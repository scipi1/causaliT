"""Softmax-competition HSIC aggregation (``hsic_aggregation: attw_softmax``).

The row-wise softmax removes the all-zero trivial solution of the
unnormalised attention-weighted variant: each row's weights sum to 1, so the
mass has to go somewhere and sparsity emerges from the within-row
competition.  The self-edge is excluded BEFORE the softmax.
"""
import pytest
import torch

from causaliT.utils.hsic_utils import (
    hsic_attention_softmax,
    hsic_attention_weighted,
    hsic_evidence_max_pair_weights,
    hsic_posterior_pair_weights,
    hsic_softmax_pair_weights,
    row_entropy_stats,
)


def _dep_data(batch=256, n=3, seed=0):
    """Square data where residual 1 depends on source 0 only."""
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(batch, n, generator=g)
    res = torch.randn(batch, n, generator=g)
    res[:, 1] = 0.8 * src[:, 0] + 0.2 * res[:, 1]   # only pair (1, 0) dependent
    return src, res


class TestPairWeights:
    def test_uniform_logits_give_uniform_offdiag_weights(self):
        att = torch.zeros(3, 3)
        w = hsic_softmax_pair_weights(att)
        assert torch.allclose(w.sum(dim=1), torch.ones(3))
        assert torch.allclose(w.diagonal(), torch.zeros(3))
        off = w[~torch.eye(3, dtype=torch.bool)]
        assert torch.allclose(off, torch.full_like(off, 0.5))

    def test_rectangular_no_diagonal_removed_by_default(self):
        att = torch.zeros(2, 4)
        w = hsic_softmax_pair_weights(att)
        assert torch.allclose(w, torch.full((2, 4), 0.25))

    def test_rectangular_diagonal_offset_excludes_self_edge(self):
        att = torch.zeros(2, 4)
        w = hsic_softmax_pair_weights(att, diagonal_offset=2)
        assert torch.allclose(w.sum(dim=1), torch.ones(2))
        assert w[0, 2].item() == 0.0 and w[1, 3].item() == 0.0
        off0 = torch.tensor([w[0, 0], w[0, 1], w[0, 3]])
        assert torch.allclose(off0, torch.full((3,), 1 / 3))

    def test_inf_entries_get_zero_weight(self):
        att = torch.tensor([[0.0, 1.0, float("-inf")]])
        w = hsic_softmax_pair_weights(att)
        assert w[0, 2].item() == 0.0
        assert torch.allclose(w.sum(dim=1), torch.ones(1))

    def test_fully_masked_row_is_zero_not_nan(self):
        att = torch.full((2, 2), float("-inf"))
        w = hsic_softmax_pair_weights(att)
        assert torch.isfinite(w).all()
        assert torch.allclose(w, torch.zeros_like(w))

    def test_gradient_flows_through_softmax(self):
        att = torch.randn(3, 3, requires_grad=True)
        w = hsic_softmax_pair_weights(att)
        w.sum().backward()
        assert att.grad is not None
        assert torch.isfinite(att.grad).all()


class TestAntiTrivialSolution:
    def test_zero_logits_still_penalise_dependence(self):
        """attw with zero attention -> 0 loss (trivial); softmax must NOT."""
        src, res = _dep_data()
        zero_att = torch.zeros(3, 3)
        plain = hsic_attention_weighted(
            source_values=src, residuals=res, attention_weights=zero_att,
            sigma=1.0,
        )
        comp = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=zero_att,
            sigma=1.0, pair_weight_mode="softmax_logits",
        )
        assert plain.item() == pytest.approx(0.0, abs=1e-12)
        assert comp.item() > 0.0  # uniform spread still pays the dependent pair

    def test_dropping_dependent_edge_raises_loss(self):
        """Moving mass AWAY from the dependent pair must reduce the loss."""
        src, res = _dep_data()
        att_parent = torch.zeros(3, 3)
        att_parent[1, 0] = 5.0      # concentrate on the dependent pair
        att_away = torch.zeros(3, 3)
        att_away[1, 2] = 5.0        # concentrate on an independent pair
        l_parent = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att_parent,
            sigma=1.0, pair_weight_mode="softmax_logits",
        )
        l_away = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att_away,
            sigma=1.0, pair_weight_mode="softmax_logits",
        )
        assert l_away.item() < l_parent.item()


class TestSelfEdge:
    def test_square_diagonal_excluded_from_matrix_and_weights(self):
        src, res = _dep_data()
        att = torch.zeros(3, 3)
        val, mat = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att,
            sigma=1.0, return_matrix=True,
        )
        assert torch.isnan(mat.diagonal()).all()
        assert torch.isfinite(val)

    def test_split_mode_self_edge_excluded_via_offset(self):
        g = torch.Generator().manual_seed(1)
        n_s, n_x, b = 2, 2, 128
        src = torch.randn(b, n_s + n_x, generator=g)   # combined [S ; X]
        res = torch.randn(b, n_x, generator=g)
        att = torch.zeros(n_x, n_s + n_x)
        val, mat = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att,
            sigma=1.0, return_matrix=True, diagonal_offset=n_s,
        )
        assert torch.isnan(mat[0, 2]) and torch.isnan(mat[1, 3])
        assert torch.isfinite(mat[0, :2]).all() and torch.isfinite(mat[1, :2]).all()
        assert torch.isfinite(val)

    def test_self_edge_would_otherwise_win(self):
        """HSIC(X_i, r_i) is huge; with the self-edge in play it must win."""
        g = torch.Generator().manual_seed(2)
        x = torch.randn(256, 3, generator=g)
        res = x.clone()                    # residual i == source i: irreducible
        att = torch.zeros(3, 3)
        val, mat = hsic_attention_softmax(
            source_values=x, residuals=res, attention_weights=att,
            sigma=1.0, return_matrix=True,
        )
        from causaliT.utils.hsic_utils import hsic as _hsic
        self_hsic = _hsic(x[:, 0], res[:, 0], sigma=1.0)
        assert val.item() < 0.2 * self_hsic.item()


class TestContract:
    def test_shape_mismatch_raises(self):
        src, res = _dep_data()
        with pytest.raises(ValueError, match="does not match"):
            hsic_attention_softmax(
                source_values=src, residuals=res,
                attention_weights=torch.zeros(3, 4), sigma=1.0,
            )

    def test_gradient_wrt_attention_is_finite(self):
        src, res = _dep_data()
        att = torch.zeros(3, 3, requires_grad=True)
        val = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att, sigma=1.0,
            pair_weight_mode="softmax_logits",
        )
        val.backward()
        assert att.grad is not None
        assert torch.isfinite(att.grad).all()
        # d loss / d logit_j = w_j (H_j - sum_k w_k H_k): the dependent
        # pair (1,0) has the largest H, so its logit gradient is POSITIVE
        # (increasing it raises the loss -> the optimiser pushes it down).
        row = att.grad[1]
        assert row[0] > row[2]

class TestPosteriorPairWeights:
    """posterior mode: softmax over log p == L1 row-renormalisation of the
    gate posterior.  A closed gate (p = 0) gets weight EXACTLY 0."""

    def test_zero_gate_gets_exactly_zero_weight(self):
        att = torch.tensor([[0.0, 0.8, 0.0], [0.5, 0.0, 0.0]])
        w = hsic_posterior_pair_weights(att)
        assert w[0, 0].item() == 0.0 and w[0, 2].item() == 0.0
        assert w[0, 1].item() == pytest.approx(1.0)
        assert torch.allclose(w.sum(dim=1), torch.ones(2))

    def test_renormalisation_matches_closed_form(self):
        att = torch.tensor([[0.2, 0.5, 0.1], [0.9, 0.3, 0.6]])
        w = hsic_posterior_pair_weights(att)
        assert torch.allclose(w, att / att.sum(dim=1, keepdim=True))

    def test_square_diagonal_excluded_before_normalisation(self):
        att = torch.tensor([[0.9, 0.4, 0.0],
                            [0.4, 0.9, 0.2],
                            [0.0, 0.2, 0.9]])
        w = hsic_posterior_pair_weights(att)
        assert torch.allclose(w.diagonal(), torch.zeros(3))
        assert torch.allclose(w.sum(dim=1), torch.ones(3))
        assert w[0, 1].item() == pytest.approx(1.0)   # only off-diag in row 0

    def test_fully_closed_row_is_zero_not_nan(self):
        att = torch.zeros(2, 3)
        w = hsic_posterior_pair_weights(att)
        assert torch.isfinite(w).all()
        assert torch.allclose(w, torch.zeros_like(w))

    def test_split_mode_self_edge_excluded_via_offset(self):
        att = torch.full((2, 4), 0.5)
        w = hsic_posterior_pair_weights(att, diagonal_offset=2)
        assert w[0, 2].item() == 0.0 and w[1, 3].item() == 0.0
        assert torch.allclose(w.sum(dim=1), torch.ones(2))

    def test_gradient_flows(self):
        att = torch.rand(3, 3, requires_grad=True)
        hsic_posterior_pair_weights(att).sum().backward()
        assert att.grad is not None and torch.isfinite(att.grad).all()

    def test_direction_gate_routes_descendant_pressure(self):
        """Self-block factorisation p = p_exist * d: zeroing ONLY the
        antisymmetric direction gate of the anti-causal orientation closes
        the pair exactly, while the symmetric counterpart keeps its mass."""
        p_exist = torch.full((3, 3), 0.8)
        d = torch.full((3, 3), 0.5)
        d[0, 2] = 0.0          # descendant orientation (2 anti-causal for 0)
        d[2, 0] = 1.0          # true orientation stays fully open
        p = p_exist * d
        p.fill_diagonal_(0.0)
        w = hsic_posterior_pair_weights(p)
        assert w[0, 2].item() == 0.0, "closed direction gate must close the pair"
        assert w[2, 0].item() > 0.0, "true orientation keeps its weight"


class TestPosteriorAggregation:
    def test_default_mode_is_posterior(self):
        src, res = _dep_data()
        # Posterior concentrated on the dependent pair vs spread out.
        att_parent = torch.zeros(3, 3); att_parent[1, 0] = 0.9
        att_away = torch.zeros(3, 3); att_away[1, 2] = 0.9
        l_parent = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att_parent,
            sigma=1.0,
        )
        l_away = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att_away,
            sigma=1.0,
        )
        assert l_away.item() < l_parent.item()

    def test_closed_gates_pay_nothing(self):
        """The descendant/spurious pair with p = 0 contributes exactly 0."""
        src, res = _dep_data()
        att = torch.zeros(3, 3)
        att[1, 0] = 0.9   # only the true parent gate open
        val, mat = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att,
            sigma=1.0, return_matrix=True,
        )
        from causaliT.utils.hsic_utils import hsic as _hsic
        # Row 1 pays exactly HSIC(src_0, res_1); all other rows are closed.
        expected = _hsic(src[:, 0], res[:, 1], sigma=1.0) / 3.0
        assert val.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_unknown_mode_raises(self):
        src, res = _dep_data()
        with pytest.raises(ValueError, match="pair_weight_mode"):
            hsic_attention_softmax(
                source_values=src, residuals=res,
                attention_weights=torch.zeros(3, 3), sigma=1.0,
                pair_weight_mode="bogus",
            )


class TestRowEntropyStats:
    def test_uniform_row_has_max_entropy(self):
        w = torch.full((2, 4), 0.25)
        stats = row_entropy_stats(w)
        import math
        assert stats["mean"] == pytest.approx(math.log(4))
        assert stats["norm_mean"] == pytest.approx(1.0)
        assert stats["eff_competitors_mean"] == pytest.approx(4.0)

    def test_one_hot_row_has_zero_entropy(self):
        w = torch.zeros(2, 3); w[:, 0] = 1.0
        stats = row_entropy_stats(w)
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["eff_competitors_mean"] == pytest.approx(1.0)

    def test_zero_rows_excluded(self):
        w = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
        stats = row_entropy_stats(w)
        import math
        assert stats["mean"] == pytest.approx(math.log(2))
        assert stats["max"] == stats["min"]

    def test_all_zero_is_safe(self):
        stats = row_entropy_stats(torch.zeros(2, 2))
        assert all(v == 0.0 for v in stats.values())

    def test_non_row_stochastic_weights_are_normalised(self):
        """evidence_max rows (leader = 1 + subordinates) must NOT yield
        entropies above ln K: rows are renormalised before the entropy and
        the raw mass is reported separately."""
        import math
        w = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.25, 0.0]])  # sums 3, 1.25
        stats = row_entropy_stats(w)
        assert stats["mean"] == pytest.approx(
            0.5 * (math.log(3) + (-(0.8 * math.log(0.8)
                                   + 0.2 * math.log(0.2)))))
        assert stats["norm_mean"] <= 1.0 + 1e-9
        assert stats["eff_competitors_mean"] <= 3.0 + 1e-6
        assert stats["row_mass_mean"] == pytest.approx(0.5 * (3.0 + 1.25))

    def test_row_stochastic_modes_unaffected(self):
        w = torch.full((2, 4), 0.25)
        stats = row_entropy_stats(w)
        assert stats["row_mass_mean"] == pytest.approx(1.0)

    def test_binary_gate_entropy_closed_form(self):
        """k open gates of K competitors: H = ln(k*e + K-k) - k*e/(k*e+K-k)
        for the LEGACY [0,1]-logit softmax; posterior mode gives ln(k)."""
        import math
        k, K = 3, 20
        att = torch.zeros(1, K); att[0, :k] = 1.0
        w_legacy = hsic_softmax_pair_weights(att)
        z = k * math.e + (K - k)
        h_legacy = row_entropy_stats(w_legacy)["mean"]
        assert h_legacy == pytest.approx(math.log(z) - k * math.e / z, rel=1e-5)
        w_post = hsic_posterior_pair_weights(att)
        assert row_entropy_stats(w_post)["mean"] == pytest.approx(math.log(k))

class TestEvidenceMaxWeights:
    """evidence_max: s = p * exp(-H/tau), max-normalised per row."""

    def test_leader_gets_unit_weight(self):
        p = torch.tensor([[0.5, 0.8, 0.2]])
        H = torch.zeros(1, 3)
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert w.max().item() == pytest.approx(1.0)
        assert w[0, 1].item() == pytest.approx(1.0)   # argmax p leads

    def test_descendant_suppressed_exponentially(self):
        """High-H (descendant) pair loses weight ~ exp(-H/tau) despite a
        strong gate."""
        p = torch.tensor([[0.9, 0.9]])
        H = torch.tensor([[0.0, 5.0]])
        w = hsic_evidence_max_pair_weights(p, H, tau=0.5)
        import math
        assert w[0, 0].item() == pytest.approx(1.0)
        assert w[0, 1].item() == pytest.approx(math.exp(-10.0), rel=1e-4)

    def test_closed_gate_stays_zero(self):
        p = torch.tensor([[0.0, 0.7]])
        H = torch.tensor([[10.0, 0.1]])   # closed pair has huge H; still 0
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert w[0, 0].item() == 0.0

    def test_non_dilutive_additivity(self):
        """Opening a below-leader gate only ADDS cost: row loss rises."""
        src, res = _dep_data()
        p_one = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])
        p_two = p_one.clone(); p_two[1, 2] = 0.45   # add subordinate gate
        l_one = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=p_one,
            sigma=1.0, pair_weight_mode="evidence_max", tilt_tau=1.0,
        )
        l_two = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=p_two,
            sigma=1.0, pair_weight_mode="evidence_max", tilt_tau=1.0,
        )
        # H(2,?) terms ~ 0 for row 2; row 1 gains w*H >= 0 -> never diluted
        assert l_two.item() >= l_one.item() - 1e-8

    def test_h0_pairs_are_l0_neutral(self):
        """H ~ 0 pairs keep w = p/p_lead: HSIC neither rewards nor punishes
        them, so L0 removes them unopposed."""
        p = torch.tensor([[0.4, 0.8]])
        H = torch.zeros(1, 2)
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert w[0, 0].item() == pytest.approx(0.5)
        assert w[0, 1].item() == pytest.approx(1.0)

    def test_maximum_budget_not_exact(self):
        """Support can fall BELOW the budget: no forced mass anywhere."""
        p = torch.tensor([[0.9, 0.001, 0.0]])
        H = torch.tensor([[0.0, 3.0, 1.0]])
        w = hsic_evidence_max_pair_weights(p, H, tau=0.1)
        assert w[0, 0].item() == pytest.approx(1.0)
        assert w[0, 1].item() < 1e-8
        assert w[0, 2].item() == 0.0

    def test_self_edge_excluded_before_max(self):
        p = torch.tensor([[0.9, 0.4, 0.1],
                          [0.4, 0.9, 0.2],
                          [0.1, 0.2, 0.9]])
        H = torch.zeros(3, 3)
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert torch.allclose(w.diagonal(), torch.zeros(3))
        assert w[0, 1].item() == pytest.approx(1.0)  # new leader of row 0

    def test_nan_evidence_excluded(self):
        p = torch.tensor([[0.5, 0.5]])
        H = torch.tensor([[0.0, float("nan")]])
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert w[0, 1].item() == 0.0
        assert torch.isfinite(w).all()

    def test_empty_row_is_zero(self):
        p = torch.zeros(2, 3)
        H = torch.zeros(2, 3)
        w = hsic_evidence_max_pair_weights(p, H, tau=1.0)
        assert torch.allclose(w, torch.zeros_like(w))

    def test_auto_tau_is_mad_scaled(self):
        """auto = 1.4826 * MAD of the valid evidence entries."""
        import math
        E = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
        med = E.median()
        mad = (E - med).abs().median()
        tau_auto = float(1.4826 * mad)
        p = torch.full((1, 4), 0.5)
        w_auto = hsic_evidence_max_pair_weights(p, E, tau="auto")
        w_ref = hsic_evidence_max_pair_weights(p, E, tau=tau_auto)
        assert torch.allclose(w_auto, w_ref)
        assert math.isfinite(tau_auto) and tau_auto > 0

    def test_bad_tau_raises(self):
        with pytest.raises(ValueError, match="tau"):
            hsic_evidence_max_pair_weights(
                torch.ones(1, 2), torch.zeros(1, 2), tau=0.0
            )

    def test_gradient_flows_to_gates(self):
        p = torch.rand(3, 3, requires_grad=True)
        H = torch.rand(3, 3)
        hsic_evidence_max_pair_weights(p, H, tau=1.0).sum().backward()
        assert p.grad is not None and torch.isfinite(p.grad).all()


class TestEvidenceMaxAggregation:
    def test_return_weights_contract(self):
        src, res = _dep_data()
        out = hsic_attention_softmax(
            source_values=src, residuals=res,
            attention_weights=torch.full((3, 3), 0.5), sigma=1.0,
            pair_weight_mode="evidence_max", return_weights=True,
        )
        val, mat, w = out
        assert mat is None          # return_matrix not requested
        assert w.shape == (3, 3)
        assert not w.requires_grad  # returned weights are detached
        assert torch.isfinite(val)

    def test_evidence_beats_blind_posterior_on_descendant(self):
        """With equal gates, the descendant (H > 0) must end up with LESS
        weight than the fitted parent (H ~ 0) -- the tilt does it."""
        g = torch.Generator().manual_seed(7)
        b = 512
        x0 = torch.randn(b, generator=g)
        x1 = 0.9 * x0 + 0.1 * torch.randn(b, generator=g)   # child of 0
        src = torch.stack([x0, x1], dim=1)
        res = torch.stack([x0, 0.5 * x1 + 0.5 * torch.randn(b, generator=g)],
                          dim=1)
        # Row 1: source 0 is the (fitted) parent -> H ~ small; source 1 is
        # the descendant (self-edge of the pair view) -> H large.
        att = torch.tensor([[0.0, 0.8], [0.8, 0.0]])
        _, _, w = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att,
            sigma=1.0, pair_weight_mode="evidence_max", tilt_tau=0.05,
            return_weights=True,
        )
        # row 1 must concentrate on source 0 (the parent it was fitted on)
        assert w[1, 0].item() > 0.9
