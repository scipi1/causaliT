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
    hsic_softmax_pair_weights,
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
            sigma=1.0,
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
            sigma=1.0,
        )
        l_away = hsic_attention_softmax(
            source_values=src, residuals=res, attention_weights=att_away,
            sigma=1.0,
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
        )
        val.backward()
        assert att.grad is not None
        assert torch.isfinite(att.grad).all()
        # d loss / d logit_j = w_j (H_j - sum_k w_k H_k): the dependent
        # pair (1,0) has the largest H, so its logit gradient is POSITIVE
        # (increasing it raises the loss -> the optimiser pushes it down).
        row = att.grad[1]
        assert row[0] > row[2]
