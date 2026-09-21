"""Tests for TopKGate (source-side top-k blanking of gate posteriors).

Run with:  pytest tests/test_topk_gate.py -v

Design under test
-----------------
``TopKGate`` blanks all but the top-k gate entries of every query row before
the value aggregation.  The generic class owns validity masking, per-row
budgets, the annealing clock and diagnostics; the registered rule
(``noisy_hard_k``) owns the selection mask and its gradient policy
(detached hard mask; excluded gates rely on posterior-level losses).

Two levels are covered:

1. Rule/class unit tests: exactness, detachment, slack exploration,
   annealing, hard masks, per-row budgets, diagnostics, registry.
2. Integration: ``GatedCrossAttention`` / ``GatedSelfAttention`` with an
   attached TopKGate blank the applied weight A while the returned
   posterior stays raw; the flag-off path is bit-identical.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.topk_gate import TopKGate
from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.modules.gated_self_attention import GatedSelfAttention


BATCH = 2
L_ROWS = 4
S_COLS = 5
D_QK = 16
D_MODEL = 8


def _gates(requires_grad=False):
    g = torch.tensor(
        [
            [0.9, 0.1, 0.4, 0.8, 0.2],
            [0.3, 0.7, 0.6, 0.05, 0.5],
            [0.2, 0.2, 0.9, 0.1, 0.6],
            [0.5, 0.4, 0.3, 0.2, 0.85],
        ],
        dtype=torch.float32,
    )
    g = g.unsqueeze(0).expand(BATCH, L_ROWS, S_COLS).clone()
    if requires_grad:
        g.requires_grad_(True)
    return g


# ===========================================================================
# Part 1 - Registry and construction
# ===========================================================================


class TestRegistry:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown TopKGate method"):
            TopKGate(k=2, method="does_not_exist")

    def test_noisy_hard_k_registered_with_detached_policy(self):
        gate = TopKGate(k=2)
        assert gate.method == "noisy_hard_k"
        assert gate.mask_grad is False

    def test_every_registered_method_declares_gradient_policy(self):
        for name, fn in TopKGate._METHODS.items():
            assert hasattr(fn, "mask_grad"), f"method {name!r} lacks mask_grad"

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError):
            TopKGate(k=0)
        with pytest.raises(ValueError):
            TopKGate(k=[1, 0, 2])


# ===========================================================================
# Part 2 - Exact hard top-k (eval mode)
# ===========================================================================


class TestExactTopK:
    def test_eval_selects_exactly_top_k(self):
        gate = TopKGate(k=2)
        gate.eval()
        g = _gates()
        out = gate(g)
        mask = gate.last_mask  # batch-mean of a constant mask = the mask
        assert torch.allclose(mask.sum(dim=-1), torch.full((L_ROWS,), 2.0))
        expected = torch.zeros_like(g[0])
        for l in range(L_ROWS):
            top = g[0, l].topk(2).indices
            expected[l, top] = g[0, l, top]
        assert torch.allclose(out[0], expected)
        assert torch.allclose(out, g * mask.unsqueeze(0))

    def test_per_row_budget_vector(self):
        gate = TopKGate(k=[1, 2, 1, 3])
        gate.eval()
        g = _gates()
        gate(g)
        counts = gate.last_mask.sum(dim=-1)
        assert torch.allclose(counts, torch.tensor([1.0, 2.0, 1.0, 3.0]))

    def test_per_row_budget_length_mismatch_raises(self):
        gate = TopKGate(k=[1, 2])
        gate.eval()
        with pytest.raises(ValueError, match="one budget per query row"):
            gate(_gates())

    def test_fewer_allowed_than_k_passes_all_allowed(self):
        gate = TopKGate(k=4)
        gate.eval()
        g = _gates()
        hm = torch.ones(L_ROWS, S_COLS)
        hm[0] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])  # row 0: only 2 allowed
        out = gate(g, hard_mask=hm)
        assert torch.allclose(out[0, 0], g[0, 0] * hm[0])

    def test_deterministic_across_calls(self):
        gate = TopKGate(k=2)
        gate.eval()
        g = _gates()
        assert torch.allclose(gate(g), gate(g))


# ===========================================================================
# Part 3 - Gradient policy (noisy_hard_k: detached mask)
# ===========================================================================


class TestGradientPolicy:
    def test_mask_is_detached(self):
        gate = TopKGate(k=2)
        gate.eval()
        g = _gates(requires_grad=True)
        out = gate(g)
        out.sum().backward()
        # Gradient is the (detached) mask itself: 1 on selected, 0 on excluded.
        mask = gate.last_mask.unsqueeze(0).expand_as(g)
        assert torch.allclose(g.grad, mask)

    def test_excluded_gate_gets_zero_value_path_gradient(self):
        gate = TopKGate(k=1)
        gate.eval()
        g = _gates(requires_grad=True)
        gate(g).sum().backward()
        excluded = gate.last_mask == 0
        assert torch.all(g.grad[0][excluded] == 0.0)

    def test_generic_class_never_detaches_rule_mask(self):
        """The framework must not sever a future differentiable rule's grad:
        a rule returning a grad-carrying mask sees that grad survive."""
        def soft_rule(module, scores, k_row, valid, slack):
            return torch.sigmoid(scores)  # deliberately grad-carrying
        soft_rule.mask_grad = True
        TopKGate._METHODS["_test_soft"] = soft_rule
        try:
            gate = TopKGate(k=2, method="_test_soft")
            gate.eval()
            g = _gates(requires_grad=True)
            gate(g).sum().backward()
            assert g.grad is not None and torch.all(g.grad != 0.0)
        finally:
            del TopKGate._METHODS["_test_soft"]


# ===========================================================================
# Part 4 - Slack exploration and annealing (training mode)
# ===========================================================================


class TestNoisySlack:
    def test_slack_zero_training_equals_eval(self):
        gate = TopKGate(k=2, slack_init=0)
        g = _gates()
        gate.eval()
        ref = gate(g)
        gate.train()
        assert torch.allclose(gate(g), ref)

    def test_slack_explores_sub_k_gates(self):
        """With slack, the rank-(k+j) gate is included with P(eps >= j) > 0."""
        torch.manual_seed(0)
        gate = TopKGate(k=1, slack_init=3)  # static slack, S=5
        gate.train()
        g = _gates()
        selected = torch.zeros(L_ROWS, S_COLS)
        n_trials = 200
        for _ in range(n_trials):
            gate(g)
            selected += gate.last_mask
        freq = selected / n_trials
        # Rank-0 gate always selected; rank 1..3 gates selected sometimes
        # (P(eps >= j) = (4-j)/4 for j = 1..3); rank-4 gate never.
        for l in range(L_ROWS):
            order = g[0, l].argsort(descending=True)
            assert freq[l, order[0]].item() == 1.0
            assert freq[l, order[1]].item() > 0.3   # P(eps>=1) = 0.75
            assert freq[l, order[4]].item() == 0.0

    def test_per_row_slack_gives_independent_budgets(self):
        torch.manual_seed(0)
        gate = TopKGate(k=1, slack_init=3, per_row_slack=True)
        gate.train()
        g = _gates()
        counts = []
        for _ in range(50):
            gate(g)
            counts.append(gate.last_mask.sum(dim=-1))
        counts = torch.stack(counts)  # (50, L)
        # Rows must not move in lockstep (independent eps per row).
        assert not torch.allclose(counts.std(dim=0), torch.zeros(L_ROWS))

    def test_shared_slack_moves_rows_in_lockstep(self):
        torch.manual_seed(0)
        gate = TopKGate(k=1, slack_init=3, per_row_slack=False)
        gate.train()
        g = _gates()
        for _ in range(20):
            gate(g)
            counts = gate.last_mask.sum(dim=-1)
            assert torch.all(counts == counts[0])

    def test_annealing_converges_to_exact_top_k(self):
        gate = TopKGate(k=1, slack_init=4, slack_final=0, annealing_batches=10)
        gate.train()
        g = _gates()
        slacks = [gate._current_slack(S_COLS, 1)]
        for _ in range(10):
            gate(g)
            slacks.append(gate._current_slack(S_COLS, 1))
        assert slacks[0] == 4 and slacks[-1] == 0
        assert slacks == sorted(slacks, reverse=True)  # monotone decay

    def test_auto_slack_init_resolves_to_full_exploration(self):
        gate = TopKGate(k=2, annealing_batches=10)  # slack_init=None -> S - k
        gate.train()
        assert gate._current_slack(S_COLS, 2) == S_COLS - 2

    def test_clock_advances_but_no_blanking_when_phase_inactive(self):
        gate = TopKGate(k=1, slack_init=4, annealing_batches=10)
        gate.set_phase_active(False)
        gate.train()
        g = _gates()
        out = gate(g)
        assert torch.allclose(out, g)              # passthrough
        assert gate._step.item() == 1              # clock still advanced
        assert gate.last_mask is None              # no stale diagnostics



# ===========================================================================
# Part 5 - Validity mask (hard mask, diagonal) and diagnostics
# ===========================================================================


class TestValidityAndDiagnostics:
    def test_forbidden_entries_never_selected_nor_ranked(self):
        gate = TopKGate(k=2)
        gate.eval()
        g = _gates()
        hm = torch.ones(L_ROWS, S_COLS)
        hm[:, 0] = 0.0   # forbid the strongest gate of row 0
        out = gate(g, hard_mask=hm)
        assert torch.all(out[:, :, 0] == 0.0)
        # Row 0 still selects 2 gates among the ALLOWED set.
        assert gate.last_mask[0].sum().item() == 2.0

    def test_exclude_diagonal(self):
        gate = TopKGate(k=1, exclude_diagonal=True)
        gate.eval()
        g = torch.eye(4).unsqueeze(0) * 5.0 + torch.rand(1, 4, 4) * 0.1
        out = gate(g)
        assert torch.allclose(out.diagonal(dim1=-2, dim2=-1),
                              torch.zeros(1, 4))
        assert gate.last_mask.diagonal(dim1=-2, dim2=-1).sum().item() == 0.0

    def test_exclude_diagonal_requires_square(self):
        gate = TopKGate(k=1, exclude_diagonal=True)
        gate.eval()
        with pytest.raises(ValueError, match="square block"):
            gate(_gates())  # (B, 4, 5) is not square

    def test_margin_diagnostic(self):
        gate = TopKGate(k=2)
        gate.eval()
        gate(_gates())
        # Row 0 gates: [0.9, 0.1, 0.4, 0.8, 0.2] -> g_(2)=0.8, g_(3)=0.4.
        assert gate.last_margin.shape == (L_ROWS,)
        assert gate.last_margin[0].item() == pytest.approx(0.4, abs=1e-6)

    def test_margin_nan_when_budget_covers_allowed_set(self):
        gate = TopKGate(k=2)
        gate.eval()
        hm = torch.ones(2, 3)
        hm[0, 2] = 0.0   # row 0: only 2 allowed -> margin undefined
        gate(torch.rand(1, 2, 3), hard_mask=hm)
        assert torch.isnan(gate.last_margin[0])
        assert torch.isfinite(gate.last_margin[1])



# ===========================================================================
# Part 6 - Integration with the gated attention modules
# ===========================================================================


class TestGatedCrossAttentionIntegration:
    def test_applied_weight_blanked_posterior_raw(self):
        att = GatedCrossAttention(attention_dropout=0.0)
        att.topk_gate = TopKGate(k=1)
        att.eval()
        att.topk_gate.eval()
        # Batch size 1: the recorded batch-mean diagnostics are per-sample.
        q = torch.randn(1, L_ROWS, D_QK)
        k = torch.randn(1, S_COLS, D_QK)
        v = torch.randn(1, S_COLS, D_MODEL)
        out, posterior, _ = att(q, k, v)
        gate = att.topk_gate
        # out equals einsum of the BLANKED applied weight with the values
        # (eval mask is deterministic and constant across the batch).
        expected = torch.einsum(
            "bls,bsd->bld",
            gate.last_gates_in.unsqueeze(0) * gate.last_mask.unsqueeze(0),
            v,
        )
        assert torch.allclose(out, expected, atol=1e-5)
        # The returned posterior is the RAW gate posterior (not blanked).
        assert torch.allclose(posterior.mean(dim=0), att.last_p_edge_on, atol=1e-6)

    def test_flag_off_bit_identical(self):
        torch.manual_seed(1)
        att = GatedCrossAttention(attention_dropout=0.0)
        att.eval()
        q = torch.randn(BATCH, L_ROWS, D_QK)
        k = torch.randn(BATCH, S_COLS, D_QK)
        v = torch.randn(BATCH, S_COLS, D_MODEL)
        ref_out, ref_post, ref_aux = att(q, k, v)
        torch.manual_seed(1)
        att2 = GatedCrossAttention(attention_dropout=0.0)  # topk_gate=None
        att2.eval()
        out2, post2, aux2 = att2(q, k, v)
        assert torch.allclose(ref_out, out2)
        assert torch.allclose(ref_post, post2)
        assert torch.allclose(ref_aux["l0_penalty"], aux2["l0_penalty"])


class TestGatedSelfAttentionIntegration:
    def test_square_block_topk_with_diagonal_exclusion(self):
        n = 5
        att = GatedSelfAttention(attention_dropout=0.0)
        att.topk_gate = TopKGate(k=2, exclude_diagonal=True)
        att.eval()
        att.topk_gate.eval()
        q = torch.randn(1, n, D_QK)
        k = torch.randn(1, n, D_QK)
        v = torch.randn(1, n, D_MODEL)
        out, posterior, _ = att(q, k, v)
        gate = att.topk_gate
        assert torch.all(gate.last_mask.diagonal(dim1=-2, dim2=-1) == 0.0)
        assert torch.all(gate.last_mask.sum(dim=-1) <= 2.0)
        expected = torch.einsum(
            "bnm,bmd->bnd",
            gate.last_gates_in.unsqueeze(0) * gate.last_mask.unsqueeze(0),
            v,
        )
        assert torch.allclose(out, expected, atol=1e-5)



# ===========================================================================
# Part 7 - Layer-level plumbing (AttentionSelectorLayer)
# ===========================================================================


VALUE_COL = 0
VAR_COL = 1
VOCAB_S = 4
VOCAB_X = 5


def _embed_cfg(vocab, d_model=D_MODEL):
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value", "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure", "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_layer(topk_k=None, **topk_kw):
    from causaliT.core.architectures.attention_selector import (
        AttentionSelectorLayer,
    )
    return AttentionSelectorLayer(
        model="test_topk",
        ds_embed_S=_embed_cfg(VOCAB_S),
        ds_embed_X=_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
        self_attention_type="GatedSelfAttention",
        n_heads=1,
        dropout_emb=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        dropout_qkv=0.0,
        attention_dropout=0.0,
        activation="relu",
        norm="layer",
        use_final_norm=False,
        device="cpu",
        out_dim=1,
        d_ff=32,
        d_model=D_MODEL,
        d_qk=D_MODEL,  # remove_*_projection=True (default) requires d_qk == d_model
        S_seq_len=3,
        X_seq_len=4,
        topk_k=topk_k,
        **topk_kw,
    )


class TestLayerPlumbing:
    def test_flag_off_no_gate_anywhere(self):
        m = _make_layer(topk_k=None)
        assert m.attention.inner_attention.topk_gate is None
        assert m.self_attention.inner_attention.topk_gate is None
        assert m.topk_gate_main is None and m.topk_gate_self is None

    def test_split_mode_gates_on_both_blocks(self):
        m = _make_layer(topk_k=2, topk_slack_init=3, topk_annealing_batches=100)
        cross_gate = m.attention.inner_attention.topk_gate
        self_gate = m.self_attention.inner_attention.topk_gate
        assert isinstance(cross_gate, TopKGate)
        assert isinstance(self_gate, TopKGate)
        assert cross_gate.k == 2 and self_gate.k == 2
        # Cross block (S->X) has no diagonal; self block (X->X) does.
        assert cross_gate.exclude_diagonal is False
        assert self_gate.exclude_diagonal is True
        assert cross_gate.annealing_batches == 100
        # Layer attributes are plain references to the same objects.
        assert m.topk_gate_main is cross_gate
        assert m.topk_gate_self is self_gate

    def test_gate_not_duplicated_in_state_dict(self):
        m = _make_layer(topk_k=2)
        # The gates are owned by the inner attentions only: no state_dict key
        # may live at the layer top level (duplicate registration would break
        # checkpoint loading).
        assert not any(k.startswith("topk_gate") for k in m.state_dict())

    def test_homogeneous_mode_gate_is_square_excluding_diagonal(self):
        from causaliT.core.architectures.attention_selector import (
            AttentionSelectorLayer,
        )
        m = AttentionSelectorLayer(
            model="test_topk_hom",
            ds_embed_S=_embed_cfg(VOCAB_S),
            ds_embed_X=_embed_cfg(VOCAB_S + VOCAB_X),
            comps_embed_S="summation",
            comps_embed_X="svfa",
            attention_type="GatedCrossAttention",
            self_attention_type="GatedSelfAttention",
            homogeneous_nodes=True,
            n_heads=1,
            dropout_emb=0.0,
            dropout_attn_out=0.0,
            dropout_ff=0.0,
            dropout_qkv=0.0,
            attention_dropout=0.0,
            activation="relu",
            norm="layer",
            use_final_norm=False,
            device="cpu",
            out_dim=1,
            d_ff=32,
            d_model=D_MODEL,
            d_qk=D_MODEL,  # remove_*_projection=True (default) requires d_qk == d_model
            S_seq_len=3,
            X_seq_len=4,
            topk_k=9,
        )
        gate = m.attention.inner_attention.topk_gate
        assert isinstance(gate, TopKGate)
        assert gate.exclude_diagonal is True
        assert gate.k == 9

