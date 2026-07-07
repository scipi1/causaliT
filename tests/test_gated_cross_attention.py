"""Tests for GatedCrossAttention (disentangled structure-gate x recon-gain).

Run with:  pytest tests/test_gated_cross_attention.py -v

Design under test
-----------------
The edge weight is factorised ``A_ij = z_ij * g_ij`` where

* ``z_ij`` — Hard-Concrete **structure gate** (from the structural query/key,
  classified STRUCTURAL → driven by HSIC + L0), and
* ``g_ij`` — bounded sigmoid **reconstruction gain** (from the SEPARATE gain
  query/key, classified RECONSTRUCTION → driven by the MSE loss).

Two levels are covered:

1. Module level (``GatedCrossAttention``): shapes, the ``A = z*g`` contract,
   the ``gain_query``/``gain_key`` requirement, and the gate-open reconstruction
   recovery (anti-run_0: a non-null gain alone reconstructs a scaled target).
2. Layer level (``AttentionSelectorLayer`` with ``attention_type=
   "GatedCrossAttention"``): construction wiring, forward shapes, the
   gradient-routing split (gain params → reconstruction, gate Q/K → structural),
   the shared-mode guard, and a backward smoke test.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants (mirrors tests/test_atsel_free_query.py layout)
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ===========================================================================
# Part 1 — Module-level tests (GatedCrossAttention in isolation)
# ===========================================================================


def _proj(B, L, E):
    return torch.randn(B, L, E, requires_grad=True)


class TestModuleForward:
    def test_shapes_3d_value(self):
        att = GatedCrossAttention(register_entropy=True)
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        out, A, aux = att(q, k, v, gain_query=gq, gain_key=gk)
        assert out.shape == (BATCH, X_SEQ_LEN, D_MODEL)
        assert A.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert aux["l0_penalty"].dim() == 0            # scalar
        assert aux["entropy"].shape == (BATCH, X_SEQ_LEN)

    def test_shapes_4d_value(self):
        att = GatedCrossAttention()
        H = 3
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, H, D_MODEL)
        out, A, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        assert out.shape == (BATCH, X_SEQ_LEN, H, D_MODEL)

    def test_requires_gain_projections(self):
        att = GatedCrossAttention()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        with pytest.raises(ValueError):
            att(q, k, v)   # no gain_query / gain_key

    def test_rejects_multihead_struct(self):
        att = GatedCrossAttention()
        q = torch.randn(BATCH, X_SEQ_LEN, 2, D_QK)   # 4-D struct query
        k = torch.randn(BATCH, S_SEQ_LEN, 2, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        with pytest.raises(ValueError):
            att(q, k, v, gain_query=gq, gain_key=gk)


class TestFactorisationContract:
    def test_A_equals_gate_times_gain_and_bounded(self):
        """A = z * g with both factors in [0,1] → A in [0,1]."""
        att = GatedCrossAttention(attention_dropout=0.0)
        att.eval()   # deterministic gate
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        _, A, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        assert torch.all(A >= 0.0) and torch.all(A <= 1.0)

    def test_hard_mask_zeros_forbidden_edges(self):
        att = GatedCrossAttention()
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        mask = 1.0 - torch.eye(X_SEQ_LEN)   # no self-loops
        _, A, _ = att(q, k, v, hard_mask=mask, gain_query=gq, gain_key=gk)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))

    def test_gate_open_recovers_gain_weighted_target(self):
        """Anti-run_0: with the gate forced fully open, the bounded sigmoid gain
        ALONE linearly mixes the value stream — reconstruction is possible even
        though the structure score is saturated (constant)."""
        att = GatedCrossAttention()
        att.eval()
        # Large positive structural logits → z ≈ 1 for every edge.
        q = torch.full((1, 2, D_QK), 5.0)
        k = torch.full((1, 2, D_QK), 5.0)
        gq = _proj(1, 2, D_QK)
        gk = _proj(1, 2, D_QK)
        v = torch.randn(1, 2, D_MODEL)
        out, A, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        # out must equal A @ v exactly (aggregation contract) and A must be > 0
        # (non-null) — i.e. information actually flows.
        assert torch.allclose(out, torch.einsum("bls,bsd->bld", A, v), atol=1e-5)
        assert A.abs().sum() > 0.0

    def test_sparsity_signal_is_gate_only(self):
        """score_tensor_for_sparsity / last_p_edge_on are functions of the GATE,
        never the gain — so structure regularisers act on structure only."""
        att = GatedCrossAttention()
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        att(q, k, v, gain_query=gq, gain_key=gk)
        assert att.score_tensor_for_sparsity.shape == (X_SEQ_LEN, S_SEQ_LEN)
        assert torch.all(att.score_tensor_for_sparsity >= 0.0)
        assert torch.all(att.score_tensor_for_sparsity <= 1.0)


def _manual_gain(gq, gk, gain_tau=1.0):
    """Reproduce the module's bounded sigmoid gain g = sigmoid(<gq,gk>/(tau*sqrt(E)))."""
    import math
    E = gq.shape[-1]
    logit = torch.einsum("ble,bse->bls", gq, gk) / (gain_tau * math.sqrt(E))
    return torch.sigmoid(logit)


class TestOptunaProtocol:
    """Constant-score capacity protocol: the STRUCTURE gate is frozen at a
    constant (0/1) while the reconstruction gain stays learnable."""

    def test_protocol_one_freezes_gate_A_equals_gain(self):
        att = GatedCrossAttention(optuna_protocol=1.0, attention_dropout=0.0)
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        _, A, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        assert torch.allclose(A, _manual_gain(gq, gk), atol=1e-5)

    def test_protocol_half_scales_gain(self):
        att = GatedCrossAttention(optuna_protocol=0.5, attention_dropout=0.0)
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        _, A, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        assert torch.allclose(A, 0.5 * _manual_gain(gq, gk), atol=1e-5)

    def test_protocol_gate_qk_receive_no_gradient(self):
        """Freezing the gate must detach the structural query/key from the loss."""
        att = GatedCrossAttention(optuna_protocol=1.0, attention_dropout=0.0)
        att.train()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        out, _, _ = att(q, k, v, gain_query=gq, gain_key=gk)
        out.sum().backward()
        assert q.grad is None or q.grad.abs().sum() == 0
        assert k.grad is None or k.grad.abs().sum() == 0
        assert gq.grad is not None and gq.grad.abs().sum() > 0

    def test_protocol_still_masks_self_loops(self):
        att = GatedCrossAttention(optuna_protocol=1.0, attention_dropout=0.0)
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        mask = 1.0 - torch.eye(X_SEQ_LEN)
        _, A, _ = att(q, k, v, hard_mask=mask, gain_query=gq, gain_key=gk)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))


class TestOracleGate:
    """Oracle mode: the ground-truth adjacency (hard_mask) IS the structure
    gate; only the learned reconstruction gain modulates magnitude."""

    def test_oracle_A_equals_mask_times_gain(self):
        att = GatedCrossAttention(attention_dropout=0.0)
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, X_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        mask = (torch.rand(X_SEQ_LEN, X_SEQ_LEN) > 0.5).float()
        out, A, _ = att(q, k, v, hard_mask=mask, oracle=True, gain_query=gq, gain_key=gk)
        expected_A = mask.unsqueeze(0) * _manual_gain(gq, gk)
        assert torch.allclose(A, expected_A, atol=1e-5)
        assert torch.allclose(out, torch.einsum("bls,bsd->bld", A, v), atol=1e-5)

    def test_oracle_requires_hard_mask(self):
        att = GatedCrossAttention()
        att.eval()
        q, k = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        gq, gk = _proj(BATCH, X_SEQ_LEN, D_QK), _proj(BATCH, S_SEQ_LEN, D_QK)
        v = torch.randn(BATCH, S_SEQ_LEN, D_MODEL)
        with pytest.raises(ValueError):
            att(q, k, v, oracle=True, gain_query=gq, gain_key=gk)


# ===========================================================================
# Part 2 — Layer-level tests (AttentionSelectorLayer + GatedCrossAttention)
# ===========================================================================



def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_gated_model(
    gain_stream_source: str = "separate",
    orthogonal_fixed: bool = False,
    optuna_protocol=None,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_gated",
        optuna_protocol=optuna_protocol,

        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
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
        d_ff=D_FF,
        d_model=D_MODEL,
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        orthogonal_fixed=orthogonal_fixed,
        gain_stream_source=gain_stream_source,
    )


def _make_inputs():
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


class TestLayerConstruction:
    def test_is_gated_flag(self):
        m = _make_gated_model()
        assert m.is_gated is True

    def test_gain_embeddings_present_in_separate_mode(self):
        m = _make_gated_model(gain_stream_source="separate")
        assert m.gain_q_embed_X is not None
        assert m.gain_k_embed_S is not None
        assert m.gain_k_embed_X is not None
        # and the layer-level gain projections exist and are the right shape
        assert m.attention.gain_q_proj is not None
        assert m.attention.gain_k_proj is not None

    def test_shared_mode_requires_orthogonal_fixed(self):
        with pytest.raises(ValueError):
            _make_gated_model(gain_stream_source="shared", orthogonal_fixed=False)
        # allowed when the struct frame is frozen
        m = _make_gated_model(gain_stream_source="shared", orthogonal_fixed=True)
        assert m.gain_q_embed_X is None   # no separate tables in shared mode


class TestLayerForward:
    def test_forward_shapes(self):
        m = _make_gated_model()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        assert "l0_penalty" in aux and aux["l0_penalty"] is not None

    def test_x_self_loops_zero(self):
        m = _make_gated_model()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        _, att_xx = m.split_attention(attn)          # (B, L_X, L_X)
        diag = torch.diagonal(att_xx, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))


class TestRoutingSeparation:
    def test_gain_params_are_reconstruction_gate_qk_structural(self):
        m = _make_gated_model()
        structural, reconstruction = classify_parameters(m)
        struct_ids = {id(p) for p in structural}
        recon_ids = {id(p) for p in reconstruction}

        # Gain stream (embeddings + projections) → RECONSTRUCTION.
        gain_modules = [
            m.gain_q_embed_X, m.gain_k_embed_S, m.gain_k_embed_X,
            m.attention.gain_q_proj, m.attention.gain_k_proj,
        ]
        for mod in gain_modules:
            for p in mod.parameters():
                assert id(p) in recon_ids, "gain params must be RECONSTRUCTION"
                assert id(p) not in struct_ids

        # Structural gate Q/K projections → STRUCTURAL.
        for p in m.attention.query_projection.parameters():
            assert id(p) in struct_ids
        for p in m.attention.key_projection.parameters():
            assert id(p) in struct_ids


class TestBackward:
    def test_gain_embedding_receives_gradient(self):
        m = _make_gated_model()
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()
        w = m.gain_q_embed_X.embedding.weight
        assert w.grad is not None and w.grad.abs().sum() > 0, (
            "gain_q_embed_X must receive gradient from the reconstruction output."
        )

    def test_l0_penalty_backward_reaches_gate_qk(self):
        m = _make_gated_model()
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        _, _, aux = m.forward_with_actual(source, x_blanked, x_actual)
        aux["l0_penalty"].backward()
        w = m.attention.query_projection.weight
        assert w.grad is not None and w.grad.abs().sum() > 0, (
            "L0 penalty must flow into the structural gate query projection."
        )


class TestOptunaProtocolLayer:
    """End-to-end wiring: model.kwargs.optuna_protocol must reach the GCA gate
    through AttentionSelectorLayer -> AttentionLayer -> GatedCrossAttention."""

    def test_protocol_freezes_gate_qk_but_trains_gain(self):
        m = _make_gated_model(optuna_protocol=1.0)
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()

        # Reconstruction gain still learns from the output ...
        gain_w = m.gain_q_embed_X.embedding.weight
        assert gain_w.grad is not None and gain_w.grad.abs().sum() > 0, (
            "gain stream must still receive reconstruction gradient under protocol."
        )
        # ... while the frozen structural gate Q/K get no reconstruction signal.
        gate_w = m.attention.query_projection.weight
        assert gate_w.grad is None or gate_w.grad.abs().sum() == 0, (
            "structural gate Q/K must be detached from the loss under optuna_protocol."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


