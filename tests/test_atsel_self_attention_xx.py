"""Tests for the direction-aware X→X split in ``AttentionSelectorLayer``.

Run with:  pytest tests/test_atsel_self_attention_xx.py -v

Motivation
----------
The single combined cross-attention parametrises the X→X interaction with the
SAME (undirected) mechanism used for S→X, so it cannot express edge direction
and produces two-cycles / double edges.  Setting ``self_attention_type=
"GatedSelfAttention"`` SPLITS the layer into:

  * S→X  via the cross ``attention_type`` block  (keys/values = S only)
  * X→X  via a dedicated ``GatedSelfAttention`` block whose antisymmetric
         direction gate enforces ``d_ij + d_ji = 1`` (two-cycle suppression)

The two attention outputs are summed into ONE value residual stream, so the
unified representation of X (reconstructed from both S and X) is preserved.

These tests verify:

1. Construction wiring (``self_attention`` present iff the type is set).
2. Forward shapes + combined-attention round-trip equals ``cat(attn_sx, attn_xx)``.
3. Two-cycle suppression on the X→X directed posterior (``p_ij + p_ji <= 1``).
4. ``get_score_tensor_for_sparsity`` returns the ``(L_X, L_S+L_X)`` combined tensor.
5. Gradient routing classifies the self block's Q/K as STRUCTURAL and its
   gain / value / out projections as RECONSTRUCTION.
6. ``oracle=True`` raises ``NotImplementedError`` in split mode.
7. Backward smoke: gradients reach the self-block structural params.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 5
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_model(
    self_attention_type=None,
    free_query_embedding: bool = True,
    gain_stream_source: str = "separate",
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="svfa",
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
        struct_embedding_type="standard_learnable",
        free_query_embedding=free_query_embedding,
        gain_stream_source=gain_stream_source,
        self_attention_type=self_attention_type,
    )


def _make_inputs():
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Construction wiring
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_split_flag(self):
        assert _make_model(self_attention_type="GatedSelfAttention").split_xx is True
        assert _make_model(self_attention_type=None).split_xx is False

    def test_self_attention_present_only_when_enabled(self):
        m_on = _make_model(self_attention_type="GatedSelfAttention")
        m_off = _make_model(self_attention_type=None)
        assert m_on.self_attention is not None
        assert m_off.self_attention is None

    def test_cross_block_keys_are_S_only_in_split(self):
        m = _make_model(self_attention_type="GatedSelfAttention")
        # Split cross block attends S only → key_seq_len == S_SEQ_LEN.
        assert m.cross_mask.shape == (X_SEQ_LEN, S_SEQ_LEN)
        assert m.self_mask.shape == (X_SEQ_LEN, X_SEQ_LEN)
        # self mask is off-diagonal (no self loops)
        assert torch.equal(torch.diagonal(m.self_mask), torch.zeros(X_SEQ_LEN))

    def test_invalid_self_attention_type_raises(self):
        with pytest.raises(ValueError):
            _make_model(self_attention_type="NotARealBlock")


# ---------------------------------------------------------------------------
# 2. Forward shapes + combined-attention round-trip
# ---------------------------------------------------------------------------


class TestForwardShapes:
    def test_shapes(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        assert "l0_penalty" in aux

    def test_round_trip_split(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        att_sx, att_xx = model.split_attention(attn)
        assert att_sx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert att_xx.shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        assert torch.allclose(torch.cat([att_sx, att_xx], dim=-1), attn)

    def test_single_mode_still_works(self):
        """Regression guard: self_attention_type=None reproduces the combined block."""
        model = _make_model(self_attention_type=None)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 3. Two-cycle suppression on the X→X directed posterior
# ---------------------------------------------------------------------------


class TestDirectionality:
    def test_two_cycle_suppressed(self):
        """The X→X directed posterior satisfies p_ij + p_ji <= 1 (Toeplitz split).

        In eval mode d_ij + d_ji == 1, so the directed posterior
        p_directed_ij + p_directed_ji == p_edge_undirected <= 1: a genuine
        i→j edge forces j→i toward 0, killing two-cycles.
        """
        model = _make_model(self_attention_type="GatedSelfAttention")
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        model.forward_with_actual(source, x_blanked, x_actual)
        score = model.get_score_tensor_for_sparsity()   # (L_X, L_S+L_X)
        xx = score[:, S_SEQ_LEN:]                        # (L_X, L_X)
        pair_sum = xx + xx.t()
        assert torch.all(pair_sum <= 1.0 + 1e-4), (
            "Directed X→X posterior must satisfy p_ij + p_ji <= 1 "
            "(antisymmetric direction gate)."
        )


# ---------------------------------------------------------------------------
# 4. Combined score tensor
# ---------------------------------------------------------------------------


class TestScoreTensor:
    def test_combined_score_shape(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        model.forward_with_actual(source, x_blanked, x_actual)
        score = model.get_score_tensor_for_sparsity()
        assert score is not None
        assert score.shape == (X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 5. Gradient routing of the self block
# ---------------------------------------------------------------------------


class TestParameterClassification:
    def test_self_block_qk_structural_gain_reconstruction(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        structural, reconstruction = classify_parameters(model)
        struct_ids = {id(p) for p in structural}
        recon_ids = {id(p) for p in reconstruction}

        named = dict(model.self_attention.named_parameters())
        # Q/K projections → structural
        for key in named:
            if "query_projection" in key or "key_projection" in key:
                assert id(named[key]) in struct_ids, f"{key} should be structural"
            if "gain_q_proj" in key or "gain_k_proj" in key:
                assert id(named[key]) in recon_ids, f"{key} should be reconstruction"
            if "value_projection" in key or "out_projection" in key:
                assert id(named[key]) in recon_ids, f"{key} should be reconstruction"


# ---------------------------------------------------------------------------
# 6. Oracle guard
# ---------------------------------------------------------------------------


class TestOracleGuard:
    def test_oracle_raises_in_split_mode(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        source, x_actual, x_blanked = _make_inputs()
        with pytest.raises(NotImplementedError):
            model.forward_with_actual(source, x_blanked, x_actual, oracle=True)


# ---------------------------------------------------------------------------
# 7. Backward smoke
# ---------------------------------------------------------------------------


class TestBackward:
    def test_self_block_receives_gradient(self):
        model = _make_model(self_attention_type="GatedSelfAttention")
        model.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()

        # At least one structural Q/K param of the self block must get a grad.
        got_grad = False
        for name, p in model.self_attention.named_parameters():
            if ("query_projection" in name or "key_projection" in name) and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    got_grad = True
        assert got_grad, "Self-block Q/K must receive a non-zero gradient."


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
