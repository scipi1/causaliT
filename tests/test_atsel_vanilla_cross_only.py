"""Tests for the VANILLA-TRANSFORMER arm of ``AttentionSelectorLayer``.

Run with:  pytest tests/test_atsel_vanilla_cross_only.py -v

The vanilla benchmark is ``self_attention_type=None`` +
``attention_type="ScaledDotSoftmax"`` + ``comps_embed="summation"``: ONE
combined block whose keys/values are ``[S_actual ; X_actual]``, so a SINGLE
softmax normalises over the S and X candidate parents JOINTLY.  Two separate
softmax blocks would produce two independently normalised simplices that need a
fusion rule, which is a modelling choice and not part of a vanilla transformer.

These tests verify:

1. Construction wiring (cross-only flags, single combined mask, no self block).
2. Forward shapes and the canonical ``(B, L_X, L_S+L_X)`` posterior layout.
3. Softmax semantics: rows sum to 1 and the X-block diagonal is exactly 0
   (a node may not attend to itself).
4. No structural machinery: the sparsity score is None, so the L0 / NOTEARS
   terms are inert.
5. Backward smoke: gradients reach the Q/K projections.
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer


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


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
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


def _make_vanilla(n_heads: int = 1) -> AttentionSelectorLayer:
    """The exact benchmark configuration: cross-only + softmax + summation."""
    return AttentionSelectorLayer(
        model="vanilla",
        ds_embed_S=_embed_cfg(VOCAB_S),
        ds_embed_X=_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotSoftmax",
        self_attention_type=None,
        n_heads=n_heads,
        dropout_emb=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        dropout_qkv=0.0,
        attention_dropout=0.0,
        activation="gelu",
        norm="layer",
        use_final_norm=True,
        device="cpu",
        out_dim=1,
        d_ff=D_FF,
        d_model=D_MODEL,
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        struct_embedding_type="standard_learnable",
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
# 1. Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_cross_only_flags(self):
        m = _make_vanilla()
        assert m.cross_only is True
        assert m.split_xx is False
        assert m.homogeneous_nodes is False
        assert m.self_attention is None

    def test_single_combined_mask(self):
        m = _make_vanilla()
        names = {n for n, _ in m.named_buffers()}
        assert "combined_mask" in names
        assert "cross_mask" not in names and "self_mask" not in names

        mask = m.combined_mask
        assert mask.shape == (X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        # S block fully connected, X block off-diagonal.
        assert torch.all(mask[:, :S_SEQ_LEN] == 1.0)
        xx = mask[:, S_SEQ_LEN:]
        assert torch.equal(torch.diagonal(xx), torch.zeros(X_SEQ_LEN))

    def test_inner_attention_is_vanilla_softmax(self):
        from causaliT.core.modules import ScaledDotSoftmax
        m = _make_vanilla()
        assert isinstance(m.attention.inner_attention, ScaledDotSoftmax)


# ---------------------------------------------------------------------------
# 2. Forward shapes
# ---------------------------------------------------------------------------


class TestForward:
    def test_shapes(self):
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)
        assert isinstance(aux, dict)

    def test_split_attention_round_trip(self):
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        att_sx, att_xx = m.split_attention(attn)
        assert att_sx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert att_xx.shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        assert torch.allclose(torch.cat([att_sx, att_xx], dim=-1), attn)

    def test_multihead_shapes(self):
        m = _make_vanilla(n_heads=4)
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape[-1] == S_SEQ_LEN + X_SEQ_LEN


# ---------------------------------------------------------------------------
# 3. Softmax semantics: joint normalisation over S and X
# ---------------------------------------------------------------------------


class TestSoftmaxSemantics:
    def test_rows_sum_to_one(self):
        """S and X parents compete on ONE simplex (the point of cross-only)."""
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_no_self_loops(self):
        """The X-block diagonal is exactly zero: a node cannot copy itself."""
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        xx = attn[:, :, S_SEQ_LEN:]
        assert torch.all(torch.diagonal(xx, dim1=1, dim2=2) == 0.0)

    def test_attention_is_non_negative(self):
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        _, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert torch.all(attn >= 0.0)


# ---------------------------------------------------------------------------
# 4. No structural machinery
# ---------------------------------------------------------------------------


class TestNoStructuralMachinery:
    def test_sparsity_score_is_none(self):
        """Vanilla softmax publishes no score, so L0 / NOTEARS terms are inert."""
        m = _make_vanilla()
        m.eval()
        source, x_actual, x_blanked = _make_inputs()
        m.forward_with_actual(source, x_blanked, x_actual)
        assert m.get_score_tensor_for_sparsity() is None

    def test_no_value_structure_injection(self):
        m = _make_vanilla()
        assert m.inject_value_structure is False
        assert m.inject_value_structure_query is False
        assert m.val_id_embed_S is None and m.val_id_embed_X is None

    def test_no_gain_stream(self):
        m = _make_vanilla()
        assert m.is_gated is False
        # The reconstruction-gain stream has been removed entirely.
        assert not hasattr(m, "gain_q_embed_X")
        assert not hasattr(m, "gain_k_embed_S")


# ---------------------------------------------------------------------------
# 5. Backward smoke
# ---------------------------------------------------------------------------


class TestBackward:
    def test_gradients_reach_qk(self):
        m = _make_vanilla()
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()

        got_grad = False
        for name, p in m.attention.named_parameters():
            if ("query_projection" in name or "key_projection" in name):
                if p.grad is not None and p.grad.abs().sum() > 0:
                    got_grad = True
        assert got_grad, "Q/K projections must receive a non-zero gradient."


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
