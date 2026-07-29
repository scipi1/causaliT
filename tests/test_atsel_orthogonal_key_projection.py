"""Tests for AttentionSelectorLayer ``key_projection_type="orthogonal"``.

Run with:  pytest tests/test_atsel_orthogonal_key_projection.py -v

Motivation
----------
``OrthogonalMaskEmbedding`` makes the *raw* keys of different variables
mutually orthogonal (disjoint dimension blocks).  But the attention logit is
computed on the *projected* keys ``k W_K``.  With an unconstrained ``W_K`` the
Gram matrix ``W_K^T W_K`` is an arbitrary SPD matrix, so

    <k_i W_K, k_j W_K> = k_i (W_K^T W_K) k_j^T

is generally non-zero even when ``k_i ⊥ k_j``.  The orthogonality is destroyed
at the projection step.

``key_projection_type="orthogonal"`` constrains ``W_K`` to an **isometry**
(``W_K^T W_K = I``) via ``OrthogonalLinear`` (Cayley parametrisation).  An
isometry preserves inner products, so orthogonal raw keys stay orthogonal after
projection.  These tests verify:

1. Construction wiring: ``key_projection`` is ``OrthogonalLinear`` iff the flag
   is set, and ``verify_orthonormality()`` holds (with ``orthogonal_key_scale``
   both on and off).
2. Guard: ``key_projection_type="orthogonal"`` with ``d_qk < d_model`` raises.
3. **Orthogonality preservation**: projecting two orthogonal raw keys keeps
   their inner product at (approximately) zero -- the core property.
4. **Inner-product preservation up to scale**: with the learnable scale, the
   projected inner products equal ``scale^2`` times the raw ones.
5. Forward pass is unchanged in shape (orthogonal embedding + orthogonal key
   projection), and the baseline ``"linear"`` uses a plain ``nn.Linear``.

Column convention (production): value at column 0, variable-ID at column 1.
``d_qk == d_model`` (isometry requires ``d_qk >= d_model``) and ``n_heads == 1``.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear


# ---------------------------------------------------------------------------
# Constants (d_qk == d_model so the orthogonal isometry is valid)
# ---------------------------------------------------------------------------

D_MODEL = 14        # 14 = 7 vars * 2 dims/var (exact tiling for orth embedding)
D_FF = 32
D_QK = 14           # must be >= d_model for orthogonal key projection
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_model(
    key_projection_type: str = "orthogonal",
    orthogonal_key_scale: bool = True,
    orthogonal_struct_embedding: bool = True,
    d_qk: int = D_QK,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotProduct",
        # MANDATORY since the legacy cross-only variant was removed.
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
        d_ff=D_FF,
        d_model=D_MODEL,
        d_qk=d_qk,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type=(
            "orthogonal_learnable"
            if orthogonal_struct_embedding
            else "standard_learnable"
        ),
        key_projection_type=key_projection_type,
        orthogonal_key_scale=orthogonal_key_scale,
    )


def _make_inputs():
    """(source, x_actual, x_blanked): value at col 0, 1-indexed var-ID at col 1."""
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
    def test_flag_stored(self):
        assert _make_model("orthogonal").key_projection_type == "orthogonal"
        assert _make_model("linear").key_projection_type == "linear"

    def test_orthogonal_uses_orthogonal_linear(self):
        m = _make_model("orthogonal")
        assert isinstance(m.attention.key_projection, OrthogonalLinear)

    def test_linear_uses_plain_linear(self):
        m = _make_model("linear")
        assert isinstance(m.attention.key_projection, nn.Linear)

    @pytest.mark.parametrize("use_scale", [True, False])
    def test_weight_is_orthonormal(self, use_scale):
        m = _make_model("orthogonal", orthogonal_key_scale=use_scale)
        # verify_orthonormality checks the UNSCALED weight columns W^T W = I.
        assert m.attention.key_projection.verify_orthonormality(atol=1e-4)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="key_projection_type"):
            _make_model("banana")


# ---------------------------------------------------------------------------
# 2. Isometry guard: d_qk >= d_model
# ---------------------------------------------------------------------------


class TestIsometryGuard:
    def test_dqk_lt_dmodel_raises(self):
        with pytest.raises(ValueError, match="d_qk >= d_model"):
            _make_model("orthogonal", d_qk=D_MODEL - 2)

    def test_linear_allows_small_dqk(self):
        # No isometry constraint for "linear" → smaller d_qk is fine.
        m = _make_model("linear", d_qk=D_MODEL - 2)
        assert isinstance(m.attention.key_projection, nn.Linear)


# ---------------------------------------------------------------------------
# 3. Orthogonality preservation (the core property)
# ---------------------------------------------------------------------------


class TestOrthogonalityPreservation:
    def test_orthogonal_raw_keys_stay_orthogonal_after_projection(self):
        """Different-variable keys (disjoint blocks) stay ⊥ after W_K."""
        m = _make_model("orthogonal", orthogonal_key_scale=True)
        m.eval()
        source, x_actual, _ = _make_inputs()

        with torch.no_grad():
            # Raw orthogonal key embeddings for S and X (disjoint blocks).
            s_keys = m.orth_embed_S(source)      # (B, L_S, d)
            x_keys = m.orth_embed_X(x_actual)    # (B, L_X, d)
            raw = torch.cat([s_keys, x_keys], dim=1)   # (B, L_S+L_X, d)

            # Sanity: raw keys of distinct variables are orthogonal.
            gram_raw = raw[0] @ raw[0].T               # (L, L)
            off_raw = gram_raw - torch.diag(torch.diag(gram_raw))
            assert torch.allclose(off_raw, torch.zeros_like(off_raw), atol=1e-5), (
                "Pre-condition failed: raw orthogonal-mask keys are not mutually "
                "orthogonal."
            )

            # Project through the shared (isometric) W_K.
            proj = m.attention.key_projection(raw)     # (B, L, d_qk)
            gram_proj = proj[0] @ proj[0].T
            off_proj = gram_proj - torch.diag(torch.diag(gram_proj))

        assert torch.allclose(off_proj, torch.zeros_like(off_proj), atol=1e-4), (
            "Projected keys must remain mutually orthogonal when W_K is an "
            "isometry (W_K^T W_K = I)."
        )

    def test_linear_projection_generally_breaks_orthogonality(self):
        """Contrast: an unconstrained W_K generally destroys orthogonality."""
        m = _make_model("linear")
        m.eval()
        # Perturb the linear weights away from any accidental isometry.
        with torch.no_grad():
            m.attention.key_projection.weight += (
                torch.randn_like(m.attention.key_projection.weight) * 0.5
            )
        source, x_actual, _ = _make_inputs()
        with torch.no_grad():
            raw = torch.cat([m.orth_embed_S(source), m.orth_embed_X(x_actual)], dim=1)
            proj = m.attention.key_projection(raw)
            gram = proj[0] @ proj[0].T
            off = gram - torch.diag(torch.diag(gram))
        assert off.abs().max() > 1e-3, (
            "Unconstrained linear W_K is expected to mix the disjoint blocks and "
            "break orthogonality of the projected keys."
        )


# ---------------------------------------------------------------------------
# 4. Inner-product preservation up to scale^2
# ---------------------------------------------------------------------------


class TestInnerProductPreservation:
    def test_isometry_preserves_inner_products_up_to_scale(self):
        m = _make_model("orthogonal", orthogonal_key_scale=True)
        m.eval()
        proj_layer = m.attention.key_projection
        scale = proj_layer.get_scale()

        x = torch.randn(5, D_MODEL)
        y = torch.randn(5, D_MODEL)
        with torch.no_grad():
            wx = proj_layer(x)
            wy = proj_layer(y)
            lhs = (wx * wy).sum(dim=-1)              # <Wx, Wy>
            rhs = (scale ** 2) * (x * y).sum(dim=-1)  # scale^2 <x, y>
        assert torch.allclose(lhs, rhs, atol=1e-3), (
            "An isometry (with scalar scale s) must satisfy "
            "<Wx, Wy> = s^2 <x, y>."
        )


# ---------------------------------------------------------------------------
# 5. Forward pass smoke
# ---------------------------------------------------------------------------


class TestForward:
    @pytest.mark.parametrize("orth_embed", [True, False])
    def test_forward_shapes_with_orthogonal_projection(self, orth_embed):
        m = _make_model(
            "orthogonal",
            orthogonal_struct_embedding=orth_embed,
        )
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_backward_reaches_key_projection(self):
        m = _make_model("orthogonal")
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()
        skew = m.attention.key_projection.skew_params
        assert skew.grad is not None and skew.grad.abs().sum() > 0, (
            "The Cayley skew parameters of the orthogonal W_K must receive a "
            "non-zero gradient."
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
