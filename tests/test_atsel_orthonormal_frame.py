"""Tests for the fixed dense orthonormal structural embedding (``orthogonal_fixed``).

Run with:  pytest tests/test_atsel_orthonormal_frame.py -v

Motivation
----------
``OrthogonalMaskEmbedding`` guarantees cross-variable orthogonality with disjoint
binary blocks -- each variable lives on ``d_model // n_vars`` axis-aligned
dimensions, and ``d_model % n_vars`` dimensions are left idle.  The observed
downside is limited expressivity / "wasted" dimensions.

``FixedOrthonormalEmbedding`` (selected via ``struct_embedding_type="orthogonal_fixed"``)
keeps EXACT cross-variable orthogonality while using dense frozen rows that span
the FULL ``d_model`` space (no idle dims), analogous to fixed sinusoidal positional
encodings but with a genuinely orthonormal frame.  S and X share ONE frame
(same seed + total_variables) via disjoint row slices, so all ``L_S + L_X`` rows
are pairwise orthonormal -- including S-vs-X.

These tests verify:

1. **Frame orthonormality**: ``frame @ frame^T = scale^2 I`` for a single group,
   and the concatenated S+X frame is mutually orthonormal (joint S-perp-X).
2. **Frozen**: the module has no trainable parameters and the frame buffer does
   not change after a backward/optimizer step.
3. **Value-independence**: the output depends only on the variable ID, not the
   value column.
4. **Determinism**: ``"dct"`` frames are seed-independent; ``"random"`` frames are
   reproducible given a seed and vary across seeds.
5. **Guards**: ``total_variables > d_model``, bad ``frame_type``, bad ``row_offset``.
6. **Model wiring**: ``struct_embedding_type="orthogonal_fixed"`` populates
   ``orth_embed_{S,X}`` with ``FixedOrthonormalEmbedding``; forward shapes are
   unchanged; the model's S and X frames are mutually orthogonal; an invalid
   ``struct_embedding_type`` raises.
7. **Korth preservation**: projecting the orthonormal raw keys through an
   isometric ``W_K`` keeps them orthogonal.

Column convention (production): value at column 0, variable-ID at column 1.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules import FixedOrthonormalEmbedding
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16          # >= d_model for the orthogonal (isometric) key projection
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


def _var_input(num_vars: int, batch: int = BATCH, value=None) -> torch.Tensor:
    """(batch, num_vars, 2) tensor: value at col 0, 1-indexed var-ID at col 1."""
    x = torch.zeros(batch, num_vars, 2)
    if value is None:
        x[:, :, VALUE_COL] = torch.randn(batch, num_vars)
    else:
        x[:, :, VALUE_COL] = value
    x[:, :, VAR_COL] = torch.arange(1, num_vars + 1).float().unsqueeze(0).repeat(batch, 1)
    return x


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
    orthogonal_fixed: bool = True,
    orthogonal_struct_embedding: bool = False,
    key_projection_type: str = "linear",
    frame_type: str = "random",
    d_qk: int = D_QK,
    struct_embedding_type: str = None,
) -> AttentionSelectorLayer:
    # Translate the legacy boolean helper flags to the single struct_embedding_type
    # key.  ``struct_embedding_type`` (when passed explicitly) overrides the flags,
    # allowing tests to inject invalid values.
    if struct_embedding_type is None:
        if orthogonal_fixed:
            struct_embedding_type = "orthogonal_fixed"
        elif orthogonal_struct_embedding:
            struct_embedding_type = "orthogonal_learnable"
        else:
            struct_embedding_type = "standard_learnable"
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotProduct",
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
        struct_embedding_type=struct_embedding_type,
        orthogonal_fixed_frame_type=frame_type,
        key_projection_type=key_projection_type,
    )


def _make_inputs():
    source = _var_input(S_SEQ_LEN)
    x_actual = _var_input(X_SEQ_LEN)
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Frame orthonormality
# ---------------------------------------------------------------------------


class TestFrameOrthonormality:
    @pytest.mark.parametrize("frame_type", ["random", "dct"])
    def test_single_group_frame_is_orthonormal(self, frame_type):
        emb = FixedOrthonormalEmbedding(
            num_variables=5, d_model=D_MODEL, frame_type=frame_type, seed=7
        )
        gram = emb.frame @ emb.frame.T
        assert torch.allclose(gram, torch.eye(5), atol=1e-5), (
            f"{frame_type} frame rows must be orthonormal (frame @ frame^T = I)."
        )

    def test_scale_gives_scale_squared_identity(self):
        scale = 2.5
        emb = FixedOrthonormalEmbedding(
            num_variables=4, d_model=D_MODEL, seed=1, scale=scale
        )
        gram = emb.frame @ emb.frame.T
        assert torch.allclose(gram, (scale ** 2) * torch.eye(4), atol=1e-4), (
            "Scaling rows by s must give frame @ frame^T = s^2 I."
        )

    def test_joint_S_X_frame_is_mutually_orthonormal(self):
        """Two instances sharing seed+total build disjoint slices of one frame."""
        total = S_SEQ_LEN + X_SEQ_LEN
        emb_S = FixedOrthonormalEmbedding(
            num_variables=S_SEQ_LEN, d_model=D_MODEL,
            total_variables=total, row_offset=0, seed=42,
        )
        emb_X = FixedOrthonormalEmbedding(
            num_variables=X_SEQ_LEN, d_model=D_MODEL,
            total_variables=total, row_offset=S_SEQ_LEN, seed=42,
        )
        full = torch.cat([emb_S.frame, emb_X.frame], dim=0)   # (total, d)
        gram = full @ full.T
        assert torch.allclose(gram, torch.eye(total), atol=1e-5), (
            "All L_S + L_X rows (including S-vs-X) must be mutually orthonormal."
        )

    def test_different_row_offset_different_seed_breaks_joint_orthogonality(self):
        """Sanity: independent frames (different seeds) are NOT jointly orthogonal."""
        total = S_SEQ_LEN + X_SEQ_LEN
        emb_S = FixedOrthonormalEmbedding(
            num_variables=S_SEQ_LEN, d_model=D_MODEL,
            total_variables=total, row_offset=0, seed=1,
        )
        emb_X = FixedOrthonormalEmbedding(
            num_variables=X_SEQ_LEN, d_model=D_MODEL,
            total_variables=total, row_offset=S_SEQ_LEN, seed=999,
        )
        full = torch.cat([emb_S.frame, emb_X.frame], dim=0)
        gram = full @ full.T
        off = gram - torch.diag(torch.diag(gram))
        assert off.abs().max() > 1e-3, (
            "Independent frames (different seeds) should NOT be jointly orthogonal."
        )


# ---------------------------------------------------------------------------
# 2. Frozen (no trainable parameters, buffer unchanged after a step)
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_no_trainable_parameters(self):
        emb = FixedOrthonormalEmbedding(num_variables=4, d_model=D_MODEL, seed=0)
        assert list(emb.parameters()) == [], (
            "FixedOrthonormalEmbedding must have no trainable parameters (frame is a buffer)."
        )

    def test_frame_is_a_buffer(self):
        emb = FixedOrthonormalEmbedding(num_variables=4, d_model=D_MODEL, seed=0)
        buffer_names = {n for n, _ in emb.named_buffers()}
        assert "frame" in buffer_names

    def test_frame_unchanged_after_backward_step(self):
        """A downstream loss must not alter the frame (no grad flows into it)."""
        m = _make_model(orthogonal_fixed=True)
        before_S = m.orth_embed_S.frame.clone()
        before_X = m.orth_embed_X.frame.clone()
        m.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = m.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1.0)
        opt.step()
        assert torch.equal(m.orth_embed_S.frame, before_S)
        assert torch.equal(m.orth_embed_X.frame, before_X)


# ---------------------------------------------------------------------------
# 3. Value independence (identity-only embedding)
# ---------------------------------------------------------------------------


class TestValueIndependence:
    def test_output_ignores_value_column(self):
        emb = FixedOrthonormalEmbedding(num_variables=4, d_model=D_MODEL, seed=3)
        x_a = _var_input(4, value=1.0)
        x_b = _var_input(4, value=-7.5)   # same var IDs, different values
        out_a = emb(x_a)
        out_b = emb(x_b)
        assert torch.equal(out_a, out_b), (
            "FixedOrthonormalEmbedding must depend only on variable identity."
        )

    def test_same_var_id_maps_to_same_row(self):
        emb = FixedOrthonormalEmbedding(num_variables=4, d_model=D_MODEL, seed=3)
        x = _var_input(4)
        out = emb(x)
        # Row for var-ID 1 (index 0) equals frame[0] across the batch.
        assert torch.allclose(out[0, 0], emb.frame[0], atol=1e-6)
        assert torch.allclose(out[1, 0], emb.frame[0], atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_random_frame_reproducible_for_seed(self):
        a = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL, seed=123)
        b = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL, seed=123)
        assert torch.equal(a.frame, b.frame)

    def test_random_frame_varies_with_seed(self):
        a = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL, seed=1)
        b = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL, seed=2)
        assert not torch.equal(a.frame, b.frame)

    def test_dct_frame_seed_independent(self):
        a = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL,
                                      frame_type="dct", seed=1)
        b = FixedOrthonormalEmbedding(num_variables=6, d_model=D_MODEL,
                                      frame_type="dct", seed=99999)
        assert torch.equal(a.frame, b.frame), "DCT frame must be seed-independent."


# ---------------------------------------------------------------------------
# 5. Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_total_variables_gt_d_model_raises(self):
        with pytest.raises(ValueError, match="d_model >= total_variables"):
            FixedOrthonormalEmbedding(num_variables=20, d_model=8, seed=0)

    def test_bad_frame_type_raises(self):
        with pytest.raises(ValueError, match="frame_type"):
            FixedOrthonormalEmbedding(num_variables=3, d_model=D_MODEL,
                                      frame_type="banana", seed=0)

    def test_row_offset_overflow_raises(self):
        with pytest.raises(ValueError, match="row_offset"):
            FixedOrthonormalEmbedding(num_variables=4, d_model=D_MODEL,
                                      total_variables=5, row_offset=3, seed=0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            FixedOrthonormalEmbedding(num_variables=3, d_model=D_MODEL,
                                      seed=0, scale=-1.0)


# ---------------------------------------------------------------------------
# 6. Model wiring
# ---------------------------------------------------------------------------


class TestModelWiring:
    def test_flag_stored_and_modules_are_fixed_orthonormal(self):
        m = _make_model(orthogonal_fixed=True)
        assert m.struct_embedding_type == "orthogonal_fixed"
        assert m.orthogonal_fixed is True
        assert isinstance(m.orth_embed_S, FixedOrthonormalEmbedding)
        assert isinstance(m.orth_embed_X, FixedOrthonormalEmbedding)

    def test_off_leaves_orth_embeds_none(self):
        m = _make_model(orthogonal_fixed=False)
        assert m.struct_embedding_type == "standard_learnable"
        assert m.orth_embed_S is None and m.orth_embed_X is None

    def test_model_S_and_X_frames_are_mutually_orthogonal(self):
        m = _make_model(orthogonal_fixed=True)
        full = torch.cat([m.orth_embed_S.frame, m.orth_embed_X.frame], dim=0)
        gram = full @ full.T
        assert torch.allclose(gram, torch.eye(S_SEQ_LEN + X_SEQ_LEN), atol=1e-5)

    def test_invalid_struct_embedding_type_raises(self):
        # The single struct_embedding_type key replaces the previous pair of
        # mutually-exclusive booleans; an unknown value must raise.
        with pytest.raises(ValueError, match="struct_embedding_type"):
            _make_model(struct_embedding_type="orthogonal_both")

    def test_forward_shapes(self):
        m = _make_model(orthogonal_fixed=True)
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_forward_runs_with_dct_frame(self):
        m = _make_model(orthogonal_fixed=True, frame_type="dct")
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = m.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# 7. Korth preservation (isometric W_K keeps orthogonal keys orthogonal)
# ---------------------------------------------------------------------------


class TestKorthPreservation:
    def test_orthogonal_raw_keys_stay_orthogonal_after_projection(self):
        m = _make_model(orthogonal_fixed=True, key_projection_type="orthogonal")
        assert isinstance(m.attention.key_projection, OrthogonalLinear)
        m.eval()
        source, x_actual, _ = _make_inputs()
        with torch.no_grad():
            raw = torch.cat([m.orth_embed_S(source), m.orth_embed_X(x_actual)], dim=1)
            # Pre-condition: raw dense keys of distinct variables are orthogonal.
            gram_raw = raw[0] @ raw[0].T
            off_raw = gram_raw - torch.diag(torch.diag(gram_raw))
            assert off_raw.abs().max() < 1e-4

            proj = m.attention.key_projection(raw)
            gram_proj = proj[0] @ proj[0].T
            off_proj = gram_proj - torch.diag(torch.diag(gram_proj))
        assert off_proj.abs().max() < 1e-4, (
            "Isometric W_K must keep the dense orthonormal keys mutually orthogonal."
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
