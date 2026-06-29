"""
Tests for batch_key_dropout (BKD) support in AttentionSelectorLayer.

Verifies:
1. AttentionSelectorLayer accepts BKD constructor params without error.
2. The inner attention module has a non-None BatchConsistentKeyDropout sub-module
   when batch_key_dropout is set.
3. BKD is None when batch_key_dropout=None (default / disabled).
4. A forward pass with BKD enabled completes without error and produces the
   correct output shape.
5. _build_stage_config in anm_staged_trainer now applies batch_key_dropout_p
   to AttentionSelectorLayer kwargs (regression test for the previous skip).
"""

import pytest
import torch

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer


# ---------------------------------------------------------------------------
# Minimal factory helper
# ---------------------------------------------------------------------------

D_MODEL = 16
# vocab sizes: must be larger than seq_len so variable IDs are valid
VOCAB_S = 8
VOCAB_X = 8


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Standard summation embedding config (variable ID + scalar value)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": 0,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
            {
                "idx": 1,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
        ],
    }


def _make_atsel(
    S_seq_len: int = 3,
    X_seq_len: int = 4,
    d_model: int = D_MODEL,
    batch_key_dropout: float = None,
    batch_key_dropout_p_final: float = None,
    batch_key_dropout_annealing_batches: int = None,
) -> AttentionSelectorLayer:
    """Build a minimal AttentionSelectorLayer for testing."""
    return AttentionSelectorLayer(
        model="AttentionSelectorLayer",
        ds_embed_S=_embed_cfg(VOCAB_S, d_model),
        ds_embed_X=_embed_cfg(VOCAB_X, d_model),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="CausalCrossAttention",
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
        d_model=d_model,
        d_qk=d_model,
        S_seq_len=S_seq_len,
        X_seq_len=X_seq_len,
        batch_key_dropout=batch_key_dropout,
        batch_key_dropout_p_final=batch_key_dropout_p_final,
        batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
    )


def _make_inputs(S_len: int = 3, X_len: int = 4, batch: int = 4):
    """
    Create minimal (S, X, X_blanked) tensors.
    Feature layout: [:, :, 0] = variable ID (int as float), [:, :, 1] = value.
    """
    S = torch.zeros(batch, S_len, 2)
    S[:, :, 0] = torch.randint(1, VOCAB_S, (batch, S_len)).float()
    S[:, :, 1] = torch.randn(batch, S_len)

    X = torch.zeros(batch, X_len, 2)
    X[:, :, 0] = torch.randint(1, VOCAB_X, (batch, X_len)).float()
    X[:, :, 1] = torch.randn(batch, X_len)

    X_blanked = X.clone()
    X_blanked[:, :, 1] = 0.0  # blank value column

    return S, X, X_blanked


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAttentionSelectorLayerBKD:

    def test_bkd_disabled_by_default(self):
        """batch_key_dropout=None → inner attention has no BKD sub-module."""
        model = _make_atsel(batch_key_dropout=None)
        inner = model.attention.inner_attention
        bkd = getattr(inner, "batch_key_dropout", "ATTR_MISSING")
        # When BKD is disabled the attribute should be None (not missing).
        assert bkd is None, (
            f"Expected batch_key_dropout=None on inner attention, got {bkd!r}"
        )

    def test_bkd_enabled_creates_sub_module(self):
        """batch_key_dropout=0.5 → inner attention has a BatchConsistentKeyDropout."""
        from causaliT.core.modules.attention import BatchConsistentKeyDropout  # noqa: F401 (import check)

        model = _make_atsel(batch_key_dropout=0.5)
        inner = model.attention.inner_attention
        bkd = getattr(inner, "batch_key_dropout", None)
        assert bkd is not None, (
            "Expected a BatchConsistentKeyDropout sub-module on inner_attention "
            "when batch_key_dropout=0.5, but got None."
        )
        # Verify it's the right type.
        assert hasattr(bkd, "p_init"), (
            f"BKD sub-module {bkd!r} does not look like BatchConsistentKeyDropout "
            "(missing 'p_init' attribute)."
        )

    def test_bkd_p_init_stored_correctly(self):
        """The p_init value stored on BKD matches the constructor argument."""
        model = _make_atsel(batch_key_dropout=0.3)
        inner = model.attention.inner_attention
        bkd = inner.batch_key_dropout
        assert bkd is not None
        assert abs(float(bkd.p_init) - 0.3) < 1e-6, (
            f"Expected bkd.p_init=0.3, got {bkd.p_init}"
        )

    def test_forward_with_bkd_enabled(self):
        """Forward pass completes and returns correct output shape with BKD."""
        B, S_len, X_len = 4, 3, 4
        model = _make_atsel(
            S_seq_len=S_len,
            X_seq_len=X_len,
            batch_key_dropout=0.5,
        )
        model.eval()

        S, X, X_blanked = _make_inputs(S_len=S_len, X_len=X_len, batch=B)

        pred_x, attn_weights, _aux = model.forward_with_actual(
            source_tensor=S,
            x_blanked=X_blanked,
            x_actual=X,
        )

        assert pred_x.shape == (B, X_len, 1), (
            f"Expected pred_x shape ({B}, {X_len}, 1), got {pred_x.shape}"
        )
        assert attn_weights.shape == (B, X_len, S_len + X_len), (
            f"Expected attn_weights shape ({B}, {X_len}, {S_len + X_len}), "
            f"got {attn_weights.shape}"
        )

    def test_forward_without_bkd(self):
        """Forward pass without BKD is unaffected (baseline sanity check)."""
        B, S_len, X_len = 4, 3, 4
        model = _make_atsel(S_seq_len=S_len, X_seq_len=X_len, batch_key_dropout=None)
        model.eval()

        S, X, X_blanked = _make_inputs(S_len=S_len, X_len=X_len, batch=B)

        pred_x, attn_weights, _aux = model.forward_with_actual(
            source_tensor=S,
            x_blanked=X_blanked,
            x_actual=X,
        )
        assert pred_x.shape == (B, X_len, 1)

    def test_bkd_zero_prob_is_identity(self):
        """batch_key_dropout=0.0 registers BKD with p_init=0 (no masking)."""
        model = _make_atsel(batch_key_dropout=0.0)
        inner = model.attention.inner_attention
        bkd = inner.batch_key_dropout
        assert bkd is not None
        assert abs(float(bkd.p_init) - 0.0) < 1e-6


class TestBuildStageConfigBKDForAtsel:
    """
    Regression tests: _build_stage_config must now apply batch_key_dropout_p
    to AttentionSelectorLayer (previously it was silently skipped).
    """

    def _make_base_config(self):
        return {
            "model": {
                "model_object": "AttentionSelectorLayer",
                "kwargs": {
                    "batch_key_dropout": None,
                    "batch_key_dropout_p_final": None,
                    "batch_key_dropout_annealing_batches": None,
                },
            },
            "training": {
                "max_epochs": 10,
                "k_fold": 1,
            },
        }

    def test_bkd_p_applied_to_atsel(self):
        """batch_key_dropout_p in stage_spec is now forwarded to model kwargs."""
        from causaliT.training.anm_staged_trainer import _build_stage_config

        base = self._make_base_config()
        stage_spec = {"name": "test_stage", "batch_key_dropout_p": 0.6}

        result = _build_stage_config(base, stage_spec, stage_idx=0)

        mk = result["model"]["kwargs"]
        assert mk["batch_key_dropout"] == pytest.approx(0.6), (
            f"Expected batch_key_dropout=0.6, got {mk['batch_key_dropout']}"
        )
        assert mk["batch_key_dropout_p_final"] == pytest.approx(0.6), (
            f"Expected batch_key_dropout_p_final=0.6 (no intra-stage annealing), "
            f"got {mk['batch_key_dropout_p_final']}"
        )
        assert mk["batch_key_dropout_annealing_batches"] is None, (
            "Expected batch_key_dropout_annealing_batches=None (step annealing "
            f"disabled), got {mk['batch_key_dropout_annealing_batches']}"
        )

    def test_bkd_p_not_set_leaves_kwargs_unchanged(self):
        """If stage_spec has no batch_key_dropout_p, model kwargs are untouched."""
        from causaliT.training.anm_staged_trainer import _build_stage_config

        base = self._make_base_config()
        stage_spec = {"name": "test_stage"}

        result = _build_stage_config(base, stage_spec, stage_idx=0)

        mk = result["model"]["kwargs"]
        assert mk["batch_key_dropout"] is None
        assert mk["batch_key_dropout_p_final"] is None
        assert mk["batch_key_dropout_annealing_batches"] is None
