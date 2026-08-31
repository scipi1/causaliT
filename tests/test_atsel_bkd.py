"""
Tests for batch_key_dropout (BKD) support in AttentionSelectorLayer.

Verifies:
1. AttentionSelectorLayer accepts BKD constructor params without error.
2. The inner attention module has a non-None BatchConsistentKeyDropout sub-module
   when batch_key_dropout is set.
3. BKD is None when batch_key_dropout=None (default / disabled).
4. A forward pass with BKD enabled completes without error and produces the
   correct output shape.
6. BKD stage-transition state_dict compatibility:
   - BatchConsistentKeyDropout._step_count is only in state_dict when annealing
     is active (Fix 1).
   - AttentionSelectorForecaster.on_load_checkpoint strips unexpected BKD keys
     from legacy checkpoints (Fix 2, recon→struct direction).
   - on_load_checkpoint fills missing BKD keys from the current model
     (Fix 2, struct→recon direction).
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


# ---------------------------------------------------------------------------
# Helper: build a minimal AttentionSelectorForecaster config dict
# ---------------------------------------------------------------------------

def _make_forecaster_config(
    S_seq_len: int = 3,
    X_seq_len: int = 3,
    d_model: int = D_MODEL,
    batch_key_dropout=None,
    batch_key_dropout_p_final=None,
    batch_key_dropout_annealing_batches=None,
) -> dict:
    """Return a minimal config dict accepted by AttentionSelectorForecaster.__init__."""
    return {
        "data": {
            "val_idx": 1,           # feature index 1 = value column
            "S_seq_len": S_seq_len,
            "X_seq_len": X_seq_len,
            "dataset": "dummy",
        },
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(VOCAB_S, d_model),
                "ds_embed_X": _embed_cfg(VOCAB_X, d_model),
                "comps_embed_S": "summation",
                "comps_embed_X": "summation",
                "attention_type": "CausalCrossAttention",
                # MANDATORY since the legacy cross-only variant was removed.
                "self_attention_type": "GatedSelfAttention",
                "n_heads": 1,

                "dropout_emb": 0.0,
                "dropout_attn_out": 0.0,
                "dropout_ff": 0.0,
                "dropout_qkv": 0.0,
                "attention_dropout": 0.0,
                "activation": "relu",
                "norm": "layer",
                "use_final_norm": False,
                "device": "cpu",
                "out_dim": 1,
                "d_ff": 32,
                "d_model": d_model,
                "d_qk": d_model,
                "S_seq_len": S_seq_len,
                "X_seq_len": X_seq_len,
                "batch_key_dropout": batch_key_dropout,
                "batch_key_dropout_p_final": batch_key_dropout_p_final,
                "batch_key_dropout_annealing_batches": batch_key_dropout_annealing_batches,
            },
        },
        "training": {
            "loss_fn": "mse",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "use_gradient_routing": False,
            "lambda_recon": 1.0,
            "lambda_hsic": 0.0,
            "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0,
            "lambda_l0": 0.0,
            "kappa": 0.0,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": False,
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "use_oracle_attention": False,
            "use_hard_masks": False,
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }


class TestBKDStageTransitionStateDict:
    """
    Regression tests for the BKD stage-transition state_dict mismatch fix.

    The original bug: loading a checkpoint saved by a stage with
    ``batch_key_dropout=0.8`` into a stage model with ``batch_key_dropout=null``
    raised::

        RuntimeError: unexpected key(s) in state_dict:
            "model.attention.inner_attention.batch_key_dropout._step_count"

    Fix 1 (BatchConsistentKeyDropout): ``_step_count`` is no longer registered
    as a buffer when step-counter annealing is disabled — new checkpoints won't
    contain the key at all.

    Fix 2 (AttentionSelectorForecaster.on_load_checkpoint): legacy checkpoints
    that still contain the key are handled gracefully by stripping (or filling)
    any ``batch_key_dropout.*`` key that doesn't match the current model.
    """

    # ------------------------------------------------------------------
    # Fix 1: BatchConsistentKeyDropout state_dict behaviour
    # ------------------------------------------------------------------

    def test_bkd_no_annealing_step_count_not_in_state_dict(self):
        """
        When annealing is disabled (annealing_batches=None), _step_count must
        NOT appear in state_dict.  This is the root-cause fix: checkpoints
        saved after this change will never contain the offending key.
        """
        from causaliT.core.modules.extra_layers import BatchConsistentKeyDropout

        bkd = BatchConsistentKeyDropout(p_init=0.8)
        sd_keys = list(bkd.state_dict().keys())
        assert "_step_count" not in sd_keys, (
            f"_step_count should NOT be in state_dict when annealing is off, "
            f"but found keys: {sd_keys}"
        )

    def test_bkd_with_annealing_step_count_in_state_dict(self):
        """
        When annealing IS active (p_final + annealing_batches both set),
        _step_count MUST appear in state_dict so that annealing progress
        is preserved across checkpoint save/load.
        """
        from causaliT.core.modules.extra_layers import BatchConsistentKeyDropout

        bkd = BatchConsistentKeyDropout(
            p_init=0.8, p_final=0.0, annealing_batches=1000
        )
        sd_keys = list(bkd.state_dict().keys())
        assert "_step_count" in sd_keys, (
            f"_step_count MUST be in state_dict when annealing is active, "
            f"but found keys: {sd_keys}"
        )

    # ------------------------------------------------------------------
    # Fix 2: on_load_checkpoint — recon→struct direction (unexpected key)
    # ------------------------------------------------------------------

    def test_on_load_checkpoint_strips_unexpected_bkd_keys(self):
        """
        Core regression for the H9 crash.

        Simulates loading a legacy checkpoint (saved by a stage with BKD enabled,
        before Fix 1) into a stage model with batch_key_dropout=None.  The
        checkpoint contains 'model.attention.inner_attention.batch_key_dropout._step_count'.

        After on_load_checkpoint the key must be stripped so that
        load_state_dict(strict=True) no longer raises RuntimeError.
        """
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        # Stage 2 model: BKD disabled (no batch_key_dropout module)
        model_no_bkd = AttentionSelectorForecaster(
            _make_forecaster_config(batch_key_dropout=None)
        )

        # Simulate a legacy checkpoint from Stage 1 (BKD enabled, pre-Fix 1).
        # The key that caused the crash is injected manually.
        legacy_ckpt_state = dict(model_no_bkd.state_dict())  # base keys are correct
        offending_key = (
            "model.attention.inner_attention.batch_key_dropout._step_count"
        )
        legacy_ckpt_state[offending_key] = torch.tensor(42, dtype=torch.long)

        fake_checkpoint = {"state_dict": legacy_ckpt_state}

        # This must NOT raise.
        model_no_bkd.on_load_checkpoint(fake_checkpoint)

        # The offending key must have been removed.
        assert offending_key not in fake_checkpoint["state_dict"], (
            "on_load_checkpoint did not strip the unexpected BKD key."
        )

        # After stripping, strict load must succeed.
        model_no_bkd.load_state_dict(fake_checkpoint["state_dict"], strict=True)

    # ------------------------------------------------------------------
    # Fix 2: on_load_checkpoint — struct→recon direction (missing key)
    # ------------------------------------------------------------------

    def test_on_load_checkpoint_fills_missing_bkd_keys(self):
        """
        Symmetric direction: loading a checkpoint saved by a no-BKD stage into
        a model with BKD + annealing enabled.

        The checkpoint is missing 'model.attention.inner_attention.batch_key_dropout._step_count'.
        on_load_checkpoint must fill it from the current model's state_dict so that
        load_state_dict(strict=True) succeeds (step counter resets to 0).
        """
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        # Stage 3 model: BKD enabled WITH annealing (so _step_count IS a buffer).
        model_with_bkd = AttentionSelectorForecaster(
            _make_forecaster_config(
                batch_key_dropout=0.6,
                batch_key_dropout_p_final=0.0,
                batch_key_dropout_annealing_batches=1000,
            )
        )

        bkd_key = (
            "model.attention.inner_attention.batch_key_dropout._step_count"
        )
        # Verify the model actually has the buffer (sanity check for test validity).
        assert bkd_key in model_with_bkd.state_dict(), (
            f"Test setup error: expected {bkd_key!r} in state_dict of "
            "BKD+annealing model, but it is missing."
        )

        # Checkpoint from Stage 2 (no BKD) — does NOT contain the buffer.
        ckpt_state = {
            k: v
            for k, v in model_with_bkd.state_dict().items()
            if "batch_key_dropout" not in k
        }
        fake_checkpoint = {"state_dict": ckpt_state}

        # on_load_checkpoint must fill the missing BKD key.
        model_with_bkd.on_load_checkpoint(fake_checkpoint)

        assert bkd_key in fake_checkpoint["state_dict"], (
            "on_load_checkpoint did not fill the missing BKD key."
        )

        # Strict load must succeed (step counter is reset to 0 from current model).
        model_with_bkd.load_state_dict(fake_checkpoint["state_dict"], strict=True)


# ---------------------------------------------------------------------------
# BKD dropped-key exclusion from the HSIC pair set
# ---------------------------------------------------------------------------

class TestBKDHsicExclusion:
    """hsic_bkd_exclude_dropped: exclude BKD-dropped sources from HSIC."""

    # -- keep-mask export, BatchConsistentKeyDropout sub-module style --------
    def test_keep_mask_exported_submodule_bkd(self):
        model = _make_atsel(batch_key_dropout=1.0)
        S, X, X_blanked = _make_inputs()
        model.train()
        model.forward_with_actual(
            source_tensor=S, x_blanked=X_blanked, x_actual=X
        )
        bkd = model.attention.inner_attention.batch_key_dropout
        assert bkd._last_key_mask is not None
        assert not bool(bkd._last_key_mask.any()), "p=1.0 must drop every key"
        model.eval()
        model.forward_with_actual(
            source_tensor=S, x_blanked=X_blanked, x_actual=X
        )
        assert bkd._last_key_mask is None, "eval must not expose a keep mask"

    # -- keep-mask export, inline-BKD style (GatedSelfAttention) -------------
    def test_keep_mask_exported_inline_bkd(self):
        model = _make_atsel(batch_key_dropout=1.0)
        inner_self = model.self_attention.inner_attention
        if getattr(inner_self, "_bkd_p0", None) is None:
            pytest.skip("BKD not wired to the self-attention block here")
        S, X, X_blanked = _make_inputs()
        model.train()
        model.forward_with_actual(
            source_tensor=S, x_blanked=X_blanked, x_actual=X
        )
        keep = getattr(inner_self, "last_bkd_keep", None)
        assert keep is not None, "inline BKD must export last_bkd_keep"
        assert float(keep.sum()) == 0.0, "p=1.0 must drop every key"
        model.eval()
        model.forward_with_actual(
            source_tensor=S, x_blanked=X_blanked, x_actual=X
        )
        assert inner_self.last_bkd_keep is None, "eval must reset the mask"

    # -- forecaster: feature flag --------------------------------------------
    def test_exclude_disabled_by_default(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        fc = AttentionSelectorForecaster(
            _make_forecaster_config(batch_key_dropout=0.5)
        )
        assert fc.hsic_bkd_exclude_dropped is False
        assert fc._build_bkd_keep_mask(3, 6, torch.device("cpu")) is None

    # -- forecaster: split-mode column layout --------------------------------
    def test_split_mode_column_layout(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=0.5)
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        # Cross-attn (S side): drop S_0 and S_2.  Self-attn (X side): drop X_1.
        fc.model.attention.inner_attention.last_bkd_keep = torch.tensor(
            [0.0, 1.0, 0.0]
        )
        fc.model.self_attention.inner_attention.last_bkd_keep = torch.tensor(
            [1.0, 0.0, 1.0]
        )
        mask = fc._build_bkd_keep_mask(3, 6, torch.device("cpu"))
        assert mask is not None and mask.shape == (3, 6)
        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]).expand(3, 6)
        assert torch.equal(mask, expected), f"got {mask[0]}"

    def test_module_without_mask_defaults_to_kept(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=0.5)
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.model.attention.inner_attention.last_bkd_keep = torch.tensor(
            [0.0, 1.0, 1.0]
        )
        # self-attention keep mask is None (e.g. eval / no BKD there).
        fc.model.self_attention.inner_attention.last_bkd_keep = None
        mask = fc._build_bkd_keep_mask(3, 6, torch.device("cpu"))
        assert mask is not None
        assert torch.equal(
            mask[0], torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ), f"got {mask[0]}"

    # -- forecaster: end-to-end _step with every key dropped -----------------
    def test_all_dropped_batch_yields_zero_hsic(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=1.0)
        cfg["training"]["lambda_hsic"] = 0.1
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.log = lambda *a, **k: None  # no trainer attached in tests
        fc.train()
        S, X, _ = _make_inputs(S_len=3, X_len=3, batch=4)
        total_loss, _, _ = fc._step((S, X), "train")
        assert torch.isfinite(total_loss), f"non-finite loss: {total_loss}"
        assert fc._last_hsic_bkd_kept_frac == 0.0

    # -- forecaster: no BKD application -> no exclusion ----------------------
    def test_eval_step_not_excluded(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=1.0)
        cfg["training"]["lambda_hsic"] = 0.1
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.log = lambda *a, **k: None
        fc.eval()
        S, X, _ = _make_inputs(S_len=3, X_len=3, batch=4)
        total_loss, _, _ = fc._step((S, X), "val")
        assert torch.isfinite(total_loss)
        assert fc._last_hsic_bkd_kept_frac == 1.0



    # -- forecaster: homogeneous-mode AND-combine ----------------------------
    def test_homogeneous_mode_and_combines(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=0.5)
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.homogeneous_nodes = True
        # Square layout: cross keep AND self keep (a pair is excluded only
        # when the key was dropped in EVERY module that could route it).
        fc.model.attention.inner_attention.last_bkd_keep = torch.tensor(
            [0.0, 1.0, 0.0]
        )
        fc.model.self_attention.inner_attention.last_bkd_keep = torch.tensor(
            [0.0, 0.0, 1.0]
        )
        mask = fc._build_bkd_keep_mask(3, 3, torch.device("cpu"))
        assert mask is not None and mask.shape == (3, 3)
        assert torch.equal(
            mask[0], torch.tensor([0.0, 0.0, 0.0])
        ), f"got {mask[0]}"

    def test_homogeneous_mode_single_module(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=0.5)
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.homogeneous_nodes = True
        fc.model.attention.inner_attention.last_bkd_keep = torch.tensor(
            [1.0, 0.0, 1.0]
        )
        fc.model.self_attention.inner_attention.last_bkd_keep = None
        mask = fc._build_bkd_keep_mask(3, 3, torch.device("cpu"))
        assert mask is not None
        assert torch.equal(mask[0], torch.tensor([1.0, 0.0, 1.0]))

    # -- shape mismatch -> warn + no exclusion -------------------------------
    def test_size_mismatch_falls_back_to_none(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        cfg = _make_forecaster_config(batch_key_dropout=0.5)
        cfg["training"]["hsic_bkd_exclude_dropped"] = True
        fc = AttentionSelectorForecaster(cfg)
        fc.model.attention.inner_attention.last_bkd_keep = torch.tensor(
            [1.0, 0.0]
        )  # wrong length for n_sources=6
        fc.model.self_attention.inner_attention.last_bkd_keep = None
        assert fc._build_bkd_keep_mask(3, 6, torch.device("cpu")) is None

