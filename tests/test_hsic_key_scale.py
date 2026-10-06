"""HSIC-derived key scaling (RESIT-style source prior).

Covers the two layers of the feature:

* ``GatedSelfAttention.set_hsic_key_scale`` / ``_structural_raw``: the
  detached per-key multiplier scales column j of the raw score by ``w_j``.
* ``AttentionSelectorForecaster``: config parsing + guards, and the
  ``_update_hsic_key_scale`` normalisation / floor / EMA / warmup logic.
"""
import pytest
import torch

from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from test_dropout_selection import _make_forecaster_config


def _raw(mod, q, k):
    return mod._structural_raw(q, k)


class TestModuleKeyScale:
    def test_default_disabled(self):
        mod = GatedSelfAttention()
        assert mod._hsic_key_scale is None

    def test_columns_scaled(self):
        torch.manual_seed(0)
        mod = GatedSelfAttention()
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        base = _raw(mod, q, k)
        w = torch.tensor([1.0, 0.5, 0.25, 1.0])
        mod.set_hsic_key_scale(w)
        scaled = _raw(mod, q, k)
        # Column j of raw must scale by exactly w_j.
        torch.testing.assert_close(
            scaled, base * w.view(1, 1, -1), rtol=1e-5, atol=1e-6
        )

    def test_setter_detaches(self):
        mod = GatedSelfAttention()
        w = torch.ones(3, requires_grad=True)
        mod.set_hsic_key_scale(w)
        assert not mod._hsic_key_scale.requires_grad

    def test_none_disables(self):
        mod = GatedSelfAttention()
        mod.set_hsic_key_scale(torch.ones(3))
        mod.set_hsic_key_scale(None)
        assert mod._hsic_key_scale is None

    def test_wrong_length_raises(self):
        mod = GatedSelfAttention()
        mod.set_hsic_key_scale(torch.ones(5))
        with pytest.raises(ValueError, match="hsic_key_scale shape"):
            _raw(mod, torch.randn(1, 4, 8), torch.randn(1, 4, 8))



def _resolve(**training):
    """Real forecaster with the fixed-orthonormal key stack required by the
    feature (the dropout fixture defaults to learnable key projections)."""
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["data"]["val_idx"] = 0
    for embed in ("ds_embed_S", "ds_embed_X"):
        for mod in cfg["model"]["kwargs"][embed]["modules"]:
            mod["idx"] = 1 if mod["label"] == "variable" else 0
            mod["role"] = "structure" if mod["label"] == "variable" else "value"
    cfg["model"]["kwargs"]["comps_embed_S"] = "svfa"
    cfg["model"]["kwargs"]["comps_embed_X"] = "svfa"
    cfg["model"]["kwargs"]["struct_embedding_type"] = "orthogonal_fixed"
    cfg["model"]["kwargs"]["remove_key_projection"] = True
    cfg["training"].update(training)
    return AttentionSelectorForecaster(cfg)


class TestConfigParsing:
    def test_default_disabled(self):
        o = _resolve()
        assert not o.hsic_key_scale_enabled
        assert o._hsic_key_scale_state is None

    def test_enabled_with_defaults(self):
        o = _resolve(hsic_key_scale={"enabled": True})
        assert o.hsic_key_scale_enabled
        assert o.hsic_key_scale_ema == 0.9
        assert o.hsic_key_scale_floor == 0.1
        assert o.hsic_key_scale_warmup_epochs == 0
        assert o.hsic_key_scale_exponent == 1.0

    def test_requires_fixed_orthonormal_keys(self):
        torch.manual_seed(0)
        cfg = _make_forecaster_config()
        cfg["data"]["val_idx"] = 0
        for embed in ("ds_embed_S", "ds_embed_X"):
            for mod in cfg["model"]["kwargs"][embed]["modules"]:
                mod["idx"] = 1 if mod["label"] == "variable" else 0
                mod["role"] = "structure" if mod["label"] == "variable" else "value"
        cfg["model"]["kwargs"]["comps_embed_S"] = "svfa"
        cfg["model"]["kwargs"]["comps_embed_X"] = "svfa"
        # remove_key_projection=False (fixture default): learnable key path.
        cfg["training"]["hsic_key_scale"] = {"enabled": True}
        with pytest.raises(ValueError, match="FIXED orthonormal key frame"):
            AttentionSelectorForecaster(cfg)

    def test_requires_pairwise_objective(self):
        with pytest.raises(ValueError, match="hsic_objective='pairwise'"):
            _resolve(hsic_key_scale={"enabled": True},
                     hsic_objective="dhsic_residual")

    @pytest.mark.parametrize("bad", [{"ema": 1.0}, {"floor": -0.1},
                                     {"exponent": 0.0}])
    def test_invalid_hyperparams_raise(self, bad):
        with pytest.raises(ValueError, match="hsic_key_scale"):
            _resolve(hsic_key_scale=dict(bad, enabled=True))


class TestWeightUpdate:
    def _armed(self, **hks):
        o = _resolve(hsic_key_scale=dict({"enabled": True, "ema": 0.0}, **hks))
        o.log = lambda *a, **k: None  # no logger attached in unit tests
        return o

    def _mat(self):
        # Row 0 high (source-like), rows 1-2 low (sink-like).
        return torch.tensor([
            [0.8, 0.9, 0.7],
            [0.05, 0.02, 0.03],
            [0.10, 0.08, 0.06],
        ])

    def test_max_normalisation_and_floor(self):
        o = self._armed(floor=0.1)
        o._update_hsic_key_scale(self._mat(), None, "train")
        w = o._hsic_key_scale_state
        assert w is not None
        assert float(w.max()) == pytest.approx(1.0)     # top node pinned to 1
        assert float(w[0]) == pytest.approx(1.0)        # highest row
        assert float(w.min()) >= 0.1                    # floor respected
        # Ordering preserved: row 0 >> rows 1, 2.
        assert float(w[0]) > float(w[1]) and float(w[0]) > float(w[2])

    def test_pushed_into_modules(self):
        o = self._armed()
        o._update_hsic_key_scale(self._mat(), None, "train")
        mods = [m for m in o.modules()
                if isinstance(m, GatedSelfAttention)]
        assert mods, "fixture must contain a GatedSelfAttention"
        for m in mods:
            assert m._hsic_key_scale is not None
            torch.testing.assert_close(
                m._hsic_key_scale, o._hsic_key_scale_state)

    def test_ema_smoothing(self):
        o = self._armed()
        o.hsic_key_scale_ema = 0.5
        o._update_hsic_key_scale(self._mat(), None, "train")
        first = o._hsic_key_scale_state.clone()
        # First update: 0.5 * ones + 0.5 * w_new (state starts at 1).
        assert float(first[0]) == pytest.approx(1.0)
        assert 0.5 < float(first[1]) < 1.0
        o._update_hsic_key_scale(self._mat(), None, "train")
        second = o._hsic_key_scale_state
        # Converging toward w_new (< first for sink rows).
        assert float(second[1]) < float(first[1])

    def test_nan_row_keeps_previous_weight(self):
        o = self._armed()
        mat = self._mat()
        o._update_hsic_key_scale(mat, None, "train")
        before = o._hsic_key_scale_state.clone()
        mat2 = mat.clone()
        mat2[1] = float("nan")   # row fully excluded this batch
        mat2[2] = torch.tensor([0.2, 0.16, 0.12])  # row 2 signal changes
        o._update_hsic_key_scale(mat2, None, "train")
        after = o._hsic_key_scale_state
        assert float(after[1]) == float(before[1])
        assert float(after[2]) != float(before[2])  # other rows still update

    def test_val_stage_skipped(self):
        o = self._armed()
        o._update_hsic_key_scale(self._mat(), None, "val")
        assert o._hsic_key_scale_state is None

    def test_warmup_gating(self):
        o = self._armed(warmup_epochs=5)
        o._update_hsic_key_scale(self._mat(), None, "train")
        assert o._hsic_key_scale_state is None  # current_epoch 0 < 5

    def test_all_nan_matrix_skipped(self):
        o = self._armed()
        o._update_hsic_key_scale(
            torch.full((3, 3), float("nan")), None, "train")
        assert o._hsic_key_scale_state is None

    def test_disabled_is_noop(self):
        o = _resolve()
        o._update_hsic_key_scale(self._mat(), None, "train")
        assert o._hsic_key_scale_state is None

