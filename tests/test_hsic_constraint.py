"""Tests for the HSIC-as-constraint (Lagrangian / augmented Lagrangian) mode.

Run with:  pytest tests/test_hsic_constraint.py -v

Background
----------
``training.hsic_constraint.enabled=True`` replaces the fixed-weight HSIC
penalty ``lambda_hsic * HSIC`` by the constraint term of

    min_thetaS  L0 + NOTEARS        s.t.  HSIC(thetaS) <= tolerance

    L = L0 + NOTEARS + lam * (HSIC - eps) + (rho/2) * relu(HSIC - eps)^2

with per-epoch dual ascent on ``lam`` (EMA of the raw train HSIC) and
optional NOTEARS-style rho escalation.  The mode is mutually exclusive with
the HSIC-supervision safeguards (gradient_surgery, *_max_hsic_pct,
lambda_hsic > 0), because it REVERSES the HSIC vs L0/NOTEARS relationship.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from tests.test_atsel_reg_safeguard import _make_forecaster_config, _make_batch


def _constraint_cfg(**overrides):
    """Config with the constraint enabled and no conflicting options."""
    cfg = _make_forecaster_config(lambda_hsic=0.0)
    hc = {
        "enabled": True,
        "tolerance": 0.0,
        "dual_init": 0.0,
        "dual_lr": 1.0,
        "dual_max": 100.0,
        "rho_init": 0.0,
        "rho_mult": 2.0,
        "rho_max": 1e6,
        "ema": 0.9,
    }
    hc.update(overrides)
    cfg["training"]["hsic_constraint"] = hc
    return cfg


class TestConfigValidation:
    def test_default_disabled(self):
        model = AttentionSelectorForecaster(_make_forecaster_config())
        assert model.hsic_constraint_enabled is False

    def test_conflicting_gradient_surgery_raises(self):
        cfg = _constraint_cfg()
        cfg["training"]["gradient_surgery"] = True
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    def test_conflicting_kappa_cap_raises(self):
        cfg = _constraint_cfg()
        cfg["training"]["kappa_max_hsic_pct"] = 0.5
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    def test_conflicting_l0_cap_raises(self):
        cfg = _constraint_cfg()
        cfg["training"]["lambda_l0_max_hsic_pct"] = 0.5
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    def test_conflicting_lambda_hsic_raises(self):
        cfg = _constraint_cfg()
        cfg["training"]["lambda_hsic"] = 1.0
        with pytest.raises(ValueError, match="mutually exclusive"):
            AttentionSelectorForecaster(cfg)

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance"):
            AttentionSelectorForecaster(_constraint_cfg(tolerance=-0.1))

    def test_zero_dual_lr_raises(self):
        with pytest.raises(ValueError, match="dual_lr"):
            AttentionSelectorForecaster(_constraint_cfg(dual_lr=0.0))

    def test_bad_ema_raises(self):
        with pytest.raises(ValueError, match="ema"):
            AttentionSelectorForecaster(_constraint_cfg(ema=1.0))

    def test_negative_dual_init_raises(self):
        with pytest.raises(ValueError, match="dual_init"):
            AttentionSelectorForecaster(_constraint_cfg(dual_init=-1.0))

    def test_negative_rho_init_raises(self):
        with pytest.raises(ValueError, match="rho_init"):
            AttentionSelectorForecaster(_constraint_cfg(rho_init=-1.0))

    def test_dual_state_initialised(self):
        model = AttentionSelectorForecaster(
            _constraint_cfg(dual_init=3.0, rho_init=1.0)
        )
        assert model._hsic_dual_lambda == 3.0
        assert model._hsic_rho == 1.0
        assert model._hsic_constraint_ema is None
        assert model._hsic_constraint_prev_violation is None


class TestConstraintTerm:
    def test_term_matches_fixed_lambda_when_pure_lagrangian(self):
        """lam*(h - 0) with lam=2 must equal 2 * (1.0 * h) of the legacy path."""
        model = AttentionSelectorForecaster(_constraint_cfg(dual_init=2.0))
        batch = _make_batch()
        model.eval()
        with torch.no_grad():
            model._step(batch, stage="val")
        reg_constraint = float(model._last_hsic_reg)

        # Same model, legacy fixed-weight path with lambda_hsic = 1.
        model.hsic_constraint_enabled = False
        model.lambda_hsic = 1.0
        with torch.no_grad():
            model._step(batch, stage="val")
        reg_fixed = float(model._last_hsic_reg)

        assert reg_constraint == pytest.approx(2.0 * reg_fixed, rel=1e-5)

    def test_tolerance_and_rho_algebra(self):
        """hsic_reg == lam*(h-eps) + (rho/2)*relu(h-eps)^2 exactly."""
        eps, lam, rho = 0.001, 1.5, 4.0
        model = AttentionSelectorForecaster(
            _constraint_cfg(tolerance=eps, dual_init=lam, rho_init=rho)
        )
        batch = _make_batch()
        model.eval()
        with torch.no_grad():
            model._step(batch, stage="val")
        reg = float(model._last_hsic_reg)
        # Recover h from the legacy path (lambda_hsic = 1), same model/batch.
        model.hsic_constraint_enabled = False
        model.lambda_hsic = 1.0
        with torch.no_grad():
            model._step(batch, stage="val")
        h = float(model._last_hsic_reg)
        expected = lam * (h - eps) + 0.5 * rho * max(h - eps, 0.0) ** 2
        assert reg == pytest.approx(expected, rel=1e-4)

    def test_ema_updates_on_train_only(self):
        model = AttentionSelectorForecaster(_constraint_cfg())
        batch = _make_batch()
        model.train()
        with torch.no_grad():
            model._step(batch, stage="train")
        assert model._hsic_constraint_ema is not None
        first = model._hsic_constraint_ema

        model.eval()
        with torch.no_grad():
            model._step(_make_batch(seed=1), stage="val")
        assert model._hsic_constraint_ema == first  # untouched by val

        model.train()
        with torch.no_grad():
            model._step(_make_batch(seed=2), stage="train")
        assert model._hsic_constraint_ema != first  # EMA moved



class TestDualAscent:
    def _model(self, **hc):
        return AttentionSelectorForecaster(_constraint_cfg(**hc))

    def test_lambda_grows_while_violated(self):
        m = self._model(dual_lr=2.0)
        m._hsic_constraint_ema = 0.5
        m._update_hsic_dual()
        assert m._hsic_dual_lambda == pytest.approx(2.0 * 0.5)
        m._update_hsic_dual()
        assert m._hsic_dual_lambda == pytest.approx(2.0 * 1.0)

    def test_lambda_stops_at_tolerance(self):
        m = self._model(dual_init=5.0, tolerance=0.1, dual_lr=1.0)
        m._hsic_constraint_ema = 0.05  # below tolerance -> violation < 0
        m._update_hsic_dual()
        assert m._hsic_dual_lambda == pytest.approx(5.0 - 0.05)

    def test_lambda_floored_at_zero(self):
        m = self._model(dual_init=0.1, dual_lr=10.0)
        m._hsic_constraint_ema = 0.0
        m.hsic_tol = 1.0
        m._update_hsic_dual()
        assert m._hsic_dual_lambda == 0.0

    def test_lambda_capped_at_dual_max(self):
        m = self._model(dual_init=99.0, dual_lr=10.0, dual_max=100.0)
        m._hsic_constraint_ema = 1.0
        m._update_hsic_dual()
        assert m._hsic_dual_lambda == 100.0

    def test_no_update_without_ema(self):
        m = self._model(dual_init=1.0)
        m._update_hsic_dual()  # EMA is None -> no-op
        assert m._hsic_dual_lambda == 1.0

    def test_rho_escalates_on_stalled_violation(self):
        m = self._model(rho_init=1.0, rho_mult=2.0)
        m._hsic_constraint_ema = 1.0
        m._update_hsic_dual()                      # prev = None -> no escalate
        assert m._hsic_rho == 1.0
        m._hsic_constraint_ema = 0.9               # > 0.25 * 1.0 -> escalate
        m._update_hsic_dual()
        assert m._hsic_rho == 2.0

    def test_rho_not_escalated_when_improving(self):
        m = self._model(rho_init=1.0, rho_mult=2.0)
        m._hsic_constraint_ema = 1.0
        m._update_hsic_dual()
        m._hsic_constraint_ema = 0.2               # <= 0.25 * 1.0 -> improving
        m._update_hsic_dual()
        assert m._hsic_rho == 1.0

    def test_rho_never_escalates_from_zero(self):
        m = self._model(rho_init=0.0, rho_mult=2.0)
        m._hsic_constraint_ema = 1.0
        m._update_hsic_dual()
        m._update_hsic_dual()
        assert m._hsic_rho == 0.0


class TestCheckpointAndGradients:
    def test_checkpoint_round_trip(self):
        model = AttentionSelectorForecaster(
            _constraint_cfg(dual_init=1.0, rho_init=2.0)
        )
        model._hsic_dual_lambda = 7.5
        model._hsic_rho = 8.0
        model._hsic_constraint_ema = 0.3
        model._hsic_constraint_prev_violation = 0.25

        checkpoint = {"state_dict": dict(model.state_dict())}
        model.on_save_checkpoint(checkpoint)

        fresh = AttentionSelectorForecaster(
            _constraint_cfg(dual_init=1.0, rho_init=2.0)
        )
        fresh.on_load_checkpoint(checkpoint)
        assert fresh._hsic_dual_lambda == 7.5
        assert fresh._hsic_rho == 8.0
        assert fresh._hsic_constraint_ema == 0.3
        assert fresh._hsic_constraint_prev_violation == 0.25

    def test_load_predating_checkpoint_keeps_init(self):
        model = AttentionSelectorForecaster(
            _constraint_cfg(dual_init=1.0, rho_init=2.0)
        )
        checkpoint = {"state_dict": dict(model.state_dict())}
        model.on_load_checkpoint(checkpoint)  # no "hsic_constraint" key
        assert model._hsic_dual_lambda == 1.0
        assert model._hsic_rho == 2.0

    def test_gradients_flow_through_constraint_term(self):
        model = AttentionSelectorForecaster(_constraint_cfg(dual_init=1.0))
        model.train()
        total_loss, _, _ = model._step(_make_batch(), stage="train")
        total_loss.backward()
        grads = [
            p.grad for p in model.model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert grads, "no parameter received gradient through the constraint term"


# ---------------------------------------------------------------------------
# Oracle-SHD constraint source (hsic_constraint.source = "oracle_shd")
# ---------------------------------------------------------------------------

import tempfile

import numpy as np
import pandas as pd


def _oracle_cfg(tmp_root, **hc_overrides):
    """Constraint config with source=oracle_shd + a tiny GT dataset on disk."""
    rng = np.random.default_rng(0)
    cross = (rng.random((3, 3)) > 0.5).astype(float)   # dec_cross (L_X, L_S)
    selfm = (rng.random((3, 3)) > 0.5).astype(float)   # dec_self  (L_X, L_X)
    np.fill_diagonal(selfm, 0.0)
    ds = Path(tmp_root) / "dummy"
    ds.mkdir(parents=True)
    pd.DataFrame(cross).to_csv(ds / "dec1_cross_att_mask.csv")
    pd.DataFrame(selfm).to_csv(ds / "dec1_self_att_mask.csv")

    cfg = _make_forecaster_config(lambda_hsic=0.0)
    cfg["training"]["hard_mask_files"] = {
        "dec_cross": "dec1_cross_att_mask.csv",
        "dec_self": "dec1_self_att_mask.csv",
    }
    hc = {
        "enabled": True,
        "source": "oracle_shd",
        "tolerance": 0.0,
        "dual_init": 0.0,
        "dual_lr": 1.0,
        "dual_max": 100.0,
        "rho_init": 0.0,
        "ema": 0.9,
    }
    hc.update(hc_overrides)
    cfg["training"]["hsic_constraint"] = hc
    return cfg, tmp_root, np.concatenate([cross, selfm], axis=1)


class TestOracleShd:
    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            AttentionSelectorForecaster(_constraint_cfg(source="nope"))

    def test_hard_masks_leakage_guard(self):
        cfg, data_dir, _ = _oracle_cfg(tempfile.mkdtemp())
        cfg["training"]["use_hard_masks"] = True
        with pytest.raises(ValueError, match="use_hard_masks"):
            AttentionSelectorForecaster(cfg, data_dir=data_dir)

    def test_oracle_attention_leakage_guard(self):
        cfg, data_dir, _ = _oracle_cfg(tempfile.mkdtemp())
        cfg["training"]["use_oracle_attention"] = True
        with pytest.raises(ValueError, match="use_oracle_attention"):
            AttentionSelectorForecaster(cfg, data_dir=data_dir)

    def test_gt_loaded_and_never_in_forward_path(self):
        cfg, data_dir, gt_combined = _oracle_cfg(tempfile.mkdtemp())
        model = AttentionSelectorForecaster(cfg, data_dir=data_dir)
        # GT present as the dedicated buffer, with the split-mode layout.
        assert hasattr(model, "oracle_shd_gt")
        assert tuple(model.oracle_shd_gt.shape) == (3, 6)
        assert torch.allclose(model.oracle_shd_gt,
                              torch.tensor(gt_combined, dtype=torch.float32))
        # Leakage channel closed: no oracle_combined_mask registered.
        assert not hasattr(model, "oracle_combined_mask")

    def test_ema_tracks_expected_shd_not_hsic(self):
        cfg, data_dir, gt_combined = _oracle_cfg(tempfile.mkdtemp())
        model = AttentionSelectorForecaster(cfg, data_dir=data_dir)
        model.eval()  # deterministic posterior; stage string drives the EMA
        batch = _make_batch()
        with torch.no_grad():
            model._step(batch, stage="train")
        assert model._hsic_constraint_ema is not None
        # Recompute the expected SHD by hand from the same posterior.
        with torch.no_grad():
            _, att, _ = model.forward(*batch)
            post = att.mean(dim=0)
            gt = torch.tensor(gt_combined, dtype=post.dtype)
            expected = (gt * (1.0 - post) + (1.0 - gt) * post).mean()
        assert model._hsic_constraint_ema == pytest.approx(
            float(expected), rel=1e-6
        )

    def test_gradients_flow_through_oracle_shd(self):
        cfg, data_dir, _ = _oracle_cfg(tempfile.mkdtemp(), dual_init=1.0)
        model = AttentionSelectorForecaster(cfg, data_dir=data_dir)
        model.train()
        total_loss, _, _ = model._step(_make_batch(), stage="train")
        total_loss.backward()
        grads = [
            p.grad for p in model.model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert grads, "no gradient through the oracle-SHD constraint term"
