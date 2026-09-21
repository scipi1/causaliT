"""
Tests for the HSIC-constraint per-rung tolerance calibration and dual-ascent
control (AttentionSelectorForecaster).

Covers:
1. ``dual_ascent_structure_only``: dual ascent + EMA are gated to structure
   phases via ``hsic_constraint_on_phase_switch(phase=...)``.
2. ``dual_pause_below_keys``: ascent is paused at BKD rungs whose key budget
   is below the configured floor (constraint unreachable by construction).
3. ``dual_lr_down``: satisfied constraints release lambda at the asymmetric
   downward rate.
4. ``calibrate_tolerance``: structure-phase entry arms the permutation-null
   calibration; ``_finalize_hsic_null_calibration`` sets
   ``tolerance = quantile(null) * margin``.
5. Checkpoint round-trip of the calibrated tolerance.
"""

import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from tests.test_atsel_bkd import _make_forecaster_config


def _make_fc(hsic_constraint: dict) -> AttentionSelectorForecaster:
    cfg = _make_forecaster_config()
    cfg["training"]["hsic_constraint"] = hsic_constraint
    return AttentionSelectorForecaster(cfg)


_BASE_HC = {
    "enabled": True,
    "source": "hsic",
    "tolerance": 0.01,
    "dual_init": 0.0,
    "dual_lr": 1.0,
    "dual_max": 1000.0,
    "rho_init": 0.0,
    "ema": 0.9,
}


class TestDualAscentGating:
    def test_structure_only_gating(self):
        fc = _make_fc({**_BASE_HC, "dual_ascent_structure_only": True})
        fc._hsic_constraint_ema = 0.5  # >> tolerance -> would ascend

        fc.hsic_constraint_on_phase_switch(phase="reconstruct", bkd_min_keys=2)
        assert fc._hsic_dual_active is False
        fc._hsic_constraint_ema = 0.5  # reset by the switch; re-arm manually
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda == 0.0, "ascent during reconstruct phase"

        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=2)
        assert fc._hsic_dual_active is True
        fc._hsic_constraint_ema = 0.5
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda == pytest.approx(1.0 * (0.5 - 0.01))

    def test_gating_disabled_by_config(self):
        fc = _make_fc({**_BASE_HC, "dual_ascent_structure_only": False})
        fc.hsic_constraint_on_phase_switch(phase="reconstruct", bkd_min_keys=2)
        assert fc._hsic_dual_active is True
        fc._hsic_constraint_ema = 0.5
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda > 0.0

    def test_low_rung_pause(self):
        fc = _make_fc({**_BASE_HC, "dual_pause_below_keys": 4})
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=2)
        fc._hsic_constraint_ema = 0.5
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda == 0.0, "ascent at unreachable rung"

        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=6)
        fc._hsic_constraint_ema = 0.5
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda > 0.0, "pause persisted at high rung"

    def test_asymmetric_downward_lr(self):
        fc = _make_fc({**_BASE_HC, "dual_init": 10.0, "dual_lr_down": 5.0})
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=10)
        fc._hsic_constraint_ema = 0.001  # below tolerance 0.01
        fc._update_hsic_dual()
        # violation = -0.009, downward rate 5.0 -> lambda 10 - 0.045
        assert fc._hsic_dual_lambda == pytest.approx(10.0 - 5.0 * 0.009)

    def test_lambda_clips_at_zero(self):
        fc = _make_fc({**_BASE_HC, "dual_init": 1e-6, "dual_lr_down": 5.0})
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=10)
        fc._hsic_constraint_ema = 0.0
        fc._update_hsic_dual()
        assert fc._hsic_dual_lambda == 0.0



class TestToleranceCalibration:
    def test_armed_only_at_structure_entry(self):
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True,
                       "calibration_batches": 3})
        fc.hsic_constraint_on_phase_switch(phase="reconstruct", bkd_min_keys=2)
        assert fc._hsic_null_calib_remaining == 0
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=2)
        assert fc._hsic_null_calib_remaining == 3
        assert fc._hsic_null_calib_round == 1

    def test_not_armed_when_disabled(self):
        fc = _make_fc(_BASE_HC)
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=2)
        assert fc._hsic_null_calib_remaining == 0

    def test_finalize_sets_quantile_times_margin(self):
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True,
                       "calibration_margin": 2.0, "calibration_quantile": 0.9})
        fc._hsic_null_samples = [float(i) for i in range(1, 101)]  # 1..100
        fc._finalize_hsic_null_calibration()
        import numpy as np
        expected = float(np.quantile(np.arange(1.0, 101.0), 0.9)) * 2.0
        assert fc.hsic_tol == pytest.approx(expected)
        assert fc._hsic_null_samples == []

    def test_finalize_keeps_tolerance_on_too_few_samples(self):
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True})
        fc._hsic_null_samples = [0.1, float("nan")]
        fc._finalize_hsic_null_calibration()
        assert fc.hsic_tol == 0.01  # unchanged

    def test_collect_sample_plain_aggregation(self):
        """End-to-end null sample collection through the plain masked path."""
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True,
                       "calibration_batches": 1, "calibration_permutations": 3})
        fc.hsic_constraint_on_phase_switch(phase="structure", bkd_min_keys=2)
        torch.manual_seed(0)
        B, T, Ssrc = 64, 3, 6
        combined = torch.randn(B, Ssrc)
        residuals = torch.randn(B, T)
        fc._collect_hsic_null_sample(
            combined_source=combined,
            residuals=residuals,
            attention_weights=None,
            bkd_keep_mask=None,
            hsic_pair_mask=None,
        )
        # calibration_batches=1 -> finalized immediately, tolerance replaced
        assert fc._hsic_null_calib_remaining == 0
        assert fc.hsic_tol > 0.0


class TestCheckpointRoundTrip:
    def test_tolerance_persisted(self):
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True})
        fc.hsic_tol = 0.123
        ckpt = {}
        fc.on_save_checkpoint(ckpt)
        assert ckpt["hsic_constraint"]["tolerance"] == pytest.approx(0.123)

        fc2 = _make_fc({**_BASE_HC, "calibrate_tolerance": True})
        fc2.on_load_checkpoint(dict(ckpt, state_dict={}))
        assert fc2.hsic_tol == pytest.approx(0.123)

    def test_missing_tolerance_is_backward_compatible(self):
        fc = _make_fc(_BASE_HC)
        fc.on_load_checkpoint({"hsic_constraint": {"dual_lambda": 3.0},
                               "state_dict": {}})
        assert fc._hsic_dual_lambda == 3.0
        assert fc.hsic_tol == 0.01  # config default preserved


class TestFitStartArming:
    """Static-trainer path: with no phase controller, the permutation-null
    calibration must arm once at fit start (otherwise the fallback tolerance
    would govern the whole run)."""

    def test_fit_start_arms_calibration(self):
        fc = _make_fc({**_BASE_HC, "calibrate_tolerance": True,
                       "calibration_batches": 5})
        assert fc._hsic_null_calib_remaining == 0
        fc.on_fit_start()
        assert fc._hsic_null_calib_remaining == 5
        assert fc._hsic_null_calib_round == 1
        # dual stays active (no phase gating without the adaptive controller)
        assert fc._hsic_dual_active is True

    def test_fit_start_no_arming_when_disabled(self):
        fc = _make_fc(_BASE_HC)
        fc.on_fit_start()
        assert fc._hsic_null_calib_remaining == 0
        assert fc._hsic_null_calib_round == 0
