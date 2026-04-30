"""
Smoke test for d_ff_mult / d_qk_mult resolution in
``causaliT.training.experiment_control.update_config``.

These tests pin the affine-multiplier convention introduced after the
1:1:1 ``d_model_set / d_ff / d_qk`` default:

    d_ff = round(d_ff_mult * d_model_set)
    d_qk = max(1, round(d_qk_mult * d_model_set))

and verify that explicit absolute overrides still win over multipliers.
"""
from __future__ import annotations

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from causaliT.training.experiment_control import update_config


def _make_cfg(d_model_set=24, d_ff_mult=2.0, d_qk_mult=0.5,
              d_ff=None, d_qk=None):
    return OmegaConf.create({
        "experiment": {
            "d_model_set": d_model_set,
            "d_ff_mult": d_ff_mult,
            "d_qk_mult": d_qk_mult,
            "d_ff": d_ff,
            "d_qk": d_qk,
        },
    })


def test_default_2x_and_half_multipliers():
    cfg = _make_cfg()
    update_config(cfg)
    assert cfg.experiment.d_ff == 48      # 2 * 24
    assert cfg.experiment.d_qk == 12      # 24 / 2


def test_d_ff_explicit_override_wins():
    cfg = _make_cfg(d_ff=37)
    update_config(cfg)
    assert cfg.experiment.d_ff == 37      # multiplier ignored
    assert cfg.experiment.d_qk == 12      # still resolved


def test_d_qk_explicit_override_wins():
    cfg = _make_cfg(d_qk=7)
    update_config(cfg)
    assert cfg.experiment.d_ff == 48
    assert cfg.experiment.d_qk == 7


def test_no_multiplier_means_no_change():
    """If both multipliers are absent, d_ff/d_qk remain None."""
    cfg = OmegaConf.create({
        "experiment": {"d_model_set": 24, "d_ff": None, "d_qk": None}
    })
    update_config(cfg)
    assert cfg.experiment.d_ff is None
    assert cfg.experiment.d_qk is None


def test_d_qk_floor_at_1():
    """Tiny d_model with d_qk_mult=0.5 should round to a positive int >= 1."""
    cfg = _make_cfg(d_model_set=1, d_qk_mult=0.1)
    update_config(cfg)
    # 0.1 * 1 = 0.1 -> round(0) -> floored to 1
    assert cfg.experiment.d_qk == 1


def test_non_integer_multiplier_rounded():
    """d_ff_mult=2.5 with d_model_set=10 -> d_ff = round(25.0) = 25."""
    cfg = _make_cfg(d_model_set=10, d_ff_mult=2.5, d_qk_mult=0.5)
    update_config(cfg)
    assert cfg.experiment.d_ff == 25
    assert cfg.experiment.d_qk == 5
