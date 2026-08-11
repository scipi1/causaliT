"""
The ATE arms must declare a trainer their base config can actually run.

The first campaign lost all 30 baseline runs to a contradiction that only
surfaced inside the train array: `vanilla` and `cheater` asked for the adaptive
trainer while pinning `training.use_gradient_routing: false`, which
`adaptive_trainer` rejects at start-up.  These tests catch it offline, and the
plan-time guard catches it at submission instead of 15 jobs later.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from causaliT.euler_sweep.euler_sweep.dagsweep_parallel import _check_trainer_supported

ATE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "7_PUBLISH" / "ATE"
ARMS = ["svfa", "vanilla", "cheater"]


def _arm_paths(arm):
    arm_dir = ATE_DIR / arm
    return arm_dir / "atesweep.yaml", arm_dir / "config_atsel.yaml"


@pytest.mark.parametrize("arm", ARMS)
def test_arm_trainer_matches_gradient_routing(arm):
    """`adaptive` requires gradient routing; `standard` works either way."""
    sweep_path, config_path = _arm_paths(arm)
    if not sweep_path.exists():
        pytest.skip(f"arm {arm} not present")

    spec = OmegaConf.load(sweep_path)
    config = OmegaConf.load(config_path)
    routing = bool(config.training.use_gradient_routing)

    for phase, trainer in (
        ("training.trainer", spec.training.trainer),
        ("optuna.trainer", spec.optuna.get("trainer", "standard")),
    ):
        if str(trainer) == "adaptive":
            assert routing, (
                f"{arm}: {phase}=adaptive but use_gradient_routing=false; "
                "adaptive_trainer cannot freeze the parameter groups."
            )


@pytest.mark.parametrize("arm", ["vanilla", "cheater"])
def test_baseline_arms_train_with_standard_trainer(arm):
    """The baselines have no structural loss, so there is no phase to alternate."""
    sweep_path, _ = _arm_paths(arm)
    if not sweep_path.exists():
        pytest.skip(f"arm {arm} not present")
    assert str(OmegaConf.load(sweep_path).training.trainer) == "standard"


def test_plan_guard_rejects_adaptive_without_routing(tmp_path):
    config_path = tmp_path / "config_atsel.yaml"
    OmegaConf.save(OmegaConf.create({"training": {"use_gradient_routing": False}}),
                   config_path)

    with pytest.raises(ValueError, match="use_gradient_routing"):
        _check_trainer_supported(str(config_path), "training.trainer", "adaptive")

    # standard never needs the parameter groups
    _check_trainer_supported(str(config_path), "training.trainer", "standard")


def test_plan_guard_accepts_adaptive_with_routing(tmp_path):
    config_path = tmp_path / "config_atsel.yaml"
    OmegaConf.save(OmegaConf.create({"training": {"use_gradient_routing": True}}),
                   config_path)
    _check_trainer_supported(str(config_path), "training.trainer", "adaptive")
