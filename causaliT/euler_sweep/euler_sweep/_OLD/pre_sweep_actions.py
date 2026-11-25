"""
Pre-Sweep Actions: Hook factories for the calibrated sweep pipeline.

A ``pre_sweep_fn`` is any callable with the signature::

    pre_sweep_fn(config, data_dir, save_dir) -> dict

It is called by ``run_sequential_sweep`` **once** before the parameter grid is
expanded.  Its return value is merged into the base config so that every
sweep combination automatically inherits the calibrated values.

Factories defined here
----------------------
make_calibration_pre_sweep
    Runs ``calibrate_group_l1`` and returns calibrated lambda values.
    Use this when the score-sparsity sweep should run on top of the
    dataset-specific group sparsity found by calibration.

make_noop_pre_sweep
    Identity hook — returns an empty dict.
    Useful for testing or when calibration is already done.

Typical usage in cli.py::

    from causaliT.euler_sweep.euler_sweep.pre_sweep_actions import (
        make_calibration_pre_sweep,
    )
    pre_fn = make_calibration_pre_sweep(seed=42)
    run_sequential_sweep(..., pre_sweep_fn=pre_fn)
"""

import copy
import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def make_noop_pre_sweep() -> Callable:
    """Return a pre_sweep_fn that does nothing (returns empty dict)."""
    def _noop(config: dict, data_dir: str, save_dir: str) -> dict:
        return {}
    return _noop


def make_calibration_pre_sweep(seed: int = 42) -> Callable:
    """
    Factory for a pre_sweep_fn that calibrates group L1 sparsity.

    The returned callable runs ``calibrate_group_l1`` once on the base config
    and returns a dict whose keys mirror what
    ``config_operations.apply_calibration_to_config`` would write:

    - ``training.lambda_group_l1``
    - ``training.lambda_hsic_cross``
    - ``training.lambda_hsic_self``
    - ``staged_training.lambda_group_l1``
    - ``staged_training.lambda_hsic_cross_suggested``
    - ``staged_training.lambda_hsic_self_suggested``
    - ``staged_training.calibration_checkpoint``

    These are flat-merged into the sweep's base config BEFORE the Cartesian
    product of lambda_score values is generated, so every score-sparsity run
    automatically uses the calibrated group sparsity.

    Args:
        seed: Random seed for the calibration runs.

    Returns:
        A callable ``(config, data_dir, save_dir) -> dict``.
    """
    def _calibrate(config: dict, data_dir: str, save_dir: str) -> dict:
        from causaliT.training.calibration import calibrate_group_l1
        from causaliT.training.config_operations import apply_calibration_to_config

        print("\n" + "=" * 70)
        print("PRE-SWEEP ACTION: Calibrating group L1 sparsity")
        print("=" * 70)

        cal_result = calibrate_group_l1(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            seed=seed,
        )

        calibrated_config = apply_calibration_to_config(config, cal_result)

        # Extract the flat key→value overrides to merge into sweep base config
        overrides = {
            "training": {
                "lambda_group_l1": calibrated_config["training"]["lambda_group_l1"],
                "lambda_hsic_cross": calibrated_config["training"]["lambda_hsic_cross"],
                "lambda_hsic_self": calibrated_config["training"]["lambda_hsic_self"],
            },
            "staged_training": dict(calibrated_config.get("staged_training", {})),
        }

        # Persist so the sweep can be resumed without re-running calibration
        cal_summary_path = Path(save_dir) / "pre_sweep_calibration.json"
        with open(cal_summary_path, "w") as f:
            json.dump(
                {
                    "lambda_group_optimal": cal_result.lambda_group_optimal,
                    "lambda_hsic_cross_suggested": cal_result.lambda_hsic_cross_suggested,
                    "lambda_hsic_self_suggested": cal_result.lambda_hsic_self_suggested,
                    "base_ratio_cross": cal_result.base_ratio_cross,
                    "base_ratio_self": cal_result.base_ratio_self,
                    "converged": cal_result.converged,
                    "checkpoint_path": cal_result.checkpoint_path,
                },
                f,
                indent=2,
            )

        print(f"\nPre-sweep calibration complete:")
        print(f"  lambda_group*    = {cal_result.lambda_group_optimal:.2e}")
        print(f"  lambda_hsic_cross = {cal_result.lambda_hsic_cross_suggested:.4f}")
        print(f"  lambda_hsic_self  = {cal_result.lambda_hsic_self_suggested:.4f}")
        print(f"  Converged: {cal_result.converged}")
        print(f"  Summary: {cal_summary_path}\n")

        return overrides

    return _calibrate


def load_pre_sweep_calibration(save_dir: str) -> Optional[dict]:
    """
    Load previously saved pre-sweep calibration results (for sweep resumption).

    Args:
        save_dir: The sweep save directory.

    Returns:
        The calibration override dict (same structure as returned by the
        calibration hook), or ``None`` if no prior calibration was found.
    """
    cal_path = Path(save_dir) / "pre_sweep_calibration.json"
    if not cal_path.exists():
        return None

    with open(cal_path, "r") as f:
        data = json.load(f)

    return {
        "training": {
            "lambda_group_l1": data["lambda_group_optimal"],
            "lambda_hsic_cross": data["lambda_hsic_cross_suggested"],
            "lambda_hsic_self": data["lambda_hsic_self_suggested"],
        },
        "staged_training": {
            "lambda_group_l1": data["lambda_group_optimal"],
            "lambda_hsic_cross_suggested": data["lambda_hsic_cross_suggested"],
            "lambda_hsic_self_suggested": data["lambda_hsic_self_suggested"],
            "calibration_checkpoint": data.get("checkpoint_path", ""),
        },
    }
