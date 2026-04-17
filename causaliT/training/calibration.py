"""
Calibration Stage: Two-step calibration for gradient balance.

This module implements Stage 0 of the staged training pipeline with TWO steps:

STEP 1: SPARSITY CALIBRATION (λ_group search)
    Goal: Make the HSIC landscape non-flat via embedding sparsity
    Metric: BASE gradient ratios (train_grad_ratio_*) - independent of λ_hsic
    Process: Binary search for λ_group that brings base ratio toward 1.0
    Output: λ_group_optimal, final_base_ratio_cross, final_base_ratio_self

STEP 2: LAMBDA SELECTION & VERIFICATION
    Goal: Balance actual learning signals between reconstruction and HSIC
    Process: Set λ_hsic = final_base_ratio (makes update_ratio ≈ 1.0)
    Verification: Re-train with suggested λ values and check update_ratio
    Output: suggested λ_hsic values, final_update_ratio

Key insight: When the model has too much capacity (large d_model or low sparsity),
the HSIC loss landscape becomes flat - any parameter configuration achieves low HSIC.
Step 1 constrains capacity via group L1 sparsity to make the HSIC landscape meaningful.
Step 2 then uses λ_hsic to balance the now-meaningful gradients.

IMPORTANT: This module tracks BOTH metrics:
- train_hsic_cross_grad_norm: ||∇ hsic_value|| (BASE gradient, for Step 1)
- train_hsic_cross_update_norm: λ * ||∇ hsic_value|| (UPDATE signal, for Step 2)

The calibration uses min(ratio_cross, ratio_self) as the convergence criterion
to ensure NEITHER HSIC signal is drowned out by reconstruction.

Usage:
    result = calibrate_group_l1(config, data_dir, save_dir)
    # result contains: lambda_group_optimal, lambda_hsic_cross_suggested,
    #                  lambda_hsic_self_suggested, final_update_ratio, checkpoint
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, NamedTuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CALIBRATION RESULT
# =============================================================================

class CalibrationResult(NamedTuple):
    """Result from two-phase calibration stage.

    Attributes:
        lambda_group_optimal: Calibrated group L1 coefficient (from Phase 1)
        lambda_hsic_cross_suggested: Suggested λ_hsic_cross value (= BASE ratio cross)
        lambda_hsic_self_suggested: Suggested λ_hsic_self value (= BASE ratio self)
        checkpoint_path: Path to verification checkpoint (Phase 2)
        base_ratio_cross: Final BASE gradient ratio for cross-attention (Phase 1)
        base_ratio_self: Final BASE gradient ratio for self-attention (Phase 1)
        update_ratio_cross: Final UPDATE ratio for cross-attention (Phase 2)
        update_ratio_self: Final UPDATE ratio for self-attention (Phase 2)
        phase1_converged: Whether Phase 1 (sparsity search) converged
        phase2_converged: Whether Phase 2 (verification) passed
        converged: Overall calibration success (phase1 AND phase2)
    """
    lambda_group_optimal: float
    lambda_hsic_cross_suggested: float
    lambda_hsic_self_suggested: float
    checkpoint_path: str
    base_ratio_cross: float
    base_ratio_self: float
    update_ratio_cross: float
    update_ratio_self: float
    phase1_converged: bool
    phase2_converged: bool
    converged: bool


# =============================================================================
# GRADIENT NORM TRACKER CALLBACK
# =============================================================================

class GradientNormTracker(Callback):
    """
    Callback to track BOTH base gradients AND update signals for two-step
    calibration.  Injected as ``extra_callbacks`` into ``train_single_fold``.

    FOR STEP 1 (Sparsity Calibration):
    - train_recon_grad_norm: ||∇ recon_loss||_F
    - train_hsic_cross_grad_norm: ||∇ hsic_value||_F (BASE, no lambda)
    - train_hsic_self_grad_norm: ||∇ hsic_self_value||_F (BASE, no lambda)
    - train_grad_ratio_cross / _self / _min

    FOR STEP 2 (Lambda Verification):
    - train_hsic_cross_update_norm: λ * ||∇ hsic_value||_F
    - train_hsic_self_update_norm: λ * ||∇ hsic_self_value||_F
    - train_update_ratio_cross / _self / _min
    """

    def __init__(self):
        super().__init__()
        # Base gradient tracking (Step 1)
        self.recon_grad_norms: List[float] = []
        self.hsic_cross_grad_norms: List[float] = []
        self.hsic_self_grad_norms: List[float] = []
        self.ratios_cross: List[float] = []
        self.ratios_self: List[float] = []
        self.ratios_min: List[float] = []

        # Update signal tracking (Step 2)
        self.hsic_cross_update_norms: List[float] = []
        self.hsic_self_update_norms: List[float] = []
        self.update_ratios_cross: List[float] = []
        self.update_ratios_self: List[float] = []
        self.update_ratios_min: List[float] = []

        # Per-batch accumulators for current epoch
        self._batch_recon_norms: List[float] = []

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule):
        self._batch_recon_norms = []

    def on_after_backward(self, trainer: Trainer, pl_module: pl.LightningModule):
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self._batch_recon_norms.append(total_norm ** 0.5)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics

        # ── BASE GRADIENTS (Step 1) ───────────────────────────────────────────
        recon_norm = metrics.get("train_recon_grad_norm", None)
        hsic_cross_norm = metrics.get("train_hsic_cross_grad_norm", None)
        hsic_self_norm = metrics.get("train_hsic_self_grad_norm", None)
        ratio_cross = metrics.get("train_grad_ratio_cross", None)
        ratio_self = metrics.get("train_grad_ratio_self", None)
        ratio_min = metrics.get("train_grad_ratio_min", None)

        if recon_norm is not None:
            self.recon_grad_norms.append(float(recon_norm.item()))
        elif self._batch_recon_norms:
            self.recon_grad_norms.append(
                float(np.sqrt(np.mean([n ** 2 for n in self._batch_recon_norms])))
            )

        self.hsic_cross_grad_norms.append(
            float(hsic_cross_norm.item()) if hsic_cross_norm is not None else 0.0
        )
        self.hsic_self_grad_norms.append(
            float(hsic_self_norm.item()) if hsic_self_norm is not None else 0.0
        )

        if ratio_cross is not None:
            self.ratios_cross.append(float(ratio_cross.item()))
        elif self.recon_grad_norms and self.hsic_cross_grad_norms[-1] > 1e-10:
            self.ratios_cross.append(
                float(self.recon_grad_norms[-1] / self.hsic_cross_grad_norms[-1])
            )

        if ratio_self is not None:
            self.ratios_self.append(float(ratio_self.item()))
        elif self.recon_grad_norms and self.hsic_self_grad_norms[-1] > 1e-10:
            self.ratios_self.append(
                float(self.recon_grad_norms[-1] / self.hsic_self_grad_norms[-1])
            )

        if ratio_min is not None:
            self.ratios_min.append(float(ratio_min.item()))
        else:
            ratios = []
            if self.ratios_cross:
                ratios.append(self.ratios_cross[-1])
            if self.ratios_self:
                ratios.append(self.ratios_self[-1])
            if ratios:
                self.ratios_min.append(min(ratios))

        # ── UPDATE SIGNALS (Step 2) ───────────────────────────────────────────
        hsic_cross_update = metrics.get("train_hsic_cross_update_norm", None)
        hsic_self_update = metrics.get("train_hsic_self_update_norm", None)
        update_ratio_cross = metrics.get("train_update_ratio_cross", None)
        update_ratio_self = metrics.get("train_update_ratio_self", None)
        update_ratio_min = metrics.get("train_update_ratio_min", None)

        self.hsic_cross_update_norms.append(
            float(hsic_cross_update.item()) if hsic_cross_update is not None else 0.0
        )
        self.hsic_self_update_norms.append(
            float(hsic_self_update.item()) if hsic_self_update is not None else 0.0
        )

        if update_ratio_cross is not None:
            self.update_ratios_cross.append(float(update_ratio_cross.item()))
        if update_ratio_self is not None:
            self.update_ratios_self.append(float(update_ratio_self.item()))
        if update_ratio_min is not None:
            self.update_ratios_min.append(float(update_ratio_min.item()))

        # Log for Lightning dashboard
        if self.ratios_min:
            pl_module.log("calibration_grad_ratio_min", self.ratios_min[-1], on_epoch=True)
        if self.update_ratios_min:
            pl_module.log("calibration_update_ratio_min", self.update_ratios_min[-1], on_epoch=True)

    # ── BASE GETTERS ─────────────────────────────────────────────────────────

    def get_mean_ratio_min(self) -> float:
        return float(np.mean(self.ratios_min)) if self.ratios_min else float("inf")

    def get_final_ratio_min(self) -> float:
        return self.ratios_min[-1] if self.ratios_min else float("inf")

    def get_mean_ratio_cross(self) -> float:
        return float(np.mean(self.ratios_cross)) if self.ratios_cross else float("inf")

    def get_mean_ratio_self(self) -> float:
        return float(np.mean(self.ratios_self)) if self.ratios_self else float("inf")

    # ── UPDATE GETTERS ────────────────────────────────────────────────────────

    def get_mean_update_ratio_min(self) -> float:
        return float(np.mean(self.update_ratios_min)) if self.update_ratios_min else float("inf")

    def get_final_update_ratio_min(self) -> float:
        return self.update_ratios_min[-1] if self.update_ratios_min else float("inf")

    def get_mean_update_ratio_cross(self) -> float:
        return float(np.mean(self.update_ratios_cross)) if self.update_ratios_cross else float("inf")

    def get_mean_update_ratio_self(self) -> float:
        return float(np.mean(self.update_ratios_self)) if self.update_ratios_self else float("inf")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _find_last_checkpoint(run_dir: Path) -> Optional[str]:
    """
    Locate the most recent checkpoint written by ``train_single_fold``.

    ``train_single_fold`` creates a ``k_0/checkpoints/`` subfolder via the
    standard ``get_checkpoint_callback``.  This helper finds ``last.ckpt`` or
    falls back to any ``.ckpt`` file in that folder.

    Args:
        run_dir: The ``save_dir`` passed to ``train_single_fold``.

    Returns:
        Absolute path string to the checkpoint, or ``None`` if none found.
    """
    ckpt_dir = run_dir / "k_0" / "checkpoints"
    if not ckpt_dir.exists():
        return None

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    # Fall back to the newest .ckpt file
    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(ckpt_files[-1]) if ckpt_files else None


def _build_calibration_config(config: dict, lambda_group: float, epochs: int) -> dict:
    """
    Build a short-run config for one calibration trial.

    Changes applied:
    - ``training.lambda_group_l1``      = lambda_group
    - ``training.log_gradient_norms``   = True
    - ``training.use_hsic_annealing``   = False
    - ``training.max_epochs``           = epochs  (short trial)
    - ``training.k_fold``               = 1       (single fold for speed)
    - ``training.save_ckpt_every_n_epochs`` guaranteed to exist

    Args:
        config:       Base configuration dict.
        lambda_group: Group-L1 coefficient for this trial.
        epochs:       Number of epochs to run.

    Returns:
        A deep copy of ``config`` with the above overrides applied.
    """
    config_cal = copy.deepcopy(config)
    config_cal["training"]["lambda_group_l1"] = float(lambda_group)
    config_cal["training"]["log_gradient_norms"] = True
    config_cal["training"]["use_hsic_annealing"] = False
    config_cal["training"]["max_epochs"] = int(epochs)
    config_cal["training"]["k_fold"] = 1
    # Ensure checkpoint-every-n-epochs exists (train_single_fold requires it)
    config_cal["training"].setdefault("save_ckpt_every_n_epochs", epochs)
    return config_cal


def _train_and_measure_gradient_ratio(
    config: dict,
    data_dir: str,
    save_dir: Path,
    lambda_group: float,
    epochs: int,
    seed: int,
    ratio_type: str = "base",
    run_name: str = None,
) -> Tuple[float, float, float, str, "GradientNormTracker"]:
    """
    Train for a fixed number of epochs and measure gradient ratios.

    Uses ``train_single_fold`` as the execution primitive, injecting a
    ``GradientNormTracker`` callback to capture gradient norms without
    modifying the training loop.

    Args:
        config:       Base configuration dictionary.
        data_dir:     Root data directory.
        save_dir:     Parent directory for this calibration run.
        lambda_group: Group-L1 coefficient to test.
        epochs:       Number of training epochs.
        seed:         Random seed.
        ratio_type:   ``"base"`` → use Phase-1 gradient ratios;
                      ``"update"`` → use Phase-2 update-signal ratios.
        run_name:     Subfolder name inside ``save_dir``
                      (default: ``lambda_{value:.2e}``).

    Returns:
        (ratio_min, ratio_cross, ratio_self, checkpoint_path, grad_tracker)
    """
    from causaliT.training.trainer import (
        create_model_instance,
        get_dataloader,
        train_single_fold,
        _make_fold_splits,
    )

    if run_name is None:
        run_name = f"lambda_{lambda_group:.2e}"
    run_dir = save_dir / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    # Build a short-run config
    config_cal = _build_calibration_config(config, lambda_group, epochs)

    # Resolve OmegaConf interpolations and populate dataset metadata.
    # When called from the sweep pre-hook, config arrives as a plain dict
    # with unresolved ${...} references (because the sweeper passes
    # resolve=False to preserve interpolations for per-combo overrides).
    # We must resolve them here before creating the model.
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset
    config_cal_omega = OmegaConf.create(config_cal)
    if data_dir is not None:
        config_cal_omega = populate_seq_lengths_from_dataset(config_cal_omega, data_dir)
    config_cal = OmegaConf.to_container(config_cal_omega, resolve=True)

    seed_everything(seed)
    model = create_model_instance(config_cal, data_dir)
    dm = get_dataloader(config_cal, data_dir, cluster=False, seed=seed)
    dm.prepare_data()

    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config_cal, dm, seed, data_dir=data_dir
    )
    train_local_idx, val_local_idx = fold_splits[0]

    grad_tracker = GradientNormTracker()

    train_single_fold(
        config=config_cal,
        model=model,
        dm=dm,
        fold=0,
        train_local_idx=train_local_idx,
        val_local_idx=val_local_idx,
        test_idx=test_idx,
        train_val_idx=train_val_idx,
        save_dir=str(run_dir),
        trainable_params=0,  # not needed for calibration
        cluster=False,
        extra_callbacks=[grad_tracker],
    )

    # Extract the ratios according to calibration phase
    if ratio_type == "base":
        ratio_min = grad_tracker.get_mean_ratio_min()
        ratio_cross = grad_tracker.get_mean_ratio_cross()
        ratio_self = grad_tracker.get_mean_ratio_self()
    else:  # "update"
        ratio_min = grad_tracker.get_mean_update_ratio_min()
        ratio_cross = grad_tracker.get_mean_update_ratio_cross()
        ratio_self = grad_tracker.get_mean_update_ratio_self()

    # Find checkpoint written by train_single_fold
    checkpoint_path = _find_last_checkpoint(run_dir)
    if checkpoint_path is None:
        logger.warning(f"No checkpoint found in {run_dir / 'k_0' / 'checkpoints'}")
        checkpoint_path = ""

    # Persist per-run metrics for post-hoc analysis
    metrics = {
        "lambda_group": lambda_group,
        "epochs": epochs,
        "ratio_type": ratio_type,
        # BASE ratios (Phase 1)
        "base_ratio_min": grad_tracker.get_mean_ratio_min(),
        "base_ratio_cross": grad_tracker.get_mean_ratio_cross(),
        "base_ratio_self": grad_tracker.get_mean_ratio_self(),
        # UPDATE ratios (Phase 2)
        "update_ratio_min": grad_tracker.get_mean_update_ratio_min(),
        "update_ratio_cross": grad_tracker.get_mean_update_ratio_cross(),
        "update_ratio_self": grad_tracker.get_mean_update_ratio_self(),
        # Per-epoch history
        "recon_grad_norms": grad_tracker.recon_grad_norms,
        "hsic_cross_grad_norms": grad_tracker.hsic_cross_grad_norms,
        "hsic_self_grad_norms": grad_tracker.hsic_self_grad_norms,
        "hsic_cross_update_norms": grad_tracker.hsic_cross_update_norms,
        "hsic_self_update_norms": grad_tracker.hsic_self_update_norms,
        "ratios_cross_per_epoch": grad_tracker.ratios_cross,
        "ratios_self_per_epoch": grad_tracker.ratios_self,
        "ratios_min_per_epoch": grad_tracker.ratios_min,
        "update_ratios_cross_per_epoch": grad_tracker.update_ratios_cross,
        "update_ratios_self_per_epoch": grad_tracker.update_ratios_self,
        "update_ratios_min_per_epoch": grad_tracker.update_ratios_min,
    }
    with open(run_dir / "calibration_metrics.json", "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )

    return ratio_min, ratio_cross, ratio_self, checkpoint_path, grad_tracker


# =============================================================================
# MAIN CALIBRATION ENTRY POINT
# =============================================================================

def calibrate_group_l1(
    config: dict,
    data_dir: str,
    save_dir: str,
    seed: int = 42,
) -> CalibrationResult:
    """
    Run calibration to find optimal λ_group for gradient balance.

    This is the main entry point for Stage 0 (Calibration).

    Process:
    1. Binary search over λ_group values (Phase 1 – BASE gradient ratios)
    2. For each λ_group, train for ``calibration_epochs`` and measure ratios
    3. Find λ_group where min(ratio_cross, ratio_self) ≈ 1.0
    4. Compute separate HSIC multipliers for cross and self
    5. Verification run with suggested λ_hsic values (Phase 2 – UPDATE ratios)

    The convergence criterion uses min(ratio_cross, ratio_self) to ensure
    NEITHER HSIC signal is drowned out.

    Args:
        config:   Configuration dictionary with ``staged_training`` section.
        data_dir: Root data directory.
        save_dir: Directory for calibration outputs.
        seed:     Random seed.

    Returns:
        CalibrationResult namedtuple.
    """
    cal_dir = Path(save_dir) / "calibration"
    cal_dir.mkdir(exist_ok=True, parents=True)

    staged_config = config.get("staged_training", {})
    cal_epochs = int(staged_config.get("calibration_epochs", 10))
    balance_threshold = float(staged_config.get("calibration_balance_threshold", 2.0))
    lambda_range = staged_config.get("calibration_lambda_group_range", [1e-5, 1e-1])
    max_iterations = int(staged_config.get("calibration_max_iterations", 10))

    print(f"\n{'='*70}")
    print("CALIBRATION: Finding optimal λ_group for SEPARATE HSIC gradient balance")
    print(f"{'='*70}")
    print(f"  Epochs per trial: {cal_epochs}")
    print(f"  Balance threshold: {balance_threshold}")
    print(f"  λ_group search range: [{lambda_range[0]:.2e}, {lambda_range[1]:.2e}]")
    print(
        f"  Convergence: min(ratio_cross, ratio_self) in "
        f"[{1/balance_threshold:.2f}, {balance_threshold:.2f}]"
    )

    # =========================================================================
    # PHASE 1: Binary search for λ_group
    # =========================================================================
    lambda_low = float(lambda_range[0])
    lambda_high = float(lambda_range[1])
    lambda_optimal = None
    best_ratio_min = None
    best_ratio_cross = None
    best_ratio_self = None
    best_checkpoint = None
    trials = []

    for iteration in range(int(max_iterations)):
        lambda_mid = float(np.sqrt(lambda_low * lambda_high))  # geometric midpoint

        print(f"\n  Iteration {iteration + 1}/{max_iterations}: testing λ_group = {lambda_mid:.2e}")

        ratio_min, ratio_cross, ratio_self, checkpoint_path, _ = _train_and_measure_gradient_ratio(
            config=config,
            data_dir=data_dir,
            save_dir=cal_dir,
            lambda_group=lambda_mid,
            epochs=cal_epochs,
            seed=seed,
            ratio_type="base",
        )

        trials.append(
            {
                "iteration": iteration + 1,
                "lambda_group": lambda_mid,
                "ratio_min": ratio_min,
                "ratio_cross": ratio_cross,
                "ratio_self": ratio_self,
                "checkpoint": checkpoint_path,
            }
        )

        print(f"    → Ratio (cross) = {ratio_cross:.3f}")
        print(f"    → Ratio (self)  = {ratio_self:.3f}")
        print(f"    → Ratio (min)   = {ratio_min:.3f} (target: ~1.0)")

        if 1 / balance_threshold <= ratio_min <= balance_threshold:
            print(
                f"    ✓ Balanced! min ratio in [{1/balance_threshold:.2f}, {balance_threshold:.2f}]"
            )
            lambda_optimal = lambda_mid
            best_ratio_min = ratio_min
            best_ratio_cross = ratio_cross
            best_ratio_self = ratio_self
            best_checkpoint = checkpoint_path
            break
        elif ratio_min > balance_threshold:
            print(f"    → Recon dominates. Increasing sparsity (λ_group ↑)")
            lambda_low = lambda_mid
        else:
            print(f"    → HSIC dominates. Decreasing sparsity (λ_group ↓)")
            lambda_high = lambda_mid

        # Track best so far (closest to 1.0)
        if best_ratio_min is None or abs(ratio_min - 1.0) < abs(best_ratio_min - 1.0):
            lambda_optimal = lambda_mid
            best_ratio_min = ratio_min
            best_ratio_cross = ratio_cross
            best_ratio_self = ratio_self
            best_checkpoint = checkpoint_path

    if lambda_optimal is None:
        lambda_optimal = lambda_mid  # type: ignore[possibly-undefined]
        best_ratio_min = ratio_min
        best_ratio_cross = ratio_cross
        best_ratio_self = ratio_self
        best_checkpoint = checkpoint_path
        print(f"\n  Warning: Calibration did not converge within {max_iterations} iterations.")

    # =========================================================================
    # COMPUTE SEPARATE λ_hsic MULTIPLIERS
    # =========================================================================
    lambda_hsic_cross = config["training"].get(
        "lambda_hsic_cross", config["training"].get("lambda_hsic", 0.1)
    )
    lambda_hsic_self = config["training"].get("lambda_hsic_self", 0.0)

    suggested_lambda_hsic_cross = (
        best_ratio_cross if best_ratio_cross != float("inf") else lambda_hsic_cross
    )
    suggested_lambda_hsic_self = (
        best_ratio_self if best_ratio_self != float("inf") else lambda_hsic_self
    )

    print(f"\n  PHASE 1 COMPLETE – SEPARATE λ_hsic RECOMMENDATIONS:")
    print(f"    Cross-attention (S→X):")
    print(f"      BASE ratio = {best_ratio_cross:.3f}")
    print(f"      Suggested λ_hsic_cross = {suggested_lambda_hsic_cross:.4f}")
    if lambda_hsic_self > 0:
        print(f"    Self-attention (X→X):")
        print(f"      BASE ratio = {best_ratio_self:.3f}")
        print(f"      Suggested λ_hsic_self = {suggested_lambda_hsic_self:.4f}")
    else:
        print(f"    Self-attention (X→X): disabled (λ_hsic_self = 0)")

    # =========================================================================
    # PHASE 2: Verification run with suggested λ_hsic values
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Verifying suggested λ_hsic values")
    print(f"{'='*70}")
    print(f"  Running {cal_epochs} epochs with suggested λ values…")
    print(f"  λ_group       = {lambda_optimal:.2e}")
    print(f"  λ_hsic_cross  = {suggested_lambda_hsic_cross:.4f}")
    print(f"  λ_hsic_self   = {suggested_lambda_hsic_self:.4f}")

    config_verify = copy.deepcopy(config)
    config_verify["training"]["lambda_hsic_cross"] = float(suggested_lambda_hsic_cross)
    config_verify["training"]["lambda_hsic_self"] = float(suggested_lambda_hsic_self)

    verify_ratio_min, verify_ratio_cross, verify_ratio_self, verify_checkpoint, verify_tracker = \
        _train_and_measure_gradient_ratio(
            config=config_verify,
            data_dir=data_dir,
            save_dir=cal_dir,
            lambda_group=lambda_optimal,
            epochs=cal_epochs,
            seed=seed,
            ratio_type="update",
            run_name="verification",
        )

    print(f"\n  PHASE 2 RESULTS (UPDATE ratios – should be ≈ 1.0):")
    print(f"    UPDATE ratio (cross) = {verify_ratio_cross:.3f}")
    print(f"    UPDATE ratio (self)  = {verify_ratio_self:.3f}")
    print(f"    UPDATE ratio (min)   = {verify_ratio_min:.3f}")

    verification_converged = 1 / balance_threshold <= verify_ratio_min <= balance_threshold
    if verification_converged:
        print(
            f"    ✓ VERIFICATION PASSED! Update ratio in "
            f"[{1/balance_threshold:.2f}, {balance_threshold:.2f}]"
        )
    else:
        print(
            f"    ⚠ VERIFICATION WARNING: Update ratio outside "
            f"[{1/balance_threshold:.2f}, {balance_threshold:.2f}]"
        )

    phase1_converged = 1 / balance_threshold <= best_ratio_min <= balance_threshold
    converged = phase1_converged and verification_converged

    # =========================================================================
    # SAVE CALIBRATION SUMMARY
    # =========================================================================
    summary = {
        "phase1": {
            "lambda_group_optimal": lambda_optimal,
            "base_ratio_min": best_ratio_min,
            "base_ratio_cross": best_ratio_cross,
            "base_ratio_self": best_ratio_self,
            "converged": phase1_converged,
            "trials": trials,
            "checkpoint": best_checkpoint,
        },
        "phase2": {
            "suggested_lambda_hsic_cross": suggested_lambda_hsic_cross,
            "suggested_lambda_hsic_self": suggested_lambda_hsic_self,
            "update_ratio_min": verify_ratio_min,
            "update_ratio_cross": verify_ratio_cross,
            "update_ratio_self": verify_ratio_self,
            "converged": verification_converged,
            "checkpoint": verify_checkpoint,
        },
        "converged": converged,
        "verification_checkpoint": verify_checkpoint,
        "original_lambda_hsic_cross": lambda_hsic_cross,
        "original_lambda_hsic_self": lambda_hsic_self,
    }
    with open(cal_dir / "calibration_summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )

    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Phase 1 (Sparsity):  λ_group* = {lambda_optimal:.2e}")
    print(f"    BASE ratio (cross) = {best_ratio_cross:.3f}")
    print(f"    BASE ratio (self)  = {best_ratio_self:.3f}")
    print(f"    Phase 1 converged: {phase1_converged}")
    print(f"  Phase 2 (Verification):")
    print(f"    λ_hsic_cross = {suggested_lambda_hsic_cross:.4f}")
    print(f"    λ_hsic_self  = {suggested_lambda_hsic_self:.4f}")
    print(f"    UPDATE ratio (cross) = {verify_ratio_cross:.3f}")
    print(f"    UPDATE ratio (self)  = {verify_ratio_self:.3f}")
    print(f"    Phase 2 converged: {verification_converged}")
    print(f"  Overall converged: {converged}")
    print(f"  Final checkpoint: {verify_checkpoint}")
    print(f"{'='*70}\n")

    return CalibrationResult(
        lambda_group_optimal=lambda_optimal,
        lambda_hsic_cross_suggested=suggested_lambda_hsic_cross,
        lambda_hsic_self_suggested=suggested_lambda_hsic_self,
        checkpoint_path=verify_checkpoint,
        base_ratio_cross=best_ratio_cross,
        base_ratio_self=best_ratio_self,
        update_ratio_cross=verify_ratio_cross,
        update_ratio_self=verify_ratio_self,
        phase1_converged=phase1_converged,
        phase2_converged=verification_converged,
        converged=converged,
    )
