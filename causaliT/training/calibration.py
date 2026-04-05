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

import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


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


class GradientNormTracker(Callback):
    """
    Callback to track BOTH base gradients AND update signals for two-step calibration.
    
    This callback tracks:
    
    FOR STEP 1 (Sparsity Calibration):
    - train_recon_grad_norm: ||∇ recon_loss||_F
    - train_hsic_cross_grad_norm: ||∇ hsic_value||_F (BASE, no lambda)
    - train_hsic_self_grad_norm: ||∇ hsic_self_value||_F (BASE, no lambda)
    - train_grad_ratio_cross: recon / hsic_cross_base
    - train_grad_ratio_self: recon / hsic_self_base
    - train_grad_ratio_min: min of the two
    
    FOR STEP 2 (Lambda Verification):
    - train_hsic_cross_update_norm: λ * ||∇ hsic_value||_F (UPDATE signal)
    - train_hsic_self_update_norm: λ * ||∇ hsic_self_value||_F (UPDATE signal)
    - train_update_ratio_cross: recon / hsic_cross_update
    - train_update_ratio_self: recon / hsic_self_update
    - train_update_ratio_min: min of the two
    
    Attributes:
        recon_grad_norms: List of ||∇Recon||_F per epoch
        hsic_cross_grad_norms: List of BASE ||∇HSIC_cross||_F per epoch
        hsic_self_grad_norms: List of BASE ||∇HSIC_self||_F per epoch
        hsic_cross_update_norms: List of UPDATE ||λ * ∇HSIC_cross||_F per epoch
        hsic_self_update_norms: List of UPDATE ||λ * ∇HSIC_self||_F per epoch
        ratios_cross: List of BASE recon/hsic_cross ratios per epoch
        ratios_self: List of BASE recon/hsic_self ratios per epoch
        ratios_min: List of min(BASE ratio_cross, BASE ratio_self) per epoch
        update_ratios_cross: List of UPDATE recon/hsic_cross_update ratios per epoch
        update_ratios_self: List of UPDATE recon/hsic_self_update ratios per epoch
        update_ratios_min: List of min(UPDATE ratio_cross, UPDATE ratio_self) per epoch
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
        """Reset batch accumulators at the start of each epoch."""
        self._batch_recon_norms = []
    
    def on_after_backward(self, trainer: Trainer, pl_module: pl.LightningModule):
        """
        Called after loss.backward() but before optimizer.step().
        
        Store total gradient norm as fallback for reconstruction.
        """
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self._batch_recon_norms.append(total_norm)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        """
        Aggregate gradient norms from forecaster's logged metrics.
        
        The forecaster logs these metrics when log_gradient_norms=True:
        
        BASE GRADIENTS (Step 1):
        - train_recon_grad_norm: ||∇L_recon||_F
        - train_hsic_cross_grad_norm: ||∇ hsic_value||_F (base, no lambda)
        - train_hsic_self_grad_norm: ||∇ hsic_self_value||_F (base, no lambda)
        - train_grad_ratio_cross: recon / hsic_cross_base
        - train_grad_ratio_self: recon / hsic_self_base
        - train_grad_ratio_min: min of the two
        
        UPDATE SIGNALS (Step 2):
        - train_hsic_cross_update_norm: λ * ||∇ hsic_value||_F
        - train_hsic_self_update_norm: λ * ||∇ hsic_self_value||_F
        - train_update_ratio_cross: recon / hsic_cross_update
        - train_update_ratio_self: recon / hsic_self_update
        - train_update_ratio_min: min of the two
        """
        metrics = trainer.callback_metrics
        
        # =====================================================================
        # BASE GRADIENTS (Step 1: Sparsity Calibration)
        # =====================================================================
        recon_norm = metrics.get("train_recon_grad_norm", None)
        hsic_cross_norm = metrics.get("train_hsic_cross_grad_norm", None)
        hsic_self_norm = metrics.get("train_hsic_self_grad_norm", None)
        ratio_cross = metrics.get("train_grad_ratio_cross", None)
        ratio_self = metrics.get("train_grad_ratio_self", None)
        ratio_min = metrics.get("train_grad_ratio_min", None)
        
        # Store recon norm
        if recon_norm is not None:
            self.recon_grad_norms.append(float(recon_norm.item()))
        elif self._batch_recon_norms:
            epoch_recon_norm = np.sqrt(np.mean([n**2 for n in self._batch_recon_norms]))
            self.recon_grad_norms.append(float(epoch_recon_norm))
        
        # Store HSIC cross BASE norm
        if hsic_cross_norm is not None:
            self.hsic_cross_grad_norms.append(float(hsic_cross_norm.item()))
        else:
            self.hsic_cross_grad_norms.append(0.0)
        
        # Store HSIC self BASE norm
        if hsic_self_norm is not None:
            self.hsic_self_grad_norms.append(float(hsic_self_norm.item()))
        else:
            self.hsic_self_grad_norms.append(0.0)
        
        # Store BASE ratios
        if ratio_cross is not None:
            self.ratios_cross.append(float(ratio_cross.item()))
        elif self.recon_grad_norms and self.hsic_cross_grad_norms[-1] > 1e-10:
            self.ratios_cross.append(float(self.recon_grad_norms[-1] / self.hsic_cross_grad_norms[-1]))
        
        if ratio_self is not None:
            self.ratios_self.append(float(ratio_self.item()))
        elif self.recon_grad_norms and self.hsic_self_grad_norms[-1] > 1e-10:
            self.ratios_self.append(float(self.recon_grad_norms[-1] / self.hsic_self_grad_norms[-1]))
        
        if ratio_min is not None:
            self.ratios_min.append(float(ratio_min.item()))
        else:
            # Compute min manually
            ratios = []
            if self.ratios_cross:
                ratios.append(self.ratios_cross[-1])
            if self.ratios_self:
                ratios.append(self.ratios_self[-1])
            if ratios:
                self.ratios_min.append(min(ratios))
        
        # =====================================================================
        # UPDATE SIGNALS (Step 2: Lambda Verification)
        # =====================================================================
        hsic_cross_update = metrics.get("train_hsic_cross_update_norm", None)
        hsic_self_update = metrics.get("train_hsic_self_update_norm", None)
        update_ratio_cross = metrics.get("train_update_ratio_cross", None)
        update_ratio_self = metrics.get("train_update_ratio_self", None)
        update_ratio_min = metrics.get("train_update_ratio_min", None)
        
        # Store HSIC cross UPDATE norm
        if hsic_cross_update is not None:
            self.hsic_cross_update_norms.append(float(hsic_cross_update.item()))
        else:
            self.hsic_cross_update_norms.append(0.0)
        
        # Store HSIC self UPDATE norm
        if hsic_self_update is not None:
            self.hsic_self_update_norms.append(float(hsic_self_update.item()))
        else:
            self.hsic_self_update_norms.append(0.0)
        
        # Store UPDATE ratios
        if update_ratio_cross is not None:
            self.update_ratios_cross.append(float(update_ratio_cross.item()))
        
        if update_ratio_self is not None:
            self.update_ratios_self.append(float(update_ratio_self.item()))
        
        if update_ratio_min is not None:
            self.update_ratios_min.append(float(update_ratio_min.item()))
        
        # Log for monitoring
        if self.ratios_min:
            pl_module.log("calibration_grad_ratio_min", self.ratios_min[-1], on_epoch=True)
        if self.update_ratios_min:
            pl_module.log("calibration_update_ratio_min", self.update_ratios_min[-1], on_epoch=True)
    
    # =========================================================================
    # BASE GRADIENT GETTERS (Step 1)
    # =========================================================================
    
    def get_mean_ratio_min(self) -> float:
        """Get mean of min(BASE ratio_cross, BASE ratio_self) across all epochs."""
        if not self.ratios_min:
            return float('inf')
        return float(np.mean(self.ratios_min))
    
    def get_final_ratio_min(self) -> float:
        """Get min BASE ratio from final epoch."""
        if not self.ratios_min:
            return float('inf')
        return self.ratios_min[-1]
    
    def get_mean_ratio_cross(self) -> float:
        """Get mean BASE cross-attention ratio across all epochs."""
        if not self.ratios_cross:
            return float('inf')
        return float(np.mean(self.ratios_cross))
    
    def get_mean_ratio_self(self) -> float:
        """Get mean BASE self-attention ratio across all epochs."""
        if not self.ratios_self:
            return float('inf')
        return float(np.mean(self.ratios_self))
    
    # =========================================================================
    # UPDATE SIGNAL GETTERS (Step 2)
    # =========================================================================
    
    def get_mean_update_ratio_min(self) -> float:
        """Get mean of min(UPDATE ratio_cross, UPDATE ratio_self) across all epochs."""
        if not self.update_ratios_min:
            return float('inf')
        return float(np.mean(self.update_ratios_min))
    
    def get_final_update_ratio_min(self) -> float:
        """Get min UPDATE ratio from final epoch."""
        if not self.update_ratios_min:
            return float('inf')
        return self.update_ratios_min[-1]
    
    def get_mean_update_ratio_cross(self) -> float:
        """Get mean UPDATE cross-attention ratio across all epochs."""
        if not self.update_ratios_cross:
            return float('inf')
        return float(np.mean(self.update_ratios_cross))
    
    def get_mean_update_ratio_self(self) -> float:
        """Get mean UPDATE self-attention ratio across all epochs."""
        if not self.update_ratios_self:
            return float('inf')
        return float(np.mean(self.update_ratios_self))


def _train_and_measure_gradient_ratio(
    config: dict,
    data_dir: str,
    save_dir: Path,
    lambda_group: float,
    epochs: int,
    seed: int,
    ratio_type: str = "base",
    run_name: str = None,
) -> Tuple[float, float, float, str, GradientNormTracker]:
    """
    Train for a few epochs and measure gradient ratios.
    
    This function supports two phases of calibration:
    - Phase 1 (ratio_type="base"): Uses BASE gradient ratios (λ-independent)
      to find optimal λ_group via binary search
    - Phase 2 (ratio_type="update"): Uses UPDATE ratios (learning signals, λ-dependent)
      to verify that suggested λ_hsic values achieve balance
    
    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        save_dir: Directory to save calibration run outputs
        lambda_group: Group L1 coefficient to test
        epochs: Number of epochs to train
        seed: Random seed
        ratio_type: "base" for Phase 1 (gradient norms), "update" for Phase 2 (learning signals)
        run_name: Optional custom name for run directory (default: lambda_{value})
        
    Returns:
        Tuple of (ratio_min, ratio_cross, ratio_self, checkpoint_path, grad_tracker)
    """
    from causaliT.training.trainer import create_model_instance, get_dataloader
    
    # Setup run directory
    if run_name is None:
        run_name = f"lambda_{lambda_group:.2e}"
    run_dir = save_dir / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Reset seed for reproducibility
    seed_everything(seed)
    
    # Modify config for this calibration run
    config_cal = config.copy()
    
    config_cal["training"]["lambda_group_l1"] = float(lambda_group)
    config_cal["training"]["log_gradient_norms"] = True  # Enable gradient logging
    
    # Disable any annealing during calibration - need constant hyperparameters
    config_cal["training"]["use_hsic_annealing"] = False

    # Create model and dataloader
    model = create_model_instance(config_cal, data_dir)
    dm = get_dataloader(config_cal, data_dir, cluster=False, seed=seed)
    dm.setup(stage='fit')
    
    # Setup gradient tracker callback
    grad_tracker = GradientNormTracker()
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=epochs,
        default_root_dir=str(run_dir),
        callbacks=[grad_tracker],
        logger=CSVLogger(save_dir=str(run_dir), name="cal_logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
    )
    
    # Train
    trainer.fit(model, dm)
    
    # Get gradient ratios based on ratio_type
    if ratio_type == "base":
        # Phase 1: BASE gradient ratios (λ-independent)
        ratio_min = grad_tracker.get_mean_ratio_min()
        ratio_cross = grad_tracker.get_mean_ratio_cross()
        ratio_self = grad_tracker.get_mean_ratio_self()
    else:  # "update"
        # Phase 2: UPDATE ratios (learning signals, λ-dependent)
        ratio_min = grad_tracker.get_mean_update_ratio_min()
        ratio_cross = grad_tracker.get_mean_update_ratio_cross()
        ratio_self = grad_tracker.get_mean_update_ratio_self()
    
    # Save checkpoint
    checkpoint_path = str(run_dir / "checkpoint.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    
    # Save metrics (include both base and update for completeness)
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
        # Per-epoch details
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
    with open(run_dir / "calibration_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    return ratio_min, ratio_cross, ratio_self, checkpoint_path, grad_tracker


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
    1. Binary search over λ_group values
    2. For each λ_group, train for calibration_epochs and measure gradient ratios
    3. Find λ_group where min(ratio_cross, ratio_self) ≈ 1 (both HSIC signals balanced)
    4. Compute separate multipliers for cross and self HSIC
    5. Return calibrated values and checkpoint
    
    The convergence criterion uses min(ratio_cross, ratio_self) to ensure
    NEITHER HSIC signal is drowned out. If only one HSIC is active, that
    ratio is used.
    
    Args:
        config: Configuration dictionary with staged_training section
        data_dir: Data directory path
        save_dir: Save directory for calibration outputs
        seed: Random seed
        
    Returns:
        CalibrationResult with:
        - lambda_group_optimal: Calibrated group L1 coefficient
        - lambda_hsic_cross_multiplier: Suggested multiplier for λ_hsic_cross
        - lambda_hsic_self_multiplier: Suggested multiplier for λ_hsic_self
        - checkpoint_path: Path to best calibration checkpoint
        - ratio_cross, ratio_self: Final ratios for reference
        - converged: Whether calibration converged within threshold
    """
    # Setup calibration directory
    cal_dir = Path(save_dir) / "calibration"
    cal_dir.mkdir(exist_ok=True, parents=True)
    
    # Get calibration parameters from config
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
    print(f"  Convergence: min(ratio_cross, ratio_self) in [{1/balance_threshold:.2f}, {balance_threshold:.2f}]")
    
    # Binary search for optimal λ_group
    lambda_low = float(lambda_range[0])
    lambda_high = float(lambda_range[1])
    lambda_optimal = None
    best_ratio_min = None
    best_ratio_cross = None
    best_ratio_self = None
    best_checkpoint = None
    
    # Track all trials
    trials = []
    
    for iteration in range(int(max_iterations)):
        # Geometric mean for log-scale search
        lambda_mid = float(np.sqrt(lambda_low * lambda_high))
        
        print(f"\n  Iteration {iteration + 1}/{max_iterations}: testing λ_group = {lambda_mid:.2e}")
        
        # Train and measure using BASE gradient ratios (Phase 1)
        ratio_min, ratio_cross, ratio_self, checkpoint_path, _ = _train_and_measure_gradient_ratio(
            config=config,
            data_dir=data_dir,
            save_dir=cal_dir,
            lambda_group=lambda_mid,
            epochs=cal_epochs,
            seed=seed,
            ratio_type="base",  # Phase 1: BASE gradient ratios
        )
        
        trials.append({
            "iteration": iteration + 1,
            "lambda_group": lambda_mid,
            "ratio_min": ratio_min,
            "ratio_cross": ratio_cross,
            "ratio_self": ratio_self,
            "checkpoint": checkpoint_path,
        })
        
        print(f"    → Ratio (cross) = {ratio_cross:.3f}")
        print(f"    → Ratio (self)  = {ratio_self:.3f}")
        print(f"    → Ratio (min)   = {ratio_min:.3f} (target: ~1.0)")
        
        # Check if balanced using min ratio
        if 1/balance_threshold <= ratio_min <= balance_threshold:
            print(f"    ✓ Balanced! min ratio in [{1/balance_threshold:.2f}, {balance_threshold:.2f}]")
            lambda_optimal = lambda_mid
            best_ratio_min = ratio_min
            best_ratio_cross = ratio_cross
            best_ratio_self = ratio_self
            best_checkpoint = checkpoint_path
            break
        elif ratio_min > balance_threshold:
            # Recon gradient dominates both HSICs → increase sparsity
            print(f"    → Recon dominates. Increasing sparsity (λ_group ↑)")
            lambda_low = lambda_mid
        else:
            # At least one HSIC gradient dominates → decrease sparsity
            print(f"    → HSIC dominates. Decreasing sparsity (λ_group ↓)")
            lambda_high = lambda_mid
        
        # Track best so far (closest to 1.0)
        if best_ratio_min is None or abs(ratio_min - 1.0) < abs(best_ratio_min - 1.0):
            lambda_optimal = lambda_mid
            best_ratio_min = ratio_min
            best_ratio_cross = ratio_cross
            best_ratio_self = ratio_self
            best_checkpoint = checkpoint_path
    
    # If didn't converge, use best found
    if lambda_optimal is None:
        lambda_optimal = lambda_mid
        best_ratio_min = ratio_min
        best_ratio_cross = ratio_cross
        best_ratio_self = ratio_self
        best_checkpoint = checkpoint_path
        print(f"\n  Warning: Calibration did not converge within {max_iterations} iterations.")
    
    # =========================================================================
    # COMPUTE SEPARATE MULTIPLIERS FOR CROSS AND SELF HSIC
    # =========================================================================
    # The multiplier tells us how to scale the current λ_hsic to balance gradients.
    # multiplier = ratio means: new_lambda = old_lambda * ratio to get balanced
    # We want the EFFECTIVE gradient (λ * base_grad) ≈ recon_grad
    # So: λ_new * base_grad = recon_grad → λ_new = recon_grad / base_grad = ratio
    
    # For cross-attention HSIC
    if best_ratio_cross is not None and best_ratio_cross != float('inf'):
        lambda_hsic_cross_multiplier = best_ratio_cross
    else:
        lambda_hsic_cross_multiplier = 1.0
    
    # For self-attention HSIC
    if best_ratio_self is not None and best_ratio_self != float('inf'):
        lambda_hsic_self_multiplier = best_ratio_self
    else:
        lambda_hsic_self_multiplier = 1.0
    
    # Print recommendations
    lambda_hsic_cross = config["training"].get("lambda_hsic_cross", 
                            config["training"].get("lambda_hsic", 0.1))
    lambda_hsic_self = config["training"].get("lambda_hsic_self", 0.0)
    
    # Compute suggested lambda values
    suggested_lambda_hsic_cross = best_ratio_cross if best_ratio_cross != float('inf') else lambda_hsic_cross
    suggested_lambda_hsic_self = best_ratio_self if best_ratio_self != float('inf') else lambda_hsic_self
    
    print(f"\n  PHASE 1 COMPLETE - SEPARATE λ_hsic RECOMMENDATIONS:")
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
    # PHASE 2: VERIFICATION RUN WITH SUGGESTED LAMBDA VALUES
    # =========================================================================
    # Now train with the suggested λ_hsic values and verify that UPDATE ratios ≈ 1.0
    # This confirms the calibration worked correctly
    
    print(f"\n{'='*70}")
    print("PHASE 2: Verifying suggested λ_hsic values")
    print(f"{'='*70}")
    print(f"  Running {cal_epochs} epochs with suggested λ values...")
    print(f"  λ_group = {lambda_optimal:.2e}")
    print(f"  λ_hsic_cross = {suggested_lambda_hsic_cross:.4f}")
    print(f"  λ_hsic_self = {suggested_lambda_hsic_self:.4f}")
    
    # Create config with suggested lambda values for verification
    import copy
    config_verify = copy.deepcopy(config)
    config_verify["training"]["lambda_hsic_cross"] = float(suggested_lambda_hsic_cross)
    config_verify["training"]["lambda_hsic_self"] = float(suggested_lambda_hsic_self)
    
    # Run verification with UPDATE ratio type
    verify_ratio_min, verify_ratio_cross, verify_ratio_self, verify_checkpoint, verify_tracker = _train_and_measure_gradient_ratio(
        config=config_verify,
        data_dir=data_dir,
        save_dir=cal_dir,
        lambda_group=lambda_optimal,
        epochs=cal_epochs,
        seed=seed,
        ratio_type="update",  # Phase 2: Check UPDATE ratios
        run_name="verification",
    )
    
    print(f"\n  PHASE 2 RESULTS (UPDATE ratios - should be ≈ 1.0):")
    print(f"    UPDATE ratio (cross) = {verify_ratio_cross:.3f}")
    print(f"    UPDATE ratio (self)  = {verify_ratio_self:.3f}")
    print(f"    UPDATE ratio (min)   = {verify_ratio_min:.3f}")
    
    # Check if verification passed
    verification_converged = 1/balance_threshold <= verify_ratio_min <= balance_threshold
    if verification_converged:
        print(f"    ✓ VERIFICATION PASSED! Update ratio in [{1/balance_threshold:.2f}, {balance_threshold:.2f}]")
    else:
        print(f"    ⚠ VERIFICATION WARNING: Update ratio outside [{1/balance_threshold:.2f}, {balance_threshold:.2f}]")
        print(f"      This may indicate gradients changed significantly during training.")
    
    phase1_converged = 1/balance_threshold <= best_ratio_min <= balance_threshold
    converged = phase1_converged and verification_converged
    
    # Save calibration summary with both phases
    summary = {
        # Phase 1 results
        "phase1": {
            "lambda_group_optimal": lambda_optimal,
            "base_ratio_min": best_ratio_min,
            "base_ratio_cross": best_ratio_cross,
            "base_ratio_self": best_ratio_self,
            "converged": phase1_converged,
            "trials": trials,
            "checkpoint": best_checkpoint,
        },
        # Phase 2 results
        "phase2": {
            "suggested_lambda_hsic_cross": suggested_lambda_hsic_cross,
            "suggested_lambda_hsic_self": suggested_lambda_hsic_self,
            "update_ratio_min": verify_ratio_min,
            "update_ratio_cross": verify_ratio_cross,
            "update_ratio_self": verify_ratio_self,
            "converged": verification_converged,
            "checkpoint": verify_checkpoint,
        },
        # Overall
        "converged": converged,
        "verification_checkpoint": verify_checkpoint,
        # Config reference
        "original_lambda_hsic_cross": lambda_hsic_cross,
        "original_lambda_hsic_self": lambda_hsic_self,
    }
    
    with open(cal_dir / "calibration_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    
    print(f"\n{'='*70}")
    print(f"CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Phase 1 (Sparsity):")
    print(f"    λ_group* = {lambda_optimal:.2e}")
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
