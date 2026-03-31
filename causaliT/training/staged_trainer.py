"""
Staged Training Orchestrator: Coordinates calibration → causal_init → training.

This module implements the complete staged training pipeline that:
1. (Optional) Stage 0: Calibrates group L1 sparsity for gradient balance
2. (Optional) Stage 1: Runs causal initialization with HSIC-dominated loss
3. Stage 2: Runs standard training with optional checkpoint loading

The pipeline handles all combinations of stage configurations:
- calibration ON  + causal_init ON  → cal_ckpt → init_ckpt → train
- calibration ON  + causal_init OFF → cal_ckpt → train
- calibration OFF + causal_init ON  → fresh → init_ckpt → train
- calibration OFF + causal_init OFF → train (fresh) - standard behavior

This design allows for:
- Ablation studies (enable/disable each stage independently)
- Flexibility (skip calibration if λ_group is known)
- Reproducibility (checkpoints save exact state between stages)

Usage:
    from causaliT.training.staged_trainer import staged_trainer
    
    df_metrics = staged_trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=False,
    )
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def staged_trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str = "NA",
    resume_ckpt: str = None,
    plot_pred_check: bool = False,
    debug: bool = False,
    best: bool = False,
) -> pd.DataFrame:
    """
    Run staged training pipeline.
    
    This is the main entry point for the staged training infrastructure.
    It orchestrates the following stages:
    
    Stage 0 (Calibration - optional):
        Finds optimal λ_group such that ||∇Recon|| ≈ ||∇HSIC||.
        This ensures the HSIC signal remains meaningful during training.
        
    Stage 1 (Causal Init - optional):
        Trains with HSIC-dominated loss to initialize the model toward
        causal structure before standard fitting-focused training.
        
    Stage 2 (Main Training):
        Standard training with optional checkpoint loading from previous stages.
        Uses HSIC annealing to transition from structure learning to fitting.
    
    Args:
        config: Configuration dictionary containing model, training, and
               staged_training settings
        data_dir: Path to data directory
        save_dir: Path to save outputs (checkpoints, logs, etc.)
        cluster: Whether running on a compute cluster
        experiment_tag: Tag for experiment tracking
        resume_ckpt: Optional checkpoint to resume from (overrides staged checkpoints)
        plot_pred_check: Whether to plot prediction checks
        debug: Enable debug mode
        best: Use best checkpoint metrics instead of final epoch
        
    Returns:
        DataFrame with training metrics for each fold
    """
    from causaliT.training.calibration import calibrate_group_l1
    from causaliT.training.causal_initialization import run_causal_initialization
    from causaliT.training.trainer import trainer
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset
    
    # Get staged training configuration
    staged_config = config.get("staged_training", {})
    use_calibration = staged_config.get("use_calibration", False)
    use_causal_init = staged_config.get("use_causal_init", False)
    
    seed = config["training"].get("seed", 42)
    
    # If resume_ckpt is provided, skip staged training and go directly to main training
    if resume_ckpt is not None:
        print(f"\nResume checkpoint provided: {resume_ckpt}")
        print("Skipping staged training (calibration/causal_init) and resuming main training.")
        starting_checkpoint = resume_ckpt
    else:
        starting_checkpoint = None
    
    # Populate sequence lengths from dataset metadata (needed for all stages)
    config = populate_seq_lengths_from_dataset(config, data_dir)
    
    # Create staged training summary
    staged_summary = {
        "use_calibration": use_calibration,
        "use_causal_init": use_causal_init,
        "resume_from": resume_ckpt,
        "stages_completed": [],
    }
    
    # =========================================================================
    # STAGE 0: CALIBRATION
    # =========================================================================
    # Store calibration results for use in causal init
    # If calibration runs: suggested values are set in config, multipliers = 1.0
    # If calibration doesn't run: multipliers = None (use causal_init defaults)
    suggested_lambda_hsic_cross = None
    suggested_lambda_hsic_self = None
    
    if use_calibration and starting_checkpoint is None:
        print("\n" + "="*70)
        print("STAGED TRAINING: STAGE 0 - CALIBRATION")
        print("="*70)
        
        cal_result = calibrate_group_l1(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            seed=seed,
        )
        
        # Unpack calibration result (CalibrationResult namedtuple)
        # Note: The suggested lambda values are the BASE ratios from Phase 1
        # They should be used directly as lambda_hsic values for balanced gradients
        lambda_group = cal_result.lambda_group_optimal
        suggested_lambda_hsic_cross = cal_result.lambda_hsic_cross_suggested
        suggested_lambda_hsic_self = cal_result.lambda_hsic_self_suggested
        cal_ckpt = cal_result.checkpoint_path
        
        # Update config with calibrated values
        if "staged_training" not in config:
            config["staged_training"] = {}
        config["staged_training"]["lambda_group_l1"] = lambda_group
        config["staged_training"]["calibration_checkpoint"] = cal_ckpt
        config["staged_training"]["lambda_hsic_cross_suggested"] = suggested_lambda_hsic_cross
        config["staged_training"]["lambda_hsic_self_suggested"] = suggested_lambda_hsic_self
        
        # Also update the training config to use the suggested lambda values
        config["training"]["lambda_hsic_cross"] = suggested_lambda_hsic_cross
        config["training"]["lambda_hsic_self"] = suggested_lambda_hsic_self
        
        # Update starting checkpoint for next stage
        starting_checkpoint = cal_ckpt
        
        staged_summary["stages_completed"].append("calibration")
        staged_summary["lambda_group_optimal"] = float(lambda_group)
        staged_summary["lambda_hsic_cross_suggested"] = float(suggested_lambda_hsic_cross)
        staged_summary["lambda_hsic_self_suggested"] = float(suggested_lambda_hsic_self)
        staged_summary["base_ratio_cross"] = float(cal_result.base_ratio_cross) if cal_result.base_ratio_cross else None
        staged_summary["base_ratio_self"] = float(cal_result.base_ratio_self) if cal_result.base_ratio_self else None
        staged_summary["update_ratio_cross"] = float(cal_result.update_ratio_cross) if cal_result.update_ratio_cross else None
        staged_summary["update_ratio_self"] = float(cal_result.update_ratio_self) if cal_result.update_ratio_self else None
        staged_summary["phase1_converged"] = cal_result.phase1_converged
        staged_summary["phase2_converged"] = cal_result.phase2_converged
        staged_summary["calibration_converged"] = cal_result.converged
        staged_summary["calibration_checkpoint"] = cal_ckpt
        
        print(f"Calibration complete: λ_group = {lambda_group:.2e}")
        print(f"  Suggested λ_hsic_cross = {suggested_lambda_hsic_cross:.4f}")
        print(f"  Suggested λ_hsic_self = {suggested_lambda_hsic_self:.4f}")
        print(f"  Phase 1 converged: {cal_result.phase1_converged}")
        print(f"  Phase 2 converged: {cal_result.phase2_converged}")
    
    # =========================================================================
    # STAGE 1: CAUSAL INITIALIZATION
    # =========================================================================
    if use_causal_init and resume_ckpt is None:
        print("\n" + "="*70)
        print("STAGED TRAINING: STAGE 1 - CAUSAL INITIALIZATION")
        print("="*70)
        
        # If calibration ran, config already has suggested lambda values
        # Pass multiplier=1.0 (or None to use default) since values are already calibrated
        # If calibration didn't run, pass None to use causal_init's default multiplier
        hsic_cross_mult_for_init = 1.0 if suggested_lambda_hsic_cross is not None else None
        hsic_self_mult_for_init = 1.0 if suggested_lambda_hsic_self is not None else None
        
        init_ckpt = run_causal_initialization(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            starting_checkpoint=starting_checkpoint,  # From calibration OR None
            hsic_cross_multiplier=hsic_cross_mult_for_init,  # 1.0 if calibrated, None otherwise
            hsic_self_multiplier=hsic_self_mult_for_init,    # 1.0 if calibrated, None otherwise
            seed=seed,
        )
        
        config["staged_training"]["causal_init_checkpoint"] = init_ckpt
        starting_checkpoint = init_ckpt
        
        staged_summary["stages_completed"].append("causal_init")
        staged_summary["causal_init_checkpoint"] = init_ckpt
        
        print("Causal initialization complete")
    
    # =========================================================================
    # STAGE 2: MAIN TRAINING
    # =========================================================================
    print("\n" + "="*70)
    print("STAGED TRAINING: STAGE 2 - MAIN TRAINING")
    print("="*70)
    
    # Apply group L1 if calibrated (or manually specified)
    lambda_group = staged_config.get("lambda_group_l1", None)
    if lambda_group is not None:
        config["training"]["lambda_group_l1"] = lambda_group
        print(f"Using λ_group = {lambda_group:.2e}")
    
    # Configure HSIC annealing for smooth transition from causal init
    if use_causal_init:
        # Start with high HSIC (from causal init) and anneal down
        hsic_mult = staged_config.get("causal_init_hsic_multiplier", 10.0)
        base_hsic = config["training"].get("lambda_hsic_cross",
                        config["training"].get("lambda_hsic", 0.1))
        
        config["training"]["use_hsic_annealing"] = True
        config["training"]["hsic_lambda_start"] = base_hsic * hsic_mult
        config["training"]["hsic_lambda_end"] = base_hsic
        
        # Set annealing epochs if not specified
        if "hsic_anneal_epochs" not in config["training"] or config["training"]["hsic_anneal_epochs"] is None:
            config["training"]["hsic_anneal_epochs"] = int(config["training"]["max_epochs"] * 0.5)
        
        print(f"HSIC annealing: {config['training']['hsic_lambda_start']:.4f} → {config['training']['hsic_lambda_end']:.4f}")
    
    # Run standard training
    # Pass the checkpoint from staged training (if any) as resume_ckpt
    df_metrics = trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
        resume_ckpt=starting_checkpoint,  # From causal_init OR calibration OR None
        plot_pred_check=plot_pred_check,
        debug=debug,
        best=best,
    )
    
    staged_summary["stages_completed"].append("main_training")
    staged_summary["final_checkpoint"] = str(Path(save_dir) / "k_0" / "checkpoints" / "last.ckpt")
    
    # =========================================================================
    # SAVE STAGED TRAINING SUMMARY
    # =========================================================================
    summary_path = Path(save_dir) / "staged_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(staged_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("STAGED TRAINING COMPLETE")
    print("="*70)
    print(f"Stages completed: {' → '.join(staged_summary['stages_completed'])}")
    print(f"Summary saved to: {summary_path}")
    
    return df_metrics


def run_staged_training_from_config(
    config_path: str,
    data_dir: str,
    save_dir: str,
    cluster: bool = False,
    experiment_tag: str = "NA",
) -> pd.DataFrame:
    """
    Convenience function to run staged training from a config file path.
    
    Args:
        config_path: Path to YAML config file
        data_dir: Path to data directory
        save_dir: Path to save outputs
        cluster: Whether running on cluster
        experiment_tag: Experiment tag
        
    Returns:
        DataFrame with training metrics
    """
    config = OmegaConf.load(config_path)
    
    return staged_trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
    )


def check_staged_training_config(config: dict) -> dict:
    """
    Validate and provide defaults for staged training configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict with validation results and warnings
    """
    result = {
        "valid": True,
        "warnings": [],
        "info": [],
    }
    
    staged = config.get("staged_training", {})
    training = config.get("training", {})
    
    # Check if calibration is enabled but HSIC is disabled
    use_calibration = staged.get("use_calibration", False)
    use_causal_init = staged.get("use_causal_init", False)
    lambda_hsic = training.get("lambda_hsic_cross", training.get("lambda_hsic", 0))
    
    if use_calibration and lambda_hsic == 0:
        result["warnings"].append(
            "Calibration is enabled but λ_hsic = 0. Calibration requires HSIC to be active."
        )
    
    if use_causal_init and lambda_hsic == 0:
        result["warnings"].append(
            "Causal init is enabled but λ_hsic = 0. Causal init requires HSIC to be active."
        )
    
    # Check calibration parameters
    if use_calibration:
        cal_epochs = staged.get("calibration_epochs", 10)
        if cal_epochs < 5:
            result["warnings"].append(
                f"Calibration epochs ({cal_epochs}) may be too few for reliable gradient estimation."
            )
        result["info"].append(f"Calibration: {cal_epochs} epochs")
    
    # Check causal init parameters
    if use_causal_init:
        init_epochs = staged.get("causal_init_epochs", 20)
        hsic_mult = staged.get("causal_init_hsic_multiplier", 10.0)
        if hsic_mult < 2:
            result["warnings"].append(
                f"HSIC multiplier ({hsic_mult}) may be too low for effective causal initialization."
            )
        result["info"].append(f"Causal init: {init_epochs} epochs, HSIC multiplier = {hsic_mult}")
    
    # Check HSIC annealing if using causal init
    if use_causal_init and not training.get("use_hsic_annealing", False):
        result["info"].append(
            "HSIC annealing will be auto-enabled for smooth transition from causal init."
        )
    
    return result
