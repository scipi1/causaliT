"""
Staged Training Orchestrator: Coordinates calibration -> causal_init -> score_cv -> training.

Pipeline stages (each independently toggleable):
- Stage 0: Calibration         (use_calibration)      → λ_group, λ_hsic
- Stage 1: Causal Init         (use_causal_init)      → structural checkpoint
- Stage 2: Score Sparsity CV   (use_score_sparsity_cv)→ λ_score
- Stage 3: Main Training       (always)               → final model

Pipeline combinations (all 8 combinations are valid):
- cal ON  + init ON  + cv ON  -> cal -> init -> cv -> train
- cal ON  + init ON  + cv OFF -> cal -> init -> train
- cal ON  + init OFF + cv ON  -> cal -> cv -> train
- cal ON  + init OFF + cv OFF -> cal -> train
- cal OFF + init ON  + cv ON  -> init -> cv -> train
- cal OFF + init ON  + cv OFF -> init -> train
- cal OFF + init OFF + cv ON  -> cv -> train
- cal OFF + init OFF + cv OFF -> train (standard)

All config changes are delegated to pure functions in config_operations.py.
"""

import json
import logging
from pathlib import Path

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
    Run the staged training pipeline (calibration -> causal_init -> score_cv -> training).

    Stage 0 -- Calibration (staged_training.use_calibration):
        Binary search for lambda_group s.t. ||grad_Recon|| ~ ||grad_HSIC||.

    Stage 1 -- Causal Init (staged_training.use_causal_init):
        Pre-trains with HSIC-dominated loss to break symmetry.

    Stage 2 -- Score Sparsity CV (staged_training.use_score_sparsity_cv):
        Selects optimal λ_score via k-fold cross-validation.
        Uses the same lambda for cross- and self-attention score sparsity.

    Stage 3 -- Main Training:
        Standard k-fold training. HSIC annealing auto-configured after init.

    Args:
        config:          Configuration dictionary.
        data_dir:        Path to data directory.
        save_dir:        Path to save all outputs.
        cluster:         Whether running on a compute cluster.
        experiment_tag:  Tag for experiment tracking.
        resume_ckpt:     Skip Stages 0-1 and resume from this checkpoint.
        plot_pred_check: Passed to trainer (currently unused).
        debug:           Enable debug mode.
        best:            Use best-checkpoint metrics instead of final epoch.

    Returns:
        pd.DataFrame with training metrics for each fold.
    """
    from causaliT.training.calibration import calibrate_group_l1
    from causaliT.training.causal_initialization import run_causal_initialization
    from causaliT.training.score_sparsity_cv import run_score_sparsity_cv
    from causaliT.training.config_operations import (
        apply_calibration_to_config,
        configure_main_training_from_staged,
    )
    from causaliT.training.trainer import trainer
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset

    staged_config = config.get("staged_training", {})
    use_calibration = staged_config.get("use_calibration", False)
    use_causal_init = staged_config.get("use_causal_init", False)
    use_score_sparsity_cv = staged_config.get("use_score_sparsity_cv", False)
    seed = config["training"].get("seed", 42)

    # Resume checkpoint skips Stages 0-1 and does a full Lightning resume
    # (model weights + optimizer state + epoch counter) — crash recovery.
    # Pipeline-internal stage transitions use warm-start (weights only).
    if resume_ckpt is not None:
        if not cluster:
            print(f"\nResume checkpoint provided: {resume_ckpt}")
            print("Skipping staged training (calibration/causal_init).")
            print("Using full Lightning resume (weights + optimizer + epoch).")
        starting_checkpoint = None   # no warm-start needed
        full_resume_ckpt = resume_ckpt  # full Lightning resume for main training
        use_calibration = False
        use_causal_init = False
    else:
        starting_checkpoint = None   # warm-start checkpoint from pipeline stages
        full_resume_ckpt = None      # no full resume

    # Populate sequence lengths once (all stages need this)
    config = populate_seq_lengths_from_dataset(config, data_dir)

    staged_summary = {
        "use_calibration": use_calibration,
        "use_causal_init": use_causal_init,
        "use_score_sparsity_cv": use_score_sparsity_cv,
        "resume_from": resume_ckpt,
        "stages_completed": [],
    }

    # =========================================================================
    # STAGE 0: CALIBRATION
    # =========================================================================
    if use_calibration:
        if not cluster:
            print("\n" + "=" * 70)
            print("STAGED TRAINING: STAGE 0 - CALIBRATION")
            print("=" * 70)

        cal_result = calibrate_group_l1(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            seed=seed,
        )

        config = apply_calibration_to_config(config, cal_result)
        starting_checkpoint = cal_result.checkpoint_path

        staged_summary.update({
            "lambda_group_optimal": float(cal_result.lambda_group_optimal),
            "lambda_hsic_cross_suggested": float(cal_result.lambda_hsic_cross_suggested),
            "lambda_hsic_self_suggested": float(cal_result.lambda_hsic_self_suggested),
            "base_ratio_cross": float(cal_result.base_ratio_cross) if cal_result.base_ratio_cross else None,
            "base_ratio_self": float(cal_result.base_ratio_self) if cal_result.base_ratio_self else None,
            "update_ratio_cross": float(cal_result.update_ratio_cross) if cal_result.update_ratio_cross else None,
            "update_ratio_self": float(cal_result.update_ratio_self) if cal_result.update_ratio_self else None,
            "phase1_converged": cal_result.phase1_converged,
            "phase2_converged": cal_result.phase2_converged,
            "calibration_converged": cal_result.converged,
            "calibration_checkpoint": cal_result.checkpoint_path,
        })
        staged_summary["stages_completed"].append("calibration")

        if not cluster:
            print(f"Calibration complete: lambda_group = {cal_result.lambda_group_optimal:.2e}")
            print(f"  lambda_hsic_cross = {cal_result.lambda_hsic_cross_suggested:.4f}")
            print(f"  lambda_hsic_self  = {cal_result.lambda_hsic_self_suggested:.4f}")

    # =========================================================================
    # STAGE 1: CAUSAL INITIALIZATION
    # =========================================================================
    if use_causal_init and resume_ckpt is None:
        n_seeds = staged_config.get("causal_init_n_seeds", 1)
        if not cluster:
            print("\n" + "=" * 70)
            print("STAGED TRAINING: STAGE 1 - CAUSAL INITIALIZATION")
            print(f"  (n_seeds={n_seeds})")
            print("=" * 70)

        # If calibration ran, config already has calibrated lambda_hsic values;
        # multipliers of 1.0 are correct. Otherwise keep None (uses defaults).
        hsic_cross_mult = 1.0 if use_calibration else None
        hsic_self_mult = 1.0 if use_calibration else None

        init_ckpt = run_causal_initialization(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            starting_checkpoint=starting_checkpoint,
            hsic_cross_multiplier=hsic_cross_mult,
            hsic_self_multiplier=hsic_self_mult,
            seed=seed,
            cluster=cluster,
        )

        config["staged_training"]["causal_init_checkpoint"] = init_ckpt
        starting_checkpoint = init_ckpt

        staged_summary["stages_completed"].append("causal_init")
        staged_summary["causal_init_checkpoint"] = init_ckpt

        if not cluster:
            print("Causal initialization complete")

    # =========================================================================
    # STAGE 2: SCORE SPARSITY CROSS-VALIDATION
    # =========================================================================
    if use_score_sparsity_cv:
        if not cluster:
            print("\n" + "=" * 70)
            print("STAGED TRAINING: STAGE 2 - SCORE SPARSITY CV")
            print("=" * 70)

        best_lambda_score = run_score_sparsity_cv(
            config=config,
            data_dir=data_dir,
            save_dir=save_dir,
            starting_checkpoint=starting_checkpoint,
            seed=seed,
            cluster=cluster,
        )

        # Store the suggested value in staged_training config
        # (configure_main_training_from_staged will propagate it)
        config["staged_training"]["lambda_score_suggested"] = best_lambda_score

        staged_summary["stages_completed"].append("score_sparsity_cv")
        staged_summary["best_lambda_score"] = best_lambda_score

        if not cluster:
            print(f"Score sparsity CV complete: λ_score = {best_lambda_score}")

    # =========================================================================
    # STAGE 3: MAIN TRAINING
    # =========================================================================
    if not cluster:
        print("\n" + "=" * 70)
        print("STAGED TRAINING: STAGE 3 - MAIN TRAINING")
        print("=" * 70)

    # Apply HSIC annealing wiring and lambda_group propagation
    config_main = configure_main_training_from_staged(config)

    if use_causal_init:
        anneal_epochs = config_main["training"].get("hsic_anneal_epochs")
        cross_start = config_main["training"].get("hsic_lambda_cross_start", 0)
        cross_end = config_main["training"].get("hsic_lambda_cross_end", 0)
        self_start = config_main["training"].get("hsic_lambda_self_start", 0)
        self_end = config_main["training"].get("hsic_lambda_self_end", 0)
        if not cluster:
            print(f"HSIC annealing over {anneal_epochs} epochs:")
            print(f"  Cross: {cross_start:.4f} -> {cross_end:.4f}")
            print(f"  Self:  {self_start:.4f} -> {self_end:.4f}")

    if use_score_sparsity_cv and not cluster:
        lambda_cs = config_main["training"].get("lambda_cross_score_sparse", 0)
        lambda_ss = config_main["training"].get("lambda_self_score_sparse", 0)
        print(f"Score sparsity: λ_cross={lambda_cs}, λ_self={lambda_ss}")

    df_metrics = trainer(
        config=config_main,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
        resume_ckpt=full_resume_ckpt,        # full Lightning resume (crash recovery)
        warm_start_ckpt=starting_checkpoint,  # weights-only (stage transitions)
        plot_pred_check=plot_pred_check,
        debug=debug,
        best=best,
    )

    staged_summary["stages_completed"].append("main_training")
    staged_summary["final_checkpoint"] = str(
        Path(save_dir) / "k_0" / "checkpoints" / "last.ckpt"
    )

    # Save summary
    summary_path = Path(save_dir) / "staged_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(staged_summary, f, indent=2)

    if not cluster:
        print("\n" + "=" * 70)
        print("STAGED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Stages completed: {' -> '.join(staged_summary['stages_completed'])}")
        print(f"Summary saved to: {summary_path}")

    return df_metrics


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def run_staged_training_from_config(
    config_path: str,
    data_dir: str,
    save_dir: str,
    cluster: bool = False,
    experiment_tag: str = "NA",
) -> pd.DataFrame:
    """Run staged training directly from a YAML config path."""
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
    Validate staged training configuration and return warnings/info.

    Returns:
        Dict with ``valid``, ``warnings``, and ``info`` keys.
    """
    result = {"valid": True, "warnings": [], "info": []}
    staged = config.get("staged_training", {})
    training = config.get("training", {})

    use_calibration = staged.get("use_calibration", False)
    use_causal_init = staged.get("use_causal_init", False)
    use_score_cv = staged.get("use_score_sparsity_cv", False)
    lambda_hsic = training.get("lambda_hsic_cross", training.get("lambda_hsic", 0))

    if use_calibration and lambda_hsic == 0:
        result["warnings"].append(
            "Calibration enabled but lambda_hsic = 0. HSIC must be active for calibration."
        )
    if use_causal_init and lambda_hsic == 0:
        result["warnings"].append(
            "Causal init enabled but lambda_hsic = 0. HSIC must be active for causal init."
        )
    if use_calibration:
        cal_epochs = staged.get("calibration_epochs", 10)
        if cal_epochs < 5:
            result["warnings"].append(
                f"calibration_epochs={cal_epochs} may be too few for reliable gradient estimation."
            )
        result["info"].append(f"Calibration: {cal_epochs} epochs")
    if use_causal_init:
        init_epochs = staged.get("causal_init_epochs", 20)
        hsic_mult = staged.get("causal_init_hsic_multiplier", 10.0)
        if hsic_mult < 2:
            result["warnings"].append(
                f"causal_init_hsic_multiplier={hsic_mult} may be too low for effective init."
            )
        result["info"].append(
            f"Causal init: {init_epochs} epochs, HSIC multiplier = {hsic_mult}"
        )
    if use_causal_init and not training.get("use_hsic_annealing", False):
        result["info"].append(
            "HSIC annealing will be auto-enabled for smooth transition from causal init."
        )
    if use_score_cv:
        cv_folds = staged.get("score_sparsity_cv_folds", 5)
        cv_epochs = staged.get("score_sparsity_cv_epochs", 20)
        candidates = staged.get(
            "score_sparsity_lambda_candidates",
            [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
        )
        n_candidates = len(candidates)
        total_runs = n_candidates * cv_folds
        result["info"].append(
            f"Score sparsity CV: {n_candidates} lambdas × {cv_folds} folds "
            f"= {total_runs} runs, {cv_epochs} epochs each"
        )
        if cv_epochs < 5:
            result["warnings"].append(
                f"score_sparsity_cv_epochs={cv_epochs} may be too few for "
                "reliable score sparsity selection."
            )
        if n_candidates < 2:
            result["warnings"].append(
                f"score_sparsity_lambda_candidates has only {n_candidates} "
                "value(s). Need at least 2 for meaningful cross-validation."
            )

    return result
