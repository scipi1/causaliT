"""
Simplified evaluation function for seed sweep experiments.

Computes mean and std over seeds for ATE metrics, test MAE, SHD, and MEC distance.
Outputs two CSV files: one for per-intervention ATE metrics, one for experiment-wide metrics.

Example:
    >>> from causaliT.evaluation.eval_funs import eval_seed_sweep
    >>> df_ate, df_exp = eval_seed_sweep("experiments/baseline/euler/vanilla_transformer_scm1_61555008")
"""

import re
import json
from os.path import join, exists, isdir
from os import makedirs, listdir
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd


def _parse_seed_folder_name(folder_name: str) -> Optional[int]:
    """Parse seed from folder name (expected format: *_seed_{seed}*)."""
    match = re.search(r'seed_(\d+)', folder_name)
    return int(match.group(1)) if match else None


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found or error."""
    if not exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _parse_intervention_value(intervention: str) -> Optional[float]:
    """Parse intervention value from string (e.g., 'S2=-1.7' -> -1.7)."""
    match = re.search(r'=(-?[\d.]+)', intervention)
    return float(match.group(1)) if match else None


def _extract_fit_diagnostics(run_path: str) -> Optional[Dict[str, float]]:
    """
    Extract fit diagnostics (train_loss, generalization gap) from a seed run.
    
    Reads the training CSV log from k_0/logs/csv/version_0/metrics.csv,
    finds the epoch with best val_loss, and computes:
    - best_train_loss: train_loss at the best val_loss epoch
    - best_val_loss: best validation loss achieved
    - generalization_gap: val_loss - train_loss at best epoch
    - gap_ratio: (val_loss - train_loss) / train_loss
    
    Returns None if training CSV is not found or metrics are missing.
    """
    # Find the k-fold directory (typically k_0 for single-fold)
    kfold_dirs = sorted([
        d for d in listdir(run_path)
        if isdir(join(run_path, d)) and d.startswith("k_")
    ])
    
    if not kfold_dirs:
        return None
    
    # Use first fold
    metrics_path = join(run_path, kfold_dirs[0], "logs", "csv", "version_0", "metrics.csv")
    if not exists(metrics_path):
        return None
    
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    
    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        return None
    
    # Group by epoch, take first non-NaN for each metric
    if "epoch" in df.columns:
        df_epoch = df.groupby("epoch").first().reset_index()
    else:
        df_epoch = df
    
    # Drop rows where either is NaN
    valid = df_epoch.dropna(subset=["train_loss", "val_loss"])
    if len(valid) == 0:
        return None
    
    # Find best val_loss epoch
    best_idx = valid["val_loss"].idxmin()
    best_row = valid.loc[best_idx]
    
    best_train = float(best_row["train_loss"])
    best_val = float(best_row["val_loss"])
    gen_gap = best_val - best_train
    gap_ratio = gen_gap / best_train if best_train > 1e-12 else float('inf')
    
    return {
        "best_train_loss": best_train,
        "best_val_loss": best_val,
        "generalization_gap": gen_gap,
        "gap_ratio": gap_ratio,
    }


def eval_seed_sweep(sweep_experiment: str, show_plots: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a seed sweep experiment and compute mean ± std over seeds.
    
    Args:
        sweep_experiment: Path to sweep experiment folder
        
    Returns:
        Tuple of:
        - df_ate: ATE metrics per (intervention, variable) with mean/std over seeds
        - df_experiment: Experiment-wide metrics (test_mae, shd, mec) with mean/std over seeds
        
    Output Files:
        eval/eval_seed_sweep/files/ate_summary.csv
        eval/eval_seed_sweep/files/experiment_summary.csv
    """
    combinations_dir = join(sweep_experiment, "sweeper", "runs", "combinations")
    
    if not exists(combinations_dir):
        raise FileNotFoundError(f"No sweep runs found at: {combinations_dir}")
    
    # Setup output directory
    eval_path_files = join(sweep_experiment, "eval", "eval_seed_sweep", "files")
    makedirs(eval_path_files, exist_ok=True)
    
    # Discover seed folders
    run_folders = sorted([
        d for d in listdir(combinations_dir)
        if isdir(join(combinations_dir, d))
    ])
    
    # Collect data
    ate_records = []
    experiment_records = []
    
    for run_folder in run_folders:
        run_path = join(combinations_dir, run_folder)
        seed = _parse_seed_folder_name(run_folder)
        
        if seed is None:
            continue
        
        # Load data sources
        ate_data = _load_json(join(run_path, "eval", "eval_ate_mc", "files", "ate_metrics_mc.json"))
        kfold_data = _load_json(join(run_path, "kfold_summary.json"))
        dag_data = _load_json(join(run_path, "eval", "eval_attention_scores", "files", "dag_metrics.json"))
        
        # Extract experiment-wide metrics
        exp_record = {"seed": seed}
        
        if kfold_data:
            stats = kfold_data.get("statistics", {})
            if "test_x_mae" in stats:
                exp_record["test_mae"] = stats["test_x_mae"].get("mean")
            if "val_x_mae" in stats:
                exp_record["val_mae"] = stats["val_x_mae"].get("mean")
            if "val_loss_x" in stats:
                exp_record["val_loss"] = stats["val_loss_x"].get("mean")
        
        # Extract fit diagnostics from training CSV logs
        fit_diag = _extract_fit_diagnostics(run_path)
        if fit_diag:
            exp_record.update(fit_diag)
        
        if dag_data:
            if "soft_hamming_cross" in dag_data:
                exp_record["shd_cross"] = dag_data["soft_hamming_cross"].get("mean")
            if "mec_distance" in dag_data:
                exp_record["mec_distance"] = dag_data["mec_distance"].get("mean")
        
        experiment_records.append(exp_record)
        
        # Extract per-intervention-variable ATE metrics
        if ate_data:
            per_interv = ate_data.get("per_intervention_variable", [])
            for entry in per_interv:
                intervention = entry.get("intervention", "unknown")
                abs_error = entry.get("abs_error_mean")
                
                # Compute scaled_error = abs_error / |intervention_value|
                interv_value = _parse_intervention_value(intervention)
                scaled_error = None
                if abs_error is not None and interv_value is not None and interv_value != 0:
                    scaled_error = abs_error / abs(interv_value)
                
                ate_records.append({
                    "seed": seed,
                    "intervention": intervention,
                    "variable": entry.get("variable", "unknown"),
                    "true_ate": entry.get("true_ate"),
                    "model_ate": entry.get("model_ate_mean"),
                    "abs_error": abs_error,
                    "scaled_error": scaled_error,
                })
    
    if not ate_records:
        raise ValueError("No valid seed runs found")
    
    # Create DataFrames
    df_ate = pd.DataFrame(ate_records)
    df_exp = pd.DataFrame(experiment_records)
    
    # =========================================================================
    # Aggregate ATE metrics by (intervention, variable)
    # =========================================================================
    agg_ate = df_ate.groupby(["intervention", "variable"]).agg({
        "true_ate": "first",
        "model_ate": ["mean", "std"],
        "abs_error": ["mean", "std"],
        "scaled_error": ["mean", "std"],
        "seed": "count",
    })
    
    # Flatten column names
    agg_ate.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in agg_ate.columns
    ]
    agg_ate = agg_ate.rename(columns={"seed_count": "n_seeds", "true_ate_first": "true_ate"})
    agg_ate = agg_ate.reset_index()
    
    # Reorder columns
    col_order_ate = [
        "intervention", "variable", "true_ate",
        "model_ate_mean", "model_ate_std",
        "abs_error_mean", "abs_error_std",
        "scaled_error_mean", "scaled_error_std",
        "n_seeds",
    ]
    agg_ate = agg_ate[[c for c in col_order_ate if c in agg_ate.columns]]
    
    # =========================================================================
    # Aggregate experiment-wide metrics
    # =========================================================================
    exp_summary = {}
    for col in ["test_mae", "val_mae", "val_loss",
                "best_train_loss", "best_val_loss", "generalization_gap", "gap_ratio",
                "shd_cross", "mec_distance"]:
        if col in df_exp.columns:
            values = df_exp[col].dropna()
            if len(values) > 0:
                exp_summary[f"{col}_mean"] = values.mean()
                exp_summary[f"{col}_std"] = values.std()
    
    exp_summary["n_seeds"] = len(df_exp)
    df_exp_summary = pd.DataFrame([exp_summary])
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    ate_path = join(eval_path_files, "ate_summary.csv")
    exp_path = join(eval_path_files, "experiment_summary.csv")
    
    agg_ate.to_csv(ate_path, index=False)
    df_exp_summary.to_csv(exp_path, index=False)
    
    print(f"Saved: {ate_path}")
    print(f"Saved: {exp_path}")
    
    return agg_ate, df_exp_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate seed sweep (mean ± std over seeds)")
    parser.add_argument("sweep_experiment", help="Path to sweep experiment folder")
    args = parser.parse_args()
    
    df_ate, df_exp = eval_seed_sweep(args.sweep_experiment)
    print("\n--- ATE Summary ---")
    print(df_ate.to_string())
    print("\n--- Experiment Summary ---")
    print(df_exp.to_string())
