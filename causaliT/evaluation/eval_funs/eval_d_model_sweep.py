"""
Evaluation function for d_model × seed sweeps.

This module provides functions for analyzing sweeps where model complexity (d_model)
and training seed are varied to study the relationship between:
- Model complexity (d_model, trainable parameters)
- Causality metrics (Soft Hamming Distance, DAG confidence)
- Independence metrics (HSIC)
- Prediction performance (test loss, R², MAE)

The main use case is to identify a minimum model complexity threshold above which
HSIC can serve as a proxy for correct causal structure learning.

Example:
    >>> from causaliT.evaluation.eval_funs.eval_d_model_sweep import eval_d_model_sweep
    >>> df = eval_d_model_sweep("experiments/noise_aware_single/scm1/euler/sweep_d_model_60863422")
    >>> # Load CSV into notebook for plotting and aggregation
    >>> import pandas as pd
    >>> df = pd.read_csv("experiments/.../eval/eval_d_model_sweep/files/sweep_summary.csv")
"""

import re
import json
from os.path import join, exists, isdir
from os import makedirs, listdir
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Import shared utilities
from .eval_utils import (
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_run_folder_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse sweep run folder name to extract d_model and seed.
    
    Expected format: sweep_d_model_combo_d_model_set_{d_model}_seed_{seed}
    """
    pattern = re.compile(r'd_model_set_(\d+)_seed_(\d+)')
    match = pattern.search(folder_name)
    
    if match:
        return {
            'd_model': int(match.group(1)),
            'seed': int(match.group(2)),
        }
    
    return {'d_model': None, 'seed': None}


def _load_kfold_summary(run_path: str) -> Optional[Dict[str, Any]]:
    """Load kfold_summary.json from a run folder."""
    kfold_path = join(run_path, "kfold_summary.json")
    
    if not exists(kfold_path):
        return None
    
    try:
        with open(kfold_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {kfold_path}: {e}")
        return None


def _load_dag_metrics(run_path: str) -> Optional[Dict[str, Any]]:
    """Load dag_metrics.json from eval_attention_scores."""
    dag_path = join(run_path, "eval", "eval_attention_scores", "files", "dag_metrics.json")
    
    if not exists(dag_path):
        return None
    
    try:
        with open(dag_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {dag_path}: {e}")
        return None


def _load_ate_metrics(run_path: str) -> Optional[Dict[str, Any]]:
    """Load ate_metrics.json from eval_interventions."""
    ate_path = join(run_path, "eval", "eval_do", "default", "files", "ate_metrics.json")
    
    if not exists(ate_path):
        return None
    
    try:
        with open(ate_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {ate_path}: {e}")
        return None


def _load_training_hsic_per_fold(run_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load training metrics.csv from all k-folds and compute HSIC statistics over epochs.
    
    Returns:
        Dict mapping fold_name -> {
            'train_hsic_mean_over_epochs': X,
            'train_hsic_max_over_epochs': X,
            'train_hsic_min_over_epochs': X,
            'train_hsic_final': X,
            'val_hsic_mean_over_epochs': X,
            ...
        }
    """
    result = {}
    
    kfold_dirs = sorted([
        d for d in listdir(run_path)
        if isdir(join(run_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        return result
    
    for kfold_dir in kfold_dirs:
        metrics_path = join(run_path, kfold_dir, 'logs', 'csv', 'version_0', 'metrics.csv')
        
        if not exists(metrics_path):
            continue
        
        try:
            df = pd.read_csv(metrics_path)
            fold_stats = {}
            
            if 'train_hsic' in df.columns:
                train_hsic = df['train_hsic'].dropna().values
                if len(train_hsic) > 0:
                    fold_stats['train_hsic_mean_over_epochs'] = float(np.mean(train_hsic))
                    fold_stats['train_hsic_max_over_epochs'] = float(np.max(train_hsic))
                    fold_stats['train_hsic_min_over_epochs'] = float(np.min(train_hsic))
                    fold_stats['train_hsic_final'] = float(train_hsic[-1])
            
            if 'val_hsic' in df.columns:
                val_hsic = df['val_hsic'].dropna().values
                if len(val_hsic) > 0:
                    fold_stats['val_hsic_mean_over_epochs'] = float(np.mean(val_hsic))
                    fold_stats['val_hsic_max_over_epochs'] = float(np.max(val_hsic))
                    fold_stats['val_hsic_min_over_epochs'] = float(np.min(val_hsic))
                    fold_stats['val_hsic_final'] = float(val_hsic[-1])
            
            result[kfold_dir] = fold_stats
                    
        except Exception as e:
            print(f"Warning: Could not load metrics from {metrics_path}: {e}")
            continue
    
    return result


def _parse_tensor_string(value):
    """Parse tensor(X.XX) string to float."""
    if not isinstance(value, str):
        return value
    match = re.match(r'^tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)$', value.strip())
    if match:
        return float(match.group(1))
    return value


# =============================================================================
# Main Evaluation Function
# =============================================================================

def eval_d_model_sweep(
    sweep_experiment: str,
    output_csv: str = "sweep_summary.csv",
    show_plots:str = None                       # to keep signature, not used 
) -> pd.DataFrame:
    """
    Evaluate a d_model × seed sweep experiment.
    
    Extracts per-fold metrics from all runs in the sweep. NO aggregation is performed;
    the user handles aggregation in their notebook.
    
    Args:
        sweep_experiment: Path to the sweep experiment folder
                         (containing sweeper/runs/combinations/)
        output_csv: Name of output CSV file (saved in eval/eval_d_model_sweep/files/)
        
    Returns:
        pd.DataFrame: Raw per-fold data with one row per (d_model, seed, fold) combination
        
    Output Files:
        - files/{output_csv}: Main CSV with all per-fold metrics
        - files/sweep_labels.json: Column descriptions
        
    CSV Columns:
        Identification:
            - run_name: Folder name of the run
            - d_model: Model dimensionality
            - seed: Training seed
            - fold: K-fold index (k_0, k_1, ...)
            
        Causality Metrics (per fold):
            - shd_cross: Soft Hamming Distance (S→X)
            - shd_self: Soft Hamming Distance (X→X)
            
        HSIC (per fold):
            - test_hsic: HSIC at end of training (test set)
            - val_hsic: HSIC at end of training (val set)
            - train_hsic_mean_over_epochs: Average HSIC over all training epochs
            - train_hsic_max_over_epochs: Maximum HSIC during training
            - train_hsic_min_over_epochs: Minimum HSIC during training
            - train_hsic_final: HSIC at last training epoch
            - val_hsic_mean_over_epochs, val_hsic_max_over_epochs, etc.
            
        Test Performance (per fold):
            - test_loss: Test loss (NLL for NoiseAware)
            - test_r2: Test R²
            - test_mae: Test MAE
            - test_rmse: Test RMSE
            
    Example:
        >>> from causaliT.evaluation.eval_funs import eval_d_model_sweep
        >>> df = eval_d_model_sweep("experiments/noise_aware_single/scm1/euler/sweep_d_model_60863422")
        >>> 
        >>> # Aggregate in notebook
        >>> df_agg = df.groupby(['d_model', 'seed']).agg({
        ...     'shd_cross': ['mean', 'std'],
        ...     'test_hsic': ['mean', 'std'],
        ... }).reset_index()
    """
    # Validate experiment path
    combinations_dir = join(sweep_experiment, "sweeper", "runs", "combinations")
    
    if not exists(combinations_dir):
        raise FileNotFoundError(
            f"No sweep runs found. Expected: {combinations_dir}\n"
            f"Make sure this is a sweep experiment with sweeper/runs/combinations/"
        )
    
    # Extract experiment ID from path
    match = re.search(r'([^/\\]+)$', sweep_experiment)
    exp_id = match.group(1) if match else "unknown"
    
    # Setup evaluation directories
    eval_path_root = join(sweep_experiment, "eval", "eval_d_model_sweep")
    eval_path_files = join(eval_path_root, "files")
    eval_path_cline = join(eval_path_root, "cline")
    
    makedirs(eval_path_files, exist_ok=True)
    makedirs(eval_path_cline, exist_ok=True)
    
    print(f"Sweep Experiment ID: {exp_id}")
    print(f"Combinations directory: {combinations_dir}")
    
    # =========================================================================
    # Save documentation files
    # =========================================================================
    
    sweep_labels = {
        "description": "d_model × seed sweep evaluation - per-fold raw data (no aggregation)",
        "purpose": "Identify minimum model complexity threshold where HSIC can proxy for correct causal structure",
        "columns": {
            "run_name": "Folder name of the sweep run",
            "d_model": "Model dimensionality (embedding/hidden dimension)",
            "seed": "Training seed (creates different DAG hypothesis)",
            "fold": "K-fold identifier (k_0, k_1, ...)",
            "shd_cross": "Soft Hamming Distance for S→X edges (lower is better)",
            "shd_self": "Soft Hamming Distance for X→X edges",
            "test_hsic": "HSIC on test set (lower = more independent residuals)",
            "val_hsic": "HSIC on validation set",
            "train_hsic_mean_over_epochs": "Average train HSIC over all training epochs",
            "train_hsic_max_over_epochs": "Maximum train HSIC during training",
            "train_hsic_min_over_epochs": "Minimum train HSIC during training",
            "train_hsic_final": "Train HSIC at last training epoch",
            "val_hsic_mean_over_epochs": "Average val HSIC over all training epochs",
            "val_hsic_max_over_epochs": "Maximum val HSIC during training",
            "val_hsic_min_over_epochs": "Minimum val HSIC during training",
            "val_hsic_final": "Val HSIC at last training epoch",
            "test_loss": "Test loss (NLL for NoiseAware)",
            "test_r2": "Test R²",
            "test_mae": "Test MAE",
            "test_rmse": "Test RMSE",
        },
        "note": "This file contains per-fold data. Aggregate in your notebook as needed.",
    }
    _save_variable_labels(eval_path_files, sweep_labels, "sweep_labels.json")
    
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, None,
        description="d_model × seed sweep evaluation - per-fold raw data for HSIC vs causality analysis",
        files_info={
            output_csv: "Per-fold metrics (one row per d_model × seed × fold)",
            "sweep_labels.json": "Column descriptions",
        },
        column_documentation=sweep_labels["columns"]
    )
    
    _create_cline_template(eval_path_cline, "eval_d_model_sweep", exp_id)
    
    # =========================================================================
    # Discover all sweep runs
    # =========================================================================
    
    print("\n--- Discovering sweep runs ---")
    
    run_folders = sorted([
        d for d in listdir(combinations_dir)
        if isdir(join(combinations_dir, d))
    ])
    
    print(f"Found {len(run_folders)} run folders")
    
    # =========================================================================
    # Extract metrics from each run
    # =========================================================================
    
    print("\n--- Extracting metrics from runs ---")
    
    all_records = []
    
    for run_folder in run_folders:
        run_path = join(combinations_dir, run_folder)
        
        # Parse d_model and seed from folder name
        parsed = _parse_run_folder_name(run_folder)
        d_model = parsed['d_model']
        seed = parsed['seed']
        
        if d_model is None:
            print(f"  ⊘ Skipping {run_folder}: Could not parse d_model/seed")
            continue
        
        # Load data sources
        kfold_data = _load_kfold_summary(run_path)
        dag_data = _load_dag_metrics(run_path)
        ate_data = _load_ate_metrics(run_path)
        hsic_training_data = _load_training_hsic_per_fold(run_path)
        
        if kfold_data is None:
            print(f"  ⊘ Skipping {run_folder}: No kfold_summary.json")
            continue
        
        fold_results = kfold_data.get("fold_results", {})
        
        # Process each fold
        for fold_id, fold_data in fold_results.items():
            fold_name = f"k_{fold_id}" if not fold_id.startswith("k_") else fold_id
            metrics = fold_data.get("metrics", {})
            
            record = {
                "run_name": run_folder,
                "d_model": d_model,
                "seed": seed,
                "fold": fold_name,
            }
            
            # Extract per-fold metrics from kfold_summary
            for key in ["test_hsic", "val_hsic", "test_loss", "val_loss", 
                        "test_nll", "val_nll", "test_x_r2", "val_x_r2",
                        "test_x_mae", "val_x_mae", "test_x_rmse", "val_x_rmse"]:
                if key in metrics:
                    val = _parse_tensor_string(metrics[key])
                    # Rename for clarity
                    out_key = key
                    if key == "test_x_r2":
                        out_key = "test_r2"
                    elif key == "val_x_r2":
                        out_key = "val_r2"
                    elif key == "test_x_mae":
                        out_key = "test_mae"
                    elif key == "val_x_mae":
                        out_key = "val_mae"
                    elif key == "test_x_rmse":
                        out_key = "test_rmse"
                    elif key == "val_x_rmse":
                        out_key = "val_rmse"
                    elif key == "test_nll":
                        out_key = "test_loss"  # Merge NLL into test_loss
                    elif key == "val_nll":
                        out_key = "val_loss"
                    record[out_key] = val
            
            # Extract per-fold SHD from dag_metrics
            if dag_data:
                for edge_type in ["cross", "self"]:
                    shd_key = f"soft_hamming_{edge_type}"
                    if shd_key in dag_data and isinstance(dag_data[shd_key], dict):
                        per_fold_shd = dag_data[shd_key].get("per_fold", {})
                        if fold_name in per_fold_shd:
                            record[f"shd_{edge_type}"] = per_fold_shd[fold_name]
            
            # Extract training HSIC statistics for this fold
            if fold_name in hsic_training_data:
                fold_hsic = hsic_training_data[fold_name]
                for hsic_key, hsic_val in fold_hsic.items():
                    record[hsic_key] = hsic_val
            
            all_records.append(record)
        
        print(f"  ✓ {run_folder}: d_model={d_model}, seed={seed}, {len(fold_results)} folds")
    
    if not all_records:
        raise ValueError("No valid runs found in the sweep")
    
    # =========================================================================
    # Create DataFrame and save
    # =========================================================================
    
    df = pd.DataFrame(all_records)
    df = df.sort_values(["d_model", "seed", "fold"]).reset_index(drop=True)
    
    # Save CSV
    df.to_csv(join(eval_path_files, output_csv), index=False)
    print(f"\n✓ Saved: {output_csv} ({len(df)} rows)")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("Sweep Evaluation Complete!")
    print(f"{'='*60}")
    print(f"  Total rows: {len(df)}")
    print(f"  d_model values: {sorted(df['d_model'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Folds per run: {df.groupby(['d_model', 'seed'])['fold'].count().unique()}")
    print(f"  Results saved to: {eval_path_files}")
    print(f"{'='*60}")
    
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate d_model × seed sweep for HSIC vs causality metrics"
    )
    parser.add_argument("sweep_experiment", help="Path to sweep experiment folder")
    parser.add_argument("--output-csv", default="sweep_summary.csv",
                       help="Output CSV filename (default: sweep_summary.csv)")
    
    args = parser.parse_args()
    
    eval_d_model_sweep(
        sweep_experiment=args.sweep_experiment,
        output_csv=args.output_csv,
    )
