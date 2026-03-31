"""
Evaluation function for seed sweep experiments (paper reporting).

This module provides functions for aggregating statistics across multiple seeds
for paper tables. It extracts:
- ATE (Average Treatment Effect) errors per intervention
- Test loss and prediction metrics
- DAG recovery metrics (SHD, MEC distance)

Example:
    >>> from causaliT.evaluation.eval_funs import eval_seed_sweep
    >>> df = eval_seed_sweep("experiments/baseline/euler/vanilla_transformer_scm1_61555008")
    >>> # Results saved to eval/eval_seed_sweep/files/
"""

import re
import json
from os.path import join, exists, isdir
from os import makedirs, listdir
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import shared utilities
from .eval_utils import (
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    DEFAULT_PLOT_FORMAT,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_seed_folder_name(folder_name: str) -> Optional[int]:
    """
    Parse seed from sweep run folder name.
    
    Expected format: *_seed_{seed} or *_seed_{seed}_*
    """
    pattern = re.compile(r'seed_(\d+)')
    match = pattern.search(folder_name)
    
    if match:
        return int(match.group(1))
    return None


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


def _load_ate_metrics(run_path: str) -> Optional[Dict[str, Any]]:
    """Load ate_metrics.json from eval_ate folder."""
    ate_path = join(run_path, "eval", "eval_ate", "files", "ate_metrics.json")
    
    if not exists(ate_path):
        return None
    
    try:
        with open(ate_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {ate_path}: {e}")
        return None


def _load_dag_metrics(run_path: str) -> Optional[Dict[str, Any]]:
    """Load dag_metrics.json from eval_attention_scores folder."""
    dag_path = join(run_path, "eval", "eval_attention_scores", "files", "dag_metrics.json")
    
    if not exists(dag_path):
        return None
    
    try:
        with open(dag_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {dag_path}: {e}")
        return None


def _aggregate_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def _extract_intervention_ate_errors(
    ate_data: Dict[str, Any]
) -> Dict[str, List[Dict[str, float]]]:
    """
    Extract ATE absolute errors grouped by intervention from ate_metrics.json.
    
    Returns:
        Dict mapping intervention label (e.g., "S1=0.5") to list of per-variable errors
    """
    per_interv = ate_data.get("per_intervention_variable", [])
    
    grouped = {}
    for entry in per_interv:
        intervention = entry.get("intervention", "unknown")
        abs_error = entry.get("abs_error_mean")
        variable = entry.get("variable", "unknown")
        
        if intervention not in grouped:
            grouped[intervention] = []
        
        if abs_error is not None:
            grouped[intervention].append({
                "variable": variable,
                "abs_error": abs_error,
                "true_ate": entry.get("true_ate"),
                "model_ate": entry.get("model_ate_raw_mean"),
            })
    
    return grouped


# =============================================================================
# Main Evaluation Function
# =============================================================================

def eval_seed_sweep(
    sweep_experiment: str,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a seed sweep experiment and aggregate statistics for paper reporting.
    
    This function collects metrics from multiple seed runs and computes mean ± std
    statistics suitable for paper tables.
    
    Args:
        sweep_experiment: Path to sweep experiment folder containing
                         sweeper/runs/combinations/
        show_plots: If True, display summary plots
        
    Returns:
        pd.DataFrame: Summary statistics with mean and std columns
        
    Output Files:
        - eval/eval_seed_sweep/files/summary_stats.csv
            Summary table with mean ± std for all metrics (one row per metric)
        - eval/eval_seed_sweep/files/ate_by_intervention.csv
            ATE errors broken down by intervention (for paper tables)
        - eval/eval_seed_sweep/files/raw_per_seed.csv
            Raw per-seed data for custom analysis
        - eval/eval_seed_sweep/fig/ate_by_intervention.png
            Bar chart of ATE errors by intervention
            
    Metrics Aggregated:
        1. Test Performance: test_loss, test_mae, test_r2, test_rmse
        2. ATE Errors: Overall and per-intervention (S1, S2, S3, S5)
        3. DAG Recovery: SHD_cross, SHD_self, MEC_distance, MEC_membership_rate
        
    Example:
        >>> from causaliT.evaluation.eval_funs import eval_seed_sweep
        >>> df = eval_seed_sweep("experiments/baseline/euler/vanilla_transformer_scm1_61555008")
        >>> print(df)  # Summary stats with mean, std columns
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
    eval_path_root = join(sweep_experiment, "eval", "eval_seed_sweep")
    eval_path_fig = join(eval_path_root, "fig")
    eval_path_files = join(eval_path_root, "files")
    eval_path_cline = join(eval_path_root, "cline")
    
    makedirs(eval_path_fig, exist_ok=True)
    makedirs(eval_path_files, exist_ok=True)
    makedirs(eval_path_cline, exist_ok=True)
    
    print(f"Seed Sweep Experiment ID: {exp_id}")
    print(f"Combinations directory: {combinations_dir}")
    
    # =========================================================================
    # Save documentation files
    # =========================================================================
    
    sweep_labels = {
        "description": "Seed sweep evaluation for paper reporting",
        "purpose": "Aggregate statistics across multiple training seeds to report mean ± std",
        "columns": {
            "seed": "Training seed",
            "test_loss": "Test loss (MSE or NLL)",
            "test_mae": "Test Mean Absolute Error",
            "test_r2": "Test R² score",
            "test_rmse": "Test Root Mean Squared Error",
            "ate_overall_mae": "Mean ATE absolute error across all interventions",
            "shd_cross": "Soft Hamming Distance for S→X edges (lower is better)",
            "shd_self": "Soft Hamming Distance for X→X edges (lower is better)",
            "mec_distance": "Markov Equivalence Class distance (0 = same MEC)",
            "mec_in_class": "Whether learned DAG is in true MEC (0 or 1)",
        },
        "ate_interventions": {
            "S1": "Negative control (dangling, no children) - should have zero effect",
            "S2": "Positive control (one-to-one → X1) - tests simple causal learning",
            "S3": "Structure test (one-to-many → X2, X3) - tests one-to-many learning",
            "S5": "Confounding test (many-to-one → X4) - tests confounded parent learning",
        },
    }
    _save_variable_labels(eval_path_files, sweep_labels, "sweep_labels.json")
    
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Seed sweep evaluation - aggregated statistics for paper reporting",
        files_info={
            "summary_stats.csv": "Summary table with mean ± std for all metrics",
            "ate_by_intervention.csv": "ATE errors broken down by intervention",
            "raw_per_seed.csv": "Raw per-seed data for custom analysis",
        },
        column_documentation=sweep_labels["columns"]
    )
    
    _create_cline_template(eval_path_cline, "eval_seed_sweep", exp_id)
    
    # =========================================================================
    # Discover all seed runs
    # =========================================================================
    
    print("\n--- Discovering seed runs ---")
    
    run_folders = sorted([
        d for d in listdir(combinations_dir)
        if isdir(join(combinations_dir, d))
    ])
    
    print(f"Found {len(run_folders)} run folders")
    
    # =========================================================================
    # Extract metrics from each seed
    # =========================================================================
    
    print("\n--- Extracting metrics from seeds ---")
    
    all_records = []
    ate_by_intervention_records = []
    
    for run_folder in run_folders:
        run_path = join(combinations_dir, run_folder)
        
        # Parse seed from folder name
        seed = _parse_seed_folder_name(run_folder)
        
        if seed is None:
            print(f"  ⊘ Skipping {run_folder}: Could not parse seed")
            continue
        
        # Load data sources
        kfold_data = _load_kfold_summary(run_path)
        ate_data = _load_ate_metrics(run_path)
        dag_data = _load_dag_metrics(run_path)
        
        record = {
            "run_name": run_folder,
            "seed": seed,
        }
        
        # ---------------------------------------------------------------------
        # Extract test metrics from kfold_summary
        # ---------------------------------------------------------------------
        if kfold_data:
            stats = kfold_data.get("statistics", {})
            
            # Test loss
            if "test_loss" in stats:
                record["test_loss"] = stats["test_loss"].get("mean")
            elif "test_loss_x" in stats:
                record["test_loss"] = stats["test_loss_x"].get("mean")
            
            # Test MAE
            if "test_x_mae" in stats:
                record["test_mae"] = stats["test_x_mae"].get("mean")
            
            # Test R²
            if "test_x_r2" in stats:
                record["test_r2"] = stats["test_x_r2"].get("mean")
            
            # Test RMSE
            if "test_x_rmse" in stats:
                record["test_rmse"] = stats["test_x_rmse"].get("mean")
            
            # HSIC metrics
            if "test_hsic_cross" in stats:
                record["test_hsic_cross"] = stats["test_hsic_cross"].get("mean")
            if "test_hsic_self" in stats:
                record["test_hsic_self"] = stats["test_hsic_self"].get("mean")
        
        # ---------------------------------------------------------------------
        # Extract ATE metrics
        # ---------------------------------------------------------------------
        if ate_data:
            summary = ate_data.get("summary", {})
            record["ate_overall_mae"] = summary.get("mean_absolute_error")
            record["ate_overall_median"] = summary.get("median_absolute_error")
            
            # Extract per-intervention errors
            interv_errors = _extract_intervention_ate_errors(ate_data)
            
            for intervention, var_errors in interv_errors.items():
                # Aggregate across variables for this intervention
                abs_errors = [e["abs_error"] for e in var_errors if e["abs_error"] is not None]
                
                if abs_errors:
                    interv_mean = np.mean(abs_errors)
                    interv_key = f"ate_{intervention.replace('=', '_').replace('.', 'p').replace('-', 'n')}"
                    record[interv_key] = interv_mean
                    
                    # Also store detailed per-variable records
                    for var_entry in var_errors:
                        ate_by_intervention_records.append({
                            "seed": seed,
                            "intervention": intervention,
                            "variable": var_entry["variable"],
                            "abs_error": var_entry["abs_error"],
                            "true_ate": var_entry["true_ate"],
                            "model_ate": var_entry["model_ate"],
                        })
        
        # ---------------------------------------------------------------------
        # Extract DAG recovery metrics
        # ---------------------------------------------------------------------
        if dag_data:
            # Soft Hamming Distance
            if "soft_hamming_cross" in dag_data:
                shd_cross = dag_data["soft_hamming_cross"]
                record["shd_cross"] = shd_cross.get("mean")
            
            if "soft_hamming_self" in dag_data:
                shd_self = dag_data["soft_hamming_self"]
                record["shd_self"] = shd_self.get("mean")
            
            # MEC distance
            if "mec_distance" in dag_data:
                mec = dag_data["mec_distance"]
                record["mec_distance"] = mec.get("mean")
            
            # MEC membership rate
            if "mec_membership_rate" in dag_data:
                record["mec_in_class"] = dag_data["mec_membership_rate"]
            
            # DAG confidence
            if "dag_confidence_cross" in dag_data:
                record["dag_confidence_cross"] = dag_data["dag_confidence_cross"]
            if "dag_confidence_self" in dag_data:
                record["dag_confidence_self"] = dag_data["dag_confidence_self"]
        
        all_records.append(record)
        print(f"  ✓ {run_folder}: seed={seed}")
    
    if not all_records:
        raise ValueError("No valid seed runs found in the sweep")
    
    # =========================================================================
    # Create DataFrames
    # =========================================================================
    
    df_raw = pd.DataFrame(all_records)
    df_raw = df_raw.sort_values("seed").reset_index(drop=True)
    
    df_ate_interv = pd.DataFrame(ate_by_intervention_records)
    if not df_ate_interv.empty:
        df_ate_interv = df_ate_interv.sort_values(["intervention", "seed", "variable"]).reset_index(drop=True)
    
    # =========================================================================
    # Compute summary statistics CSV
    # =========================================================================
    
    summary_records = []
    
    # Columns to aggregate (skip non-numeric)
    skip_cols = ["run_name", "seed"]
    metric_cols = [c for c in df_raw.columns if c not in skip_cols]
    
    for col in metric_cols:
        values = df_raw[col].dropna().tolist()
        if values:
            summary_records.append({
                "metric": col,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_seeds": len(values),
            })
    
    df_summary = pd.DataFrame(summary_records)
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    
    # Summary stats CSV
    df_summary.to_csv(join(eval_path_files, "summary_stats.csv"), index=False)
    print(f"\n✓ Saved: summary_stats.csv ({len(df_summary)} metrics)")
    
    # Raw per-seed CSV
    df_raw.to_csv(join(eval_path_files, "raw_per_seed.csv"), index=False)
    print(f"✓ Saved: raw_per_seed.csv ({len(df_raw)} seeds)")
    
    # ATE by intervention CSV
    if not df_ate_interv.empty:
        df_ate_interv.to_csv(join(eval_path_files, "ate_by_intervention.csv"), index=False)
        print(f"✓ Saved: ate_by_intervention.csv ({len(df_ate_interv)} rows)")
    
    # =========================================================================
    # Create plots
    # =========================================================================
    
    # Extract ATE by intervention stats for plotting
    ate_interv_cols = [c for c in df_raw.columns if c.startswith("ate_") and c not in ["ate_overall_mae", "ate_overall_median"]]
    if ate_interv_cols:
        ate_by_intervention = {}
        for col in ate_interv_cols:
            values = df_raw[col].dropna().tolist()
            label = col.replace("ate_", "").replace("_", "=").replace("p", ".").replace("n", "-")
            ate_by_intervention[label] = _aggregate_stats(values)
        
        _plot_ate_by_intervention(
            ate_by_intervention,
            save_path=join(eval_path_fig, f"ate_by_intervention_{exp_id}.{DEFAULT_PLOT_FORMAT}"),
            show_plot=show_plots,
        )
    
    # DAG metrics plot
    dag_metrics_cols = ["shd_cross", "shd_self", "mec_distance"]
    dag_metrics = {}
    for col in dag_metrics_cols:
        if col in df_raw.columns:
            values = df_raw[col].dropna().tolist()
            dag_metrics[col] = _aggregate_stats(values)
    
    if dag_metrics:
        _plot_dag_metrics(
            dag_metrics,
            save_path=join(eval_path_fig, f"dag_metrics_{exp_id}.{DEFAULT_PLOT_FORMAT}"),
            show_plot=show_plots,
        )
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    _print_summary(df_summary, len(df_raw), exp_id)
    
    return df_summary


# =============================================================================
# Plotting Functions
# =============================================================================

def _plot_ate_by_intervention(
    ate_by_intervention: Dict[str, Dict[str, float]],
    save_path: str,
    show_plot: bool = False,
) -> None:
    """Create bar chart of ATE errors by intervention with error bars."""
    
    # Sort interventions by source variable
    labels = sorted(ate_by_intervention.keys())
    means = [ate_by_intervention[k]["mean"] for k in labels]
    stds = [ate_by_intervention[k]["std"] for k in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    
    ax.set_xlabel("Intervention")
    ax.set_ylabel("Mean Absolute ATE Error")
    ax.set_title("ATE Error by Intervention (mean ± std across seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def _plot_dag_metrics(
    dag_metrics: Dict[str, Dict[str, float]],
    save_path: str,
    show_plot: bool = False,
) -> None:
    """Create bar chart of DAG recovery metrics with error bars."""
    
    # Filter to plottable metrics
    plot_metrics = ["shd_cross", "shd_self", "mec_distance"]
    labels = []
    means = []
    stds = []
    
    for metric in plot_metrics:
        if metric in dag_metrics and dag_metrics[metric]["mean"] is not None:
            labels.append(metric.replace("_", " ").title())
            means.append(dag_metrics[metric]["mean"])
            stds.append(dag_metrics[metric]["std"] if dag_metrics[metric]["std"] else 0)
    
    if not labels:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='coral', edgecolor='black')
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value (lower is better)")
    ax.set_title("DAG Recovery Metrics (mean ± std across seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Summary Printing
# =============================================================================

def _print_summary(df_summary: pd.DataFrame, n_seeds: int, exp_id: str) -> None:
    """Print a formatted summary of the seed sweep evaluation."""
    
    print(f"\n{'='*70}")
    print("SEED SWEEP EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment: {exp_id}")
    print(f"Number of seeds: {n_seeds}")
    
    print(f"\n--- Metrics Summary ---")
    for _, row in df_summary.iterrows():
        metric = row["metric"]
        mean = row["mean"]
        std = row["std"]
        print(f"  {metric}: {mean:.6f} ± {std:.6f}")
    
    print(f"{'='*70}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate seed sweep experiments for paper reporting"
    )
    parser.add_argument("sweep_experiment", help="Path to sweep experiment folder")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots interactively")
    
    args = parser.parse_args()
    
    eval_seed_sweep(
        sweep_experiment=args.sweep_experiment,
        show_plots=args.show_plots,
    )
