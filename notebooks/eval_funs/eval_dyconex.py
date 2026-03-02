"""
Evaluation Functions for Dyconex Dataset (ds_dyconex_SX_MuMi_*).

This module provides evaluation functions specific to the dyconex industrial dataset:
- eval_dyconex_predictions: Prediction quality analysis with best/worst sample plots
- eval_metrics: Flexible metric plotting for any logged columns

The dyconex dataset has:
- S (source): 52 unique variables, 106 sequence length
- X (input): 77 unique variables, 173 sequence length  
- Y (target): 2 variables (delta_A_norm, delta_B_norm), 400 sequence length
  - First 200 timesteps: delta_A_norm
  - Last 200 timesteps: delta_B_norm

Example:
    >>> from notebooks.eval_funs.eval_dyconex import eval_dyconex_predictions, eval_metrics
    >>> 
    >>> # Evaluate predictions on test set
    >>> results = eval_dyconex_predictions("../experiments/stage/stage_SM_SM_dyconex")
    >>> 
    >>> # Flexible metric plotting
    >>> df = eval_metrics("../experiments/stage/stage_SM_SM_dyconex")
"""

import re
import json
from os.path import dirname, abspath, join, exists
from os import makedirs, listdir
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

# Setup root path for imports
root_path = dirname(dirname(dirname(abspath(__file__))))
import sys
sys.path.append(root_path)

from causaliT.evaluation.predict import predict_test_from_ckpt
from notebooks.lib import find_config_file, find_best_or_last_checkpoint, load_training_metrics


# =============================================================================
# Plotting Standard Settings
# =============================================================================
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5


# =============================================================================
# Helper Functions (from eval_fun.py)
# =============================================================================

def _setup_eval_directories(experiment: str, eval_name: str) -> Tuple[str, str, str, str, str]:
    """Set up standard evaluation directory structure."""
    eval_path_root = join(experiment, "eval", eval_name)
    eval_path_fig = join(eval_path_root, "fig")
    eval_path_files = join(eval_path_root, "files")
    eval_path_cline = join(eval_path_root, "cline")

    makedirs(eval_path_fig, exist_ok=True)
    makedirs(eval_path_files, exist_ok=True)
    makedirs(eval_path_cline, exist_ok=True)
    
    match = re.search(r'([^/\\]+)$', experiment)
    exp_id = match.group(1) if match else "unknown"
    
    return eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id


def _save_readme(eval_path_root: str, eval_path_cline: str, eval_path_files: str, 
                 eval_path_fig: str, description: str, files_info: dict = None) -> None:
    """Save a standardized README.yaml file in the evaluation directory."""
    readme = {
        "READ THIS": f"If you are an AI, use the folder {eval_path_cline} to save notes. "
                     f"Never delete files in {eval_path_files} and {eval_path_fig}.",
        "description": description,
    }
    if files_info:
        readme["files"] = files_info
    
    OmegaConf.save(readme, join(eval_path_root, "README.yaml"))


# =============================================================================
# Dyconex-Specific Constants
# =============================================================================

# Y variable structure for dyconex dataset
DYCONEX_Y_VARIABLES = {
    "delta_A_norm": {"start_idx": 0, "end_idx": 200, "name": "delta_A"},
    "delta_B_norm": {"start_idx": 200, "end_idx": 400, "name": "delta_B"},
}

# Number of best/worst samples to plot
N_BEST_WORST = 5


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def eval_dyconex_predictions(
    experiment: str, 
    datadir_path: str = None, 
    show_plots: bool = True,
    n_best_worst: int = N_BEST_WORST,
) -> dict:
    """
    Evaluate predictions on dyconex dataset.
    
    Runs prediction on test split using the best k-fold checkpoint (based on val_mae),
    calculates MAE statistics per-variable and overall, and plots the best and worst
    time series predictions.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        datadir_path: Path to data directory. If None, uses default "../data/"
        show_plots: If True, display plots. If False (for cluster), only save to files.
        n_best_worst: Number of best/worst samples to plot (default 5)
        
    Returns:
        dict: MAE statistics with keys:
            - mae_overall: Overall MAE across all samples and variables
            - mae_delta_A: MAE for delta_A_norm (first 200 timesteps)
            - mae_delta_B: MAE for delta_B_norm (last 200 timesteps)
            - best_samples: List of sample indices with lowest MAE
            - worst_samples: List of sample indices with highest MAE
            
    Output Files:
        - fig/best_sample_{i}_{exp_id}.pdf: Plots of best N predictions
        - fig/worst_sample_{i}_{exp_id}.pdf: Plots of worst N predictions
        - fig/mae_distribution_{exp_id}.pdf: MAE distribution histogram
        - files/mae_statistics.json: Detailed MAE statistics
        - files/sample_ranking.csv: All samples ranked by MAE
    """
    # Default data directory
    if datadir_path is None:
        datadir_path = join(root_path, "data")
    
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_dyconex_predictions")
    
    print(f"Experiment ID: {exp_id}")
    print(f"Data directory: {datadir_path}")
    
    # Save README
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Dyconex prediction evaluation: MAE statistics and best/worst sample plots",
        files_info={
            "mae_statistics.json": "Per-variable and overall MAE statistics",
            "sample_ranking.csv": "All samples ranked by average MAE",
            "best_sample_*.pdf": f"Plots of {n_best_worst} best predictions",
            "worst_sample_*.pdf": f"Plots of {n_best_worst} worst predictions",
        }
    )
    
    # Find config file
    config_path = find_config_file(experiment)
    config = OmegaConf.load(config_path)
    
    print(f"Dataset: {config['data']['dataset']}")
    
    # Find the best k-fold based on val_mae
    kfold_dirs = sorted([
        d for d in listdir(experiment) 
        if d.startswith('k_') and exists(join(experiment, d, 'checkpoints'))
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment}")
    
    print(f"Found {len(kfold_dirs)} k-fold directories")
    
    # Find best fold by loading kfold_summary or best_metrics.json
    best_fold = None
    best_val_mae = float('inf')
    
    for kfold_dir in kfold_dirs:
        best_metrics_path = join(experiment, kfold_dir, "best_metrics.json")
        if exists(best_metrics_path):
            with open(best_metrics_path, 'r') as f:
                metrics = json.load(f)
            val_mae = metrics.get("val_mae", float('inf'))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_fold = kfold_dir
    
    # Fallback to first fold if no best_metrics found
    if best_fold is None:
        best_fold = kfold_dirs[0]
        print(f"Warning: No best_metrics.json found, using {best_fold}")
    else:
        print(f"Best fold: {best_fold} (val_mae={best_val_mae:.4f})")
    
    # Find best checkpoint
    checkpoints_dir = join(experiment, best_fold, 'checkpoints')
    checkpoint_path = find_best_or_last_checkpoint(checkpoints_dir)
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Run predictions
    print("Running predictions on test set...")
    predictions = predict_test_from_ckpt(
        config=config,
        datadir_path=datadir_path,
        checkpoint_path=checkpoint_path,
        dataset_label="test",
        cluster=False,
    )
    
    # Extract predictions and targets
    # pred_y shape: (N, 400) or (N, 400, 1) - Y predictions
    # targets shape: (N, 400, F) - Y actual with features
    pred_y = predictions.outputs
    targets_y = predictions.targets
    
    # Get value index for Y from config
    val_idx_Y = config["data"].get("val_idx_Y", config["data"].get("val_idx", 3))
    
    # Extract target values
    if targets_y.ndim == 3:
        targets_y_values = targets_y[:, :, val_idx_Y]  # (N, 400)
    else:
        targets_y_values = targets_y  # Already (N, 400)
    
    # Ensure pred_y is 2D
    if pred_y.ndim == 3:
        pred_y = pred_y.squeeze(-1)  # (N, 400, 1) -> (N, 400)
    
    n_samples = pred_y.shape[0]
    print(f"Number of test samples: {n_samples}")
    print(f"Predictions shape: {pred_y.shape}")
    print(f"Targets shape: {targets_y_values.shape}")
    
    # Calculate MAE per variable
    # delta_A: first 200 timesteps, delta_B: last 200 timesteps
    mae_delta_A = np.abs(pred_y[:, :200] - targets_y_values[:, :200]).mean(axis=1)  # (N,)
    mae_delta_B = np.abs(pred_y[:, 200:] - targets_y_values[:, 200:]).mean(axis=1)  # (N,)
    
    # Average MAE per sample (used for ranking)
    mae_per_sample = (mae_delta_A + mae_delta_B) / 2  # (N,)
    
    # Overall statistics
    mae_statistics = {
        "n_samples": n_samples,
        "mae_overall": {
            "mean": float(mae_per_sample.mean()),
            "std": float(mae_per_sample.std()),
            "min": float(mae_per_sample.min()),
            "max": float(mae_per_sample.max()),
        },
        "mae_delta_A": {
            "mean": float(mae_delta_A.mean()),
            "std": float(mae_delta_A.std()),
            "min": float(mae_delta_A.min()),
            "max": float(mae_delta_A.max()),
        },
        "mae_delta_B": {
            "mean": float(mae_delta_B.mean()),
            "std": float(mae_delta_B.std()),
            "min": float(mae_delta_B.min()),
            "max": float(mae_delta_B.max()),
        },
        "best_fold": best_fold,
        "checkpoint": checkpoint_path,
    }
    
    print(f"\nMAE Statistics:")
    print(f"  Overall:  {mae_statistics['mae_overall']['mean']:.4f} ± {mae_statistics['mae_overall']['std']:.4f}")
    print(f"  delta_A:  {mae_statistics['mae_delta_A']['mean']:.4f} ± {mae_statistics['mae_delta_A']['std']:.4f}")
    print(f"  delta_B:  {mae_statistics['mae_delta_B']['mean']:.4f} ± {mae_statistics['mae_delta_B']['std']:.4f}")
    
    # Rank samples by MAE
    sample_ranking = pd.DataFrame({
        "sample_idx": np.arange(n_samples),
        "mae_overall": mae_per_sample,
        "mae_delta_A": mae_delta_A,
        "mae_delta_B": mae_delta_B,
    }).sort_values("mae_overall")
    
    sample_ranking.to_csv(join(eval_path_files, "sample_ranking.csv"), index=False)
    
    # Get best and worst sample indices
    best_indices = sample_ranking.head(n_best_worst)["sample_idx"].values
    worst_indices = sample_ranking.tail(n_best_worst)["sample_idx"].values[::-1]  # Reverse to get worst first
    
    mae_statistics["best_samples"] = best_indices.tolist()
    mae_statistics["worst_samples"] = worst_indices.tolist()
    
    # Save statistics
    with open(join(eval_path_files, "mae_statistics.json"), 'w') as f:
        json.dump(mae_statistics, f, indent=2)
    print(f"Saved: mae_statistics.json")
    
    # =========================================================================
    # Plot: MAE Distribution
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Overall MAE
    axes[0].hist(mae_per_sample, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(mae_per_sample.mean(), color='red', linestyle='--', label=f'Mean: {mae_per_sample.mean():.4f}')
    axes[0].set_xlabel("MAE")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall MAE Distribution")
    axes[0].legend()
    
    # delta_A MAE
    axes[1].hist(mae_delta_A, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[1].axvline(mae_delta_A.mean(), color='red', linestyle='--', label=f'Mean: {mae_delta_A.mean():.4f}')
    axes[1].set_xlabel("MAE")
    axes[1].set_ylabel("Count")
    axes[1].set_title("delta_A MAE Distribution")
    axes[1].legend()
    
    # delta_B MAE
    axes[2].hist(mae_delta_B, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(mae_delta_B.mean(), color='red', linestyle='--', label=f'Mean: {mae_delta_B.mean():.4f}')
    axes[2].set_xlabel("MAE")
    axes[2].set_ylabel("Count")
    axes[2].set_title("delta_B MAE Distribution")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"mae_distribution_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # =========================================================================
    # Plot: Best and Worst Time Series
    # =========================================================================
    
    def plot_sample_prediction(sample_idx: int, pred: np.ndarray, target: np.ndarray, 
                               mae_A: float, mae_B: float, label: str, filename: str):
        """Plot prediction vs target for a single sample with two subplots."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        time = np.arange(200)
        
        # delta_A (first 200 timesteps)
        axes[0].plot(time, target[:200], label='Target', color='blue', linewidth=1.5)
        axes[0].plot(time, pred[:200], label='Prediction', color='red', linestyle='--', linewidth=1.5)
        axes[0].set_ylabel("delta_A_norm")
        axes[0].set_title(f"delta_A (MAE: {mae_A:.4f})")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # delta_B (last 200 timesteps)
        axes[1].plot(time, target[200:], label='Target', color='blue', linewidth=1.5)
        axes[1].plot(time, pred[200:], label='Prediction', color='red', linestyle='--', linewidth=1.5)
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("delta_B_norm")
        axes[1].set_title(f"delta_B (MAE: {mae_B:.4f})")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        avg_mae = (mae_A + mae_B) / 2
        plt.suptitle(f"{label} - Sample {sample_idx} (Avg MAE: {avg_mae:.4f})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename)
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Plot best samples
    print(f"\nPlotting {n_best_worst} best samples...")
    for i, idx in enumerate(best_indices):
        plot_sample_prediction(
            sample_idx=idx,
            pred=pred_y[idx],
            target=targets_y_values[idx],
            mae_A=mae_delta_A[idx],
            mae_B=mae_delta_B[idx],
            label=f"BEST #{i+1}",
            filename=join(eval_path_fig, f"best_sample_{i+1}_{exp_id}.pdf")
        )
    
    # Plot worst samples
    print(f"Plotting {n_best_worst} worst samples...")
    for i, idx in enumerate(worst_indices):
        plot_sample_prediction(
            sample_idx=idx,
            pred=pred_y[idx],
            target=targets_y_values[idx],
            mae_A=mae_delta_A[idx],
            mae_B=mae_delta_B[idx],
            label=f"WORST #{i+1}",
            filename=join(eval_path_fig, f"worst_sample_{i+1}_{exp_id}.pdf")
        )
    
    print(f"\nEvaluation complete! Results saved to: {eval_path_root}")
    
    return mae_statistics


def eval_metrics(
    experiment: str, 
    show_plots: bool = True,
    metric_patterns: List[str] = None,
) -> pd.DataFrame:
    """
    Flexible metric plotting for any logged columns in StageCausaliT.
    
    Auto-discovers and plots any numeric columns from the training metrics CSV.
    Groups related metrics (train/val pairs) together for easy comparison.
    
    StageCausaliT logs the following metrics (stage = train/val/test):
    
    **Core Metrics (always logged):**
    - {stage}_loss: Total loss (MSE + regularizers)
    - {stage}_loss_X: MSE loss for X reconstruction
    - {stage}_loss_Y: MSE loss for Y prediction
    - {stage}_mae_X: Mean Absolute Error for X
    - {stage}_mae_Y: Mean Absolute Error for Y
    - {stage}_mae: Combined MAE (X and Y)
    - {stage}_r2_X: R² score for X
    - {stage}_r2_Y: R² score for Y
    
    **Conditional Metrics (based on config):**
    
    If log_entropy=True:
    - {stage}_dec1_cross_entropy: Decoder 1 cross-attention entropy (S→X)
    - {stage}_dec1_self_entropy: Decoder 1 self-attention entropy (X→X)
    - {stage}_dec2_cross_entropy: Decoder 2 cross-attention entropy (X→Y)
    - {stage}_dec2_self_entropy: Decoder 2 self-attention entropy (Y→Y)
    
    If log_acyclicity=True:
    - {stage}_notears: NOTEARS acyclicity constraint value
    
    If log_sparsity=True:
    - {stage}_sparsity_self: Self-attention L1 sparsity penalty
    - {stage}_sparsity_cross: Cross-attention L1 sparsity penalty
    - {stage}_sparsity_total: Total sparsity regularization
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True, display plots. If False (for cluster), only save to files.
        metric_patterns: Optional list of patterns to filter columns.
                        If None, discovers and plots all numeric columns with train/val pairs.
                        E.g., ["mae", "loss", "r2", "entropy", "sparsity", "notears"]
        
    Returns:
        pd.DataFrame: Combined training metrics from all k-folds
        
    Output Files:
        - fig/{metric_name}_{exp_id}.pdf: Plot for each discovered metric
        - files/available_metrics.json: List of all discovered metrics
        - files/metrics_labels.json: Description of StageCausaliT metrics
        
    Example:
        >>> # Plot all metrics
        >>> df = eval_metrics("../experiments/stage/my_experiment")
        >>> 
        >>> # Plot only loss and MAE metrics
        >>> df = eval_metrics("../experiments/stage/my_experiment", metric_patterns=["loss", "mae"])
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_metrics")
    
    print(f"Experiment ID: {exp_id}")
    
    # Save README
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Flexible metric plotting for all logged training metrics",
        files_info={
            "available_metrics.json": "List of all discovered metric columns",
            "{metric}_{exp_id}.pdf": "Individual plots for each metric",
        }
    )
    
    # Load training metrics
    df = load_training_metrics(experiment)
    df = df.groupby(["kfold", "epoch"]).first().reset_index()
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter by patterns if provided
    if metric_patterns is not None:
        filtered_cols = []
        for col in numeric_cols:
            for pattern in metric_patterns:
                if pattern.lower() in col.lower():
                    filtered_cols.append(col)
                    break
        numeric_cols = filtered_cols
    
    # Exclude common non-metric columns
    exclude_cols = ['epoch', 'step']
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Group metrics into train/val pairs
    metric_pairs = {}  # base_name -> {"train": col, "val": col}
    standalone_metrics = []
    
    for col in numeric_cols:
        if col.startswith("train_"):
            base_name = col[6:]  # Remove "train_" prefix
            if base_name not in metric_pairs:
                metric_pairs[base_name] = {}
            metric_pairs[base_name]["train"] = col
        elif col.startswith("val_"):
            base_name = col[4:]  # Remove "val_" prefix
            if base_name not in metric_pairs:
                metric_pairs[base_name] = {}
            metric_pairs[base_name]["val"] = col
        elif col.startswith("test_"):
            # Skip test metrics (evaluated once)
            pass
        else:
            standalone_metrics.append(col)
    
    # Save available metrics
    available_metrics = {
        "paired_metrics": list(metric_pairs.keys()),
        "standalone_metrics": standalone_metrics,
        "total_columns": len(numeric_cols),
    }
    with open(join(eval_path_files, "available_metrics.json"), 'w') as f:
        json.dump(available_metrics, f, indent=2)
    
    print(f"Found {len(metric_pairs)} paired metrics and {len(standalone_metrics)} standalone metrics")
    
    # Plot paired metrics
    for base_name, cols in metric_pairs.items():
        train_col = cols.get("train")
        val_col = cols.get("val")
        
        # Skip if we don't have both train and val
        if train_col is None or val_col is None:
            continue
        
        # Skip if columns don't exist or are all NaN
        if train_col not in df.columns or val_col not in df.columns:
            continue
        if df[train_col].isna().all() or df[val_col].isna().all():
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot validation (solid) and training (dashed)
        sns.lineplot(data=df, x="epoch", y=val_col, hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y=train_col, hue="kfold", ax=ax, 
                    legend=False, linestyle=":")
        
        # Determine if log scale is appropriate
        min_val = min(df[val_col].min(), df[train_col].min())
        max_val = max(df[val_col].max(), df[train_col].max())
        if min_val > 0 and max_val / min_val > 100:
            ax.set_yscale("log")
        
        ax.set_ylabel(base_name)
        ax.set_title(f"{base_name}\nval (solid) | train (dashed)")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"{base_name}_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"  ✓ Plotted: {base_name}")
    
    # Plot standalone metrics
    for col in standalone_metrics:
        if col not in df.columns or df[col].isna().all():
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df, x="epoch", y=col, hue="kfold", ax=ax)
        
        # Log scale if appropriate
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val > 0 and max_val / min_val > 100:
            ax.set_yscale("log")
        
        ax.set_ylabel(col)
        ax.set_title(col)
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"{col}_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        print(f"  ✓ Plotted: {col}")
    
    print(f"\nEvaluation complete! Results saved to: {eval_path_root}")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Example usage for dyconex evaluation functions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Dyconex evaluation functions")
    parser.add_argument("experiment", help="Path to experiment folder")
    parser.add_argument("--datadir", default=None, help="Path to data directory")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots (for cluster)")
    parser.add_argument("--function", choices=["predictions", "metrics", "all"], 
                        default="all", help="Which evaluation to run")
    
    args = parser.parse_args()
    
    show_plots = not args.no_show
    
    if args.function in ["predictions", "all"]:
        print("\n" + "="*60)
        print("Running eval_dyconex_predictions...")
        print("="*60)
        eval_dyconex_predictions(args.experiment, args.datadir, show_plots=show_plots)
    
    if args.function in ["metrics", "all"]:
        print("\n" + "="*60)
        print("Running eval_metrics...")
        print("="*60)
        eval_metrics(args.experiment, show_plots=show_plots)
