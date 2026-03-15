"""
Evaluation Functions for Dyconex Dataset (ds_dyconex_SX_MuMi_*).

This module provides evaluation functions specific to the dyconex industrial dataset:
- eval_dyconex_predictions: Prediction quality analysis with best/worst sample plots

Note: For training metrics plotting, use eval_train_metrics from eval_training.py,
      which provides generalized metric plotting for any experiment.

The dyconex dataset has:
- S (source): 52 unique variables, 106 sequence length
- X (input): 77 unique variables, 173 sequence length  
- Y (target): 2 variables (delta_A_norm, delta_B_norm), 400 sequence length
  - First 200 timesteps: delta_A_norm
  - Last 200 timesteps: delta_B_norm

Example:
    >>> from notebooks.eval_funs.eval_dyconex import eval_dyconex_predictions
    >>> from notebooks.eval_funs.eval_training import eval_train_metrics
    >>> 
    >>> # Evaluate predictions on test set
    >>> results = eval_dyconex_predictions("../experiments/stage/stage_SM_SM_dyconex")
    >>> 
    >>> # Flexible metric plotting (use generalized function)
    >>> df = eval_train_metrics("../experiments/stage/stage_SM_SM_dyconex")
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
from .eval_utils import DEFAULT_PLOT_FORMAT

# Import from local eval_funs modules (self-contained)
from .eval_lib import (
    find_config_file,
    find_best_or_last_checkpoint,
    load_training_metrics,
)


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
    plt.savefig(join(eval_path_fig, f"mae_distribution_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
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
            filename=join(eval_path_fig, f"best_sample_{i+1}_{exp_id}.{DEFAULT_PLOT_FORMAT}")
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
            filename=join(eval_path_fig, f"worst_sample_{i+1}_{exp_id}.{DEFAULT_PLOT_FORMAT}")
        )
    
    print(f"\nEvaluation complete! Results saved to: {eval_path_root}")
    
    return mae_statistics


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Example usage for dyconex evaluation functions.
    
    For training metrics plotting, use eval_train_metrics from eval_training.py:
        from notebooks.eval_funs.eval_training import eval_train_metrics
        df = eval_train_metrics(experiment, show_plots=show_plots)
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
        print("Running eval_train_metrics (generalized)...")
        print("="*60)
        # Import the generalized function from eval_training
        from .eval_training import eval_train_metrics
        eval_train_metrics(args.experiment, show_plots=show_plots)
