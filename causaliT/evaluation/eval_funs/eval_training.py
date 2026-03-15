"""
Training metrics evaluation functions for CausaliT experiments.

This module provides functions for analyzing training metrics:
- eval_train_metrics: Analyze training metrics (loss curves, regularization terms)

The function auto-discovers and plots any logged train/val metric pairs.
"""

import json
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

# Import shared utilities
from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    _plot_metric_pair,
    _discover_metric_pairs,
    _filter_metric_pairs,
    _is_column_plottable,
    compute_instability_metrics,
    DEFAULT_PLOT_FORMAT,
)

# Import from local eval_funs modules (self-contained)
from .eval_lib import load_training_metrics


# =============================================================================
# Default Metric Configuration
# =============================================================================

# Default plot options for common metrics
DEFAULT_PLOT_OPTIONS = {
    "loss": {"ylabel": "Loss (MSE)", "use_log_scale": "auto"},
    "loss_X": {"ylabel": "Loss X (MSE)", "use_log_scale": "auto"},
    "loss_Y": {"ylabel": "Loss Y (MSE)", "use_log_scale": "auto"},
    "mae": {"ylabel": "MAE", "use_log_scale": "auto"},
    "mae_X": {"ylabel": "MAE X", "use_log_scale": "auto"},
    "mae_Y": {"ylabel": "MAE Y", "use_log_scale": "auto"},
    "r2": {"ylabel": "R²", "use_log_scale": "never"},
    "r2_X": {"ylabel": "R² X", "use_log_scale": "never"},
    "r2_Y": {"ylabel": "R² Y", "use_log_scale": "never"},
    "notears": {"ylabel": "NOTEARS", "use_log_scale": "auto"},
    "hsic": {"ylabel": "HSIC (raw)", "use_log_scale": "always"},
    "hsic_reg": {"ylabel": "HSIC Regularizer", "use_log_scale": "always"},
    "sparsity_cross": {"ylabel": "Sparsity Cross", "use_log_scale": "always"},
    "sparsity_self": {"ylabel": "Sparsity Self", "use_log_scale": "always"},
    "sparsity_total": {"ylabel": "Sparsity Total", "use_log_scale": "always"},
    "dec1_cross_entropy": {"ylabel": "Dec1 Cross Entropy", "use_log_scale": "auto"},
    "dec1_self_entropy": {"ylabel": "Dec1 Self Entropy", "use_log_scale": "auto"},
    "dec2_cross_entropy": {"ylabel": "Dec2 Cross Entropy", "use_log_scale": "auto"},
    "dec2_self_entropy": {"ylabel": "Dec2 Self Entropy", "use_log_scale": "auto"},
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def eval_train_metrics(
    experiment: str, 
    show_plots: bool = False,
    metric_patterns: List[str] = None,
    plot_options: Dict[str, dict] = None,
    order_of_magnitude_threshold: float = 2.0,
    figsize: tuple = (8, 5),
) -> pd.DataFrame:
    """
    Evaluate and visualize training metrics from an experiment.
    
    Auto-discovers and plots all train/val metric pairs logged during training.
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
        show_plots: If True (default), display plots interactively. If False, only save to files.
        metric_patterns: Optional list of patterns to filter which metrics to plot.
                        If None, discovers and plots all numeric train/val pairs.
                        E.g., ["mae", "loss", "r2", "entropy", "sparsity", "notears"]
        plot_options: Optional dict mapping metric base names to plot options.
                     Each value is a dict with keys like {"ylabel": "Label", "use_log_scale": "auto"}.
                     Overrides DEFAULT_PLOT_OPTIONS.
        order_of_magnitude_threshold: For "auto" log scale mode, use log if max/min > 10^threshold 
                     (default: 2.0 = 100x span). Allows log scale even with some negative values
                     (up to 50%), clipping negatives to a safe minimum.
        figsize: Figure size for each plot (default: (8, 5))
        
    Returns:
        pd.DataFrame: Combined training metrics from all k-folds with columns:
            - kfold: Fold identifier (e.g., "k_0", "k_1")
            - epoch: Training epoch
            - train_loss, val_loss: Training and validation loss
            - Additional columns for any logged regularization terms
            
    Output Files:
        - fig/{metric_name}_{exp_id}.pdf: Plot for each discovered metric
        - fig/metrics_corr_{exp_id}.pdf: Correlation heatmap
        - files/matrix_corr.csv: Correlation matrix data
        - files/available_metrics.json: List of all discovered metrics
        - files/metric_labels.json: Descriptions for all metrics
        
    Example:
        >>> # Plot all metrics
        >>> df = eval_train_metrics("../experiments/single/local/my_experiment")
        >>> 
        >>> # Plot only loss and MAE metrics
        >>> df = eval_train_metrics("../experiments/stage/my_experiment", metric_patterns=["loss", "mae"])
        >>> 
        >>> # Custom labels
        >>> df = eval_train_metrics(
        ...     "../experiments/stage/my_experiment",
        ...     plot_options={"loss": {"ylabel": "Custom Loss Label"}}
        ... )
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_train_metrics")
    
    metrics_corr_filename = "matrix_corr.csv"

    # Define metric descriptions for AI interpretation
    metric_labels = {
        "description": "Training metrics logged during model optimization",
        "metric_descriptions": {
            "train_loss": "Training set MSE loss (prediction error)",
            "val_loss": "Validation set MSE loss (generalization error)",
            "test_loss": "Test set MSE loss (final evaluation)",
            "train_loss_X": "Training set MSE loss for X reconstruction",
            "val_loss_X": "Validation set MSE loss for X reconstruction",
            "train_loss_Y": "Training set MSE loss for Y prediction",
            "val_loss_Y": "Validation set MSE loss for Y prediction",
            "train_mae": "Training set Mean Absolute Error",
            "val_mae": "Validation set Mean Absolute Error",
            "train_mae_X": "Training set MAE for X",
            "val_mae_X": "Validation set MAE for X",
            "train_mae_Y": "Training set MAE for Y",
            "val_mae_Y": "Validation set MAE for Y",
            "train_r2": "Training set R² score (explained variance)",
            "val_r2": "Validation set R² score",
            "train_r2_X": "Training set R² for X predictions",
            "val_r2_X": "Validation set R² for X predictions",
            "train_r2_Y": "Training set R² for Y predictions",
            "val_r2_Y": "Validation set R² for Y predictions",
            "train_notears": "NOTEARS acyclicity constraint on training set (0 = DAG)",
            "val_notears": "NOTEARS acyclicity constraint on validation set",
            "train_hsic": "HSIC value (residuals vs S) - training",
            "val_hsic": "HSIC value (residuals vs S) - validation",
            "train_hsic_reg": "HSIC independence regularization (lambda * hsic) - training",
            "val_hsic_reg": "HSIC independence regularization (lambda * hsic) - validation",
            "train_sparsity_cross": "L1 sparsity on cross-attention (S→X edges)",
            "val_sparsity_cross": "L1 sparsity on cross-attention - validation",
            "train_sparsity_self": "L1 sparsity on self-attention (X→X edges)",
            "val_sparsity_self": "L1 sparsity on self-attention - validation",
            "train_sparsity_total": "Total L1 sparsity (self + cross)",
            "val_sparsity_total": "Total L1 sparsity - validation",
            "train_dec1_cross_entropy": "Decoder 1 cross-attention entropy (S→X) - training",
            "val_dec1_cross_entropy": "Decoder 1 cross-attention entropy (S→X) - validation",
            "train_dec1_self_entropy": "Decoder 1 self-attention entropy (X→X) - training",
            "val_dec1_self_entropy": "Decoder 1 self-attention entropy (X→X) - validation",
            "train_dec2_cross_entropy": "Decoder 2 cross-attention entropy (X→Y) - training",
            "val_dec2_cross_entropy": "Decoder 2 cross-attention entropy (X→Y) - validation",
            "train_dec2_self_entropy": "Decoder 2 self-attention entropy (Y→Y) - training",
            "val_dec2_self_entropy": "Decoder 2 self-attention entropy (Y→Y) - validation",
        },
        "interpretation": {
            "lower_is_better": [
                "train_loss", "val_loss", "test_loss",
                "train_loss_X", "val_loss_X", "train_loss_Y", "val_loss_Y",
                "train_mae", "val_mae", "train_mae_X", "val_mae_X", "train_mae_Y", "val_mae_Y",
                "train_notears", "val_notears",
                "train_hsic", "val_hsic",
                "train_hsic_reg", "val_hsic_reg",
                "train_sparsity_cross", "val_sparsity_cross",
                "train_sparsity_self", "val_sparsity_self", 
                "train_sparsity_total", "val_sparsity_total",
            ],
            "higher_is_better": [
                "train_r2", "val_r2", "test_r2",
                "train_r2_X", "val_r2_X", "train_r2_Y", "val_r2_Y",
            ],
        },
        "column_documentation": {
            "kfold": "Cross-validation fold identifier (k_0, k_1, ...)",
            "epoch": "Training epoch number",
            "step": "Training step (batch) number",
        }
    }
    _save_variable_labels(eval_path_files, metric_labels, "metric_labels.json")

    # Save README with column documentation
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation folder contains plots of the metrics logged during training.",
        files_info={
            "matrix_corr.csv": "Pairwise correlation matrix between all training metrics",
            "metric_labels.json": "Descriptions and interpretation guide for all metrics",
            "available_metrics.json": "List of all discovered metric columns",
        },
        column_documentation=metric_labels["column_documentation"]
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_train_metrics", exp_id)
    
    print(f"Experiment ID: {exp_id}")
    
    # Load and preprocess metrics
    df = load_training_metrics(experiment)
    df = df.groupby(["kfold", "epoch"]).first().reset_index()
    
    # Discover metric pairs
    metric_pairs = _discover_metric_pairs(df)
    
    # Filter by patterns if provided
    if metric_patterns is not None:
        metric_pairs = _filter_metric_pairs(metric_pairs, metric_patterns)
    
    # Save available metrics info
    available_metrics = {
        "discovered_pairs": list(metric_pairs.keys()),
        "total_pairs": len(metric_pairs),
        "filters_applied": metric_patterns if metric_patterns else "none",
    }
    with open(join(eval_path_files, "available_metrics.json"), 'w') as f:
        json.dump(available_metrics, f, indent=2)
    
    print(f"Found {len(metric_pairs)} metric pairs to plot")
    
    # Merge default options with user-provided options
    merged_options = DEFAULT_PLOT_OPTIONS.copy()
    if plot_options:
        for key, opts in plot_options.items():
            if key in merged_options:
                merged_options[key].update(opts)
            else:
                merged_options[key] = opts
    
    # Plot each metric pair
    plotted_count = 0
    for base_name, cols in metric_pairs.items():
        train_col = cols.get("train")
        val_col = cols.get("val")
        
        if train_col is None or val_col is None:
            continue
        
        # Get plot options for this metric
        opts = merged_options.get(base_name, {})
        ylabel = opts.get("ylabel", None)
        use_log_scale = opts.get("use_log_scale", "auto")
        title = opts.get("title", None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use helper function to plot
        success = _plot_metric_pair(
            df=df,
            train_col=train_col,
            val_col=val_col,
            ax=ax,
            ylabel=ylabel,
            title=title,
            use_log_scale=use_log_scale,
            order_of_magnitude_threshold=order_of_magnitude_threshold,
        )
        
        if success:
            plt.tight_layout()
            plt.savefig(join(eval_path_fig, f"{base_name}_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
            if show_plots:
                plt.show()
            else:
                plt.close()
            print(f"  ✓ Plotted: {base_name}")
            plotted_count += 1
        else:
            plt.close()
            print(f"  ⊘ Skipped: {base_name} (insufficient data)")
    
    # Compute correlation matrix (excluding test metrics)
    df_no_test = df[[c for c in df.columns if "test" not in c]]
    numeric_df = df_no_test.select_dtypes(include=['number'])
    
    if len(numeric_df.columns) > 1:
        df_corr = numeric_df.corr().abs()
        ranked = df_corr.unstack().sort_values(ascending=False)
        ranked[ranked < 1].to_csv(join(eval_path_files, metrics_corr_filename))
        
        # Plot: Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr, ax=ax, cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title("Metric Correlation Matrix")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"metrics_corr_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        print(f"  ✓ Plotted: correlation heatmap")
    
    # =========================================================================
    # Compute Training Instability Metrics
    # =========================================================================
    print("\n--- Computing Training Instability Metrics ---")
    
    instability_results = compute_training_instability(
        df=df,
        metric_col="val_loss",
        window=5,
    )
    
    # Save instability metrics to JSON
    instability_filename = "training_instability.json"
    with open(join(eval_path_files, instability_filename), 'w') as f:
        json.dump(instability_results, f, indent=2)
    print(f"  ✓ Saved: {instability_filename}")
    
    # Print summary
    print(f"\n  Instability Metrics Summary (val_loss):")
    for metric_name, metric_data in instability_results["metrics"].items():
        mean_val = metric_data["mean"]
        std_val = metric_data["std"]
        print(f"    {metric_name}: {mean_val:.6f} ± {std_val:.6f}")
    
    print(f"\nEvaluation complete!")
    print(f"  Plotted {plotted_count} metric pairs")
    print(f"  Results saved to: {eval_path_root}")
    
    return df


def compute_training_instability(
    df: pd.DataFrame,
    metric_col: str = "val_loss",
    window: int = 5,
) -> dict:
    """
    Compute training instability metrics from training curves.
    
    Computes per-fold instability metrics (spike ratio, CV, max jump, trend instability)
    and aggregates them across k-folds.
    
    Args:
        df: DataFrame with columns ['kfold', 'epoch', metric_col]
        metric_col: Column name to compute instability for (default: 'val_loss')
        window: Moving average window for trend instability (default: 5)
        
    Returns:
        dict with structure:
            {
                "metric_col": "val_loss",
                "window": 5,
                "n_folds": 5,
                "metrics": {
                    "spike_ratio": {"mean": X, "std": Y, "per_fold": {...}},
                    "cv": {"mean": X, "std": Y, "per_fold": {...}},
                    "max_jump": {"mean": X, "std": Y, "per_fold": {...}},
                    "trend_instability": {"mean": X, "std": Y, "per_fold": {...}},
                }
            }
            
    Example:
        >>> df = load_training_metrics("../experiments/my_experiment")
        >>> df = df.groupby(["kfold", "epoch"]).first().reset_index()
        >>> results = compute_training_instability(df, metric_col="val_loss")
        >>> print(f"Spike ratio: {results['metrics']['spike_ratio']['mean']:.3f}")
    """
    if metric_col not in df.columns:
        print(f"Warning: Metric column '{metric_col}' not found in DataFrame")
        return {
            "metric_col": metric_col,
            "window": window,
            "n_folds": 0,
            "metrics": {},
            "error": f"Column '{metric_col}' not found",
        }
    
    # Get unique k-folds
    kfolds = df["kfold"].unique()
    n_folds = len(kfolds)
    
    # Compute instability metrics for each fold
    fold_metrics = {
        "spike_ratio": {},
        "cv": {},
        "max_jump": {},
        "trend_instability": {},
    }
    
    for kfold in kfolds:
        # Get loss values sorted by epoch for this fold
        fold_df = df[df["kfold"] == kfold].sort_values("epoch")
        loss_values = fold_df[metric_col].values
        
        # Filter out NaN values
        clean_values = loss_values[np.isfinite(loss_values)]
        
        if len(clean_values) < 2:
            print(f"  Warning: {kfold} has insufficient data points ({len(clean_values)})")
            continue
        
        # Compute instability metrics for this fold
        fold_instability = compute_instability_metrics(clean_values, window=window)
        
        for metric_name, value in fold_instability.items():
            fold_metrics[metric_name][kfold] = value
    
    # Aggregate across folds
    aggregated_metrics = {}
    for metric_name, per_fold_dict in fold_metrics.items():
        if not per_fold_dict:
            aggregated_metrics[metric_name] = {
                "mean": None,
                "std": None,
                "per_fold": {},
            }
            continue
        
        values = np.array(list(per_fold_dict.values()))
        aggregated_metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": per_fold_dict,
        }
    
    return {
        "metric_col": metric_col,
        "window": window,
        "n_folds": n_folds,
        "metrics": aggregated_metrics,
    }
