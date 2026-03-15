"""
Attention Necessity Score (ANS) evaluation functions for CausaliT experiments.

This module provides functions for analyzing ANS sweep experiments to determine
if attention is necessary for model performance. ANS measures whether learned
attention improves predictions compared to uniform (bypass) attention.

The ANS framework:
- ANS(λ) = Loss(ablated, λ) - Loss(full, λ)
- λ = embedding L1 regularization strength (controls embedding capacity)
- ablated = uniform attention (attention_bypass=True)
- full = learned attention (attention_bypass=False)

Interpretation:
- ANS > 0: Attention improves performance → attention is useful
- ANS ≈ 0: Attention doesn't help → embeddings alone suffice (overcapacity)
- ANS < 0: Shouldn't happen (attention shouldn't hurt)

Example:
    >>> from causaliT.evaluation.eval_funs.eval_ans import eval_ans
    >>> results = eval_ans("experiments/single/scm6/euler/ANS_single_Lie_SM_scm6_SVFA_59792944")
    >>> print(f"λ_critical: {results['lambda_critical']}")
"""

import re
import json
from os.path import join, exists, isdir, basename
from os import makedirs, listdir
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from omegaconf import OmegaConf

# Import shared utilities
from .eval_utils import (
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    DEFAULT_PLOT_FORMAT,
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
# Helper Functions
# =============================================================================

def _discover_ans_sweep_runs(sweep_dir: str) -> List[Dict[str, Any]]:
    """
    Discover all sweep runs in an ANS sweep experiment.
    
    Parses run folder names to extract lambda_embed_l1 and attention_bypass values.
    
    Args:
        sweep_dir: Path to the sweeper/runs/combinations/ directory
        
    Returns:
        List of dicts with keys:
            - path: Full path to run folder
            - name: Folder name
            - lambda_embed_l1: float value of embedding L1 regularization
            - attention_bypass: bool (True=ablated, False=full)
            
    Example:
        >>> runs = _discover_ans_sweep_runs("experiment/sweeper/runs/combinations")
        >>> print(runs[0])
        {'path': '...', 'lambda_embed_l1': 0.0, 'attention_bypass': False}
    """
    if not exists(sweep_dir) or not isdir(sweep_dir):
        raise FileNotFoundError(f"Sweep runs directory not found: {sweep_dir}")
    
    runs = []
    
    # Pattern to match: ..._lambda_embed_l1_{value}_attention_bypass_{True/False}
    pattern = re.compile(
        r'lambda_embed_l1_([0-9.e+-]+)_attention_bypass_(True|False)',
        re.IGNORECASE
    )
    
    for folder_name in listdir(sweep_dir):
        folder_path = join(sweep_dir, folder_name)
        
        if not isdir(folder_path):
            continue
        
        match = pattern.search(folder_name)
        if match:
            lambda_val_str = match.group(1)
            bypass_str = match.group(2)
            
            # Parse lambda value
            try:
                lambda_val = float(lambda_val_str)
            except ValueError:
                print(f"Warning: Could not parse lambda value '{lambda_val_str}' in {folder_name}")
                continue
            
            # Parse bypass boolean
            attention_bypass = bypass_str.lower() == 'true'
            
            runs.append({
                'path': folder_path,
                'name': folder_name,
                'lambda_embed_l1': lambda_val,
                'attention_bypass': attention_bypass,
            })
    
    if not runs:
        raise ValueError(f"No valid ANS sweep runs found in {sweep_dir}")
    
    # Sort by lambda, then by bypass
    runs.sort(key=lambda x: (x['lambda_embed_l1'], x['attention_bypass']))
    
    return runs


def _load_kfold_metrics(run_path: str, metric: str = "val_loss") -> Dict[int, float]:
    """
    Load per-fold metric values from a run's kfold_summary.json.
    
    Args:
        run_path: Path to the run folder containing kfold_summary.json
        metric: Name of the metric to extract (default: "val_loss")
        
    Returns:
        Dict mapping fold index (int) to metric value (float)
        
    Example:
        >>> metrics = _load_kfold_metrics("path/to/run", "val_loss")
        >>> print(metrics)
        {0: 0.00011, 1: 0.00010, 2: 0.00010, ...}
    """
    kfold_path = join(run_path, "kfold_summary.json")
    
    if not exists(kfold_path):
        raise FileNotFoundError(f"kfold_summary.json not found: {kfold_path}")
    
    with open(kfold_path, 'r') as f:
        kfold_data = json.load(f)
    
    fold_results = kfold_data.get("fold_results", {})
    
    if not fold_results:
        raise ValueError(f"No fold_results in {kfold_path}")
    
    metrics = {}
    for fold_id, fold_data in fold_results.items():
        fold_idx = int(fold_id)
        fold_metrics = fold_data.get("metrics", {})
        
        if metric not in fold_metrics:
            print(f"Warning: Metric '{metric}' not found in fold {fold_id} of {run_path}")
            continue
        
        value = fold_metrics[metric]
        
        # Handle tensor string format if present
        if isinstance(value, str) and value.startswith("tensor("):
            match = re.match(r'^tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)$', value.strip())
            if match:
                value = float(match.group(1))
            else:
                print(f"Warning: Could not parse tensor string '{value}'")
                continue
        
        metrics[fold_idx] = float(value)
    
    return metrics


def _compute_paired_ans(
    full_metrics: Dict[int, float],
    ablated_metrics: Dict[int, float],
) -> Tuple[List[float], List[int]]:
    """
    Compute paired ANS values for matching folds.
    
    ANS = Loss(ablated) - Loss(full)
    
    Args:
        full_metrics: Dict mapping fold_idx to loss for full model (bypass=False)
        ablated_metrics: Dict mapping fold_idx to loss for ablated model (bypass=True)
        
    Returns:
        Tuple of (ans_values, fold_indices):
            - ans_values: List of ANS values for each paired fold
            - fold_indices: List of fold indices that were successfully paired
    """
    common_folds = set(full_metrics.keys()) & set(ablated_metrics.keys())
    
    if not common_folds:
        return [], []
    
    fold_indices = sorted(common_folds)
    ans_values = []
    
    for fold_idx in fold_indices:
        # ANS = Loss(ablated) - Loss(full)
        # Positive ANS means attention helps (lower loss with learned attention)
        ans = ablated_metrics[fold_idx] - full_metrics[fold_idx]
        ans_values.append(ans)
    
    return ans_values, fold_indices


# =============================================================================
# Main Evaluation Function
# =============================================================================

def eval_ans(
    experiment: str,
    metric: str = "val_loss",
    significance_level: float = 0.05,
    show_plots: bool = True,
) -> dict:
    """
    Evaluate Attention Necessity Score (ANS) from a sweep experiment.
    
    Computes ANS for each lambda value by comparing validation loss between
    the full model (learned attention) and ablated model (uniform attention).
    
    ANS(λ) = Loss(ablated, λ) - Loss(full, λ)
    
    Args:
        experiment: Path to the ANS sweep experiment folder
                   (containing sweeper/runs/combinations/)
        metric: Metric to use for ANS computation (default: "val_loss")
        significance_level: Alpha for one-tailed t-test (default: 0.05)
        show_plots: If True, display plots. If False, only save to files.
        
    Returns:
        dict: ANS evaluation results with keys:
            - ans_curve: DataFrame with lambda, ANS stats, and p-values
            - lambda_critical: Smallest λ where ANS is significantly > 0 (or None)
            - interpretation: Text interpretation of results
            - per_fold_data: Raw per-fold ANS values for each lambda
            
    Output Files:
        - fig/ans_curve_{exp_id}.pdf: ANS vs λ with per-fold lines
        - fig/loss_comparison_{exp_id}.pdf: Full vs Ablated losses
        - files/ans_results.json: Complete ANS statistics
        - files/ans_summary.csv: Tabular summary
        - files/ans_labels.json: Interpretation guide
        
    Example:
        >>> results = eval_ans("experiments/ANS_single_Lie_SM_scm6_SVFA_59792944")
        >>> print(f"λ_critical: {results['lambda_critical']}")
        >>> print(f"Interpretation: {results['interpretation']}")
    """
    # Validate experiment path
    sweep_combinations_dir = join(experiment, "sweeper", "runs", "combinations")
    
    if not exists(sweep_combinations_dir):
        # Try alternative path structure (if experiment IS the sweeper folder)
        if exists(join(experiment, "runs", "combinations")):
            sweep_combinations_dir = join(experiment, "runs", "combinations")
        else:
            raise FileNotFoundError(
                f"No sweep runs found. Expected: {sweep_combinations_dir}\n"
                f"Make sure this is an ANS sweep experiment with sweeper/runs/combinations/"
            )
    
    # Setup evaluation directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_ans")
    
    print(f"Experiment ID: {exp_id}")
    print(f"Sweep runs directory: {sweep_combinations_dir}")
    print(f"Metric: {metric}")
    
    # =========================================================================
    # Save documentation files
    # =========================================================================
    
    ans_labels = {
        "description": "Attention Necessity Score (ANS) evaluation results",
        "ans_definition": "ANS(λ) = Loss(ablated, λ) - Loss(full, λ)",
        "interpretation": {
            "ANS > 0": "Attention improves performance → attention is useful",
            "ANS ≈ 0": "Attention doesn't help → embeddings alone suffice",
            "ANS < 0": "Shouldn't happen (attention shouldn't hurt)",
        },
        "lambda_critical": "Smallest λ where ANS is significantly positive (p < α)",
        "columns": {
            "lambda_embed_l1": "Embedding L1 regularization strength",
            "ANS_mean": "Mean ANS across k-folds",
            "ANS_std": "Standard deviation of ANS across k-folds",
            "p_value": "One-tailed t-test p-value (H0: ANS ≤ 0, H1: ANS > 0)",
            "significant": "Whether ANS > 0 is statistically significant",
            "full_loss_mean": "Mean validation loss for full model (bypass=False)",
            "ablated_loss_mean": "Mean validation loss for ablated model (bypass=True)",
        },
    }
    _save_variable_labels(eval_path_files, ans_labels, "ans_labels.json")
    
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Attention Necessity Score (ANS) evaluation for determining if attention is necessary.",
        files_info={
            "ans_results.json": "Complete ANS statistics with per-fold data",
            "ans_summary.csv": "Tabular summary of ANS per lambda",
            "ans_labels.json": "Interpretation guide for ANS metrics",
        },
        column_documentation=ans_labels["columns"]
    )
    
    _create_cline_template(eval_path_cline, "eval_ans", exp_id)
    
    # =========================================================================
    # Discover and load sweep runs
    # =========================================================================
    
    print("\n--- Discovering sweep runs ---")
    runs = _discover_ans_sweep_runs(sweep_combinations_dir)
    print(f"Found {len(runs)} sweep runs")
    
    # Group runs by lambda value
    runs_by_lambda: Dict[float, Dict[str, Any]] = {}
    for run in runs:
        lam = run['lambda_embed_l1']
        bypass = run['attention_bypass']
        
        if lam not in runs_by_lambda:
            runs_by_lambda[lam] = {'full': None, 'ablated': None}
        
        if bypass:
            runs_by_lambda[lam]['ablated'] = run
        else:
            runs_by_lambda[lam]['full'] = run
    
    # Verify we have pairs
    complete_lambdas = [lam for lam, pair in runs_by_lambda.items() 
                        if pair['full'] is not None and pair['ablated'] is not None]
    
    if not complete_lambdas:
        raise ValueError("No complete (full, ablated) pairs found for any lambda value")
    
    complete_lambdas.sort()
    print(f"Complete lambda pairs: {complete_lambdas}")
    
    # =========================================================================
    # Compute ANS for each lambda
    # =========================================================================
    
    print(f"\n--- Computing ANS (metric: {metric}) ---")
    
    ans_results = []
    per_fold_data = {}  # lambda -> {fold_idx: ans_value}
    loss_data = {
        'full': {},      # lambda -> {fold_idx: loss}
        'ablated': {},   # lambda -> {fold_idx: loss}
    }
    
    for lam in complete_lambdas:
        full_run = runs_by_lambda[lam]['full']
        ablated_run = runs_by_lambda[lam]['ablated']
        
        try:
            # Load per-fold metrics
            full_metrics = _load_kfold_metrics(full_run['path'], metric)
            ablated_metrics = _load_kfold_metrics(ablated_run['path'], metric)
            
            # Store for plotting
            loss_data['full'][lam] = full_metrics
            loss_data['ablated'][lam] = ablated_metrics
            
            # Compute paired ANS
            ans_values, fold_indices = _compute_paired_ans(full_metrics, ablated_metrics)
            
            if not ans_values:
                print(f"  λ={lam}: No valid fold pairs found")
                continue
            
            # Store per-fold ANS
            per_fold_data[lam] = {fold_idx: ans_val 
                                  for fold_idx, ans_val in zip(fold_indices, ans_values)}
            
            # Compute statistics
            ans_array = np.array(ans_values)
            ans_mean = float(np.mean(ans_array))
            ans_std = float(np.std(ans_array, ddof=1)) if len(ans_array) > 1 else 0.0
            
            # One-tailed t-test: H0: ANS <= 0, H1: ANS > 0
            if len(ans_array) > 1:
                t_stat, two_sided_p = stats.ttest_1samp(ans_array, 0)
                # Convert to one-tailed (we care about ANS > 0)
                if t_stat > 0:
                    p_value = two_sided_p / 2
                else:
                    p_value = 1 - two_sided_p / 2
            else:
                t_stat = float('nan')
                p_value = 1.0  # Can't conclude anything with n=1
            
            significant = bool(p_value < significance_level and ans_mean > 0)
            
            # Mean losses for comparison
            full_loss_mean = float(np.mean(list(full_metrics.values())))
            ablated_loss_mean = float(np.mean(list(ablated_metrics.values())))
            
            ans_results.append({
                'lambda_embed_l1': float(lam),
                'ANS_mean': ans_mean,
                'ANS_std': ans_std,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': significant,
                'n_folds': int(len(ans_values)),
                'full_loss_mean': full_loss_mean,
                'ablated_loss_mean': ablated_loss_mean,
            })
            
            sig_marker = "*" if significant else ""
            print(f"  λ={lam}: ANS={ans_mean:.6f} ± {ans_std:.6f}, "
                  f"p={p_value:.4f}{sig_marker}")
            
        except Exception as e:
            print(f"  λ={lam}: Error - {e}")
            continue
    
    if not ans_results:
        raise ValueError("Failed to compute ANS for any lambda value")
    
    # Create DataFrame
    df_ans = pd.DataFrame(ans_results)
    
    # =========================================================================
    # Find λ_critical
    # =========================================================================
    
    significant_lambdas = df_ans[df_ans['significant']]['lambda_embed_l1'].tolist()
    
    if significant_lambdas:
        lambda_critical = min(significant_lambdas)
        print(f"\n✓ λ_critical = {lambda_critical} (smallest λ where ANS > 0 is significant)")
    else:
        lambda_critical = None
        print(f"\n✗ No λ_critical found (ANS never significantly positive)")
    
    # =========================================================================
    # Generate interpretation
    # =========================================================================
    
    # Check if ANS is ever positive
    any_positive_ans = any(r['ANS_mean'] > 0 for r in ans_results)
    all_positive_ans = all(r['ANS_mean'] > 0 for r in ans_results)
    
    if lambda_critical is not None:
        if lambda_critical == 0.0:
            interpretation = (
                "Attention is NECESSARY even without embedding regularization. "
                "The model relies on attention structure for predictions."
            )
        else:
            interpretation = (
                f"Attention becomes necessary when embedding capacity is constrained "
                f"(λ ≥ {lambda_critical}). Below this threshold, embeddings alone suffice."
            )
    elif any_positive_ans:
        interpretation = (
            "ANS is positive for some λ values but not statistically significant. "
            "Consider running with more k-folds or stronger regularization."
        )
    else:
        interpretation = (
            "ANS is near zero or negative for all λ values. "
            "Attention may not be necessary for this task, or the model has too much "
            "embedding capacity to require structured attention."
        )
    
    print(f"\nInterpretation: {interpretation}")
    
    # =========================================================================
    # Plot 1: ANS Curve with per-fold lines
    # =========================================================================
    
    print("\n--- Generating plots ---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colormap for folds
    cmap = plt.get_cmap('tab10')
    n_folds = max(len(v) for v in per_fold_data.values()) if per_fold_data else 5
    fold_colors = {i: cmap(i / max(n_folds - 1, 1)) for i in range(n_folds)}
    
    # Get all lambdas with data
    lambdas_sorted = sorted(per_fold_data.keys())
    
    # Plot per-fold lines (thin, dotted)
    for fold_idx in range(n_folds):
        fold_lambdas = []
        fold_ans = []
        
        for lam in lambdas_sorted:
            if fold_idx in per_fold_data.get(lam, {}):
                fold_lambdas.append(lam)
                fold_ans.append(per_fold_data[lam][fold_idx])
        
        if fold_lambdas:
            ax.plot(fold_lambdas, fold_ans, 
                   linestyle=':', linewidth=1, alpha=0.6,
                   color=fold_colors[fold_idx], label=f'k_{fold_idx}')
    
    # Plot mean line (thick, solid)
    mean_lambdas = df_ans['lambda_embed_l1'].values
    mean_ans = df_ans['ANS_mean'].values
    ax.plot(mean_lambdas, mean_ans, 
           'k-', linewidth=2.5, label='Mean', zorder=10)
    
    # Plot confidence band (±1 std)
    std_ans = df_ans['ANS_std'].values
    ax.fill_between(mean_lambdas, mean_ans - std_ans, mean_ans + std_ans,
                   alpha=0.2, color='gray', label='±1 std')
    
    # Reference line at ANS=0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='ANS=0')
    
    # Mark λ_critical if found
    if lambda_critical is not None:
        ax.axvline(x=lambda_critical, color='green', linestyle='--', linewidth=1.5,
                  label=f'λ_critical={lambda_critical}')
    
    # Mark significant points
    sig_mask = df_ans['significant'].values
    if sig_mask.any():
        ax.scatter(mean_lambdas[sig_mask], mean_ans[sig_mask], 
                  marker='*', s=200, c='green', zorder=15, label='Significant')
    
    ax.set_xscale('log')
    ax.set_xlabel('λ_embed_l1 (log scale)')
    ax.set_ylabel(f'ANS = Loss(ablated) - Loss(full)')
    ax.set_title(f'Attention Necessity Score (ANS) Curve\nMetric: {metric}')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"ans_curve_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"  ✓ Saved: ans_curve_{exp_id}.pdf")
    
    # =========================================================================
    # Plot 2: Loss Comparison (Full vs Ablated)
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot per-fold lines for full model (solid thin)
    for fold_idx in range(n_folds):
        fold_lambdas = []
        fold_loss_full = []
        fold_loss_ablated = []
        
        for lam in lambdas_sorted:
            if (fold_idx in loss_data['full'].get(lam, {}) and 
                fold_idx in loss_data['ablated'].get(lam, {})):
                fold_lambdas.append(lam)
                fold_loss_full.append(loss_data['full'][lam][fold_idx])
                fold_loss_ablated.append(loss_data['ablated'][lam][fold_idx])
        
        if fold_lambdas:
            # Full model (solid)
            ax.plot(fold_lambdas, fold_loss_full,
                   linestyle='-', linewidth=1, alpha=0.4,
                   color=fold_colors[fold_idx])
            # Ablated model (dashed)
            ax.plot(fold_lambdas, fold_loss_ablated,
                   linestyle='--', linewidth=1, alpha=0.4,
                   color=fold_colors[fold_idx])
    
    # Plot mean lines (thick)
    ax.plot(df_ans['lambda_embed_l1'], df_ans['full_loss_mean'],
           'b-', linewidth=2.5, label='Full (learned attention)', zorder=10)
    ax.plot(df_ans['lambda_embed_l1'], df_ans['ablated_loss_mean'],
           'r--', linewidth=2.5, label='Ablated (uniform attention)', zorder=10)
    
    # Mark λ_critical
    if lambda_critical is not None:
        ax.axvline(x=lambda_critical, color='green', linestyle='--', linewidth=1.5,
                  label=f'λ_critical={lambda_critical}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('λ_embed_l1 (log scale)')
    ax.set_ylabel(f'{metric} (log scale)')
    ax.set_title(f'Full vs Ablated Model Loss Comparison\nGap = ANS')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"loss_comparison_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    print(f"  ✓ Saved: loss_comparison_{exp_id}.pdf")
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    # Save summary CSV
    df_ans.to_csv(join(eval_path_files, "ans_summary.csv"), index=False)
    print(f"  ✓ Saved: ans_summary.csv")
    
    # Save complete results JSON
    results_json = {
        "experiment": experiment,
        "exp_id": exp_id,
        "metric": metric,
        "significance_level": significance_level,
        "lambda_critical": lambda_critical,
        "interpretation": interpretation,
        "lambdas_tested": lambdas_sorted,
        "ans_statistics": ans_results,
        "per_fold_data": {str(k): v for k, v in per_fold_data.items()},
        "loss_data": {
            "full": {str(k): v for k, v in loss_data['full'].items()},
            "ablated": {str(k): v for k, v in loss_data['ablated'].items()},
        },
    }
    
    with open(join(eval_path_files, "ans_results.json"), 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Saved: ans_results.json")
    
    # =========================================================================
    # Return results
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("ANS Evaluation Complete!")
    print(f"  λ_critical: {lambda_critical}")
    print(f"  Interpretation: {interpretation}")
    print(f"  Results saved to: {eval_path_root}")
    print('='*60)
    
    return {
        "ans_curve": df_ans,
        "lambda_critical": lambda_critical,
        "interpretation": interpretation,
        "per_fold_data": per_fold_data,
        "loss_data": loss_data,
        "experiment": experiment,
        "metric": metric,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Attention Necessity Score (ANS) from a sweep experiment"
    )
    parser.add_argument("experiment", help="Path to ANS sweep experiment folder")
    parser.add_argument("--metric", default="val_loss", 
                       help="Metric to use for ANS (default: val_loss)")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level for λ_critical (default: 0.05)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (save to files only)")
    
    args = parser.parse_args()
    
    eval_ans(
        experiment=args.experiment,
        metric=args.metric,
        significance_level=args.alpha,
        show_plots=not args.no_show,
    )
