"""
Manifest update functions for CausaliT experiments.

This module provides functions for updating the experiments manifest:
- fix_kfold_summary, enrich_kfold_summary: Data cleaning
- update_experiments_manifest, load_experiments_manifest: Manifest operations
- batch_update_manifest: Batch processing
"""

import re
import json
from os.path import dirname, abspath, join, exists, isdir
from os import makedirs, listdir
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Import root_path from eval_utils
from .eval_utils import root_path

# Default manifest location
MANIFEST_PATH = join(root_path, "experiments", "experiments_manifest.csv")


# =============================================================================
# Data Cleaning Functions
# =============================================================================

def fix_kfold_summary(experiment: str) -> bool:
    """
    Fix kfold_summary.json files that have tensor string values like "tensor(0.0005)".
    
    Converts tensor strings to proper numeric values for machine readability.
    Creates a backup of the original file before modifying.
    
    Args:
        experiment: Path to the experiment folder containing kfold_summary.json
        
    Returns:
        bool: True if file was modified, False if no changes needed
        
    Example:
        >>> fix_kfold_summary("../experiments/single/local/my_experiment")
        Fixed: kfold_summary.json (backup: kfold_summary.json.bak)
    """
    import json
    
    filepath = join(experiment, "kfold_summary.json")
    
    if not exists(filepath):
        print(f"No kfold_summary.json found in {experiment}")
        return False
    
    def _parse_tensor_string(value):
        """Parse tensor(X.XX) string to float."""
        if not isinstance(value, str):
            return value
        match = re.match(r'^tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)$', value.strip())
        if match:
            return float(match.group(1))
        return value
    
    def _fix_dict_recursive(d):
        """Recursively fix all tensor strings in a dictionary."""
        fixed = {}
        for key, value in d.items():
            if isinstance(value, dict):
                fixed[key] = _fix_dict_recursive(value)
            elif isinstance(value, str) and value.startswith("tensor("):
                fixed[key] = _parse_tensor_string(value)
            else:
                fixed[key] = value
        return fixed
    
    # Load the file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if there are any tensor strings to fix
    original_str = json.dumps(data)
    if "tensor(" not in original_str:
        print(f"No tensor strings found in {filepath}")
        return False
    
    # Fix the data
    fixed_data = _fix_dict_recursive(data)
    
    # Create backup
    backup_path = filepath + ".bak"
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Write fixed data
    with open(filepath, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"Fixed: {filepath} (backup created)")
    return True


def enrich_kfold_summary(experiment: str) -> bool:
    """
    Enrich kfold_summary.json with aggregated statistics for all metrics.
    
    Computes min/max/mean/std from fold_results for metrics like:
    val_loss, test_r2, test_x_r2, val_x_r2, etc.
    
    This ensures that update_experiments_manifest() can find the statistics
    it needs (e.g., val_loss min, test_r2 max).
    
    Args:
        experiment: Path to the experiment folder containing kfold_summary.json
        
    Returns:
        bool: True if file was modified, False if no changes needed
        
    Example:
        >>> enrich_kfold_summary("../experiments/single/local/my_experiment")
        Enriched: kfold_summary.json with 12 new statistics
    """
    import json
    
    filepath = join(experiment, "kfold_summary.json")
    
    if not exists(filepath):
        print(f"No kfold_summary.json found in {experiment}")
        return False
    
    # Load the file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    fold_results = data.get("fold_results", {})
    if not fold_results:
        print(f"No fold_results found in {filepath}")
        return False
    
    # Collect all numeric metrics from all folds
    all_metrics = {}  # metric_name -> list of values
    
    for fold_id, fold_data in fold_results.items():
        metrics = fold_data.get("metrics", {})
        for metric_name, value in metrics.items():
            # Skip non-numeric values (e.g., paths)
            if isinstance(value, (int, float)):
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
    
    if not all_metrics:
        print(f"No numeric metrics found in fold_results")
        return False
    
    # Compute statistics for each metric
    statistics = data.get("statistics", {})
    new_stats_count = 0
    
    for metric_name, values in all_metrics.items():
        # Skip if already computed
        if metric_name in statistics:
            continue
        
        if len(values) > 0:
            values_array = np.array(values)
            statistics[metric_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
            }
            new_stats_count += 1
    
    if new_stats_count == 0:
        print(f"No new statistics to add to {filepath}")
        return False
    
    # Update data
    data["statistics"] = statistics
    
    # Create backup
    backup_path = filepath + ".bak"
    with open(backup_path, 'w') as f:
        # Read original for backup
        pass
    with open(filepath, 'r') as f:
        original_content = f.read()
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    # Write enriched data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Enriched: {filepath} with {new_stats_count} new statistics")
    return True


# =============================================================================
# Manifest Functions
# =============================================================================

# Default manifest location
MANIFEST_PATH = join(root_path, "experiments", "experiments_manifest.csv")


def update_experiments_manifest(
    experiment: str, 
    manifest_path: str = None
) -> pd.DataFrame:
    """
    Update the experiments manifest CSV with metadata from an experiment.
    
    Extracts metadata from the experiment's config and kfold_summary.json,
    lists available evaluations, and updates/adds a row in the manifest.
    
    Args:
        experiment: Path to the experiment folder
        manifest_path: Path to manifest CSV. If None, uses default location
                      (experiments/experiments_manifest.csv)
        
    Returns:
        pd.DataFrame: The updated manifest DataFrame
        
    Manifest Columns:
        - exp_id: Experiment identifier (folder name)
        - path: Full path to experiment
        - dataset: Dataset name (from config)
        - architecture: Model architecture (from config)
        - attention: Self-attention type (from config)
        - cross_attention: Cross-attention type (from config)
        - best_val_loss: Best validation loss across folds
        - best_test_r2: Best test R² across folds
        - num_folds: Number of k-folds
        - last_evaluated: Timestamp of last evaluation
        - available_evals: List of available evaluation directories
        
    Example:
        >>> update_experiments_manifest("../experiments/single/local/my_experiment")
        Updated manifest: my_experiment
    """
    import json
    from datetime import datetime
    from os import listdir
    from os.path import isdir
    
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    
    # Extract experiment ID
    match = re.search(r'([^/\\]+)$', experiment)
    exp_id = match.group(1) if match else "unknown"
    
    # Initialize metadata with defaults
    metadata = {
        "exp_id": exp_id,
        "path": experiment,
        "dataset": None,
        "architecture": None,
        "attention": None,
        "cross_attention": None,
        # Training hyperparameters
        "max_epochs": None,
        "learning_rate": None,
        "batch_size": None,
        "optimizer": None,
        "k_fold": None,
        # Model architecture
        "d_model": None,
        "n_heads": None,
        "n_layers": None,
        # Regularization
        "gamma_entropy": None,
        "kappa_acyclic": None,
        "lambda_sparse": None,
        "lambda_sparse_cross": None,
        "lambda_hsic": None,
        "lambda_decisive": None,
        "lambda_decisive_cross": None,
        "lambda_tau": None,
        "lambda_l1_self_scores": None,
        "lambda_l1_cross_scores": None,
        "lambda_embed_l1": None,
        "lambda_entropy_self": None,
        "lambda_entropy_cross": None,
        "lambda_kl": None,
        "lambda_noise_prior": None,
        # Results
        "best_val_loss": None,
        "best_test_r2": None,
        "num_folds": None,
        # Test loss statistics across k-folds
        "test_loss_mean": None,
        "test_loss_std": None,
        "test_loss_best": None,
        "test_loss_worst": None,
        # DAG recovery metrics (statistics only - per-fold details are in dag_metrics.json)
        "soft_hamming_cross_best": None,
        "soft_hamming_cross_mean": None,
        "soft_hamming_cross_worst": None,
        "soft_hamming_self_best": None,
        "soft_hamming_self_mean": None,
        "soft_hamming_self_worst": None,
        "dag_confidence_cross": None,  # DAG consistency across folds (1=identical, 0=max disagreement)
        "dag_confidence_self": None,
        "dag_source": None,  # "phi" or "attention"
        # ATE (Average Treatment Effect) metrics from eval_interventions
        "ate_mean_abs_error": None,  # Mean absolute error: avg over interventions, variables, folds
        "ate_std_abs_error": None,   # Std of absolute errors across all comparisons
        "ate_median_abs_error": None,  # Median absolute error
        "ate_mean_rel_error": None,  # Mean relative error (where defined)
        "ate_n_comparisons": None,   # Number of intervention-variable-fold comparisons
        # HSIC (independence regularization)
        "final_hsic_mean": None,  # Mean HSIC across folds at final epoch
        "final_hsic_std": None,   # Std HSIC across folds
        # Training instability metrics (from eval_train_metrics)
        "val_loss_spike_ratio_mean": None,
        "val_loss_spike_ratio_std": None,
        "val_loss_cv_mean": None,
        "val_loss_cv_std": None,
        "val_loss_max_jump_mean": None,
        "val_loss_max_jump_std": None,
        "val_loss_trend_instability_mean": None,
        "val_loss_trend_instability_std": None,
        # Metadata
        "last_evaluated": datetime.now().isoformat(timespec="seconds"),
        "available_evals": "[]",
    }
    
    # Try to load config
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if config_files:
        config_path = join(experiment, config_files[0])
        try:
            config = OmegaConf.load(config_path)
            metadata["dataset"] = config.get("data", {}).get("dataset")
            metadata["architecture"] = config.get("model", {}).get("model_object")
            
            # Extract attention types (location varies by architecture)
            model_config = config.get("model", {})
            model_kwargs = model_config.get("kwargs", {})
            
            # Self-attention type
            if "self_attention" in model_config:
                metadata["attention"] = model_config.get("self_attention")
            elif "decoder_self_attention" in model_config:
                metadata["attention"] = model_config.get("decoder_self_attention")
            elif "dec_self_attention_type" in model_kwargs:
                metadata["attention"] = model_kwargs.get("dec_self_attention_type")
            
            # Cross-attention type
            if "cross_attention" in model_config:
                metadata["cross_attention"] = model_config.get("cross_attention")
            elif "decoder_cross_attention" in model_config:
                metadata["cross_attention"] = model_config.get("decoder_cross_attention")
            elif "dec_cross_attention_type" in model_kwargs:
                metadata["cross_attention"] = model_kwargs.get("dec_cross_attention_type")
            
            # Training hyperparameters
            training_config = config.get("training", {})
            metadata["max_epochs"] = training_config.get("max_epochs")
            metadata["learning_rate"] = training_config.get("lr")
            metadata["batch_size"] = training_config.get("batch_size")
            metadata["optimizer"] = training_config.get("optimizer")
            metadata["k_fold"] = training_config.get("k_fold")
            
            # Regularization parameters
            metadata["gamma_entropy"] = training_config.get("gamma")
            metadata["kappa_acyclic"] = training_config.get("kappa")
            metadata["lambda_sparse"] = training_config.get("lambda_sparse")
            metadata["lambda_sparse_cross"] = training_config.get("lambda_sparse_cross")
            metadata["lambda_hsic"] = training_config.get("lambda_hsic")
            metadata["lambda_decisive"] = training_config.get("lambda_decisive")
            metadata["lambda_decisive_cross"] = training_config.get("lambda_decisive_cross")
            metadata["lambda_tau"] = training_config.get("lambda_tau")
            metadata["lambda_l1_self_scores"] = training_config.get("lambda_l1_self_scores")
            metadata["lambda_l1_cross_scores"] = training_config.get("lambda_l1_cross_scores")
            metadata["lambda_embed_l1"] = training_config.get("lambda_embed_l1")
            metadata["lambda_entropy_self"] = training_config.get("lambda_entropy_self")
            metadata["lambda_entropy_cross"] = training_config.get("lambda_entropy_cross")
            metadata["lambda_kl"] = training_config.get("lambda_kl")
            metadata["lambda_noise_prior"] = training_config.get("lambda_noise_prior")
            
            # Model architecture details
            embed_dim = model_config.get("embed_dim", {})
            metadata["d_model"] = embed_dim.get("d_model") or model_kwargs.get("d_model")
            metadata["n_heads"] = model_kwargs.get("n_heads")
            # Number of layers (varies by architecture)
            metadata["n_layers"] = (
                model_kwargs.get("dec_layers") or 
                model_kwargs.get("num_decoder_layers") or
                model_kwargs.get("n_encoder_layers")
            )
            
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Try to load kfold_summary
    kfold_path = join(experiment, "kfold_summary.json")
    if exists(kfold_path):
        try:
            with open(kfold_path, 'r') as f:
                kfold_data = json.load(f)
            
            metadata["num_folds"] = kfold_data.get("total_folds")
            
            # Get best metrics from statistics or fold_results
            stats = kfold_data.get("statistics", {})
            if "val_loss" in stats:
                metadata["best_val_loss"] = stats["val_loss"].get("min")
            
            if "test_r2" in stats:
                metadata["best_test_r2"] = stats["test_r2"].get("max")
            elif "test_x_r2" in stats:
                metadata["best_test_r2"] = stats["test_x_r2"].get("max")
            
            # Load test loss statistics across k-folds
            if "test_loss" in stats:
                metadata["test_loss_mean"] = stats["test_loss"].get("mean")
                metadata["test_loss_std"] = stats["test_loss"].get("std")
                metadata["test_loss_best"] = stats["test_loss"].get("min")
                metadata["test_loss_worst"] = stats["test_loss"].get("max")
                print(f"Loaded test loss stats from kfold_summary")
                
        except Exception as e:
            print(f"Warning: Could not load kfold_summary: {e}")
    
    # List available evaluations
    eval_dir = join(experiment, "eval")
    if exists(eval_dir) and isdir(eval_dir):
        evals = [d for d in listdir(eval_dir) if isdir(join(eval_dir, d))]
        metadata["available_evals"] = json.dumps(evals)
    
    # Load DAG metrics from eval_attention_scores if available
    dag_metrics_path = join(experiment, "eval", "eval_attention_scores", "files", "dag_metrics.json")
    if exists(dag_metrics_path):
        try:
            with open(dag_metrics_path, 'r') as f:
                dag_metrics = json.load(f)
            
            # Extract soft Hamming metrics statistics from nested structure
            # New format: {"soft_hamming_cross": {"best": X, "mean": Y, "worst": Z, "std": W, "per_fold": {...}}}
            for key in ["soft_hamming_cross", "soft_hamming_self"]:
                if key in dag_metrics:
                    metric_data = dag_metrics[key]
                    # Check if it's the new nested format
                    if isinstance(metric_data, dict) and "best" in metric_data:
                        metadata[f"{key}_best"] = metric_data.get("best")
                        metadata[f"{key}_mean"] = metric_data.get("mean")
                        metadata[f"{key}_worst"] = metric_data.get("worst")
                    else:
                        # Legacy format: single value (backward compatibility)
                        metadata[f"{key}_mean"] = metric_data
            
            # Get the DAG source (phi or attention)
            if "soft_hamming_cross_source" in dag_metrics:
                metadata["dag_source"] = dag_metrics["soft_hamming_cross_source"]
            elif "soft_hamming_self_source" in dag_metrics:
                metadata["dag_source"] = dag_metrics["soft_hamming_self_source"]
            
            # Extract DAG confidence metrics
            for key in ["dag_confidence_cross", "dag_confidence_self"]:
                if key in dag_metrics:
                    metadata[key] = dag_metrics[key]
                
            print(f"Loaded DAG metrics from {dag_metrics_path}")
        except Exception as e:
            print(f"Warning: Could not load DAG metrics: {e}")
    
    # Load final HSIC from training metrics (eval_train_metrics output or kfold_summary)
    # First try kfold_summary statistics
    if exists(kfold_path):
        try:
            with open(kfold_path, 'r') as f:
                kfold_data = json.load(f)
            stats = kfold_data.get("statistics", {})
            
            # Check for HSIC in statistics (may be logged as val_hsic_reg or similar)
            for hsic_key in ["val_hsic_reg", "val_hsic", "hsic_reg", "hsic"]:
                if hsic_key in stats:
                    metadata["final_hsic_mean"] = stats[hsic_key].get("mean")
                    metadata["final_hsic_std"] = stats[hsic_key].get("std")
                    print(f"Loaded HSIC from kfold_summary: {hsic_key}")
                    break
        except Exception as e:
            print(f"Warning: Could not load HSIC from kfold_summary: {e}")
    
    # If HSIC not in kfold_summary, try to extract from training metrics CSV files
    if metadata["final_hsic_mean"] is None:
        try:
            # Find all k-fold directories and extract final epoch HSIC
            kfold_dirs = sorted([
                d for d in listdir(experiment) 
                if isdir(join(experiment, d)) and d.startswith('k_')
            ])
            
            final_hsic_values = []
            for kfold_dir in kfold_dirs:
                metrics_path = join(experiment, kfold_dir, 'logs', 'csv', 'version_0', 'metrics.csv')
                if exists(metrics_path):
                    df_metrics = pd.read_csv(metrics_path)
                    # Get final epoch HSIC (validation)
                    hsic_col = None
                    for col in ["val_hsic_reg", "val_hsic", "hsic_reg"]:
                        if col in df_metrics.columns:
                            hsic_col = col
                            break
                    if hsic_col:
                        # Get the last non-NaN value
                        final_hsic = df_metrics[hsic_col].dropna().iloc[-1] if not df_metrics[hsic_col].dropna().empty else None
                        if final_hsic is not None:
                            final_hsic_values.append(final_hsic)
            
            if final_hsic_values:
                metadata["final_hsic_mean"] = float(np.mean(final_hsic_values))
                metadata["final_hsic_std"] = float(np.std(final_hsic_values))
                print(f"Loaded HSIC from training metrics CSV: mean={metadata['final_hsic_mean']:.6f}")
        except Exception as e:
            print(f"Warning: Could not load HSIC from training metrics: {e}")
    
    # Load training instability metrics from eval_train_metrics output
    instability_path = join(experiment, "eval", "eval_train_metrics", "files", "training_instability.json")
    if exists(instability_path):
        try:
            with open(instability_path, 'r') as f:
                instability_data = json.load(f)
            
            metrics = instability_data.get("metrics", {})
            
            # Extract instability metrics
            for metric_name in ["spike_ratio", "cv", "max_jump", "trend_instability"]:
                if metric_name in metrics:
                    metric_data = metrics[metric_name]
                    mean_val = metric_data.get("mean")
                    std_val = metric_data.get("std")
                    
                    if mean_val is not None:
                        metadata[f"val_loss_{metric_name}_mean"] = mean_val
                    if std_val is not None:
                        metadata[f"val_loss_{metric_name}_std"] = std_val
            
            print(f"Loaded training instability from {instability_path}")
        except Exception as e:
            print(f"Warning: Could not load training instability: {e}")
    
    # Load ATE (Average Treatment Effect) metrics from eval_interventions output
    # First try the default intervention directory
    ate_metrics_path = join(experiment, "eval", "eval_do", "default", "files", "ate_metrics.json")
    if exists(ate_metrics_path):
        try:
            with open(ate_metrics_path, 'r') as f:
                ate_data = json.load(f)
            
            # Extract summary statistics
            summary = ate_data.get("summary", {})
            
            if summary:
                metadata["ate_mean_abs_error"] = summary.get("mean_absolute_error")
                metadata["ate_std_abs_error"] = summary.get("std_absolute_error")
                metadata["ate_median_abs_error"] = summary.get("median_absolute_error")
                metadata["ate_mean_rel_error"] = summary.get("mean_relative_error")
                metadata["ate_n_comparisons"] = summary.get("n_total_comparisons")
                
                if metadata["ate_mean_abs_error"] is not None:
                    print(f"Loaded ATE metrics from {ate_metrics_path}")
                    print(f"  ATE Mean Abs Error: {metadata['ate_mean_abs_error']:.4f}")
        except Exception as e:
            print(f"Warning: Could not load ATE metrics: {e}")
    
    # Load existing manifest or create new one
    if exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
    else:
        manifest_df = pd.DataFrame()
    
    # Update or append row
    if len(manifest_df) > 0 and exp_id in manifest_df["exp_id"].values:
        # Update existing row
        for col, val in metadata.items():
            manifest_df.loc[manifest_df["exp_id"] == exp_id, col] = val
        print(f"Updated manifest: {exp_id}")
    else:
        # Append new row
        new_row = pd.DataFrame([metadata])
        manifest_df = pd.concat([manifest_df, new_row], ignore_index=True)
        print(f"Added to manifest: {exp_id}")
    
    # Ensure directory exists and save
    makedirs(dirname(manifest_path), exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    
    return manifest_df


def load_experiments_manifest(manifest_path: str = None) -> pd.DataFrame:
    """
    Load the experiments manifest CSV.
    
    Args:
        manifest_path: Path to manifest CSV. If None, uses default location.
        
    Returns:
        pd.DataFrame: The manifest DataFrame, or empty DataFrame if not found.
        
    Example:
        >>> manifest = load_experiments_manifest()
        >>> # Filter by dataset
        >>> scm6_exps = manifest[manifest["dataset"] == "scm6"]
    """
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    
    if exists(manifest_path):
        return pd.read_csv(manifest_path)
    else:
        print(f"Manifest not found at {manifest_path}")
        return pd.DataFrame()


# =============================================================================
# Batch Manifest Update (Local Use)
# =============================================================================

def batch_update_manifest(
    experiments: List[str] = None,
    experiments_folders: List[str] = None,
    manifest_path: str = None,
) -> pd.DataFrame:
    """
    Update the experiments manifest for multiple experiments.
    
    This function is designed for local use after syncing experiments from the cluster.
    It reads existing evaluation results (generated by run_all_evaluations on the cluster)
    and updates the manifest CSV without recomputing evaluations.
    
    Args:
        experiments: List of individual experiment paths to update
        experiments_folders: List of folders containing experiments. All subdirectories
                            will be discovered and added to the experiments list.
        manifest_path: Path to manifest CSV. If None, uses default location
                      (experiments/experiments_manifest.csv)
        
    Returns:
        pd.DataFrame: The updated manifest DataFrame
        
    Example:
        >>> from notebooks.eval_funs import batch_update_manifest
        >>> 
        >>> # Update manifest for specific experiments
        >>> manifest = batch_update_manifest(
        ...     experiments=["../experiments/single/euler/my_exp_1", "../experiments/single/euler/my_exp_2"]
        ... )
        >>> 
        >>> # Update manifest for all experiments in folders
        >>> manifest = batch_update_manifest(
        ...     experiments_folders=["../experiments/single/euler", "../experiments/stage/euler"]
        ... )
    """
    from os import listdir
    from os.path import isdir
    
    # Initialize experiments list
    all_experiments = []
    
    if experiments:
        all_experiments.extend(experiments)
    
    # Discover experiments from folders
    if experiments_folders:
        for folder in experiments_folders:
            if exists(folder) and isdir(folder):
                for subdir in listdir(folder):
                    subdir_path = join(folder, subdir)
                    if isdir(subdir_path):
                        # Check if it looks like an experiment folder (has config or k_* folders)
                        contents = listdir(subdir_path)
                        has_config = any(f.startswith("config") and f.endswith(".yaml") for f in contents)
                        has_kfold = any(f.startswith("k_") for f in contents)
                        if has_config or has_kfold:
                            all_experiments.append(subdir_path)
    
    if not all_experiments:
        print("No experiments found to update.")
        return pd.DataFrame()
    
    print(f"\n{'='*60}")
    print(f"Batch updating manifest for {len(all_experiments)} experiments")
    print('='*60)
    
    # Update manifest for each experiment
    success_count = 0
    error_count = 0
    
    for exp in all_experiments:
        try:
            print(f"\n--- Updating: {exp} ---")
            update_experiments_manifest(exp, manifest_path)
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            error_count += 1
    
    # Load and return final manifest
    manifest_df = load_experiments_manifest(manifest_path)
    
    print(f"\n{'='*60}")
    print(f"Batch Update Summary:")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Total experiments in manifest: {len(manifest_df)}")
    print('='*60)
    
    return manifest_df

