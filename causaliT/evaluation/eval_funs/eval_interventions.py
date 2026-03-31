"""
ATE (Average Treatment Effect) evaluation for CausaliT experiments.

This module computes E[X | do(S=s)] predictions and compares them to ground-truth
causal effects from the SCM.

Key function:
- eval_ate: Evaluate ATE predictions using interventions from dataset metadata
"""

import json
from os.path import join, exists
from os import makedirs, listdir

import pandas as pd
from omegaconf import OmegaConf
from typing import Dict, List, Optional

# Import shared utilities
from .eval_utils import root_path, load_dataset_metadata

# Import from project modules
from causaliT.evaluation.predict import create_intervention_fn

# Import from local eval_funs modules
from .eval_lib import predict_from_experiment


# =============================================================================
# ATE Utility Functions
# =============================================================================

def load_ate_ground_truth(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """Load ATE ground truth from dataset directory."""
    ate_path = join(datadir_path, dataset_name, "ate_ground_truth.json")
    if exists(ate_path):
        with open(ate_path, 'r') as f:
            return json.load(f)
    return None


def load_normalization_stats(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """Load normalization statistics from dataset directory."""
    norm_path = join(datadir_path, dataset_name, "normalization.json")
    if exists(norm_path):
        with open(norm_path, 'r') as f:
            return json.load(f)
    return None


def get_interventions_from_ground_truth(ate_ground_truth: dict) -> Dict[str, List[float]]:
    """
    Extract intervention configuration from ate_ground_truth.json.
    
    The ate_ground_truth.json contains an 'interventions' key with structure:
    {
        "S1": {"values": [0.5], "type": "in_distribution", "role": "negative_control"},
        "S2": {"values": [-1.7], "type": "in_distribution", "role": "positive_control"},
        ...
    }
    
    Returns:
        Dict mapping source variable names to lists of intervention values.
        E.g., {"S1": [0.5], "S2": [-1.7], "S3": [-0.5, 1.0], "S5": [-0.8, 2.5]}
    """
    interventions_meta = ate_ground_truth.get("interventions", {})
    
    if not interventions_meta:
        # Fallback: infer from analytical/monte_carlo keys if no explicit interventions
        # Parse keys like "S1=0.5" -> {"S1": [0.5]}
        inferred = {}
        for method in ["analytical", "monte_carlo"]:
            if method in ate_ground_truth:
                for key in ate_ground_truth[method].keys():
                    if "=" in key:
                        var, val = key.split("=")
                        if var not in inferred:
                            inferred[var] = []
                        inferred[var].append(float(val))
        return inferred
    
    return {src: meta["values"] for src, meta in interventions_meta.items()}


def denormalize_value(normalized_value: float, norm_stats: dict, category: str = "input") -> float:
    """De-normalize a value using the normalization statistics."""
    if category not in norm_stats:
        return normalized_value
    
    stats = norm_stats[category]
    method = stats.get("method", "minmax")
    
    if method == "minmax":
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 1)
        return normalized_value * (max_val - min_val) + min_val
    elif method == "standardize":
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 1)
        return normalized_value * std_val + mean_val
    else:
        return normalized_value


def compute_ate_metrics(
    df: pd.DataFrame,
    ate_ground_truth: dict,
    norm_stats: dict,
    input_labels: List[str],
) -> pd.DataFrame:
    """
    Compute ATE metrics: model_ATE vs true_ATE
    
    ATE = E[X | do(S=s)] - E[X | do(S=0)]  (difference, not absolute)
    
    Args:
        df: DataFrame with predictions (columns: intervention, pred_feat_0, pos_idx, kfold)
        ate_ground_truth: Dict from ate_ground_truth.json (new format with 'ate' key)
        norm_stats: Dict from normalization.json
        input_labels: List of input variable names (e.g., ["X1", "X2", ...])
        
    Returns:
        DataFrame with ATE metrics per intervention × variable × fold
    """
    # Use analytical ground truth - handle both old and new format
    gt_method = "analytical" if "analytical" in ate_ground_truth else "monte_carlo"
    gt_data = ate_ground_truth.get(gt_method, {})
    
    # New format: gt_data has 'ate', 'baseline', 'treated' keys
    # Old format: gt_data directly maps intervention -> {var: value}
    if "ate" in gt_data:
        ground_truth_ate = gt_data["ate"]  # New format
    else:
        ground_truth_ate = gt_data  # Old format (backward compatibility)
    
    ate_records = []
    
    # Get baseline predictions
    df_baseline = df[df["intervention"] == "baseline"]
    
    # Get interventions (excluding baseline)
    interventions = [i for i in df["intervention"].unique() if i != "baseline"]
    
    for intervention in interventions:
        df_treated = df[df["intervention"] == intervention]
        
        for pos_idx in df_treated["pos_idx"].unique():
            # Map position to variable name
            var_name = input_labels[int(pos_idx)] if int(pos_idx) < len(input_labels) else f"X{pos_idx+1}"
            
            for kfold in df_treated["kfold"].unique():
                # Get treated predictions for this variable/fold
                mask_treated = (df_treated["pos_idx"] == pos_idx) & (df_treated["kfold"] == kfold)
                subset_treated = df_treated[mask_treated]["pred_feat_0"]
                
                # Get baseline predictions for this variable/fold
                mask_baseline = (df_baseline["pos_idx"] == pos_idx) & (df_baseline["kfold"] == kfold)
                subset_baseline = df_baseline[mask_baseline]["pred_feat_0"]
                
                if len(subset_treated) == 0 or len(subset_baseline) == 0:
                    continue
                
                # Compute model ATE = mean(treated) - mean(baseline)
                # De-normalize both before computing difference
                treated_mean_norm = float(subset_treated.mean())
                baseline_mean_norm = float(subset_baseline.mean())
                
                treated_mean_raw = denormalize_value(treated_mean_norm, norm_stats, "input")
                baseline_mean_raw = denormalize_value(baseline_mean_norm, norm_stats, "input")
                
                model_ate = treated_mean_raw - baseline_mean_raw
                
                # Ground truth ATE lookup
                true_ate = ground_truth_ate.get(intervention, {}).get(var_name)
                
                # Compute errors
                abs_error = abs(model_ate - true_ate) if true_ate is not None else None
                rel_error = abs_error / abs(true_ate) if (true_ate and abs(true_ate) > 1e-10) else None
                
                ate_records.append({
                    "intervention": intervention,
                    "variable": var_name,
                    "kfold": kfold,
                    "model_ate": model_ate,
                    "model_treated_raw": treated_mean_raw,
                    "model_baseline_raw": baseline_mean_raw,
                    "true_ate": true_ate,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "n_samples": len(subset_treated),
                })
    
    return pd.DataFrame(ate_records)


# =============================================================================
# Main Evaluation Function
# =============================================================================

def eval_ate(experiment: str) -> pd.DataFrame:
    """
    Evaluate ATE (Average Treatment Effect) for an experiment.
    
    Interventions are loaded from the dataset's ate_ground_truth.json file.
    This allows different datasets to define their own intervention configurations.
    
    Args:
        experiment: Path to the experiment folder
        
    Returns:
        DataFrame with ATE metrics per intervention × variable × fold
        
    Output Files:
        experiment/eval/eval_ate/files/ate_metrics.csv
        experiment/eval/eval_ate/files/ate_metrics.json
    """
    print(f"Evaluating ATE for: {experiment}")
    
    # =========================================================================
    # Load metadata
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")
    
    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)
    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'")
    
    print(f"  Dataset: {dataset_name}")
    
    # =========================================================================
    # Load ATE ground truth and normalization
    # =========================================================================
    ate_ground_truth = load_ate_ground_truth(datadir_path, dataset_name)
    norm_stats = load_normalization_stats(datadir_path, dataset_name)
    
    if ate_ground_truth is None:
        raise ValueError(f"ate_ground_truth.json not found for '{dataset_name}'")
    if norm_stats is None:
        raise ValueError(f"normalization.json not found for '{dataset_name}'")
    
    # =========================================================================
    # Setup output directories
    # =========================================================================
    eval_path = join(experiment, "eval", "eval_ate", "files")
    makedirs(eval_path, exist_ok=True)
    
    predictions_file = join(eval_path, "predictions.csv")
    ate_csv_file = join(eval_path, "ate_metrics.csv")
    ate_json_file = join(eval_path, "ate_metrics.json")
    
    # =========================================================================
    # Load interventions from ate_ground_truth.json
    # =========================================================================
    intervention_config = get_interventions_from_ground_truth(ate_ground_truth)
    
    if not intervention_config:
        raise ValueError(
            f"No interventions found in ate_ground_truth.json for '{dataset_name}'. "
            f"Ensure the 'interventions' key is present or regenerate the dataset."
        )
    
    source_labels = metadata["variable_info"].get("source_labels", [])
    var_idx_map = metadata.get("variable_index_map", {})
    input_labels = metadata["variable_info"].get("input_labels", [])
    
    # Build intervention functions from config
    interventions = []
    for src_var, values in intervention_config.items():
        src_idx = var_idx_map.get(src_var)
        if src_idx is not None:
            for val in values:
                interventions.append(
                    (create_intervention_fn(interventions={src_idx: val}), f"{src_var}={val}")
                )
    
    print(f"  Interventions (from dataset): {[label for _, label in interventions]}")
    
    # =========================================================================
    # Run predictions (or load cached)
    # =========================================================================
    if exists(predictions_file):
        print("  Loading cached predictions...")
        df = pd.read_csv(predictions_file)
    else:
        print("  Running predictions...")
        # Baseline
        df = predict_from_experiment(experiment, input_conditioning_fn=None)
        df["intervention"] = "baseline"
        
        # Interventions
        for do_fn, do_label in interventions:
            df_do = predict_from_experiment(experiment, input_conditioning_fn=do_fn)
            df_do["intervention"] = do_label
            df = pd.concat([df, df_do], axis=0)
        
        df.to_csv(predictions_file, index=False)
        print(f"  Saved predictions to {predictions_file}")
    
    # =========================================================================
    # Compute ATE metrics
    # =========================================================================
    df_ate = compute_ate_metrics(
        df=df,
        ate_ground_truth=ate_ground_truth,
        norm_stats=norm_stats,
        input_labels=input_labels,
    )
    
    # Save CSV
    df_ate.to_csv(ate_csv_file, index=False)
    
    # =========================================================================
    # Compute summary and save JSON
    # =========================================================================
    abs_errors = df_ate["abs_error"].dropna()
    rel_errors = df_ate["rel_error"].dropna()
    
    # Per-intervention-variable summary
    summary_records = []
    for intervention in df_ate["intervention"].unique():
        for variable in df_ate["variable"].unique():
            mask = (df_ate["intervention"] == intervention) & (df_ate["variable"] == variable)
            subset = df_ate[mask]
            if len(subset) == 0:
                continue
            
            abs_err = subset["abs_error"].dropna()
            rel_err = subset["rel_error"].dropna()
            
            summary_records.append({
                "intervention": intervention,
                "variable": variable,
                "true_ate": float(subset["true_ate"].iloc[0]) if subset["true_ate"].notna().any() else None,
                "model_ate_mean": float(subset["model_ate"].mean()),
                "model_ate_std": float(subset["model_ate"].std()),
                "abs_error_mean": float(abs_err.mean()) if len(abs_err) > 0 else None,
                "abs_error_std": float(abs_err.std()) if len(abs_err) > 0 else None,
                "rel_error_mean": float(rel_err.mean()) if len(rel_err) > 0 else None,
                "n_folds": len(subset),
            })
    
    ate_json = {
        "dataset": dataset_name,
        "interventions": list(intervention_config.keys()),
        "summary": {
            "mean_absolute_error": float(abs_errors.mean()) if len(abs_errors) > 0 else None,
            "std_absolute_error": float(abs_errors.std()) if len(abs_errors) > 0 else None,
            "median_absolute_error": float(abs_errors.median()) if len(abs_errors) > 0 else None,
            "mean_relative_error": float(rel_errors.mean()) if len(rel_errors) > 0 else None,
            "n_comparisons": len(df_ate),
        },
        "per_intervention_variable": summary_records,
    }
    
    with open(ate_json_file, 'w') as f:
        json.dump(ate_json, f, indent=2)
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n  === ATE Results ===")
    print(f"  Mean Absolute Error: {abs_errors.mean():.4f} ± {abs_errors.std():.4f}" if len(abs_errors) > 0 else "  No ATE computed")
    if len(rel_errors) > 0:
        print(f"  Mean Relative Error: {rel_errors.mean():.2%}")
    print(f"  Saved: {ate_csv_file}")
    print(f"  Saved: {ate_json_file}")
    
    return df_ate


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

def eval_interventions(experiment: str, **kwargs) -> pd.DataFrame:
    """Backward-compatible alias for eval_ate()."""
    return eval_ate(experiment)
