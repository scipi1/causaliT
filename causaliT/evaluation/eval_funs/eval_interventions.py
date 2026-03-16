"""
Intervention evaluation functions for CausaliT experiments.

This module provides functions for analyzing model predictions under causal interventions:
- eval_interventions: Evaluate model predictions under do-calculus interventions
"""

import re
import json
from os.path import join, exists
from os import makedirs, listdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from typing import List, Tuple, Optional

# Import shared utilities
from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _create_cline_template,
    load_dataset_metadata,
    DEFAULT_PLOT_FORMAT,
)

# Import from project modules
from causaliT.evaluation.predict import create_intervention_fn

# Import from local eval_funs modules (self-contained)
from .eval_lib import predict_from_experiment
import warnings


# =============================================================================
# ATE Ground Truth Loading and De-normalization Utilities
# =============================================================================

def load_ate_ground_truth(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """
    Load ATE ground truth from dataset directory.
    
    Args:
        datadir_path: Path to data directory (e.g., "data/")
        dataset_name: Name of dataset (e.g., "scm1_linear_gaussian")
        
    Returns:
        Dict containing ATE ground truth, or None if not found
    """
    ate_path = join(datadir_path, dataset_name, "ate_ground_truth.json")
    if exists(ate_path):
        with open(ate_path, 'r') as f:
            return json.load(f)
    return None


def load_normalization_stats(datadir_path: str, dataset_name: str) -> Optional[dict]:
    """
    Load normalization statistics from dataset directory.
    
    Args:
        datadir_path: Path to data directory
        dataset_name: Name of dataset
        
    Returns:
        Dict containing normalization stats, or None if not found
    """
    norm_path = join(datadir_path, dataset_name, "normalization.json")
    if exists(norm_path):
        with open(norm_path, 'r') as f:
            return json.load(f)
    return None


def denormalize_value(
    normalized_value: float,
    norm_stats: dict,
    category: str = "input"
) -> float:
    """
    De-normalize a value using the normalization statistics.
    
    Args:
        normalized_value: The normalized value to convert back
        norm_stats: Dict containing normalization parameters
        category: "input", "source", or "target"
        
    Returns:
        De-normalized value in original scale
    """
    if category not in norm_stats:
        return normalized_value
    
    stats = norm_stats[category]
    method = stats.get("method", "minmax")
    
    if method == "minmax":
        # MinMax: normalized = (x - min) / (max - min)
        # => x = normalized * (max - min) + min
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 1)
        return normalized_value * (max_val - min_val) + min_val
    elif method == "standardize":
        # Standard: normalized = (x - mean) / std
        # => x = normalized * std + mean
        mean_val = stats.get("mean", 0)
        std_val = stats.get("std", 1)
        return normalized_value * std_val + mean_val
    else:
        return normalized_value


def compute_ate_deviation(
    df: pd.DataFrame,
    ate_ground_truth: dict,
    norm_stats: dict,
    trg_feat_1_map: dict,
    input_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute ATE deviation: |E_model[X | do(S=s)] - E_SCM[X | do(S=s)]|
    
    This computes the error between the model's predicted interventional effect
    and the ground-truth causal effect from the SCM.
    
    Args:
        df: DataFrame with predictions (must have 'intervention', 'pred_feat_0', 
            'trg_feat_1', 'kfold' columns)
        ate_ground_truth: Dict from ate_ground_truth.json
        norm_stats: Dict from normalization.json
        trg_feat_1_map: Dict mapping variable indices to names
        input_labels: List of input variable names (e.g., ["X1", "X2"]) - used to
                     correctly map prediction positions to ground truth variables
        
    Returns:
        DataFrame with columns:
            - intervention: e.g., "S1=0"
            - variable: target variable name (e.g., "X1")
            - kfold: cross-validation fold
            - model_ate_normalized: model's mean prediction (normalized scale)
            - model_ate_raw: model's mean prediction (de-normalized to raw scale)
            - true_ate: ground-truth E[X | do(S=s)] from SCM
            - abs_error: |model_ate_raw - true_ate|
            - n_samples: number of samples used
    """
    # Use analytical ground truth by default
    gt_method = "analytical"
    if gt_method not in ate_ground_truth:
        gt_method = list(ate_ground_truth.keys())[0]
        if gt_method in ["description", "do_values_used", "computation_methods"]:
            gt_method = "analytical" if "analytical" in ate_ground_truth else "monte_carlo"
    
    ground_truth = ate_ground_truth.get(gt_method, {})
    
    ate_records = []
    
    # Get unique interventions (excluding baseline)
    interventions = [i for i in df["intervention"].unique() if i != "baseline"]
    
    # Build position-to-input_label mapping
    # The model predicts input variables (X) in sequence order
    # pos_idx 0 -> input_labels[0] (e.g., "X1")
    # pos_idx 1 -> input_labels[1] (e.g., "X2")
    pos_to_input_label = {}
    if input_labels:
        for pos_idx, label in enumerate(input_labels):
            pos_to_input_label[pos_idx] = label
    
    for intervention in interventions:
        # Filter for this intervention
        df_interv = df[df["intervention"] == intervention]
        
        # Get unique positions (pos_idx) to iterate over
        # Use pos_idx if available, otherwise fall back to trg_feat_1
        if "pos_idx" in df_interv.columns:
            unique_positions = df_interv["pos_idx"].unique()
            use_pos_idx = True
        else:
            unique_positions = df_interv["trg_feat_1"].unique()
            use_pos_idx = False
        
        for pos_or_idx in unique_positions:
            # Determine the variable name for ground truth lookup
            if use_pos_idx and input_labels and int(pos_or_idx) < len(input_labels):
                # Map position to input variable name
                trg_var_name = input_labels[int(pos_or_idx)]
            else:
                # Fall back to trg_feat_1_map
                if use_pos_idx:
                    # Get corresponding trg_feat_1 value
                    trg_feat_1_val = df_interv[df_interv["pos_idx"] == pos_or_idx]["trg_feat_1"].iloc[0]
                else:
                    trg_feat_1_val = pos_or_idx
                trg_var_name = trg_feat_1_map.get(int(trg_feat_1_val), str(trg_feat_1_val))
                
                # If the variable name is a source variable (S1, S2, etc.) but ground truth 
                # has input variables (X1, X2), try to map using input_labels by position
                if input_labels and trg_var_name not in ground_truth.get(intervention, {}):
                    pos = list(df_interv["trg_feat_1"].unique()).index(trg_feat_1_val) if trg_feat_1_val in df_interv["trg_feat_1"].unique() else -1
                    if 0 <= pos < len(input_labels):
                        trg_var_name = input_labels[pos]
            
            for kfold in df_interv["kfold"].unique():
                if use_pos_idx:
                    mask = (
                        (df_interv["pos_idx"] == pos_or_idx) &
                        (df_interv["kfold"] == kfold)
                    )
                else:
                    mask = (
                        (df_interv["trg_feat_1"] == pos_or_idx) &
                        (df_interv["kfold"] == kfold)
                    )
                subset = df_interv[mask]["pred_feat_0"]
                
                if len(subset) == 0:
                    continue
                
                # Model's mean prediction (normalized)
                model_ate_normalized = float(subset.mean())
                
                # De-normalize to raw scale
                model_ate_raw = denormalize_value(
                    model_ate_normalized, 
                    norm_stats, 
                    category="input"
                )
                
                # Look up ground truth
                # intervention key format: "S1=0" matches ground_truth["S1=0"]["X1"]
                true_ate = None
                if intervention in ground_truth:
                    true_ate = ground_truth[intervention].get(trg_var_name)
                
                # Compute error
                if true_ate is not None:
                    abs_error = abs(model_ate_raw - true_ate)
                    rel_error = abs_error / abs(true_ate) if abs(true_ate) > 1e-10 else None
                else:
                    abs_error = None
                    rel_error = None
                
                ate_records.append({
                    "intervention": intervention,
                    "variable": trg_var_name,
                    "pos_idx": int(pos_or_idx) if use_pos_idx else None,
                    "kfold": kfold,
                    "model_ate_normalized": model_ate_normalized,
                    "model_ate_raw": model_ate_raw,
                    "true_ate": true_ate,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "n_samples": len(subset),
                })
    
    return pd.DataFrame(ate_records)


# =============================================================================
# Evaluation Functions
# =============================================================================

def eval_interventions(
    experiment: str, 
    interventions: Optional[List[Tuple]] = None, 
    show_plots: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate model predictions under causal interventions (do-calculus).
    
    Computes predictions with and without interventions on source variables (S)
    to analyze causal effects. By default, tests interventions S1=0, S1=1, S1=-1,
    S2=0, S2=1, S2=-1, S3=0, S3=1, S3=-1.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        interventions: Optional list of (intervention_fn, label) tuples. 
                      Each intervention_fn should be created with create_intervention_fn().
                      If None, uses default interventions on S1, S2, S3 with values 0, 1, -1.
        show_plots: If True, display plots interactively. If False (default), 
                    only save to files.
        
    Returns:
        Tuple of (df, df_dev):
            - df: Raw predictions DataFrame with columns:
                - sample_idx, pos_idx: Sample and position indices
                - pred_feat_0: Predicted value
                - trg_feat_*: Target features
                - kfold, checkpoint_name: Fold and checkpoint info
                - intervention: Intervention label or "baseline"
            - df_dev: Deviations DataFrame with columns:
                - pred_feat_0: Difference (baseline - intervention)
                - intervention: Intervention label
                - Other grouping columns
        
    Output Files:
        - fig/dev_{var}_{k}_{exp_id}.pdf: Deviation histograms per variable and fold
        - files/do.csv: Raw predictions data
        
    Example:
        >>> # Default interventions
        >>> df, df_dev = eval_interventions("../experiments/my_experiment")
        >>> 
        >>> # Custom interventions
        >>> from causaliT.evaluation.predict import create_intervention_fn
        >>> custom = [
        ...     (create_intervention_fn(interventions={1: 0.5}), "S1=0.5"),
        ...     (create_intervention_fn(interventions={2: 0.5}), "S2=0.5"),
        ... ]
        >>> df, df_dev = eval_interventions("../experiments/my_experiment", interventions=custom)
    """
    # Extract experiment ID
    match = re.search(r'([^/\\]+)$', experiment)
    exp_id = match.group(1) if match else "unknown"
    
    # =========================================================================
    # Load dataset metadata for dynamic configuration
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_name = None
    metadata = None
    
    if config_files:
        try:
            config = OmegaConf.load(join(experiment, config_files[0]))
            dataset_name = config.get("data", {}).get("dataset")
        except Exception:
            pass
    
    if dataset_name:
        datadir_path = join(root_path, "data")
        metadata = load_dataset_metadata(datadir_path, dataset_name)
        if metadata:
            print(f"Loaded metadata for dataset: {dataset_name}")
    
    # =========================================================================
    # Build trg_feat_1_map from metadata (NO FALLBACK - requires metadata)
    # =========================================================================
    if not metadata or "variable_index_map" not in metadata:
        raise ValueError(
            f"Dataset metadata not found for '{dataset_name}'. "
            f"Ensure dataset_metadata.json exists in data/{dataset_name}/"
        )
    
    # Invert the map: {name: idx} -> {idx: name}
    trg_feat_1_map = {v: k for k, v in metadata["variable_index_map"].items()}
    print(f"  Variable map loaded from metadata: {list(trg_feat_1_map.values())}")
    
    # =========================================================================
    # Setup default interventions from metadata or fallback
    # =========================================================================
    if interventions is None:
        if not metadata or "variable_info" not in metadata:
            raise ValueError(
                f"Cannot generate interventions: metadata missing for '{dataset_name}'. "
                f"Ensure dataset_metadata.json exists in data/{dataset_name}/"
            )
        
        # Generate interventions dynamically from source_labels
        source_labels = metadata["variable_info"].get("source_labels", [])
        var_idx_map = metadata.get("variable_index_map", {})
        interventions = []
        for src_var in source_labels:
            src_idx = var_idx_map.get(src_var)
            if src_idx is not None:
                for val in [0, 1, -1]:
                    interventions.append(
                        (create_intervention_fn(interventions={src_idx: val}), f"{src_var}={val}")
                    )
        print(f"  Generated {len(interventions)} interventions from metadata")
        label_dir = "default"
    else:
        do_labels = [tup[-1] for tup in interventions]
        label_dir = "_".join(do_labels)
        
    # Setup directories with intervention-specific subfolder
    eval_path_root = join(experiment, "eval", "eval_do", label_dir)
    eval_path_fig = join(eval_path_root, "fig")
    eval_path_files = join(eval_path_root, "files")
    eval_path_cline = join(eval_path_root, "cline")

    makedirs(eval_path_fig, exist_ok=True)
    makedirs(eval_path_files, exist_ok=True)
    makedirs(eval_path_cline, exist_ok=True)
    
    do_filename = "do.csv"
    summary_filename = "do_summary.csv"
    variable_labels_filename = "variable_labels.json"
    
    # Save variable labels to JSON for AI/programmatic access
    variable_labels = {
        "trg_feat_1_to_name": trg_feat_1_map,
        "description": "Maps numeric trg_feat_1 values to variable names",
    }
    with open(join(eval_path_files, variable_labels_filename), 'w') as f:
        json.dump(variable_labels, f, indent=2)
    
    # Save README with column documentation
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation folder contains predictions of the model under interventions",
        files_info={
            "do.csv": "Raw predictions for all samples × interventions × folds",
            "do_summary.csv": "Aggregated statistics (mean, std) of deviations per intervention × variable",
            "variable_labels.json": "Mapping from numeric trg_feat_1 to variable names (S1, X1, etc.)",
        },
        column_documentation={
            "sample_idx": "Sample index in dataset",
            "pos_idx": "Position index within sequence (corresponds to variable)",
            "pred_feat_0": "Predicted value",
            "trg_feat_0": "Target value (ground truth)",
            "trg_feat_1": "Variable ID (see variable_labels.json for mapping)",
            "kfold": "Cross-validation fold identifier",
            "checkpoint_name": "Model checkpoint file used",
            "intervention": "Intervention applied ('baseline' or 'S1=0', 'S2=1', etc.)",
        }
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_interventions", exp_id)
    
    # Load or compute predictions
    if exists(join(eval_path_files, do_filename)):
        df = pd.read_csv(join(eval_path_files, do_filename))
        print("Experiment already available. Data loaded!")
    else:
        # Predict baseline
        df = predict_from_experiment(experiment, input_conditioning_fn=None)
        df["intervention"] = "baseline"
    
        # Predict with each intervention
        for do_fn, do_label in interventions:
            df_do = predict_from_experiment(experiment, input_conditioning_fn=do_fn)
            df_do["intervention"] = do_label
            df = pd.concat([df, df_do], axis=0)

        df.to_csv(join(eval_path_files, do_filename))
        print("Data saved!")
        
    # Calculate deviations from baseline
    group = ["intervention", "sample_idx", "pos_idx", "trg_feat_1", "kfold", "checkpoint_name"]
    pred_label = "pred_feat_0"
    do_labels = df["intervention"].unique().tolist()
    do_labels.remove("baseline")

    df_do_list = []
    for do in do_labels:
        df_do = (
            df.set_index(group).loc["baseline"][pred_label] - 
            df.set_index(group).loc[do][pred_label]
        ).reset_index()
        df_do["intervention"] = do
        df_do_list.append(df_do)

    df_dev = pd.concat(df_do_list, axis=0)
    
    # Add variable name column for readability
    df_dev["variable"] = df_dev["trg_feat_1"].map(trg_feat_1_map)
    
    # Create summary CSV with aggregated statistics (small, AI-readable)
    summary_records = []
    for intervention in df_dev["intervention"].unique():
        for trg_var in df_dev["trg_feat_1"].unique():
            for kfold in df_dev["kfold"].unique():
                mask = (
                    (df_dev["intervention"] == intervention) & 
                    (df_dev["trg_feat_1"] == trg_var) & 
                    (df_dev["kfold"] == kfold)
                )
                subset = df_dev[mask]["pred_feat_0"]
                summary_records.append({
                    "intervention": intervention,
                    "trg_feat_1": trg_var,
                    "variable": trg_feat_1_map.get(int(trg_var), str(trg_var)),
                    "kfold": kfold,
                    "deviation_mean": subset.mean(),
                    "deviation_std": subset.std(),
                    "deviation_median": subset.median(),
                    "deviation_min": subset.min(),
                    "deviation_max": subset.max(),
                    "n_samples": len(subset),
                })
    
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(join(eval_path_files, summary_filename), index=False)
    print(f"Saved summary: {summary_filename}")
    
    # =========================================================================
    # Intervention Invariance Test (H1: non-causal interventions should have ~0 effect)
    # =========================================================================
    invariance_filename = "intervention_invariance.json"
    invariance_threshold = 0.05  # Threshold for "no effect" (mean abs deviation)
    
    # Build expected_effects from metadata or use fallback
    expected_effects = None
    
    if metadata and "causal_structure" in metadata and "expected_effects" in metadata["causal_structure"]:
        # Convert metadata format: {src: {tgt: bool}} -> {(src, tgt): bool}
        expected_effects_raw = metadata["causal_structure"]["expected_effects"]
        expected_effects = {}
        for src_var, targets in expected_effects_raw.items():
            for tgt_var, has_effect in targets.items():
                expected_effects[(src_var, tgt_var)] = has_effect
        print(f"  Loaded expected effects from metadata: {len(expected_effects)} pairs")
    
    if expected_effects:
        
        # Compute invariance test results
        invariance_tests = []
        
        for intervention in df_summary["intervention"].unique():
            # Parse intervention variable (e.g., "S1=0" → "S1")
            interv_var = intervention.split("=")[0]
            
            for _, row in df_summary[df_summary["intervention"] == intervention].iterrows():
                target_var = row["variable"]
                deviation_mean = row["deviation_mean"]
                kfold = row["kfold"]
                
                # Look up expected effect
                key = (interv_var, target_var)
                if key in expected_effects:
                    expected_effect = expected_effects[key]
                    actual_deviation = abs(deviation_mean)
                    
                    # Test passes if:
                    # - Expected no effect AND actual deviation < threshold
                    # - OR expected effect AND actual deviation > threshold
                    if expected_effect:
                        passed = actual_deviation > invariance_threshold
                    else:
                        passed = actual_deviation < invariance_threshold
                    
                    invariance_tests.append({
                        "intervention": intervention,
                        "intervention_var": interv_var,
                        "target_var": target_var,
                        "kfold": kfold,
                        "expected_effect": expected_effect,
                        "actual_deviation_mean": float(deviation_mean),
                        "actual_deviation_abs": float(actual_deviation),
                        "passed": passed,
                    })
        
        # Aggregate results
        if invariance_tests:
            df_inv = pd.DataFrame(invariance_tests)
            
            # Summary statistics
            total_tests = len(df_inv)
            passed_tests = df_inv["passed"].sum()
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            # Group by expected_effect=False (invariance tests)
            invariance_only = df_inv[df_inv["expected_effect"] == False]
            invariance_pass_rate = invariance_only["passed"].mean() if len(invariance_only) > 0 else None
            
            invariance_summary = {
                "dataset": dataset_name,
                "threshold": invariance_threshold,
                "total_tests": total_tests,
                "passed_tests": int(passed_tests),
                "overall_pass_rate": float(pass_rate),
                "invariance_tests_count": len(invariance_only),
                "invariance_pass_rate": float(invariance_pass_rate) if invariance_pass_rate is not None else None,
                "tests": invariance_tests,
            }
            
            with open(join(eval_path_files, invariance_filename), 'w') as f:
                json.dump(invariance_summary, f, indent=2)
            print(f"Saved intervention invariance: {invariance_filename}")
            print(f"  Invariance pass rate: {invariance_pass_rate:.2%}" if invariance_pass_rate else "  No invariance tests")
    else:
        print(f"  Skipping invariance test: no expected effects defined for dataset '{dataset_name}'")

    # =========================================================================
    # ATE Ground Truth Comparison (Publication-Ready Metric)
    # =========================================================================
    ate_metrics_filename = "ate_metrics.csv"
    ate_metrics_json_filename = "ate_metrics.json"
    
    # Load ATE ground truth and normalization stats
    datadir_path = join(root_path, "data")
    ate_ground_truth = load_ate_ground_truth(datadir_path, dataset_name)
    norm_stats = load_normalization_stats(datadir_path, dataset_name)
    
    if ate_ground_truth is None:
        warnings.warn(
            f"ATE ground truth not found for dataset '{dataset_name}'. "
            f"Please regenerate the dataset to include ate_ground_truth.json. "
            f"Run: SCMDataset.generate_ds() to create the ground truth file.",
            UserWarning
        )
        print(f"  Skipping ATE metrics: ate_ground_truth.json not found for '{dataset_name}'")
    elif norm_stats is None:
        warnings.warn(
            f"Normalization stats not found for dataset '{dataset_name}'. "
            f"ATE comparison requires normalization.json to de-normalize predictions.",
            UserWarning
        )
        print(f"  Skipping ATE metrics: normalization.json not found for '{dataset_name}'")
    else:
        # Get input_labels from metadata for correct variable mapping
        input_labels = metadata.get("variable_info", {}).get("input_labels", []) if metadata else []
        
        # Compute ATE deviation metrics
        df_ate = compute_ate_deviation(
            df=df,
            ate_ground_truth=ate_ground_truth,
            norm_stats=norm_stats,
            trg_feat_1_map=trg_feat_1_map,
            input_labels=input_labels,
        )
        
        if len(df_ate) > 0:
            # Save to CSV (detailed per-fold, per-intervention, per-variable)
            df_ate.to_csv(join(eval_path_files, ate_metrics_filename), index=False)
            
            # Compute summary statistics for JSON export
            ate_summary_records = []
            for intervention in df_ate["intervention"].unique():
                for variable in df_ate["variable"].unique():
                    mask = (
                        (df_ate["intervention"] == intervention) &
                        (df_ate["variable"] == variable)
                    )
                    subset = df_ate[mask]
                    
                    if len(subset) == 0:
                        continue
                    
                    # Aggregate across folds
                    abs_errors = subset["abs_error"].dropna()
                    rel_errors = subset["rel_error"].dropna()
                    
                    ate_summary_records.append({
                        "intervention": intervention,
                        "variable": variable,
                        "true_ate": float(subset["true_ate"].iloc[0]) if subset["true_ate"].notna().any() else None,
                        "model_ate_raw_mean": float(subset["model_ate_raw"].mean()),
                        "model_ate_raw_std": float(subset["model_ate_raw"].std()),
                        "abs_error_mean": float(abs_errors.mean()) if len(abs_errors) > 0 else None,
                        "abs_error_std": float(abs_errors.std()) if len(abs_errors) > 0 else None,
                        "rel_error_mean": float(rel_errors.mean()) if len(rel_errors) > 0 else None,
                        "n_folds": len(subset),
                    })
            
            # Overall summary metrics (averaged across all interventions and variables)
            all_abs_errors = df_ate["abs_error"].dropna()
            all_rel_errors = df_ate["rel_error"].dropna()
            
            ate_json = {
                "description": "ATE (Average Treatment Effect) deviation metrics comparing model predictions to SCM ground truth",
                "dataset": dataset_name,
                "computation_method": "Model predictions de-normalized and compared to analytical SCM E[X | do(S=s)]",
                "summary": {
                    "mean_absolute_error": float(all_abs_errors.mean()) if len(all_abs_errors) > 0 else None,
                    "std_absolute_error": float(all_abs_errors.std()) if len(all_abs_errors) > 0 else None,
                    "median_absolute_error": float(all_abs_errors.median()) if len(all_abs_errors) > 0 else None,
                    "mean_relative_error": float(all_rel_errors.mean()) if len(all_rel_errors) > 0 else None,
                    "n_intervention_variable_pairs": len(ate_summary_records),
                    "n_total_comparisons": len(df_ate),
                },
                "per_intervention_variable": ate_summary_records,
            }
            
            with open(join(eval_path_files, ate_metrics_json_filename), 'w') as f:
                json.dump(ate_json, f, indent=2)
            
            print(f"Saved ATE metrics: {ate_metrics_filename}")
            if all_abs_errors.notna().any():
                print(f"  Mean Absolute Error: {all_abs_errors.mean():.4f} ± {all_abs_errors.std():.4f}")
                if all_rel_errors.notna().any():
                    print(f"  Mean Relative Error: {all_rel_errors.mean():.2%}")
        else:
            print("  No ATE metrics computed (empty results)")

    # Generate plots
    for k in df_dev["kfold"].unique():
        for trg_var in df_dev["trg_feat_1"].unique():
            df_hist = df_dev.set_index(["kfold", "trg_feat_1"]).loc[k].loc[trg_var]

            fig, ax = plt.subplots()
            sns.histplot(
                data=df_hist, 
                x="pred_feat_0", 
                hue="intervention", 
                ax=ax, 
                stat="density", 
                multiple="stack", 
                bins=50
            )

            var_label = trg_feat_1_map.get(int(trg_var), str(trg_var))
            ax.set_title(r"Variable $\mathcal{Y}= $" + f"{var_label}, fold={k}")
            ax.set_xlabel(r"$\mathbb{E}[\mathcal{Y} | S:=s]$")

            plt.savefig(join(eval_path_fig, f"dev_{var_label}_{k}_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    return df, df_dev
