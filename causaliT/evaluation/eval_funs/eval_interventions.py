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
