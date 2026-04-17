"""
Attention score and DAG recovery evaluation functions for CausaliT experiments.

This module provides two main evaluation functions:

1. eval_attention_scores (FAST - always run):
   - Loads attention from best checkpoint only
   - Computes DAG recovery metrics (soft Hamming, MEC)
   - Plots attention heatmaps

2. eval_attention_evolution (SLOW - optional):
   - Loads attention from multiple checkpoints
   - Tracks how phi/attention evolves during training
   - Plots attention drift over epochs
"""

import json
from os.path import join, exists, isdir
from os import listdir
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Import shared utilities
from .eval_utils import (
    root_path,
    _setup_eval_directories,
    _save_readme,
    _save_variable_labels,
    _create_cline_template,
    find_all_checkpoints,
    _select_evenly_spaced_checkpoints,
    _compute_soft_hamming,
    _load_true_dag_mask,
    _compute_dag_confidence,
    _get_learned_dag_per_fold,
    load_dataset_metadata,
    get_architecture_config,
    ARCHITECTURE_REGISTRY,
    DEFAULT_PLOT_FORMAT,
    # MEC metrics
    _combine_attention_to_full_dag,
    _load_full_true_dag,
    _compute_mec_distance,
    _check_mec_membership,
    _find_v_structures,
    _dag_to_skeleton,
)

# Import from project modules
from causaliT.evaluation.predict import predict_test_from_ckpt
from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
from causaliT.training.forecasters.single_causal_forecaster import SingleCausalForecaster
from causaliT.training.forecasters.noise_aware_forecaster import NoiseAwareCausalForecaster

# Import from local eval_funs modules (self-contained)
from .eval_lib import (
    load_attention_data,
    load_attention_data_from_file,
    save_attention_data,
    find_config_file,
    get_architecture_type,
    extract_phi_from_model,
)
from .eval_plot_lib import plot_attention_scores, plot_attention_evolution


# =============================================================================
# Model Loading Helper
# =============================================================================

def _load_model_from_checkpoint(checkpoint_path: str, architecture_type: str):
    """Load model from checkpoint based on architecture type."""
    if architecture_type == "TransformerForecaster":
        return TransformerForecaster.load_from_checkpoint(checkpoint_path)
    elif architecture_type == "StageCausalForecaster":
        return StageCausalForecaster.load_from_checkpoint(checkpoint_path)
    elif architecture_type == "SingleCausalForecaster":
        return SingleCausalForecaster.load_from_checkpoint(checkpoint_path)
    elif architecture_type == "NoiseAwareCausalForecaster":
        return NoiseAwareCausalForecaster.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


# =============================================================================
# FAST: Final Attention Analysis (from best checkpoint)
# =============================================================================

def eval_attention_scores(experiment: str, show_plots: bool = True) -> dict:
    """
    Evaluate final attention scores and DAG recovery metrics.
    
    FAST: Only loads best checkpoint from each fold (~10-30 seconds).
    For evolution tracking across epochs, use eval_attention_evolution().
    
    This function:
    - Loads attention weights from the best checkpoint of each k-fold
    - Computes DAG recovery metrics (soft Hamming, MEC distance)
    - Plots attention score heatmaps and DAG comparisons
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        dict: DAG recovery metrics with keys:
            - soft_hamming_cross: Soft Hamming distance for S→X edges (best/mean/worst/std/per_fold)
            - soft_hamming_self: Soft Hamming distance for X→X edges (best/mean/worst/std/per_fold)
            - dag_confidence_cross: DAG consistency across folds for S→X (1=identical, 0=max disagreement)
            - dag_confidence_self: DAG consistency across folds for X→X (1=identical, 0=max disagreement)
            - mec_distance: MEC distance metrics (if computable)
        
    Output Files:
        - fig/attention_scores_{exp_id}.png: Attention score heatmaps for all folds
        - fig/dag_comparison_{fold}_{exp_id}.png: Learned vs true DAG comparison heatmaps
        - files/final_scores/: Saved attention data (can be reloaded quickly)
        - files/dag_metrics.json: DAG recovery metrics (soft Hamming + MEC + dag_confidence)
        - files/attention_labels.json: Descriptions of attention blocks
        
    Example:
        >>> metrics = eval_attention_scores("experiments/single/local/my_experiment")
        >>> print(f"Soft Hamming (cross): {metrics['soft_hamming_cross']['mean']:.4f}")
        >>> print(f"DAG Confidence (cross): {metrics['dag_confidence_cross']:.4f}")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_attention_scores")

    final_scores_dirname = "final_scores"
    dag_metrics_filename = "dag_metrics.json"
    attention_labels_filename = "attention_labels.json"

    print(f"Experiment ID: {exp_id}")
    
    # =========================================================================
    # Load dataset metadata for variable mappings
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        raise ValueError(f"No config file found in {experiment}")
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset_name = config.get("data", {}).get("dataset")
    
    if not dataset_name:
        raise ValueError("No dataset specified in experiment config.")
    
    datadir_path = join(root_path, "data")
    metadata = load_dataset_metadata(datadir_path, dataset_name)
    
    if not metadata:
        raise ValueError(f"Dataset metadata not found for '{dataset_name}'.")
    
    print(f"  Dataset: {dataset_name}")
    
    # =========================================================================
    # Build attention labels for AI interpretation
    # =========================================================================
    attention_labels = {
        "description": "Attention weights and DAG (phi) structure learned by the model",
        "attention_blocks": {arch: cfg for arch, cfg in [
            (arch, {
                k: f"Attention block: {k}" for k in config_data["attention_keys"]
            }) for arch, config_data in ARCHITECTURE_REGISTRY.items()
        ]},
        "phi_tensors": {
            "description": "Learned DAG edge probabilities (sigmoid(phi)). Values in [0,1] where 1 = edge present.",
        },
        "dag_metrics": {
            "soft_hamming": "Mean absolute difference between learned and true DAG. 0 = perfect, 1 = inverted",
        },
        "dataset": dataset_name,
    }
    
    # Add variable mapping from metadata
    if "variable_descriptions" in metadata:
        attention_labels["variable_mapping"] = metadata["variable_descriptions"]
    if "causal_structure" in metadata and "edges" in metadata["causal_structure"]:
        edges = metadata["causal_structure"]["edges"]
        edge_strs = [f"{src}→{tgt}" for src, tgt in edges]
        attention_labels["dag_structure"] = ", ".join(edge_strs)
    
    _save_variable_labels(eval_path_files, attention_labels, attention_labels_filename)

    # Save README
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Attention scores evaluation (FAST): final checkpoint analysis and DAG recovery metrics.",
        files_info={
            final_scores_dirname: "Saved attention data (npz files) for fast reloading",
            dag_metrics_filename: "Soft Hamming distance and MEC metrics comparing learned DAG to true DAG (JSON)",
            attention_labels_filename: "Descriptions of attention blocks and interpretation guide (JSON)",
        },
    )
    
    _create_cline_template(eval_path_cline, "eval_attention_scores", exp_id)
    
    # =========================================================================
    # Load or compute final attention scores
    # =========================================================================
    if exists(join(eval_path_files, final_scores_dirname)):
        final_scores_dict = load_attention_data_from_file(join(eval_path_files, final_scores_dirname))
        print("  Loaded cached attention data.")
    else:
        final_scores_dict = load_attention_data(experiment)
        save_attention_data(final_scores_dict, join(eval_path_files, final_scores_dirname), save_predictions=True)
        print("  Computed and saved attention data.")
    
    # Plot attention score heatmaps
    fig = plot_attention_scores(final_scores_dict, cmap='viridis', annotation_fontsize=8, scale_mode="row")
    plt.savefig(join(eval_path_fig, f"attention_scores_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # =========================================================================
    # Compute DAG Recovery Metrics
    # =========================================================================
    print("\n--- Computing DAG Recovery Metrics ---")
    
    architecture = final_scores_dict.architecture_type
    
    try:
        arch_config = get_architecture_config(architecture)
        blocks_to_eval = list(arch_config["blocks_to_eval"])  # copy to allow extension
        mec_keys = arch_config["mec_keys"]
    except ValueError:
        print(f"Warning: Unknown architecture {architecture}, skipping DAG metrics")
        return {}
    
    # Extend blocks_to_eval with per-layer entries when multi-layer data is available
    if architecture in ("SingleCausalForecaster", "NoiseAwareCausalForecaster"):
        import re as _re
        layer_phi_self = sorted(
            [k for k in final_scores_dict.phi_tensors.keys() if _re.match(r'^decoder_L\d+$', k)],
            key=lambda k: int(_re.search(r'L(\d+)', k).group(1))
        )
        layer_phi_cross = sorted(
            [k for k in final_scores_dict.phi_tensors.keys() if _re.match(r'^decoder_cross_L\d+$', k)],
            key=lambda k: int(_re.search(r'L(\d+)', k).group(1))
        )
        layer_att_self = sorted(
            [k for k in final_scores_dict.attention_weights.keys() if _re.match(r'^dec_self_L\d+$', k)],
            key=lambda k: int(_re.search(r'L(\d+)', k).group(1))
        )
        layer_att_cross = sorted(
            [k for k in final_scores_dict.attention_weights.keys() if _re.match(r'^dec_cross_L\d+$', k)],
            key=lambda k: int(_re.search(r'L(\d+)', k).group(1))
        )
        
        has_multi_layer = len(layer_phi_self) > 1 or len(layer_phi_cross) > 1
        
        if has_multi_layer:
            # Replace backward-compat blocks with per-layer blocks
            blocks_to_eval = []
            for att_key, phi_key in zip(layer_att_cross, layer_phi_cross):
                blocks_to_eval.append((att_key, phi_key, "dec_cross"))
            for att_key, phi_key in zip(layer_att_self, layer_phi_self):
                blocks_to_eval.append((att_key, phi_key, "dec_self"))
            print(f"  Multi-layer: evaluating {len(blocks_to_eval)} per-layer blocks")
    
    dag_metrics = {
        "dataset": dataset_name,
        "architecture": architecture,
    }
    
    # Store per-fold comparison data for plotting
    per_fold_comparison_data = []
    
    for att_key, phi_key, mask_type in blocks_to_eval:
        print(f"  Evaluating {att_key}...")
        
        # Get learned DAG for each fold
        fold_dags, source = _get_learned_dag_per_fold(final_scores_dict, att_key, phi_key)
        
        if all(dag is None for _, dag in fold_dags):
            print(f"    No data available for {att_key}")
            continue
        
        # Load true DAG mask
        true_dag = _load_true_dag_mask(datadir_path, dataset_name, mask_type)
        
        if true_dag is None:
            print(f"    No true DAG mask found for {mask_type}")
            continue
        
        # Compute per-fold soft Hamming distances
        per_fold_values = {}
        fold_sh_list = []
        
        for fold_name, learned_dag in fold_dags:
            if learned_dag is None or learned_dag.shape != true_dag.shape:
                per_fold_values[fold_name] = None
                continue
            
            soft_hamming = _compute_soft_hamming(learned_dag, true_dag)
            per_fold_values[fold_name] = soft_hamming
            fold_sh_list.append(soft_hamming)
            
            print(f"    {fold_name}: Soft Hamming ({source}) = {soft_hamming:.4f}")
            
            per_fold_comparison_data.append({
                "fold_name": fold_name,
                "block": att_key,
                "learned": learned_dag,
                "true": true_dag,
                "soft_hamming": soft_hamming,
                "source": source,
            })
        
        # Compute statistics
        if fold_sh_list:
            fold_sh_array = np.array(fold_sh_list)
            # Derive metric key from mask_type (remove prefixes)
            metric_key = f"soft_hamming_{mask_type.replace('dec_', '').replace('dec1_', '').replace('dec2_', '')}"
            
            dag_metrics[metric_key] = {
                "best": float(np.min(fold_sh_array)),
                "mean": float(np.mean(fold_sh_array)),
                "worst": float(np.max(fold_sh_array)),
                "std": float(np.std(fold_sh_array)),
                "per_fold": per_fold_values,
            }
            dag_metrics[f"{metric_key}_source"] = source
        
        # Compute DAG confidence
        valid_fold_dags = [dag for _, dag in fold_dags if dag is not None]
        if len(valid_fold_dags) >= 2:
            confidence_key = f"dag_confidence_{mask_type.replace('dec_', '').replace('dec1_', '').replace('dec2_', '')}"
            dag_metrics[confidence_key] = _compute_dag_confidence(valid_fold_dags)
            print(f"    DAG Confidence: {dag_metrics[confidence_key]:.4f}")
    
    # =========================================================================
    # Compute MEC Metrics
    # =========================================================================
    print("\n--- Computing MEC Metrics ---")
    
    cross_att_key, cross_phi_key = mec_keys["cross"]
    self_att_key, self_phi_key = mec_keys["self"]
    
    cross_fold_dags, _ = _get_learned_dag_per_fold(final_scores_dict, cross_att_key, cross_phi_key)
    self_fold_dags, _ = _get_learned_dag_per_fold(final_scores_dict, self_att_key, self_phi_key)
    
    true_full_dag = _load_full_true_dag(datadir_path, dataset_name)
    
    if true_full_dag is not None:
        true_skeleton = _dag_to_skeleton(true_full_dag)
        true_v_structures = _find_v_structures(true_full_dag)
        print(f"  True DAG: {len(true_skeleton)} edges, {len(true_v_structures)} v-structures")
        
        mec_per_fold = {}
        mec_distances = []
        mec_memberships = []
        
        for i, (fold_name, cross_dag) in enumerate(cross_fold_dags):
            if i >= len(self_fold_dags):
                continue
            _, self_dag = self_fold_dags[i]
            
            if cross_dag is None or self_dag is None:
                mec_per_fold[fold_name] = None
                continue
            
            n_X, n_S = cross_dag.shape
            learned_full_dag = _combine_attention_to_full_dag(cross_dag, self_dag, n_S, n_X)
            
            if learned_full_dag.shape != true_full_dag.shape:
                mec_per_fold[fold_name] = None
                continue
            
            mec_dist, mec_details = _compute_mec_distance(learned_full_dag, true_full_dag)
            in_mec, membership_details = _check_mec_membership(learned_full_dag, true_full_dag)
            
            mec_distances.append(mec_dist)
            mec_memberships.append(in_mec)
            
            mec_per_fold[fold_name] = {
                "mec_distance": mec_dist,
                "in_mec": in_mec,
                "skeleton_recall": mec_details["skeleton_recall"],
                "skeleton_precision": mec_details["skeleton_precision"],
                "v_structure_recall": mec_details["v_structure_recall"],
                "v_structure_precision": mec_details["v_structure_precision"],
            }
            
            print(f"    {fold_name}: MEC distance = {mec_dist:.4f}, in MEC = {in_mec}")
        
        if mec_distances:
            mec_dist_array = np.array(mec_distances)
            dag_metrics["mec_distance"] = {
                "best": float(np.min(mec_dist_array)),
                "mean": float(np.mean(mec_dist_array)),
                "worst": float(np.max(mec_dist_array)),
                "std": float(np.std(mec_dist_array)),
                "per_fold": mec_per_fold,
            }
            dag_metrics["mec_membership_rate"] = float(np.mean(mec_memberships))
            dag_metrics["n_true_v_structures"] = len(true_v_structures)
    else:
        print("  Could not load full true DAG for MEC computation")
    
    # Save DAG metrics
    with open(join(eval_path_files, dag_metrics_filename), 'w') as f:
        json.dump(dag_metrics, f, indent=2)
    print(f"\n  Saved: {dag_metrics_filename}")
    
    # =========================================================================
    # Plot DAG comparisons
    # =========================================================================
    if per_fold_comparison_data:
        fold_data_groups = defaultdict(list)
        for data in per_fold_comparison_data:
            fold_data_groups[data["fold_name"]].append(data)
        
        for fold_name, fold_data_list in fold_data_groups.items():
            n_blocks = len(fold_data_list)
            fig, axes = plt.subplots(n_blocks, 2, figsize=(8, 3 * n_blocks), squeeze=False)
            
            for idx, data in enumerate(fold_data_list):
                # Learned DAG
                ax_learned = axes[idx, 0]
                im = ax_learned.imshow(data["learned"], vmin=0, vmax=1, cmap='viridis')
                ax_learned.set_title(f"Learned ({data['source']})\n{data['block']}\nSH={data['soft_hamming']:.3f}")
                ax_learned.set_xlabel("Sources")
                ax_learned.set_ylabel("Targets")
                plt.colorbar(im, ax=ax_learned)
                
                for i in range(data["learned"].shape[0]):
                    for j in range(data["learned"].shape[1]):
                        ax_learned.text(j, i, f"{data['learned'][i,j]:.2f}", 
                                       ha='center', va='center', color='white', fontsize=8)
                
                # True DAG
                ax_true = axes[idx, 1]
                im = ax_true.imshow(data["true"], vmin=0, vmax=1, cmap='viridis')
                ax_true.set_title(f"True DAG\n{data['block']}")
                ax_true.set_xlabel("Sources")
                ax_true.set_ylabel("Targets")
                plt.colorbar(im, ax=ax_true)
                
                for i in range(data["true"].shape[0]):
                    for j in range(data["true"].shape[1]):
                        ax_true.text(j, i, f"{int(data['true'][i,j])}", 
                                    ha='center', va='center', color='white', fontsize=10)
            
            plt.suptitle(f"Fold: {fold_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(join(eval_path_fig, f"dag_comparison_{fold_name}_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    print(f"\n✓ eval_attention_scores complete!")
    return dag_metrics


# =============================================================================
# SLOW: Attention Evolution Tracking (across training epochs)
# =============================================================================

def eval_attention_evolution(
    experiment: str,
    n_evaluations: int = 10,
    show_plots: bool = True,
) -> pd.DataFrame:
    """
    Track attention/phi evolution during training.
    
    SLOW: Loads multiple checkpoints per fold (~2-5 minutes depending on n_evaluations).
    Run separately when evolution analysis is needed.
    
    This function tracks how learned DAG structure (attention scores and phi tensors) 
    evolve during training from initialization.
    
    Args:
        experiment: Path to the experiment folder containing config and k_* folders
        n_evaluations: Number of checkpoints to evaluate (evenly distributed).
                      Default is 10. Set to 0 for ALL checkpoints (slowest).
        show_plots: If True (default), display plots. If False, only save to files.
        
    Returns:
        pd.DataFrame with columns:
            - kfold: fold identifier (e.g., "k_0", "k_1")
            - epoch: epoch number (0 for initialization)
            - {block}_{i}{j}_mean: mean attention score across samples
            - {block}_{i}{j}_std: std of attention scores
            - phi_{block}_{i}{j}: learned DAG probability (sigmoid(phi))
            
    Output Files:
        - fig/attention_drift_{exp_id}.png: Attention evolution over training
        - files/scores_evol.csv: Attention evolution data
        
    Example:
        >>> # Track evolution with 10 checkpoints (default, ~2-5 min)
        >>> df = eval_attention_evolution("experiments/my_experiment")
        >>> 
        >>> # Track ALL checkpoints (slower, more detailed)
        >>> df = eval_attention_evolution("experiments/my_experiment", n_evaluations=0)
    """
    import os
    import traceback
    
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_attention_evolution")
    
    scores_evolution_filename = "scores_evol.csv"
    
    print(f"Experiment ID: {exp_id}")
    print(f"n_evaluations: {n_evaluations if n_evaluations > 0 else 'ALL'}")
    
    # Save README
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="Attention evolution tracking (SLOW): how attention/phi evolves during training.",
        files_info={
            scores_evolution_filename: "Attention scores evolution over training epochs (CSV)",
        },
    )
    
    _create_cline_template(eval_path_cline, "eval_attention_evolution", exp_id)
    
    # Check for cached results
    if exists(join(eval_path_files, scores_evolution_filename)):
        df = pd.read_csv(join(eval_path_files, scores_evolution_filename))
        print(f"  Loaded cached evolution data: {len(df)} rows")
    else:
        # Compute evolution data
        df = _load_attention_evolution_data(experiment, n_evaluations)
        df.to_csv(join(eval_path_files, scores_evolution_filename), index=False)
        print(f"  Computed and saved evolution data: {len(df)} rows")
    
    # Plot evolution
    if len(df) > 0:
        fig = plot_attention_evolution(df, aggregate_folds=False, include_phi=True)
        plt.savefig(join(eval_path_fig, f"attention_drift_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"\n✓ eval_attention_evolution complete!")
    return df


def _load_attention_evolution_data(
    experiment_path: str,
    n_evaluations: int = 10,
    datadir_path: str = None,
    dataset_label: str = "test",
) -> pd.DataFrame:
    """
    Load attention scores across training epochs (internal helper).
    
    Args:
        experiment_path: Path to experiment folder
        n_evaluations: Number of checkpoints to evaluate (0 = all)
        datadir_path: Path to data directory
        dataset_label: One of ["train", "test", "all"]
        
    Returns:
        pd.DataFrame with attention evolution data
    """
    import traceback
    import re as _re
    
    if datadir_path is None:
        datadir_path = join(root_path, "data")
    
    # Find config
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    
    # Get architecture config
    architecture_type = get_architecture_type(config)
    print(f"  Architecture: {architecture_type}")
    
    try:
        arch_config = get_architecture_config(architecture_type)
        attention_keys = arch_config["attention_keys"]
        phi_keys = arch_config["phi_keys"]
    except ValueError:
        # Fallback for unknown architecture
        attention_keys = ["decoder", "cross"]
        phi_keys = ["decoder", "cross"]
    
    # Find k-fold directories
    kfold_dirs = sorted([
        d for d in listdir(experiment_path) 
        if isdir(join(experiment_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment_path}")
    
    print(f"  Found {len(kfold_dirs)} k-fold directories")
    
    all_records = []
    
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            epoch_checkpoints = find_all_checkpoints(checkpoints_dir)
            total_checkpoints = len(epoch_checkpoints)
            print(f"\n  {kfold_dir}: {total_checkpoints} checkpoints")
            
            if not epoch_checkpoints:
                continue
            
            # Select checkpoints
            if n_evaluations and n_evaluations > 0:
                selected_checkpoints = _select_evenly_spaced_checkpoints(epoch_checkpoints, n_evaluations)
                print(f"    Selected {len(selected_checkpoints)} checkpoints")
            else:
                selected_checkpoints = epoch_checkpoints
            
            # Storage for initial values (for computing diffs)
            init_attention = {}
            init_phi = {}
            
            for epoch, checkpoint_path in selected_checkpoints:
                record = {'kfold': kfold_dir, 'epoch': epoch}
                
                try:
                    # Run predictions to get attention weights
                    predictions = predict_test_from_ckpt(
                        config=config,
                        datadir_path=datadir_path,
                        checkpoint_path=checkpoint_path,
                        dataset_label=dataset_label,
                        cluster=False,
                    )
                    
                    att_weights = predictions.attention_weights
                    
                    # Load model for phi
                    model = _load_model_from_checkpoint(checkpoint_path, architecture_type)
                    phi_dict = extract_phi_from_model(model, architecture_type)
                    
                    # Process attention weights (backward-compat + per-layer keys)
                    if att_weights is not None:
                        # Build list of all attention keys to track: backward-compat + per-layer
                        all_att_keys = list(attention_keys)
                        per_layer_att_keys = [k for k in att_weights.keys() if _re.match(r'^(dec_self|dec_cross)_L\d+$', k)]
                        all_att_keys.extend(sorted(per_layer_att_keys))
                        
                        for att_key in all_att_keys:
                            att_tensor = att_weights.get(att_key)
                            if att_tensor is None:
                                continue
                            
                            if att_tensor.ndim == 2:
                                att_tensor = np.expand_dims(att_tensor, axis=0)
                            
                            if att_key not in init_attention:
                                init_attention[att_key] = att_tensor
                            
                            mean_att = att_tensor.mean(axis=0)
                            std_att = att_tensor.std(axis=0)
                            
                            n_rows, n_cols = mean_att.shape
                            for i in range(n_rows):
                                for j in range(n_cols):
                                    record[f"{att_key}_{i}{j}_mean"] = mean_att[i, j]
                                    record[f"{att_key}_{i}{j}_std"] = std_att[i, j]
                            
                            # Compute diff from init
                            if att_key in init_attention:
                                init_att = init_attention[att_key]
                                min_batch = min(att_tensor.shape[0], init_att.shape[0])
                                diff = att_tensor[:min_batch] - init_att[:min_batch]
                                diff_mean = diff.mean(axis=0)
                                diff_std = diff.std(axis=0)
                                
                                for i in range(n_rows):
                                    for j in range(n_cols):
                                        record[f"{att_key}_{i}{j}_diff_mean"] = diff_mean[i, j]
                                        record[f"{att_key}_{i}{j}_diff_std"] = diff_std[i, j]
                    
                    # Process phi tensors (backward-compat + per-layer keys)
                    all_phi_keys = list(phi_keys)
                    per_layer_phi_keys = [k for k in phi_dict.keys() if _re.match(r'^(decoder|decoder_cross)_L\d+$', k)]
                    all_phi_keys.extend(sorted(per_layer_phi_keys))
                    
                    for phi_key in all_phi_keys:
                        phi_tensor = phi_dict.get(phi_key)
                        if phi_tensor is None:
                            continue
                        
                        if phi_key not in init_phi:
                            init_phi[phi_key] = phi_tensor
                        
                        n_rows, n_cols = phi_tensor.shape
                        for i in range(n_rows):
                            for j in range(n_cols):
                                record[f"phi_{phi_key}_{i}{j}"] = phi_tensor[i, j]
                        
                        if phi_key in init_phi:
                            phi_diff = phi_tensor - init_phi[phi_key]
                            for i in range(n_rows):
                                for j in range(n_cols):
                                    record[f"phi_{phi_key}_{i}{j}_diff"] = phi_diff[i, j]
                    
                    all_records.append(record)
                    print(f"    ✓ epoch {epoch}")
                    
                except Exception as e:
                    print(f"    ✗ epoch {epoch}: {e}")
                    continue
            
        except Exception as e:
            print(f"  ✗ {kfold_dir}: {e}")
            continue
    
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"\n  Loaded {len(df)} rows from {df['kfold'].nunique()} folds")
        return df
    else:
        print("  Warning: No records processed")
        return pd.DataFrame()


# =============================================================================
# Backward Compatibility: Original Function Name
# =============================================================================

def load_attention_evolution(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    input_conditioning_fn = None,
    n_evaluations: int = 10,
) -> pd.DataFrame:
    """
    [DEPRECATED] Use eval_attention_evolution() instead.
    
    This function is kept for backward compatibility.
    """
    print("Warning: load_attention_evolution() is deprecated. Use eval_attention_evolution() instead.")
    return _load_attention_evolution_data(
        experiment_path=experiment_path,
        n_evaluations=n_evaluations,
        datadir_path=datadir_path,
        dataset_label=dataset_label,
    )
