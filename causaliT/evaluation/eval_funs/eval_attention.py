"""
Attention score and DAG recovery evaluation functions for CausaliT experiments.

This module provides functions for analyzing attention weights and DAG recovery:
- load_attention_evolution: Track attention evolution during training
- eval_attention_scores: Evaluate attention scores and DAG recovery metrics
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
    DEFAULT_PLOT_FORMAT,
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
# Attention Evolution Functions
# =============================================================================

def load_attention_evolution(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    input_conditioning_fn = None,
    n_evaluations: int = 10,
) -> pd.DataFrame:
    """
    Load attention scores and phi tensors across training epochs to track their evolution.
    
    This function tracks how learned DAG structure (attention scores and phi tensors) 
    evolve during training from initialization. For each selected checkpoint across all 
    k-folds, it computes the difference from initialization at the sample level, then 
    aggregates to mean and std.
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        datadir_path: Path to data directory. If None, uses "../data/" relative to project root
        dataset_label: One of ["train", "test", "all"]
        input_conditioning_fn: Optional function to condition inputs before forward pass
        n_evaluations: Number of checkpoints to evaluate (evenly distributed across epochs).
                      If 0 or None, evaluates ALL checkpoints (original behavior).
                      Default is 10, ensuring consistent evaluation time regardless of total epochs.
        
    Returns:
        pd.DataFrame with columns:
            - kfold: fold identifier (e.g., "k_0", "k_1")
            - epoch: epoch number (0 for initialization)
            
            For each attention block (e.g., dec1_self, dec2_cross):
            - {block}_{i}{j}_mean: mean attention score across samples
            - {block}_{i}{j}_std: std of attention scores across samples
            - {block}_{i}{j}_diff_mean: mean of (score_t - score_0) across samples
            - {block}_{i}{j}_diff_std: std of (score_t - score_0) across samples
            
            For each phi tensor (when available):
            - phi_{block}_{i}{j}: learned DAG probability (sigmoid(phi))
            - phi_{block}_{i}{j}_diff: difference from initialization
            
    Example:
        >>> from notebooks.eval_funs.eval_attention import load_attention_evolution
        >>> 
        >>> # Load attention evolution with 10 evaluation points (default)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6")
        >>> 
        >>> # Load ALL checkpoints (slower, more detailed)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6", n_evaluations=0)
    """
    import os
    import traceback
    
    # Default data directory
    if datadir_path is None:
        datadir_path = join(root_path, "data")
    
    # Find config file
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    
    # Determine architecture type
    architecture_type = get_architecture_type(config)
    print(f"Detected architecture: {architecture_type}")
    
    # Determine which attention keys to track based on architecture
    if architecture_type == "TransformerForecaster":
        attention_keys = ["encoder", "decoder", "cross"]
        phi_keys = ["encoder", "decoder", "cross"]
    elif architecture_type == "StageCausalForecaster":
        attention_keys = ["dec1_self", "dec1_cross", "dec2_self", "dec2_cross"]
        phi_keys = ["decoder1", "decoder1_cross", "decoder2", "decoder2_cross"]
    elif architecture_type == "SingleCausalForecaster":
        attention_keys = ["dec_self", "dec_cross"]
        phi_keys = ["decoder", "decoder_cross"]
    elif architecture_type == "NoiseAwareCausalForecaster":
        # Same structure as SingleCausalForecaster
        attention_keys = ["dec_self", "dec_cross"]
        phi_keys = ["decoder", "decoder_cross"]
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    
    # Find all k-fold directories
    kfold_dirs = sorted([
        d for d in listdir(experiment_path) 
        if isdir(join(experiment_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment_path}")
    
    print(f"Found {len(kfold_dirs)} k-fold directories: {kfold_dirs}")
    
    all_records = []
    
    # Process each k-fold
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            # Find all checkpoints sorted by epoch
            epoch_checkpoints = find_all_checkpoints(checkpoints_dir)
            total_checkpoints = len(epoch_checkpoints)
            print(f"\n{kfold_dir}: Found {total_checkpoints} checkpoints")
            
            if not epoch_checkpoints:
                print(f"  ✗ No checkpoints found for {kfold_dir}")
                continue
            
            # Select evenly spaced checkpoints if n_evaluations is specified
            if n_evaluations and n_evaluations > 0:
                selected_checkpoints = _select_evenly_spaced_checkpoints(epoch_checkpoints, n_evaluations)
                print(f"  Selected {len(selected_checkpoints)} checkpoints for evaluation (n_evaluations={n_evaluations})")
                epochs_selected = [ep for ep, _ in selected_checkpoints]
                print(f"  Epochs: {epochs_selected}")
            else:
                selected_checkpoints = epoch_checkpoints
                print(f"  Evaluating ALL {len(selected_checkpoints)} checkpoints")
            
            # Storage for initial attention scores (for computing diffs)
            init_attention = {}  # key -> (B, Q, K) array
            init_phi = {}  # key -> (Q, K) array
            
            # Process each selected checkpoint
            for epoch, checkpoint_path in selected_checkpoints:
                print(f"  Processing epoch {epoch}: {os.path.basename(checkpoint_path)}")
                
                record = {
                    'kfold': kfold_dir,
                    'epoch': epoch,
                }
                
                try:
                    # Run predictions to get attention weights
                    predictions = predict_test_from_ckpt(
                        config=config,
                        datadir_path=datadir_path,
                        checkpoint_path=checkpoint_path,
                        dataset_label=dataset_label,
                        cluster=False,
                        input_conditioning_fn=input_conditioning_fn
                    )
                    
                    att_weights = predictions.attention_weights
                    
                    # Load model to extract phi tensors
                    if architecture_type == "TransformerForecaster":
                        model = TransformerForecaster.load_from_checkpoint(checkpoint_path)
                    elif architecture_type == "StageCausalForecaster":
                        model = StageCausalForecaster.load_from_checkpoint(checkpoint_path)
                    elif architecture_type == "SingleCausalForecaster":
                        model = SingleCausalForecaster.load_from_checkpoint(checkpoint_path)
                    elif architecture_type == "NoiseAwareCausalForecaster":
                        model = NoiseAwareCausalForecaster.load_from_checkpoint(checkpoint_path)
                    
                    phi_dict = extract_phi_from_model(model, architecture_type)
                    
                    # Process attention weights
                    if att_weights is not None:
                        for att_key in attention_keys:
                            att_tensor = att_weights.get(att_key)
                            
                            if att_tensor is None:
                                continue
                            
                            # Ensure 3D: (B, Q, K)
                            if att_tensor.ndim == 2:
                                att_tensor = np.expand_dims(att_tensor, axis=0)
                            
                            # For epoch 0 (or first evaluated epoch), store as initial
                            if att_key not in init_attention:
                                init_attention[att_key] = att_tensor
                            
                            # Compute mean and std across samples
                            mean_att = att_tensor.mean(axis=0)  # (Q, K)
                            std_att = att_tensor.std(axis=0)  # (Q, K)
                            
                            # Flatten and add to record
                            n_rows, n_cols = mean_att.shape
                            for i in range(n_rows):
                                for j in range(n_cols):
                                    record[f"{att_key}_{i}{j}_mean"] = mean_att[i, j]
                                    record[f"{att_key}_{i}{j}_std"] = std_att[i, j]
                            
                            # Compute sample-wise diff from initialization
                            if att_key in init_attention:
                                init_att = init_attention[att_key]
                                
                                # Handle batch size mismatch by using min size
                                min_batch = min(att_tensor.shape[0], init_att.shape[0])
                                diff = att_tensor[:min_batch] - init_att[:min_batch]  # (B, Q, K)
                                
                                diff_mean = diff.mean(axis=0)  # (Q, K)
                                diff_std = diff.std(axis=0)  # (Q, K)
                                
                                for i in range(n_rows):
                                    for j in range(n_cols):
                                        record[f"{att_key}_{i}{j}_diff_mean"] = diff_mean[i, j]
                                        record[f"{att_key}_{i}{j}_diff_std"] = diff_std[i, j]
                            else:
                                # No init available, set diff to 0
                                for i in range(n_rows):
                                    for j in range(n_cols):
                                        record[f"{att_key}_{i}{j}_diff_mean"] = 0.0
                                        record[f"{att_key}_{i}{j}_diff_std"] = 0.0
                    
                    # Process phi tensors
                    for phi_key in phi_keys:
                        phi_tensor = phi_dict.get(phi_key)
                        
                        if phi_tensor is None:
                            continue
                        
                        # For first evaluated checkpoint, store as initial
                        if phi_key not in init_phi:
                            init_phi[phi_key] = phi_tensor
                        
                        # Flatten and add to record
                        n_rows, n_cols = phi_tensor.shape
                        for i in range(n_rows):
                            for j in range(n_cols):
                                record[f"phi_{phi_key}_{i}{j}"] = phi_tensor[i, j]
                        
                        # Compute diff from initialization
                        if phi_key in init_phi:
                            phi_diff = phi_tensor - init_phi[phi_key]
                            for i in range(n_rows):
                                for j in range(n_cols):
                                    record[f"phi_{phi_key}_{i}{j}_diff"] = phi_diff[i, j]
                        else:
                            for i in range(n_rows):
                                for j in range(n_cols):
                                    record[f"phi_{phi_key}_{i}{j}_diff"] = 0.0
                    
                    all_records.append(record)
                    print(f"    ✓ Processed epoch {epoch}")
                    
                except Exception as e:
                    print(f"    ✗ Error processing epoch {epoch}: {e}")
                    traceback.print_exc()
                    continue
            
        except Exception as e:
            print(f"  ✗ Error processing {kfold_dir}: {e}")
            traceback.print_exc()
            continue
    
    # Build DataFrame
    if all_records:
        df = pd.DataFrame(all_records)
        print(f"\nLoaded attention evolution: {len(df)} rows from {df['kfold'].nunique()} folds")
        return df
    else:
        print("Warning: No records were successfully processed")
        return pd.DataFrame()


def eval_attention_scores(experiment: str, show_plots: bool = True) -> dict:
    """
    Evaluate attention scores, DAG (phi) evolution, and DAG recovery metrics.
    
    Loads attention weights from the best checkpoint of each k-fold and tracks
    how attention scores and learned DAG probabilities (phi) evolve during training.
    Also computes DAG recovery metrics by comparing learned phi/attention to true DAG.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        dict: DAG recovery metrics with keys:
            - soft_hamming_cross: Soft Hamming distance for S→X edges (best/mean/worst/std/per_fold)
            - soft_hamming_self: Soft Hamming distance for X→X edges (best/mean/worst/std/per_fold)
            - dag_confidence_cross: DAG consistency across folds for S→X (1=identical, 0=max disagreement)
            - dag_confidence_self: DAG consistency across folds for X→X (1=identical, 0=max disagreement)
        
    Output Files:
        - fig/attention_scores_{exp_id}.pdf: Attention score heatmaps for all folds
        - fig/attention_drift_{exp_id}.pdf: Attention evolution over training
        - fig/dag_comparison_{exp_id}.pdf: Learned vs true DAG comparison heatmaps
        - files/final_scores/: Saved attention data (can be reloaded quickly)
        - files/scores_evol.csv: Attention evolution data
        - files/dag_metrics.json: DAG recovery metrics (soft Hamming + dag_confidence)
        
    Notes:
        - Supports TransformerForecaster, StageCausalForecaster, and SingleCausalForecaster
        - Results are cached; delete files/ contents to recompute
        - DAG metrics compare phi (if available) or mean attention scores to true DAG masks
        - dag_confidence = 1 - 2*mean(std of edges across folds), measuring fold consistency
        
    Example:
        >>> metrics = eval_attention_scores("../experiments/single/local/my_experiment")
        >>> print(f"Soft Hamming (cross): {metrics['soft_hamming_cross']['mean']:.4f}")
        >>> print(f"DAG Confidence (cross): {metrics['dag_confidence_cross']:.4f}")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_attention_scores")

    final_scores_dirname = "final_scores"
    scores_evolution_filename = "scores_evol.csv"
    dag_metrics_filename = "dag_metrics.json"
    attention_labels_filename = "attention_labels.json"

    # =========================================================================
    # Attention labels for AI interpretation
    # =========================================================================
    
    # Generic attention block descriptions (architecture-dependent, not dataset-specific)
    attention_labels = {
        "description": "Attention weights and DAG (phi) structure learned by the model",
        "attention_blocks": {
            "SingleCausalForecaster": {
                "dec_cross": "Cross-attention: S → X (source variables influence intermediate variables)",
                "dec_self": "Self-attention: X → X (intermediate variables influence each other)",
            },
            "NoiseAwareCausalForecaster": {
                "dec_cross": "Cross-attention: S → X (source variables influence intermediate variables)",
                "dec_self": "Self-attention: X → X (intermediate variables influence each other)",
            },
            "StageCausalForecaster": {
                "dec1_cross": "Stage 1 Cross-attention: S → X (source to intermediate)",
                "dec1_self": "Stage 1 Self-attention: X → X (intermediate to intermediate)",
                "dec2_cross": "Stage 2 Cross-attention: X → Y (intermediate to output)",
                "dec2_self": "Stage 2 Self-attention: Y → Y (output to output)",
            },
            "TransformerForecaster": {
                "encoder": "Encoder self-attention",
                "decoder": "Decoder self-attention",
                "cross": "Decoder cross-attention (encoder → decoder)",
            },
        },
        "phi_tensors": {
            "description": "Learned DAG edge probabilities (sigmoid(phi)). Values in [0,1] where 1 = edge present.",
            "interpretation": "phi is learned by LieAttention and CausalCrossAttention modules",
        },
        "dag_metrics": {
            "soft_hamming": "Mean absolute difference between learned and true DAG. 0 = perfect, 1 = inverted",
            "source": "'phi' if LieAttention/CausalCrossAttention used, else 'attention' (mean attention scores)",
        },
        "matrix_indexing": {
            "rows": "Target variables (queries) - the variables being predicted",
            "columns": "Source variables (keys) - the variables providing information",
            "value_ij": "Attention weight from source j to target i (how much target i attends to source j)",
        },
    }
    
    # =========================================================================
    # Load dataset metadata for variable mappings (NO FALLBACK - requires metadata)
    # =========================================================================
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_name = None
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
            # Build variable mapping from metadata
            variable_mapping = {}
            
            # Add variable descriptions
            if "variable_descriptions" in metadata:
                variable_mapping.update(metadata["variable_descriptions"])
            
            # Add DAG structure description from edges
            if "causal_structure" in metadata and "edges" in metadata["causal_structure"]:
                edges = metadata["causal_structure"]["edges"]
                edge_strs = [f"{src}→{tgt}" for src, tgt in edges]
                variable_mapping["dag_structure"] = ", ".join(edge_strs) + " (true causal DAG)"
            
            attention_labels["variable_mapping"] = variable_mapping
            attention_labels["dataset"] = dataset_name
            print(f"  Loaded variable mappings from metadata for dataset: {dataset_name}")
        else:
            raise ValueError(
                f"Dataset metadata not found for '{dataset_name}'. "
                f"Ensure dataset_metadata.json exists in data/{dataset_name}/"
            )
    else:
        raise ValueError(
            f"No dataset specified in experiment config. "
            f"Cannot load variable mappings."
        )
    
    _save_variable_labels(eval_path_files, attention_labels, attention_labels_filename)

    # Save README with column documentation
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation folder contains attention scores from test predictions and DAG recovery metrics.",
        files_info={
            final_scores_dirname: "Saved attention data (npz files) for fast reloading",
            scores_evolution_filename: "Attention scores evolution over training epochs (CSV)",
            dag_metrics_filename: "Soft Hamming distance comparing learned DAG to true DAG (JSON)",
            attention_labels_filename: "Descriptions of attention blocks and interpretation guide (JSON)",
        },
        column_documentation={
            "kfold": "Cross-validation fold identifier (k_0, k_1, ...)",
            "epoch": "Training epoch number",
            "{block}_{i}{j}_mean": "Mean attention from source j to target i (averaged across samples)",
            "{block}_{i}{j}_diff_mean": "Change in attention from initialization",
            "phi_{block}_{i}{j}": "Learned DAG probability for edge j→i",
        }
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_attention_scores", exp_id)

    print(f"Experiment ID: {exp_id}")
    
    # Load or compute final attention scores
    if exists(join(eval_path_files, final_scores_dirname)):
        final_scores_dict = load_attention_data_from_file(join(eval_path_files, final_scores_dirname))
        print("Experiment already available. Data loaded!")
    else:
        final_scores_dict = load_attention_data(experiment)
        save_attention_data(final_scores_dict, join(eval_path_files, final_scores_dirname), save_predictions=True)
        print("Data saved!")
    
    # Plot: Attention score heatmaps
    fig = plot_attention_scores(final_scores_dict, cmap='viridis', annotation_fontsize=8, scale_mode="row")
    plt.savefig(join(eval_path_fig, f"attention_scores_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Load or compute attention evolution
    if exists(join(eval_path_files, scores_evolution_filename)):
        df = pd.read_csv(join(eval_path_files, scores_evolution_filename))
        print("Experiment already available. Data loaded!")
    else:
        df = load_attention_evolution(experiment, n_evaluations=10)
        df.to_csv(join(eval_path_files, scores_evolution_filename))
        print("Data saved!")
    
    # Plot: Attention evolution
    fig = plot_attention_evolution(df, aggregate_folds=False, include_phi=True)
    plt.savefig(join(eval_path_fig, f"attention_drift_{exp_id}.{DEFAULT_PLOT_FORMAT}"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # =========================================================================
    # DAG Recovery Metrics (Per-Fold)
    # =========================================================================
    print("\n--- Computing DAG Recovery Metrics (Per-Fold) ---")
    
    # Load config to get dataset name
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        print("Warning: No config file found, skipping DAG metrics")
        return {}
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset = config.get("data", {}).get("dataset")
    
    if dataset is None:
        print("Warning: No dataset specified in config, skipping DAG metrics")
        return {}
    
    # Data directory (relative to project root)
    datadir_path = join(root_path, "data")
    
    # Initialize metrics dict
    dag_metrics = {
        "dataset": dataset,
        "architecture": final_scores_dict.architecture_type,
    }
    
    # Define which attention blocks to evaluate based on architecture
    architecture = final_scores_dict.architecture_type
    
    if architecture == "SingleCausalForecaster":
        blocks_to_eval = [
            ("dec_cross", "decoder_cross", "dec_cross"),
            ("dec_self", "decoder", "dec_self"),
        ]
    elif architecture == "NoiseAwareCausalForecaster":
        # Same structure as SingleCausalForecaster
        blocks_to_eval = [
            ("dec_cross", "decoder_cross", "dec_cross"),
            ("dec_self", "decoder", "dec_self"),
        ]
    elif architecture == "StageCausalForecaster":
        blocks_to_eval = [
            ("decoder1_cross", "decoder1_cross", "dec1_cross"),
            ("decoder1_self", "decoder1", "dec1_self"),
            ("decoder2_cross", "decoder2_cross", "dec2_cross"),
            ("decoder2_self", "decoder2", "dec2_self"),
        ]
    elif architecture == "TransformerForecaster":
        blocks_to_eval = [
            ("cross", "cross", "dec_cross"),
            ("decoder", "decoder", "dec_self"),
        ]
    else:
        print(f"Warning: Unknown architecture {architecture}, skipping DAG metrics")
        return {}
    
    # Store per-fold comparison data for plotting
    per_fold_comparison_data = []
    
    for att_key, phi_key, mask_type in blocks_to_eval:
        print(f"  Evaluating {att_key}...")
        
        # Get learned DAG for each fold separately
        fold_dags, source = _get_learned_dag_per_fold(final_scores_dict, att_key, phi_key)
        
        if all(dag is None for _, dag in fold_dags):
            print(f"    No data available for {att_key}")
            continue
        
        # Load true DAG mask
        true_dag = _load_true_dag_mask(datadir_path, dataset, mask_type)
        
        if true_dag is None:
            print(f"    No true DAG mask found for {mask_type}")
            continue
        
        # Compute per-fold soft Hamming distances
        per_fold_values = {}
        fold_sh_list = []
        
        for fold_name, learned_dag in fold_dags:
            if learned_dag is None:
                print(f"    {fold_name}: No data available")
                per_fold_values[fold_name] = None
                continue
            
            # Check shape compatibility
            if learned_dag.shape != true_dag.shape:
                print(f"    {fold_name}: Shape mismatch: learned {learned_dag.shape} vs true {true_dag.shape}")
                per_fold_values[fold_name] = None
                continue
            
            # Compute soft Hamming distance for this fold
            soft_hamming = _compute_soft_hamming(learned_dag, true_dag)
            per_fold_values[fold_name] = soft_hamming
            fold_sh_list.append(soft_hamming)
            
            print(f"    {fold_name}: Soft Hamming ({source}) = {soft_hamming:.4f}")
            
            # Store for per-fold plotting
            per_fold_comparison_data.append({
                "fold_name": fold_name,
                "block": att_key,
                "learned": learned_dag,
                "true": true_dag,
                "soft_hamming": soft_hamming,
                "source": source,
            })
        
        # Compute statistics across folds
        if fold_sh_list:
            fold_sh_array = np.array(fold_sh_list)
            metric_key = f"soft_hamming_{mask_type.replace('dec_', '').replace('dec1_', '').replace('dec2_', '')}"
            
            dag_metrics[metric_key] = {
                "best": float(np.min(fold_sh_array)),
                "mean": float(np.mean(fold_sh_array)),
                "worst": float(np.max(fold_sh_array)),
                "std": float(np.std(fold_sh_array)),
                "per_fold": per_fold_values,
            }
            dag_metrics[f"{metric_key}_source"] = source
            
            print(f"    Statistics: best={np.min(fold_sh_array):.4f}, mean={np.mean(fold_sh_array):.4f}, worst={np.max(fold_sh_array):.4f}, std={np.std(fold_sh_array):.4f}")
        
        # Compute DAG confidence (consistency across folds)
        valid_fold_dags = [dag for _, dag in fold_dags if dag is not None]
        if len(valid_fold_dags) >= 2:
            confidence_key = f"dag_confidence_{mask_type.replace('dec_', '').replace('dec1_', '').replace('dec2_', '')}"
            confidence = _compute_dag_confidence(valid_fold_dags)
            dag_metrics[confidence_key] = confidence
            print(f"    DAG Confidence: {confidence:.4f}")
        elif len(valid_fold_dags) == 1:
            confidence_key = f"dag_confidence_{mask_type.replace('dec_', '').replace('dec1_', '').replace('dec2_', '')}"
            dag_metrics[confidence_key] = 1.0
            print(f"    DAG Confidence: 1.0 (single fold)")
    
    # Save DAG metrics to JSON
    with open(join(eval_path_files, dag_metrics_filename), 'w') as f:
        json.dump(dag_metrics, f, indent=2)
    print(f"  Saved: {dag_metrics_filename}")
    
    # Plot: Per-fold DAG comparison heatmaps (one PDF per fold)
    if per_fold_comparison_data:
        # Group by fold_name
        fold_data_groups = defaultdict(list)
        for data in per_fold_comparison_data:
            fold_data_groups[data["fold_name"]].append(data)
        
        # Generate one plot per fold
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
                
                # Add value annotations
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
                
                # Add value annotations
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
    
    return dag_metrics
