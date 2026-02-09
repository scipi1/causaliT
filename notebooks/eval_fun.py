"""
Evaluation Functions for CausaliT Experiments.

This module provides standardized evaluation functions for analyzing trained models.
Each function creates a structured output directory within the experiment folder containing:
- fig/: Generated plots and visualizations (PDF format)
- files/: Data files (CSV, intermediate results)
- cline/: Workspace for AI assistants to save notes

The evaluation functions support automatic caching - if results already exist in the
expected locations, they will be loaded instead of recomputed.

Functions:
    eval_train_metrics: Analyze training metrics (loss curves, regularization terms)
    eval_embed: Analyze embedding evolution and cosine similarities
    eval_attention_scores: Analyze attention weights and phi (DAG) evolution
    eval_interventions: Analyze model predictions under causal interventions

Example:
    >>> from notebooks.eval_fun import eval_train_metrics, eval_attention_scores
    >>> 
    >>> experiment = "../experiments/single/local/my_experiment"
    >>> 
    >>> # Evaluate training metrics
    >>> df_metrics = eval_train_metrics(experiment)
    >>> 
    >>> # Evaluate attention scores
    >>> eval_attention_scores(experiment)

Dependencies:
    - notebooks.lib: Data loading utilities
    - notebooks.plot_lib: Plotting utilities
    - causaliT.evaluation.predict: Prediction and intervention functions
"""

import re
from os.path import dirname, abspath, join, exists
from os import makedirs
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

# Setup root path for local imports
# Use __file__ to get the directory of this script, then go up one level to project root
root_path = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(root_path)

import torch
from causaliT.evaluation.predict import create_intervention_fn, predict_test_from_ckpt
from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
from causaliT.training.forecasters.single_causal_forecaster import SingleCausalForecaster
from notebooks.lib import (
    load_training_metrics,
    load_embeddings_evolution,
    load_attention_data,
    load_attention_data_from_file,
    save_attention_data,
    predict_from_experiment,
    find_config_file,
    get_architecture_type,
    extract_phi_from_model,
)
from notebooks.plot_lib import plot_attention_scores, plot_attention_evolution


# =============================================================================
# DAG Recovery Metrics
# =============================================================================

def _compute_soft_hamming(learned: np.ndarray, true: np.ndarray) -> float:
    """
    Compute soft Hamming distance between learned and true DAG adjacency matrices.
    
    Soft Hamming distance = mean(|learned_ij - true_ij|)
    
    This extends the standard Hamming distance to continuous predictions:
    - 0.0 = perfect match (all edges correctly predicted)
    - 1.0 = completely wrong (all edges inverted)
    
    Args:
        learned: Learned adjacency matrix with values in [0, 1] (e.g., phi or attention scores)
        true: True binary adjacency matrix with values in {0, 1}
        
    Returns:
        float: Soft Hamming distance in [0, 1]
        
    Example:
        >>> learned = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> true = np.array([[1, 0], [0, 1]])
        >>> _compute_soft_hamming(learned, true)
        0.15  # Average absolute difference
    """
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")
    
    return float(np.mean(np.abs(learned - true)))


def _load_true_dag_mask(
    datadir_path: str, 
    dataset: str, 
    mask_type: str
) -> Optional[np.ndarray]:
    """
    Load true DAG adjacency mask from the dataset folder.
    
    Mask files are CSV with:
    - Rows = target variables (X1, X2, ... or Y1, Y2, ...)
    - Columns = source variables (S1, S2, S3 or X1, X2, ...)
    - Values = 0 or 1 indicating edge presence
    
    Args:
        datadir_path: Path to the data directory
        dataset: Dataset name (e.g., "scm6")
        mask_type: Type of mask to load. One of:
            - "dec_cross" or "dec1_cross": S → X mask (file: dec1_cross_att_mask.csv)
            - "dec_self" or "dec1_self": X → X mask (file: dec1_self_att_mask.csv)
            - "dec2_cross": X → Y mask (file: dec2_cross_att_mask.csv)
            - "dec2_self": Y → Y mask (file: dec2_self_att_mask.csv)
            
    Returns:
        np.ndarray: Binary adjacency matrix, or None if file not found
        
    Example:
        >>> mask = _load_true_dag_mask("../data", "scm6", "dec_cross")
        >>> print(mask)
        [[1 0 0]    # X1 ← S1
         [0 1 1]]   # X2 ← S2, S3
    """
    # Map mask type to filename
    mask_file_map = {
        "dec_cross": "dec1_cross_att_mask.csv",
        "dec1_cross": "dec1_cross_att_mask.csv",
        "decoder_cross": "dec1_cross_att_mask.csv",
        "dec_self": "dec1_self_att_mask.csv",
        "dec1_self": "dec1_self_att_mask.csv",
        "decoder": "dec1_self_att_mask.csv",
        "dec2_cross": "dec2_cross_att_mask.csv",
        "decoder2_cross": "dec2_cross_att_mask.csv",
        "dec2_self": "dec2_self_att_mask.csv",
        "decoder2": "dec2_self_att_mask.csv",
    }
    
    filename = mask_file_map.get(mask_type)
    if filename is None:
        print(f"Warning: Unknown mask type '{mask_type}'")
        return None
    
    filepath = join(datadir_path, dataset, filename)
    
    if not exists(filepath):
        print(f"Warning: Mask file not found: {filepath}")
        return None
    
    try:
        # Load CSV with first column as index (variable names)
        df = pd.read_csv(filepath, index_col=0)
        return df.values.astype(float)
    except Exception as e:
        print(f"Warning: Failed to load mask {filepath}: {e}")
        return None


def _get_learned_dag(
    attention_data,
    attention_key: str,
    phi_key: str,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Extract the learned DAG from attention data (averaged across folds).
    
    Priority:
    1. If phi tensor is available (LieAttention, CausalCrossAttention), use it
    2. Otherwise, use mean attention scores across test samples
    
    Args:
        attention_data: AttentionData object from load_attention_data()
        attention_key: Key for attention weights (e.g., "dec_cross", "dec_self")
        phi_key: Key for phi tensor (e.g., "decoder_cross", "decoder")
        
    Returns:
        Tuple of (learned_dag, source):
            - learned_dag: np.ndarray with shape (n_targets, n_sources), values in [0,1]
            - source: "phi" or "attention" indicating which was used
    """
    # Try phi first (preferred - it's the learned DAG structure)
    phi_list = attention_data.phi_tensors.get(phi_key, [])
    phi_available = any(p is not None for p in phi_list)
    
    if phi_available:
        # Average phi across k-folds
        phi_arrays = [p for p in phi_list if p is not None]
        learned_dag = np.mean(phi_arrays, axis=0)
        return learned_dag, "phi"
    
    # Fall back to mean attention scores
    att_list = attention_data.attention_weights.get(attention_key, [])
    att_available = any(a is not None for a in att_list)
    
    if att_available:
        # For each fold, compute mean attention across samples, then average across folds
        fold_means = []
        for att in att_list:
            if att is not None:
                # att shape: (B, n_targets, n_sources) or (n_targets, n_sources)
                if att.ndim == 3:
                    fold_means.append(att.mean(axis=0))  # Mean over samples
                else:
                    fold_means.append(att)
        
        if fold_means:
            learned_dag = np.mean(fold_means, axis=0)
            return learned_dag, "attention"
    
    return None, "none"


def _get_learned_dag_per_fold(
    attention_data,
    attention_key: str,
    phi_key: str,
) -> Tuple[List[Tuple[str, Optional[np.ndarray]]], str]:
    """
    Extract learned DAG for each fold separately (no averaging).
    
    Priority:
    1. If phi tensor is available (LieAttention, CausalCrossAttention), use it
    2. Otherwise, use mean attention scores across test samples
    
    Args:
        attention_data: AttentionData object from load_attention_data()
        attention_key: Key for attention weights (e.g., "dec_cross", "dec_self")
        phi_key: Key for phi tensor (e.g., "decoder_cross", "decoder")
        
    Returns:
        Tuple of (fold_dags, source):
            - fold_dags: List of (fold_name, dag_array) tuples where dag_array has 
              shape (n_targets, n_sources) with values in [0,1], or None if unavailable
            - source: "phi" or "attention" indicating which was used
            
    Example:
        >>> fold_dags, source = _get_learned_dag_per_fold(attention_data, "dec_cross", "decoder_cross")
        >>> for fold_name, dag in fold_dags:
        ...     if dag is not None:
        ...         print(f"{fold_name}: shape={dag.shape}")
    """
    # Derive fold names from checkpoint_paths or generate default names
    # checkpoint_paths have format like: ".../k_0/checkpoints/best_checkpoint.ckpt"
    fold_names = []
    if attention_data.checkpoint_paths:
        for ckpt_path in attention_data.checkpoint_paths:
            # Extract k_X from the path
            match = re.search(r'(k_\d+)', ckpt_path)
            if match:
                fold_names.append(match.group(1))
            else:
                fold_names.append(f"fold_{len(fold_names)}")
    else:
        # Determine number of folds from phi_tensors or attention_weights
        phi_list = attention_data.phi_tensors.get(phi_key, [])
        att_list = attention_data.attention_weights.get(attention_key, [])
        num_folds = max(len(phi_list), len(att_list))
        fold_names = [f"k_{i}" for i in range(num_folds)]
    
    # Try phi first (preferred - it's the learned DAG structure)
    phi_list = attention_data.phi_tensors.get(phi_key, [])
    phi_available = any(p is not None for p in phi_list)
    
    if phi_available:
        fold_dags = []
        for i, fold_name in enumerate(fold_names):
            if i < len(phi_list) and phi_list[i] is not None:
                fold_dags.append((fold_name, phi_list[i]))
            else:
                fold_dags.append((fold_name, None))
        return fold_dags, "phi"
    
    # Fall back to mean attention scores
    att_list = attention_data.attention_weights.get(attention_key, [])
    att_available = any(a is not None for a in att_list)
    
    if att_available:
        fold_dags = []
        for i, fold_name in enumerate(fold_names):
            if i < len(att_list) and att_list[i] is not None:
                att = att_list[i]
                # att shape: (B, n_targets, n_sources) or (n_targets, n_sources)
                if att.ndim == 3:
                    fold_dag = att.mean(axis=0)  # Mean over samples
                else:
                    fold_dag = att
                fold_dags.append((fold_name, fold_dag))
            else:
                fold_dags.append((fold_name, None))
        return fold_dags, "attention"
    
    # No data available
    return [(fn, None) for fn in fold_names], "none"

# =============================================================================
# Plotting Standard Settings
# =============================================================================
plt.rcParams['figure.dpi'] = 100  # 360 for publication, 100 for notebook visualization
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


# =============================================================================
# Helper Functions
# =============================================================================

def _setup_eval_directories(experiment: str, eval_name: str) -> Tuple[str, str, str, str, str]:
    """
    Set up standard evaluation directory structure.
    
    Creates the following directory structure within the experiment folder:
        experiment/
        └── eval/
            └── {eval_name}/
                ├── fig/      (for plots)
                ├── files/    (for data files)
                └── cline/    (for AI assistant notes)
    
    Args:
        experiment: Path to the experiment folder
        eval_name: Name of the evaluation (e.g., "eval_train_metrics")
        
    Returns:
        Tuple of (eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id)
    """
    eval_path_root = join(experiment, "eval", eval_name)
    eval_path_fig = join(eval_path_root, "fig")
    eval_path_files = join(eval_path_root, "files")
    eval_path_cline = join(eval_path_root, "cline")

    makedirs(eval_path_fig, exist_ok=True)
    makedirs(eval_path_files, exist_ok=True)
    makedirs(eval_path_cline, exist_ok=True)
    
    # Extract experiment ID from path (last component)
    match = re.search(r'([^/\\]+)$', experiment)
    exp_id = match.group(1) if match else "unknown"
    
    return eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id


def _save_readme(eval_path_root: str, eval_path_cline: str, eval_path_files: str, 
                 eval_path_fig: str, description: str, files_info: dict = None,
                 column_documentation: dict = None) -> None:
    """
    Save a standardized README.yaml file in the evaluation directory.
    
    Args:
        eval_path_root: Root path for the evaluation
        eval_path_cline: Path to cline directory
        eval_path_files: Path to files directory
        eval_path_fig: Path to figures directory
        description: Description of the evaluation
        files_info: Optional dict mapping file descriptions to filenames
        column_documentation: Optional dict documenting CSV column meanings
    """
    readme = {
        "READ THIS": f"If you are an AI, use the folder {eval_path_cline} to save notes and documents. "
                     f"Never delete files in {eval_path_files} and {eval_path_fig}.",
        "description": description,
    }
    if files_info:
        readme["files"] = files_info
    if column_documentation:
        readme["column_documentation"] = column_documentation
    
    OmegaConf.save(readme, join(eval_path_root, "README.yaml"))


def _save_variable_labels(eval_path_files: str, labels: dict, filename: str = "variable_labels.json") -> None:
    """
    Save variable labels JSON file for AI-friendly data interpretation.
    
    Args:
        eval_path_files: Path to files directory
        labels: Dict containing variable mappings and descriptions
        filename: Output filename (default: variable_labels.json)
    """
    import json
    with open(join(eval_path_files, filename), 'w') as f:
        json.dump(labels, f, indent=2)


def _create_cline_template(eval_path_cline: str, eval_name: str, exp_id: str) -> None:
    """
    Create a markdown template for AI analysis notes in the cline directory.
    
    Args:
        eval_path_cline: Path to cline directory
        eval_name: Name of the evaluation function
        exp_id: Experiment identifier
    """
    from datetime import datetime
    
    template_path = join(eval_path_cline, "cline_notes.md")
    
    # Only create if doesn't exist (don't overwrite existing notes)
    if exists(template_path):
        return
    
    template = f"""# AI Analysis Notes - {eval_name}

## Experiment: {exp_id}
## Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
## Last Updated: 

---

### Key Findings
- [ ] Finding 1
- [ ] Finding 2

### Anomalies or Concerns
- 

### Questions for Human Review
- 

### Suggested Follow-up Experiments
- 

---

### Detailed Observations

<!-- Add detailed analysis below -->

"""
    
    with open(template_path, 'w') as f:
        f.write(template)


# =============================================================================
# Checkpoint Discovery Functions
# =============================================================================

def find_all_checkpoints(checkpoints_dir: str) -> List[Tuple[int, str]]:
    """
    Find all checkpoints in a directory and return them sorted by epoch.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        List of (epoch, checkpoint_path) tuples sorted by epoch
        
    Example:
        >>> checkpoints = find_all_checkpoints("experiments/my_exp/k_0/checkpoints")
        >>> # Returns: [(0, "path/epoch0-initial.ckpt"), (5, "path/epoch=5-train_loss=0.01.ckpt"), ...]
    """
    from os.path import isdir
    from os import listdir
    
    if not exists(checkpoints_dir) or not isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    checkpoint_files = [f for f in listdir(checkpoints_dir) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    epoch_checkpoints = []
    
    # Pattern for regular checkpoints: epoch={num}-train_loss={loss}.ckpt
    epoch_pattern = re.compile(r'epoch=(\d+)')
    # Pattern for initial checkpoint: epoch0-initial.ckpt
    initial_pattern = re.compile(r'epoch0-initial\.ckpt')
    
    for ckpt in checkpoint_files:
        # Skip best_checkpoint.ckpt as it's a duplicate
        if ckpt == 'best_checkpoint.ckpt':
            continue
            
        full_path = join(checkpoints_dir, ckpt)
        
        # Check for initial checkpoint
        if initial_pattern.match(ckpt):
            epoch_checkpoints.append((0, full_path))
            continue
        
        # Check for regular epoch checkpoint
        match = epoch_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))
            epoch_checkpoints.append((epoch, full_path))
    
    # Sort by epoch
    epoch_checkpoints.sort(key=lambda x: x[0])
    
    return epoch_checkpoints


def _select_evenly_spaced_checkpoints(
    epoch_checkpoints: List[Tuple[int, str]], 
    n_evaluations: int
) -> List[Tuple[int, str]]:
    """
    Select n_evaluations checkpoints evenly distributed across the training epochs.
    
    Always includes the first (epoch 0) and last checkpoint if available.
    
    Args:
        epoch_checkpoints: List of (epoch, checkpoint_path) tuples sorted by epoch
        n_evaluations: Number of checkpoints to select
        
    Returns:
        List of selected (epoch, checkpoint_path) tuples
        
    Example:
        >>> # 100 checkpoints, select 10 evenly spaced
        >>> checkpoints = [(i, f"epoch={i}.ckpt") for i in range(0, 100, 1)]
        >>> selected = _select_evenly_spaced_checkpoints(checkpoints, 10)
        >>> # Returns approximately: [(0, ...), (11, ...), (22, ...), ..., (99, ...)]
    """
    n_total = len(epoch_checkpoints)
    
    if n_total <= n_evaluations or n_evaluations <= 0:
        # Return all checkpoints if we have fewer than requested or n_evaluations is 0/negative
        return epoch_checkpoints
    
    if n_evaluations == 1:
        # Just return the last checkpoint
        return [epoch_checkpoints[-1]]
    
    if n_evaluations == 2:
        # Return first and last
        return [epoch_checkpoints[0], epoch_checkpoints[-1]]
    
    # Select evenly spaced indices, always including first (0) and last (n_total-1)
    indices = [0]  # Always include first
    
    # Calculate intermediate indices
    step = (n_total - 1) / (n_evaluations - 1)
    for i in range(1, n_evaluations - 1):
        idx = int(round(i * step))
        if idx not in indices:  # Avoid duplicates
            indices.append(idx)
    
    indices.append(n_total - 1)  # Always include last
    
    # Remove duplicates and sort
    indices = sorted(set(indices))
    
    return [epoch_checkpoints[i] for i in indices]


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
        >>> from notebooks.eval_fun import load_attention_evolution
        >>> 
        >>> # Load attention evolution with 10 evaluation points (default)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6")
        >>> 
        >>> # Load ALL checkpoints (slower, more detailed)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6", n_evaluations=0)
        >>> 
        >>> # Plot evolution of a specific attention entry over epochs
        >>> import matplotlib.pyplot as plt
        >>> for kfold in df['kfold'].unique():
        ...     fold_data = df[df['kfold'] == kfold]
        ...     plt.plot(fold_data['epoch'], fold_data['dec1_self_00_diff_mean'], label=kfold)
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('Attention score change from init')
        >>> plt.legend()
        >>> plt.show()
    """
    import os
    from os.path import isdir
    from os import listdir
    
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
                    import traceback
                    traceback.print_exc()
                    continue
            
        except Exception as e:
            print(f"  ✗ Error processing {kfold_dir}: {e}")
            import traceback
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
        # Results
        "best_val_loss": None,
        "best_test_r2": None,
        "num_folds": None,
        # DAG recovery metrics (statistics only - per-fold details are in dag_metrics.json)
        "soft_hamming_cross_best": None,
        "soft_hamming_cross_mean": None,
        "soft_hamming_cross_worst": None,
        "soft_hamming_self_best": None,
        "soft_hamming_self_mean": None,
        "soft_hamming_self_worst": None,
        "dag_source": None,  # "phi" or "attention"
        # HSIC (independence regularization)
        "final_hsic_mean": None,  # Mean HSIC across folds at final epoch
        "final_hsic_std": None,   # Std HSIC across folds
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
# Evaluation Functions
# =============================================================================

def eval_train_metrics(experiment: str, show_plots: bool = True) -> pd.DataFrame:
    """
    Evaluate and visualize training metrics from an experiment.
    
    Loads training metrics (loss, regularization terms) from all k-folds and generates:
    - Loss curves (train/val) over epochs
    - NOTEARS constraint evolution (if logged)
    - HSIC regularization evolution (if logged)
    - Sparsity regularization evolution (if logged)
    - Correlation heatmap between all metrics
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        pd.DataFrame: Combined training metrics from all k-folds with columns:
            - kfold: Fold identifier (e.g., "k_0", "k_1")
            - epoch: Training epoch
            - train_loss, val_loss: Training and validation loss
            - Additional columns for any logged regularization terms
            
    Output Files:
        - fig/loss_{exp_id}.pdf: Loss curves
        - fig/notears_{exp_id}.pdf: NOTEARS curves (if available)
        - fig/hsic_{exp_id}.pdf: HSIC curves (if available)
        - fig/sparsity_*_{exp_id}.pdf: Sparsity curves (if available)
        - fig/metrics_corr_{exp_id}.pdf: Correlation heatmap
        - files/matrix_corr.csv: Correlation matrix data
        
    Example:
        >>> df = eval_train_metrics("../experiments/single/local/my_experiment")
        >>> print(df.columns.tolist())
        >>> df.groupby('kfold')['val_loss'].min()  # Best loss per fold
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
            "train_notears": "NOTEARS acyclicity constraint on training set (0 = DAG)",
            "val_notears": "NOTEARS acyclicity constraint on validation set",
            "train_hsic_reg": "HSIC independence regularization (residuals vs parents) - training",
            "val_hsic_reg": "HSIC independence regularization - validation",
            "train_sparsity_cross": "L1 sparsity on cross-attention (S→X edges)",
            "val_sparsity_cross": "L1 sparsity on cross-attention - validation",
            "train_sparsity_self": "L1 sparsity on self-attention (X→X edges)",
            "val_sparsity_self": "L1 sparsity on self-attention - validation",
            "train_sparsity_total": "Total L1 sparsity (self + cross)",
            "val_sparsity_total": "Total L1 sparsity - validation",
            "test_r2": "Test set R² score (explained variance)",
            "test_x_r2": "Test set R² for X predictions (SingleCausal/StageCausal)",
        },
        "interpretation": {
            "lower_is_better": ["train_loss", "val_loss", "test_loss", "train_notears", "val_notears", 
                               "train_hsic_reg", "val_hsic_reg", "train_sparsity_cross", "val_sparsity_cross",
                               "train_sparsity_self", "val_sparsity_self", "train_sparsity_total", "val_sparsity_total"],
            "higher_is_better": ["test_r2", "test_x_r2"],
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
        },
        column_documentation=metric_labels["column_documentation"]
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_train_metrics", exp_id)
    
    print(f"Experiment ID: {exp_id}")
    
    # Load and preprocess metrics
    df = load_training_metrics(experiment)
    df = df.groupby(["kfold", "epoch"]).first().reset_index()
    
    # Compute correlation matrix (excluding test metrics)
    df_no_test = df[[c for c in df.columns if "test" not in c]]
    numeric_df = df_no_test.select_dtypes(include=['number'])
    df_corr = numeric_df.corr().abs()
    ranked = df_corr.unstack().sort_values(ascending=False)
    ranked[ranked < 1].to_csv(join(eval_path_files, metrics_corr_filename))
    
    # Plot: Loss curves
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="epoch", y="val_loss", hue="kfold", ax=ax)
    sns.lineplot(data=df, x="epoch", y="train_loss", hue="kfold", ax=ax, legend=False, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("val solid | train dashed")
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"loss_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot: NOTEARS (if available)
    if all(s in df.columns for s in ["train_notears", "val_notears"]):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="epoch", y="val_notears", hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y="train_notears", hue="kfold", ax=ax, legend=False, linestyle=":")
        ax.set_ylabel("NOTEARS")
        ax.set_title("val solid | train dashed")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"notears_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # Plot: HSIC (if available)
    if all(s in df.columns for s in ["train_hsic_reg", "val_hsic_reg"]):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="epoch", y="val_hsic_reg", hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y="train_hsic_reg", hue="kfold", ax=ax, legend=False, linestyle=":")
        ax.set_ylabel("HSIC")
        ax.set_yscale("log")
        ax.set_title("val solid | train dashed")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"hsic_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # Plot: Sparsity Cross (if available)
    if all(s in df.columns for s in ["train_sparsity_cross", "val_sparsity_cross"]):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="epoch", y="val_sparsity_cross", hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y="train_sparsity_cross", hue="kfold", ax=ax, legend=False, linestyle=":")
        ax.set_ylabel("Sparsity Cross")
        ax.set_yscale("log")
        ax.set_title("val solid | train dashed")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"sparsity_cross_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # Plot: Sparsity Self (if available)
    if all(s in df.columns for s in ["train_sparsity_self", "val_sparsity_self"]):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="epoch", y="val_sparsity_self", hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y="train_sparsity_self", hue="kfold", ax=ax, legend=False, linestyle=":")
        ax.set_ylabel("Sparsity Self")
        ax.set_yscale("log")
        ax.set_title("val solid | train dashed")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"sparsity_self_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # Plot: Sparsity Total (if available)
    if all(s in df.columns for s in ["train_sparsity_total", "val_sparsity_total"]):
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x="epoch", y="val_sparsity_total", hue="kfold", ax=ax)
        sns.lineplot(data=df, x="epoch", y="train_sparsity_total", hue="kfold", ax=ax, legend=False, linestyle=":")
        ax.set_ylabel("Sparsity Total")
        ax.set_yscale("log")
        ax.set_title("val solid | train dashed")
        plt.tight_layout()
        plt.savefig(join(eval_path_fig, f"sparsity_tot_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
    # Plot: Correlation heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df_corr, ax=ax)
    plt.savefig(join(eval_path_fig, f"metrics_corr_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
        
    return df


def eval_embed(experiment: str, show_plots: bool = True) -> None:
    """
    Evaluate embedding evolution during training.
    
    Analyzes how source (S) and intermediate (X) embeddings evolve during training
    by computing cosine similarities between embedding vectors across epochs.
    This is specific to SingleCausalForecaster architecture with orthogonal embeddings.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        None (results are saved to files and displayed)
        
    Output Files:
        - fig/cosine_similarities_kfold_{k}_{exp_id}.pdf: Cosine similarity evolution per fold
        - fig/final_cosine_similarities_{exp_id}.pdf: Final epoch similarities across folds
        - fig/cosine_similarities_correlation_matrix_{exp_id}.pdf: Correlation heatmap
        - files/emb_df.csv: Embedding data (cached for subsequent runs)
        - files/emb_sim_corr.csv: Cosine similarity correlation matrix
        
    Notes:
        - Computes cosine similarities between S1, S2, S3 (sources) and X1, X2 (intermediates)
        - Results are cached in files/ directory; delete to recompute
        
    Example:
        >>> eval_embed("../experiments/single/local/my_experiment")
    """
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_embedding_evolution")
    
    emb_dataframe_filename = "emb_df.csv"
    emb_sim_corr_filename = "emb_sim_corr.csv"
    embedding_labels_filename = "embedding_labels.json"

    # =========================================================================
    # Embedding labels for AI interpretation
    # =========================================================================
    
    # TODO: Dataset-specific variable mappings are hard-coded below.
    # TODO: When adding new datasets, create a new entry in DATASET_EMBEDDING_MAPPINGS.
    # TODO: Consider loading these from the dataset folder (e.g., data/{dataset}/variable_labels.json)
    # Note: scm6 (non-linear) and scm7 (linear) share the same causal structure
    _SCM6_FAMILY_EMBEDDING_MAPPING = {
        "source_variables": ["S1", "S2", "S3"],
        "intermediate_variables": ["X1", "X2"],
        "output_variables": ["Y1", "Y2"],
        "cosine_similarity_pairs": {
            "cos_S1_X1": "Cosine similarity between S1 and X1 embeddings",
            "cos_S1_X2": "Cosine similarity between S1 and X2 embeddings",
            "cos_S2_X1": "Cosine similarity between S2 and X1 embeddings",
            "cos_S2_X2": "Cosine similarity between S2 and X2 embeddings",
            "cos_S3_X1": "Cosine similarity between S3 and X1 embeddings",
            "cos_S3_X2": "Cosine similarity between S3 and X2 embeddings",
            "cos_X1_X2": "Cosine similarity between X1 and X2 embeddings",
        },
        "expected_causal_relations": {
            "cos_S1_X1": "Should be high (S1 → X1 in true DAG)",
            "cos_S2_X2": "Should be high (S2 → X2 in true DAG)",
            "cos_S3_X2": "Should be high (S3 → X2 in true DAG)",
            "cos_S1_X2": "Should be low (no direct edge S1 → X2)",
            "cos_S2_X1": "Should be low (no direct edge S2 → X1)",
            "cos_S3_X1": "Should be low (no direct edge S3 → X1)",
        },
    }
    DATASET_EMBEDDING_MAPPINGS = {
        "scm6": _SCM6_FAMILY_EMBEDDING_MAPPING,  # Non-linear SCM
        "scm7": _SCM6_FAMILY_EMBEDDING_MAPPING,  # Linear SCM (same causal structure)
        # TODO: Add more datasets here as they are created
    }
    
    embedding_labels = {
        "description": "Embedding evolution analysis - tracking cosine similarities between variable embeddings during training",
        "interpretation": {
            "high_cosine_similarity": "Variables with similar embeddings are expected to have causal relationships",
            "low_cosine_similarity": "Variables with orthogonal embeddings should not have direct causal edges",
            "evolution_over_epochs": "Embeddings typically start random and converge to reflect causal structure",
        },
        "embedding_types": {
            "embedding_S": "Source variable embeddings (OrthogonalMaskEmbedding for SingleCausal)",
            "embedding_X": "Intermediate variable embeddings (ModularEmbedding)",
        },
        "column_documentation": {
            "kfold": "Cross-validation fold identifier (k_0, k_1, ...)",
            "epoch": "Training epoch number",
            "S1, S2, S3": "Embedding vectors for source variables (flattened)",
            "X1, X2": "Embedding vectors for intermediate variables (flattened)",
            "cos_*_*": "Cosine similarity between two variable embeddings",
        },
    }
    
    # Try to get dataset from config to add dataset-specific labels
    from os import listdir as _listdir
    config_files = [f for f in _listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_name = None
    if config_files:
        try:
            config = OmegaConf.load(join(experiment, config_files[0]))
            dataset_name = config.get("data", {}).get("dataset")
        except Exception:
            pass
    
    # Add dataset-specific mapping if available
    if dataset_name and dataset_name in DATASET_EMBEDDING_MAPPINGS:
        embedding_labels["variable_mapping"] = DATASET_EMBEDDING_MAPPINGS[dataset_name]
        embedding_labels["dataset"] = dataset_name
    else:
        # TODO: Unknown dataset - using generic placeholder
        embedding_labels["variable_mapping"] = {
            "note": f"No variable mapping defined for dataset '{dataset_name}'. Add to DATASET_EMBEDDING_MAPPINGS.",
        }
        embedding_labels["dataset"] = dataset_name or "unknown"
    
    _save_variable_labels(eval_path_files, embedding_labels, embedding_labels_filename)

    # Save README with column documentation
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation folder contains evaluations of the learned embeddings.",
        files_info={
            emb_dataframe_filename: "Embedding data with cosine similarities (CSV)",
            emb_sim_corr_filename: "Correlation matrix of cosine similarities (CSV)",
            embedding_labels_filename: "Variable descriptions and interpretation guide (JSON)",
        },
        column_documentation=embedding_labels["column_documentation"]
    )
    
    # Create cline notes template
    _create_cline_template(eval_path_cline, "eval_embedding_evolution", exp_id)

    print(f"Experiment ID: {exp_id}")

    # Helper functions for embedding processing
    # TODO: Hard-coded for SCM with 3 source variables (S1, S2, S3) and 2 intermediate variables (X1, X2)
    # TODO: Make num_vars configurable based on dataset/experiment config
    def _compute_srow(row):
        """Compute source embedding vectors from orthogonal mask embedding components."""
        num_vars = 3  # TODO: Hard-coded number of source variables
        ab = np.asarray(row["embedding_S_value_embedding_weight"]) + \
             np.asarray(row["embedding_S_value_embedding_bias"])
        c = np.asarray(row["embedding_S_binary_masks"])
        L = ab.shape[0]

        if c.size != num_vars * L:
            raise ValueError(f"Expected c length {num_vars*L}, got {c.size}")

        c3 = c.reshape(num_vars, L)
        return c3

    def _unpack(row, label, num_vars):
        """Unpack flattened weight array into per-variable vectors."""
        arr = np.asarray(row[label])
        L = arr.size // num_vars
        return tuple(arr.reshape(num_vars, L))

    def _rowwise_cosine(df, col1, col2, eps=1e-12):
        """Compute row-wise cosine similarity between two vector columns."""
        X = np.stack(df[col1].to_numpy())
        Y = np.stack(df[col2].to_numpy())
        num = np.einsum("ij,ij->i", X, Y)
        den = (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)).clip(min=eps)
        return num / den

    # Load or compute embedding data
    if exists(join(eval_path_files, emb_dataframe_filename)):
        df_scm = pd.read_csv(join(eval_path_files, emb_dataframe_filename))
    else:
        df = load_embeddings_evolution(experiment)

        group_cols = ["kfold", "epoch"]
        source_emb_name_single = [
            "embedding_S_value_embedding_weight",
            "embedding_S_value_embedding_bias",
            "embedding_S_binary_masks"
        ]
        source_emb = df.set_index("embedding_name").loc[source_emb_name_single].reset_index()

        # Process source embeddings
        from functools import partial
        gS = (
            source_emb
            .groupby(group_cols + ["embedding_name"])["weight"]
            .first()
            .unstack("embedding_name")
        )
        gS["temp_res"] = gS.apply(_compute_srow, axis=1)
        # TODO: Hard-coded variable names S1, S2, S3 - should be dynamically generated
        gS[["S1", "S2", "S3"]] = gS.apply(
            partial(_unpack, label="temp_res", num_vars=3), 
            axis=1, 
            result_type="expand"
        )
        gS = gS.drop(columns=source_emb_name_single + ["temp_res"])
        df_S = gS.reset_index()

        # Process intermediate X embeddings
        # TODO: Hard-coded embedding name 'embedding_X_var1_nn_embedding_embedding_weight'
        # TODO: Hard-coded num_vars=8 and indices [4, 5] for X1, X2 extraction
        df_X = df.set_index("embedding_name").loc['embedding_X_var1_nn_embedding_embedding_weight'].reset_index()
        df_X[["X1", "X2"]] = df_X.apply(
            partial(_unpack, label="weight", num_vars=8), 
            axis=1, 
            result_type="expand"
        )[[4, 5]]
        df_X = df_X.drop(columns=["embedding_name", "weight", "type", "shape", "component"])

        df_scm = pd.concat([df_S.set_index(group_cols), df_X.set_index(group_cols)], axis=1).reset_index()

    # Calculate cosine similarities
    # TODO: Hard-coded cosine similarity pairs for S1, S2, S3 and X1, X2
    # TODO: Should dynamically generate pairs based on number of source/intermediate variables
    df_scm["cos_S1_X1"] = _rowwise_cosine(df_scm, "S1", "X1")
    df_scm["cos_S1_X2"] = _rowwise_cosine(df_scm, "S1", "X2")
    df_scm["cos_S2_X1"] = _rowwise_cosine(df_scm, "S2", "X1")
    df_scm["cos_S2_X2"] = _rowwise_cosine(df_scm, "S2", "X2")
    df_scm["cos_S3_X1"] = _rowwise_cosine(df_scm, "S3", "X1")
    df_scm["cos_S3_X2"] = _rowwise_cosine(df_scm, "S3", "X2")
    df_scm["cos_X1_X2"] = _rowwise_cosine(df_scm, "X1", "X2")

    # TODO: Hard-coded list of variable pairs
    var_pairs = ["cos_S1_X1", "cos_S1_X2", "cos_S2_X1", "cos_S2_X2", 
                 "cos_S3_X1", "cos_S3_X2", "cos_X1_X2"]
    
    # Save correlation matrix
    df_corr = df_scm[var_pairs].corr().abs()
    ranked = df_corr.unstack().sort_values(ascending=False)
    ranked[ranked < 1].to_csv(join(eval_path_files, emb_sim_corr_filename))
    
    # Plot: Cosine similarity evolution per fold
    group_cols = ["kfold", "epoch"]
    df_sim_plot = df_scm.melt(
        id_vars=group_cols, 
        value_vars=var_pairs, 
        var_name="variables", 
        value_name="cosine_similarity"
    )
    
    for k in df_sim_plot["kfold"].unique():
        fig, ax = plt.subplots()
        sns.lineplot(
            df_sim_plot.set_index("kfold").loc[k], 
            x="epoch", 
            y="cosine_similarity", 
            hue="variables", 
            ax=ax
        )
        plt.title(f"Fold: {k}")
        plt.savefig(join(eval_path_fig, f"cosine_similarities_kfold_{k}_{exp_id}.pdf"))
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Plot: Final epoch similarities
    fig, ax = plt.subplots()
    max_epoch = max(df_sim_plot["epoch"].unique())
    sns.scatterplot(
        df_sim_plot[df_sim_plot["epoch"] == max_epoch], 
        y="variables", 
        x="cosine_similarity", 
        hue="kfold", 
        s=200, 
        alpha=0.6, 
        ax=ax
    )
    plt.savefig(join(eval_path_fig, f"final_cosine_similarities_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot: Correlation matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df_corr, ax=ax)
    plt.savefig(join(eval_path_fig, f"cosine_similarities_correlation_matrix_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()


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
        dict: DAG recovery metrics with keys like "soft_hamming_cross", "soft_hamming_self"
        
    Output Files:
        - fig/attention_scores_{exp_id}.pdf: Attention score heatmaps for all folds
        - fig/attention_drift_{exp_id}.pdf: Attention evolution over training
        - fig/dag_comparison_{exp_id}.pdf: Learned vs true DAG comparison heatmaps
        - files/final_scores/: Saved attention data (can be reloaded quickly)
        - files/scores_evol.csv: Attention evolution data
        - files/dag_metrics.json: DAG recovery metrics
        
    Notes:
        - Supports TransformerForecaster, StageCausalForecaster, and SingleCausalForecaster
        - Results are cached; delete files/ contents to recompute
        - DAG metrics compare phi (if available) or mean attention scores to true DAG masks
        
    Example:
        >>> metrics = eval_attention_scores("../experiments/single/local/my_experiment")
        >>> print(f"Soft Hamming (cross): {metrics['soft_hamming_cross']:.4f}")
    """
    import json
    from os import listdir
    
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
    
    # TODO: Dataset-specific variable mappings are hard-coded below.
    # TODO: When adding new datasets, create a new entry in DATASET_VARIABLE_MAPPINGS.
    # TODO: Consider loading these from the dataset folder (e.g., data/{dataset}/variable_labels.json)
    # Note: scm6 (non-linear) and scm7 (linear) share the same causal structure
    _SCM6_FAMILY_MAPPING = {
        "S1": "Source variable 1 (index 0 in S) - exogenous",
        "S2": "Source variable 2 (index 1 in S) - exogenous", 
        "S3": "Source variable 3 (index 2 in S) - exogenous",
        "X1": "Intermediate variable 1 (index 0 in X) - X1 ← S1",
        "X2": "Intermediate variable 2 (index 1 in X) - X2 ← S2, S3, X1",
        "Y1": "Output variable 1 (index 0 in Y)",
        "Y2": "Output variable 2 (index 1 in Y)",
        "dag_structure": "S1→X1, S2→X2, S3→X2, X1→X2 (true causal DAG)",
    }
    DATASET_VARIABLE_MAPPINGS = {
        "scm6": _SCM6_FAMILY_MAPPING,  # Non-linear SCM
        "scm7": _SCM6_FAMILY_MAPPING,  # Linear SCM (same causal structure as scm6)
        # TODO: Add more datasets here as they are created
    }
    
    # Try to get dataset from config to add dataset-specific labels
    from os import listdir
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_name = None
    if config_files:
        try:
            config = OmegaConf.load(join(experiment, config_files[0]))
            dataset_name = config.get("data", {}).get("dataset")
        except Exception:
            pass
    
    # Add dataset-specific mapping if available
    if dataset_name and dataset_name in DATASET_VARIABLE_MAPPINGS:
        attention_labels["variable_mapping"] = DATASET_VARIABLE_MAPPINGS[dataset_name]
        attention_labels["dataset"] = dataset_name
    else:
        # TODO: Unknown dataset - using generic placeholder
        attention_labels["variable_mapping"] = {
            "note": f"No variable mapping defined for dataset '{dataset_name}'. Add to DATASET_VARIABLE_MAPPINGS.",
        }
        attention_labels["dataset"] = dataset_name or "unknown"
    
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
    plt.savefig(join(eval_path_fig, f"attention_scores_{exp_id}.pdf"))
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
    plt.savefig(join(eval_path_fig, f"attention_drift_{exp_id}.pdf"))
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
        # S → X (cross) and X → X (self)
        blocks_to_eval = [
            ("dec_cross", "decoder_cross", "dec_cross"),  # (att_key, phi_key, mask_type)
            ("dec_self", "decoder", "dec_self"),
        ]
    elif architecture == "StageCausalForecaster":
        # decoder1: S → X, decoder2: X → Y
        blocks_to_eval = [
            ("decoder1_cross", "decoder1_cross", "dec1_cross"),
            ("decoder1_self", "decoder1", "dec1_self"),
            ("decoder2_cross", "decoder2_cross", "dec2_cross"),
            ("decoder2_self", "decoder2", "dec2_self"),
        ]
    elif architecture == "TransformerForecaster":
        # encoder, decoder, cross
        blocks_to_eval = [
            ("cross", "cross", "dec_cross"),
            ("decoder", "decoder", "dec_self"),
        ]
    else:
        print(f"Warning: Unknown architecture {architecture}, skipping DAG metrics")
        return {}
    
    # Store per-fold comparison data for plotting
    per_fold_comparison_data = []  # List of (fold_name, block, learned_dag, true_dag, soft_hamming, source)
    
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
            dag_metrics[f"{metric_key}_source"] = source  # "phi" or "attention"
            
            print(f"    Statistics: best={np.min(fold_sh_array):.4f}, mean={np.mean(fold_sh_array):.4f}, worst={np.max(fold_sh_array):.4f}, std={np.std(fold_sh_array):.4f}")
    
    # Save DAG metrics to JSON
    with open(join(eval_path_files, dag_metrics_filename), 'w') as f:
        json.dump(dag_metrics, f, indent=2)
    print(f"  Saved: {dag_metrics_filename}")
    
    # Plot: Per-fold DAG comparison heatmaps (one PDF per fold)
    if per_fold_comparison_data:
        # Group by fold_name
        from collections import defaultdict
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
            plt.savefig(join(eval_path_fig, f"dag_comparison_{fold_name}_{exp_id}.pdf"))
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    return dag_metrics


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
        show_images: If True, display plots interactively. If False (default), 
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
    
    # Setup default interventions if not provided
    # TODO: Hard-coded default interventions for 3 source variables (S1, S2, S3)
    # TODO: Should read number of source variables from experiment config
    if interventions is None:
        interventions = [
            (create_intervention_fn(interventions={1: 0}), "S1=0"),
            (create_intervention_fn(interventions={2: 0}), "S2=0"),
            (create_intervention_fn(interventions={3: 0}), "S3=0"),
            (create_intervention_fn(interventions={1: 1}), "S1=1"),
            (create_intervention_fn(interventions={2: 1}), "S2=1"),
            (create_intervention_fn(interventions={3: 1}), "S3=1"),
            (create_intervention_fn(interventions={1: -1}), "S1=-1"),
            (create_intervention_fn(interventions={2: -1}), "S2=-1"),
            (create_intervention_fn(interventions={3: -1}), "S3=-1"),
        ]
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
    
    # Variable label mapping
    # TODO: Hard-coded variable label mapping for SCM with S1-S3, X1-X2, Y1-Y2
    # TODO: Should be dynamically generated based on dataset/experiment config
    trg_feat_1_map = {
        1: "S1",
        2: "S2",
        3: "S3",
        4: "X1",
        5: "X2",
        6: "Y1",
        7: "Y2",
    }
    
    # LaTeX labels for plots
    trg_feat_1_latex = {
        1: "$S_1$",
        2: "$S_2$",
        3: "$S_3$",
        4: "$X_1$",
        5: "$X_2$",
        6: "$Y_1$",
        7: "$Y_2$",
    }
    
    # Save variable labels to JSON for AI/programmatic access
    import json
    variable_labels = {
        "trg_feat_1_to_name": trg_feat_1_map,
        "trg_feat_1_to_latex": trg_feat_1_latex,
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
    # TODO: Hard-coded expected effects for scm6 dataset
    # TODO: When adding new datasets, add their expected_effects to DATASET_EXPECTED_EFFECTS
    # Note: scm6 (non-linear) and scm7 (linear) share the same causal structure
    _SCM6_FAMILY_EXPECTED_EFFECTS = {
        # Format: (intervention_var, target_var): expected_to_have_effect (True/False)
        # True DAG: S1→X1, S2→X2, S3→X2, X1→X2
        ("S1", "X1"): True,   # S1 causes X1
        ("S1", "X2"): True,   # S1 → X1 → X2 (indirect)
        ("S2", "X1"): False,  # S2 does not cause X1
        ("S2", "X2"): True,   # S2 causes X2
        ("S3", "X1"): False,  # S3 does not cause X1
        ("S3", "X2"): True,   # S3 causes X2
        # S variables don't affect themselves
        ("S1", "S1"): False,  # Intervention doesn't affect own prediction
        ("S1", "S2"): False,
        ("S1", "S3"): False,
        ("S2", "S1"): False,
        ("S2", "S2"): False,
        ("S2", "S3"): False,
        ("S3", "S1"): False,
        ("S3", "S2"): False,
        ("S3", "S3"): False,
    }
    DATASET_EXPECTED_EFFECTS = {
        "scm6": _SCM6_FAMILY_EXPECTED_EFFECTS,  # Non-linear SCM
        "scm7": _SCM6_FAMILY_EXPECTED_EFFECTS,  # Linear SCM (same causal structure)
        # TODO: Add more datasets here as they are created
    }
    
    # Try to determine dataset from config
    from os import listdir as _list_dir
    config_files_int = [f for f in _list_dir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    dataset_int = None
    if config_files_int:
        try:
            config_int = OmegaConf.load(join(experiment, config_files_int[0]))
            dataset_int = config_int.get("data", {}).get("dataset")
        except Exception:
            pass
    
    invariance_filename = "intervention_invariance.json"
    invariance_threshold = 0.05  # Threshold for "no effect" (mean abs deviation)
    
    if dataset_int and dataset_int in DATASET_EXPECTED_EFFECTS:
        expected_effects = DATASET_EXPECTED_EFFECTS[dataset_int]
        
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
                "dataset": dataset_int,
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
        print(f"  Skipping invariance test: no expected effects defined for dataset '{dataset_int}'")

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

            plt.savefig(join(eval_path_fig, f"dev_{var_label}_{k}_{exp_id}.pdf"))
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    return df, df_dev


def eval_embedding_dag_correlation(experiment: str, show_plots: bool = True) -> dict:
    """
    Evaluate correlation between embedding similarity and learned DAG structure.
    
    This function tests H6: whether causally-linked variables have more similar
    embeddings than non-linked variables. It combines data from:
    - eval_embed: cosine similarities between variable embeddings
    - eval_attention_scores: learned DAG structure (phi or attention)
    
    For each variable pair (e.g., S1-X1), it correlates:
    - Embedding cosine similarity (from eval_embed)
    - Learned edge probability (from eval_attention_scores phi tensor)
    - True edge indicator (from dataset DAG masks)
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        show_plots: If True (default), display plots interactively. If False, only save to files.
        
    Returns:
        dict: Correlation results with keys:
            - embedding_dag_correlation: Pearson correlation between embedding sim and learned DAG
            - embedding_true_dag_correlation: Correlation between embedding sim and true DAG
            - separation_score: Difference in mean embedding sim (causal vs non-causal pairs)
            
    Output Files:
        - fig/embedding_dag_scatter_{exp_id}.pdf: Scatter plot of embedding sim vs DAG prob
        - fig/embedding_separation_{exp_id}.pdf: Box plot comparing causal vs non-causal pairs
        - files/embedding_dag_correlation.json: Detailed correlation results
        - files/embedding_dag_data.csv: Raw data for further analysis
        
    Notes:
        - Requires eval_embed and eval_attention_scores to be run first
        - Currently supports SingleCausalForecaster with scm6 dataset
        
    Example:
        >>> results = eval_embedding_dag_correlation("../experiments/single/local/my_experiment")
        >>> print(f"Embedding-DAG correlation: {results['embedding_dag_correlation']:.3f}")
    """
    import json
    from os import listdir
    from scipy import stats
    
    # Setup directories
    eval_path_root, eval_path_fig, eval_path_files, eval_path_cline, exp_id = \
        _setup_eval_directories(experiment, "eval_embedding_dag_correlation")
    
    correlation_filename = "embedding_dag_correlation.json"
    data_filename = "embedding_dag_data.csv"
    
    # =========================================================================
    # Load prerequisite data
    # =========================================================================
    
    # Check if eval_embed was run
    emb_data_path = join(experiment, "eval", "eval_embedding_evolution", "files", "emb_df.csv")
    if not exists(emb_data_path):
        print(f"Error: eval_embed must be run first. Missing: {emb_data_path}")
        print("Run: eval_embed(experiment)")
        return {}
    
    # Check if eval_attention_scores was run
    dag_metrics_path = join(experiment, "eval", "eval_attention_scores", "files", "dag_metrics.json")
    att_data_path = join(experiment, "eval", "eval_attention_scores", "files", "final_scores")
    if not exists(dag_metrics_path):
        print(f"Error: eval_attention_scores must be run first. Missing: {dag_metrics_path}")
        print("Run: eval_attention_scores(experiment)")
        return {}
    
    # Load embedding cosine similarities (final epoch, mean across folds)
    df_emb = pd.read_csv(emb_data_path)
    
    # Get final epoch data
    final_epoch = df_emb["epoch"].max()
    df_final = df_emb[df_emb["epoch"] == final_epoch]
    
    # Compute mean cosine similarities across folds for final epoch
    # TODO: Hard-coded cosine similarity column names for scm6
    cos_cols = ["cos_S1_X1", "cos_S1_X2", "cos_S2_X1", "cos_S2_X2", "cos_S3_X1", "cos_S3_X2"]
    mean_cos_sims = {}
    for col in cos_cols:
        if col in df_final.columns:
            mean_cos_sims[col] = df_final[col].mean()
    
    if not mean_cos_sims:
        print("Error: No cosine similarity columns found in embedding data")
        return {}
    
    print(f"Loaded embedding similarities from final epoch {final_epoch}")
    
    # Load learned DAG from attention scores
    with open(dag_metrics_path, 'r') as f:
        dag_metrics = json.load(f)
    
    # Load attention data to get phi values
    attention_data = load_attention_data_from_file(att_data_path)
    
    # Get architecture type
    architecture = attention_data.architecture_type
    
    # Extract learned DAG (phi or attention) - mean across folds
    # TODO: Hard-coded for SingleCausalForecaster dec_cross (S→X)
    if architecture == "SingleCausalForecaster":
        phi_key = "decoder_cross"
        att_key = "dec_cross"
    else:
        print(f"Warning: eval_embedding_dag_correlation not yet implemented for {architecture}")
        return {}
    
    learned_dag, source = _get_learned_dag(attention_data, att_key, phi_key)
    
    if learned_dag is None:
        print("Error: Could not extract learned DAG from attention data")
        return {}
    
    print(f"Loaded learned DAG from {source}: shape={learned_dag.shape}")
    
    # Load true DAG mask
    config_files = [f for f in listdir(experiment) if f.startswith("config") and f.endswith(".yaml")]
    if not config_files:
        print("Error: No config file found")
        return {}
    
    config = OmegaConf.load(join(experiment, config_files[0]))
    dataset = config.get("data", {}).get("dataset")
    
    datadir_path = join(root_path, "data")
    true_dag = _load_true_dag_mask(datadir_path, dataset, "dec_cross")
    
    if true_dag is None:
        print("Error: Could not load true DAG mask")
        return {}
    
    print(f"Loaded true DAG: shape={true_dag.shape}")
    
    # =========================================================================
    # Map cosine similarities to DAG edges
    # =========================================================================
    
    # TODO: Hard-coded mapping for scm6 (S1, S2, S3) → (X1, X2)
    # Format: cos_col → (target_idx, source_idx) in the DAG matrix
    # DAG matrix is (n_targets=2, n_sources=3) where rows=X, cols=S
    COS_TO_DAG_MAP = {
        "cos_S1_X1": (0, 0),  # X1 ← S1
        "cos_S2_X1": (0, 1),  # X1 ← S2
        "cos_S3_X1": (0, 2),  # X1 ← S3
        "cos_S1_X2": (1, 0),  # X2 ← S1
        "cos_S2_X2": (1, 1),  # X2 ← S2
        "cos_S3_X2": (1, 2),  # X2 ← S3
    }
    
    # Build data for correlation
    records = []
    for cos_col, (target_idx, source_idx) in COS_TO_DAG_MAP.items():
        if cos_col not in mean_cos_sims:
            continue
        
        cos_sim = mean_cos_sims[cos_col]
        learned_edge = learned_dag[target_idx, source_idx]
        true_edge = true_dag[target_idx, source_idx]
        
        records.append({
            "pair": cos_col.replace("cos_", ""),
            "source_var": cos_col.split("_")[1],  # e.g., "S1"
            "target_var": cos_col.split("_")[2],  # e.g., "X1"
            "embedding_cosine_sim": cos_sim,
            "learned_dag_prob": learned_edge,
            "true_dag_edge": int(true_edge),
            "is_causal": bool(true_edge > 0.5),
        })
    
    df_data = pd.DataFrame(records)
    df_data.to_csv(join(eval_path_files, data_filename), index=False)
    print(f"Saved: {data_filename}")
    
    # =========================================================================
    # Compute correlations
    # =========================================================================
    
    # Correlation: embedding similarity vs learned DAG probability
    corr_emb_learned, pval_emb_learned = stats.pearsonr(
        df_data["embedding_cosine_sim"], 
        df_data["learned_dag_prob"]
    )
    
    # Correlation: embedding similarity vs true DAG (binary)
    corr_emb_true, pval_emb_true = stats.pearsonr(
        df_data["embedding_cosine_sim"], 
        df_data["true_dag_edge"]
    )
    
    # Separation score: mean embedding sim for causal pairs - mean for non-causal
    causal_pairs = df_data[df_data["is_causal"]]
    non_causal_pairs = df_data[~df_data["is_causal"]]
    
    mean_causal = causal_pairs["embedding_cosine_sim"].mean() if len(causal_pairs) > 0 else 0
    mean_non_causal = non_causal_pairs["embedding_cosine_sim"].mean() if len(non_causal_pairs) > 0 else 0
    separation_score = mean_causal - mean_non_causal
    
    # Statistical test: are causal pairs significantly more similar?
    if len(causal_pairs) > 0 and len(non_causal_pairs) > 0:
        t_stat, t_pval = stats.ttest_ind(
            causal_pairs["embedding_cosine_sim"],
            non_causal_pairs["embedding_cosine_sim"]
        )
    else:
        t_stat, t_pval = None, None
    
    # =========================================================================
    # Save results
    # =========================================================================
    
    results = {
        "dataset": dataset,
        "architecture": architecture,
        "dag_source": source,
        "n_pairs": len(df_data),
        "n_causal_pairs": len(causal_pairs),
        "n_non_causal_pairs": len(non_causal_pairs),
        "embedding_dag_correlation": float(corr_emb_learned),
        "embedding_dag_correlation_pval": float(pval_emb_learned),
        "embedding_true_dag_correlation": float(corr_emb_true),
        "embedding_true_dag_correlation_pval": float(pval_emb_true),
        "mean_embedding_sim_causal": float(mean_causal),
        "mean_embedding_sim_non_causal": float(mean_non_causal),
        "separation_score": float(separation_score),
        "separation_ttest_stat": float(t_stat) if t_stat is not None else None,
        "separation_ttest_pval": float(t_pval) if t_pval is not None else None,
        "pair_details": records,
    }
    
    with open(join(eval_path_files, correlation_filename), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {correlation_filename}")
    
    # =========================================================================
    # Plots
    # =========================================================================
    
    # Plot 1: Scatter of embedding similarity vs learned DAG probability
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = df_data["is_causal"].map({True: "green", False: "red"})
    ax.scatter(df_data["learned_dag_prob"], df_data["embedding_cosine_sim"], 
               c=colors, s=200, alpha=0.7)
    
    # Add pair labels
    for _, row in df_data.iterrows():
        ax.annotate(row["pair"], (row["learned_dag_prob"], row["embedding_cosine_sim"]),
                   fontsize=10, ha='center', va='bottom')
    
    ax.set_xlabel("Learned DAG Probability (phi)")
    ax.set_ylabel("Embedding Cosine Similarity")
    ax.set_title(f"Embedding vs DAG Correlation\nr={corr_emb_learned:.3f}, p={pval_emb_learned:.3f}")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Causal pair'),
        Patch(facecolor='red', alpha=0.7, label='Non-causal pair'),
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"embedding_dag_scatter_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot 2: Box plot comparing causal vs non-causal pairs
    fig, ax = plt.subplots(figsize=(6, 6))
    
    box_data = [
        non_causal_pairs["embedding_cosine_sim"].values,
        causal_pairs["embedding_cosine_sim"].values,
    ]
    bp = ax.boxplot(box_data, labels=["Non-causal", "Causal"], patch_artist=True)
    bp["boxes"][0].set_facecolor("red")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("green")
    bp["boxes"][1].set_alpha(0.5)
    
    # Overlay individual points
    for i, (data, color) in enumerate([(non_causal_pairs, "red"), (causal_pairs, "green")]):
        x = np.ones(len(data)) * (i + 1) + np.random.normal(0, 0.05, len(data))
        ax.scatter(x, data["embedding_cosine_sim"], c=color, alpha=0.7, s=100)
    
    ax.set_ylabel("Embedding Cosine Similarity")
    ax.set_title(f"Embedding Separation by Causality\nSeparation={separation_score:.3f}, p={t_pval:.3f}" if t_pval else "Embedding Separation by Causality")
    
    plt.tight_layout()
    plt.savefig(join(eval_path_fig, f"embedding_separation_{exp_id}.pdf"))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # =========================================================================
    # Save README and cline template
    # =========================================================================
    
    _save_readme(
        eval_path_root, eval_path_cline, eval_path_files, eval_path_fig,
        description="This evaluation analyzes correlation between embedding similarity and DAG structure (H6).",
        files_info={
            correlation_filename: "Correlation statistics and summary results (JSON)",
            data_filename: "Raw data: embedding similarity, learned DAG, true DAG per pair (CSV)",
        },
        column_documentation={
            "pair": "Variable pair name (e.g., 'S1_X1')",
            "embedding_cosine_sim": "Cosine similarity between variable embeddings (from eval_embed)",
            "learned_dag_prob": "Learned edge probability from phi tensor",
            "true_dag_edge": "True DAG edge indicator (0 or 1)",
            "is_causal": "Whether this is a true causal pair",
        }
    )
    
    _create_cline_template(eval_path_cline, "eval_embedding_dag_correlation", exp_id)
    
    # Print summary
    print(f"\n=== Embedding-DAG Correlation Summary ===")
    print(f"Embedding vs Learned DAG: r={corr_emb_learned:.3f} (p={pval_emb_learned:.3f})")
    print(f"Embedding vs True DAG:    r={corr_emb_true:.3f} (p={pval_emb_true:.3f})")
    print(f"Mean embedding sim (causal):     {mean_causal:.3f}")
    print(f"Mean embedding sim (non-causal): {mean_non_causal:.3f}")
    print(f"Separation score:                {separation_score:.3f}")
    if t_pval is not None:
        print(f"Separation t-test p-value:       {t_pval:.3f}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Run batch evaluation on multiple experiments.
    
    Edit the lists below to configure:
    - experiments: List of individual experiment paths to evaluate
    - experiments_folders: List of folders containing experiments (all subdirectories will be added)
    - eval_functions: List of evaluation functions to run (in order)
    """
    from os import listdir
    from os.path import isdir
    
    # -------------------------------------------------------------------------
    # CONFIGURE: Individual experiments to evaluate
    # -------------------------------------------------------------------------
    exp_dir = join(root_path, "experiments")
    experiments: List[str] = [
        join(exp_dir, "single/euler/single_Lie_CC_scm6_54803384"),
        join(exp_dir, "single/euler/single_Lie_CC_scm6_54834710"),
        join(exp_dir, "single/euler/single_Lie_CC_scm6_54916195"),
        join(exp_dir, "single/euler/single_Lie_CC_scm6_54946595"),
        join(exp_dir, "single/euler/single_Lie_CC_scm6_55015699"),
        join(exp_dir, "single/euler/single_Lie_CC_scm7_55058272"),
        # Add more individual experiment paths here...
    ]
    
    # -------------------------------------------------------------------------
    # CONFIGURE: Folders containing experiments (all subdirectories will be added)
    # -------------------------------------------------------------------------
    experiments_folders: List[str] = [
        #join(exp_dir, "single/euler")
        # join(exp_dir, "single/local"),  # Uncomment to add all experiments in this folder
        # join(exp_dir, "stage/local"),   # Uncomment to add all experiments in this folder
    ]
    
    # Discover experiments from folders and add to experiments list
    for folder in experiments_folders:
        if exists(folder) and isdir(folder):
            for subdir in listdir(folder):
                subdir_path = join(folder, subdir)
                if isdir(subdir_path):
                    experiments.append(subdir_path)
    
    # -------------------------------------------------------------------------
    # CONFIGURE: Evaluation functions to run (in order)
    # -------------------------------------------------------------------------
    eval_functions = [
        eval_train_metrics,
        eval_attention_scores,
        eval_embed,
        eval_interventions,
    ]
    
    # -------------------------------------------------------------------------
    # Run evaluations (show_plots=False for batch mode)
    # -------------------------------------------------------------------------
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp}")
        print('='*60)
        
        # First, fix any tensor strings in kfold_summary.json
        print(f"\n--- Running fix_kfold_summary ---")
        try:
            fix_kfold_summary(exp)
        except Exception as e:
            print(f"Error in fix_kfold_summary: {e}")
        
        # Enrich kfold_summary.json with aggregated statistics
        print(f"\n--- Running enrich_kfold_summary ---")
        try:
            enrich_kfold_summary(exp)
        except Exception as e:
            print(f"Error in enrich_kfold_summary: {e}")
        
        # Then run all evaluation functions
        for eval_fn in eval_functions:
            print(f"\n--- Running {eval_fn.__name__} ---")
            try:
                # Pass show_plots=False for batch mode (no interactive display)
                eval_fn(exp, show_plots=False)
            except Exception as e:
                print(f"Error in {eval_fn.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        # Finally, update the experiments manifest
        print(f"\n--- Updating experiments manifest ---")
        try:
            update_experiments_manifest(exp)
        except Exception as e:
            print(f"Error updating manifest: {e}")
