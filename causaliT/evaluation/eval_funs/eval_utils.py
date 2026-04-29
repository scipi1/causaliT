"""
Shared utility functions for CausaliT evaluation.

This module provides helper functions used across all evaluation modules:
- Directory setup and file management
- Checkpoint discovery
- DAG recovery metrics
- Plotting settings
"""

import re
import json
from os.path import dirname, abspath, join, exists
from os import makedirs, listdir
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Setup root path for local imports
# Go up FOUR levels: eval_funs -> evaluation -> causaliT -> project root
root_path = dirname(dirname(dirname(dirname(abspath(__file__)))))

# =============================================================================
# Plot Configuration
# =============================================================================
# Default plot format for all evaluation functions
# Change to "pdf" for publication quality, "png" for faster rendering
DEFAULT_PLOT_FORMAT = "png"


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
# Architecture Registry
# =============================================================================
# Centralized configuration for all supported model architectures.
# This eliminates repetitive architecture-specific if/elif blocks.

ARCHITECTURE_REGISTRY = {
    "SingleCausalForecaster": {
        "attention_keys": ["dec_self", "dec_cross"],
        "phi_keys": ["decoder", "decoder_cross"],
        "blocks_to_eval": [
            # (attention_key, phi_key, mask_type)
            ("dec_cross", "decoder_cross", "dec_cross"),
            ("dec_self", "decoder", "dec_self"),
        ],
        "mec_keys": {
            "cross": ("dec_cross", "decoder_cross"),
            "self": ("dec_self", "decoder"),
        },
    },
    "NoiseAwareCausalForecaster": {
        # Same structure as SingleCausalForecaster
        "attention_keys": ["dec_self", "dec_cross"],
        "phi_keys": ["decoder", "decoder_cross"],
        "blocks_to_eval": [
            ("dec_cross", "decoder_cross", "dec_cross"),
            ("dec_self", "decoder", "dec_self"),
        ],
        "mec_keys": {
            "cross": ("dec_cross", "decoder_cross"),
            "self": ("dec_self", "decoder"),
        },
    },
    "SingleCausalResForecaster": {
        # SVFA dual-residual variant of SingleCausalForecaster.
        # Attention/phi shape is identical: the dual-residual decoder only
        # adds extra value/output projections (`value_projection_struct`,
        # `out_projection_struct`) that affect the X_struct update path —
        # the (Q, K)-derived attention pattern keys are unchanged.
        "attention_keys": ["dec_self", "dec_cross"],
        "phi_keys": ["decoder", "decoder_cross"],
        "blocks_to_eval": [
            ("dec_cross", "decoder_cross", "dec_cross"),
            ("dec_self", "decoder", "dec_self"),
        ],
        "mec_keys": {
            "cross": ("dec_cross", "decoder_cross"),
            "self": ("dec_self", "decoder"),
        },
    },
    "StageCausalForecaster": {
        "attention_keys": ["dec1_self", "dec1_cross", "dec2_self", "dec2_cross"],
        "phi_keys": ["decoder1", "decoder1_cross", "decoder2", "decoder2_cross"],
        "blocks_to_eval": [
            ("decoder1_cross", "decoder1_cross", "dec1_cross"),
            ("decoder1_self", "decoder1", "dec1_self"),
            ("decoder2_cross", "decoder2_cross", "dec2_cross"),
            ("decoder2_self", "decoder2", "dec2_self"),
        ],
        "mec_keys": {
            # For stage causal, focus on first decoder (S→X, X→X)
            "cross": ("decoder1_cross", "decoder1_cross"),
            "self": ("decoder1_self", "decoder1"),
        },
    },
    "TransformerForecaster": {
        "attention_keys": ["encoder", "decoder", "cross"],
        "phi_keys": ["encoder", "decoder", "cross"],
        "blocks_to_eval": [
            ("cross", "cross", "dec_cross"),
            ("decoder", "decoder", "dec_self"),
        ],
        "mec_keys": {
            "cross": ("cross", "cross"),
            "self": ("decoder", "decoder"),
        },
    },
}


def get_architecture_config(architecture_type: str) -> dict:
    """
    Get architecture-specific configuration from the registry.
    
    Args:
        architecture_type: Name of the architecture (e.g., "SingleCausalForecaster")
        
    Returns:
        dict: Configuration with keys:
            - attention_keys: List of attention weight keys to track
            - phi_keys: List of phi tensor keys to extract
            - blocks_to_eval: List of (att_key, phi_key, mask_type) tuples for DAG metrics
            - mec_keys: Dict with 'cross' and 'self' keys for MEC computation
            
    Raises:
        ValueError: If architecture is not in the registry
        
    Example:
        >>> config = get_architecture_config("SingleCausalForecaster")
        >>> config["attention_keys"]
        ['dec_self', 'dec_cross']
    """
    if architecture_type not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture_type}. "
            f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[architecture_type]


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


def _compute_standard_shd(
    learned: np.ndarray,
    true: np.ndarray,
    threshold: float = 0.5,
    is_cross_attention: bool = False,
) -> dict:
    """
    Compute the standard Structural Hamming Distance (Tsamardinos et al., 2006).
    
    This is the metric used by NOTEARS, DAG-GNN, GraN-DAG, DCDI and other
    causal discovery methods:
    
        SHD = missing_edges + extra_edges + reversed_edges
    
    The learned continuous adjacency is first binarized at `threshold`.
    
    Convention for reversed edges: each reversed edge counts as 1 mistake
    (consistent with GraN-DAG / Lachapelle et al. 2020). Reversed edges are
    only counted for square (self-attention) matrices where directionality
    is meaningful. For cross-attention (S→X), edges are always directional
    so only missing/extra are counted.
    
    Args:
        learned: Learned adjacency matrix with values in [0, 1]
        true: True binary adjacency matrix with values in {0, 1}
        threshold: Binarization threshold (default: 0.5)
        is_cross_attention: If True, skip reversal counting (not applicable
                           for rectangular S→X matrices)
        
    Returns:
        dict with keys:
            - shd: Total standard SHD (int)
            - missing: Number of true edges not predicted (int)
            - extra: Number of predicted edges not in true graph (int)
            - reversed: Number of reversed edges (int, 0 for cross-attention)
            - n_true_edges: Number of edges in true graph (int)
            - n_pred_edges: Number of edges in predicted graph (int)
            - threshold: Threshold used for binarization
            
    Example:
        >>> learned = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> true = np.array([[1, 0], [0, 1]])
        >>> result = _compute_standard_shd(learned, true)
        >>> result['shd']
        0  # Perfect after thresholding at 0.5
    """
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")
    
    pred = (learned >= threshold).astype(int)
    true_bin = true.astype(int)
    
    n_true_edges = int(true_bin.sum())
    n_pred_edges = int(pred.sum())
    
    # Missing: true edge absent in prediction
    missing = int(((true_bin == 1) & (pred == 0)).sum())
    # Extra: predicted edge absent in true graph
    extra = int(((true_bin == 0) & (pred == 1)).sum())
    
    # Reversed edges: only for square self-attention matrices
    reversed_edges = 0
    if not is_cross_attention and pred.shape[0] == pred.shape[1]:
        n = pred.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                # pred has i→j but true has j→i
                if pred[i, j] == 1 and pred[j, i] == 0 and true_bin[i, j] == 0 and true_bin[j, i] == 1:
                    reversed_edges += 1
                # pred has j→i but true has i→j
                elif pred[j, i] == 1 and pred[i, j] == 0 and true_bin[j, i] == 0 and true_bin[i, j] == 1:
                    reversed_edges += 1
    
    shd = missing + extra + reversed_edges
    
    return {
        'shd': shd,
        'missing': missing,
        'extra': extra,
        'reversed': reversed_edges,
        'n_true_edges': n_true_edges,
        'n_pred_edges': n_pred_edges,
        'threshold': threshold,
    }


def _compute_zeroness_metrics(learned: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute zero-ness metrics: how close to 0 are the non-edges?
    
    Measures the quality of the learned DAG in terms of edge/non-edge separation.
    For Toeplitz self-attention (initialized at 0.5), non-edges must be actively
    driven towards 0 — this metric captures that progress.
    
    Args:
        learned: Learned adjacency matrix with values in [0, 1]
        true: True binary adjacency matrix with values in {0, 1}
        
    Returns:
        dict with keys:
            - mean_nonedge: Mean |att| where true=0 (lower = better "blackness")
            - max_nonedge: Max |att| where true=0 (worst false positive)
            - mean_edge: Mean att where true=1 (higher = better "whiteness")
            - min_edge: Min att where true=1 (weakest true edge)
            - contrast: mean_edge - mean_nonedge (higher = better separation)
            - n_edges: Number of true edges
            - n_nonedges: Number of true non-edges
            
    Example:
        >>> learned = np.array([[0.9, 0.05], [0.02, 0.85]])
        >>> true = np.array([[1, 0], [0, 1]])
        >>> m = _compute_zeroness_metrics(learned, true)
        >>> m['mean_nonedge']  # ~0.035
        >>> m['contrast']      # ~0.84
    """
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")
    
    edge_mask = true.astype(bool)
    nonedge_mask = ~edge_mask
    
    nonedge_vals = np.abs(learned[nonedge_mask])
    edge_vals = learned[edge_mask]
    
    mean_nonedge = float(nonedge_vals.mean()) if len(nonedge_vals) > 0 else float('nan')
    max_nonedge = float(nonedge_vals.max()) if len(nonedge_vals) > 0 else float('nan')
    mean_edge = float(edge_vals.mean()) if len(edge_vals) > 0 else float('nan')
    min_edge = float(edge_vals.min()) if len(edge_vals) > 0 else float('nan')
    
    contrast = mean_edge - mean_nonedge
    
    return {
        'mean_nonedge': mean_nonedge,
        'max_nonedge': max_nonedge,
        'mean_edge': mean_edge,
        'min_edge': min_edge,
        'contrast': contrast,
        'n_edges': int(edge_mask.sum()),
        'n_nonedges': int(nonedge_mask.sum()),
    }


def load_dataset_metadata(
    datadir_path: str,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    """
    Load dataset metadata from the dataset folder.
    
    The metadata file (dataset_metadata.json) is generated by SCMDataset.generate_ds()
    and contains all information needed for dataset-agnostic evaluation:
    - variable_info: source_labels, input_labels, target_labels, counts
    - causal_structure: direct_edges, transitive_closure, expected_effects
    - variable_index_map: mapping from variable names to indices
    
    Args:
        datadir_path: Path to the data directory
        dataset: Dataset name (e.g., "scm6", "scm1_linear_gaussian")
        
    Returns:
        dict: Dataset metadata, or None if file not found
        
    Example:
        >>> metadata = load_dataset_metadata("../data", "scm6")
        >>> source_labels = metadata["variable_info"]["source_labels"]  # ["S1", "S2", "S3"]
        >>> expected_effects = metadata["causal_structure"]["expected_effects"]
    """
    filepath = join(datadir_path, dataset, "dataset_metadata.json")
    
    if not exists(filepath):
        print(f"Warning: dataset_metadata.json not found at {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load dataset metadata: {e}")
        return None


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


def _compute_dag_confidence(fold_dags: List[np.ndarray]) -> float:
    """
    Compute DAG confidence metric across k-folds.
    
    DAG confidence measures how consistent the learned DAG structure is across
    different cross-validation folds. It is computed as:
    
        dag_confidence = 1 - 2 * mean(std(edge_ij across folds))
    
    Interpretation:
    - 1.0 = Maximum confidence: All folds learned exactly the same DAG
    - 0.0 = Minimum confidence: Maximum disagreement across folds
           (e.g., half folds have edge=0, half have edge=1 for all edges)
    
    The factor of 2 normalizes because the maximum standard deviation for 
    values in [0,1] is 0.5 (when values are perfectly split between 0 and 1).
    
    Args:
        fold_dags: List of learned DAG adjacency matrices, one per fold.
                   Each matrix has shape (n_targets, n_sources) with values in [0,1].
                   
    Returns:
        float: DAG confidence score in [0, 1]
        
    Example:
        >>> # Perfect agreement across 3 folds
        >>> dag1 = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> fold_dags = [dag1, dag1, dag1]
        >>> _compute_dag_confidence(fold_dags)
        1.0
        
        >>> # Some disagreement
        >>> dag2 = np.array([[0.7, 0.3], [0.4, 0.6]])
        >>> fold_dags = [dag1, dag2]
        >>> confidence = _compute_dag_confidence(fold_dags)
        >>> 0 < confidence < 1
        True
    """
    if len(fold_dags) < 2:
        # With fewer than 2 folds, we can't compute meaningful confidence
        # Return 1.0 (no evidence of disagreement)
        return 1.0
    
    # Stack DAGs: (K, n_targets, n_sources)
    stacked = np.stack(fold_dags, axis=0)
    
    # Compute std for each edge across folds: (n_targets, n_sources)
    edge_std = np.std(stacked, axis=0)
    
    # Confidence = 1 - 2 * mean(std)
    # Factor of 2 normalizes since max std for [0,1] values is 0.5
    confidence = 1.0 - 2.0 * np.mean(edge_std)
    
    # Clip to [0, 1] to handle numerical edge cases
    return float(np.clip(confidence, 0.0, 1.0))


def _should_use_log_scale(
    values: np.ndarray,
    order_of_magnitude_threshold: float = 2.0,
    max_negative_fraction: float = 0.5,
) -> Tuple[bool, Optional[float]]:
    """
    Determine if log scale should be used based on data characteristics.
    
    Log scale is recommended when:
    1. Value span exceeds the order of magnitude threshold (e.g., max/min > 100 for 2 orders)
    2. Less than max_negative_fraction of values are negative (allows log with some negatives)
    
    Args:
        values: Array of numeric values
        order_of_magnitude_threshold: Use log scale if max/min > 10^threshold (default: 2.0 = 100x span)
        max_negative_fraction: Maximum fraction of negative values allowed for log scale (default: 0.5)
        
    Returns:
        Tuple of (use_log_scale, safe_min_value):
            - use_log_scale: True if log scale should be used
            - safe_min_value: Minimum safe positive value to replace negatives (None if no log scale)
    """
    # Filter out NaN and infinite values
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) == 0:
        return False, None
    
    # Get positive values only
    positive_values = clean_values[clean_values > 0]
    
    # If no positive values, can't use log scale
    if len(positive_values) == 0:
        return False, None
    
    # Check fraction of negative/zero values
    non_positive_count = np.sum(clean_values <= 0)
    negative_fraction = non_positive_count / len(clean_values)
    
    # If too many negatives, don't use log scale
    if negative_fraction >= max_negative_fraction:
        return False, None
    
    # Check if value span exceeds order of magnitude threshold
    # Using ratio of max/min positive values
    min_positive = positive_values.min()
    max_positive = positive_values.max()
    
    # Avoid division issues
    if min_positive <= 0 or max_positive <= 0:
        return False, None
    
    ratio = max_positive / min_positive
    spans_required_orders = ratio > (10 ** order_of_magnitude_threshold)
    
    if spans_required_orders:
        # Compute safe value: 1/10 of minimum positive value
        safe_min_value = min_positive / 10.0
        return True, safe_min_value
    
    return False, None


def _is_column_plottable(
    df: 'pd.DataFrame',
    col: str,
    min_valid_entries: int = 2,
) -> bool:
    """
    Check if a column has enough valid (non-zero, non-NaN) entries to be worth plotting.
    
    Args:
        df: DataFrame containing the column
        col: Column name to check
        min_valid_entries: Minimum number of valid entries required (default: 2)
        
    Returns:
        bool: True if column is plottable
    """
    if col not in df.columns:
        return False
    
    values = df[col].values
    
    # Count non-NaN and non-zero entries
    valid_mask = np.isfinite(values) & (values != 0)
    n_valid = np.sum(valid_mask)
    
    return n_valid >= min_valid_entries


def _plot_metric_pair(
    df: 'pd.DataFrame',
    train_col: str,
    val_col: str,
    ax: 'plt.Axes',
    ylabel: str = None,
    title: str = None,
    use_log_scale: str = "auto",
    order_of_magnitude_threshold: float = 2.0,
) -> bool:
    """
    Plot a train/val metric pair on the given axes.
    
    Handles validation, plotting with seaborn, and optional log scale.
    When using log scale with some negative values, negatives are clipped
    to a safe minimum value.
    
    Args:
        df: DataFrame with columns ['kfold', 'epoch', train_col, val_col]
        train_col: Name of training metric column
        val_col: Name of validation metric column
        ax: Matplotlib axes to plot on
        ylabel: Y-axis label (default: inferred from column name)
        title: Plot title (default: "val (solid) | train (dashed)")
        use_log_scale: "auto" (default), "always", or "never"
        order_of_magnitude_threshold: For "auto" mode, use log if max/min > 10^threshold (default: 2.0)
        
    Returns:
        bool: True if plot was created, False if skipped due to validation
    """
    import seaborn as sns
    
    # Validate columns exist
    if train_col not in df.columns or val_col not in df.columns:
        return False
    
    # Check if both columns have enough valid data
    if not _is_column_plottable(df, train_col) or not _is_column_plottable(df, val_col):
        return False
    
    # Determine log scale and safe minimum value
    apply_log = False
    safe_min = None
    
    if use_log_scale == "always":
        apply_log = True
        # Get safe minimum from positive values
        all_values = np.concatenate([
            df[train_col].dropna().values,
            df[val_col].dropna().values
        ])
        positive_values = all_values[all_values > 0]
        if len(positive_values) > 0:
            safe_min = positive_values.min() / 10.0
        else:
            apply_log = False  # Can't use log if no positive values
    elif use_log_scale == "auto":
        all_values = np.concatenate([
            df[train_col].dropna().values,
            df[val_col].dropna().values
        ])
        apply_log, safe_min = _should_use_log_scale(all_values, order_of_magnitude_threshold)
    # "never" -> apply_log stays False
    
    # Prepare data for plotting (clip negatives if using log scale)
    if apply_log and safe_min is not None:
        df_plot = df.copy()
        df_plot[train_col] = df_plot[train_col].clip(lower=safe_min)
        df_plot[val_col] = df_plot[val_col].clip(lower=safe_min)
    else:
        df_plot = df
    
    # Plot validation (solid) and training (dashed)
    sns.lineplot(data=df_plot, x="epoch", y=val_col, hue="kfold", ax=ax)
    sns.lineplot(data=df_plot, x="epoch", y=train_col, hue="kfold", ax=ax, 
                 legend=False, linestyle=":")
    
    # Apply log scale if determined
    if apply_log:
        ax.set_yscale("log")
    
    # Set labels
    if ylabel is None:
        # Infer from val_col by removing "val_" prefix
        ylabel = val_col[4:] if val_col.startswith("val_") else val_col
    ax.set_ylabel(ylabel)
    
    if title is None:
        title = "val (solid) | train (dashed)"
    ax.set_title(title)
    
    return True


def _discover_metric_pairs(
    df: 'pd.DataFrame',
    exclude_prefixes: List[str] = None,
    exclude_cols: List[str] = None,
) -> dict:
    """
    Discover train/val metric pairs from a DataFrame.
    
    Finds columns with matching train_* and val_* prefixes and groups them.
    
    Args:
        df: DataFrame with metric columns
        exclude_prefixes: Prefixes to exclude (default: ["test_"])
        exclude_cols: Specific columns to exclude (default: ["epoch", "step"])
        
    Returns:
        dict: Mapping of base_name -> {"train": col_name, "val": col_name}
              Only includes pairs where both train and val columns exist.
    """
    if exclude_prefixes is None:
        exclude_prefixes = ["test_"]
    if exclude_cols is None:
        exclude_cols = ["epoch", "step", "kfold"]
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter out excluded columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    numeric_cols = [c for c in numeric_cols if not any(c.startswith(p) for p in exclude_prefixes)]
    
    metric_pairs = {}
    
    for col in numeric_cols:
        if col.startswith("train_"):
            base_name = col[6:]  # Remove "train_" prefix
            if base_name not in metric_pairs:
                metric_pairs[base_name] = {}
            metric_pairs[base_name]["train"] = col
        elif col.startswith("val_"):
            base_name = col[4:]  # Remove "val_" prefix
            if base_name not in metric_pairs:
                metric_pairs[base_name] = {}
            metric_pairs[base_name]["val"] = col
    
    # Filter to only include complete pairs
    complete_pairs = {
        name: cols for name, cols in metric_pairs.items()
        if "train" in cols and "val" in cols
    }
    
    return complete_pairs


def _filter_metric_pairs(
    metric_pairs: dict,
    patterns: List[str] = None,
) -> dict:
    """
    Filter metric pairs by patterns.
    
    Args:
        metric_pairs: dict from _discover_metric_pairs
        patterns: List of patterns to match (case-insensitive substring match)
                 If None, returns all pairs
                 
    Returns:
        dict: Filtered metric pairs
    """
    if patterns is None:
        return metric_pairs
    
    filtered = {}
    for name, cols in metric_pairs.items():
        for pattern in patterns:
            if pattern.lower() in name.lower():
                filtered[name] = cols
                break
    
    return filtered


# =============================================================================
# Training Instability Metrics
# =============================================================================

def _compute_spike_ratio(values: np.ndarray) -> float:
    """
    Compute the fraction of epochs where loss increased (spikes).
    
    A spike is defined as any epoch where loss[t+1] > loss[t].
    Higher values indicate more unstable training.
    
    Args:
        values: Array of loss values over epochs (should be sorted by epoch)
        
    Returns:
        float: Fraction of epochs with loss increase, in [0, 1]
        
    Example:
        >>> values = np.array([1.0, 0.9, 0.95, 0.8, 0.85, 0.7])  # 2 spikes out of 5 transitions
        >>> _compute_spike_ratio(values)
        0.4
    """
    if len(values) < 2:
        return 0.0
    
    # Filter out NaN values
    clean_values = values[np.isfinite(values)]
    if len(clean_values) < 2:
        return 0.0
    
    diffs = np.diff(clean_values)
    return float(np.sum(diffs > 0) / len(diffs))


def _compute_coefficient_of_variation(values: np.ndarray) -> float:
    """
    Compute the coefficient of variation (CV) of loss values.
    
    CV = std / mean, a normalized measure of variability.
    Higher values indicate more volatile training.
    
    Args:
        values: Array of loss values over epochs
        
    Returns:
        float: Coefficient of variation (unitless)
        
    Example:
        >>> values = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        >>> cv = _compute_coefficient_of_variation(values)
    """
    # Filter out NaN values
    clean_values = values[np.isfinite(values)]
    if len(clean_values) == 0:
        return 0.0
    
    mean_val = np.mean(clean_values)
    if mean_val == 0 or not np.isfinite(mean_val):
        return 0.0
    
    return float(np.std(clean_values) / mean_val)


def _compute_max_jump(values: np.ndarray) -> float:
    """
    Compute the largest single-epoch loss increase.
    
    This captures the magnitude of the worst training instability event.
    
    Args:
        values: Array of loss values over epochs (should be sorted by epoch)
        
    Returns:
        float: Maximum positive difference between consecutive epochs.
               Returns 0 if loss never increases.
               
    Example:
        >>> values = np.array([1.0, 0.9, 1.2, 0.8])  # max jump is 0.3 (0.9 -> 1.2)
        >>> _compute_max_jump(values)
        0.3
    """
    if len(values) < 2:
        return 0.0
    
    # Filter out NaN values
    clean_values = values[np.isfinite(values)]
    if len(clean_values) < 2:
        return 0.0
    
    diffs = np.diff(clean_values)
    max_increase = np.max(diffs)
    
    # Return 0 if loss never increases
    return float(max(0.0, max_increase))


def _compute_trend_instability(values: np.ndarray, window: int = 5) -> float:
    """
    Compute trend instability as residual variance from smoothed trend.
    
    Uses a moving average to estimate the trend, then computes the standard
    deviation of residuals (actual - smoothed). Higher values indicate the
    loss oscillates more around its trend.
    
    Args:
        values: Array of loss values over epochs (should be sorted by epoch)
        window: Moving average window size (default: 5 epochs)
        
    Returns:
        float: Standard deviation of residuals from smoothed trend
        
    Example:
        >>> values = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7])
        >>> instability = _compute_trend_instability(values, window=3)
    """
    # Filter out NaN values
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) < window:
        # Not enough data for moving average, return simple std
        return float(np.std(clean_values)) if len(clean_values) > 0 else 0.0
    
    # Compute moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(clean_values, kernel, mode='valid')
    
    # Align original values with smoothed (center alignment)
    start_idx = window // 2
    end_idx = start_idx + len(smoothed)
    aligned_values = clean_values[start_idx:end_idx]
    
    # Compute residuals
    residuals = aligned_values - smoothed
    
    return float(np.std(residuals))


def compute_instability_metrics(
    values: np.ndarray,
    window: int = 5,
) -> dict:
    """
    Compute all training instability metrics for a loss curve.
    
    Args:
        values: Array of loss values over epochs
        window: Moving average window for trend instability (default: 5)
        
    Returns:
        dict with keys:
            - spike_ratio: Fraction of epochs with loss increase
            - cv: Coefficient of variation (std/mean)
            - max_jump: Largest single-epoch loss increase
            - trend_instability: Residual std from smoothed trend
            
    Example:
        >>> values = np.array([1.0, 0.9, 0.95, 0.8, 0.85, 0.7, 0.65])
        >>> metrics = compute_instability_metrics(values)
        >>> print(f"Spike ratio: {metrics['spike_ratio']:.2f}")
    """
    return {
        "spike_ratio": _compute_spike_ratio(values),
        "cv": _compute_coefficient_of_variation(values),
        "max_jump": _compute_max_jump(values),
        "trend_instability": _compute_trend_instability(values, window=window),
    }


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
# Markov Equivalence Class (MEC) Metrics
# =============================================================================

def _dag_to_skeleton(dag_adj: np.ndarray) -> set:
    """
    Convert a DAG adjacency matrix to its skeleton (undirected edges).
    
    The skeleton is the set of undirected edges, represented as frozensets.
    An edge exists between i and j if either dag_adj[i,j]=1 or dag_adj[j,i]=1.
    
    Args:
        dag_adj: Binary adjacency matrix where dag_adj[i,j]=1 means j→i (edge from j to i)
        
    Returns:
        set: Set of frozenset({i, j}) representing undirected edges
        
    Example:
        >>> dag = np.array([[0, 1, 0],
        ...                 [0, 0, 1],
        ...                 [0, 0, 0]])  # 1→0, 2→1
        >>> skeleton = _dag_to_skeleton(dag)
        >>> frozenset({0, 1}) in skeleton
        True
    """
    n = dag_adj.shape[0]
    skeleton = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Edge exists if either direction is present
            if dag_adj[i, j] > 0 or dag_adj[j, i] > 0:
                skeleton.add(frozenset({i, j}))
    
    return skeleton


def _find_v_structures(dag_adj: np.ndarray) -> set:
    """
    Find all v-structures (colliders) in a DAG.
    
    A v-structure is a pattern A → B ← C where A and C are NOT adjacent.
    This function identifies all colliders by finding nodes with multiple parents
    where the parents are not adjacent to each other.
    
    Args:
        dag_adj: Binary adjacency matrix where dag_adj[i,j]=1 means j→i (edge from j to i)
        
    Returns:
        set: Set of tuples (collider_idx, parent1_idx, parent2_idx) where parent1 < parent2
             for consistent ordering
             
    Example:
        >>> # DAG: 0 → 2 ← 1 (v-structure at node 2), and 0 and 1 are NOT adjacent
        >>> dag = np.array([[0, 0, 0],
        ...                 [0, 0, 0],
        ...                 [1, 1, 0]])  # 0→2, 1→2
        >>> v_structs = _find_v_structures(dag)
        >>> (2, 0, 1) in v_structs
        True
    """
    n = dag_adj.shape[0]
    skeleton = _dag_to_skeleton(dag_adj)
    v_structures = set()
    
    for collider in range(n):
        # Find all parents of this node (nodes j where dag_adj[collider, j] = 1)
        parents = [j for j in range(n) if dag_adj[collider, j] > 0]
        
        # Check all pairs of parents
        for i, p1 in enumerate(parents):
            for p2 in parents[i + 1:]:
                # V-structure exists if parents are NOT adjacent
                if frozenset({p1, p2}) not in skeleton:
                    # Store with consistent ordering (smaller index first)
                    v_structures.add((collider, min(p1, p2), max(p1, p2)))
    
    return v_structures


def _combine_attention_to_full_dag(
    cross_adj: np.ndarray,
    self_adj: np.ndarray,
    n_source: int,
    n_intermediate: int,
) -> np.ndarray:
    """
    Combine cross-attention and self-attention matrices into a full DAG adjacency matrix.
    
    For a model with S source variables and X intermediate variables:
    - cross_adj has shape (n_X, n_S) representing S → X edges
    - self_adj has shape (n_X, n_X) representing X → X edges
    
    The combined DAG has shape (n_S + n_X, n_S + n_X):
    - Rows/cols 0:n_S correspond to source variables (no parents)
    - Rows/cols n_S:n_S+n_X correspond to intermediate variables
    
    Args:
        cross_adj: Cross-attention matrix (n_X, n_S) where entry [i,j]=1 means S_j → X_i
        self_adj: Self-attention matrix (n_X, n_X) where entry [i,j]=1 means X_j → X_i
        n_source: Number of source variables
        n_intermediate: Number of intermediate variables
        
    Returns:
        np.ndarray: Full DAG adjacency matrix of shape (n_S + n_X, n_S + n_X)
        
    Example:
        >>> cross = np.array([[1, 0], [0, 1]])  # S1→X1, S2→X2
        >>> self_att = np.array([[0, 0], [1, 0]])  # X1→X2
        >>> full_dag = _combine_attention_to_full_dag(cross, self_att, n_source=2, n_intermediate=2)
        >>> full_dag.shape
        (4, 4)
    """
    n_total = n_source + n_intermediate
    full_dag = np.zeros((n_total, n_total))
    
    # S → X edges: cross_adj[i, j] means S_j → X_i
    # In full DAG: row n_source+i, col j
    full_dag[n_source:n_source + n_intermediate, 0:n_source] = cross_adj
    
    # X → X edges: self_adj[i, j] means X_j → X_i
    # In full DAG: row n_source+i, col n_source+j
    full_dag[n_source:n_source + n_intermediate, n_source:n_source + n_intermediate] = self_adj
    
    return full_dag


def _soft_skeleton_distance(
    learned_adj: np.ndarray,
    true_skeleton: set,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute soft skeleton distance between a learned adjacency matrix and true skeleton.
    
    For continuous learned adjacency values in [0,1], this computes:
    - Recall: For true edges, how strong is max(A[i,j], A[j,i])?
    - Precision: For non-edges, how low is max(A[i,j], A[j,i])?
    - Distance: 1 - F1(recall, precision)
    
    Args:
        learned_adj: Continuous adjacency matrix with values in [0, 1]
        true_skeleton: Set of frozenset({i, j}) representing true undirected edges
        
    Returns:
        Tuple of (distance, details):
            - distance: Float in [0, 1] where 0 = perfect skeleton match
            - details: Dict with 'recall', 'precision', 'f1' scores
            
    Example:
        >>> learned = np.array([[0, 0.9, 0.1],
        ...                     [0.8, 0, 0.2],
        ...                     [0.1, 0.1, 0]])
        >>> true_skel = {frozenset({0, 1})}  # Only edge 0-1 exists
        >>> dist, details = _soft_skeleton_distance(learned, true_skel)
    """
    n = learned_adj.shape[0]
    all_pairs = {frozenset({i, j}) for i in range(n) for j in range(i + 1, n)}
    
    # For true edges: we WANT presence (either direction)
    if true_skeleton:
        true_edge_scores = []
        for edge in true_skeleton:
            i, j = tuple(edge)
            edge_strength = max(learned_adj[i, j], learned_adj[j, i])
            true_edge_scores.append(edge_strength)
        recall = float(np.mean(true_edge_scores))
    else:
        # No true edges - recall is perfect by definition
        recall = 1.0
    
    # For non-edges: we DON'T want presence
    false_pairs = all_pairs - true_skeleton
    if false_pairs:
        false_edge_scores = []
        for edge in false_pairs:
            i, j = tuple(edge)
            edge_strength = max(learned_adj[i, j], learned_adj[j, i])
            false_edge_scores.append(edge_strength)
        precision = 1.0 - float(np.mean(false_edge_scores))
    else:
        # All possible edges are true edges - precision is perfect
        precision = 1.0
    
    # F1-style combination
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    
    distance = 1.0 - f1
    
    details = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_true_edges": len(true_skeleton),
        "n_possible_edges": len(all_pairs),
    }
    
    return distance, details


def _soft_v_structure_distance(
    learned_adj: np.ndarray,
    true_v_structures: set,
    true_skeleton: set,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute soft v-structure distance between learned adjacency and true v-structures.
    
    For each true v-structure (collider at c with parents p1, p2):
    - Both p1→c and p2→c should be present (high values)
    - p1 and p2 should NOT be adjacent (low values for p1-p2 edge)
    
    The function also penalizes spurious v-structures in the learned DAG.
    
    Args:
        learned_adj: Continuous adjacency matrix with values in [0, 1]
        true_v_structures: Set of (collider, parent1, parent2) tuples
        true_skeleton: Set of frozenset({i, j}) representing true undirected edges
        
    Returns:
        Tuple of (distance, details):
            - distance: Float in [0, 1] where 0 = perfect v-structure match
            - details: Dict with 'recall', 'precision', 'f1' scores
            
    Example:
        >>> learned = np.array([[0, 0, 0],
        ...                     [0, 0, 0],
        ...                     [0.9, 0.8, 0]])  # Strong edges 0→2, 1→2
        >>> true_v = {(2, 0, 1)}  # V-structure at 2 with parents 0, 1
        >>> true_skel = {frozenset({0, 2}), frozenset({1, 2})}
        >>> dist, details = _soft_v_structure_distance(learned, true_v, true_skel)
    """
    n = learned_adj.shape[0]
    
    # Compute recall: how well are true v-structures captured?
    if true_v_structures:
        v_recall_scores = []
        for (c, p1, p2) in true_v_structures:
            # Both parent edges should be present (into collider)
            parent_strength = min(learned_adj[c, p1], learned_adj[c, p2])
            # Parents should NOT be adjacent
            no_parent_edge = 1.0 - max(learned_adj[p1, p2], learned_adj[p2, p1])
            # Combined score: both conditions must hold
            v_score = parent_strength * no_parent_edge
            v_recall_scores.append(v_score)
        recall = float(np.mean(v_recall_scores))
    else:
        # No true v-structures - recall is 1 by definition
        recall = 1.0
    
    # Compute precision: penalize spurious v-structures
    # Find potential v-structures in learned DAG (nodes with multiple strong incoming edges
    # where those sources are not adjacent)
    spurious_scores = []
    
    for collider in range(n):
        # Find potential parents (nodes with edge into collider)
        potential_parents = []
        for j in range(n):
            if j != collider and learned_adj[collider, j] > 0.3:  # threshold for "potential" edge
                potential_parents.append((j, learned_adj[collider, j]))
        
        # Check pairs of potential parents
        for i, (p1, strength1) in enumerate(potential_parents):
            for p2, strength2 in potential_parents[i + 1:]:
                # This is a potential v-structure if p1-p2 not adjacent
                p1_p2_edge = max(learned_adj[p1, p2], learned_adj[p2, p1])
                
                # If this is NOT a true v-structure, penalize it
                v_tuple = (collider, min(p1, p2), max(p1, p2))
                if v_tuple not in true_v_structures:
                    # Spurious v-structure strength
                    spurious_strength = min(strength1, strength2) * (1.0 - p1_p2_edge)
                    spurious_scores.append(spurious_strength)
    
    if spurious_scores:
        precision = 1.0 - float(np.mean(spurious_scores))
        precision = max(0.0, precision)  # Clip to [0, 1]
    else:
        precision = 1.0
    
    # F1-style combination
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    
    distance = 1.0 - f1
    
    details = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_true_v_structures": len(true_v_structures),
        "n_spurious_candidates": len(spurious_scores),
    }
    
    return distance, details


def _compute_mec_distance(
    learned_adj: np.ndarray,
    true_dag_adj: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the Markov Equivalence Class (MEC) distance between learned and true DAG.
    
    The MEC distance combines skeleton distance and v-structure distance:
        MEC_distance = (skeleton_distance + v_structure_distance) / 2
    
    A distance of 0 means the learned DAG is in the same MEC as the true DAG.
    A distance of 1 means maximum difference (completely wrong skeleton and v-structures).
    
    Args:
        learned_adj: Continuous adjacency matrix with values in [0, 1]
        true_dag_adj: Binary true DAG adjacency matrix
        
    Returns:
        Tuple of (mec_distance, details):
            - mec_distance: Float in [0, 1] where 0 = same MEC
            - details: Dict containing component distances and scores
            
    Example:
        >>> true_dag = np.array([[0, 0, 0],
        ...                      [0, 0, 0],
        ...                      [1, 1, 0]])  # V-structure: 0→2←1
        >>> learned = np.array([[0, 0, 0],
        ...                     [0, 0, 0],
        ...                     [0.9, 0.85, 0]])  # Good approximation
        >>> dist, details = _compute_mec_distance(learned, true_dag)
    """
    # Extract skeleton and v-structures from true DAG
    true_skeleton = _dag_to_skeleton(true_dag_adj)
    true_v_structures = _find_v_structures(true_dag_adj)
    
    # Compute component distances
    skel_dist, skel_details = _soft_skeleton_distance(learned_adj, true_skeleton)
    v_dist, v_details = _soft_v_structure_distance(learned_adj, true_v_structures, true_skeleton)
    
    # Combined distance (simple mean)
    mec_distance = (skel_dist + v_dist) / 2.0
    
    details = {
        "mec_distance": mec_distance,
        "skeleton_distance": skel_dist,
        "skeleton_recall": skel_details["recall"],
        "skeleton_precision": skel_details["precision"],
        "skeleton_f1": skel_details["f1"],
        "v_structure_distance": v_dist,
        "v_structure_recall": v_details["recall"],
        "v_structure_precision": v_details["precision"],
        "v_structure_f1": v_details["f1"],
        "n_true_skeleton_edges": skel_details["n_true_edges"],
        "n_true_v_structures": v_details["n_true_v_structures"],
    }
    
    return mec_distance, details


def _check_mec_membership(
    learned_adj: np.ndarray,
    true_dag_adj: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a learned DAG belongs to the Markov Equivalence Class of the true DAG.
    
    This is the binary version of MEC distance. After thresholding the learned
    adjacency matrix at `threshold`, it checks:
    1. Same skeleton: All edges (ignoring direction) must match exactly
    2. Same v-structures: All colliders must match exactly
    
    Args:
        learned_adj: Continuous adjacency matrix with values in [0, 1]
        true_dag_adj: Binary true DAG adjacency matrix
        threshold: Threshold for binarizing learned adjacency (default: 0.5)
        
    Returns:
        Tuple of (in_mec, details):
            - in_mec: Boolean indicating if learned DAG is in the same MEC
            - details: Dict with diagnostic information
            
    Example:
        >>> true_dag = np.array([[0, 0, 0],
        ...                      [0, 0, 0],
        ...                      [1, 1, 0]])  # V-structure: 0→2←1
        >>> learned = np.array([[0, 0, 0],
        ...                     [0, 0, 0],
        ...                     [0.9, 0.85, 0]])
        >>> in_mec, details = _check_mec_membership(learned, true_dag)
    """
    # Binarize learned adjacency
    learned_binary = (learned_adj >= threshold).astype(float)
    
    # Extract skeletons
    true_skeleton = _dag_to_skeleton(true_dag_adj)
    learned_skeleton = _dag_to_skeleton(learned_binary)
    
    # Check skeleton equality
    skeleton_match = (true_skeleton == learned_skeleton)
    
    # Extract v-structures
    true_v_structures = _find_v_structures(true_dag_adj)
    learned_v_structures = _find_v_structures(learned_binary)
    
    # Check v-structure equality
    v_structure_match = (true_v_structures == learned_v_structures)
    
    # In MEC if both conditions hold
    in_mec = skeleton_match and v_structure_match
    
    # Compute detailed differences
    missing_edges = true_skeleton - learned_skeleton
    extra_edges = learned_skeleton - true_skeleton
    missing_v_structures = true_v_structures - learned_v_structures
    extra_v_structures = learned_v_structures - true_v_structures
    
    details = {
        "in_mec": in_mec,
        "threshold": threshold,
        "skeleton_match": skeleton_match,
        "v_structure_match": v_structure_match,
        "n_true_skeleton_edges": len(true_skeleton),
        "n_learned_skeleton_edges": len(learned_skeleton),
        "n_missing_edges": len(missing_edges),
        "n_extra_edges": len(extra_edges),
        "missing_edges": [tuple(e) for e in missing_edges],
        "extra_edges": [tuple(e) for e in extra_edges],
        "n_true_v_structures": len(true_v_structures),
        "n_learned_v_structures": len(learned_v_structures),
        "n_missing_v_structures": len(missing_v_structures),
        "n_extra_v_structures": len(extra_v_structures),
        "missing_v_structures": list(missing_v_structures),
        "extra_v_structures": list(extra_v_structures),
    }
    
    return in_mec, details


def _load_full_true_dag(
    datadir_path: str,
    dataset: str,
) -> Optional[np.ndarray]:
    """
    Load the full true DAG adjacency matrix combining all attention mask blocks.
    
    For datasets with source variables (S → X structure), this combines:
    - dec1_cross_att_mask.csv (S → X edges)
    - dec1_self_att_mask.csv (X → X edges)
    
    Into a full (n_S + n_X) × (n_S + n_X) adjacency matrix.
    
    Args:
        datadir_path: Path to the data directory
        dataset: Dataset name
        
    Returns:
        np.ndarray: Full DAG adjacency matrix, or None if files not found
    """
    # Load cross-attention mask (S → X)
    cross_mask = _load_true_dag_mask(datadir_path, dataset, "dec1_cross")
    # Load self-attention mask (X → X)
    self_mask = _load_true_dag_mask(datadir_path, dataset, "dec1_self")
    
    if cross_mask is None or self_mask is None:
        # Try to load full DAG directly if available
        filepath = join(datadir_path, dataset, "dag_adj_mask.csv")
        if exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                return df.values.astype(float)
            except Exception as e:
                print(f"Warning: Failed to load full DAG mask: {e}")
        return None
    
    n_X, n_S = cross_mask.shape
    
    # Combine into full DAG
    full_dag = _combine_attention_to_full_dag(
        cross_adj=cross_mask,
        self_adj=self_mask,
        n_source=n_S,
        n_intermediate=n_X,
    )
    
    return full_dag
