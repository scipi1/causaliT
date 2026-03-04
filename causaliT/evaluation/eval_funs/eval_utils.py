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
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Setup root path for local imports
# Go up FOUR levels: eval_funs -> evaluation -> causaliT -> project root
root_path = dirname(dirname(dirname(dirname(abspath(__file__)))))


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
    log_scale_threshold: float = 10.0,
) -> bool:
    """
    Determine if log scale should be used based on data characteristics.
    
    Log scale is recommended when:
    1. All values are strictly positive (required for log scale)
    2. The range (max - min) exceeds the threshold
    
    Args:
        values: Array of numeric values
        log_scale_threshold: Use log scale if max - min > threshold (default: 10.0)
        
    Returns:
        bool: True if log scale should be used
    """
    # Filter out NaN and infinite values
    clean_values = values[np.isfinite(values)]
    
    if len(clean_values) == 0:
        return False
    
    # Check if all values are strictly positive
    if not np.all(clean_values > 0):
        return False
    
    # Check if range exceeds threshold
    value_range = clean_values.max() - clean_values.min()
    return value_range > log_scale_threshold


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
    log_scale_threshold: float = 10.0,
) -> bool:
    """
    Plot a train/val metric pair on the given axes.
    
    Handles validation, plotting with seaborn, and optional log scale.
    
    Args:
        df: DataFrame with columns ['kfold', 'epoch', train_col, val_col]
        train_col: Name of training metric column
        val_col: Name of validation metric column
        ax: Matplotlib axes to plot on
        ylabel: Y-axis label (default: inferred from column name)
        title: Plot title (default: "val (solid) | train (dashed)")
        use_log_scale: "auto" (default), "always", or "never"
        log_scale_threshold: For "auto" mode, use log if max-min > threshold
        
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
    
    # Plot validation (solid) and training (dashed)
    sns.lineplot(data=df, x="epoch", y=val_col, hue="kfold", ax=ax)
    sns.lineplot(data=df, x="epoch", y=train_col, hue="kfold", ax=ax, 
                 legend=False, linestyle=":")
    
    # Determine log scale
    if use_log_scale == "always":
        ax.set_yscale("log")
    elif use_log_scale == "auto":
        all_values = np.concatenate([
            df[train_col].dropna().values,
            df[val_col].dropna().values
        ])
        if _should_use_log_scale(all_values, log_scale_threshold):
            ax.set_yscale("log")
    # "never" -> no log scale
    
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
