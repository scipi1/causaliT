"""
Shared utility functions for CausaliT evaluation.

This module provides helper functions used across all evaluation modules:
- Evaluation directory setup and file management
- Dataset metadata and ground-truth DAG masks
- DAG recovery metrics (soft Hamming, SHD, zeroness, DAG confidence)
- Markov equivalence class (MEC) metrics

Note: this module no longer plots anything, so it no longer imports
matplotlib or mutates global rcParams at import time.
"""

import re
import json
from os.path import dirname, abspath, join, exists
from os import makedirs
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Setup root path for local imports
# Go up FOUR levels: eval_funs -> evaluation -> causaliT -> project root
root_path = dirname(dirname(dirname(dirname(abspath(__file__)))))


# =============================================================================
# Evaluation Directory Helpers
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
    
    # Classification metrics from the binary confusion matrix
    tp = int(((true_bin == 1) & (pred == 1)).sum())  # True Positives
    tn = int(((true_bin == 0) & (pred == 0)).sum())  # True Negatives
    
    # Rates as percentages
    tpr = 100.0 * tp / (tp + missing) if (tp + missing) > 0 else float('nan')       # Recall = TP / (TP + FN)
    fdr = 100.0 * extra / (extra + tp) if (extra + tp) > 0 else float('nan')        # FP / (FP + TP)
    precision = 100.0 * tp / (tp + extra) if (tp + extra) > 0 else float('nan')
    
    return {
        'shd': shd,
        'missing': missing,
        'extra': extra,
        'reversed': reversed_edges,
        'n_true_edges': n_true_edges,
        'n_pred_edges': n_pred_edges,
        'threshold': threshold,
        # Classification metrics (percentages)
        'tp': tp,
        'tn': tn,
        'tpr': tpr,              # True Positive Rate (Recall) [%]
        'fdr': fdr,              # False Discovery Rate [%]
        'precision': precision,  # Precision [%]
    }


def _compute_zeroness_metrics(learned: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute zero-ness metrics: how close to 0 are the non-edges?
    
    Measures the quality of the learned DAG in terms of edge/non-edge separation.
    For Toeplitz self-attention (initialized at 0.5), non-edges must be actively
    driven towards 0 - this metric captures that progress.
    
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


def _compute_mec_threshold(
    learned_adj: np.ndarray,
    true_dag_adj: np.ndarray,
) -> Tuple[Optional[float], bool]:
    """
    Compute the MEC threshold: the maximum binarisation threshold θ at which
    the learned DAG is in the same Markov Equivalence Class as the true DAG.

    Analogous to a p-value: a higher value means the scores are more
    discriminative - even aggressive pruning (high threshold) still recovers
    the correct skeleton and v-structures.

    Algorithm:
        The binarised graph ``(learned_adj >= θ)`` only changes at the unique
        score values present in ``learned_adj``.  We evaluate
        ``_check_mec_membership`` at each such value (plus 0.0) and return
        the **maximum** θ for which membership holds.

    Args:
        learned_adj:   Continuous adjacency matrix with values in [0, 1].
        true_dag_adj:  Binary true DAG adjacency matrix.

    Returns:
        Tuple of (mec_threshold, exists):
            mec_threshold : float or None
                Maximum threshold achieving MEC membership.
                ``None`` (→ stored as NaN downstream) when no threshold works.
            exists : bool
                ``True`` when at least one threshold achieves MEC membership.

    Example:
        >>> true_dag = np.array([[0, 0, 0],
        ...                      [0, 0, 0],
        ...                      [1, 1, 0]])   # V-structure 0→2←1
        >>> learned = np.array([[0,    0,    0],
        ...                     [0,    0,    0],
        ...                     [0.85, 0.75, 0]])
        >>> mec_thresh, exists = _compute_mec_threshold(learned, true_dag)
        >>> exists
        True
        >>> 0.0 < mec_thresh <= 0.75  # Must be ≤ min(true-edge scores)
        True
    """
    # Candidate thresholds: all unique score values (these are the only points
    # at which the binarised graph changes), plus:
    #   0.0          → "include all edges" baseline
    #   max + epsilon → "include no edges" endpoint (needed for empty true DAG)
    max_val = float(np.max(learned_adj)) if learned_adj.size else 0.0
    epsilon = max(max_val * 1e-9, 1e-12)
    candidates = np.unique(np.concatenate(
        [[0.0], learned_adj.ravel(), [max_val + epsilon]]
    ))

    # Evaluate from highest threshold downward so we can short-circuit.
    # We want the *maximum* θ where in_mec is True.
    best_threshold: Optional[float] = None
    for theta in candidates[::-1]:   # descending order
        in_mec, _ = _check_mec_membership(learned_adj, true_dag_adj, threshold=float(theta))
        if in_mec:
            best_threshold = float(theta)
            break   # First hit in descending order is the maximum

    if best_threshold is None:
        return None, False
    return best_threshold, True


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
