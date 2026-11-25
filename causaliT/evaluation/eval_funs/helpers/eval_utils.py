"""
Shared utility functions for CausaliT evaluation.

This module provides helper functions used across all evaluation modules:
- Evaluation directory setup and file management
- Dataset metadata and ground-truth DAG masks (loading / assembling)

Metrics are NOT here: they live in ``causaliT.metrics``.

Note: this module no longer plots anything, so it no longer imports
matplotlib or mutates global rcParams at import time.
"""

import re
import json
from os.path import join, exists
from os import makedirs
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from causaliT.paths import ROOT_DIR

# Project root, as a string (callers pass it to os.path.join).
# Taken from causaliT.paths so it stays correct regardless of how deeply
# nested this module is, and honours the PROT_ROOT override.
root_path = str(ROOT_DIR)


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
