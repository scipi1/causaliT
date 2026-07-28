"""
Config and checkpoint discovery for CausaliT evaluation.

This module is deliberately tiny: it answers three questions about an
experiment folder on disk, without importing any model code.

- ``find_config_file``            : locate the single ``config*.yaml``
- ``find_best_or_last_checkpoint``: pick a checkpoint by policy
- ``get_architecture_type``       : map ``model.model_object`` to a tag

The attention-data cache, embedding-evolution loaders, training-metric
loaders and ``predict_from_experiment`` that used to live here were removed
together with the evaluations that consumed them; prediction now goes through
``causaliT.evaluation.predict``.
"""

import re
from os import listdir
from os.path import join, exists, isdir
from typing import Dict, Optional

import numpy as np


# =============================================================================
# Config and Checkpoint Discovery
# =============================================================================

def find_config_file(folder_path: str) -> str:
    """
    Find a configuration file matching the pattern config_*.yaml in the given folder.

    Args:
        folder_path: Path to the folder to search in

    Returns:
        str: Full path to the config file

    Raises:
        FileNotFoundError: If no config file is found
        ValueError: If more than one config file is found
    """
    pattern = re.compile(r'^config(_.*)?\.yaml$')
    matching_files = []

    for filename in listdir(folder_path):
        if pattern.match(filename):
            matching_files.append(join(folder_path, filename))

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No config*.yaml found in {folder_path}")

    if len(matching_files) > 1:
        raise ValueError(f"More than one config file found in {folder_path}: {matching_files}")

    return matching_files[0]


def find_best_or_last_checkpoint(
    checkpoints_dir: str,
    checkpoint_type: str = "last"
) -> str:
    """
    Find a checkpoint from the checkpoints directory.

    Args:
        checkpoints_dir: Path to the checkpoints directory
        checkpoint_type: Which checkpoint to return:
            - "last": Last epoch checkpoint (default) - better for causal analysis
            - "best": best_checkpoint.ckpt if available, else last

    Returns:
        str: Full path to the selected checkpoint

    Raises:
        FileNotFoundError: If no checkpoints are found

    Note:
        For causal discovery, "last" is preferred because:
        - "best" selects based on prediction loss, not causal correctness
        - Causal regularizers (HSIC, sparsity) may need more epochs to converge
        - "last" represents the model's final DAG hypothesis
    """
    if not exists(checkpoints_dir) or not isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    checkpoint_files = [f for f in listdir(checkpoints_dir) if f.endswith('.ckpt')]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")

    # If checkpoint_type is "best_causal", try best_causal_checkpoint.ckpt first,
    # then fall back to best_reconstruction_checkpoint.ckpt, then to last epoch.
    if checkpoint_type == "best_causal":
        if 'best_causal_checkpoint.ckpt' in checkpoint_files:
            return join(checkpoints_dir, 'best_causal_checkpoint.ckpt')
        # Fallback: treat as "best_reconstruction"
        checkpoint_type = "best_reconstruction"

    # If checkpoint_type is "best_reconstruction", try new name then legacy name
    if checkpoint_type == "best_reconstruction":
        if 'best_reconstruction_checkpoint.ckpt' in checkpoint_files:
            return join(checkpoints_dir, 'best_reconstruction_checkpoint.ckpt')
        if 'best_checkpoint.ckpt' in checkpoint_files:
            return join(checkpoints_dir, 'best_checkpoint.ckpt')
        # Fall through to last-epoch logic

    # Legacy "best" type: try both names for backward compatibility
    if checkpoint_type == "best":
        if 'best_reconstruction_checkpoint.ckpt' in checkpoint_files:
            return join(checkpoints_dir, 'best_reconstruction_checkpoint.ckpt')
        if 'best_checkpoint.ckpt' in checkpoint_files:
            return join(checkpoints_dir, 'best_checkpoint.ckpt')

    # Otherwise (or for "last"), find the checkpoint with the highest epoch number
    epoch_pattern = re.compile(r'epoch=(\d+)')
    max_epoch = -1
    last_ckpt = None

    for ckpt in checkpoint_files:
        # Skip best_checkpoint.ckpt when looking for last
        if ckpt == 'best_checkpoint.ckpt':
            continue
        match = epoch_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                last_ckpt = ckpt

    if last_ckpt is None:
        # Fall back to first checkpoint if no epoch pattern found
        last_ckpt = checkpoint_files[0]

    return join(checkpoints_dir, last_ckpt)


# =============================================================================
# Architecture Detection
# =============================================================================

def get_architecture_type(config: dict) -> str:
    """
    Determine the architecture type from config.

    Args:
        config: OmegaConf configuration

    Returns:
        str: e.g. "TransformerForecaster", "SingleCausalForecaster",
        "NoiseAwareCausalForecaster", "AttentionSelectorForecaster"
    """
    model_obj = config["model"]["model_object"]

    if model_obj == "proT":
        return "TransformerForecaster"
    elif model_obj == "StageCausaliT":
        return "StageCausalForecaster"
    elif model_obj == "SingleCausalLayer":
        return "SingleCausalForecaster"
    elif model_obj == "SingleCausalLayerRes":
        # Treated identically to SingleCausalForecaster except for the
        # checkpoint loader (different forecaster class wraps the dual-
        # residual model). Returning a distinct tag keeps the loader
        # branches readable.
        return "SingleCausalResForecaster"
    elif model_obj == "NoiseAwareSingleCausalLayer":
        return "NoiseAwareCausalForecaster"
    elif model_obj == "NoiseAwareSingleCausalLayerRes":
        return "NoiseAwareCausalResForecaster"
    elif model_obj == "AttentionSelectorLayer":
        return "AttentionSelectorForecaster"
    else:
        raise ValueError(f"Unknown model type: {model_obj}")


# =============================================================================
# Model Introspection
# =============================================================================

def extract_phi_from_model(model, architecture_type: str) -> Dict[str, Optional[np.ndarray]]:
    """
    [DEPRECATED] Phi-learning has been removed from CausaliT.

    The learned DAG structure is now read directly from attention weights
    (mean attention over the test set) rather than from a separate learnable
    phi parameter.  This stub is kept so that old call-sites do not crash;
    it simply returns an empty dict, leaving phi_tensors unpopulated.

    All DAG metrics now use the attention-weight path in
    ``eval_dag_query.query_dag_blocks``.
    """
    return {}
