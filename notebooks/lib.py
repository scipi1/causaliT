"""
Notebook utilities for attention score analysis and model predictions.

This module provides functions to:
- Load attention weights and phi tensors from trained models
- Run predictions from experiments (all k-folds, best/last checkpoint)
- Support interventions via input_conditioning_fn

Automatic architecture detection (TransformerForecaster vs StageCausalForecaster).
"""

import os
import re
from os.path import join, exists, isdir
from os import listdir
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Local imports
from causaliT.evaluation.predict import predict_test_from_ckpt
from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster


@dataclass
class AttentionData:
    """
    Container for attention weights and phi tensors from a trained model.
    
    Attributes:
        attention_weights: Dict mapping attention type to list of arrays per k-fold
            - "encoder": encoder self-attention (TransformerForecaster only)
            - "decoder": decoder self-attention
            - "cross": cross-attention
            For StageCausalForecaster, additional keys:
            - "decoder1_self", "decoder1_cross"
            - "decoder2_self", "decoder2_cross"
        phi_tensors: Dict mapping component to list of phi arrays per k-fold
            - "encoder": encoder phi (TransformerForecaster only)
            - "decoder": decoder phi (TransformerForecaster)
            - "decoder1": decoder1 phi (StageCausalForecaster)
            - "decoder2": decoder2 phi (StageCausalForecaster)
        predictions: List of PredictionResult objects per k-fold
        config: The loaded OmegaConf config
        architecture_type: "TransformerForecaster" or "StageCausalForecaster"
        checkpoint_paths: List of checkpoint paths used per k-fold
    """
    attention_weights: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    phi_tensors: Dict[str, List[Optional[np.ndarray]]] = field(default_factory=dict)
    predictions: List[Any] = field(default_factory=list)
    config: Any = None
    architecture_type: str = ""
    checkpoint_paths: List[str] = field(default_factory=list)


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
    pattern = re.compile(r'^config_.*\.yaml$')
    matching_files = []
    
    for filename in listdir(folder_path):
        if pattern.match(filename):
            matching_files.append(join(folder_path, filename))
    
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No config_*.yaml found in {folder_path}")
    
    if len(matching_files) > 1:
        raise ValueError(f"More than one config file found in {folder_path}: {matching_files}")
    
    return matching_files[0]


def find_best_or_last_checkpoint(checkpoints_dir: str) -> str:
    """
    Find the best checkpoint if available, otherwise return the last epoch checkpoint.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        str: Full path to the selected checkpoint
        
    Raises:
        FileNotFoundError: If no checkpoints are found
    """
    if not exists(checkpoints_dir) or not isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    checkpoint_files = [f for f in listdir(checkpoints_dir) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    # Check for best_checkpoint.ckpt first
    if 'best_checkpoint.ckpt' in checkpoint_files:
        return join(checkpoints_dir, 'best_checkpoint.ckpt')
    
    # Otherwise, find the checkpoint with the highest epoch number
    epoch_pattern = re.compile(r'epoch=(\d+)')
    max_epoch = -1
    best_ckpt = None
    
    for ckpt in checkpoint_files:
        match = epoch_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_ckpt = ckpt
    
    if best_ckpt is None:
        # Fall back to first checkpoint if no epoch pattern found
        best_ckpt = checkpoint_files[0]
    
    return join(checkpoints_dir, best_ckpt)


def get_architecture_type(config: dict) -> str:
    """
    Determine the architecture type from config.
    
    Args:
        config: OmegaConf configuration
        
    Returns:
        str: "TransformerForecaster" or "StageCausalForecaster"
    """
    model_obj = config["model"]["model_object"]
    
    if model_obj == "proT":
        return "TransformerForecaster"
    elif model_obj == "StageCausaliT":
        return "StageCausalForecaster"
    else:
        raise ValueError(f"Unknown model type: {model_obj}")


def extract_phi_from_model(model, architecture_type: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract learned DAG probabilities (sigmoid(phi)) from a loaded model.
    
    This function extracts the posterior DAG structure learned by LieAttention and
    CausalCrossAttention modules. It uses `get_dag_probabilities()` which returns
    sigmoid(phi), the actual edge probabilities, rather than raw phi logits.
    
    Args:
        model: Loaded model (TransformerForecaster or StageCausalForecaster)
        architecture_type: "TransformerForecaster" or "StageCausalForecaster"
        
    Returns:
        Dict mapping component name to DAG probability array (or None if not available)
        Keys:
        - TransformerForecaster: "encoder", "decoder", "cross"
        - StageCausalForecaster: "decoder1", "decoder1_cross", "decoder2", "decoder2_cross"
    """
    phi_dict = {}
    
    def _get_dag_probs(inner_attention):
        """Helper to safely extract DAG probabilities from an attention module."""
        if hasattr(inner_attention, 'get_dag_probabilities'):
            dag_probs = inner_attention.get_dag_probabilities()
            if dag_probs is not None:
                return dag_probs.detach().cpu().numpy()
        return None
    
    if architecture_type == "TransformerForecaster":
        # Encoder self-attention DAG
        enc_inner = model.model.encoder.layers[0].global_attention.inner_attention
        phi_dict["encoder"] = _get_dag_probs(enc_inner)
        
        # Decoder self-attention DAG
        dec_self_inner = model.model.decoder.layers[0].global_self_attention.inner_attention
        phi_dict["decoder"] = _get_dag_probs(dec_self_inner)
        
        # Decoder cross-attention DAG (for CausalCrossAttention)
        dec_cross_inner = model.model.decoder.layers[0].global_cross_attention.inner_attention
        phi_dict["cross"] = _get_dag_probs(dec_cross_inner)
        
    elif architecture_type == "StageCausalForecaster":
        # Decoder1 self-attention DAG (X -> X structure)
        dec1_self_inner = model.model.decoder1.layers[0].global_self_attention.inner_attention
        phi_dict["decoder1"] = _get_dag_probs(dec1_self_inner)
        
        # Decoder1 cross-attention DAG (S -> X structure)
        dec1_cross_inner = model.model.decoder1.layers[0].global_cross_attention.inner_attention
        phi_dict["decoder1_cross"] = _get_dag_probs(dec1_cross_inner)
        
        # Decoder2 self-attention DAG (Y -> Y structure)
        dec2_self_inner = model.model.decoder2.layers[0].global_self_attention.inner_attention
        phi_dict["decoder2"] = _get_dag_probs(dec2_self_inner)
        
        # Decoder2 cross-attention DAG (X -> Y structure)
        dec2_cross_inner = model.model.decoder2.layers[0].global_cross_attention.inner_attention
        phi_dict["decoder2_cross"] = _get_dag_probs(dec2_cross_inner)
        
        # Compatibility keys
        phi_dict["encoder"] = None  # No encoder in StageCausal
        phi_dict["decoder"] = None  # For compatibility
    
    return phi_dict


def load_attention_data(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    checkpoint_type: str = "best",
    input_conditioning_fn: Callable = None,
) -> AttentionData:
    """
    Load attention weights and phi tensors from a trained experiment.
    
    This function automatically:
    - Finds the config file in the experiment folder
    - Detects the architecture type (TransformerForecaster vs StageCausalForecaster)
    - Discovers all k-fold directories
    - Loads the best or last checkpoint from each fold
    - Extracts attention weights and phi tensors
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        datadir_path: Path to data directory. If None, uses "../data/" relative to notebooks
        dataset_label: One of ["train", "test", "all"]
        checkpoint_type: "best" for best_checkpoint.ckpt, "last" for last epoch
        input_conditioning_fn: Optional function to condition inputs before forward pass
        
    Returns:
        AttentionData: Container with attention weights, phi tensors, predictions, and metadata
        
    Example:
        >>> from lib import load_attention_data
        >>> 
        >>> # Load attention data from experiment
        >>> data = load_attention_data("../experiments/euler_scm6/stage_Lie_scm6_54094964")
        >>> 
        >>> # Access attention weights across all folds
        >>> enc_self_att_list = data.attention_weights["encoder"]
        >>> dec_self_att_list = data.attention_weights["decoder"]
        >>> cross_att_list = data.attention_weights["cross"]
        >>> 
        >>> # Access phi tensors (for LieAttention)
        >>> enc_phi_list = data.phi_tensors["encoder"]
        >>> dec_phi_list = data.phi_tensors["decoder"]
        >>> 
        >>> # Print architecture type
        >>> print(f"Architecture: {data.architecture_type}")
    """
    # Default data directory
    if datadir_path is None:
        datadir_path = join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    
    # Find config file
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    
    # Determine architecture type
    architecture_type = get_architecture_type(config)
    print(f"Detected architecture: {architecture_type}")
    
    # Find all k-fold directories
    kfold_dirs = sorted([
        d for d in listdir(experiment_path) 
        if isdir(join(experiment_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment_path}")
    
    print(f"Found {len(kfold_dirs)} k-fold directories: {kfold_dirs}")
    
    # Initialize result containers
    result = AttentionData(
        config=config,
        architecture_type=architecture_type,
    )
    
    # Initialize attention weight lists based on architecture
    if architecture_type == "TransformerForecaster":
        result.attention_weights = {
            "encoder": [],
            "decoder": [],
            "cross": [],
        }
        result.phi_tensors = {
            "encoder": [],
            "decoder": [],
            "cross": [],  # Cross-attention DAG (for CausalCrossAttention)
        }
    else:  # StageCausalForecaster
        result.attention_weights = {
            "encoder": [],  # Empty for compatibility
            "decoder": [],  # Mapped to decoder2_self for compatibility
            "cross": [],    # Mapped to decoder2_cross for compatibility
            "decoder1_self": [],
            "decoder1_cross": [],
            "decoder2_self": [],
            "decoder2_cross": [],
        }
        result.phi_tensors = {
            "encoder": [],
            "decoder": [],
            "decoder1": [],
            "decoder1_cross": [],  # Cross-attention DAG (S -> X)
            "decoder2": [],
            "decoder2_cross": [],  # Cross-attention DAG (X -> Y)
        }
    
    # Process each k-fold
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            # Find checkpoint
            if checkpoint_type == "best":
                checkpoint_path = find_best_or_last_checkpoint(checkpoints_dir)
            else:
                checkpoint_path = find_best_or_last_checkpoint(checkpoints_dir)  # Same logic, finds last if no best
            
            print(f"Processing {kfold_dir}: {os.path.basename(checkpoint_path)}")
            result.checkpoint_paths.append(checkpoint_path)
            
            # Run predictions to get attention weights
            predictions = predict_test_from_ckpt(
                config=config,
                datadir_path=datadir_path,
                checkpoint_path=checkpoint_path,
                dataset_label=dataset_label,
                cluster=False,
                input_conditioning_fn=input_conditioning_fn
            )
            result.predictions.append(predictions)
            
            # Extract attention weights from predictions
            att_weights = predictions.attention_weights
            
            # Debug: print available keys
            if att_weights is not None:
                print(f"  Attention weights keys: {att_weights.keys()}")
            else:
                print(f"  Warning: attention_weights is None")
            
            if att_weights is None:
                # No attention weights returned, append None for all keys
                for key in result.attention_weights.keys():
                    result.attention_weights[key].append(None)
            elif architecture_type == "TransformerForecaster":
                result.attention_weights["encoder"].append(att_weights.get("encoder"))
                result.attention_weights["decoder"].append(att_weights.get("decoder"))
                result.attention_weights["cross"].append(att_weights.get("cross"))
            else:  # StageCausalForecaster
                # StageCausalPredictor returns keys: dec1_cross, dec1_self, dec2_cross, dec2_self
                result.attention_weights["decoder1_self"].append(att_weights.get("dec1_self"))
                result.attention_weights["decoder1_cross"].append(att_weights.get("dec1_cross"))
                result.attention_weights["decoder2_self"].append(att_weights.get("dec2_self"))
                result.attention_weights["decoder2_cross"].append(att_weights.get("dec2_cross"))
                # For compatibility with notebook code expecting encoder/decoder/cross
                result.attention_weights["encoder"].append(None)
                result.attention_weights["decoder"].append(att_weights.get("dec2_self"))
                result.attention_weights["cross"].append(att_weights.get("dec2_cross"))
            
            # Load model and extract phi tensors
            if architecture_type == "TransformerForecaster":
                model = TransformerForecaster.load_from_checkpoint(checkpoint_path)
            else:
                model = StageCausalForecaster.load_from_checkpoint(checkpoint_path)
            
            phi_dict = extract_phi_from_model(model, architecture_type)
            
            for key, value in phi_dict.items():
                if key in result.phi_tensors:
                    result.phi_tensors[key].append(value)
            
            print(f"  ✓ Successfully processed {kfold_dir}")
            
        except Exception as e:
            print(f"  ✗ Error processing {kfold_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nLoaded data from {len(result.predictions)} folds")
    return result


def plot_attention_scores(
    data: AttentionData,
    figsize: tuple = None,
    cmap: str = 'viridis',
    annotation_fontsize: int = 8,
    title_fontsize: int = 12,
    save_path: str = None,
    dpi: int = 100,
    scale_mode: str = "global",
) -> 'plt.Figure':
    """
    Plot attention scores from all K-folds and attention blocks using GridSpec.
    
    Creates a grid layout where:
    - Columns represent different K-folds
    - Rows represent different attention blocks (with optional phi rows below)
    - Each row has its own colorbar on the right
    - Color scale can be global (default) or per-row
    
    Args:
        data: AttentionData object from load_attention_data()
        figsize: Figure size as (width, height). If None, auto-calculated
        cmap: Colormap to use for heatmaps
        annotation_fontsize: Font size for mean±std annotations
        title_fontsize: Font size for subplot titles
        save_path: Optional path to save the figure
        dpi: DPI for the figure
        scale_mode: Color scale mode - "global" (default) for same scale across all plots,
                    or "row" for per-row scaling
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> data = load_attention_data("../experiments/my_experiment")
        >>> fig = plot_attention_scores(data)  # global scale (default)
        >>> fig = plot_attention_scores(data, scale_mode="row")  # row-wise scale
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Determine which attention blocks to plot (non-empty ones)
    attention_blocks = []
    phi_mapping = {}  # Maps attention block name to corresponding phi key
    
    if data.architecture_type == "TransformerForecaster":
        # Check each block for non-None data
        if any(x is not None for x in data.attention_weights.get("encoder", [])):
            attention_blocks.append("encoder")
            phi_mapping["encoder"] = "encoder"
        if any(x is not None for x in data.attention_weights.get("decoder", [])):
            attention_blocks.append("decoder")
            phi_mapping["decoder"] = "decoder"
        if any(x is not None for x in data.attention_weights.get("cross", [])):
            attention_blocks.append("cross")
            phi_mapping["cross"] = "cross"  # Cross-attention DAG (for CausalCrossAttention)
    else:  # StageCausalForecaster
        if any(x is not None for x in data.attention_weights.get("decoder1_self", [])):
            attention_blocks.append("decoder1_self")
            phi_mapping["decoder1_self"] = "decoder1"
        if any(x is not None for x in data.attention_weights.get("decoder1_cross", [])):
            attention_blocks.append("decoder1_cross")
            phi_mapping["decoder1_cross"] = "decoder1_cross"  # Cross-attention DAG (S -> X)
        if any(x is not None for x in data.attention_weights.get("decoder2_self", [])):
            attention_blocks.append("decoder2_self")
            phi_mapping["decoder2_self"] = "decoder2"
        if any(x is not None for x in data.attention_weights.get("decoder2_cross", [])):
            attention_blocks.append("decoder2_cross")
            phi_mapping["decoder2_cross"] = "decoder2_cross"  # Cross-attention DAG (X -> Y)
    
    if not attention_blocks:
        raise ValueError("No attention blocks with data found")
    
    # Determine number of K-folds
    n_folds = len(data.predictions)
    if n_folds == 0:
        raise ValueError("No predictions found in data")
    
    # Calculate number of rows: each attention block + optional phi row below self-attention blocks
    row_info = []  # List of (block_name, is_phi) tuples
    for block in attention_blocks:
        row_info.append((block, False))  # Attention row
        # Check if phi is available for this block
        phi_key = phi_mapping.get(block)
        if phi_key and any(x is not None for x in data.phi_tensors.get(phi_key, [])):
            row_info.append((block, True))  # Phi row
    
    n_rows = len(row_info)
    n_cols = n_folds + 1  # +1 for colorbar column
    
    # Calculate global min/max for consistent color scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    for block in attention_blocks:
        for att_tensor in data.attention_weights.get(block, []):
            if att_tensor is not None:
                if len(att_tensor.shape) < 3:
                    att_tensor = np.expand_dims(att_tensor, axis=0)
                mean = att_tensor.mean(axis=0)
                global_min = min(global_min, mean.min())
                global_max = max(global_max, mean.max())
    
    # Include phi tensors in global scale
    for phi_key in phi_mapping.values():
        for phi_tensor in data.phi_tensors.get(phi_key, []):
            if phi_tensor is not None:
                global_min = min(global_min, phi_tensor.min())
                global_max = max(global_max, phi_tensor.max())
    
    print(f"Global color scale: min={global_min:.4f}, max={global_max:.4f}")
    
    # Pre-compute per-row min/max for row-wise scaling
    row_scales = {}  # Maps row_idx to (vmin, vmax)
    if scale_mode == "row":
        for row_idx, (block_name, is_phi) in enumerate(row_info):
            row_min = float('inf')
            row_max = float('-inf')
            
            if is_phi:
                phi_key = phi_mapping.get(block_name)
                phi_list = data.phi_tensors.get(phi_key, [])
                for phi_tensor in phi_list:
                    if phi_tensor is not None:
                        row_min = min(row_min, phi_tensor.min())
                        row_max = max(row_max, phi_tensor.max())
            else:
                att_list = data.attention_weights.get(block_name, [])
                for att_tensor in att_list:
                    if att_tensor is not None:
                        if len(att_tensor.shape) < 3:
                            att_tensor = np.expand_dims(att_tensor, axis=0)
                        mean = att_tensor.mean(axis=0)
                        row_min = min(row_min, mean.min())
                        row_max = max(row_max, mean.max())
            
            row_scales[row_idx] = (row_min, row_max)
            print(f"Row {row_idx} ({block_name}, phi={is_phi}) scale: min={row_min:.4f}, max={row_max:.4f}")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        cell_width = 3.5
        cell_height = 3.0
        figsize = (cell_width * n_cols, cell_height * n_rows)
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Width ratios: equal for folds, narrow for colorbar
    width_ratios = [1] * n_folds + [0.05]
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios,
                           wspace=0.3, hspace=0.4)
    
    # Plot each row
    for row_idx, (block_name, is_phi) in enumerate(row_info):
        row_images = []  # Store images for colorbar
        
        # Determine vmin/vmax for this row
        if scale_mode == "row":
            vmin, vmax = row_scales[row_idx]
        else:
            vmin, vmax = global_min, global_max
        
        for col_idx in range(n_folds):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if is_phi:
                # Plot phi tensor
                phi_key = phi_mapping.get(block_name)
                phi_list = data.phi_tensors.get(phi_key, [])
                phi_tensor = phi_list[col_idx] if col_idx < len(phi_list) else None
                
                if phi_tensor is not None:
                    im = ax.imshow(phi_tensor, vmin=vmin, vmax=vmax, cmap=cmap)
                    row_images.append(im)
                    
                    # Set title for first column only
                    if col_idx == 0:
                        ax.set_ylabel(f"φ ({block_name})", fontsize=title_fontsize)
                    
                    # Set column title (k-fold) for first row only
                    if row_idx == 0:
                        ax.set_title(f"k={col_idx}", fontsize=title_fontsize)
                    
                    # Add tick labels
                    n_queries, n_keys = phi_tensor.shape
                    ax.set_xticks(range(n_keys))
                    ax.set_yticks(range(n_queries))
                    ax.set_xlabel("Keys")
                else:
                    ax.text(0.5, 0.5, "No phi", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                # Plot attention weights (mean ± std)
                att_list = data.attention_weights.get(block_name, [])
                att_tensor = att_list[col_idx] if col_idx < len(att_list) else None
                
                if att_tensor is not None:
                    if len(att_tensor.shape) < 3:
                        att_tensor = np.expand_dims(att_tensor, axis=0)
                    
                    mean = att_tensor.mean(axis=0)
                    std = att_tensor.std(axis=0)
                    
                    im = ax.imshow(mean, vmin=vmin, vmax=vmax, cmap=cmap)
                    row_images.append(im)
                    
                    # Annotate with mean ± std
                    for i in range(mean.shape[0]):
                        for j in range(mean.shape[1]):
                            ax.text(j, i, f"{mean[i, j]:.2f}\n±{std[i, j]:.2f}",
                                   ha="center", va="center", color="white",
                                   fontsize=annotation_fontsize,
                                   fontweight='bold')
                    
                    # Set ylabel (block name) for first column only
                    if col_idx == 0:
                        ax.set_ylabel(block_name, fontsize=title_fontsize)
                    
                    # Set column title (k-fold) for first row only
                    if row_idx == 0:
                        ax.set_title(f"k={col_idx}", fontsize=title_fontsize)
                    
                    # Add tick labels
                    n_queries, n_keys = mean.shape
                    ax.set_xticks(range(n_keys))
                    ax.set_yticks(range(n_queries))
                    ax.set_xlabel("Keys")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Add colorbar for this row
        if row_images:
            cbar_ax = fig.add_subplot(gs[row_idx, n_cols - 1])
            fig.colorbar(row_images[0], cax=cbar_ax)
    
    # Add overall title
    fig.suptitle(f"Attention Scores - {data.architecture_type}", fontsize=title_fontsize + 2, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


# Convenience function for quick access to attention data
def get_attention_summary(data: AttentionData) -> dict:
    """
    Get a summary of the loaded attention data.
    
    Args:
        data: AttentionData object from load_attention_data()
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        "architecture": data.architecture_type,
        "num_folds": len(data.predictions),
        "attention_keys": list(data.attention_weights.keys()),
        "phi_keys": list(data.phi_tensors.keys()),
        "checkpoint_paths": data.checkpoint_paths,
    }
    
    # Add shape info for first fold if available
    if data.predictions:
        first_pred = data.predictions[0]
        summary["input_shape"] = first_pred.inputs.shape
        summary["output_shape"] = first_pred.outputs.shape
        summary["target_shape"] = first_pred.targets.shape
    
    # Add phi availability
    summary["has_encoder_phi"] = any(p is not None for p in data.phi_tensors.get("encoder", []))
    summary["has_decoder_phi"] = any(p is not None for p in data.phi_tensors.get("decoder", []))
    if data.architecture_type == "StageCausalForecaster":
        summary["has_decoder1_phi"] = any(p is not None for p in data.phi_tensors.get("decoder1", []))
        summary["has_decoder2_phi"] = any(p is not None for p in data.phi_tensors.get("decoder2", []))
    
    return summary


# =============================================================================
# Prediction Functions (Long DataFrame Format)
# =============================================================================

def predictions_to_long_df(outputs: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """
    Convert prediction outputs and targets to a long DataFrame format.
    
    Handles different output shapes:
    - (B,) -> single value per sample
    - (B, L) -> sequence output
    - (B, L, F) -> multivariate sequence
    
    Args:
        outputs: Prediction array from model (various shapes)
        targets: Target array (B, L, D) or similar
        
    Returns:
        pd.DataFrame with columns:
            - sample_idx: sample index
            - pos_idx: position index (if sequence)
            - pred_feat_0, pred_feat_1, ...: prediction features
            - trg_feat_0, trg_feat_1, ...: target features
    """
    # Ensure outputs is at least 2D
    if outputs.ndim == 1:
        outputs = outputs[:, np.newaxis]  # (B,) -> (B, 1)
    
    # Ensure outputs is 3D: (B, L, F)
    if outputs.ndim == 2:
        outputs = outputs[:, :, np.newaxis]  # (B, L) -> (B, L, 1)
    
    # Ensure targets is at least 2D
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]
    
    # Ensure targets is 3D: (B, L, D)
    if targets.ndim == 2:
        targets = targets[:, :, np.newaxis]
    
    B, L_out, F_out = outputs.shape
    B_trg, L_trg, D_trg = targets.shape
    
    # Build long format dataframe
    records = []
    
    # Use the minimum length if they differ
    L = min(L_out, L_trg)
    
    for sample_idx in range(B):
        for pos_idx in range(L):
            record = {
                'sample_idx': sample_idx,
                'pos_idx': pos_idx,
            }
            
            # Add prediction features
            for f in range(F_out):
                record[f'pred_feat_{f}'] = outputs[sample_idx, pos_idx, f]
            
            # Add target features
            for d in range(D_trg):
                record[f'trg_feat_{d}'] = targets[sample_idx, pos_idx, d]
            
            records.append(record)
    
    return pd.DataFrame(records)


def predict_from_experiment(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    checkpoint_type: str = "best",
    input_conditioning_fn: Callable = None,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Run predictions for best/last checkpoint across all k-folds of an experiment.
    
    This function is a faster alternative to predict_nested_all_checkpoints when
    you only need predictions from the best (or last) checkpoint per fold, rather
    than all checkpoints across all epochs.
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        datadir_path: Path to data directory. If None, uses "../data/" relative to notebooks
        dataset_label: One of ["train", "test", "all"]
        checkpoint_type: "best" for best_checkpoint.ckpt, "last" for last epoch checkpoint
        input_conditioning_fn: Optional function to condition inputs before forward pass.
                              Use create_intervention_fn() from causaliT.evaluation.predict
                              to create intervention functions for causal analysis.
        save_path: Optional path to save results CSV. If None, only returns DataFrame.
        
    Returns:
        pd.DataFrame: Predictions in long format with columns:
            - sample_idx: sample index in dataset
            - pos_idx: position index within sequence
            - pred_feat_0, pred_feat_1, ...: prediction features
            - trg_feat_0, trg_feat_1, ...: target features
            - kfold: fold identifier (e.g., "k_0", "k_1")
            - checkpoint_name: name of checkpoint file used
            
    Example:
        >>> from lib import predict_from_experiment
        >>> 
        >>> # Basic usage - predictions from best checkpoint
        >>> df = predict_from_experiment("../experiments/my_experiment")
        >>> 
        >>> # With intervention (causal analysis)
        >>> from causaliT.evaluation.predict import create_intervention_fn
        >>> intervention_fn = create_intervention_fn(interventions={1: 0.5})
        >>> df_intervened = predict_from_experiment(
        ...     "../experiments/my_experiment",
        ...     input_conditioning_fn=intervention_fn
        ... )
        >>> 
        >>> # Save results to CSV
        >>> df = predict_from_experiment(
        ...     "../experiments/my_experiment",
        ...     save_path="results/predictions.csv"
        ... )
        >>> 
        >>> # Compare predictions across k-folds
        >>> df.groupby('kfold')['pred_feat_0'].mean()
    """
    # Default data directory
    if datadir_path is None:
        datadir_path = join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    
    # Find config file
    config_path = find_config_file(experiment_path)
    config = OmegaConf.load(config_path)
    
    # Determine architecture type for logging
    architecture_type = get_architecture_type(config)
    print(f"Detected architecture: {architecture_type}")
    
    # Find all k-fold directories
    kfold_dirs = sorted([
        d for d in listdir(experiment_path) 
        if isdir(join(experiment_path, d)) and d.startswith('k_')
    ])
    
    if not kfold_dirs:
        raise ValueError(f"No k-fold directories found in {experiment_path}")
    
    print(f"Found {len(kfold_dirs)} k-fold directories: {kfold_dirs}")
    
    df_list = []
    
    # Process each k-fold
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            # Find checkpoint (best or last)
            checkpoint_path = find_best_or_last_checkpoint(checkpoints_dir)
            checkpoint_name = os.path.basename(checkpoint_path)
            
            print(f"Processing {kfold_dir}: {checkpoint_name}...")
            
            # Run predictions
            predictions = predict_test_from_ckpt(
                config=config,
                datadir_path=datadir_path,
                checkpoint_path=checkpoint_path,
                dataset_label=dataset_label,
                cluster=False,
                input_conditioning_fn=input_conditioning_fn
            )
            
            # Convert to long DataFrame
            df_pred = predictions_to_long_df(
                outputs=predictions.outputs,
                targets=predictions.targets
            )
            
            # Add metadata columns
            df_pred["kfold"] = kfold_dir
            df_pred["checkpoint_name"] = checkpoint_name
            
            df_list.append(df_pred)
            print(f"  ✓ Successfully processed {kfold_dir} ({len(df_pred)} rows)")
            
        except Exception as e:
            print(f"  ✗ Error processing {kfold_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Concatenate all results
    if df_list:
        result_df = pd.concat(df_list, ignore_index=True)
        print(f"\nTotal predictions: {len(result_df)} rows from {len(df_list)} folds")
        
        # Save if path provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            result_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return result_df
    else:
        print("Warning: No predictions were successfully processed")
        return pd.DataFrame()
