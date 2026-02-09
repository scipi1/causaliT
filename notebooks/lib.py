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
from typing import Union, Tuple
from causaliT.training.forecasters.transformer_forecaster import TransformerForecaster
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
from causaliT.training.forecasters.single_causal_forecaster import SingleCausalForecaster
import json
import pickle


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
        str: "TransformerForecaster", "StageCausalForecaster", or "SingleCausalForecaster"
    """
    model_obj = config["model"]["model_object"]
    
    if model_obj == "proT":
        return "TransformerForecaster"
    elif model_obj == "StageCausaliT":
        return "StageCausalForecaster"
    elif model_obj == "SingleCausalLayer":
        return "SingleCausalForecaster"
    else:
        raise ValueError(f"Unknown model type: {model_obj}")


def extract_phi_from_model(model, architecture_type: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract learned DAG probabilities (sigmoid(phi)) from a loaded model.
    
    This function extracts the posterior DAG structure learned by LieAttention and
    CausalCrossAttention modules. It uses `get_dag_probabilities()` which returns
    sigmoid(phi), the actual edge probabilities, rather than raw phi logits.
    
    Args:
        model: Loaded model (TransformerForecaster, StageCausalForecaster, or SingleCausalForecaster)
        architecture_type: "TransformerForecaster", "StageCausalForecaster", or "SingleCausalForecaster"
        
    Returns:
        Dict mapping component name to DAG probability array (or None if not available)
        Keys:
        - TransformerForecaster: "encoder", "decoder", "cross"
        - StageCausalForecaster: "decoder1", "decoder1_cross", "decoder2", "decoder2_cross"
        - SingleCausalForecaster: "decoder", "decoder_cross"
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
        
    elif architecture_type == "SingleCausalForecaster":
        # Decoder self-attention DAG (X -> X structure)
        dec_self_inner = model.model.decoder.layers[0].global_self_attention.inner_attention
        phi_dict["decoder"] = _get_dag_probs(dec_self_inner)
        
        # Decoder cross-attention DAG (S -> X structure)
        dec_cross_inner = model.model.decoder.layers[0].global_cross_attention.inner_attention
        phi_dict["decoder_cross"] = _get_dag_probs(dec_cross_inner)
        
        # Compatibility keys
        phi_dict["encoder"] = None  # No encoder in SingleCausal
        phi_dict["cross"] = phi_dict["decoder_cross"]  # Alias for compatibility
    
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
    elif architecture_type == "StageCausalForecaster":
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
    elif architecture_type == "SingleCausalForecaster":
        result.attention_weights = {
            "encoder": [],  # Empty for compatibility
            "decoder": [],  # Mapped to dec_self for compatibility
            "cross": [],    # Mapped to dec_cross for compatibility
            "dec_self": [],
            "dec_cross": [],
        }
        result.phi_tensors = {
            "encoder": [],
            "decoder": [],
            "decoder_cross": [],  # Cross-attention DAG (S -> X)
            "cross": [],  # Alias for decoder_cross
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
            elif architecture_type == "StageCausalForecaster":
                # StageCausalPredictor returns keys: dec1_cross, dec1_self, dec2_cross, dec2_self
                result.attention_weights["decoder1_self"].append(att_weights.get("dec1_self"))
                result.attention_weights["decoder1_cross"].append(att_weights.get("dec1_cross"))
                result.attention_weights["decoder2_self"].append(att_weights.get("dec2_self"))
                result.attention_weights["decoder2_cross"].append(att_weights.get("dec2_cross"))
                # For compatibility with notebook code expecting encoder/decoder/cross
                result.attention_weights["encoder"].append(None)
                result.attention_weights["decoder"].append(att_weights.get("dec2_self"))
                result.attention_weights["cross"].append(att_weights.get("dec2_cross"))
            elif architecture_type == "SingleCausalForecaster":
                # SingleCausalPredictor returns keys: dec_cross, dec_self
                result.attention_weights["dec_self"].append(att_weights.get("dec_self"))
                result.attention_weights["dec_cross"].append(att_weights.get("dec_cross"))
                # For compatibility with notebook code expecting encoder/decoder/cross
                result.attention_weights["encoder"].append(None)
                result.attention_weights["decoder"].append(att_weights.get("dec_self"))
                result.attention_weights["cross"].append(att_weights.get("dec_cross"))
            
            # Load model and extract phi tensors
            if architecture_type == "TransformerForecaster":
                model = TransformerForecaster.load_from_checkpoint(checkpoint_path)
            elif architecture_type == "StageCausalForecaster":
                model = StageCausalForecaster.load_from_checkpoint(checkpoint_path)
            elif architecture_type == "SingleCausalForecaster":
                model = SingleCausalForecaster.load_from_checkpoint(checkpoint_path)
            
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


# Convenience function for quick access to attention data
# NOTE: Plotting functions moved to plot_lib.py:
#   - plot_attention_scores
#   - plot_attention_evolution
#   - plot_phi_evolution
# =============================================================================
# HSIC (Hilbert-Schmidt Independence Criterion) Functions
# =============================================================================

def compute_hsic(
    x: Union[np.ndarray, pd.Series, 'torch.Tensor'],
    y: Union[np.ndarray, pd.Series, 'torch.Tensor'],
    sigma: float = None,
    n_permutations: int = None,
) -> Union[float, Tuple[float, float]]:
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) between two variables.
    
    HSIC measures statistical dependence: 0 = independent, higher = more dependent.
    Uses RBF (Gaussian) kernel with median heuristic for bandwidth selection.
    
    Args:
        x: First variable - pandas Series, numpy array, or torch tensor (1D)
        y: Second variable - pandas Series, numpy array, or torch tensor (1D)
        sigma: RBF kernel bandwidth. If None, uses median heuristic (recommended)
        n_permutations: If provided, also returns p-value from permutation test
                       (e.g., 1000 permutations). P-value tests null hypothesis
                       that x and y are independent.
        
    Returns:
        If n_permutations is None:
            hsic_value: float - the HSIC statistic
        If n_permutations > 0:
            (hsic_value, p_value): tuple - HSIC and p-value from permutation test
            
    Example:
        >>> import numpy as np
        >>> from lib import compute_hsic
        >>> 
        >>> # Independent variables - low HSIC
        >>> x = np.random.randn(500)
        >>> y = np.random.randn(500)
        >>> hsic = compute_hsic(x, y)
        >>> print(f"HSIC (independent): {hsic:.6f}")
        >>> 
        >>> # Dependent variables - high HSIC
        >>> x = np.random.randn(500)
        >>> y = x + 0.1 * np.random.randn(500)  # y depends on x
        >>> hsic = compute_hsic(x, y)
        >>> print(f"HSIC (dependent): {hsic:.6f}")
        >>> 
        >>> # With p-value (permutation test)
        >>> hsic, pval = compute_hsic(x, y, n_permutations=1000)
        >>> print(f"HSIC: {hsic:.6f}, p-value: {pval:.4f}")
        >>> 
        >>> # Using pandas columns
        >>> df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
        >>> hsic = compute_hsic(df['a'], df['b'])
    """
    # Convert inputs to torch tensors
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    
    # Validate inputs
    if x_t.dim() != 1 or y_t.dim() != 1:
        raise ValueError(f"Inputs must be 1D. Got x.shape={x_t.shape}, y.shape={y_t.shape}")
    if len(x_t) != len(y_t):
        raise ValueError(f"Inputs must have same length. Got {len(x_t)} and {len(y_t)}")
    
    n = len(x_t)
    if n < 4:
        raise ValueError(f"Need at least 4 samples, got {n}")
    
    # Compute HSIC
    hsic_value = _hsic_torch(x_t, y_t, sigma=sigma)
    
    # Permutation test if requested
    if n_permutations is not None and n_permutations > 0:
        count_geq = 0
        for _ in range(n_permutations):
            perm_idx = torch.randperm(n)
            y_perm = y_t[perm_idx]
            hsic_perm = _hsic_torch(x_t, y_perm, sigma=sigma)
            if hsic_perm >= hsic_value:
                count_geq += 1
        p_value = (count_geq + 1) / (n_permutations + 1)  # +1 for continuity correction
        return float(hsic_value), float(p_value)
    
    return float(hsic_value)


def _to_tensor(x: Union[np.ndarray, pd.Series, 'torch.Tensor']) -> 'torch.Tensor':
    """Convert input to a 1D float torch tensor."""
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return x.float().flatten()


def _rbf_kernel(x: 'torch.Tensor', sigma: float) -> 'torch.Tensor':
    """
    Compute RBF (Gaussian) kernel matrix.
    
    K(i,j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    """
    # Pairwise squared distances
    x = x.unsqueeze(1)  # (n, 1)
    dists_sq = (x - x.T) ** 2  # (n, n)
    return torch.exp(-dists_sq / (2 * sigma ** 2))


def _median_heuristic(x: 'torch.Tensor') -> float:
    """
    Compute bandwidth using median heuristic.
    
    sigma = median of pairwise distances / sqrt(2)
    """
    x = x.unsqueeze(1)
    dists = torch.abs(x - x.T)
    # Get upper triangle (excluding diagonal)
    n = len(x)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    pairwise_dists = dists[mask]
    
    if len(pairwise_dists) == 0 or pairwise_dists.median() == 0:
        return 1.0  # Fallback
    
    return float(pairwise_dists.median() / np.sqrt(2))


def _hsic_torch(
    x: 'torch.Tensor',
    y: 'torch.Tensor',
    sigma: float = None
) -> 'torch.Tensor':
    """
    Compute HSIC using PyTorch.
    
    Uses the unbiased estimator: HSIC = (1/n(n-3)) * [tr(KHLH) + ...]
    Simplified to biased estimator for efficiency: HSIC = (1/(n-1)^2) * tr(KHLH)
    """
    n = len(x)
    
    # Compute bandwidths using median heuristic if not provided
    sigma_x = sigma if sigma is not None else _median_heuristic(x)
    sigma_y = sigma if sigma is not None else _median_heuristic(y)
    
    # Compute kernel matrices
    K = _rbf_kernel(x, sigma_x)
    L = _rbf_kernel(y, sigma_y)
    
    # Centering matrix H = I - (1/n) * 1*1^T
    H = torch.eye(n) - torch.ones(n, n) / n
    
    # Centered kernels
    KH = K @ H
    LH = L @ H
    
    # HSIC = (1/(n-1)^2) * tr(KH @ LH)
    # tr(A @ B) = sum(A * B.T) for efficiency
    hsic = (KH * LH.T).sum() / ((n - 1) ** 2)
    
    return hsic


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


def extract_source_values(
    inputs: np.ndarray,
    var_ids: List[int] = [1, 2],
    val_idx: int = 0,
    var_idx: int = 1
) -> Dict[int, np.ndarray]:
    """
    Extract values for specific variables from the source (S) tensor.
    
    The S tensor has shape (B, L_s, F) where each token has features [value, var_id, ...].
    This function extracts the value for each specified var_id per sample.
    
    Args:
        inputs: Source tensor (B, L_s, F) or (B, L_s) if squeezed
        var_ids: List of variable IDs to extract (default [1, 2] for S1, S2)
        val_idx: Index of the value feature (default 0)
        var_idx: Index of the variable ID feature (default 1)
        
    Returns:
        Dict mapping var_id to array of values per sample (B,)
        
    Notes:
        - If a variable ID is not found in a sample, returns NaN for that sample
        - Assumes each variable appears at most once per sample (takes first occurrence)
    """
    # Ensure inputs is 3D
    if inputs.ndim == 2:
        inputs = inputs[:, :, np.newaxis]
    
    B, L, F = inputs.shape
    result = {}
    
    for var_id in var_ids:
        values = np.full(B, np.nan)
        for b in range(B):
            # Find tokens with this variable ID
            var_id_per_token = inputs[b, :, var_idx]
            mask = (var_id_per_token == var_id)
            if mask.any():
                # Take the first occurrence
                token_idx = np.argmax(mask)
                values[b] = inputs[b, token_idx, val_idx]
        result[var_id] = values
    
    return result


def compute_mse_per_sample(
    outputs: np.ndarray,
    targets: np.ndarray,
    val_idx: int = 0
) -> np.ndarray:
    """
    Compute MSE for each sample across all positions and features.
    
    Args:
        outputs: Predictions array (B, L) or (B, L, F)
        targets: Targets array (B, L, F) or (B, L)
        val_idx: Index of the value feature in targets (used if targets has more features than outputs)
        
    Returns:
        Array of MSE values per sample (B,)
    """
    # Ensure both are at least 2D
    if outputs.ndim == 1:
        outputs = outputs[:, np.newaxis]
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]
    
    # Handle different shapes
    if outputs.ndim == 2 and targets.ndim == 3:
        # outputs: (B, L), targets: (B, L, F) - extract value feature from targets
        targets = targets[:, :, val_idx]
    elif outputs.ndim == 3 and targets.ndim == 2:
        # outputs: (B, L, F), targets: (B, L) - squeeze outputs
        outputs = outputs[:, :, val_idx]
    elif outputs.ndim == 3 and targets.ndim == 3:
        # Both 3D - ensure same number of features or use val_idx
        if outputs.shape[-1] != targets.shape[-1]:
            outputs = outputs[:, :, val_idx]
            targets = targets[:, :, val_idx]
    
    # Compute MSE: mean over positions (and features if present)
    mse = np.mean((outputs - targets) ** 2, axis=tuple(range(1, outputs.ndim)))
    return mse


def compute_abs_error_per_feature(
    outputs: np.ndarray,
    targets: np.ndarray,
    val_idx: int = 0
) -> np.ndarray:
    """
    Compute absolute error for each sample and each position/feature separately.
    
    Args:
        outputs: Predictions array (B, L) or (B, L, F)
        targets: Targets array (B, L, F) or (B, L)
        val_idx: Index of the value feature in targets (used if targets has more features than outputs)
        
    Returns:
        Array of absolute errors per sample and position (B, L)
    """
    # Ensure both are at least 2D
    if outputs.ndim == 1:
        outputs = outputs[:, np.newaxis]
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]
    
    # Handle different shapes
    if outputs.ndim == 2 and targets.ndim == 3:
        # outputs: (B, L), targets: (B, L, F) - extract value feature from targets
        targets = targets[:, :, val_idx]
    elif outputs.ndim == 3 and targets.ndim == 2:
        # outputs: (B, L, F), targets: (B, L) - squeeze outputs
        outputs = outputs[:, :, val_idx]
    elif outputs.ndim == 3 and targets.ndim == 3:
        # Both 3D - ensure same number of features or use val_idx
        if outputs.shape[-1] != targets.shape[-1]:
            outputs = outputs[:, :, val_idx]
            targets = targets[:, :, val_idx]
    
    # Compute absolute error per position
    abs_error = np.abs(outputs - targets)
    return abs_error


def predict_with_error_analysis(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    checkpoint_type: str = "best",
    input_conditioning_fn: Callable = None,
    source_var_ids: List[int] = [1, 2, 3],
    val_idx: int = 0,
    var_idx: int = 1,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Run predictions and compute per-sample error analysis with input variable values.
    
    This function extends predict_from_experiment to include:
    - Absolute error per feature for Y predictions (abs_err_y_0, abs_err_y_1, ...)
    - Absolute error per feature for X predictions (abs_err_x_0, abs_err_x_1, ...) for StageCausaliT
    - Extracted values of source variables (S1, S2, S3, etc.)
    
    Use this for analyzing which input combinations lead to higher prediction errors.
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        datadir_path: Path to data directory. If None, uses "../data/" relative to notebooks
        dataset_label: One of ["train", "test", "all"]
        checkpoint_type: "best" for best_checkpoint.ckpt, "last" for last epoch checkpoint
        input_conditioning_fn: Optional function to condition inputs before forward pass
        source_var_ids: List of variable IDs to extract from source (default [1, 2, 3] for S1, S2, S3)
        val_idx: Index of the value feature in tokens (default 0)
        var_idx: Index of the variable ID feature in tokens (default 1)
        save_path: Optional path to save results CSV
        
    Returns:
        pd.DataFrame with columns:
            - sample_idx: sample index in dataset
            - kfold: fold identifier (e.g., "k_0", "k_1")
            - checkpoint_name: name of checkpoint file used
            - s1_value, s2_value, ...: extracted source variable values (named by var_id)
            - abs_err_y_0, abs_err_y_1, ...: absolute error for each Y feature
            - abs_err_x_0, abs_err_x_1, ...: absolute error for each X feature (StageCausaliT only)
            
    Example:
        >>> from lib import predict_with_error_analysis
        >>> 
        >>> # Basic usage - analyze prediction errors
        >>> df = predict_with_error_analysis("../experiments/stage_Lie_ccross_scm6")
        >>> 
        >>> # Find samples with highest Y1 errors
        >>> high_error = df.nlargest(100, 'abs_err_y_0')
        >>> 
        >>> # Analyze errors by S1, S2 combinations
        >>> import seaborn as sns
        >>> sns.scatterplot(data=df, x='s1_value', y='s2_value', hue='abs_err_y_0')
        >>> 
        >>> # Heatmap of errors by binned S1, S2
        >>> df['s1_bin'] = pd.cut(df['s1_value'], bins=10)
        >>> df['s2_bin'] = pd.cut(df['s2_value'], bins=10)
        >>> heatmap = df.groupby(['s1_bin', 's2_bin'])['abs_err_y_0'].mean().unstack()
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
            
            # Get number of samples
            num_samples = predictions.outputs.shape[0] if predictions.outputs.ndim > 0 else 1
            
            # Determine column prefix based on architecture type
            # SingleCausalForecaster predicts X, others predict Y
            if architecture_type == "SingleCausalForecaster":
                main_output_prefix = "x"  # Main output is X predictions
            else:
                main_output_prefix = "y"  # Main output is Y predictions
            
            # Compute absolute errors for main predictions (Y for Stage/Transformer, X for Single)
            abs_err_main = compute_abs_error_per_feature(
                predictions.outputs, 
                predictions.targets,
                val_idx=val_idx
            )
            # abs_err_main shape: (B, L) where L is number of output features (positions)
            
            # Compute absolute errors for X predictions per feature (StageCausaliT only)
            abs_err_x_from_metadata = None
            if 'pred_x' in predictions.metadata and 'targets_x' in predictions.metadata:
                abs_err_x_from_metadata = compute_abs_error_per_feature(
                    predictions.metadata['pred_x'],
                    predictions.metadata['targets_x'],
                    val_idx=val_idx
                )
            
            # Extract source variable values
            source_values = extract_source_values(
                predictions.inputs,
                var_ids=source_var_ids,
                val_idx=val_idx,
                var_idx=var_idx
            )
            
            # Build DataFrame for this fold
            records = []
            for sample_idx in range(num_samples):
                record = {
                    'sample_idx': sample_idx,
                    'kfold': kfold_dir,
                    'checkpoint_name': checkpoint_name,
                }
                
                # Add absolute errors for main output features (x for Single, y for others)
                if abs_err_main.ndim == 1:
                    # Single feature
                    record[f'abs_err_{main_output_prefix}_0'] = abs_err_main[sample_idx]
                else:
                    # Multiple features (B, L)
                    for feat_idx in range(abs_err_main.shape[1]):
                        record[f'abs_err_{main_output_prefix}_{feat_idx}'] = abs_err_main[sample_idx, feat_idx]
                
                # Add absolute errors for X predictions from metadata (StageCausaliT only)
                if abs_err_x_from_metadata is not None:
                    if abs_err_x_from_metadata.ndim == 1:
                        record['abs_err_x_0'] = abs_err_x_from_metadata[sample_idx]
                    else:
                        for feat_idx in range(abs_err_x_from_metadata.shape[1]):
                            record[f'abs_err_x_{feat_idx}'] = abs_err_x_from_metadata[sample_idx, feat_idx]
                
                # Add source variable values
                for var_id in source_var_ids:
                    # Name columns by var_id: s1_value, s2_value, etc.
                    record[f's{var_id}_value'] = source_values[var_id][sample_idx]
                
                records.append(record)
            
            df_fold = pd.DataFrame(records)
            df_list.append(df_fold)
            print(f"  ✓ Successfully processed {kfold_dir} ({len(df_fold)} samples)")
            
        except Exception as e:
            print(f"  ✗ Error processing {kfold_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Concatenate all results
    if df_list:
        result_df = pd.concat(df_list, ignore_index=True)
        print(f"\nTotal samples: {len(result_df)} from {len(df_list)} folds")
        
        # Save if path provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            result_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return result_df
    else:
        print("Warning: No predictions were successfully processed")
        return pd.DataFrame()


def load_training_metrics(
    experiment_path: str,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Load training metrics from all k-folds of an experiment.
    
    For each k-fold, loads the metrics.csv from logs/csv/version_0/
    and concatenates them with fold information.
    
    Args:
        experiment_path: Path to the experiment folder containing k_* folders
        save_path: Optional path to save results CSV. If None, only returns DataFrame.
        
    Returns:
        pd.DataFrame: Combined metrics with an additional 'kfold' column containing
                     fold identifiers (e.g., "k_0", "k_1", etc.)
        
    Example:
        >>> from lib import load_training_metrics
        >>> 
        >>> # Load metrics from all folds
        >>> df = load_training_metrics("../experiments/euler/stage_SoftMax_scm6_54445164")
        >>> 
        >>> # View available columns
        >>> print(df.columns.tolist())
        >>> 
        >>> # Analyze training curves across folds
        >>> df.groupby(['kfold', 'epoch'])['train_loss'].mean()
        >>> 
        >>> # Plot validation loss over epochs for each fold
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> sns.lineplot(data=df, x='epoch', y='val_loss', hue='kfold')
        >>> plt.show()
    """
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
        metrics_path = join(experiment_path, kfold_dir, 'logs', 'csv', 'version_0', 'metrics.csv')
        
        if not exists(metrics_path):
            print(f"  ✗ Warning: metrics.csv not found for {kfold_dir}: {metrics_path}")
            continue
        
        try:
            # Load metrics CSV
            df_fold = pd.read_csv(metrics_path)
            
            # Add k-fold identifier column
            df_fold['kfold'] = kfold_dir
            
            df_list.append(df_fold)
            print(f"  ✓ Loaded {kfold_dir}: {len(df_fold)} rows")
            
        except Exception as e:
            print(f"  ✗ Error loading {kfold_dir}: {e}")
            continue
    
    # Concatenate all results
    if df_list:
        result_df = pd.concat(df_list, ignore_index=True)
        print(f"\nTotal rows: {len(result_df)} from {len(df_list)} folds")
        
        # Save if path provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            result_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return result_df
    else:
        print("Warning: No metrics files were successfully loaded")
        return pd.DataFrame()


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


def _flatten_attention_tensor(
    tensor: np.ndarray, 
    block_name: str,
    is_phi: bool = False
) -> Dict[str, float]:
    """
    Flatten a 2D attention tensor into a dictionary with indexed column names.
    
    Args:
        tensor: 2D array of shape (queries, keys)
        block_name: Name of the attention block (e.g., "dec1_self")
        is_phi: If True, prefix with "phi_"
        
    Returns:
        Dict mapping column names to values
        E.g., {"dec1_self_00": 0.5, "dec1_self_01": 0.3, ...}
    """
    result = {}
    prefix = f"phi_{block_name}" if is_phi else block_name
    
    n_rows, n_cols = tensor.shape
    for i in range(n_rows):
        for j in range(n_cols):
            col_name = f"{prefix}_{i}{j}"
            result[col_name] = tensor[i, j]
    
    return result


def load_attention_evolution(
    experiment_path: str,
    datadir_path: str = None,
    dataset_label: str = "test",
    input_conditioning_fn: Callable = None,
    n_evaluations: int = None,
) -> pd.DataFrame:
    """
    Load attention scores and phi tensors across all training epochs to track their evolution.
    
    .. deprecated::
        This function has been moved to `notebooks.eval_fun.load_attention_evolution` 
        with an improved `n_evaluations` parameter. Please use:
        
        >>> from notebooks.eval_fun import load_attention_evolution
        
        The new version supports `n_evaluations` parameter (default=10) which selects
        evenly-spaced checkpoints for consistent evaluation time regardless of total epochs.
    
    This function tracks how learned DAG structure (attention scores and phi tensors) 
    evolve during training from initialization. For each checkpoint across all k-folds,
    it computes the difference from initialization at the sample level, then aggregates
    to mean and std.
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        datadir_path: Path to data directory. If None, uses "../data/" relative to notebooks
        dataset_label: One of ["train", "test", "all"]
        input_conditioning_fn: Optional function to condition inputs before forward pass
        n_evaluations: [DEPRECATED - use eval_fun.load_attention_evolution instead]
                      Number of checkpoints to evaluate. If None, evaluates ALL checkpoints.
        
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
        >>> # DEPRECATED: Use notebooks.eval_fun.load_attention_evolution instead
        >>> from notebooks.eval_fun import load_attention_evolution
        >>> 
        >>> # Load attention evolution with 10 evaluation points (default)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6")
        >>> 
        >>> # Load ALL checkpoints (slower, more detailed)
        >>> df = load_attention_evolution("../experiments/euler/stage_Lie_scm6", n_evaluations=0)
    """
    import warnings
    warnings.warn(
        "load_attention_evolution in lib.py is deprecated. "
        "Use 'from notebooks.eval_fun import load_attention_evolution' instead. "
        "The new version supports n_evaluations parameter for faster evaluation.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Default data directory
    if datadir_path is None:
        datadir_path = join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    
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
            print(f"\n{kfold_dir}: Found {len(epoch_checkpoints)} checkpoints")
            
            if not epoch_checkpoints:
                print(f"  ✗ No checkpoints found for {kfold_dir}")
                continue
            
            # Storage for initial attention scores (for computing diffs)
            init_attention = {}  # key -> (B, Q, K) array
            init_phi = {}  # key -> (Q, K) array
            
            # Process each checkpoint
            for epoch, checkpoint_path in epoch_checkpoints:
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
                            
                            # For epoch 0, store as initial
                            if epoch == 0:
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
                        
                        # For epoch 0, store as initial
                        if epoch == 0:
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


def extract_embeddings_from_model(model, architecture_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract embedding weight matrices from a loaded model.
    
    This function extracts all learnable embedding parameters including:
    - nn.Embedding lookup tables
    - Linear projection layers in embeddings
    - Time2Vec periodic layers
    - Orthogonal mask embeddings (for SingleCausalLayer)
    
    Args:
        model: Loaded model (TransformerForecaster, StageCausalForecaster, or SingleCausalForecaster)
        architecture_type: "TransformerForecaster", "StageCausalForecaster", or "SingleCausalForecaster"
        
    Returns:
        Dict mapping embedding name to dict with:
        - "weight": np.ndarray of the embedding weights
        - "type": str describing the layer type
        - "shape": tuple of weight shape
        - "component": str indicating which model component (e.g., "shared_embedding", "embedding_S")
    """
    embeddings = {}
    
    def _extract_modular_embedding(modular_emb, prefix: str):
        """Helper to extract weights from a ModularEmbedding module."""
        for i, emb_map in enumerate(modular_emb.embed_modules_list):
            var_idx = emb_map.var_idx
            emb_module = emb_map.embedding
            
            # Determine embedding type and extract weights
            emb_type = type(emb_module).__name__
            
            if hasattr(emb_module, 'weight'):
                # nn.Embedding or similar
                weight = emb_module.weight.detach().cpu().numpy()
                key = f"{prefix}_var{var_idx}_{emb_type}"
                embeddings[key] = {
                    "weight": weight,
                    "type": emb_type,
                    "shape": weight.shape,
                    "component": prefix,
                    "var_idx": var_idx,
                }
            
            # Also extract any Linear layers within the embedding
            for name, param in emb_module.named_parameters():
                if 'weight' in name or 'bias' in name:
                    weight = param.detach().cpu().numpy()
                    key = f"{prefix}_var{var_idx}_{emb_type}_{name.replace('.', '_')}"
                    embeddings[key] = {
                        "weight": weight,
                        "type": f"{emb_type}.{name}",
                        "shape": weight.shape,
                        "component": prefix,
                        "var_idx": var_idx,
                    }
        
        # Extract spatiotemporal projection if present
        if hasattr(modular_emb, 'W_time_val'):
            weight = modular_emb.W_time_val.weight.detach().cpu().numpy()
            key = f"{prefix}_W_time_val_weight"
            embeddings[key] = {
                "weight": weight,
                "type": "Linear.weight",
                "shape": weight.shape,
                "component": prefix,
                "var_idx": None,
            }
            if modular_emb.W_time_val.bias is not None:
                bias = modular_emb.W_time_val.bias.detach().cpu().numpy()
                key = f"{prefix}_W_time_val_bias"
                embeddings[key] = {
                    "weight": bias,
                    "type": "Linear.bias",
                    "shape": bias.shape,
                    "component": prefix,
                    "var_idx": None,
                }
    
    def _extract_orthogonal_embedding(ortho_emb, prefix: str):
        """Helper to extract weights from an OrthogonalMaskEmbedding module."""
        # Value embedding linear layer
        weight = ortho_emb.value_embedding.weight.detach().cpu().numpy()
        embeddings[f"{prefix}_value_embedding_weight"] = {
            "weight": weight,
            "type": "Linear.weight",
            "shape": weight.shape,
            "component": prefix,
            "var_idx": None,
        }
        if ortho_emb.value_embedding.bias is not None:
            bias = ortho_emb.value_embedding.bias.detach().cpu().numpy()
            embeddings[f"{prefix}_value_embedding_bias"] = {
                "weight": bias,
                "type": "Linear.bias",
                "shape": bias.shape,
                "component": prefix,
                "var_idx": None,
            }
        
        # Binary masks (buffer, not trainable but useful for analysis)
        masks = ortho_emb.binary_masks.detach().cpu().numpy()
        embeddings[f"{prefix}_binary_masks"] = {
            "weight": masks,
            "type": "buffer",
            "shape": masks.shape,
            "component": prefix,
            "var_idx": None,
        }
    
    # Extract forecaster (de-embedding) weights
    def _extract_forecaster(forecaster, prefix: str):
        """Helper to extract weights from forecaster (de-embedding) layer."""
        weight = forecaster.weight.detach().cpu().numpy()
        embeddings[f"{prefix}_weight"] = {
            "weight": weight,
            "type": "Linear.weight",
            "shape": weight.shape,
            "component": prefix,
            "var_idx": None,
        }
        if forecaster.bias is not None:
            bias = forecaster.bias.detach().cpu().numpy()
            embeddings[f"{prefix}_bias"] = {
                "weight": bias,
                "type": "Linear.bias",
                "shape": bias.shape,
                "component": prefix,
                "var_idx": None,
            }
    
    if architecture_type == "TransformerForecaster":
        # Encoder embedding
        if hasattr(model.model, 'encoder_embedding'):
            _extract_modular_embedding(model.model.encoder_embedding, "encoder_embedding")
        
        # Decoder embedding
        if hasattr(model.model, 'decoder_embedding'):
            _extract_modular_embedding(model.model.decoder_embedding, "decoder_embedding")
        
        # Forecaster
        if hasattr(model.model, 'forecaster'):
            _extract_forecaster(model.model.forecaster, "forecaster")
            
    elif architecture_type == "StageCausalForecaster":
        # Shared embedding
        if hasattr(model.model, 'shared_embedding'):
            _extract_modular_embedding(model.model.shared_embedding, "shared_embedding")
        
        # Forecaster
        if hasattr(model.model, 'forecaster'):
            _extract_forecaster(model.model.forecaster, "forecaster")
            
    elif architecture_type == "SingleCausalForecaster":
        # S embedding (orthogonal)
        if hasattr(model.model, 'embedding_S'):
            _extract_orthogonal_embedding(model.model.embedding_S, "embedding_S")
        
        # X embedding (modular)
        if hasattr(model.model, 'embedding_X'):
            _extract_modular_embedding(model.model.embedding_X, "embedding_X")
        
        # Forecaster
        if hasattr(model.model, 'forecaster'):
            _extract_forecaster(model.model.forecaster, "forecaster")
    
    return embeddings


def load_embeddings_evolution(
    experiment_path: str,
) -> pd.DataFrame:
    """
    Load embedding weight matrices from all training checkpoints to track their evolution.
    
    This function extracts embedding parameters (nn.Embedding lookup tables, linear layers, etc.)
    from each saved checkpoint across all k-folds, enabling analysis of how embeddings
    evolve during training.
    
    The embeddings are returned in a format suitable for computing cosine similarities
    between checkpoints to quantify how much embeddings change during training.
    
    Args:
        experiment_path: Path to the experiment folder containing config and k_* folders
        
    Returns:
        pd.DataFrame with columns:
            - kfold: fold identifier (e.g., "k_0", "k_1")
            - epoch: epoch number (0 for initialization)
            - embedding_name: name of the embedding (e.g., "shared_embedding_var0_nn_embedding")
            - weight: flattened embedding weights as numpy array
            - type: embedding layer type (e.g., "nn_embedding", "Linear.weight")
            - shape: original shape of the weight matrix
            - component: model component (e.g., "shared_embedding", "embedding_S")
            
    Example:
        >>> from lib import load_embeddings_evolution
        >>> import numpy as np
        >>> from sklearn.metrics.pairwise import cosine_similarity
        >>> 
        >>> # Load embedding evolution for an experiment
        >>> df = load_embeddings_evolution("../experiments/single/single_Lie_CC_scm6")
        >>> 
        >>> # Get unique embedding names
        >>> print(df['embedding_name'].unique())
        >>> 
        >>> # Track evolution of a specific embedding across epochs
        >>> emb_name = "shared_embedding_var0_nn_embedding"
        >>> emb_df = df[df['embedding_name'] == emb_name]
        >>> 
        >>> # Compute cosine similarity between epoch 0 and later epochs for k_0
        >>> k0_df = emb_df[emb_df['kfold'] == 'k_0'].sort_values('epoch')
        >>> weights = np.array([w for w in k0_df['weight']])
        >>> 
        >>> # Cosine similarity matrix
        >>> cos_sim_matrix = cosine_similarity(weights)
        >>> print(f"Cosine similarity epoch 0 vs epoch 5: {cos_sim_matrix[0, 5]:.4f}")
        >>> 
        >>> # Plot cosine similarity from initial embedding over epochs
        >>> import matplotlib.pyplot as plt
        >>> epochs = k0_df['epoch'].values
        >>> sim_from_init = cos_sim_matrix[0, :]  # Row 0 = similarity with epoch 0
        >>> plt.plot(epochs, sim_from_init)
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('Cosine similarity with initial embedding')
        >>> plt.title(emb_name)
        >>> plt.show()
    """
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
    
    all_records = []
    
    # Process each k-fold
    for kfold_dir in kfold_dirs:
        kfold_path = join(experiment_path, kfold_dir)
        checkpoints_dir = join(kfold_path, 'checkpoints')
        
        try:
            # Find all checkpoints sorted by epoch
            epoch_checkpoints = find_all_checkpoints(checkpoints_dir)
            print(f"\n{kfold_dir}: Found {len(epoch_checkpoints)} checkpoints")
            
            if not epoch_checkpoints:
                print(f"  ✗ No checkpoints found for {kfold_dir}")
                continue
            
            # Process each checkpoint
            for epoch, checkpoint_path in epoch_checkpoints:
                print(f"  Processing epoch {epoch}: {os.path.basename(checkpoint_path)}")
                
                try:
                    # Load model from checkpoint
                    if architecture_type == "TransformerForecaster":
                        model = TransformerForecaster.load_from_checkpoint(checkpoint_path)
                    elif architecture_type == "StageCausalForecaster":
                        model = StageCausalForecaster.load_from_checkpoint(checkpoint_path)
                    elif architecture_type == "SingleCausalForecaster":
                        model = SingleCausalForecaster.load_from_checkpoint(checkpoint_path)
                    
                    # Extract embeddings
                    embeddings = extract_embeddings_from_model(model, architecture_type)
                    
                    # Create records for each embedding
                    for emb_name, emb_data in embeddings.items():
                        record = {
                            'kfold': kfold_dir,
                            'epoch': epoch,
                            'embedding_name': emb_name,
                            'weight': emb_data['weight'].flatten(),  # Flatten for cosine similarity
                            'type': emb_data['type'],
                            'shape': emb_data['shape'],
                            'component': emb_data['component'],
                        }
                        all_records.append(record)
                    
                    print(f"    ✓ Extracted {len(embeddings)} embeddings from epoch {epoch}")
                    
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
        print(f"\nLoaded embeddings: {len(df)} rows from {df['kfold'].nunique()} folds")
        print(f"Unique embeddings: {df['embedding_name'].nunique()}")
        return df
    else:
        print("Warning: No records were successfully processed")
        return pd.DataFrame()


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


# =============================================================================
# AttentionData Save/Load Functions
# =============================================================================

def save_attention_data(
    data: AttentionData,
    save_dir: str,
    save_predictions: bool = False,
) -> None:
    """
    Save an AttentionData object to disk.
    
    Creates the following files in save_dir:
    - attention_weights.npz: All attention weight arrays
    - phi_tensors.npz: All phi tensor arrays
    - config.yaml: The OmegaConf config
    - metadata.json: architecture_type, checkpoint_paths
    - predictions.pkl (optional): Prediction objects if save_predictions=True
    
    Args:
        data: AttentionData object to save
        save_dir: Directory to save files to (created if doesn't exist)
        save_predictions: If True, also save predictions (can be large)
        
    Example:
        >>> from lib import load_attention_data, save_attention_data
        >>> 
        >>> # Load from experiment
        >>> data = load_attention_data("../experiments/my_experiment")
        >>> 
        >>> # Save to disk
        >>> save_attention_data(data, "saved_attention/my_experiment")
        >>> 
        >>> # Save with predictions (larger file)
        >>> save_attention_data(data, "saved_attention/my_experiment", save_predictions=True)
    """
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save attention weights
    att_weights_dict = {}
    for key, arr_list in data.attention_weights.items():
        for i, arr in enumerate(arr_list):
            if arr is not None:
                att_weights_dict[f"{key}_{i}"] = arr
    
    if att_weights_dict:
        np.savez(join(save_dir, "attention_weights.npz"), **att_weights_dict)
        print(f"Saved {len(att_weights_dict)} attention weight arrays")
    
    # Save phi tensors
    phi_dict = {}
    for key, arr_list in data.phi_tensors.items():
        for i, arr in enumerate(arr_list):
            if arr is not None:
                phi_dict[f"{key}_{i}"] = arr
    
    if phi_dict:
        np.savez(join(save_dir, "phi_tensors.npz"), **phi_dict)
        print(f"Saved {len(phi_dict)} phi tensor arrays")
    
    # Save config
    if data.config is not None:
        OmegaConf.save(data.config, join(save_dir, "config.yaml"))
        print("Saved config.yaml")
    
    # Save metadata
    metadata = {
        "architecture_type": data.architecture_type,
        "checkpoint_paths": data.checkpoint_paths,
        "attention_keys": list(data.attention_weights.keys()),
        "phi_keys": list(data.phi_tensors.keys()),
        "num_folds": len(data.predictions),
    }
    with open(join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata.json")
    
    # Optionally save predictions
    if save_predictions and data.predictions:
        with open(join(save_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(data.predictions, f)
        print(f"Saved predictions.pkl ({len(data.predictions)} predictions)")
    
    print(f"\n✓ AttentionData saved to {save_dir}")


def load_attention_data_from_file(load_dir: str) -> AttentionData:
    """
    Load an AttentionData object from disk.
    
    Reconstructs an AttentionData object from files saved by save_attention_data().
    
    Args:
        load_dir: Directory containing saved attention data files
        
    Returns:
        AttentionData: Reconstructed attention data object
        
    Example:
        >>> from lib import load_attention_data_from_file
        >>> 
        >>> # Load from disk (much faster than re-running predictions)
        >>> data = load_attention_data_from_file("saved_attention/my_experiment")
        >>> 
        >>> # Access data as usual
        >>> print(f"Architecture: {data.architecture_type}")
        >>> print(f"Phi keys: {list(data.phi_tensors.keys())}")
    """
    if not exists(load_dir) or not isdir(load_dir):
        raise FileNotFoundError(f"Directory not found: {load_dir}")
    
    # Load metadata
    metadata_path = join(load_dir, "metadata.json")
    if not exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {load_dir}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load config
    config = None
    config_path = join(load_dir, "config.yaml")
    if exists(config_path):
        config = OmegaConf.load(config_path)
    
    # Initialize result
    result = AttentionData(
        config=config,
        architecture_type=metadata["architecture_type"],
        checkpoint_paths=metadata.get("checkpoint_paths", []),
    )
    
    # Initialize empty lists for each key
    num_folds = metadata.get("num_folds", 0)
    for key in metadata.get("attention_keys", []):
        result.attention_weights[key] = [None] * num_folds
    for key in metadata.get("phi_keys", []):
        result.phi_tensors[key] = [None] * num_folds
    
    # Load attention weights
    att_path = join(load_dir, "attention_weights.npz")
    if exists(att_path):
        with np.load(att_path) as npz:
            for full_key in npz.files:
                # Parse key format: "{key}_{fold_idx}"
                parts = full_key.rsplit("_", 1)
                if len(parts) == 2:
                    key, fold_idx_str = parts
                    fold_idx = int(fold_idx_str)
                    if key in result.attention_weights:
                        result.attention_weights[key][fold_idx] = npz[full_key]
        print(f"Loaded attention weights from {att_path}")
    
    # Load phi tensors
    phi_path = join(load_dir, "phi_tensors.npz")
    if exists(phi_path):
        with np.load(phi_path) as npz:
            for full_key in npz.files:
                # Parse key format: "{key}_{fold_idx}"
                parts = full_key.rsplit("_", 1)
                if len(parts) == 2:
                    key, fold_idx_str = parts
                    fold_idx = int(fold_idx_str)
                    if key in result.phi_tensors:
                        result.phi_tensors[key][fold_idx] = npz[full_key]
        print(f"Loaded phi tensors from {phi_path}")
    
    # Load predictions if available
    predictions_path = join(load_dir, "predictions.pkl")
    if exists(predictions_path):
        with open(predictions_path, "rb") as f:
            result.predictions = pickle.load(f)
        print(f"Loaded {len(result.predictions)} predictions from {predictions_path}")
    
    print(f"\n✓ AttentionData loaded from {load_dir}")
    print(f"  Architecture: {result.architecture_type}")
    print(f"  Folds: {num_folds}")
    
    return result
