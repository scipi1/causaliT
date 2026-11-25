"""
Prediction entry points: checkpoint -> ``PredictionResult`` and input
conditioning helpers (blanking, interventions).
"""

from pathlib import Path
from typing import Callable

import torch
from pytorch_lightning import seed_everything

# Local imports
from causaliT.training.core.experiment_control import update_config
from causaliT.evaluation.predictors import (
    BasePredictor,
    PredictionResult,
    AttentionSelectorPredictor,
)


def create_predictor(
    config: dict,
    checkpoint_path: Path,
    datadir_path: Path = None
) -> BasePredictor:
    """
    Factory function that returns the appropriate predictor based on config.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        datadir_path: Path to data directory (optional)
        
    Returns:
        BasePredictor: Instance of appropriate predictor class
        
    Raises:
        ValueError: If model type is not recognized
    """
    model_obj = config["model"]["model_object"]
    
    # Registry maps model types to predictor classes
    PREDICTOR_REGISTRY = {
        "AttentionSelectorLayer": AttentionSelectorPredictor,
    }
    
    predictor_class = PREDICTOR_REGISTRY.get(model_obj)
    if predictor_class is None:
        available = list(PREDICTOR_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_obj}. Available: {available}")
    
    return predictor_class(config, checkpoint_path, datadir_path)


def create_input_blanking_fn(
    beta: float,
    val_idx: int,
    seed: int = 42
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an input blanking function using Bernoulli sampling.
    
    This function generates a conditioning function that randomly blanks tokens
    in the input sequence using Bernoulli(beta) sampling. Each token position
    is independently sampled, and if sampled as 1, the value feature at that
    position is set to NaN.
    
    Args:
        beta: Probability of blanking each token (0.0 to 1.0)
        val_idx: Index of the value feature to blank
        seed: Random seed for reproducibility
        
    Returns:
        Conditioning function with signature: fn(X: torch.Tensor) -> torch.Tensor
        
    Example:
        >>> blanking_fn = create_input_blanking_fn(beta=0.2, val_idx=0, seed=42)
        >>> X_blanked = blanking_fn(X)  # 20% of tokens will have val_idx set to NaN
    """
    generator = torch.Generator().manual_seed(seed)
    
    def blanking_fn(X: torch.Tensor) -> torch.Tensor:
        """
        Blank tokens in input tensor using Bernoulli(beta) sampling.
        
        Args:
            X: Input tensor (B x L x D)
            
        Returns:
            Blanked tensor with same shape
        """
        B, L, D = X.shape
        X_blanked = X.clone()
        
        # Sample Bernoulli(beta) for each position in the batch
        # Shape: B x L
        blank_mask = torch.bernoulli(
            torch.full((B, L), beta),
            generator=generator
        ).bool()
        
        # Blank the value feature where mask is True
        X_blanked[blank_mask, val_idx] = float('nan')
        
        return X_blanked
    
    return blanking_fn


def create_intervention_fn(
    interventions: dict,
    val_idx: int = 0,
    var_idx: int = 1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create an intervention function that replaces values for specific variables.
    
    This function implements do-calculus interventions by setting the value feature
    of tokens with specific variable IDs to a fixed intervention value. This allows
    studying causal effects by intervening on source variables S.
    
    Args:
        interventions: Dictionary mapping variable IDs to intervention values.
                      Example: {1: 0.5} sets all S1 tokens to value 0.5
                               {1: 0.5, 2: 0.3} sets S1=0.5 and S2=0.3
        val_idx: Index of the value feature in the token (default: 0)
        var_idx: Index of the variable ID feature in the token (default: 1)
        
    Returns:
        Conditioning function with signature: fn(S: torch.Tensor) -> torch.Tensor
        
    Example:
        >>> # Intervene on S1 (variable_id=1), setting all S1 values to 0.5
        >>> intervention_fn = create_intervention_fn(interventions={1: 0.5})
        >>> S_intervened = intervention_fn(S)
        
        >>> # Multiple interventions: S1=0.5 and S2=0.3
        >>> intervention_fn = create_intervention_fn(interventions={1: 0.5, 2: 0.3})
        >>> S_intervened = intervention_fn(S)
        
        >>> # Use with predict_test_from_ckpt
        >>> results = predict_test_from_ckpt(
        ...     config, datadir_path, checkpoint_path,
        ...     input_conditioning_fn=intervention_fn
        ... )
    
    Notes:
        - For StageCausaliT, S is the source tensor with shape (B x L_s x F)
        - Each token typically has features [value, variable_id, ...]
        - The intervention replaces value at val_idx for tokens where
          variable_id at var_idx matches a key in the interventions dict
    """
    
    def intervention_fn(S: torch.Tensor) -> torch.Tensor:
        """
        Apply interventions to source tensor.
        
        Args:
            S: Source tensor (B x L x F) where each token has [value, var_id, ...]
            
        Returns:
            Intervened tensor with same shape
        """
        S_intervened = S.clone()
        
        # Get variable IDs for all tokens (B x L)
        variable_ids = S[:, :, var_idx]
        
        # Apply each intervention
        for target_var_id, intervention_value in interventions.items():
            # Create mask for tokens with this variable ID
            # Handle both integer and float variable IDs
            mask = (variable_ids == target_var_id)
            
            # Set value feature to intervention value where mask is True
            # Use torch.where() for explicit 3D indexing (avoids ambiguous boolean mask indexing across PyTorch versions)
            batch_indices, seq_indices = torch.where(mask)
            S_intervened[batch_indices, seq_indices, val_idx] = intervention_value
        
        return S_intervened
    
    return intervention_fn


def predict_test_from_ckpt(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    external_dataset: dict = None,
    dataset_label: str = "test",
    cluster: bool = False,
    input_conditioning_fn: Callable[[torch.Tensor], torch.Tensor] = None,
    disable_hard_masks: bool = False,
) -> PredictionResult:
    """
    Run standard prediction on specified dataset using a trained model checkpoint.
    
    This function:
    1. Creates appropriate predictor based on model type in config
    2. Loads the model from checkpoint (including hard masks if model was trained with them)
    3. Creates data module
    4. Runs prediction
    5. Returns PredictionResult object
    
    Args:
        config: Configuration dictionary
        datadir_path: Path to data directory
        checkpoint_path: Path to model checkpoint
        external_dataset: Optional dict with 'dataset', 'filename_input', 'filename_target'
        dataset_label: One of ["train", "test", "all"]
        cluster: Whether running on cluster
        input_conditioning_fn: Optional function to condition inputs before forward pass
        disable_hard_masks: If True, disables hard masks even if model was trained with them.
                           Useful for ablation studies during inference. Default False = use
                           masks if model was trained with them.
        
    Returns:
        PredictionResult: Object containing:
            - inputs: input data array
            - outputs: model predictions
            - targets: target data
            - attention_weights: dict with 'encoder', 'decoder', 'cross' keys (None for baseline)
            - metadata: dict with model info, dataset info, etc.
    """
    assert dataset_label in ["train", "test", "all"], \
        f"{dataset_label} is not a proper label!"
    
    # Set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
        
    # Update config
    config_updated = update_config(config)
    
    # Create predictor (auto-selects correct type from config)
    predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
    
    # Setup data module
    dm = predictor.create_data_module(
        external_dataset=external_dataset,
        cluster=cluster
    )
    
    # Run prediction - pass disable_hard_masks to disable masks for ablation studies
    results = predictor.predict(
        dm=dm,
        dataset_label=dataset_label,
        input_conditioning_fn=input_conditioning_fn,
        disable_hard_masks=disable_hard_masks,
    )
    
    return results
