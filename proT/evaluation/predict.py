import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import dirname, abspath, join
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from typing import Tuple, Callable
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import yaml

# Local imports
from proT.training.dataloader import ProcessDataModule
from proT.training.experiment_control import update_config
from proT.evaluation.predictors import (
    BasePredictor,
    PredictionResult,
    TransformerPredictor,
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
        "proT": TransformerPredictor,
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


def predict_test_from_ckpt(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    external_dataset: dict = None,
    dataset_label: str = "test",
    cluster: bool = False,
    input_conditioning_fn: Callable[[torch.Tensor], torch.Tensor] = None
) -> PredictionResult:
    """
    Run standard prediction on specified dataset using a trained model checkpoint.
    
    This function:
    1. Creates appropriate predictor based on model type in config
    2. Loads the model from checkpoint
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
    
    # Run prediction
    results = predictor.predict(
        dm=dm,
        dataset_label=dataset_label,
        input_conditioning_fn=input_conditioning_fn
    )
    
    return results


def predict_test_from_ckpt_adaptive(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    show_trg_max_idx: int,
    external_dataset: dict = None,
    dataset_label: str = "test",
    cluster: bool = False
) -> PredictionResult:
    """
    Run adaptive prediction with target revelation for transformer models.
    
    This function is specifically for models that support curriculum learning
    (OnlineTargetForecaster, TransformerForecaster with show_trg configuration).
    
    Args:
        config: Configuration dictionary
        datadir_path: Path to data directory
        checkpoint_path: Path to model checkpoint
        show_trg_max_idx: Max index to reveal in target sequence
        external_dataset: Optional dict with 'dataset', 'filename_input', 'filename_target'
        dataset_label: One of ["train", "test", "all"]
        cluster: Whether running on cluster
        
    Returns:
        PredictionResult: Object containing predictions and attention weights
        
    Raises:
        ValueError: If model doesn't support adaptive prediction
    """
    assert dataset_label in ["train", "test", "all"], \
        f"{dataset_label} is not a proper label!"
    
    # Set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
        
    # Update config
    config_updated = update_config(config)
    
    # Create predictor (must be TransformerPredictor)
    predictor = create_predictor(config_updated, checkpoint_path, datadir_path)
    
    # Verify it's a transformer predictor
    if not isinstance(predictor, TransformerPredictor):
        raise ValueError(
            f"Adaptive prediction with show_trg_max_idx is only supported for "
            f"transformer models. Got {type(predictor).__name__}"
        )
    
    # Setup data module
    dm = predictor.create_data_module(
        external_dataset=external_dataset,
        cluster=cluster
    )
    
    # Run adaptive prediction
    results = predictor.predict(
        dm=dm,
        dataset_label=dataset_label,
        show_trg_max_idx=show_trg_max_idx
    )
    
    return results


def _plot_attention_heatmap(
    cross_att: np.ndarray,
    input_miss: np.ndarray,
    save_path: str,
    title: str
):
    """
    Helper function to create a single attention heatmap plot with input mask.
    
    Args:
        cross_att: Cross-attention matrix (N x M) where N=target length, M=source length
        input_miss: Boolean mask for missing inputs (1 x M)
        save_path: Path to save the figure
        title: Title for the plot
    """
    N, M = cross_att.shape
    
    # Create figure and grid
    fig = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.5)

    # Main heatmap axis
    ax0 = fig.add_subplot(gs[0])
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    im0 = ax0.imshow(cross_att, cmap='viridis', aspect='auto', origin='upper')
    fig.colorbar(im0, cax=cax0, label='Value')
    ax0.set_xticks([])
    ax0.set_ylabel("Rows")
    ax0.set_title(title)

    # Boolean mask axis
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    im1 = ax1.imshow(input_miss, cmap='Greys', aspect='auto', origin='upper', vmin=0, vmax=1)
    fig.colorbar(im1, cax=cax1, ticks=[0, 1], label='Missing')
    cax1.set_yticklabels(['False', 'True'])

    ax1.set_yticks([])
    ax1.set_xlabel("Columns")
    num_labels = min(M, 10)
    step = M // num_labels if M > 10 else 1

    ax1.set_xticks(np.arange(0, M, step))
    ax1.set_xticklabels(np.arange(0, M, step))
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention heatmap to {save_path}")


def mk_quick_pred_plot(
    config: dict,
    checkpoint_path: Path,
    datadir_path: Path,
    val_idx: int, 
    save_dir: Path
):
    """
    Create quick prediction plots for visualization.
    
    Works with both transformer models (proT) and baseline models (LSTM, GRU, TCN, MLP, S6).
    Uses the predictor infrastructure to automatically handle different model types.
    Supports multi-head attention by creating separate plots for each head plus a summed plot.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        datadir_path: Path to data directory
        val_idx: Index of value feature to plot
        save_dir: Directory to save plots
    """
    # Use the predictor infrastructure which handles all model types
    results = predict_test_from_ckpt(
        config=config,
        datadir_path=datadir_path,
        checkpoint_path=checkpoint_path,
        dataset_label="test",
        cluster=False
    )
    
    # Extract data from PredictionResult
    input_array = results.inputs
    output_array = results.outputs
    target_array = results.targets
    attention_weights = results.attention_weights
    model_type = results.metadata.get('model_type', 'unknown')
    
    # Debug: Print original shapes
    print(f"[DEBUG] Original shapes - output: {output_array.shape}, target: {target_array.shape}, input: {input_array.shape}")
    
    # Normalize array dimensions to ensure consistent shape handling
    # Expected shapes: output (B, L), target (B, L, F), input (B, L, F)
    
    # Handle output array
    if len(output_array.shape) == 1:
        output_array = output_array[np.newaxis, :]  # (L,) -> (1, L)
    elif len(output_array.shape) == 0:
        raise ValueError(f"Output array has invalid shape: {output_array.shape}")
    
    # Handle target array - this is the critical part
    if len(target_array.shape) == 1:
        # (L,) -> need to add batch and feature dims
        target_array = target_array[np.newaxis, :, np.newaxis]
    elif len(target_array.shape) == 2:
        # Could be (B, L) or (L, F) - need to determine which
        # If second dim matches output length, it's likely (L, F)
        if target_array.shape[0] == output_array.shape[1]:
            # (L, F) -> (1, L, F)
            target_array = target_array[np.newaxis, :, :]
        else:
            # (B, L) -> (B, L, 1)
            target_array = target_array[:, :, np.newaxis]
    elif len(target_array.shape) == 3:
        # Already correct shape (B, L, F)
        pass
    else:
        raise ValueError(f"Target array has unexpected shape: {target_array.shape}")
    
    # Handle input array
    if len(input_array.shape) == 1:
        input_array = input_array[np.newaxis, :, np.newaxis]
    elif len(input_array.shape) == 2:
        if input_array.shape[0] == output_array.shape[1]:
            input_array = input_array[np.newaxis, :, :]
        else:
            input_array = input_array[:, :, np.newaxis]
    
    # Debug: Print normalized shapes
    print(f"[DEBUG] Normalized shapes - output: {output_array.shape}, target: {target_array.shape}, input: {input_array.shape}")
    
    # Extract first sample for plotting
    y_out = output_array[0, :]  # Shape: (L,)
    
    # Safely extract target values for the specified feature
    if target_array.shape[2] <= val_idx:
        raise ValueError(f"val_idx={val_idx} is out of bounds for target array with {target_array.shape[2]} features")
    
    y_trg = target_array[0, :, val_idx]  # Shape: (L,)
    
    print(f"[DEBUG] Extracted y_out shape: {y_out.shape}, y_trg shape: {y_trg.shape}")
    print(f"[DEBUG] y_out range: [{y_out.min():.2f}, {y_out.max():.2f}], y_trg range: [{y_trg.min():.2f}, {y_trg.max():.2f}]")
    
    x = np.arange(len(y_out))

    # Plot 1: Prediction vs Target (always created)
    fig, ax = plt.subplots()
    ax.plot(x, y_out, label='Prediction')
    ax.plot(x, y_trg, label='Target')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Predictions vs Targets')
    ax.legend()
    fig.savefig(join(save_dir, "quick_pred_plot.png"))
    plt.close(fig)
    print(f"Saved prediction plot to {join(save_dir, 'quick_pred_plot.png')}")
    
    # Plot 2: Attention Heatmap (only for transformer models)
    if attention_weights is not None and 'cross' in attention_weights:
        cross_att_array = attention_weights['cross']
        
        # Debug: Print original attention shape
        print(f"[DEBUG] Original cross-attention shape: {cross_att_array.shape}")
        
        # Handle dimensions - ensure we have at least 3D (B x N x M)
        if len(cross_att_array.shape) == 2:
            # (N, M) -> (1, N, M)
            cross_att_array = np.expand_dims(cross_att_array, axis=0)
            print(f"[DEBUG] Expanded to shape: {cross_att_array.shape}")
        
        # Prepare input mask (same for all attention plots)
        input_miss_bool = np.isnan(input_array[0, :, val_idx].squeeze())
        input_miss = input_miss_bool[np.newaxis, :].astype(int)
        
        # Check for multi-head attention (4D: B x H x N x M)
        if len(cross_att_array.shape) == 4:
            num_heads = cross_att_array.shape[1]
            print(f"[DEBUG] Multi-head attention detected: {num_heads} heads")
            
            # Plot 1: Summed attention across all heads
            cross_att_sum = cross_att_array[0].sum(axis=0)  # Sum over head dimension
            print(f"[DEBUG] Summed attention shape: {cross_att_sum.shape}")
            _plot_attention_heatmap(
                cross_att_sum,
                input_miss,
                join(save_dir, "cross_att_sum.png"),
                "Cross-Attention (Summed Across Heads)"
            )
            
            # Plot 2-N: Individual head plots
            for head_idx in range(num_heads):
                cross_att_head = cross_att_array[0, head_idx, :, :]
                print(f"[DEBUG] Head {head_idx} attention shape: {cross_att_head.shape}")
                _plot_attention_heatmap(
                    cross_att_head,
                    input_miss,
                    join(save_dir, f"cross_att_head_{head_idx}.png"),
                    f"Cross-Attention Head {head_idx}"
                )
        else:
            # Single head or pre-aggregated (3D: B x N x M)
            print("[DEBUG] Single-head attention detected (or pre-aggregated)")
            cross_att = cross_att_array[0]
            print(f"[DEBUG] Attention shape for plotting: {cross_att.shape}")
            _plot_attention_heatmap(
                cross_att,
                input_miss,
                join(save_dir, "cross_att.png"),
                "Cross-Attention Heatmap"
            )
    else:
        print(f"Baseline model ({model_type}): skipping attention heatmap (not applicable)")


# Legacy function for backward compatibility
def predict(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    debug_flag: bool = False,
    show_trg_max_idx: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY FUNCTION: Makes prediction using the old API.
    
    This function is kept for backward compatibility with existing code.
    New code should use predict_test_from_ckpt() which returns a PredictionResult.
    
    Args:
        model: transformer model (assumed to be TransformerForecaster)
        dm: data module
        dataset_label: choose between ["test", "train", "all"]
        debug_flag: Predicts one batch and stops. Defaults to False.
        show_trg_max_idx: Max index to show target

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            input_array, output_array, target_array, cross_att_array, 
            enc_self_att_array, dec_self_att_array
    """
    assert dataset_label in ["train", "test", "all"], \
        AssertionError("Invalid dataset label!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set model for prediction
    forecaster = model.to(device)
    forecaster.eval()
    
    input_list, output_list, target_list = [], [], []
    cross_att_list, enc_self_att_list, dec_self_att_list = [], [], []
    
    # prepare data module
    dm.prepare_data()
    dm.setup(stage=None)
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
    elif dataset_label == "test":
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
    elif dataset_label == "all":
        dataset = dm.all_dataloader()
        print("All data selected (default).")
        
    # loop over prediction batches
    print("Predicting...")
    for batch in tqdm(dataset):
        if isinstance(batch, (list, tuple)):
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)

        X, trg = batch
        
        with torch.no_grad():
            forecast_output, _, (enc_self_att, dec_self_att, dec_cross_att), _, _ = forecaster.forward(
                data_input=X,
                data_trg=trg,
                show_trg_max_idx=show_trg_max_idx,
            )
            
        # append batch predictions
        input_list.append(X)
        output_list.append(forecast_output)
        target_list.append(trg)
        cross_att_list.append(dec_cross_att[0])
        enc_self_att_list.append(enc_self_att[0])
        dec_self_att_list.append(dec_self_att[0])

        if debug_flag:
            print("Debug mode: stopping after one batch...")
            break
    
    # detach predictions
    input_tensor = torch.cat([t.cpu().detach() for t in input_list], dim=0)
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_tensor = torch.cat([t.cpu().detach() for t in cross_att_list], dim=0)
    enc_self_att_tensor = torch.cat([t.cpu().detach() for t in enc_self_att_list], dim=0)
    dec_self_att_tensor = torch.cat([t.cpu().detach() for t in dec_self_att_list], dim=0)
    
    # convert predictions to numpy
    input_array = input_tensor.numpy().squeeze()
    output_array = output_tensor.numpy().squeeze()
    target_array = target_tensor.numpy().squeeze()
    cross_att_array = cross_att_tensor.numpy().squeeze()
    enc_self_att_array = enc_self_att_tensor.numpy().squeeze()
    dec_self_att_array = dec_self_att_tensor.numpy().squeeze()
    
    return input_array, output_array, target_array, cross_att_array, enc_self_att_array, dec_self_att_array


if __name__ == "__main__":
    
    from os.path import dirname, abspath, join
    
    ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    print(ROOT_DIR)
    datadir_path = join(ROOT_DIR, r"data/")
    config_path = join(ROOT_DIR, r"experiments/example/config_v5_5.yaml")
    checkpoint_path = join(ROOT_DIR, r"experiments/example/k_0/checkpoints/best_checkpoint.ckpt")

    config = OmegaConf.load(config_path)
    
    # Use new API
    results = predict_test_from_ckpt(
        config, 
        datadir_path, 
        checkpoint_path, 
        dataset_label="all",
        cluster=False
    )
    
    # Access results
    print(f"Input shape: {results.inputs.shape}")
    print(f"Output shape: {results.outputs.shape}")
    print(f"Target shape: {results.targets.shape}")
    print(f"Has attention: {results.attention_weights is not None}")
    print(f"Metadata: {results.metadata}")
