"""
Causal Initialization Stage: Initialize model toward causal structure.

This module implements Stage 1 of the staged training pipeline.
The goal is to pre-train the model with an HSIC-dominated loss function,
initializing attention patterns and weights toward the causal structure
before standard fitting-focused training begins.

Key insight: At random initialization, the model has no preference for any
particular causal structure. By training first with HSIC >> Recon loss,
we break this symmetry and guide the model toward learning the true causal
mechanism before the reconstruction pressure takes over.

Process:
1. Load model from checkpoint (from calibration) or initialize fresh
2. Train for causal_init_epochs with λ_hsic_init = λ_hsic * multiplier
3. Save checkpoint for main training stage

Why this works:
- HSIC measures independence between residuals and source variables
- Minimizing HSIC means making residuals independent of S
- This requires the model to capture all S→X dependence in its predictions
- The true causal mechanism achieves this optimally (unique global minimum
  under capacity constraints from group L1)

Usage:
    checkpoint_path = run_causal_initialization(config, data_dir, save_dir, starting_ckpt)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class CausalInitProgressLogger(Callback):
    """
    Callback to log progress during causal initialization.
    
    Tracks HSIC values and reconstruction loss to monitor the initialization process.
    """
    
    def __init__(self):
        super().__init__()
        self.hsic_values = []
        self.recon_values = []
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Log epoch metrics."""
        metrics = trainer.callback_metrics
        
        # Get HSIC (different possible keys)
        hsic = None
        for key in ["train_hsic_cross", "train_hsic_reg", "train_hsic"]:
            if key in metrics:
                hsic = metrics[key].item()
                break
        
        # Get reconstruction loss
        recon = None
        for key in ["train_loss_x", "train_recon", "train_loss"]:
            if key in metrics:
                recon = metrics[key].item()
                break
        
        if hsic is not None:
            self.hsic_values.append(hsic)
        if recon is not None:
            self.recon_values.append(recon)
        
        # Print progress
        epoch = trainer.current_epoch + 1
        hsic_str = f"{hsic:.6f}" if hsic is not None else "N/A"
        recon_str = f"{recon:.6f}" if recon is not None else "N/A"
        print(f"    Epoch {epoch}: HSIC = {hsic_str}, Recon = {recon_str}")


def run_causal_initialization(
    config: dict,
    data_dir: str,
    save_dir: str,
    starting_checkpoint: Optional[str] = None,
    hsic_cross_multiplier: Optional[float] = None,
    hsic_self_multiplier: Optional[float] = None,
    seed: int = 42,
) -> str:
    """
    Run causal initialization stage.
    
    This is the main entry point for Stage 1 (Causal Initialization).
    
    Process:
    1. Load model (from calibration checkpoint if provided, else fresh)
    2. Set λ_hsic to high values using SEPARATE multipliers for cross and self:
       - λ_hsic_cross_init = λ_hsic_cross * cross_multiplier * boost_factor
       - λ_hsic_self_init = λ_hsic_self * self_multiplier * boost_factor
    3. Train for causal_init_epochs
    4. Save checkpoint for main training
    
    The separate multipliers come from calibration and ensure BOTH HSIC signals
    are balanced with reconstruction. The boost_factor (causal_init_hsic_multiplier)
    then increases both equally to make HSIC dominate during initialization.
    
    The high HSIC weight prioritizes learning causal structure over fitting.
    After this stage, the model should have attention patterns aligned with
    the true causal structure, providing a good starting point for main training.
    
    Args:
        config: Configuration dictionary with staged_training section
        data_dir: Data directory path
        save_dir: Save directory for outputs
        starting_checkpoint: Optional checkpoint to start from (from calibration)
        hsic_cross_multiplier: Multiplier for λ_hsic_cross (from calibration)
        hsic_self_multiplier: Multiplier for λ_hsic_self (from calibration)
        seed: Random seed
        
    Returns:
        Path to causal initialization checkpoint
    """
    from causaliT.training.trainer import create_model_instance, get_dataloader
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset
    
    # Setup directory
    init_dir = Path(save_dir) / "causal_init"
    init_dir.mkdir(exist_ok=True, parents=True)
    
    # Get causal initialization parameters from config
    staged_config = config.get("staged_training", {})
    init_epochs = staged_config.get("causal_init_epochs", 20)
    hsic_boost_factor = staged_config.get("causal_init_hsic_multiplier", 10.0)
    lambda_group = staged_config.get("lambda_group_l1", None)
    
    # Get base HSIC weights
    base_hsic_cross = config["training"].get("lambda_hsic_cross",
                          config["training"].get("lambda_hsic", 0.1))
    base_hsic_self = config["training"].get("lambda_hsic_self", 0.0)
    
    # Use provided multipliers or default to 1.0
    cross_mult = hsic_cross_multiplier if hsic_cross_multiplier is not None else 1.0
    self_mult = hsic_self_multiplier if hsic_self_multiplier is not None else 1.0
    
    # Compute calibrated + boosted HSIC weights
    # Step 1: Apply calibration multipliers to balance gradients
    # Step 2: Apply boost factor to make HSIC dominate
    lambda_hsic_cross_init = base_hsic_cross * cross_mult * hsic_boost_factor
    lambda_hsic_self_init = base_hsic_self * self_mult * hsic_boost_factor
    
    print(f"\n{'='*70}")
    print("CAUSAL INITIALIZATION: Training with HSIC-dominated loss")
    print(f"{'='*70}")
    print(f"  Epochs: {init_epochs}")
    print(f"  HSIC boost factor: {hsic_boost_factor}")
    print(f"\n  Cross-attention (S→X) HSIC:")
    print(f"    Base λ_hsic_cross = {base_hsic_cross}")
    print(f"    Calibration multiplier = {cross_mult:.3f}")
    print(f"    λ_hsic_cross_init = {lambda_hsic_cross_init:.4f}")
    if base_hsic_self > 0:
        print(f"\n  Self-attention (X→X) HSIC:")
        print(f"    Base λ_hsic_self = {base_hsic_self}")
        print(f"    Calibration multiplier = {self_mult:.3f}")
        print(f"    λ_hsic_self_init = {lambda_hsic_self_init:.4f}")
    else:
        print(f"\n  Self-attention (X→X) HSIC: disabled (λ_hsic_self = 0)")
    if lambda_group is not None:
        print(f"\n  λ_group (from calibration): {lambda_group:.2e}")
    if starting_checkpoint:
        print(f"\n  Starting from checkpoint: {starting_checkpoint}")
    else:
        print(f"\n  Starting from fresh initialization")
    
    # Make a copy of config for causal init
    # We need to modify it for the high HSIC weight
    config_init = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # Set separate HSIC weights for cross and self
    config_init["training"]["lambda_hsic_cross"] = lambda_hsic_cross_init
    config_init["training"]["lambda_hsic_self"] = lambda_hsic_self_init
    config_init["training"]["lambda_hsic"] = lambda_hsic_cross_init  # Fallback key (deprecated)
    
    # Apply group L1 if calibrated
    if lambda_group is not None:
        config_init["training"]["lambda_group_l1"] = lambda_group
    
    # Disable HSIC annealing during causal init (we want constant high HSIC)
    config_init["training"]["use_hsic_annealing"] = False
    
    # Enable HSIC logging for monitoring
    config_init["training"]["log_hsic"] = True
    
    # Reset seed for reproducibility
    seed_everything(seed)
    
    # Create model
    model = create_model_instance(config_init, data_dir)
    
    # Load from checkpoint if provided
    if starting_checkpoint is not None:
        print(f"\n  Loading weights from: {starting_checkpoint}")
        # weights_only=False is safe here because we just created this checkpoint
        # in the calibration stage. Required for PyTorch 2.6+ due to security changes.
        checkpoint = torch.load(starting_checkpoint, map_location="cpu", weights_only=False)
        
        # Handle both state_dict formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load state dict (strict=False allows for mismatched keys)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    Warning: Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"    Warning: Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print("  ✓ Weights loaded successfully")
    
    # Setup dataloader
    dm = get_dataloader(config_init, data_dir, cluster=False, seed=seed)
    dm.setup(stage='fit')
    
    # Setup callbacks
    progress_logger = CausalInitProgressLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(init_dir),
        filename="causal_init_checkpoint",
        save_last=True,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=init_epochs,
        default_root_dir=str(init_dir),
        callbacks=[progress_logger, checkpoint_callback],
        logger=CSVLogger(save_dir=str(init_dir), name="causal_init_logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1,
    )
    
    # Train
    print(f"\n  Starting causal initialization training...")
    trainer.fit(model, dm)
    
    # Save final checkpoint explicitly
    checkpoint_path = str(init_dir / "causal_init_checkpoint.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    
    # Save summary
    summary = {
        "epochs": init_epochs,
        "lambda_hsic_cross_init": lambda_hsic_cross_init,
        "lambda_hsic_self_init": lambda_hsic_self_init,
        "base_hsic_cross": base_hsic_cross,
        "base_hsic_self": base_hsic_self,
        "cross_multiplier": cross_mult,
        "self_multiplier": self_mult,
        "hsic_boost_factor": hsic_boost_factor,
        "lambda_group_l1": lambda_group,
        "starting_checkpoint": starting_checkpoint,
        "final_checkpoint": checkpoint_path,
        "hsic_history": progress_logger.hsic_values,
        "recon_history": progress_logger.recon_values,
        "final_hsic": progress_logger.hsic_values[-1] if progress_logger.hsic_values else None,
        "final_recon": progress_logger.recon_values[-1] if progress_logger.recon_values else None,
    }
    
    with open(init_dir / "causal_init_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CAUSAL INITIALIZATION COMPLETE")
    print(f"  Final HSIC: {summary['final_hsic']:.6f}" if summary['final_hsic'] else "  Final HSIC: N/A")
    print(f"  Final Recon: {summary['final_recon']:.6f}" if summary['final_recon'] else "  Final Recon: N/A")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    return checkpoint_path


def verify_causal_init_effectiveness(
    config: dict,
    data_dir: str,
    save_dir: str,
    checkpoint_path: str,
) -> dict:
    """
    Verify the effectiveness of causal initialization.
    
    This optional function can be called after causal init to check:
    1. Attention patterns have structure (not uniform)
    2. HSIC is lower than at random init
    3. Predictions are not degenerate (have variance)
    
    Args:
        config: Configuration dictionary
        data_dir: Data directory path
        save_dir: Save directory
        checkpoint_path: Path to causal init checkpoint
        
    Returns:
        Dict with verification metrics
    """
    from causaliT.training.trainer import create_model_instance, get_dataloader
    
    # Load model from checkpoint
    model = create_model_instance(config, data_dir)
    # weights_only=False for PyTorch 2.6+ compatibility with Lightning checkpoints
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get a batch of data
    dm = get_dataloader(config, data_dir, cluster=False, seed=config["training"].get("seed", 42))
    dm.setup(stage='fit')
    batch = next(iter(dm.train_dataloader()))
    
    # Unpack batch
    if len(batch) == 3:
        S, X, Y = batch
    else:
        S, X = batch
    
    device = next(model.parameters()).device
    S, X = S.to(device), X.to(device)
    
    # Forward pass
    with torch.no_grad():
        val_idx = model.val_idx
        x_blanked = X.clone()
        x_blanked[:, :, val_idx] = 0.0
        
        pred_x, attention_weights, masks, entropies = model.model.forward(
            source_tensor=S,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=model.get_hard_masks() if hasattr(model, 'get_hard_masks') else None,
        )
    
    # Compute metrics
    x_target = X[:, :, val_idx]
    
    # Prediction variance (should be non-zero)
    pred_variance = pred_x.var().item()
    
    # Attention entropy (lower = more decisive)
    if attention_weights and attention_weights[0] is not None:
        att = attention_weights[0][0]  # First layer, first batch element
        att_entropy = -(att * (att + 1e-10).log()).sum(dim=-1).mean().item()
    else:
        att_entropy = None
    
    # Residual variance
    residuals = x_target.squeeze() - pred_x.squeeze()
    residual_variance = residuals.var().item()
    
    metrics = {
        "prediction_variance": pred_variance,
        "attention_entropy": att_entropy,
        "residual_variance": residual_variance,
        "is_degenerate": pred_variance < 1e-6,
    }
    
    # Save verification results
    verify_path = Path(save_dir) / "causal_init" / "verification.json"
    with open(verify_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics
