"""
Gradient Jacobian Logger Callback

This callback computes and logs the Jacobian matrix (gradients) of the output tensor 
with respect to the input tensor (before embedding) during validation.
"""

import os
from typing import Any
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class GradientJacobianLogger(Callback):
    """
    Callback to compute and save Jacobian matrices during validation.
    
    Args:
        save_dir: Directory to save gradient files
        every_n_epochs: Compute gradients every N epochs (default: 5)
        enabled: Whether gradient logging is enabled (default: True)
    """
    
    def __init__(
        self,
        save_dir: str,
        every_n_epochs: int = 5,
        enabled: bool = True,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.every_n_epochs = every_n_epochs
        self.enabled = enabled
        
        # Create gradients subdirectory
        self.gradients_dir = os.path.join(save_dir, "gradients")
        if self.enabled:
            os.makedirs(self.gradients_dir, exist_ok=True)
        
        # Storage for batch gradients
        self.current_epoch_gradients = []
        
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset storage at the start of each validation epoch."""
        if not self.enabled:
            return
            
        self.current_epoch_gradients = []
        
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute Jacobian for validation batches at specified intervals."""
        
        if not self.enabled:
            return
        
        current_epoch = trainer.current_epoch
        
        # Only compute gradients every N epochs
        if current_epoch % self.every_n_epochs != 0:
            return
        
        # Get batch data
        X, Y = batch
        
        # Compute Jacobian
        try:
            jacobian = self._compute_jacobian(pl_module, X, Y)
            
            # Store Jacobian in its original shape
            self.current_epoch_gradients.append({
                'batch_idx': batch_idx,
                'jacobian': jacobian.detach().cpu().numpy()
            })
                
        except Exception as e:
            print(f"Warning: Failed to compute Jacobian for batch {batch_idx}: {str(e)}")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save accumulated gradients at the end of validation epoch."""
        
        if not self.enabled or len(self.current_epoch_gradients) == 0:
            return
        
        current_epoch = trainer.current_epoch
        
        # Only save every N epochs
        if current_epoch % self.every_n_epochs != 0:
            return
        
        # Save to disk
        save_path = os.path.join(
            self.gradients_dir, 
            f"jacobian_epoch_{current_epoch:04d}.npz"
        )
        
        # Save all Jacobians
        jacobians = {f"batch_{g['batch_idx']}": g['jacobian'] 
                    for g in self.current_epoch_gradients}
        np.savez_compressed(save_path, **jacobians)
        
        print(f"✓ Saved Jacobian matrices for epoch {current_epoch} to {save_path}")
        
        # Clear memory
        self.current_epoch_gradients = []
    
    def _compute_jacobian(
        self, 
        pl_module: pl.LightningModule, 
        X: torch.Tensor, 
        Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix using manual backward passes (Option B - optimized).
        Only computes diagonal elements: ∂output[b]/∂input[b] for each batch element.
        This is much faster than computing full cross-batch Jacobian.
        
        Args:
            pl_module: The Lightning module
            X: Input tensor (B, X_seq_len, features)
            Y: Target tensor (B, Y_seq_len, features)
            
        Returns:
            Jacobian tensor of shape (B, Y_seq_len, out_dim, X_seq_len, features)
        """
        
        # Temporarily enable gradients for validation
        with torch.enable_grad():
            # Get dimensions
            B = X.shape[0]
            X_seq_len = X.shape[1]
            features = X.shape[2]
            Y_seq_len = Y.shape[1]
            
            # Prepare decoder input (zero out target values)
            dec_input = Y.clone()
            dec_val_idx = pl_module.dec_val_idx
            dec_input[:, :, dec_val_idx] = 0.0
            
            # Enable gradients for input
            X_grad = X.detach().clone().requires_grad_(True)
            
            # Temporarily set model to train mode to enable gradient computation
            was_training = pl_module.training
            pl_module.train()
            
            # Forward pass
            forecast_output, _, _, _ = pl_module.model.forward(
                input_tensor=X_grad,
                target_tensor=dec_input,
                trg_pos_mask=None
            )
            
            # Restore model mode
            if not was_training:
                pl_module.eval()
            
            # Get output dimensions
            out_dim = forecast_output.shape[2]
            
            # Allocate result tensor: (B, Y_seq_len, out_dim, X_seq_len, features)
            jacobian = torch.zeros(B, Y_seq_len, out_dim, X_seq_len, features, 
                                  device=X.device, dtype=X.dtype)
            
            # Compute gradients for each output element
            for b in range(B):
                for t_out in range(Y_seq_len):
                    for d_out in range(out_dim):
                        # Backward pass for this specific output element
                        if X_grad.grad is not None:
                            X_grad.grad.zero_()
                        
                        forecast_output[b, t_out, d_out].backward(retain_graph=True)
                        
                        # Store gradient for this batch element only
                        jacobian[b, t_out, d_out] = X_grad.grad[b].clone()
        
        return jacobian
