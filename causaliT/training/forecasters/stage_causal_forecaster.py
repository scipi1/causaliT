"""
StageCausalForecaster: PyTorch Lightning wrapper for StageCausaliT model.

This forecaster handles training, validation, and testing for the dual-decoder
architecture with proper teacher forcing and dual loss computation.
"""

import sys
from os.path import dirname, abspath
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.stage_causal import StageCausaliT


class StageCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for StageCausaliT transformer model.
    
    This forecaster manages the dual-stage training process:
    - Stage 1: S → X reconstruction
    - Stage 2: X → Y prediction
    
    Features:
    - Dual loss computation (MSE for X and Y)
    - Teacher forcing control (training vs inference)
    - Separate metrics tracking for X and Y
    - Entropy and acyclicity regularization support
    
    Args:
        config: Configuration dictionary containing model, training, and data settings
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model = StageCausaliT(**config["model"]["kwargs"])
        
        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # Teacher forcing configuration
        self.teacher_forcing = config["training"].get("teacher_forcing", False)
        
        # Loss weighting (default equal contribution)
        self.loss_weight_x = config["training"].get("loss_weight_x", 1.0)
        self.loss_weight_y = config["training"].get("loss_weight_y", 1.0)
        
        # Logging configuration
        self.log_entropy = config["training"].get("log_entropy", False)
        self.log_acyclicity = config["training"].get("log_acyclicity", False)
        
        # Regularizers
        self.gamma = config["training"].get("gamma", 0)   # Entropy regularization
        self.kappa = config["training"].get("kappa", 0)   # Acyclicity regularization
        
        self.save_hyperparameters(config)
        
        # Metrics for X reconstruction
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()
        
        # Metrics for Y prediction
        self.mae_y = tm.MeanAbsoluteError()
        self.rmse_y = tm.MeanSquaredError(squared=False)
        self.r2_y = tm.R2Score()
        
        # Optionally freeze embeddings
        if config["training"].get("freeze_embeddings", False):
            self.freeze_embeddings()
    
    def freeze_embeddings(self):
        """
        Freeze the shared embedding layer to train only attention mechanisms.
        This sets requires_grad=False for the shared embedding system.
        """
        for param in self.model.shared_embedding.parameters():
            param.requires_grad = False
        
        print("✓ Shared embeddings frozen. Training only attention and feedforward layers.")
    
    def forward(self, data_source: torch.Tensor, data_intermediate: torch.Tensor, 
                data_target: torch.Tensor) -> Any:
        """
        Forward pass through the model.
        
        Args:
            data_source: Source nodes (S)
            data_intermediate: Intermediate variables (X)
            data_target: Target variables (Y)
            
        Returns:
            pred_x: Predicted X from decoder 1
            pred_y: Predicted Y from decoder 2
            attention_weights: Attention weights from both decoders
            masks: Masks for S, X, Y
            entropies: Attention entropies from both decoders
        """
        
        # Prepare intermediate input (blank X values for decoder 1)
        x_blanked = data_intermediate.clone()
        x_blanked[:, :, self.val_idx] = 0.0
        
        # Prepare target input (blank Y values for decoder 2)
        y_blanked = data_target.clone()
        y_blanked[:, :, self.val_idx] = 0.0
        
        # Determine whether to use teacher forcing
        # Only use teacher forcing during training, never during validation/testing
        use_tf = self.teacher_forcing and self.training
        
        # Model forward pass
        # Pass BOTH blanked (for decoder 1) and actual (for decoder 2 with teacher forcing)
        pred_x, pred_y, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            intermediate_tensor_actual=data_intermediate,
            target_tensor=y_blanked,
            use_teacher_forcing=use_tf
        )
        
        return pred_x, pred_y, attention_weights, masks, entropies
    
    def _step(self, batch, stage: str = None):
        """
        Common step logic for train/val/test.
        
        Args:
            batch: Tuple of (S, X, Y) tensors
            stage: One of "train", "val", or "test"
            
        Returns:
            total_loss: Combined loss from both stages
            pred_x: Predicted X values
            pred_y: Predicted Y values
            X: Actual X values
            Y: Actual Y values
        """
        
        S, X, Y = batch
        
        # Extract actual values for loss computation
        x_val = X[:, :, self.val_idx]
        y_val = Y[:, :, self.val_idx]
        
        # Forward pass
        pred_x, pred_y, attention_weights, masks, entropies = self.forward(
            data_source=S,
            data_intermediate=X,
            data_target=Y
        )
        
        # Unpack attention weights and entropies
        dec1_cross_att, dec1_self_att, dec2_cross_att, dec2_self_att = attention_weights
        dec1_cross_ent, dec1_self_ent, dec2_cross_ent, dec2_self_ent = entropies
        
        # Compute entropy regularization if needed
        if self.gamma > 0 or self.log_entropy:
            # Average entropy across all layers
            dec1_cross_ent_batch = torch.concat(dec1_cross_ent, dim=0).mean()
            dec1_self_ent_batch = torch.concat(dec1_self_ent, dim=0).mean()
            dec2_cross_ent_batch = torch.concat(dec2_cross_ent, dim=0).mean()
            dec2_self_ent_batch = torch.concat(dec2_self_ent, dim=0).mean()
        
        # Get learned DAG parameters for acyclicity regularization
        dec1_inner_att = self.model.decoder1.layers[0].global_self_attention.inner_attention
        dec2_inner_att = self.model.decoder2.layers[0].global_self_attention.inner_attention
        
        # phi - learned DAGs (only available for LieAttention)
        dec1_phi = getattr(dec1_inner_att, 'phi', None)
        dec2_phi = getattr(dec2_inner_att, 'phi', None)
        
        # Batch statistics (with gradients for regularization, only available for LieAttention)
        dec1_batch_mean = getattr(dec1_inner_att, 'batch_att_mean', None)
        dec1_batch_snr = getattr(dec1_inner_att, 'batch_att_snr', None)
        dec2_batch_mean = getattr(dec2_inner_att, 'batch_att_mean', None)
        dec2_batch_snr = getattr(dec2_inner_att, 'batch_att_snr', None)
        
        # Running averages (detached, used as priors, only available for LieAttention)
        dec1_runav_mean = getattr(dec1_inner_att, 'runav_att_mean', None)
        dec1_runav_snr = getattr(dec1_inner_att, 'runav_att_snr', None)
        dec2_runav_mean = getattr(dec2_inner_att, 'runav_att_mean', None)
        dec2_runav_snr = getattr(dec2_inner_att, 'runav_att_snr', None)
        
        # Entropy regularizer
        if self.gamma > 0:
            entropy_regularizer = self.gamma * (
                1.0 / dec1_cross_ent_batch + 
                1.0 / dec1_self_ent_batch + 
                1.0 / dec2_cross_ent_batch + 
                1.0 / dec2_self_ent_batch
            )
        else:
            entropy_regularizer = 0.0
        
        # Acyclicity regularizer
        if self.kappa > 0:
            acyclic_regularizer = 0.0
            
            if dec1_phi is not None:
                if dec1_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder 1 phi has shape {dec1_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec1_phi)
            
            if dec2_phi is not None:
                if dec2_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder 2 phi has shape {dec2_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec2_phi)
            
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # Prior regularizer - KL divergence between learned phi and empirical evidence
        def _get_prior_reg(phi, evidence, alpha):
            """KL divergence between learned phi and empirical evidence, weighted by SNR."""
            if phi is None or evidence is None or alpha is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            
            # Explicit KL divergence for two Bernoulli distributions p and p0
            kl = (alpha * (p * (torch.log(p + _eps) - torch.log(p0 + _eps)) + 
                         (1 - p) * (torch.log(1 - p + _eps) - torch.log(1 - p0 + _eps)))).mean()
            return kl
        
        prior_regularizer = (_get_prior_reg(dec1_phi, dec1_runav_mean, dec1_runav_snr) + 
                            _get_prior_reg(dec2_phi, dec2_runav_mean, dec2_runav_snr))
        
        # Compute losses for X and Y
        x_target = torch.nan_to_num(x_val)
        y_target = torch.nan_to_num(y_val)
        
        mse_x_per_elem = self.loss_fn(pred_x.squeeze(), x_target.squeeze())
        loss_x = mse_x_per_elem.mean()
        
        mse_y_per_elem = self.loss_fn(pred_y.squeeze(), y_target.squeeze())
        loss_y = mse_y_per_elem.mean()
        
        # Combined loss with weighting
        total_loss = (self.loss_weight_x * loss_x + 
                     self.loss_weight_y * loss_y + 
                     entropy_regularizer + 
                     acyclic_regularizer +
                     prior_regularizer)
        
        # Log individual losses
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        self.log(f"{stage}_loss_y", loss_y, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log metrics for X reconstruction
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred_x.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log metrics for Y prediction
        for name, metric in [("mae", self.mae_y), ("rmse", self.rmse_y), ("r2", self.r2_y)]:
            metric_eval = metric(pred_y.reshape(-1), y_target.reshape(-1))
            self.log(f"{stage}_y_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log entropies if requested
        if self.log_entropy:
            self.log(f"{stage}_dec1_cross_entropy", dec1_cross_ent_batch, on_step=False, on_epoch=True)
            self.log(f"{stage}_dec1_self_entropy", dec1_self_ent_batch, on_step=False, on_epoch=True)
            self.log(f"{stage}_dec2_cross_entropy", dec2_cross_ent_batch, on_step=False, on_epoch=True)
            self.log(f"{stage}_dec2_self_entropy", dec2_self_ent_batch, on_step=False, on_epoch=True)
        
        # Log acyclicity if requested
        if self.log_acyclicity:
            self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True)
        
        return total_loss, pred_x, pred_y, X, Y
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _, _, _ = self._step(batch=batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, _, _, _, _ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer (AdamW or SGD) with optional learning rate scheduler."""
        
        learning_rate = self.config["training"].get("lr", 1e-4)
        weight_decay = self.config["training"].get("weight_decay", 0.01)
        optimizer_type = self.config["training"].get("optimizer", "adamw").lower()
        
        # Select optimizer based on config
        if optimizer_type == "sgd":
            momentum = self.config["training"].get("momentum", 0.0)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adamw' or 'sgd'.")
        
        # Optional: Add learning rate scheduler
        if self.config["training"].get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        
        return optimizer
    
    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """
        NOTEARS acyclicity constraint.
        
        Args:
            A: (d, d) adjacency matrix (non-negative entries)
            
        Returns:
            Scalar acyclicity penalty
        """
        d = A.shape[0]
        expm_A = torch.matrix_exp(torch.relu(A))
        return torch.trace(expm_A) - d
