"""
StageCausalForecaster: PyTorch Lightning wrapper for StageCausaliT model.

This forecaster handles training, validation, and testing for the dual-decoder
architecture with proper teacher forcing and dual loss computation.
"""

import sys
from os.path import dirname, abspath, join
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.stage_causal import StageCausaliT
from causaliT.core.utils import load_dag_masks


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
    - Hard mask support for enforcing ground-truth DAG structure
    
    Args:
        config: Configuration dictionary containing model, training, and data settings
        data_dir: Optional data directory for loading hard masks
    """
    def __init__(self, config, data_dir: str = None):
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
        
        # Sparsity regularization - L1 penalty on edge probabilities
        # Separate coefficients for self-attention and cross-attention (cross typically needs more)
        self.lambda_sparse = config["training"].get("lambda_sparse", 0)  # Uniform sparsity coefficient
        self.lambda_sparse_cross = config["training"].get("lambda_sparse_cross", None)  # Override for cross-attention
        
        # If lambda_sparse_cross not specified, use same as lambda_sparse
        if self.lambda_sparse_cross is None:
            self.lambda_sparse_cross = self.lambda_sparse
        
        # Logging for sparsity
        self.log_sparsity = config["training"].get("log_sparsity", False)
        
        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None
        
        # ALWAYS register placeholder buffers if hard masks are enabled
        # This ensures checkpoint loading works (buffers must exist before load_state_dict)
        if self.use_hard_masks:
            self._register_hard_mask_placeholders()
        
        # Load hard masks if enabled and data_dir provided (during training)
        # This will overwrite the placeholders with actual mask values
        if self.use_hard_masks and data_dir is not None:
            self._load_hard_masks(config, data_dir)
        
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
    
    def _register_hard_mask_placeholders(self):
        """
        Register placeholder buffers for hard masks.
        
        This must be called during __init__ (before save_hyperparameters and before
        checkpoint loading) to ensure the buffers exist before load_state_dict is called.
        
        The placeholders will be overwritten by:
        - Actual mask values from checkpoint (during load_from_checkpoint)
        - Actual mask values from data directory (during training via _load_hard_masks)
        """
        # Get expected mask shapes from config
        S_len = self.config["data"]["S_seq_len"]
        X_len = self.config["data"]["X_seq_len"]
        Y_len = self.config["data"]["Y_seq_len"]
        
        # Register placeholder buffers with correct shapes
        # dec1_cross: S → X cross-attention (X_len queries, S_len keys)
        self.register_buffer('hard_mask_dec1_cross', torch.zeros(X_len, S_len))
        # dec1_self: X self-attention (X_len x X_len)
        self.register_buffer('hard_mask_dec1_self', torch.zeros(X_len, X_len))
        # dec2_cross: X → Y cross-attention (Y_len queries, X_len keys)
        self.register_buffer('hard_mask_dec2_cross', torch.zeros(Y_len, X_len))
        # dec2_self: Y self-attention (Y_len x Y_len)
        self.register_buffer('hard_mask_dec2_self', torch.zeros(Y_len, Y_len))
    
    def _load_hard_masks(self, config: dict, data_dir: str):
        """
        Load hard masks from data directory based on config.
        
        Args:
            config: Configuration dictionary with mask file paths
            data_dir: Base data directory
        """
        # Get mask filenames from config
        mask_files = config["training"].get("hard_mask_files", None)
        
        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified in config.")
            return
        
        # Construct full data path
        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)
        
        # Load masks
        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')
        
        if masks is not None:
            self._hard_masks = masks
            self._hard_masks_loaded = True
            
            # Register masks as buffers (overwrites placeholders, saved with model, moved with model)
            for name, mask in masks.items():
                self.register_buffer(f'hard_mask_{name}', mask)
            
            print(f"✓ Hard masks loaded and registered for training.")
        else:
            print("Warning: No hard masks were loaded.")
    
    def get_hard_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get hard masks dictionary, retrieving from buffers.
        
        This method retrieves masks from registered buffers. Masks can be present either:
        - From checkpoint loading (buffers populated by load_state_dict)
        - From explicit loading via _load_hard_masks during training
        
        Returns:
            Dictionary of hard masks or None if use_hard_masks is False.
        """
        if not self.use_hard_masks:
            return None
        
        # Retrieve masks from buffers (ensures correct device)
        # Buffers exist if use_hard_masks=True (registered by _register_hard_mask_placeholders)
        masks = {}
        for name in ['dec1_cross', 'dec1_self', 'dec2_cross', 'dec2_self']:
            buffer_name = f'hard_mask_{name}'
            if hasattr(self, buffer_name):
                masks[name] = getattr(self, buffer_name)
        
        return masks if masks else None
    
    def freeze_embeddings(self):
        """
        Freeze the shared embedding layer to train only attention mechanisms.
        This sets requires_grad=False for the shared embedding system.
        """
        for param in self.model.shared_embedding.parameters():
            param.requires_grad = False
        
        print("✓ Shared embeddings frozen. Training only attention and feedforward layers.")
    
    def forward(self, data_source: torch.Tensor, data_intermediate: torch.Tensor, 
                data_target: torch.Tensor, toggle_off_hard_masks: bool = False) -> Any:
        """
        Forward pass through the model.
        
        Args:
            data_source: Source nodes (S)
            data_intermediate: Intermediate variables (X)
            data_target: Target variables (Y)
            toggle_off_hard_masks: If True, disables hard masks even if model was trained
                                   with them. Useful for testing causality retention.
                                   Default False = use masks if model was trained with them.
            
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
        
        # Determine whether to use hard masks
        # Apply masks if: model was trained with masks AND toggle is not set to disable them
        apply_hard_masks = self.use_hard_masks and not toggle_off_hard_masks
        hard_masks = self.get_hard_masks() if apply_hard_masks else None
        
        # Model forward pass
        # Pass BOTH blanked (for decoder 1) and actual (for decoder 2 with teacher forcing)
        pred_x, pred_y, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            intermediate_tensor_actual=data_intermediate,
            target_tensor=y_blanked,
            use_teacher_forcing=use_tf,
            hard_masks=hard_masks,
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
        
        # Get learned DAG parameters for acyclicity and prior regularization
        # Self-attention modules (LieAttention)
        dec1_self_inner = self.model.decoder1.layers[0].global_self_attention.inner_attention
        dec2_self_inner = self.model.decoder2.layers[0].global_self_attention.inner_attention
        
        # Cross-attention modules (CausalCrossAttention)
        dec1_cross_inner = self.model.decoder1.layers[0].global_cross_attention.inner_attention
        dec2_cross_inner = self.model.decoder2.layers[0].global_cross_attention.inner_attention
        
        # phi - learned DAGs for self-attention (only available for LieAttention)
        dec1_self_phi = getattr(dec1_self_inner, 'phi', None)
        dec2_self_phi = getattr(dec2_self_inner, 'phi', None)
        
        # phi - learned DAGs for cross-attention (only available for CausalCrossAttention with DAG learning)
        dec1_cross_phi = getattr(dec1_cross_inner, 'phi', None)
        dec2_cross_phi = getattr(dec2_cross_inner, 'phi', None)
        
        # Batch statistics for self-attention (with gradients for regularization)
        dec1_self_batch_mean = getattr(dec1_self_inner, 'batch_att_mean', None)
        dec1_self_batch_snr = getattr(dec1_self_inner, 'batch_att_snr', None)
        dec2_self_batch_mean = getattr(dec2_self_inner, 'batch_att_mean', None)
        dec2_self_batch_snr = getattr(dec2_self_inner, 'batch_att_snr', None)
        
        # Batch statistics for cross-attention (with gradients for regularization)
        dec1_cross_batch_mean = getattr(dec1_cross_inner, 'batch_att_mean', None)
        dec1_cross_batch_snr = getattr(dec1_cross_inner, 'batch_att_snr', None)
        dec2_cross_batch_mean = getattr(dec2_cross_inner, 'batch_att_mean', None)
        dec2_cross_batch_snr = getattr(dec2_cross_inner, 'batch_att_snr', None)
        
        # Running averages for self-attention (detached, used as priors)
        dec1_self_runav_mean = getattr(dec1_self_inner, 'runav_att_mean', None)
        dec1_self_runav_snr = getattr(dec1_self_inner, 'runav_att_snr', None)
        dec2_self_runav_mean = getattr(dec2_self_inner, 'runav_att_mean', None)
        dec2_self_runav_snr = getattr(dec2_self_inner, 'runav_att_snr', None)
        
        # Running averages for cross-attention (detached, used as priors)
        dec1_cross_runav_mean = getattr(dec1_cross_inner, 'runav_att_mean', None)
        dec1_cross_runav_snr = getattr(dec1_cross_inner, 'runav_att_snr', None)
        dec2_cross_runav_mean = getattr(dec2_cross_inner, 'runav_att_mean', None)
        dec2_cross_runav_snr = getattr(dec2_cross_inner, 'runav_att_snr', None)
        
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
        
        # Acyclicity regularizer (only for self-attention DAGs, which are square)
        # Note: Cross-attention DAGs are bipartite and inherently acyclic, so no NOTEARS needed
        if self.kappa > 0:
            acyclic_regularizer = 0.0
            
            if dec1_self_phi is not None:
                if dec1_self_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder 1 self-attention phi has shape {dec1_self_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec1_self_phi)
            
            if dec2_self_phi is not None:
                if dec2_self_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder 2 self-attention phi has shape {dec2_self_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec2_self_phi)
            
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # Prior regularizer - KL divergence between learned phi and empirical evidence
        def _get_prior_reg(phi, evidence, alpha):
            """KL divergence between learned phi and empirical evidence, weighted by |SNR|.
            
            Uses absolute value of SNR to ensure the regularizer is always non-negative.
            SNR can be negative when batch_mean is negative (possible with GeLU activation),
            but we only care about the magnitude of confidence, not its sign.
            """
            if phi is None or evidence is None or alpha is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            
            # Use absolute value of alpha (SNR) to ensure non-negative weighting
            # SNR magnitude indicates confidence; sign is irrelevant for weighting
            alpha_abs = torch.abs(alpha)
            
            # Explicit KL divergence for two Bernoulli distributions p and p0
            # KL(p || p0) is always >= 0, and alpha_abs >= 0, so result is always >= 0
            kl = (alpha_abs * (p * (torch.log(p + _eps) - torch.log(p0 + _eps)) + 
                              (1 - p) * (torch.log(1 - p + _eps) - torch.log(1 - p0 + _eps)))).mean()
            return kl
        
        # Self-attention prior regularization (LieAttention)
        self_attention_prior = (
            _get_prior_reg(dec1_self_phi, dec1_self_runav_mean, dec1_self_runav_snr) + 
            _get_prior_reg(dec2_self_phi, dec2_self_runav_mean, dec2_self_runav_snr)
        )
        
        # Cross-attention prior regularization (CausalCrossAttention)
        cross_attention_prior = (
            _get_prior_reg(dec1_cross_phi, dec1_cross_runav_mean, dec1_cross_runav_snr) + 
            _get_prior_reg(dec2_cross_phi, dec2_cross_runav_mean, dec2_cross_runav_snr)
        )
        
        # Total prior regularizer
        prior_regularizer = self_attention_prior + cross_attention_prior
        
        # Sparsity regularizer - L1 penalty on edge probabilities
        # Encourages sparse DAG solutions by penalizing the expected number of edges
        def _get_sparsity_reg(phi):
            """L1-style sparsity penalty: sum of edge probabilities."""
            if phi is None:
                return 0.0
            # Mean of sigmoid(phi) = expected density of the graph
            return torch.sigmoid(phi).mean()
        
        # Self-attention sparsity (with lambda_sparse)
        self_attention_sparsity = (
            _get_sparsity_reg(dec1_self_phi) + 
            _get_sparsity_reg(dec2_self_phi)
        )
        
        # Cross-attention sparsity (with lambda_sparse_cross, which may be higher)
        cross_attention_sparsity = (
            _get_sparsity_reg(dec1_cross_phi) + 
            _get_sparsity_reg(dec2_cross_phi)
        )
        
        # Total sparsity regularizer (using separate coefficients if specified)
        sparsity_regularizer = (
            self.lambda_sparse * self_attention_sparsity +
            self.lambda_sparse_cross * cross_attention_sparsity
        )
        
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
                     prior_regularizer +
                     sparsity_regularizer)
        
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
        
        # Log sparsity if requested
        if self.log_sparsity:
            self.log(f"{stage}_sparsity_self", self_attention_sparsity, on_step=False, on_epoch=True)
            self.log(f"{stage}_sparsity_cross", cross_attention_sparsity, on_step=False, on_epoch=True)
            self.log(f"{stage}_sparsity_total", sparsity_regularizer, on_step=False, on_epoch=True)
        
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
