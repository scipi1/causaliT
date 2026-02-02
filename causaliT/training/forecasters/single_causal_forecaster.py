"""
SingleCausalForecaster: PyTorch Lightning wrapper for SingleCausalLayer model.

This forecaster handles training, validation, and testing for the single-decoder
architecture focusing on S → X causal learning.
"""

from typing import Any, Dict, Optional
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.utils import load_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token


class SingleCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for SingleCausalLayer transformer model.
    
    This forecaster manages training for a single causal relationship: S → X
    
    Features:
    - Single loss computation (MSE for X only)
    - Entropy and acyclicity regularization support
    - Hard mask support for enforcing ground-truth DAG structure
    - Designed for staged learning experiments
    
    Args:
        config: Configuration dictionary containing model, training, and data settings
        data_dir: Optional data directory for loading hard masks
    """
    def __init__(self, config, data_dir: str = None):
        super().__init__()
        
        self.config = config
        self.model = SingleCausalLayer(**config["model"]["kwargs"])
        
        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # Logging configuration
        self.log_entropy = config["training"].get("log_entropy", False)
        self.log_acyclicity = config["training"].get("log_acyclicity", False)
        
        # Regularizers
        self.gamma = config["training"].get("gamma", 0)   # Entropy regularization
        self.kappa = config["training"].get("kappa", 0)   # Acyclicity regularization
        
        # Sparsity regularization - L1 penalty on edge probabilities
        self.lambda_sparse = config["training"].get("lambda_sparse", 0)
        self.lambda_sparse_cross = config["training"].get("lambda_sparse_cross", None)
        
        if self.lambda_sparse_cross is None:
            self.lambda_sparse_cross = self.lambda_sparse
        
        # Logging for sparsity
        self.log_sparsity = config["training"].get("log_sparsity", False)
        
        # HSIC regularization - encourages independence between S and residuals
        # Lower HSIC values indicate better causal structure learning
        self.lambda_hsic = config["training"].get("lambda_hsic", 0)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.log_hsic = config["training"].get("log_hsic", False)
        
        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None
        
        # Register placeholder buffers if hard masks are enabled
        if self.use_hard_masks:
            self._register_hard_mask_placeholders()
        
        # Load hard masks if enabled and data_dir provided
        if self.use_hard_masks and data_dir is not None:
            self._load_hard_masks(config, data_dir)
        
        self.save_hyperparameters(config)
        
        # Metrics for X reconstruction
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()
        
        # Optionally freeze embeddings
        if config["training"].get("freeze_embeddings", False):
            self.freeze_embeddings()
        
        # Optionally freeze decoder (for training only forecaster)
        if config["training"].get("freeze_decoder", False):
            self.freeze_decoder()
        
        # Optionally freeze forecaster (for training only attention)
        if config["training"].get("freeze_forecaster", False):
            self.freeze_forecaster()
    
    def _register_hard_mask_placeholders(self):
        """Register placeholder buffers for hard masks."""
        S_len = self.config["data"]["S_seq_len"]
        X_len = self.config["data"]["X_seq_len"]
        
        # dec_cross: S → X cross-attention (X_len queries, S_len keys)
        self.register_buffer('hard_mask_dec_cross', torch.zeros(X_len, S_len))
        # dec_self: X self-attention (X_len x X_len)
        self.register_buffer('hard_mask_dec_self', torch.zeros(X_len, X_len))
    
    def _load_hard_masks(self, config: dict, data_dir: str):
        """Load hard masks from data directory based on config."""
        mask_files = config["training"].get("hard_mask_files", None)
        
        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified in config.")
            return
        
        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)
        
        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')
        
        if masks is not None:
            self._hard_masks = masks
            self._hard_masks_loaded = True
            
            for name, mask in masks.items():
                self.register_buffer(f'hard_mask_{name}', mask)
            
            print(f"✓ Hard masks loaded and registered for training.")
        else:
            print("Warning: No hard masks were loaded.")
    
    def get_hard_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get hard masks dictionary, retrieving from buffers."""
        if not self.use_hard_masks:
            return None
        
        masks = {}
        for name in ['dec_cross', 'dec_self']:
            buffer_name = f'hard_mask_{name}'
            if hasattr(self, buffer_name):
                masks[name] = getattr(self, buffer_name)
        
        return masks if masks else None
    
    def freeze_embedding_S(self):
        """Freeze the S embedding (orthogonal, should already be frozen)."""
        self.model.freeze_embedding_S()
        print("✓ S embedding frozen.")
    
    def unfreeze_embedding_S(self):
        """Unfreeze the S embedding."""
        self.model.unfreeze_embedding_S()
        print("✓ S embedding unfrozen.")
    
    def freeze_embedding_X(self):
        """Freeze the X embedding (learnable)."""
        self.model.freeze_embedding_X()
        print("✓ X embedding frozen.")
    
    def unfreeze_embedding_X(self):
        """Unfreeze the X embedding."""
        self.model.unfreeze_embedding_X()
        print("✓ X embedding unfrozen.")
    
    def freeze_embeddings(self):
        """Freeze both S and X embeddings."""
        self.freeze_embedding_S()
        self.freeze_embedding_X()
        print("✓ Both embeddings frozen.")
    
    def unfreeze_embeddings(self):
        """Unfreeze both S and X embeddings (S will unfreeze value layer only)."""
        self.unfreeze_embedding_S()
        self.unfreeze_embedding_X()
        print("✓ Both embeddings unfrozen.")
    
    def freeze_decoder(self):
        """Freeze the entire decoder (attention + FFN)."""
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        print("✓ Decoder frozen.")
    
    def unfreeze_decoder(self):
        """Unfreeze the entire decoder."""
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        print("✓ Decoder unfrozen.")
    
    def freeze_decoder_attention(self):
        """Freeze only the attention layers in the decoder."""
        for layer in self.model.decoder.layers:
            for param in layer.global_cross_attention.parameters():
                param.requires_grad = False
            for param in layer.global_self_attention.parameters():
                param.requires_grad = False
        print("✓ Decoder attention layers frozen.")
    
    def unfreeze_decoder_attention(self):
        """Unfreeze only the attention layers in the decoder."""
        for layer in self.model.decoder.layers:
            for param in layer.global_cross_attention.parameters():
                param.requires_grad = True
            for param in layer.global_self_attention.parameters():
                param.requires_grad = True
        print("✓ Decoder attention layers unfrozen.")
    
    def freeze_decoder_ffn(self):
        """Freeze only the feedforward layers in the decoder."""
        for layer in self.model.decoder.layers:
            for param in layer.linear1.parameters():
                param.requires_grad = False
            for param in layer.linear2.parameters():
                param.requires_grad = False
        print("✓ Decoder FFN layers frozen.")
    
    def unfreeze_decoder_ffn(self):
        """Unfreeze only the feedforward layers in the decoder."""
        for layer in self.model.decoder.layers:
            for param in layer.linear1.parameters():
                param.requires_grad = True
            for param in layer.linear2.parameters():
                param.requires_grad = True
        print("✓ Decoder FFN layers unfrozen.")
    
    def freeze_forecaster(self):
        """Freeze the forecaster (de-embedding) layer."""
        for param in self.model.forecaster.parameters():
            param.requires_grad = False
        print("✓ Forecaster frozen.")
    
    def unfreeze_forecaster(self):
        """Unfreeze the forecaster (de-embedding) layer."""
        for param in self.model.forecaster.parameters():
            param.requires_grad = True
        print("✓ Forecaster unfrozen.")
    
    def forward(self, data_source: torch.Tensor, data_intermediate: torch.Tensor,
                toggle_off_hard_masks: bool = False) -> Any:
        """
        Forward pass through the model.
        
        Args:
            data_source: Source nodes (S)
            data_intermediate: Intermediate variables (X)
            toggle_off_hard_masks: If True, disables hard masks even if model was trained with them.
            
        Returns:
            pred_x: Predicted X from decoder
            attention_weights: Attention weights from decoder
            masks: Masks for S, X
            entropies: Attention entropies from decoder
        """
        
        # Prepare intermediate input (blank X values)
        x_blanked = data_intermediate.clone()
        x_blanked[:, :, self.val_idx] = 0.0
        
        # Determine whether to use hard masks
        apply_hard_masks = self.use_hard_masks and not toggle_off_hard_masks
        hard_masks = self.get_hard_masks() if apply_hard_masks else None
        
        # Model forward pass
        pred_x, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=hard_masks,
        )
        
        return pred_x, attention_weights, masks, entropies
    
    def _step(self, batch, stage: str = None):
        """
        Common step logic for train/val/test.
        
        Args:
            batch: Tuple of (S, X, Y) tensors - Y is ignored
            stage: One of "train", "val", or "test"
            
        Returns:
            total_loss: Loss from X prediction
            pred_x: Predicted X values
            X: Actual X values
        """
        
        # Unpack batch (ignore Y)
        S, X, Y = batch
        
        # Extract actual values for loss computation
        x_val = X[:, :, self.val_idx]
        
        # Forward pass
        pred_x, attention_weights, masks, entropies = self.forward(
            data_source=S,
            data_intermediate=X
        )
        
        # Unpack attention weights and entropies
        dec_cross_att, dec_self_att = attention_weights
        dec_cross_ent, dec_self_ent = entropies
        
        # Compute entropy regularization if needed
        if self.gamma > 0 or self.log_entropy:
            dec_cross_ent_batch = torch.concat(dec_cross_ent, dim=0).mean()
            dec_self_ent_batch = torch.concat(dec_self_ent, dim=0).mean()
        
        # Get learned DAG parameters for acyclicity and prior regularization
        dec_self_inner = self.model.decoder.layers[0].global_self_attention.inner_attention
        dec_cross_inner = self.model.decoder.layers[0].global_cross_attention.inner_attention
        
        # phi - learned DAGs
        dec_self_phi = getattr(dec_self_inner, 'phi', None)
        dec_cross_phi = getattr(dec_cross_inner, 'phi', None)
        
        # Batch statistics (with gradients for regularization)
        dec_self_batch_mean = getattr(dec_self_inner, 'batch_att_mean', None)
        dec_self_batch_snr = getattr(dec_self_inner, 'batch_att_snr', None)
        dec_cross_batch_mean = getattr(dec_cross_inner, 'batch_att_mean', None)
        dec_cross_batch_snr = getattr(dec_cross_inner, 'batch_att_snr', None)
        
        # Running averages (detached, used as priors)
        dec_self_runav_mean = getattr(dec_self_inner, 'runav_att_mean', None)
        dec_self_runav_snr = getattr(dec_self_inner, 'runav_att_snr', None)
        dec_cross_runav_mean = getattr(dec_cross_inner, 'runav_att_mean', None)
        dec_cross_runav_snr = getattr(dec_cross_inner, 'runav_att_snr', None)
        
        # Entropy regularizer
        if self.gamma > 0:
            entropy_regularizer = self.gamma * (
                1.0 / dec_cross_ent_batch + 
                1.0 / dec_self_ent_batch
            )
        else:
            entropy_regularizer = 0.0
        
        # Acyclicity regularizer (only for self-attention DAGs)
        if self.kappa > 0:
            acyclic_regularizer = 0.0
            
            if dec_self_phi is not None:
                if dec_self_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder self-attention phi has shape {dec_self_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec_self_phi)
            
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # Prior regularizer
        def _get_prior_reg(phi, evidence, alpha):
            """KL divergence between learned phi and empirical evidence."""
            if phi is None or evidence is None or alpha is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            alpha_abs = torch.abs(alpha)
            kl = (alpha_abs * (p * (torch.log(p + _eps) - torch.log(p0 + _eps)) + 
                              (1 - p) * (torch.log(1 - p + _eps) - torch.log(1 - p0 + _eps)))).mean()
            return kl
        
        # Self and cross-attention prior regularization
        prior_regularizer = (
            _get_prior_reg(dec_self_phi, dec_self_runav_mean, dec_self_runav_snr) + 
            _get_prior_reg(dec_cross_phi, dec_cross_runav_mean, dec_cross_runav_snr)
        )
        
        # Sparsity regularizer
        def _get_sparsity_reg(phi):
            """L1-style sparsity penalty: sum of edge probabilities."""
            if phi is None:
                return 0.0
            return torch.sigmoid(phi).mean()
        
        self_attention_sparsity = _get_sparsity_reg(dec_self_phi)
        cross_attention_sparsity = _get_sparsity_reg(dec_cross_phi)
        
        sparsity_regularizer = (
            self.lambda_sparse * self_attention_sparsity +
            self.lambda_sparse_cross * cross_attention_sparsity
        )
        
        # Compute loss for X
        x_target = torch.nan_to_num(x_val)
        mse_x_per_elem = self.loss_fn(pred_x.squeeze(), x_target.squeeze())
        loss_x = mse_x_per_elem.mean()
        
        # HSIC regularizer - encourages independence between S and residuals
        # Lower HSIC = less dependence = better causal structure learning
        if self.lambda_hsic > 0 or self.log_hsic:
            # Compute residuals: (batch, seq_len_y)
            residuals = x_target.squeeze() - pred_x.squeeze()
            
            # Mean residuals across sequence length: (batch,)
            if residuals.dim() > 1:
                mean_residuals = residuals.mean(dim=1)
            else:
                mean_residuals = residuals
            
            # Extract S values: (batch, seq_len_s)
            s_values = S[:, :, self.val_idx]
            
            # Compute HSIC between each S token and mean residuals
            hsic_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma)
            hsic_regularizer = self.lambda_hsic * hsic_value
        else:
            hsic_regularizer = 0.0
            hsic_value = None
        
        # Total loss
        total_loss = (loss_x + 
                     entropy_regularizer + 
                     acyclic_regularizer +
                     prior_regularizer +
                     sparsity_regularizer +
                     hsic_regularizer)
        
        # Log loss
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log metrics for X reconstruction
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred_x.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val" and name == "mae"))
        
        # Log entropies if requested
        if self.log_entropy:
            self.log(f"{stage}_dec_cross_entropy", dec_cross_ent_batch, on_step=False, on_epoch=True)
            self.log(f"{stage}_dec_self_entropy", dec_self_ent_batch, on_step=False, on_epoch=True)
        
        # Log acyclicity if requested
        if self.log_acyclicity:
            self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True)
        
        # Log sparsity if requested
        if self.log_sparsity:
            self.log(f"{stage}_sparsity_self", self_attention_sparsity, on_step=False, on_epoch=True)
            self.log(f"{stage}_sparsity_cross", cross_attention_sparsity, on_step=False, on_epoch=True)
            self.log(f"{stage}_sparsity_total", sparsity_regularizer, on_step=False, on_epoch=True)
        
        # Log HSIC if requested
        if self.log_hsic and hsic_value is not None:
            self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
            self.log(f"{stage}_hsic_reg", hsic_regularizer, on_step=False, on_epoch=True)
        
        return total_loss, pred_x, X
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _ = self._step(batch=batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, _, _ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with optional learning rate scheduler."""
        
        learning_rate = self.config["training"].get("lr", 1e-4)
        weight_decay = self.config["training"].get("weight_decay", 0.01)
        optimizer_type = self.config["training"].get("optimizer", "adamw").lower()
        
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
