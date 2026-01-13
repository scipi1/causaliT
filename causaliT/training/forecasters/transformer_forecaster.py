# Standard library imports
import sys
from os.path import dirname, abspath
from typing import Any

# Third-party imports
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

# Local imports
from causaliT.core import ProT


class TransformerForecaster(pl.LightningModule):
    """
    Lightning wrapper for ProT transformer model.
    Supports multiple optimizers (AdamW, SGD) configurable via config.

    Args:
        config: configuration dictionary
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model = ProT(**config["model"]["kwargs"])
        
        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
            
        # Data indices
        self.dec_val_idx = config["data"]["val_idx"]
        
        # Log
        self.log_entropy = config["training"].get("log_entropy", False)
        self.log_acyclicity = config["training"].get("log_acyclicity", False)
        
        # Regularizer
        self.gamma = config["training"].get("gamma", 0)   # entropy  
        self.kappa = config["training"].get("kappa", 0)   # acyclicity
            
        self.save_hyperparameters(config)
        
        # Metrics
        self.mae   = tm.MeanAbsoluteError()
        self.rmse  = tm.MeanSquaredError(squared=False)
        self.r2    = tm.R2Score()
        
        # Optionally freeze embeddings based on config
        if config["training"].get("freeze_embeddings", False):
            self.freeze_embeddings()
    
    
    def freeze_embeddings(self):
        """
        Freeze all embedding layers to train only attention mechanisms.
        This sets requires_grad=False for all encoder and decoder embeddings.
        """
        # Freeze encoder embeddings
        for param in self.model.enc_embedding.parameters():
            param.requires_grad = False
        
        # Freeze decoder embeddings
        for param in self.model.dec_embedding.parameters():
            param.requires_grad = False
        
        print("✓ Embeddings frozen. Training only attention and feedforward layers.")
        
        
        
    def forward(self, data_input: torch.Tensor, data_trg: torch.Tensor) -> Any:
        """Forward pass through the model."""
        
        # Prepare decoder input (zero out target values)
        dec_input = data_trg.clone()
        dec_input[:,:, self.dec_val_idx] = 0.0
        trg_pos_mask = None
        
        # Model forward pass
        forecast_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.model.forward(
            input_tensor=data_input,
            target_tensor=dec_input,
            trg_pos_mask=trg_pos_mask
        )
        
        return forecast_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent)
    
    
    def _step(self, batch, stage: str=None):
        """Common step logic for train/val/test."""
        
        X, Y = batch
        trg_val = Y[:,:,self.dec_val_idx]
        
        forecast_output, (enc_self_att, dec_self_att, _), _, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(data_input=X, data_trg=Y)
        
        # Entropy regularization
        if self.gamma > 0 or self.log_entropy:
            enc_self_ent_batch = torch.concat(enc_self_ent, dim=0).mean()
            dec_self_ent_batch = torch.concat(dec_self_ent, dim=0).mean()
            dec_cross_ent_batch = torch.concat(dec_cross_ent, dim=0).mean()
        
        # Get batch statistics (with gradients) and running averages (detached) for prior regularization
        enc_inner_att = self.model.encoder.layers[0].global_attention.inner_attention
        dec_inner_att = self.model.decoder.layers[0].global_self_attention.inner_attention
        
        # phi - learned DAGs (only available for LieAttention)
        enc_phi = getattr(enc_inner_att, 'phi', None)
        dec_phi = getattr(dec_inner_att, 'phi', None)
        
        # Batch statistics (with gradients for regularization, only available for LieAttention)
        enc_batch_mean = getattr(enc_inner_att, 'batch_att_mean', None)
        enc_batch_snr = getattr(enc_inner_att, 'batch_att_snr', None)
        dec_batch_mean = getattr(dec_inner_att, 'batch_att_mean', None)
        dec_batch_snr = getattr(dec_inner_att, 'batch_att_snr', None)
        
        # Running averages (detached, used as priors, only available for LieAttention)
        enc_runav_mean = getattr(enc_inner_att, 'runav_att_mean', None)
        enc_runav_snr = getattr(enc_inner_att, 'runav_att_snr', None)
        dec_runav_mean = getattr(dec_inner_att, 'runav_att_mean', None)
        dec_runav_snr = getattr(dec_inner_att, 'runav_att_snr', None)
        
        # entropy regularizer
        if self.gamma>0:
            entropy_regularizer = self.gamma * (1.0/enc_self_ent_batch + 1.0/dec_self_ent_batch + 1.0/dec_cross_ent_batch)
        else:
            entropy_regularizer = 0.0
        
        # acyclicity regularizer    
        if self.kappa > 0:
            acyclic_regularizer = 0.0
            
            if enc_phi is not None:
                # Check that phi is 2D (single-head)
                if enc_phi.dim() != 2:
                    raise NotImplementedError(f"Acyclicity regularization only supports single-head attention. "
                                             f"Encoder phi has shape {enc_phi.shape}, expected 2D tensor.")
                acyclic_regularizer += self._notears_acyclicity(enc_phi)
            
            if dec_phi is not None:
                # Check that phi is 2D (single-head)
                if dec_phi.dim() != 2:
                    raise NotImplementedError(f"Acyclicity regularization only supports single-head attention. "
                                             f"Decoder phi has shape {dec_phi.shape}, expected 2D tensor.")
                acyclic_regularizer += self._notears_acyclicity(dec_phi)
            
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # prior regularizer - detach running averages to prevent gradient tracking
        def _get_prior_reg(phi, evidence, alpha):
            """KL divergence between learned phi and empirical evidence, weighted by SNR."""
            if phi is None or evidence is None or alpha is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            
            # explicit KL divergence for two Bernoulli distribution 0 and p0
            kl = (alpha*(p*(torch.log(p+_eps)-torch.log(p0+_eps))+(1-p)*(torch.log(1-p+_eps)-torch.log(1-p0+_eps)))).mean()
            return kl
        
        prior_regularizer = _get_prior_reg(enc_phi, enc_runav_mean, enc_runav_snr) + \
                           _get_prior_reg(dec_phi, dec_runav_mean, dec_runav_snr)
        
        # Calculate loss
        predicted_value = forecast_output
        trg = torch.nan_to_num(trg_val)
        
        mse_per_elem = self.loss_fn(predicted_value.squeeze(), trg.squeeze())
        loss = mse_per_elem.mean()
        
        # Log metrics
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            metric_eval = metric(predicted_value.reshape(-1), trg.reshape(-1))
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
            
        if self.log_entropy:
            for name, value in [("enc_self_entropy", enc_self_ent_batch), ("dec_self_entropy", dec_self_ent_batch), ("dec_cross_entropy", dec_cross_ent_batch)]:
                self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        if self.log_acyclicity:
            self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Add entropy regularization to loss
        loss = loss + entropy_regularizer + acyclic_regularizer + prior_regularizer
        
        return loss, predicted_value, Y
    
    
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
        A: (d, d) adjacency matrix (non-negative entries)
        returns: scalar acyclicity penalty
        """
        d = A.shape[0]
        expm_A = torch.matrix_exp(torch.relu(A))
        return torch.trace(expm_A) - d
