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
from proT.core import ProT


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
            
        if self.kappa > 0 or self.log_acyclicity:
            enc_self_att_batch = torch.mean(torch.concat(enc_self_att, dim=0), dim=0)
            dec_self_att_batch = torch.mean(torch.concat(dec_self_att, dim=0), dim=0)
            
        if self.gamma>0:
            entropy_regularizer = self.gamma * (1.0/enc_self_ent_batch + 1.0/dec_self_ent_batch + 1.0/dec_cross_ent_batch)
        else:
            entropy_regularizer = 0.0
            
        if self.kappa > 0:
            
            acyclic_regularizer = self.kappa * (self._notears_acyclicity(enc_self_att_batch))
        else:
            acyclic_regularizer = 0.0
        
        
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
            self.log(f"{stage}_{"notears"}", acyclic_regularizer, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Add entropy regularization to loss
        loss = loss + entropy_regularizer + acyclic_regularizer
            
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
