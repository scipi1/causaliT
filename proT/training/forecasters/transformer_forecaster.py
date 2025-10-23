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
    Simplified version with AdamW optimizer.

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
        
        # Entropy regularizer
        self.gamma = config["training"].get("gamma", 0.1)
        self.entropy_regularizer = config["training"].get("entropy_regularizer", False)
            
        self.save_hyperparameters(config)
        
        # Metrics
        self.mae   = tm.MeanAbsoluteError()
        self.rmse  = tm.MeanSquaredError(squared=False)
        self.r2    = tm.R2Score()
        
        
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
        
        forecast_output, _, _, (enc_self_ent, dec_self_ent, dec_cross_ent) = self.forward(data_input=X, data_trg=Y)
        
        # Entropy regularization
        enc_self = torch.concat(enc_self_ent, dim=0).mean()
        dec_self = torch.concat(dec_self_ent, dim=0).mean()
        dec_cross = torch.concat(dec_cross_ent, dim=0).mean()
            
        if self.entropy_regularizer:
            ent_regularizer = 1.0/enc_self + 1.0/dec_self + 1.0/dec_cross
        else:
            ent_regularizer = 0.0
        
        # Calculate loss
        predicted_value = forecast_output
        trg = torch.nan_to_num(trg_val)
        
        mse_per_elem = self.loss_fn(predicted_value.squeeze(), trg.squeeze())
        loss = mse_per_elem.mean()
        
        # Log metrics
        for name, metric in [("mae", self.mae), ("rmse", self.rmse), ("r2", self.r2)]:
            metric_eval = metric(predicted_value.reshape(-1), trg.reshape(-1))
            self.log(f"{stage}_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
            
        for name, value in [("enc_self_entropy", enc_self), ("dec_self_entropy", dec_self), ("dec_cross_entropy", dec_cross)]:
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Add entropy regularization to loss
        loss = loss + self.gamma * ent_regularizer
            
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
        """Configure AdamW optimizer with optional learning rate scheduler."""
        
        learning_rate = self.config["training"].get("base_lr", 1e-4)
        weight_decay = self.config["training"].get("weight_decay", 0.01)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
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
