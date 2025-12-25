"""
Predictor for Transformer-based models (proT variants).
"""

from typing import Any, Dict
import torch
from pathlib import Path

from .base_predictor import BasePredictor
from proT.training.forecasters.transformer_forecaster import TransformerForecaster


class TransformerPredictor(BasePredictor):
    """
    Predictor for Transformer-based forecasting models.
    
    Handles:
    - TransformerForecaster (proT)
    
    Returns predictions along with attention weights from encoder, decoder, and cross-attention.
    """
    
    def _load_model(self) -> TransformerForecaster:
        """
        Load transformer model from checkpoint.
        
        Returns:
            Loaded transformer forecaster model
        """
        # causaliT only has TransformerForecaster
        model = TransformerForecaster.load_from_checkpoint(self.checkpoint_path)
        
        # Verify model loaded correctly
        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")
        
        if not any(param.requires_grad for param in model.parameters()):
            raise RuntimeError("Model parameters seem uninitialized. Check the checkpoint path.")
        
        return model
    
    def _forward(self, X: torch.Tensor, trg: torch.Tensor, **kwargs) -> Any:
        """
        Perform forward pass through transformer model.
        
        Args:
            X: Input tensor (B x L x F)
            trg: Target tensor (B x L x F)
            **kwargs: Additional arguments (e.g., show_trg_max_idx)
            
        Returns:
            Tuple of (forecast_output, attention_tuple, enc_mask, entropy_tuple)
            Note: causaliT model does not return recon_output (unlike proT)
        """
        # Call model forward - model is already in eval mode from BasePredictor.__init__
        output = self.model.forward(
            data_input=X,
            data_trg=trg,
            **kwargs
        )
        
        return output
    
    def _process_forward_output(self, output: Any) -> Dict[str, Any]:
        """
        Process transformer model output.
        
        Args:
            output: Tuple from model.forward():
                (forecast_output, (enc_self_att, dec_self_att, dec_cross_att), 
                 enc_mask, (enc_self_ent, dec_self_ent, dec_cross_ent))
                Note: causaliT model returns 4 values (no recon_output)
        
        Returns:
            Dictionary with:
                - 'forecast': Prediction tensor
                - 'attention_weights': Dict with 'encoder', 'decoder', 'cross' attention arrays
        """
        # causaliT model returns 4 values (no recon_output)
        forecast_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask, entropy = output
        
        # Extract attention weights (take first head/layer for simplicity)
        attention_weights = {
            'encoder': enc_self_att[0] if enc_self_att else None,
            'decoder': dec_self_att[0] if dec_self_att else None,
            'cross': dec_cross_att[0] if dec_cross_att else None,
        }
        
        return {
            'forecast': forecast_output,
            'attention_weights': attention_weights
        }
