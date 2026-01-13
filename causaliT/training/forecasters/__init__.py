"""
ProT Forecasters Package

Lightning wrappers for different model architectures.
Currently supports:
- TransformerForecaster: ProT transformer model
- StageCausalForecaster: StageCausaliT dual-decoder model
"""

from .transformer_forecaster import TransformerForecaster
from .stage_causal_forecaster import StageCausalForecaster

__all__ = ['TransformerForecaster', 'StageCausalForecaster']
