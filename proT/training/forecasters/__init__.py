"""
ProT Forecasters Package

Lightning wrappers for different model architectures.
Currently supports:
- TransformerForecaster: ProT transformer model
"""

from .transformer_forecaster import TransformerForecaster

__all__ = ['TransformerForecaster']
