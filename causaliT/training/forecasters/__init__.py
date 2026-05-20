"""
ProT Forecasters Package

Lightning wrappers for different model architectures.
Currently supports:
- TransformerForecaster: ProT transformer model
- StageCausalForecaster: StageCausaliT dual-decoder model
- SingleCausalForecaster: SingleCausalLayer single-decoder model
- SingleCausalResForecaster: SingleCausalLayerRes (SVFA dual-residual variant)
- NoiseAwareCausalForecaster: Noise-aware model with Gaussian NLL training
- NoiseAwareCausalResForecaster: Noise-aware SVFA dual-residual variant
"""

from .transformer_forecaster import TransformerForecaster
from .stage_causal_forecaster import StageCausalForecaster
from .single_causal_forecaster import SingleCausalForecaster
from .single_causal_res_forecaster import SingleCausalResForecaster
from .noise_aware_forecaster import NoiseAwareCausalForecaster
from .noise_aware_res_forecaster import NoiseAwareCausalResForecaster

__all__ = [
    'TransformerForecaster',
    'StageCausalForecaster',
    'SingleCausalForecaster',
    'SingleCausalResForecaster',
    'NoiseAwareCausalForecaster',
    'NoiseAwareCausalResForecaster',
]
