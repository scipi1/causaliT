"""
ProT Core Transformer Modules

This package contains the building blocks of the transformer architecture:
- Attention mechanisms
- Encoder and decoder layers
- Embedding modules
- Extra utility layers
"""

from .attention import (
    ScaledDotAttention,
    CausalCrossAttention,
    SigmoidCrossAttention,
    HardConcreteCrossAttention,
    ToeplitzAttention,
    AttentionLayer,
)
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .embedding import ModularEmbedding
from .orthogonal_embedding import OrthogonalMaskEmbedding, FixedOrthonormalEmbedding
from .orthogonal_linear import OrthogonalLinear
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask
from .noise_layers import AmbientNoiseLayer, ReadingNoiseHead, GaussianNLLLoss, VariancePropagationTracker
from .variance_layers import IntrinsicNoiseLayer, AnalyticalVarianceHead, ResidualCovarianceLoss
from .mlp_head import MLPHead

__all__ = [
    'ScaledDotAttention',
    'CausalCrossAttention',
    'SigmoidCrossAttention',
    'HardConcreteCrossAttention',
    'ToeplitzAttention',
    'AttentionLayer',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'ModularEmbedding',
    'OrthogonalMaskEmbedding',
    'FixedOrthonormalEmbedding',
    'OrthogonalLinear',
    'Normalization',
    'UniformAttentionMask',
    # Noise-aware modules
    'AmbientNoiseLayer',
    'ReadingNoiseHead',
    'GaussianNLLLoss',
    'VariancePropagationTracker',
    # Variance-centric modules
    'IntrinsicNoiseLayer',
    'AnalyticalVarianceHead',
    'ResidualCovarianceLoss',
    # Output head
    'MLPHead',
]
