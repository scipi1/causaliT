"""
ProT Core Transformer Modules

This package contains the building blocks of the transformer architecture:
- Attention mechanisms
- Encoder and decoder layers
- Embedding modules
- Extra utility layers
"""

from .attention import ScaledDotAttention, CausalCrossAttention, SigmoidCrossAttention, AttentionLayer, ToeplitzAttention
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .embedding import ModularEmbedding
from .orthogonal_embedding import OrthogonalMaskEmbedding
from .orthogonal_linear import OrthogonalLinear
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask
from .noise_layers import AmbientNoiseLayer, ReadingNoiseHead, GaussianNLLLoss, VariancePropagationTracker
from .mlp_head import MLPHead

__all__ = [
    'ScaledDotAttention',
    'CausalCrossAttention',
    'SigmoidCrossAttention',
    'ToeplitzAttention',
    'AttentionLayer',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'ModularEmbedding',
    'OrthogonalMaskEmbedding',
    'OrthogonalLinear',
    'Normalization',
    'UniformAttentionMask',
    # Noise-aware modules
    'AmbientNoiseLayer',
    'ReadingNoiseHead',
    'GaussianNLLLoss',
    'VariancePropagationTracker',
    # Output head
    'MLPHead',
]
