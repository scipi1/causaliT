"""
ProT Core Transformer Modules

This package contains the building blocks of the transformer architecture:
- Attention mechanisms
- Encoder and decoder layers
- Embedding modules
- Extra utility layers
"""

from .attention import LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer, ToeplitzAttention, ToeplitzLieAttention
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .embedding import ModularEmbedding
from .orthogonal_embedding import OrthogonalMaskEmbedding
from .orthogonal_linear import OrthogonalLinear
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask, DAGMask
from .noise_layers import AmbientNoiseLayer, ReadingNoiseHead, GaussianNLLLoss, VariancePropagationTracker

__all__ = [
    'LieAttention',
    'ScaledDotAttention',
    'CausalCrossAttention',
    'PhiSoftMax',
    'ToeplitzLieAttention',
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
    'DAGMask',
    # Noise-aware modules
    'AmbientNoiseLayer',
    'ReadingNoiseHead',
    'GaussianNLLLoss',
    'VariancePropagationTracker',
]
