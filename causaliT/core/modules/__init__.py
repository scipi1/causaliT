"""
ProT Core Transformer Modules

This package contains the building blocks of the transformer architecture:
- Attention mechanisms
- Encoder and decoder layers
- Embedding modules
- Extra utility layers
"""

from .attention import LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .embedding import ModularEmbedding
from .orthogonal_embedding import OrthogonalMaskEmbedding
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask, DAGMask

__all__ = [
    'LieAttention',
    'ScaledDotAttention',
    'CausalCrossAttention',
    'PhiSoftMax',
    'AttentionLayer',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'ModularEmbedding',
    'OrthogonalMaskEmbedding',
    'Normalization',
    'UniformAttentionMask',
    'DAGMask',
]
