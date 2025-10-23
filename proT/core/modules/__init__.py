"""
ProT Core Transformer Modules

This package contains the building blocks of the transformer architecture:
- Attention mechanisms
- Encoder and decoder layers
- Embedding modules
- Extra utility layers
"""

from .attention import ScaledDotAttention, AttentionLayer
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .embedding import ModularEmbedding
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask

__all__ = [
    'ScaledDotAttention',
    'AttentionLayer',
    'Encoder',
    'EncoderLayer',
    'Decoder',
    'DecoderLayer',
    'ModularEmbedding',
    'Normalization',
    'UniformAttentionMask',
]
