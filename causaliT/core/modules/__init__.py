"""
Building blocks of the attention-selector architecture: attention variants,
gates, embeddings, heads and utility layers.
"""

from .attention import (
    ScaledDotSoftmax,
    CausalCrossAttention,
    SigmoidCrossAttention,
    HardConcreteCrossAttention,
    ToeplitzAttention,
    AttentionLayer,
)
from .gated_cross_attention import GatedCrossAttention
from .gated_self_attention import GatedSelfAttention
from .order_factorized_attention import OrderFactorizedSelfAttention
from .topk_gate import TopKGate
from .commutator_self_attention import CommutatorSelfAttention
from .gain_softmax import GainSoftmax

from .embedding import ModularEmbedding
from .orthogonal_embedding import OrthogonalMaskEmbedding, FixedOrthonormalEmbedding
from .orthogonal_linear import OrthogonalLinear
from .embedding_layers import *
from .extra_layers import Normalization, UniformAttentionMask
from .mlp_head import MLPHead, PerNodeMLPHead

__all__ = [
    'ScaledDotSoftmax',
    'CausalCrossAttention',
    'SigmoidCrossAttention',
    'HardConcreteCrossAttention',
    'GatedCrossAttention',
    'GatedSelfAttention',
    'OrderFactorizedSelfAttention',
    'TopKGate',
    'CommutatorSelfAttention',
    'GainSoftmax',
    'ToeplitzAttention',

    'AttentionLayer',
    'ModularEmbedding',
    'OrthogonalMaskEmbedding',
    'FixedOrthonormalEmbedding',
    'OrthogonalLinear',
    'Normalization',
    'UniformAttentionMask',
    # Output head
    'MLPHead',
    'PerNodeMLPHead',
]
