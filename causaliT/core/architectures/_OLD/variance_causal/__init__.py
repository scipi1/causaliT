"""
variance_causal: Variance-centric causal transformer architecture.

Implements VarianceCausalLayer — a fully deterministic transformer that learns
causal structure via analytical variance propagation through self-attention:

    Var(X_i) = Σ_j α_{ij}² · σ_A[j]²

See docs/documentation/NOISE_AWARE_2.md for the full design description.
"""

from causaliT.core.architectures.variance_causal.model import VarianceCausalLayer
from causaliT.core.architectures.variance_causal.decoder import (
    VarianceCausalDecoderLayer,
    VarianceCausalDecoder,
)

__all__ = [
    "VarianceCausalLayer",
    "VarianceCausalDecoderLayer",
    "VarianceCausalDecoder",
]
