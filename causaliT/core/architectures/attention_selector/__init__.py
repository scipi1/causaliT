"""
AttentionSelector: Single cross-attention block for observational causal discovery.

Research question: Can a single cross-attention block — with X tokens as queries
and [S_actual, X_actual] as keys/values (diagonal of X-X block masked) — act
as a learnable variable selector that recovers causal parents from observational data?

This architecture deliberately strips away every complexity of the full pipeline
(no self-attention, no multi-stage decoder, no embeddings-only structural signal)
to provide a minimal, interpretable test bed for the core hypothesis.
"""

from .model import AttentionSelectorLayer

__all__ = ["AttentionSelectorLayer"]
