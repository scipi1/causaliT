"""
StageCausaliT: Multi-stage causal transformer architecture.

This module implements a dual-decoder architecture with reversed attention order:
- Decoder 1: Source (S) → Intermediate (X) reconstruction
- Decoder 2: Intermediate (X) → Target (Y) prediction

Key features:
- Cross-attention before self-attention in each decoder
- Shared embedding/de-embedding system
- Teacher forcing support
"""

from causaliT.core.architectures.stage_causal.model import StageCausaliT
from causaliT.core.architectures.stage_causal.decoder import ReversedDecoder, ReversedDecoderLayer

__all__ = ['StageCausaliT', 'ReversedDecoder', 'ReversedDecoderLayer']
