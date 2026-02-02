"""
Architecture-specific implementations for causaliT.

This package contains different transformer architectures:
- stage_causal: StageCausaliT (dual-decoder with reversed attention)
- single_causal: SingleCausalLayer (single-decoder for S → X learning)
- standard: ProT (encoder-decoder, located in core/model.py)
"""

from causaliT.core.architectures.stage_causal import StageCausaliT
from causaliT.core.architectures.single_causal import SingleCausalLayer

__all__ = ['StageCausaliT', 'SingleCausalLayer']
