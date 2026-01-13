"""
Architecture-specific implementations for causaliT.

This package contains different transformer architectures:
- stage_causal: StageCausaliT (dual-decoder with reversed attention)
- standard: ProT (encoder-decoder, located in core/model.py)
"""

from causaliT.core.architectures.stage_causal import StageCausaliT

__all__ = ['StageCausaliT']
