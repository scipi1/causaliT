"""
Architecture-specific implementations for causaliT.

This package contains different transformer architectures:
- stage_causal: StageCausaliT (dual-decoder with reversed attention)
- single_causal: SingleCausalLayer (single-decoder for S -> X learning)
- single_causal_res: SingleCausalLayerRes (SVFA dual-residual variant)
- noise_aware: NoiseAwareSingleCausalLayer (noise-aware S -> X with ambient/reading noise)
- standard: ProT (encoder-decoder, located in core/model.py)
"""

from causaliT.core.architectures.stage_causal import StageCausaliT
from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.architectures.single_causal_res import SingleCausalLayerRes
from causaliT.core.architectures.noise_aware import NoiseAwareSingleCausalLayer

__all__ = [
    'StageCausaliT',
    'SingleCausalLayer',
    'SingleCausalLayerRes',
    'NoiseAwareSingleCausalLayer',
]
