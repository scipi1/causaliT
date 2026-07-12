"""
Architecture-specific implementations for causaliT.

This package contains different transformer architectures:
- stage_causal: StageCausaliT (dual-decoder with reversed attention)
- single_causal: SingleCausalLayer (single-decoder for S -> X learning)
- single_causal_res: SingleCausalLayerRes (SVFA dual-residual variant)
- noise_aware: NoiseAwareSingleCausalLayer (noise-aware S -> X with ambient/reading noise)
- variance_causal: VarianceCausalLayer (variance-centric, analytical noise propagation via alpha^2 @ sigma_A^2)
- standard: ProT (encoder-decoder, located in core/model.py)
"""

from causaliT.core.architectures.stage_causal import StageCausaliT
from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.architectures.single_causal_res import SingleCausalLayerRes
from causaliT.core.architectures.noise_aware import NoiseAwareSingleCausalLayer
from causaliT.core.architectures.variance_causal import VarianceCausalLayer
from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.architectures.self_selector import SelfSelectorLayer

__all__ = [
    'StageCausaliT',
    'SingleCausalLayer',
    'SingleCausalLayerRes',
    'NoiseAwareSingleCausalLayer',
    'VarianceCausalLayer',
    'AttentionSelectorLayer',
    'SelfSelectorLayer',
]


