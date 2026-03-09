"""
Noise-Aware Causal Transformer Architecture.

This module implements a noise-aware variant of the single causal transformer
that explicitly models ambient (process) noise and reading (measurement) noise.

Key Features:
- Ambient noise injection after cross-attention (H = H_det + σ_A * ε)
- Self-attention mixing of noisy physical states
- Probabilistic output with reading noise (X ~ N(μ, σ_R²))
- Gaussian NLL training for uncertainty quantification

Design Choices (marked for paper):
- Per-node noise parameters: σ_A[j] and σ_R[i] are node-specific
- Noise injected BEFORE W_v projection (in embedding space)
- SVFA factorization required for clean separation of structure and values

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
"""

from .model import NoiseAwareSingleCausalLayer

__all__ = ["NoiseAwareSingleCausalLayer"]
