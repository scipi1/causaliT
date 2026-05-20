"""NoiseAwareSingleCausalLayerRes — noise-aware SVFA dual-residual architecture.

Difference from ``noise_aware``:
    Both cross- and self-attention now apply a residual connection on the
    STRUCTURE stream (X_struct) in addition to the value stream (X_val),
    exactly as in ``single_causal_res``.  The structural residual uses a
    dedicated structural value projection (W_s, W_s') routed to the
    structural-loss optimizer group (HSIC, score sparsity) via gradient
    routing, while the value-stream residual remains the reconstruction-driven
    W_v path.

    Ambient noise injection (the key feature of ``noise_aware``) is
    PRESERVED: noise is still injected on H_det (value path) between
    cross- and self-attention.  X_struct is never noised — Q/K scores remain
    deterministic.

The output head is unchanged: ``(μ, log_var) = ReadingNoiseHead(X_val)``.
"""

from .model import NoiseAwareSingleCausalLayerRes

__all__ = ["NoiseAwareSingleCausalLayerRes"]
