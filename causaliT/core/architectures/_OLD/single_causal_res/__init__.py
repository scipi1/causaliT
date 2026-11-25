"""SingleCausalLayerRes - SVFA dual-residual variant of SingleCausalLayer.

Difference from ``single_causal``:
    Both cross- and self-attention now apply a residual connection on the
    STRUCTURE stream (X_struct) in addition to the value stream (X_val).
    The structural residual uses a dedicated structural value projection
    (W_s, W_s') routed to the structural-loss group (HSIC, score sparsity)
    via gradient routing, while the value-stream residual remains identical
    to the original ``single_causal`` semantics (reconstruction-driven W_v).

See ``docs/SVFA_DUAL_RESIDUAL.md`` for the mathematical statement.
"""

from .model import SingleCausalLayerRes

__all__ = ["SingleCausalLayerRes"]
