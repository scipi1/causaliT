"""
Structural losses of the attention-selector forecaster.

Modules:
    acyclicity    -- h(A) penalties (notears, logdet, nilpotent)
    independence  -- residual-independence objectives (dHSIC, anti-adjacency
                     log-ratio, joint non-descendant test) + their configs

Build from config:
    build_acyclicity_penalty(cfg["training"])   -> callable h(A)
    AntiAdjacencyConfig.from_config(cfg["training"])
    resolve_hsic_objective(cfg["training"])     -> "pairwise" | "dhsic_residual"

Coding Assistant:
    * Define and document every structural loss HERE.  The forecaster only
      collects model state (forwards, order mask M, selector pi, bandwidths,
      null scale) and calls these functions.
    * Optimisation around the loss (lambda / ALM duals, safeguards, null
      calibration, gradient routing) stays in the forecaster / trainer.
    * Pairwise HSIC kernels and aggregations live in ``causaliT.utils.hsic_utils``.
"""

from causaliT.training.structural_losses.acyclicity import (
    ACYCLICITY_FNS,
    AcyclicityPenalty,
    build_acyclicity_penalty,
    logdet,
    nilpotent,
    notears,
)
from causaliT.training.structural_losses.independence import (
    HSIC_OBJECTIVES,
    AntiAdjacencyConfig,
    KernelOpts,
    anti_adjacency_loss,
    dhsic_residual_loss,
    joint_nd_loss,
    resolve_hsic_objective,
)

__all__ = [
    "ACYCLICITY_FNS",
    "AcyclicityPenalty",
    "build_acyclicity_penalty",
    "logdet",
    "nilpotent",
    "notears",
    "HSIC_OBJECTIVES",
    "AntiAdjacencyConfig",
    "KernelOpts",
    "anti_adjacency_loss",
    "dhsic_residual_loss",
    "joint_nd_loss",
    "resolve_hsic_objective",
]
