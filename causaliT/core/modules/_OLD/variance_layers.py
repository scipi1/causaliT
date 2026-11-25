"""
Variance-centric noise layers for the VarianceCausalLayer architecture.

This module implements the components for variance-centric causal learning
as described in docs/documentation/NOISE_AWARE_2.md.

Key design:
- No noise is sampled during the forward pass (fully deterministic).
- σ_A[j] (IntrinsicNoiseLayer) is a learnable parameter that represents
  the intrinsic exogenous variance of node j.
- Variance is propagated analytically through the self-attention weights:
      Var(X_i) = Σ_j α_{ij}² · σ_A[j]²
- The full predicted covariance matrix (Wright's formula) is:
      Σ_model = α · diag(σ_A²) · α^T
  and can be matched to the empirical residual covariance during training.

References:
    docs/documentation/NOISE_AWARE_2.md
    Wright (1934): "The method of path coefficients."
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class IntrinsicNoiseLayer(nn.Module):
    """
    Per-node learnable variance parameter σ[i].

    A single σ[i] per node serves both roles:
    - **Diagonal** (own variance): σ[i]² is node i's intrinsic exogenous noise.
    - **Off-diagonal propagation**: σ[j]² is what node j contributes to
      downstream nodes via the self-attention weights α_ij.

    The resulting variance formula is:
        Var(X_i) = σ[i]²  +  Σ_{j≠i} α_{ij}² · σ[j]²
                 = (α_sq + I) @ σ²     [matrix form]

    This ensures that the same quantity that describes how much noise node j
    carries in the data is the quantity that propagates to j's children — no
    separate "source noise" vs "residual noise" distinction.

    In a correctly identified DAG:
    - Root nodes: large σ (all variance is intrinsic, no parent terms).
    - Leaf nodes: small σ (most variance inherited via off-diagonal terms).

    Unlike AmbientNoiseLayer, forward() is a pure identity — no noise is
    ever sampled or added to any tensor.

    Args:
        num_nodes:  Number of X nodes.
        init_sigma: Initial σ value (default 0.1).
    """

    def __init__(
        self,
        num_nodes: int,
        init_sigma: float = 0.1,
        # Legacy kwargs accepted but ignored for backward config compat.
        init_sigma_A: Optional[float] = None,
        init_sigma_R: Optional[float] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # If init_sigma_A was supplied (old config), honour it as init_sigma.
        _init = init_sigma_A if init_sigma_A is not None else init_sigma

        # Clamping bounds for numerical stability.
        self._log_min = -10.0   # σ >= exp(-10) ≈ 4.5e-5
        self._log_max = 2.0     # σ <= exp(2)  ≈ 7.4

        # Single per-node log-scale parameter.
        self.log_sigma = nn.Parameter(
            torch.full((num_nodes,), math.log(_init))
        )

    @property
    def sigma(self) -> torch.Tensor:
        """Per-node intrinsic noise std (positive, numerically stable)."""
        return torch.exp(self.log_sigma.clamp(self._log_min, self._log_max))

    # Aliases kept for any external code that references the old names.
    @property
    def sigma_A(self) -> torch.Tensor:
        return self.sigma

    @property
    def sigma_R(self) -> torch.Tensor:
        return self.sigma

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Identity — no noise added.  Exists for API compatibility."""
        return H

    def get_variance_contribution(self) -> torch.Tensor:
        """Returns σ² per node."""
        return self.sigma ** 2

    def __repr__(self):
        return (
            f"IntrinsicNoiseLayer(num_nodes={self.num_nodes}, "
            f"sigma_mean={self.sigma.mean().item():.4f})"
        )


class AnalyticalVarianceHead(nn.Module):
    """
    Analytical variance prediction from self-attention weights and σ_A.

    Implements the first-order linear variance propagation formula:

        Var(X_i) = Σ_j α_{ij}² · σ_A[j]²

    which is exact when self-attention is linear in V (always true by
    construction) and the output head is linear or approximately linear.

    Also exposes compute_sigma_model() for the full Wright covariance matrix:

        Σ_model = α · diag(σ_A²) · α^T

    Args:
        eps: Small floor for pred_var to avoid log(0) (default 1e-6).
        log_var_min: Lower clamp on log_var for stability (default -10).
        log_var_max: Upper clamp on log_var for stability (default 5).
    """

    def __init__(
        self,
        eps: float = 1e-6,
        log_var_min: float = -10.0,
        log_var_max: float = 5.0,
    ):
        super().__init__()
        self.eps = eps
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(
        self,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        sigma_R: Optional[torch.Tensor] = None,   # kept for API compat, ignored
    ) -> torch.Tensor:
        """
        Compute analytical log-variance with unified σ per node.

        Formula:
            Var(X_i) = σ[i]²  +  Σ_j α_{ij}² · σ[j]²
                     = (α_sq + I) @ σ²

        The diagonal term σ[i]² is the node's own intrinsic noise; the
        off-diagonal terms α_{ij}² · σ[j]² are variance inherited from
        parents.  The SAME σ[j] that gives node j its own variance is the
        one that propagates to its children — no separate source/residual
        distinction.

        Args:
            alpha:  Self-attention weight matrix.
                    Shape (B, L, L) or (B, H, L, L); averaged over heads.
            sigma:  Single per-node intrinsic std, shape (L,).
            sigma_R: Accepted but ignored (backward-compat only).

        Returns:
            log_var: (B, L, 1) — log of predicted marginal variance per node.
        """
        # Handle multi-head: average over heads.
        if alpha.dim() == 4:
            alpha = alpha.mean(dim=1)   # (B, L, L)

        B, L, _ = alpha.shape
        sigma_sq = sigma ** 2           # (L,)
        alpha_sq = alpha ** 2           # (B, L, L)

        # Add identity so each node's OWN σ² contributes to its variance.
        eye = torch.eye(L, device=alpha.device, dtype=alpha.dtype).unsqueeze(0)
        alpha_sq_with_diag = alpha_sq + eye     # (B, L, L)

        # Var(X_i) = (α_sq + I) @ σ²  →  (B, L)
        pred_var = alpha_sq_with_diag @ sigma_sq

        pred_var = pred_var.clamp(min=self.eps)

        log_var = torch.log(pred_var)   # (B, L)
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)

        # Expand to (B, L, 1) to match out_dim=1 NLL interface.
        return log_var.unsqueeze(-1)

    def compute_sigma_model(
        self,
        alpha: torch.Tensor,
        sigma_A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full predicted covariance matrix (Wright's formula).

            Σ_model = α · diag(σ_A²) · α^T    shape (L, L)

        Args:
            alpha: (B, L, L) or (L, L) self-attention weights.
            sigma_A: (L,) intrinsic std per node.

        Returns:
            Sigma_model: (L, L) symmetric predicted covariance matrix.
        """
        if alpha.dim() == 4:
            alpha = alpha.mean(dim=1)       # (B, L, L)
        if alpha.dim() == 3:
            alpha = alpha.mean(dim=0)       # (L, L) — batch mean
        sigma_sq = sigma_A ** 2             # (L,)
        # α · diag(σ²) · α^T = (α * σ²) @ α^T
        return (alpha * sigma_sq.unsqueeze(0)) @ alpha.t()


class ResidualCovarianceLoss(nn.Module):
    """
    Covariance matching loss (Wright's path formula).

    Matches the empirical covariance of residuals to the model-predicted
    covariance derived from the causal attention weights and intrinsic variances:

        L_cov = ‖Σ_emp_res − Σ_model‖²_F

    where:
        Σ_emp_res = (R − mean(R))^T (R − mean(R)) / B
        Σ_model   = α · diag(σ_A²) · α^T

    This loss provides a direct structural signal: the pattern of co-variation
    among X nodes (after removing S contribution) must be explained by the
    directed causal graph encoded in α.

    Note: Reliable estimates require B >> L (batch size >> number of nodes).

    Args:
        normalize: If True, divide by L² to make the loss scale-invariant
                   to the number of nodes. Default False.
    """

    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        residuals: torch.Tensor,
        alpha: torch.Tensor,
        sigma_A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute covariance matching loss.

        Args:
            residuals: (B, L) reconstruction residuals (x − μ).
            alpha: (B, L, L) or (B, H, L, L) self-attention weights.
            sigma_A: (L,) intrinsic noise std per node.

        Returns:
            Scalar loss: ‖Σ_emp_res − Σ_model‖²_F
        """
        B, L = residuals.shape[:2]

        # --- Empirical residual covariance ---
        R = residuals                                       # (B, L)
        R_c = R - R.mean(dim=0, keepdim=True)              # centre
        Sigma_emp = R_c.t() @ R_c / B                      # (L, L)

        # --- Model covariance (Wright's formula) ---
        if alpha.dim() == 4:
            alpha = alpha.mean(dim=1)                       # (B, L, L)
        if alpha.dim() == 3:
            alpha_mean = alpha.mean(dim=0)                  # (L, L)
        else:
            alpha_mean = alpha                              # (L, L) already

        sigma_sq = sigma_A ** 2                             # (L,)
        Sigma_model = (alpha_mean * sigma_sq.unsqueeze(0)) @ alpha_mean.t()  # (L, L)

        # --- Frobenius norm squared ---
        diff = Sigma_emp - Sigma_model                      # (L, L)
        loss = (diff ** 2).sum()

        if self.normalize:
            loss = loss / (L ** 2)

        return loss
