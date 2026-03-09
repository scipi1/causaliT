"""
Noise injection layers for noise-aware causal transformers.

This module implements the noise components described in the noise-aware transformer
architecture, separating ambient (process) noise from reading (measurement) noise.

Design Choices (marked for paper):
- Per-node noise parameters: σ_A[j] and σ_R[i] are node-specific learnable parameters
- Noise injected BEFORE W_v projection: Ambient noise is in embedding space

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmbientNoiseLayer(nn.Module):
    """
    Injects node-level ambient (process) noise into hidden representations.
    
    This layer models environmental variability in the physical system:
    - ambient temperature fluctuations
    - environmental disturbances  
    - imperfect actuation
    - system drift
    
    The noisy hidden state is computed as:
        H = H_det + σ_A * ε, where ε ~ N(0, I)
    
    Design Choice: Per-node noise σ_A[j] for each source node j.
    Design Choice: Noise applied BEFORE W_v projection (in embedding space).
    
    Args:
        num_nodes: Number of nodes in the sequence (X_seq_len)
        d_model: Embedding dimension (for per-dimension noise if enabled)
        init_sigma: Initial σ_A value (default 0.01 for near-deterministic start)
        per_dimension: If True, use per-dimension noise σ_A[j, d]. Default False (per-node only).
        
    Attributes:
        log_sigma_A: Learnable log-scale ambient noise parameters.
                    Shape (num_nodes,) if per_dimension=False, else (num_nodes, d_model).
    """
    
    def __init__(
        self, 
        num_nodes: int, 
        d_model: int = None,
        init_sigma: float = 0.01,
        per_dimension: bool = False
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.per_dimension = per_dimension
        
        # Initialize log(σ_A) for positivity via exp()
        # Small init_sigma starts training near deterministic regime
        if per_dimension and d_model is not None:
            # Per-node per-dimension: σ_A[j, d]
            self.log_sigma_A = nn.Parameter(
                torch.full((num_nodes, d_model), math.log(init_sigma))
            )
        else:
            # Per-node only: σ_A[j]
            self.log_sigma_A = nn.Parameter(
                torch.full((num_nodes,), math.log(init_sigma))
            )
        
        # Clamping bounds for numerical stability
        self.log_sigma_min = -10.0  # σ >= exp(-10) ≈ 4.5e-5
        self.log_sigma_max = 2.0    # σ <= exp(2) ≈ 7.4
    
    @property
    def sigma_A(self) -> torch.Tensor:
        """Returns the ambient noise standard deviation (clamped for stability)."""
        log_sigma_clamped = self.log_sigma_A.clamp(self.log_sigma_min, self.log_sigma_max)
        return torch.exp(log_sigma_clamped)
    
    def forward(self, H_det: torch.Tensor, inject_noise: bool = True) -> torch.Tensor:
        """
        Add ambient noise to deterministic hidden representation.
        
        H = H_det + σ_A * ε, where ε ~ N(0, I)
        
        Args:
            H_det: (B, L, d_model) deterministic hidden representation from cross-attention
            inject_noise: If False, return H_det unchanged (useful for deterministic inference)
            
        Returns:
            H: (B, L, d_model) noisy hidden representation
        """
        if not inject_noise or not self.training:
            # No noise at inference (deterministic mean) unless explicitly requested
            return H_det
        
        # Get noise scale
        sigma_A = self.sigma_A  # (L,) or (L, d_model)
        
        # Sample noise
        eps = torch.randn_like(H_det)  # (B, L, d_model)
        
        # Apply noise based on parameterization
        if self.per_dimension:
            # Per-node per-dimension: σ_A[j, d] * ε[b, j, d]
            # sigma_A shape: (L, d_model) -> expand to (1, L, d_model)
            noise = sigma_A.unsqueeze(0) * eps
        else:
            # Per-node only: σ_A[j] * ε[b, j, d]
            # sigma_A shape: (L,) -> expand to (1, L, 1)
            noise = sigma_A.unsqueeze(0).unsqueeze(-1) * eps
        
        return H_det + noise
    
    def get_variance_contribution(self) -> torch.Tensor:
        """
        Returns the variance contribution from ambient noise per node.
        
        This is σ_A² for each node, useful for variance propagation analysis.
        
        Returns:
            torch.Tensor: Shape (num_nodes,) variance per node
        """
        sigma = self.sigma_A
        if self.per_dimension:
            # Average over dimensions
            return (sigma ** 2).mean(dim=-1)
        return sigma ** 2
    
    def __repr__(self):
        return (f"AmbientNoiseLayer(num_nodes={self.num_nodes}, "
                f"per_dimension={self.per_dimension}, "
                f"sigma_A_mean={self.sigma_A.mean().item():.4f})")


class ReadingNoiseHead(nn.Module):
    """
    Probabilistic output head with per-node reading (measurement) noise.
    
    This head predicts a Gaussian distribution for each output:
        X_i ~ N(μ_i, τ_i²)
    
    Where:
        μ_i = g_μ(U_i)  - learned mean projection
        τ_i² = σ_R[i]²  - per-node reading noise variance
    
    The reading noise models sensor measurement uncertainty, separate from
    the ambient noise that propagates through the causal structure.
    
    Note: The total predictive variance includes both propagated ambient noise
    and local reading noise. The model learns to separate these contributions.
    
    Args:
        d_model: Input dimension from transformer
        num_nodes: Number of output nodes (X_seq_len)
        out_dim: Output dimension per node (typically 1 for scalar values)
        init_sigma_R: Initial reading noise σ_R (default 0.05)
        learn_variance: If True, predict heteroscedastic variance. If False, use fixed σ_R.
        
    Attributes:
        head_mu: Linear projection for mean prediction
        log_sigma_R: Learnable log-scale reading noise per node
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_nodes: int, 
        out_dim: int = 1,
        init_sigma_R: float = 0.05,
        learn_variance: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.learn_variance = learn_variance
        
        # Mean prediction head
        self.head_mu = nn.Linear(d_model, out_dim, bias=False)
        
        # Per-node reading noise: σ_R[i]
        # Initialize with log(init_sigma_R) for positivity via exp()
        self.log_sigma_R = nn.Parameter(
            torch.full((num_nodes,), math.log(init_sigma_R)),
            requires_grad=learn_variance
        )
        
        # Variance clamping for numerical stability (applied to log_var)
        self.log_var_min = -10.0  # τ² >= exp(-10) ≈ 4.5e-5
        self.log_var_max = 5.0    # τ² <= exp(5) ≈ 148
    
    @property
    def sigma_R(self) -> torch.Tensor:
        """Returns the reading noise standard deviation (clamped for stability)."""
        log_sigma_clamped = self.log_sigma_R.clamp(
            self.log_var_min / 2, self.log_var_max / 2
        )
        return torch.exp(log_sigma_clamped)
    
    def forward(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict distribution parameters from aggregated representation.
        
        Args:
            U: (B, L, d_model) aggregated representation after self-attention mixing
            
        Returns:
            mu: (B, L, out_dim) predicted mean
            log_var: (B, L, out_dim) clamped log-variance (log(τ²))
        """
        # Mean prediction
        mu = self.head_mu(U)  # (B, L, out_dim)
        
        # Reading noise contribution to variance: τ² = σ_R²
        # log_var = log(σ_R²) = 2 * log(σ_R)
        log_var = 2 * self.log_sigma_R  # (L,)
        
        # Expand to match mu shape: (B, L, out_dim)
        log_var = log_var.unsqueeze(0).unsqueeze(-1).expand_as(mu)
        
        # Clamp for numerical stability
        log_var = log_var.clamp(self.log_var_min, self.log_var_max)
        
        return mu, log_var
    
    def get_variance(self) -> torch.Tensor:
        """Returns τ² = σ_R² for each node."""
        return self.sigma_R ** 2
    
    def sample(self, U: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the predictive distribution.
        
        Args:
            U: (B, L, d_model) aggregated representation
            num_samples: Number of samples to draw
            
        Returns:
            samples: (num_samples, B, L, out_dim) samples from N(μ, τ²)
        """
        mu, log_var = self.forward(U)
        std = torch.exp(0.5 * log_var)
        
        # Sample: X ~ N(μ, τ²)
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(mu)
            samples.append(mu + std * eps)
        
        return torch.stack(samples, dim=0)
    
    def __repr__(self):
        return (f"ReadingNoiseHead(d_model={self.d_model}, num_nodes={self.num_nodes}, "
                f"out_dim={self.out_dim}, sigma_R_mean={self.sigma_R.mean().item():.4f})")


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss with numerical stability.
    
    Computes:
        L = (x - μ)² / (2τ²) + log(τ)
          = (x - μ)² / (2 * exp(log_var)) + 0.5 * log_var
    
    The log(τ) term naturally penalizes unnecessarily large variance,
    preventing the model from explaining everything as noise.
    
    Args:
        eps: Small constant for numerical stability (default 1e-6)
        reduction: 'none', 'mean', or 'sum'
        full: If True, include constant term (0.5 * log(2π))
    """
    
    def __init__(
        self, 
        eps: float = 1e-6, 
        reduction: str = 'mean',
        full: bool = False
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.full = full
    
    def forward(
        self, 
        mu: torch.Tensor, 
        target: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss.
        
        Args:
            mu: (B, ...) predicted mean
            target: (B, ...) target values
            log_var: (B, ...) log-variance (log(τ²))
            
        Returns:
            loss: Scalar or per-element loss depending on reduction
        """
        # Ensure variance is positive with eps clamping
        var = torch.exp(log_var).clamp(min=self.eps)
        
        # NLL = (x - μ)² / (2τ²) + 0.5 * log(τ²)
        #     = 0.5 * [(x - μ)² / τ² + log(τ²)]
        mse = (target - mu) ** 2
        nll = 0.5 * (mse / var + log_var)
        
        if self.full:
            # Add constant: 0.5 * log(2π)
            nll = nll + 0.5 * math.log(2 * math.pi)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class VariancePropagationTracker(nn.Module):
    """
    Tracks variance propagation through the causal structure.
    
    For causal discovery, downstream nodes should inherit upstream variance:
        Var(X_i) = Σ_j α_ij² σ_A[j]² + σ_R[i]²
    
    This tracker computes and logs the variance contributions from different
    sources, which can provide statistical signal for causal direction.
    
    Args:
        num_nodes: Number of nodes in the graph
    """
    
    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Buffers for tracking (not parameters)
        self.register_buffer('propagated_var', torch.zeros(num_nodes))
        self.register_buffer('local_var', torch.zeros(num_nodes))
    
    def update(
        self, 
        attention_weights: torch.Tensor, 
        sigma_A_squared: torch.Tensor,
        sigma_R_squared: torch.Tensor
    ):
        """
        Update variance tracking based on attention weights.
        
        Args:
            attention_weights: (B, L, S) or (B, H, L, S) attention weights
            sigma_A_squared: (S,) ambient noise variance per source node
            sigma_R_squared: (L,) reading noise variance per target node
        """
        # Handle multi-head by averaging
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=1)  # (B, L, S)
        else:
            attn = attention_weights  # (B, L, S)
        
        # Average over batch
        attn_mean = attn.mean(dim=0)  # (L, S)
        
        # Propagated variance: Σ_j α_ij² σ_A[j]²
        propagated = (attn_mean ** 2 @ sigma_A_squared)  # (L,)
        
        with torch.no_grad():
            self.propagated_var.copy_(propagated)
            self.local_var.copy_(sigma_R_squared)
    
    def get_total_variance(self) -> torch.Tensor:
        """Returns total variance per node: propagated + local."""
        return self.propagated_var + self.local_var
    
    def get_variance_ratio(self) -> torch.Tensor:
        """Returns ratio of propagated to total variance per node."""
        total = self.get_total_variance()
        return self.propagated_var / (total + 1e-8)


# Utility functions

def softplus_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Inverse of softplus function.
    
    softplus(y) = log(1 + exp(β*y)) / β
    softplus_inverse(x) = log(exp(β*x) - 1) / β
    
    Useful for initializing parameters that will be passed through softplus.
    """
    return torch.log(torch.exp(beta * x) - 1) / beta


def init_noise_params(
    num_nodes: int,
    sigma_A_init: float = 0.01,
    sigma_R_init: float = 0.05,
    use_softplus: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize noise parameters for ambient and reading noise.
    
    Args:
        num_nodes: Number of nodes
        sigma_A_init: Initial ambient noise standard deviation
        sigma_R_init: Initial reading noise standard deviation
        use_softplus: If True, return params for softplus. If False, for exp.
        
    Returns:
        Tuple of (log_sigma_A, log_sigma_R) initial parameters
    """
    if use_softplus:
        # For softplus: σ = softplus(param) = log(1 + exp(param))
        log_sigma_A = softplus_inverse(torch.full((num_nodes,), sigma_A_init))
        log_sigma_R = softplus_inverse(torch.full((num_nodes,), sigma_R_init))
    else:
        # For exp: σ = exp(param)
        log_sigma_A = torch.full((num_nodes,), math.log(sigma_A_init))
        log_sigma_R = torch.full((num_nodes,), math.log(sigma_R_init))
    
    return log_sigma_A, log_sigma_R
