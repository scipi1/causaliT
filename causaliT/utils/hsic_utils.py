"""
HSIC (Hilbert-Schmidt Independence Criterion) utilities for causal learning.

This module provides differentiable HSIC computation for use in training
regularization, encouraging independence between inputs and residuals.

HSIC measures statistical dependence: 0 = independent, higher = more dependent.
Uses RBF (Gaussian) kernel for computing kernel matrices.
"""

import torch


def rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel matrix.
    
    K(i,j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    
    Args:
        x: Input tensor of shape (n,) - 1D vector of n samples
        sigma: Kernel bandwidth parameter
        
    Returns:
        Kernel matrix of shape (n, n)
    """
    x = x.unsqueeze(1)  # (n, 1)
    dists_sq = (x - x.T) ** 2  # (n, n) pairwise squared distances
    return torch.exp(-dists_sq / (2 * sigma ** 2))


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute differentiable HSIC (Hilbert-Schmidt Independence Criterion).
    
    HSIC measures non-linear statistical dependence between two variables.
    HSIC = 0 if and only if X and Y are independent.
    
    Uses the biased estimator: HSIC = (1/(n-1)^2) * tr(KHLH)
    where K, L are kernel matrices and H is the centering matrix.
    
    Args:
        x: First variable tensor of shape (n,) - 1D vector of n samples
        y: Second variable tensor of shape (n,) - 1D vector of n samples
        sigma: RBF kernel bandwidth (same for both x and y)
        
    Returns:
        Scalar HSIC value (differentiable)
        
    Example:
        >>> x = torch.randn(100)
        >>> y = torch.randn(100)  # Independent
        >>> hsic_val = hsic(x, y, sigma=1.0)
        >>> # hsic_val should be close to 0
        
        >>> y_dep = x + 0.1 * torch.randn(100)  # Dependent
        >>> hsic_val_dep = hsic(x, y_dep, sigma=1.0)
        >>> # hsic_val_dep should be > 0
    """
    n = len(x)
    
    # Compute kernel matrices
    K = rbf_kernel(x, sigma)  # (n, n)
    L = rbf_kernel(y, sigma)  # (n, n)
    
    # Centering matrix H = I - (1/n) * 1*1^T
    H = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(n, n, device=x.device, dtype=x.dtype) / n
    
    # Centered kernels
    KH = K @ H
    LH = L @ H
    
    # HSIC = (1/(n-1)^2) * tr(KH @ LH)
    # Efficient computation: tr(A @ B) = sum(A * B.T)
    hsic_value = (KH * LH.T).sum() / ((n - 1) ** 2)
    
    return hsic_value


def hsic_per_token(
    s_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute HSIC between each token position in S and the mean residuals.
    
    This function computes HSIC(S_i, mean_residuals) for each token position i,
    measuring how much information from each source token is NOT captured by
    the model (remaining in residuals).
    
    Lower HSIC values indicate better causal structure learning - the model
    has successfully captured the causal relationship from S to X.
    
    Args:
        s_values: Source values tensor of shape (batch, seq_len_s)
        residuals: Mean residuals tensor of shape (batch,)
        sigma: RBF kernel bandwidth
        
    Returns:
        Mean HSIC across all token positions (scalar)
    """
    batch_size, seq_len_s = s_values.shape
    
    # Compute HSIC for each token position
    hsic_values = []
    for i in range(seq_len_s):
        s_token = s_values[:, i]  # (batch,)
        hsic_i = hsic(s_token, residuals, sigma=sigma)
        hsic_values.append(hsic_i)
    
    # Return mean across all positions
    return torch.stack(hsic_values).mean()
