"""
HSIC (Hilbert-Schmidt Independence Criterion) utilities for causal learning.

This module provides differentiable HSIC computation for use in training
regularization, encouraging independence between inputs and residuals.

HSIC measures statistical dependence: 0 = independent, higher = more dependent.
Uses RBF (Gaussian) kernel for computing kernel matrices.

Supports **adaptive bandwidth** via the median heuristic (Gretton et al., 2012):
    σ = median( ||x_i − x_j|| )  over all pairs i < j
This ensures the RBF kernel stays well-conditioned even as residuals shrink
during training, preventing the kernel matrix from collapsing to all-ones.
"""

import torch


def _median_bandwidth(x: torch.Tensor) -> torch.Tensor:
    """
    Compute kernel bandwidth using the median heuristic.
    
    σ = median( ||x_i − x_j|| )  for all i < j
    
    This is the standard data-driven bandwidth selection for RBF kernels
    (Gretton et al., 2012 - "A Kernel Two-Sample Test").
    
    The median heuristic ensures the kernel matrix is neither too peaked
    (all off-diagonal entries ≈ 0) nor too flat (all entries ≈ 1),
    keeping the HSIC estimator well-conditioned regardless of data scale.
    
    Args:
        x: Input tensor of shape (n,) - 1D vector of n samples
        
    Returns:
        Scalar bandwidth σ (detached, no gradient flow through bandwidth)
    """
    x_col = x.detach().unsqueeze(1)  # (n, 1)
    dists = torch.abs(x_col - x_col.T)  # (n, n) pairwise absolute distances
    
    # Extract upper triangle (i < j) to avoid zero self-distances
    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
    pairwise_dists = dists[mask]
    
    if pairwise_dists.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    
    med = pairwise_dists.median()
    
    # Clamp to avoid degenerate bandwidth (numerical safety)
    return torch.clamp(med, min=1e-5)


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
    adaptive_bandwidth: bool = False,
) -> torch.Tensor:
    """
    Compute differentiable HSIC (Hilbert-Schmidt Independence Criterion).
    
    HSIC measures non-linear statistical dependence between two variables.
    HSIC = 0 if and only if X and Y are independent.
    
    Uses the biased estimator: HSIC = (1/(n-1)^2) * tr(KHLH)
    where K, L are kernel matrices and H is the centering matrix.
    
    When ``adaptive_bandwidth=True``, the ``sigma`` argument is ignored and
    the bandwidth is computed separately for each variable using the median
    heuristic (Gretton et al., 2012).  This makes the HSIC scale-invariant:
    as residuals shrink during training, the kernel automatically adapts,
    preventing artificial HSIC collapse.
    
    Args:
        x: First variable tensor of shape (n,) - 1D vector of n samples
        y: Second variable tensor of shape (n,) - 1D vector of n samples
        sigma: RBF kernel bandwidth (used when adaptive_bandwidth=False)
        adaptive_bandwidth: If True, use median heuristic for bandwidth
            selection (separate σ_x, σ_y). The ``sigma`` arg is ignored.
        
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
        
        >>> # Adaptive bandwidth — scale-invariant
        >>> hsic_val_adaptive = hsic(x, y_dep, adaptive_bandwidth=True)
    """
    n = len(x)
    
    # Compute kernel matrices (with per-variable bandwidth when adaptive)
    if adaptive_bandwidth:
        sigma_x = _median_bandwidth(x)
        sigma_y = _median_bandwidth(y)
        K = rbf_kernel(x, sigma_x)  # (n, n)
        L = rbf_kernel(y, sigma_y)  # (n, n)
    else:
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
    adaptive_bandwidth: bool = False,
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
        sigma: RBF kernel bandwidth (ignored when adaptive_bandwidth=True)
        adaptive_bandwidth: If True, use median heuristic per variable pair
        
    Returns:
        Mean HSIC across all token positions (scalar)
    """
    batch_size, seq_len_s = s_values.shape
    
    # Compute HSIC for each token position
    hsic_values = []
    for i in range(seq_len_s):
        s_token = s_values[:, i]  # (batch,)
        hsic_i = hsic(s_token, residuals, sigma=sigma, adaptive_bandwidth=adaptive_bandwidth)
        hsic_values.append(hsic_i)
    
    # Return mean across all positions
    return torch.stack(hsic_values).mean()


def hsic_per_x_pair(
    x_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
) -> torch.Tensor:
    """
    Compute HSIC between X values and per-X residuals for self-attention DAG validation.
    
    For each pair (i, j) where i != j, computes HSIC(X_j, residual_i).
    This measures whether the residual for X_i is independent of X_j,
    which is relevant for self-attention DAG learning.
    
    If the true DAG has X_j → X_i, then X_i's residual should be independent of X_j
    (the parent is properly accounted for). If the model learns the wrong direction,
    X_i's residual will still depend on X_j.
    
    Args:
        x_values: X variable values of shape (batch, seq_len_x)
        residuals: Per-X residuals of shape (batch, seq_len_x) - i.e., x_target - x_pred
        sigma: RBF kernel bandwidth (ignored when adaptive_bandwidth=True)
        adaptive_bandwidth: If True, use median heuristic per variable pair
        
    Returns:
        Mean HSIC across all (i, j) pairs where i != j (scalar)
        
    Example:
        >>> x_values = torch.randn(100, 3)  # 3 X variables
        >>> residuals = torch.randn(100, 3)  # Per-variable residuals
        >>> hsic_x = hsic_per_x_pair(x_values, residuals, sigma=1.0)
        >>> # Computes HSIC for pairs: (X_1, res_0), (X_2, res_0), (X_0, res_1), etc.
    """
    batch_size, seq_len_x = x_values.shape
    
    hsic_values = []
    for i in range(seq_len_x):
        # Residual for X_i
        res_i = residuals[:, i]  # (batch,)
        
        # Compute HSIC against all OTHER X values
        for j in range(seq_len_x):
            if i != j:
                x_j = x_values[:, j]  # (batch,)
                hsic_ij = hsic(x_j, res_i, sigma=sigma, adaptive_bandwidth=adaptive_bandwidth)
                hsic_values.append(hsic_ij)
    
    # Return mean across all (i, j) pairs
    if len(hsic_values) == 0:
        # Edge case: single X variable, no pairs
        return torch.tensor(0.0, device=x_values.device, dtype=x_values.dtype)
    
    return torch.stack(hsic_values).mean()


def hsic_attention_weighted(
    source_values: torch.Tensor,
    residuals: torch.Tensor,
    attention_weights: torch.Tensor,
    sigma: float = 1.0,
    exclude_diagonal: bool = False,
    adaptive_bandwidth: bool = False,
) -> torch.Tensor:
    """
    Attention-weighted HSIC for causal structure regularization.
    
    Computes: sum(att[i,j] * HSIC(source_j, residual_i)) / sum(att)
    
    The attention weight acts as a "confidence" factor: the model is penalized
    proportionally to how much it relies on each edge. High penalty occurs when:
    - The model strongly attends to a source (high att[i,j])
    - But the residual still depends on that source (high HSIC)
    
    For self-attention (X→X):
        - source_values = X values (batch, seq_len_x)
        - residuals = per-X residuals (batch, seq_len_x)
        - attention_weights = self-attention (seq_len_x, seq_len_x)
        - exclude_diagonal = True (X_i shouldn't attend to itself)
        
    For cross-attention (S→X):
        - source_values = S values (batch, seq_len_s)
        - residuals = per-X residuals (batch, seq_len_x)
        - attention_weights = cross-attention (seq_len_x, seq_len_s)
        - exclude_diagonal = False (no diagonal concept)
    
    Args:
        source_values: Source variable values (batch, seq_len_source) - S or X values
        residuals: Per-target residuals (batch, seq_len_target)
        attention_weights: Attention matrix (seq_len_target, seq_len_source) - averaged over batch
        sigma: RBF kernel bandwidth (ignored when adaptive_bandwidth=True)
        exclude_diagonal: If True, skip diagonal entries (for self-attention)
        adaptive_bandwidth: If True, use median heuristic per variable pair
        
    Returns:
        Normalized attention-weighted HSIC (scalar)
    """
    seq_len_target = residuals.shape[1]
    seq_len_source = source_values.shape[1]
    
    weighted_hsic_sum = torch.tensor(0.0, device=source_values.device, dtype=source_values.dtype)
    weight_sum = torch.tensor(0.0, device=source_values.device, dtype=source_values.dtype)
    
    for i in range(seq_len_target):
        res_i = residuals[:, i]  # (batch,)
        
        for j in range(seq_len_source):
            # Skip diagonal for self-attention
            if exclude_diagonal and i == j:
                continue
            
            source_j = source_values[:, j]  # (batch,)
            weight_ij = attention_weights[i, j]  # scalar
            
            hsic_ij = hsic(source_j, res_i, sigma=sigma, adaptive_bandwidth=adaptive_bandwidth)
            weighted_hsic_sum = weighted_hsic_sum + weight_ij * hsic_ij
            weight_sum = weight_sum + weight_ij
    
    # Normalize by total attention weight (avoid division by zero)
    if weight_sum > 1e-8:
        return weighted_hsic_sum / weight_sum
    else:
        return torch.tensor(0.0, device=source_values.device, dtype=source_values.dtype)
