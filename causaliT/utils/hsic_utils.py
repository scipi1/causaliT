"""
HSIC (Hilbert-Schmidt Independence Criterion) utilities for causal learning.

This module provides differentiable HSIC computation for use in training
regularization, encouraging independence between inputs and residuals.

HSIC measures statistical dependence: 0 = independent, higher = more dependent.
Uses RBF (Gaussian) kernel for computing kernel matrices.

Supports two HSIC estimators:

1. **Biased HSIC** (default): (1/(n-1)^2) * tr(KHLH)
   Standard estimator. With adaptive bandwidth, tends to create a noise floor
   because the bandwidth self-normalizes.

2. **Normalized HSIC (nHSIC)** (Ma et al., AAAI 2020): tr(K̃ · L̃)
   where K̃ = K̄(K̄ + mεI)^{-1}, K̄ = HKH (centered kernel).
   Uses Tikhonov regularization to damp eigenvalues, reducing noise sensitivity.
   Better behaved at small sample sizes and less susceptible to the adaptive
   bandwidth floor effect.

Supports **adaptive bandwidth** via the median heuristic (Gretton et al., 2012):
    σ = median( ||x_i − x_j|| )  over all pairs i < j
This ensures the RBF kernel stays well-conditioned even as residuals shrink
during training, preventing the kernel matrix from collapsing to all-ones.

Config options:
    hsic_mode: "biased" | "normalized"  (default: "biased")
    nhsic_epsilon: float  (regularization for nHSIC, default: 0.01)
    hsic_bandwidth_multipliers: optional positive scales for multi-bandwidth RBF
        (e.g. [0.5, 1.0, 2.0] applied to the fixed/median base bandwidth)
"""

from typing import Optional, Sequence

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


def _validate_bandwidth_multipliers(
    bandwidth_multipliers: Optional[Sequence[float]],
) -> Optional[tuple]:
    """Validate multipliers applied to the base RBF bandwidth.

    ``None`` disables the multiscale path and preserves the legacy kernel.
    An empty sequence is treated the same way, which is convenient for configs.
    """
    if bandwidth_multipliers is None:
        return None
    multipliers = tuple(float(m) for m in bandwidth_multipliers)
    if not multipliers:
        return None
    if any(m <= 0.0 for m in multipliers):
        raise ValueError(
            "hsic bandwidth multipliers must all be positive, got "
            f"{multipliers}"
        )
    return multipliers


def rbf_multiscale_kernel(
    x: torch.Tensor,
    sigma: float,
    bandwidth_multipliers: Sequence[float],
) -> torch.Tensor:
    """Mean of RBF kernels evaluated at several scaled bandwidths.

    Each component has unit diagonal, so the averaged kernel also has unit
    diagonal.  The multipliers scale the base bandwidth (fixed ``sigma`` or
    the median heuristic), giving sensitivity to both narrower and broader
    dependence structure without changing the HSIC estimator itself.
    """
    multipliers = _validate_bandwidth_multipliers(bandwidth_multipliers)
    if multipliers is None:
        return rbf_kernel(x, sigma)
    kernels = [rbf_kernel(x, sigma * m) for m in multipliers]
    return torch.stack(kernels, dim=0).mean(dim=0)


def dirac_kernel(x: torch.Tensor, tolerance: float = 1e-5) -> torch.Tensor:
    """
    Compute Dirac delta kernel matrix for discrete/categorical variables.

    K(i,j) = 1  if |x_i - x_j| < tolerance
             0  otherwise

    This is the natural kernel for discrete/categorical data where the RBF
    kernel may not provide sufficient resolution. For discrete S variables
    with few levels (3-11), the RBF kernel matrix has very low effective rank
    and HSIC loses sensitivity to dependence structure.

    The Dirac kernel is a valid positive definite kernel (it's the inner product
    in the feature space where each discrete value maps to a one-hot vector).

    Args:
        x: Input tensor of shape (n,) - 1D vector of n samples with discrete values
        tolerance: Floating-point tolerance for equality comparison

    Returns:
        Kernel matrix of shape (n, n) with binary entries
    """
    x_col = x.detach().unsqueeze(1)  # (n, 1) — detach like _median_bandwidth
    dists = torch.abs(x_col - x_col.T)  # (n, n)
    # Use sigmoid approximation for differentiability through the kernel
    # Sharp sigmoid: σ(-(d - tol) * sharpness) ≈ step function
    # But for HSIC, we only need gradient through the *other* variable's kernel
    # (S kernel is fixed, gradient flows through residual kernel L)
    # So we can use hard threshold safely:
    return (dists < tolerance).float()


def hsic_from_kernels(
    K: torch.Tensor,
    L: torch.Tensor,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Compute HSIC from pre-computed kernel matrices.

    This allows mixing different kernel types (e.g., Dirac for discrete S,
    RBF for continuous residuals). The centering and HSIC formula are
    applied directly to the given kernel matrices.

    Args:
        K: Kernel matrix for first variable, shape (n, n)
        L: Kernel matrix for second variable, shape (n, n)
        mode: "biased" (standard HSIC) or "normalized" (nHSIC)
        nhsic_epsilon: Regularization for nHSIC (default 0.01)

    Returns:
        Scalar HSIC value (differentiable through L; K is typically detached)
    """
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - torch.ones(n, n, device=K.device, dtype=K.dtype) / n

    if mode == "normalized":
        K_bar = H @ K @ H
        L_bar = H @ L @ H
        reg = n * nhsic_epsilon * torch.eye(n, device=K.device, dtype=K.dtype)
        K_tilde = K_bar @ torch.linalg.solve(K_bar + reg, torch.eye(n, device=K.device, dtype=K.dtype))
        L_tilde = L_bar @ torch.linalg.solve(L_bar + reg, torch.eye(n, device=K.device, dtype=K.dtype))
        return (K_tilde * L_tilde.T).sum()
    else:
        KH = K @ H
        LH = L @ H
        return (KH * LH.T).sum() / ((n - 1) ** 2)


def _compute_kernel_matrices(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    bandwidth_multipliers: Optional[Sequence[float]] = None,
) -> tuple:
    """
    Compute RBF kernel matrices for x and y.

    Shared helper that handles adaptive bandwidth selection and the optional
    multiscale RBF mixture.

    Args:
        x, y: 1D tensors of shape (n,)
        sigma: Fixed bandwidth (used when adaptive_bandwidth=False)
        adaptive_bandwidth: If True, use median heuristic per variable
        bandwidth_multipliers: Optional positive scales applied to the base
            bandwidth.  None/empty preserves the legacy single-bandwidth RBF.

    Returns:
        (K, L): Kernel matrices of shape (n, n) each
    """
    multipliers = _validate_bandwidth_multipliers(bandwidth_multipliers)
    if adaptive_bandwidth:
        sigma_x = _median_bandwidth(x)
        sigma_y = _median_bandwidth(y)
    elif isinstance(sigma, (list, tuple)):
        # Frozen per-side bandwidths, e.g. latched at the start of a training
        # stage: (sigma_source, sigma_residual).  Freezing restores the
        # residual-magnitude channel of the HSIC gradient that the per-batch
        # median heuristic divides out (RBF + median heuristic is exactly
        # scale-equivariant).
        sigma_x, sigma_y = sigma[0], sigma[1]
    else:
        sigma_x = sigma_y = sigma

    if multipliers is None:
        K = rbf_kernel(x, sigma_x)
        L = rbf_kernel(y, sigma_y)
    else:
        K = rbf_multiscale_kernel(x, sigma_x, multipliers)
        L = rbf_multiscale_kernel(y, sigma_y, multipliers)
    return K, L


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    bandwidth_multipliers: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Compute differentiable HSIC (Hilbert-Schmidt Independence Criterion).

    HSIC measures non-linear statistical dependence between two variables.
    HSIC = 0 if and only if X and Y are independent.

    Supports two modes:

    **"biased"** (default): Standard biased estimator.
        HSIC = (1/(n-1)^2) * tr(KHLH)
        where K, L are kernel matrices and H is the centering matrix.

    **"normalized"**: Normalized HSIC (nHSIC) from Ma et al., AAAI 2020.
        nHSIC = tr(K̃ · L̃)
        where K̃ = K̄(K̄ + nεI)^{-1}, K̄ = HKH.
        The Tikhonov regularization (nεI) damps eigenvalues, making the
        statistic more stable at small sample sizes and reducing the
        noise floor caused by adaptive bandwidth.

    When ``adaptive_bandwidth=True``, the ``sigma`` argument is ignored and
    the bandwidth is computed separately for each variable using the median
    heuristic (Gretton et al., 2012).

    Args:
        x: First variable tensor of shape (n,) - 1D vector of n samples
        y: Second variable tensor of shape (n,) - 1D vector of n samples
        sigma: RBF kernel bandwidth (used when adaptive_bandwidth=False)
        adaptive_bandwidth: If True, use median heuristic for bandwidth
            selection (separate σ_x, σ_y). The ``sigma`` arg is ignored.
        mode: "biased" (standard HSIC) or "normalized" (nHSIC).
        nhsic_epsilon: Regularization constant for nHSIC (default 0.01).
            Only used when mode="normalized".

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

        >>> # Normalized HSIC — better behaved at small batch sizes
        >>> nhsic_val = hsic(x, y_dep, mode="normalized", adaptive_bandwidth=True)
    """
    n = len(x)
    K, L = _compute_kernel_matrices(
        x, y, sigma, adaptive_bandwidth, bandwidth_multipliers
    )

    # Centering matrix H = I - (1/n) * 1*1^T
    H = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(n, n, device=x.device, dtype=x.dtype) / n

    if mode == "normalized":
        # Normalized HSIC (Ma et al., AAAI 2020, Eq. 5)
        # K̄ = HKH (centered kernel)
        K_bar = H @ K @ H
        L_bar = H @ L @ H

        # K̃ = K̄ (K̄ + nεI)^{-1}  — Tikhonov-regularized normalized kernel
        reg = n * nhsic_epsilon * torch.eye(n, device=x.device, dtype=x.dtype)
        K_tilde = K_bar @ torch.linalg.solve(K_bar + reg, torch.eye(n, device=x.device, dtype=x.dtype))
        L_tilde = L_bar @ torch.linalg.solve(L_bar + reg, torch.eye(n, device=x.device, dtype=x.dtype))

        # nHSIC = tr(K̃ · L̃)
        # Efficient: tr(AB) = sum(A * B.T)
        nhsic_value = (K_tilde * L_tilde.T).sum()
        return nhsic_value
    else:
        # Standard biased HSIC
        KH = K @ H
        LH = L @ H

        # HSIC = (1/(n-1)^2) * tr(KH @ LH)
        hsic_value = (KH * LH.T).sum() / ((n - 1) ** 2)
        return hsic_value


def _compute_cross_hsic_pair(
    s_i: torch.Tensor,
    res_j: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Compute HSIC for a single (source, residual) pair with kernel selection.

    When source_kernel="dirac", uses Dirac kernel for S (discrete) and RBF
    for residuals (continuous), computing HSIC via hsic_from_kernels().
    When source_kernel="rbf" (default), uses standard RBF for both.

    Args:
        s_i: Source variable values (n,)
        res_j: Residual values (n,)
        sigma: RBF bandwidth (ignored when adaptive_bandwidth=True)
        adaptive_bandwidth: If True, use median heuristic for residual bandwidth
        mode: "biased" or "normalized"
        nhsic_epsilon: Regularization for nHSIC
        source_kernel: "rbf" (default) or "dirac" (for discrete S)

    Returns:
        Scalar HSIC value
    """
    if source_kernel == "dirac":
        # Dirac kernel for S (discrete), RBF for residuals (continuous)
        K = dirac_kernel(s_i)
        if adaptive_bandwidth:
            sigma_res = _median_bandwidth(res_j)
        elif isinstance(sigma, (list, tuple)):
            sigma_res = sigma[1]
        else:
            sigma_res = sigma
        L = rbf_multiscale_kernel(res_j, sigma_res, bandwidth_multipliers)
        return hsic_from_kernels(K, L, mode=mode, nhsic_epsilon=nhsic_epsilon)
    else:
        # Standard: RBF for both
        return hsic(s_i, res_j, sigma=sigma, adaptive_bandwidth=adaptive_bandwidth,
                    mode=mode, nhsic_epsilon=nhsic_epsilon,
                    bandwidth_multipliers=bandwidth_multipliers)


def hsic_pair_matrix(
    source_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    exclude_diagonal: bool = False,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute pairwise residual-HSIC for all target/source variable pairs.

    This is the matrix-valued counterpart of the scalar HSIC regularizers used
    during training.  It reuses the same kernels and estimators via
    ``_compute_cross_hsic_pair``:

    - RBF or Dirac source kernels for S→X/cross diagnostics.
    - RBF source kernels for X→X/self diagnostics.
    - Biased or normalized HSIC, including adaptive bandwidth.

    Args:
        source_values: Candidate parent values, shape ``(batch, L_source)``.
        residuals: Per-target residuals, shape ``(batch, L_target)``.
        sigma: Fixed RBF bandwidth when ``adaptive_bandwidth=False``.
        adaptive_bandwidth: If True, use the median heuristic per pair.
        mode: ``"biased"`` or ``"normalized"``.
        nhsic_epsilon: Tikhonov regularization for normalized HSIC.
        source_kernel: ``"rbf"`` or ``"dirac"`` for source values.
        exclude_diagonal: If True and the matrix is square, diagonal entries
            are set to ``NaN``. Useful for self-attention DAG summaries where
            self-loops are not valid candidate parents.
        pair_mask: Optional ``(L_target, L_source)`` tensor.  Pairs whose mask
            entry is exactly ``0`` are SKIPPED entirely (set to ``NaN``) instead
            of being computed, which saves the kernel work for those pairs.
            Non-zero entries are computed normally — the *weighting* itself is
            the caller's job (see :func:`hsic_cross_per_pair`).  Used by the
            descendant-exclusion mask (``causaliT.utils.descendant_mask``).

    Returns:
        Tensor of shape ``(L_target, L_source)`` where entry ``[j, i]`` is
        ``HSIC(source_i, residual_j)``.
    """
    seq_len_source = source_values.shape[1]
    seq_len_target = residuals.shape[1]

    if pair_mask is not None and tuple(pair_mask.shape) != (seq_len_target, seq_len_source):
        raise ValueError(
            f"pair_mask shape {tuple(pair_mask.shape)} does not match the HSIC "
            f"matrix shape {(seq_len_target, seq_len_source)}."
        )

    rows = []
    for j in range(seq_len_target):
        res_j = residuals[:, j]
        vals = []
        for i in range(seq_len_source):
            if exclude_diagonal and seq_len_source == seq_len_target and i == j:
                vals.append(torch.tensor(float("nan"), device=source_values.device, dtype=source_values.dtype))
                continue
            if pair_mask is not None and float(pair_mask[j, i]) == 0.0:
                # Fully excluded pair: skip the kernel computation entirely.
                vals.append(torch.tensor(float("nan"), device=source_values.device, dtype=source_values.dtype))
                continue
            source_i = source_values[:, i]
            # Per-variable frozen bandwidths: sigma may be (src, res) where each
            # side is a scalar or a 1D tensor/list indexed by variable.
            sig = sigma
            if isinstance(sigma, (list, tuple)) and not adaptive_bandwidth:
                s_src, s_res = sigma
                sig = (s_src[i] if torch.is_tensor(s_src) or isinstance(s_src, (list, tuple))
                       else s_src,
                       s_res[j] if torch.is_tensor(s_res) or isinstance(s_res, (list, tuple))
                       else s_res)
            vals.append(
                _compute_cross_hsic_pair(
                    source_i,
                    res_j,
                    sigma=sig,
                    adaptive_bandwidth=adaptive_bandwidth,
                    mode=mode,
                    nhsic_epsilon=nhsic_epsilon,
                    source_kernel=source_kernel,
                    bandwidth_multipliers=bandwidth_multipliers,
                )
            )
        rows.append(torch.stack(vals))

    if not rows:
        return torch.empty((seq_len_target, seq_len_source), device=source_values.device, dtype=source_values.dtype)
    return torch.stack(rows, dim=0)


def hsic_cross_per_pair(
    s_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute HSIC between each S variable and each X residual (per-pair).

    For each pair (i, j), computes HSIC(S_i, residual_j), measuring whether
    the residual for X_j still depends on source variable S_i.

    This provides **edge-level** gradient signal for cross-attention DAG learning,
    unlike hsic_per_token which averages residuals across X positions and loses
    per-edge structural information.

    If the true DAG has S_i → X_j, then X_j's residual should be independent
    of S_i when the model correctly captures the causal relationship.

    Args:
        s_values: Source variable values of shape (batch, seq_len_s)
        residuals: Per-X residuals of shape (batch, seq_len_x) - i.e., x_target - x_pred
        sigma: RBF kernel bandwidth (ignored when adaptive_bandwidth=True)
        adaptive_bandwidth: If True, use median heuristic per variable pair
        source_kernel: "rbf" (default) or "dirac" (for discrete S variables).
            When "dirac", uses Dirac delta kernel for S and RBF for residuals.
        pair_mask: Optional ``(seq_len_x, seq_len_s)`` per-pair WEIGHT matrix.
            When given, the result is the weighted mean
            ``sum(w * HSIC) / sum(w)`` instead of the plain mean.  Pairs with
            weight exactly ``0`` are skipped entirely (no kernel work).  The
            mask must be detached by the caller — see
            ``causaliT.utils.descendant_mask.build_hsic_pair_mask``, which
            produces the descendant-exclusion mask.  ``None`` (default)
            reproduces the plain unweighted mean exactly.

    Returns:
        Mean (or ``pair_mask``-weighted mean) HSIC across all (S_i, res_j) pairs
        (scalar)

    Example:
        >>> s_values = torch.randn(100, 5)   # 5 S variables
        >>> residuals = torch.randn(100, 5)   # 5 X residuals
        >>> hsic_val = hsic_cross_per_pair(s_values, residuals, adaptive_bandwidth=True)
        >>> # Computes HSIC for all 25 pairs: (S_0, res_0), (S_0, res_1), ...
        >>> # With discrete S:
        >>> hsic_val = hsic_cross_per_pair(s_values, residuals, source_kernel="dirac")
    """
    hsic_mat = hsic_pair_matrix(
        source_values=s_values,
        residuals=residuals,
        sigma=sigma,
        adaptive_bandwidth=adaptive_bandwidth,
        mode=mode,
        nhsic_epsilon=nhsic_epsilon,
        source_kernel=source_kernel,
        bandwidth_multipliers=bandwidth_multipliers,
        exclude_diagonal=False,
        pair_mask=pair_mask,
    )

    if hsic_mat.numel() == 0:
        return torch.tensor(0.0, device=s_values.device, dtype=s_values.dtype)

    if pair_mask is None:
        return hsic_mat.mean()

    # Weighted mean over the pairs that were actually computed.  Skipped pairs
    # (weight == 0) come back as NaN and are dropped here, so a zero weight and
    # a skipped computation agree exactly.
    weights = pair_mask.to(device=hsic_mat.device, dtype=hsic_mat.dtype)
    valid = ~torch.isnan(hsic_mat)
    weight_sum = weights[valid].sum()
    if float(weight_sum) <= 1e-12:
        # Everything excluded: no structural signal available this step.
        return torch.zeros((), device=s_values.device, dtype=s_values.dtype)
    return (weights[valid] * hsic_mat[valid]).sum() / weight_sum


def hsic_per_token(
    s_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
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
        source_kernel: "rbf" (default) or "dirac" (for discrete S variables)

    Returns:
        Mean HSIC across all token positions (scalar)
    """
    batch_size, seq_len_s = s_values.shape

    # Compute HSIC for each token position
    hsic_values = []
    for i in range(seq_len_s):
        s_token = s_values[:, i]  # (batch,)
        hsic_i = _compute_cross_hsic_pair(
            s_token, residuals, sigma=sigma, adaptive_bandwidth=adaptive_bandwidth,
            mode=mode, nhsic_epsilon=nhsic_epsilon, source_kernel=source_kernel,
            bandwidth_multipliers=bandwidth_multipliers)
        hsic_values.append(hsic_i)

    # Return mean across all positions
    return torch.stack(hsic_values).mean()


def hsic_per_x_pair(
    x_values: torch.Tensor,
    residuals: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    bandwidth_multipliers: Optional[Sequence[float]] = None,
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
    hsic_mat = hsic_pair_matrix(
        source_values=x_values,
        residuals=residuals,
        sigma=sigma,
        adaptive_bandwidth=adaptive_bandwidth,
        mode=mode,
        nhsic_epsilon=nhsic_epsilon,
        source_kernel="rbf",
        bandwidth_multipliers=bandwidth_multipliers,
        exclude_diagonal=True,
    )

    valid = hsic_mat[~torch.isnan(hsic_mat)]
    if valid.numel() == 0:
        # Edge case: single X variable, no pairs
        return torch.tensor(0.0, device=x_values.device, dtype=x_values.dtype)

    return valid.mean()


def hsic_attention_weighted(
    source_values: torch.Tensor,
    residuals: torch.Tensor,
    attention_weights: torch.Tensor,
    sigma: float = 1.0,
    exclude_diagonal: bool = False,
    adaptive_bandwidth: bool = False,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
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
        source_kernel: "rbf" (default) or "dirac" (for discrete S in cross-attention).
            Only relevant for cross-attention (exclude_diagonal=False).
            For self-attention, X is always continuous → always uses RBF.

    Returns:
        Normalized attention-weighted HSIC (scalar)
    """
    effective_source_kernel = source_kernel if not exclude_diagonal else "rbf"
    hsic_mat = hsic_pair_matrix(
        source_values=source_values,
        residuals=residuals,
        sigma=sigma,
        adaptive_bandwidth=adaptive_bandwidth,
        mode=mode,
        nhsic_epsilon=nhsic_epsilon,
        source_kernel=effective_source_kernel,
        bandwidth_multipliers=bandwidth_multipliers,
        exclude_diagonal=exclude_diagonal,
    )

    valid = ~torch.isnan(hsic_mat)
    weights = attention_weights.to(device=source_values.device, dtype=source_values.dtype)
    if weights.shape != hsic_mat.shape:
        raise ValueError(
            f"attention_weights shape {tuple(weights.shape)} does not match "
            f"HSIC matrix shape {tuple(hsic_mat.shape)}"
        )

    weighted_hsic_sum = (weights[valid] * hsic_mat[valid]).sum()
    weight_sum = weights[valid].sum()

    # Normalize by total attention weight (avoid division by zero)
    if weight_sum > 1e-8:
        return weighted_hsic_sum / weight_sum
    else:
        return torch.tensor(0.0, device=source_values.device, dtype=source_values.dtype)

# ---------------------------------------------------------------------------
# Null calibration for the LOO Bayes multiplier
# (see docs/ideas/CONDITIONAL_HSIC_COUNTERPROPOSAL.md, Section 3)
# ---------------------------------------------------------------------------


def hsic_null_calibration(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = True,
    n_permutations: int = 100,
    generator: Optional[torch.Generator] = None,
    chunk: int = 10,
) -> dict:
    """Biased HSIC with a permutation-calibrated gamma null.

    Computes the biased HSIC statistic for ``(x, y)`` and calibrates its null
    distribution empirically: ``y`` is permuted ``n_permutations`` times and
    the gamma null density is moment-matched to the permutation statistics.
    This replaces the hand-tuned ``exp(-lambda * H)`` pseudo-likelihood of the
    original proposal with a real density under the independence model.

    Only the biased estimator is supported: its permutation statistics are
    cheap (O(n^2) per permutation after the kernel matrices are built), while
    nHSIC would need one O(n^3) linear solve per permutation.

    Args:
        x, y: 1-D tensors of n samples (detached inside).
        sigma: RBF bandwidth (ignored when adaptive_bandwidth=True).
        adaptive_bandwidth: median-heuristic bandwidth (default True).
        n_permutations: number of permutation statistics for the null.
        generator: torch.Generator for reproducible permutations.
        chunk: permutation batch size (memory guard: chunk x n x n tensors).

    Returns:
        Dict with detached scalars:
        ``stat`` (HSIC value), ``p_empirical`` ((1 + #{perm >= stat}) / (S+1)),
        ``p_gamma`` (survival of the fitted gamma at stat),
        ``log_q0`` (log null density at stat — the "likelihood under
        independence" for the Bayes multiplier), ``alpha``, ``beta``
        (gamma shape / rate).
    """
    x = x.detach()
    y = y.detach()
    n = len(x)
    K, L = _compute_kernel_matrices(x, y, sigma, adaptive_bandwidth, None)
    H = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(
        n, n, device=x.device, dtype=x.dtype
    ) / n
    K_bar = H @ K @ H
    L_bar = H @ L @ H
    norm = (n - 1) ** 2
    stat = (K_bar * L_bar.T).sum() / norm

    # Permutation statistics: tr(K_bar @ pi L_bar pi^T) / norm, chunked.
    perm_stats = []
    remaining = n_permutations
    while remaining > 0:
        c = min(chunk, remaining)
        idx = torch.stack(
            [torch.randperm(n, generator=generator, device=x.device)
             for _ in range(c)]
        )  # (c, n)
        L_perm = L_bar[idx.unsqueeze(2), idx.unsqueeze(1)]  # (c, n, n)
        perm_stats.append((K_bar.unsqueeze(0) * L_perm).sum(dim=(-1, -2)) / norm)
        remaining -= c
    perm_stats = torch.cat(perm_stats)

    # Moment-matched gamma fit to the permutation null.
    mu = perm_stats.mean()
    var = perm_stats.var(unbiased=False).clamp_min(1e-20)
    alpha = (mu * mu / var).clamp_min(1e-6)          # shape
    beta = (mu / var).clamp_min(1e-12)               # rate

    stat_c = stat.clamp_min(1e-30)
    log_q0 = (
        (alpha - 1.0) * torch.log(stat_c)
        - beta * stat
        + alpha * torch.log(beta)
        - torch.lgamma(alpha)
    )
    p_gamma = torch.special.gammaincc(alpha, beta * stat).clamp(1e-12, 1.0)
    p_empirical = (1.0 + (perm_stats >= stat).sum()) / (n_permutations + 1.0)

    return {
        "stat": stat,
        "p_empirical": p_empirical,
        "p_gamma": p_gamma,
        "log_q0": log_q0,
        "alpha": alpha,
        "beta": beta,
    }


def bayes_multiplier(
    log_q0_plus: torch.Tensor,
    log_q0_minus: torch.Tensor,
    p_m: torch.Tensor,
) -> torch.Tensor:
    """Bayes edge multiplier gamma from calibrated null log-densities.

        gamma = q0(H+) * P_m / (q0(H+) * P_m + q0(H^{-i}) * (1 - P_m))

    computed in log-space (log-sum-exp).  All inputs are treated as constants
    (detached); the result carries no gradient.  ``p_m`` is clamped away from
    exactly 0/1 for the log transform, which leaves gamma numerically 0 resp.
    1 at the extremes.

    Semantics: gamma -> 1 when the edge is load-bearing (the masked residual
    is dependent, so the null density at H^{-i} vanishes); gamma -> P_m when
    the two worlds are indistinguishable (redundant edge — pruning is left to
    the sparsity prior, NOT to this multiplier).
    """
    p = p_m.detach().clamp(1e-8, 1.0 - 1e-8)
    log_num = log_q0_plus.detach() + torch.log(p)
    log_den = torch.logaddexp(log_num, log_q0_minus.detach() + torch.log1p(-p))
    return torch.exp(log_num - log_den).clamp(0.0, 1.0)

