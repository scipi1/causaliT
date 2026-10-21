"""Causal-order estimates from raw residual-HSIC nodal contributions.

The estimators in this module are diagnostics: they deliberately ignore the
masks and pair weights used by the structural training loss. In particular,
descendant pairs are RETAINED because they carry the RESIT-style upstream
signal, while the irreducible self pair HSIC(X_i, r_i) is excluded by default.

The fixed-residual iterative estimator is a RESIT-inspired heuristic, not a
full RESIT implementation: residuals are not refitted after removing a sink.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from causaliT.utils.hsic_utils import hsic_pair_matrix, hsic_row_means


@dataclass(frozen=True)
class HsicCausalOrderResult:
    """Result of a residual-HSIC causal-order estimate.

    Attributes:
        scores: Raw nodal upstreamness scores. High means source-like; low
            means sink-like.
        order: Source-to-sink node indices.
        ranks: Zero-based position of each node in ``order``.
        sink_sequence: Sink-removal order for ``iterative_sink``.
        pair_matrix: Raw pair matrix when ``return_pair_matrix=True``.
        null_mean/null_std: Row-wise permutation-null summaries.
        zscores: Null-standardized observed row scores.
        p_values: Empirical upper-tail null probabilities.
        method: Ordering method used.
    """

    scores: torch.Tensor
    order: torch.Tensor
    ranks: torch.Tensor
    sink_sequence: Optional[torch.Tensor]
    pair_matrix: Optional[torch.Tensor]
    null_mean: Optional[torch.Tensor]
    null_std: Optional[torch.Tensor]
    zscores: Optional[torch.Tensor]
    p_values: Optional[torch.Tensor]
    method: str


def build_offdiagonal_pair_mask(
    n_targets: int,
    n_sources: int,
    diagonal_offset: Optional[int] = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a pair mask with the target's own source column set to zero.

    Use ``diagonal_offset=0`` for homogeneous square matrices, ``S_seq_len``
    for split ``[S, X]`` sources, and ``None`` for pure cross-attention.
    """
    if n_targets <= 0 or n_sources <= 0:
        raise ValueError(
            f"n_targets and n_sources must be positive, got "
            f"{n_targets} and {n_sources}."
        )
    mask = torch.ones(n_targets, n_sources, device=device, dtype=dtype)
    if diagonal_offset is None:
        return mask
    offset = int(diagonal_offset)
    rows = torch.arange(n_targets, device=device)
    cols = rows + offset
    valid = (cols >= 0) & (cols < n_sources)
    if bool(valid.any()):
        mask[rows[valid], cols[valid]] = 0.0
    return mask


def _validate_inputs(
    source_values: torch.Tensor,
    residuals: torch.Tensor,
) -> None:
    if source_values.dim() != 2 or residuals.dim() != 2:
        raise ValueError(
            "source_values and residuals must be 2-D (batch, nodes), got "
            f"{tuple(source_values.shape)} and {tuple(residuals.shape)}."
        )
    if source_values.shape[0] != residuals.shape[0]:
        raise ValueError(
            "source_values and residuals must share the batch dimension, got "
            f"{source_values.shape[0]} and {residuals.shape[0]}."
        )
    if source_values.shape[0] < 2:
        raise ValueError("at least two samples are required for HSIC.")
    if residuals.shape[1] < 2:
        raise ValueError("at least two target nodes are required for ordering.")


def _ranks_from_order(order: torch.Tensor) -> torch.Tensor:
    """Return the zero-based rank of every node in a source-to-sink order."""
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device)
    return ranks


def _fixed_residual_iterative_sink(
    pair_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Peel sinks from a fixed square pair matrix and return source order.

    This recomputes row means over the active node set after each removal, but
    does NOT refit residuals. It is therefore a cheap RESIT-inspired peeling
    heuristic rather than strict RESIT.
    """
    if pair_matrix.shape[0] != pair_matrix.shape[1]:
        raise ValueError(
            "iterative_sink requires a square homogeneous pair matrix, got "
            f"{tuple(pair_matrix.shape)}. Use row_mean for split layouts."
        )
    n = pair_matrix.shape[0]
    active = torch.arange(n, device=pair_matrix.device)
    sinks = []
    for _ in range(n - 1):
        active_mat = pair_matrix[active][:, active]
        active_scores = hsic_row_means(active_mat)
        finite = torch.isfinite(active_scores)
        if not bool(finite.any()):
            raise ValueError("iterative sink removal found an all-NaN pair matrix.")
        # A NaN row means no measurable evidence; treat it as maximally
        # sink-like rather than letting NaN ordering leak into the result.
        active_scores = torch.where(
            finite,
            active_scores,
            torch.full_like(active_scores, float("inf")),
        )
        sink_pos = int(torch.argmin(active_scores).item())
        sinks.append(active[sink_pos])
        active = torch.cat([active[:sink_pos], active[sink_pos + 1:]])
    sinks.append(active[0])
    sink_sequence = torch.stack(sinks)
    order = torch.flip(sink_sequence, dims=(0,))
    return order, sink_sequence


def estimate_hsic_causal_order(
    source_values: torch.Tensor,
    residuals: torch.Tensor,
    *,
    method: str = "row_mean",
    diagonal_offset: Optional[int] = 0,
    sigma: float = 1.0,
    adaptive_bandwidth: bool = True,
    mode: str = "biased",
    nhsic_epsilon: float = 0.01,
    source_kernel: str = "rbf",
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    null_permutations: int = 0,
    null_seed: int = 20240817,
    null_epsilon: float = 1e-12,
    return_pair_matrix: bool = False,
) -> HsicCausalOrderResult:
    """Estimate a causal order from raw residual-HSIC nodal contributions.

    The observed score of target node ``i`` is the unweighted row mean
    ``mean_{j != i} HSIC(source_j, residual_i)``. High scores are
    source-like. Descendant terms are retained; no training-loss mask or
    attention/direction weight is applied.

    ``null_permutations > 0`` independently permutes every residual column
    across samples for each replicate. This estimates row-wise independence
    nulls without refitting residuals. The returned empirical ``p_values``
    are upper-tail probabilities: high means compatible with independence
    (sink-like), while high ``zscores`` mean source-like.
    """
    _validate_inputs(source_values, residuals)
    method = str(method)
    if method not in {"row_mean", "iterative_sink"}:
        raise ValueError(
            f"method must be 'row_mean' or 'iterative_sink', got {method!r}."
        )
    if int(null_permutations) < 0:
        raise ValueError(
            f"null_permutations must be non-negative, got {null_permutations}."
        )

    n_targets = residuals.shape[1]
    n_sources = source_values.shape[1]
    pair_mask = build_offdiagonal_pair_mask(
        n_targets,
        n_sources,
        diagonal_offset=diagonal_offset,
        device=source_values.device,
        dtype=source_values.dtype,
    )

    pair_matrix = hsic_pair_matrix(
        source_values=source_values,
        residuals=residuals,
        sigma=sigma,
        adaptive_bandwidth=adaptive_bandwidth,
        mode=mode,
        nhsic_epsilon=nhsic_epsilon,
        source_kernel=source_kernel,
        bandwidth_multipliers=bandwidth_multipliers,
        pair_mask=pair_mask,
    ).detach()
    scores = hsic_row_means(pair_matrix).detach()
    if not bool(torch.isfinite(scores).any()):
        raise ValueError("causal-order scores are all NaN.")

    if method == "row_mean":
        order = torch.argsort(scores, descending=True, stable=True)
        sink_sequence = None
    else:
        order, sink_sequence = _fixed_residual_iterative_sink(pair_matrix)
    ranks = _ranks_from_order(order)

    null_mean = null_std = zscores = p_values = None
    if int(null_permutations) > 0:
        batch, n_null_cols = residuals.shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(null_seed))
        null_scores = []
        for _ in range(int(null_permutations)):
            # One independent sample permutation per residual column; gather
            # evaluates all row nulls in one pair-matrix pass.
            perms = torch.stack(
                [
                    torch.randperm(batch, generator=generator)
                    for _ in range(n_null_cols)
                ],
                dim=1,
            ).to(device=residuals.device)
            residuals_perm = torch.gather(residuals, dim=0, index=perms)
            null_mat = hsic_pair_matrix(
                source_values=source_values,
                residuals=residuals_perm,
                sigma=sigma,
                adaptive_bandwidth=adaptive_bandwidth,
                mode=mode,
                nhsic_epsilon=nhsic_epsilon,
                source_kernel=source_kernel,
                bandwidth_multipliers=bandwidth_multipliers,
                pair_mask=pair_mask,
            ).detach()
            null_scores.append(hsic_row_means(null_mat).detach())
        null_stack = torch.stack(null_scores)
        if not bool(torch.isfinite(null_stack).all()):
            raise ValueError("permutation-null causal-order scores contain NaN.")
        null_mean = null_stack.mean(dim=0)
        null_std = null_stack.std(dim=0, unbiased=False)
        zscores = (scores - null_mean) / (null_std + float(null_epsilon))
        exceedances = (null_stack >= scores.unsqueeze(0)).sum(dim=0)
        p_values = (1.0 + exceedances.to(scores.dtype)) / (
            float(null_permutations) + 1.0
        )

    return HsicCausalOrderResult(
        scores=scores,
        order=order.detach(),
        ranks=ranks.detach(),
        sink_sequence=(
            sink_sequence.detach() if sink_sequence is not None else None
        ),
        pair_matrix=pair_matrix if return_pair_matrix else None,
        null_mean=null_mean,
        null_std=null_std,
        zscores=zscores,
        p_values=p_values,
        method=method,
    )
