"""
Estimated adjacency -> canonical causaliT DAG blocks.

This module owns the two conventions that would otherwise be easy to get
silently wrong.

**1. Orientation.**  NOTEARS, DAGMA and the linear SEM literature write the
model as ``X = X W + noise``, so ``W[i, j] != 0`` means ``i -> j``: rows are
parents.  causaliT's DAG masks and every metric built on them use the opposite
layout - rows are *children*, columns are parents (``dec_cross`` has shape
``(L_X, L_S)``: X rows attending to S columns).  The conversion is therefore a
single transpose, done here and nowhere else::

    A = W.T        # A[child, parent]

A transposed adjacency would still produce plausible-looking SHD numbers while
scoring the reverse graph, which is why ``tests/test_benchmarks.py`` feeds the
ground-truth ``W`` through this function and asserts zero SHD.

**2. Scores.**  ``write_dag_report`` expects edge *probabilities* in ``[0, 1]``
and binarises at ``dag_threshold`` (0.5 by default).  Weighted methods return
magnitudes on an arbitrary scale, so ``to_edge_scores`` maps them to:

    ``binary`` (default) - ``1.0`` where ``|W| >= w_threshold``, else ``0.0``.
        This reproduces the papers' own reporting (they threshold at 0.3 and
        report a discrete graph), and makes the metrics insensitive to weight
        scale.
    ``scaled``           - ``|W|`` divided by its maximum, then zeroed below
        ``w_threshold``.  Keeps a notion of confidence for the soft metrics.

PC has no weights at all: its CPDAG gives directed edges (confident, ``1.0``)
and undirected edges (orientation undetermined within the equivalence class,
``0.5``).  Encoding the ambiguity as exactly ``0.5`` is deliberate: at the
default ``dag_threshold=0.5`` an undirected edge counts as present in both
directions, which is the honest reading of a CPDAG.
"""

from typing import Dict, Optional

import numpy as np

from causaliT.evaluation.eval_funs.helpers.eval_dag_query import query_dag_blocks

#: Score modes accepted by ``to_edge_scores``.
SCORE_MODES = ("binary", "scaled")


def to_canonical_adjacency(W: np.ndarray) -> np.ndarray:
    """
    Convert a paper-orientation weighted adjacency to causaliT orientation.

    Args:
        W: ``(N, N)`` matrix with ``W[parent, child]`` semantics.

    Returns:
        ``(N, N)`` matrix with ``A[child, parent]`` semantics.

    Raises:
        ValueError: *W* is not square 2-D.
    """
    A = np.asarray(W, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Expected a square (N, N) adjacency, got shape {A.shape}."
        )
    return A.T.copy()


def to_edge_scores(
    A: np.ndarray,
    w_threshold: float = 0.3,
    score_mode: str = "binary",
    is_binary: bool = False,
) -> np.ndarray:
    """
    Map a canonical-orientation adjacency to edge scores in ``[0, 1]``.

    Args:
        A: ``(N, N)`` adjacency, ``A[child, parent]``, raw weights.
        w_threshold: Papers' magnitude threshold (default 0.3).  Applied to
            ``|A|``; below it, an edge is reported as absent.
        score_mode: ``binary`` (1.0 / 0.0) or ``scaled`` (``|A| / max|A|``).
        is_binary: Set for methods that already return a discrete graph (PC).
            The threshold is then skipped and values are passed through, so the
            CPDAG's 0.5 for undirected edges survives.

    Returns:
        ``(N, N)`` score matrix with zero diagonal (self-loops are not part of
        any DAG and would distort the zeroness statistics).

    Raises:
        ValueError: unknown *score_mode*.
    """
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"Unknown score_mode '{score_mode}'; expected one of {SCORE_MODES}."
        )

    scores = np.abs(np.asarray(A, dtype=float))

    if not is_binary:
        # ``>=`` matches the papers' own pruning (`W[abs(W) < w_threshold] = 0`)
        # and the ``>=`` binarisation used by the causaliT metrics, so an edge
        # exactly at the threshold is kept everywhere rather than in some places.
        keep = scores >= float(w_threshold)
        if score_mode == "binary":
            scores = keep.astype(float)
        else:  # scaled
            peak = scores.max() if scores.size else 0.0
            scores = np.where(keep, scores / peak, 0.0) if peak > 0 else np.zeros_like(scores)

    np.fill_diagonal(scores, 0.0)
    return np.clip(scores, 0.0, 1.0)


def cpdag_to_scores(directed: np.ndarray, undirected: np.ndarray) -> np.ndarray:
    """
    Combine the directed and undirected parts of a CPDAG into edge scores.

    Orientation-agnostic: both inputs must simply use the *same* convention, and
    the output follows it.  ``pc.py`` passes paper orientation
    (``[parent, child]``) so that the result can travel through
    ``adjacency_to_blocks`` like any other method's ``W``.

    Args:
        directed: ``(N, N)`` boolean/0-1 matrix of the edges the algorithm
            oriented.
        undirected: ``(N, N)`` symmetric boolean/0-1 matrix for edges whose
            orientation is not identifiable from observational data alone.

    Returns:
        ``(N, N)`` scores: ``1.0`` directed, ``0.5`` undirected, ``0.0`` absent.
        A directed edge wins over an undirected one for the same pair.
    """
    scores = 0.5 * (np.asarray(undirected, dtype=float) > 0)
    scores = np.where(np.asarray(directed, dtype=float) > 0, 1.0, scores)
    np.fill_diagonal(scores, 0.0)
    return scores


def forbid_edges_into_sources(scores: np.ndarray, L_S: int) -> np.ndarray:
    """
    Zero every edge pointing *into* a source variable.

    Source variables are exogenous by construction in causaliT's SCMs, so an
    edge into them is necessarily a false positive.  This is background
    knowledge the benchmarks do not have, so it is **off by default**
    (``benchmark.forbid_into_sources``); enabling it makes the comparison more
    favourable to the benchmarks and must be reported when used.

    Args:
        scores: ``(N, N)`` canonical scores (``[child, parent]``).
        L_S: Number of leading source columns/rows.

    Returns:
        A copy with the first ``L_S`` *rows* (source children) zeroed.
    """
    out = np.array(scores, dtype=float, copy=True)
    if L_S > 0:
        out[:L_S, :] = 0.0
    return out


def adjacency_to_blocks(
    W: np.ndarray,
    L_S: int,
    L_X: int,
    w_threshold: float = 0.3,
    score_mode: str = "binary",
    is_binary: bool = False,
    forbid_into_sources: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Full conversion: paper adjacency -> canonical ``{cross, self}`` DAG blocks.

    The square ``(N, N)`` score matrix is handed to ``query_dag_blocks``, the
    same shape-based classifier the models use.  Its homogeneous ``Rule 1``
    (square ``(N, N)`` with ``L_S > 0``) selects the X child rows and splits the
    columns at ``L_S``, yielding exactly the ``cross`` ``(L_X, L_S)`` and
    ``self`` ``(L_X, L_X)`` blocks that ``write_dag_report`` compares against
    ``dec_cross`` / ``dec_self``.  Reusing that function - rather than slicing
    here - guarantees benchmarks and models are evaluated on identically derived
    blocks.

    Args:
        W: ``(N, N)`` estimated adjacency in paper orientation.
        L_S, L_X: Source / intermediate variable counts, ``N = L_S + L_X``.
        w_threshold: Magnitude threshold (ignored when *is_binary*).
        score_mode: ``binary`` or ``scaled``.
        is_binary: The method already returns a discrete graph (PC).
        forbid_into_sources: Apply the exogeneity constraint (see above).
        verbose: Forwarded to ``query_dag_blocks``.

    Returns:
        ``{"cross": (L_X, L_S), "self": (L_X, L_X)}``.

    Raises:
        ValueError: *W* has the wrong shape for ``L_S + L_X``.
    """
    A = to_canonical_adjacency(W)
    N = int(L_S) + int(L_X)
    if A.shape != (N, N):
        raise ValueError(
            f"Estimated adjacency has shape {A.shape} but the dataset has "
            f"N = L_S + L_X = {L_S} + {L_X} = {N} variables."
        )

    scores = to_edge_scores(
        A, w_threshold=w_threshold, score_mode=score_mode, is_binary=is_binary
    )
    if forbid_into_sources:
        scores = forbid_edges_into_sources(scores, L_S=int(L_S))

    return query_dag_blocks(
        {"att_combined": scores}, L_S=int(L_S), L_X=int(L_X), verbose=verbose
    )


def count_edges(scores: np.ndarray, threshold: float = 0.5) -> int:
    """
    Number of edges at or above *threshold* (diagnostic for ``benchmark_run.json``).

    Uses ``>=``, the same comparison the causaliT metrics binarise with, so the
    reported count equals the number of edges actually scored.  For PC this
    means an undirected edge (0.5) counts once per direction, which is what its
    CPDAG asserts.
    """
    return int((np.asarray(scores, dtype=float) >= threshold).sum())


def is_dag(scores: np.ndarray, threshold: float = 0.5) -> Optional[bool]:
    """
    Whether the thresholded graph is acyclic (Kahn's algorithm on ``[child, parent]``).

    Reported as a sanity flag: the continuous methods guarantee acyclicity up to
    their tolerance, PC returns a CPDAG whose undirected edges appear in both
    directions and therefore is expected to report ``False``.
    """
    A = np.asarray(scores, dtype=float) >= threshold
    n = A.shape[0]
    in_degree = A.sum(axis=1).astype(int)      # parents per child
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        children = np.nonzero(A[:, node])[0]
        for child in children:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))
    return visited == n


__all__ = [
    "SCORE_MODES",
    "to_canonical_adjacency",
    "to_edge_scores",
    "cpdag_to_scores",
    "forbid_edges_into_sources",
    "adjacency_to_blocks",
    "count_edges",
    "is_dag",
]
