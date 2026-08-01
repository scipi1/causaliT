"""
PC algorithm (Spirtes & Glymour, 1991), constraint-based baseline.

Uses ``causallearn.search.ConstraintBased.PC.pc`` (``pip install causal-learn``),
the reference implementation maintained by the CMU/Tsinghua group.

Defaults: ``alpha=0.05`` with the Fisher-z conditional independence test, the
standard choice for continuous data and the package's own default.  ``stable=True``
selects PC-stable (order-independent skeleton), which is what practically
everybody reports today.

**Output is a CPDAG, not a DAG.**  causal-learn encodes the graph in a
``GeneralGraph`` whose adjacency uses -1 (tail) / 1 (arrowhead) endpoint marks:

    ``G[i, j] == -1 and G[j, i] == 1``  ->  ``i -> j``   (oriented)
    ``G[i, j] == -1 and G[j, i] == -1`` ->  ``i -- j``   (undirected)

Undirected edges are genuinely unoriented within the Markov equivalence class,
so they are reported with score ``0.5`` while oriented edges get ``1.0`` (see
``postprocess.cpdag_to_scores``).  A bidirected edge (``1``/``1``), which PC can
emit in the presence of conflicts, is treated as undirected.

The wrapper returns the scores in *paper orientation* (``[parent, child]``) so
that the common ``postprocess.adjacency_to_blocks`` path applies, with
``is_binary=True`` to keep the 0.5 values intact.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult
from causaliT.benchmarks.postprocess import cpdag_to_scores

#: Endpoint marks used by causal-learn's GeneralGraph adjacency.
TAIL, ARROW = -1, 1

#: Package defaults; ``indep_test='fisherz'`` is the continuous-data standard.
DEFAULT_PARAMS: Dict[str, Any] = {
    "alpha": 0.05,
    "indep_test": "fisherz",
    "stable": True,
    "uc_rule": 0,
    "uc_priority": 2,
}


def _cpdag_matrices(graph: np.ndarray) -> tuple:
    """
    Split a causal-learn endpoint matrix into directed / undirected parts.

    Args:
        graph: ``(N, N)`` matrix of endpoint marks (see module docstring).

    Returns:
        ``(directed, undirected)`` boolean matrices in **paper orientation**
        (``[parent, child]``).
    """
    G = np.asarray(graph)
    n = G.shape[0]
    directed = np.zeros((n, n), dtype=bool)
    undirected = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == TAIL and b == ARROW:       # i -> j
                directed[i, j] = True
            elif a == ARROW and b == TAIL:     # j -> i
                directed[j, i] = True
            else:                              # i -- j, or conflicting marks
                undirected[i, j] = True
                undirected[j, i] = True

    return directed, undirected


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Run PC on *X* and return CPDAG edge scores.

    Args:
        X: ``(n_samples, N)`` design matrix, columns ordered ``[S..., X...]``.
        **params: Overrides of :data:`DEFAULT_PARAMS`.

    Returns:
        :class:`BenchmarkResult` with ``W`` holding **scores** rather than
        weights (``1.0`` oriented, ``0.5`` undirected) in paper orientation and
        ``is_binary=True`` so ``postprocess`` passes them through untouched.
        ``extra`` records how many edges were left unoriented, which is the
        fair-comparison caveat to quote for a constraint-based method.
    """
    from causallearn.search.ConstraintBased.PC import pc  # optional dependency

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)   # no weights to threshold

    X = np.ascontiguousarray(np.asarray(X, dtype=float))

    start = time.perf_counter()
    cg = pc(
        X,
        alpha=float(merged["alpha"]),
        indep_test=str(merged["indep_test"]),
        stable=bool(merged["stable"]),
        uc_rule=int(merged["uc_rule"]),
        uc_priority=int(merged["uc_priority"]),
        show_progress=False,
    )
    seconds = time.perf_counter() - start

    graph = np.asarray(cg.G.graph)
    directed, undirected = _cpdag_matrices(graph)
    scores = cpdag_to_scores(directed, undirected)

    return BenchmarkResult(
        W=scores,
        seconds=seconds,
        params=merged,
        is_binary=True,
        extra={
            "implementation": "causallearn.search.ConstraintBased.PC.pc",
            "returns": "CPDAG (1.0 = oriented, 0.5 = undirected)",
            "n_directed": int(directed.sum()),
            "n_undirected": int(undirected.sum() // 2),
        },
    )
