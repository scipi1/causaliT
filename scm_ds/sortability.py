"""Sortability analytics for simulated DAG datasets.

Implements the order-alignment measures of

* Reisach, Seiler & Weichwald (2021), "Beware of the Simulated DAG! Causal
  Discovery Benchmarks May Be Easy To Game" (NeurIPS 2021,
  doi:10.48550/arXiv.2102.13647) -- **var-sortability**, and
* Reisach, Tami, Seiler, Chambaz & Weichwald (2023), "A Scale-Invariant
  Sorting Criterion to Find a Causal Order in Additive Noise Models"
  (NeurIPS 2023, doi:10.48550/arXiv.2303.18211) -- **R2-sortability**.

The implementations replicate the authors' reference library CausalDisco
(https://github.com/CausalDisco/CausalDisco, ``CausalDisco/analytics.py``):

* agreement is measured over DIRECTED PATHS (powers of the adjacency matrix
  weight each ancestor pair by its number of connecting paths), and
* a tied pair contributes 1/2 (see ``order_alignment_paths``).

Convention: adjacency matrices follow the CausalDisco convention
``W[i, j] != 0`` == directed edge ``i -> j`` (i is a parent of j).
"""

from __future__ import annotations

import numpy as np


def order_alignment_paths(W: np.ndarray, scores: np.ndarray, tol: float = 0.0) -> float:
    """Path-weighted agreement between the causal order and the score order.

    Args:
        W: (d, d) weighted/binary DAG adjacency, ``W[i, j] != 0`` = edge i->j.
        scores: (d,) score per variable; agreement means ancestors score lower.
        tol: non-negative tolerance for score comparisons (ties count 1/2).

    Returns:
        Scalar in [0, 1]; NaN when the graph has no edges.
    """
    assert tol >= 0.0, "tol must be non-negative"
    E = W != 0
    Ek = E.copy()
    n_paths = 0
    n_correctly_ordered_paths = 0

    # scores as a row vector; differences[i, j] > 0 iff score_j > score_i
    scores = np.asarray(scores, dtype=float).reshape(1, -1)
    differences = scores - scores.T

    # See arXiv:2102.13647 Section 3.1 and arXiv:2303.18211 Equation (3).
    for _ in range(len(E) - 1):
        n_paths += Ek.sum()
        # 1/2 per correctly ordered or tied pair ...
        n_correctly_ordered_paths += (Ek * (differences >= 0 - tol)).sum() / 2
        # ... plus another 1/2 per strictly correctly ordered pair
        n_correctly_ordered_paths += (Ek * (differences > 0 + tol)).sum() / 2
        Ek = Ek.dot(E)
    if n_paths == 0:
        return float("nan")
    return float(n_correctly_ordered_paths / n_paths)


def var_sortability(X: np.ndarray, W: np.ndarray, tol: float = 0.0) -> float:
    """Var-sortability: agreement of the causal order with increasing marginal
    variance (Reisach et al. 2021).  Scale-dependent by construction.

    Args:
        X: (n_samples, d) data matrix.
        W: (d, d) DAG adjacency, ``W[i, j] != 0`` = edge i->j.
    """
    scores = np.var(X, axis=0, ddof=1)
    return order_alignment_paths(W, scores, tol=tol)


def _r2_coefficients(X: np.ndarray) -> np.ndarray:
    """R2 of each variable regressed on ALL other variables.

    Uses the partial-correlation identity ``R2_k = 1 - 1 / inv(corr)_[k, k]``
    (as in CausalDisco's ``r2coeff``), falling back to explicit per-variable
    OLS when the correlation matrix is singular.
    """
    C = np.corrcoef(X.T)
    try:
        return 1.0 - 1.0 / np.diag(np.linalg.inv(C))
    except np.linalg.LinAlgError:
        from sklearn.linear_model import LinearRegression

        d = X.shape[1]
        r2s = np.zeros(d)
        lr = LinearRegression()
        for kcol in range(d):
            others = np.arange(d) != kcol
            lr.fit(X[:, others], X[:, kcol])
            r2s[kcol] = lr.score(X[:, others], X[:, kcol])
        return r2s


def r2_sortability(X: np.ndarray, W: np.ndarray, tol: float = 0.0) -> float:
    """R2-sortability: agreement of the causal order with increasing R2
    (Reisach et al. 2023).  Scale-invariant by construction.

    Args:
        X: (n_samples, d) data matrix.
        W: (d, d) DAG adjacency, ``W[i, j] != 0`` = edge i->j.
    """
    scores = _r2_coefficients(X)
    return order_alignment_paths(W, scores, tol=tol)
