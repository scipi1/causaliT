"""
Var- and R2-sortnregress (Reisach et al., NeurIPS 2021; NeurIPS 2023).

Trivial baselines with no causal learning: order nodes by marginal variance
(or by R2 from all other nodes), then regress each node on its predecessors
with LassoLarsIC.  If they score well on a dataset, the simulated data leaks
the order and the other methods' numbers must be read with care.

Implementation: ``CausalDisco`` (``pip install --no-deps causaldisco``), the
authors' package.  The returned coefficient matrix is already in paper
orientation (``W[parent, child]``); its ``|W|`` is thresholded downstream like
any other weighted method.  The order is the sorting criterion itself.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

#: No hyperparameters: LassoLarsIC chooses its own penalty.
DEFAULT_PARAMS: Dict[str, Any] = {}


def _fit(criterion: str, X: np.ndarray, params: Dict[str, Any]) -> BenchmarkResult:
    from CausalDisco.analytics import r2coeff
    from CausalDisco.baselines import r2_sort_regress, var_sort_regress

    merged = {**params}
    merged.pop("w_threshold", None)
    seed = int(merged.pop("seed", 0))
    X = np.ascontiguousarray(np.asarray(X, dtype=float))

    start = time.perf_counter()
    if criterion == "var":
        W = var_sort_regress(X)
        key = np.var(X, axis=0)
    else:
        W = r2_sort_regress(X)
        key = r2coeff(X.T)
    seconds = time.perf_counter() - start

    return BenchmarkResult(
        W=np.asarray(W, dtype=float),
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=False,
        output_type="weighted",
        order=[int(i) for i in np.argsort(key, kind="stable")],
        extra={"implementation": f"CausalDisco.baselines.{criterion}_sort_regress"},
    )


def fit_var(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """Var-sortnregress."""
    return _fit("var", X, params)


def fit_r2(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """R2-sortnregress."""
    return _fit("r2", X, params)
