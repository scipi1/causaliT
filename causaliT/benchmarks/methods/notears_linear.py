"""
NOTEARS, linear SEM (Zheng et al., NeurIPS 2018).

Thin wrapper around the vendored paper code
(``causaliT.benchmarks.vendor.notears.linear.notears_linear``).

Paper defaults (``notears/linear.py`` signature, unchanged across graph sizes):
``lambda1=0.1``, ``loss_type='l2'``, ``max_iter=100``, ``h_tol=1e-8``,
``rho_max=1e16``.

``w_threshold`` is deliberately **not** part of the fit: the wrapper passes
``0.0`` so the returned ``W`` keeps its raw magnitudes, and thresholding happens
later in ``postprocess``.  That way ``benchmark_run.json`` stores the full
weighted estimate and any threshold can be re-scored offline without refitting.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult
from causaliT.benchmarks.vendor.notears.linear import notears_linear

#: Paper hyperparameters; size-independent (no hidden layer in the linear model).
DEFAULT_PARAMS: Dict[str, Any] = {
    "lambda1": 0.1,
    "loss_type": "l2",
    "max_iter": 100,
    "h_tol": 1e-8,
    "rho_max": 1e16,
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Fit linear NOTEARS on *X*.

    Args:
        X: ``(n_samples, N)`` design matrix, columns ordered ``[S..., X...]``.
        **params: Overrides of :data:`DEFAULT_PARAMS`.

    Returns:
        :class:`BenchmarkResult` with the raw (unthresholded) ``W`` in paper
        orientation (``W[i, j] != 0`` means ``i -> j``).
    """
    merged = {**DEFAULT_PARAMS, **params}
    # Not a solver argument: the caller thresholds in postprocess instead.
    merged.pop("w_threshold", None)

    X = np.asarray(X, dtype=float)

    start = time.perf_counter()
    W = notears_linear(
        X,
        lambda1=float(merged["lambda1"]),
        loss_type=str(merged["loss_type"]),
        max_iter=int(merged["max_iter"]),
        h_tol=float(merged["h_tol"]),
        rho_max=float(merged["rho_max"]),
        w_threshold=0.0,
    )
    seconds = time.perf_counter() - start

    return BenchmarkResult(
        W=np.asarray(W, dtype=float),
        seconds=seconds,
        params=merged,
        is_binary=False,
        extra={"implementation": "vendored notears.linear.notears_linear"},
    )
