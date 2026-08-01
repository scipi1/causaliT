"""
DAGMA, linear SEM (Bello, Aragam, Ravikumar, NeurIPS 2022).

Uses the authors' own ``dagma`` package (``pip install dagma``), so there is
nothing to vendor: ``dagma.linear.DagmaLinear`` is the reference implementation.

Paper defaults for the linear model, taken from the package's own documented
example: ``loss_type='l2'``, ``lambda1=0.02``, ``w_threshold=0.3``, and the
log-det schedule ``T=5``, ``mu_init=1.0``, ``mu_factor=0.1``,
``s=[1.0, 0.9, 0.8, 0.7, 0.6]`` (one ``s`` per outer iteration ``T``),
``warm_iter=3e4``, ``max_iter=6e4``, ``lr=3e-4``.

As with NOTEARS, ``w_threshold=0`` is passed to the solver so the returned ``W``
keeps its magnitudes and the threshold is applied in ``postprocess``.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

#: Paper / package defaults.  ``s`` must have either length 1 or length ``T``.
DEFAULT_PARAMS: Dict[str, Any] = {
    "loss_type": "l2",
    "lambda1": 0.02,
    "T": 5,
    "mu_init": 1.0,
    "mu_factor": 0.1,
    "s": [1.0, 0.9, 0.8, 0.7, 0.6],
    "warm_iter": 30000,
    "max_iter": 60000,
    "lr": 3e-4,
    "beta_1": 0.99,
    "beta_2": 0.999,
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Fit linear DAGMA on *X*.

    Args:
        X: ``(n_samples, N)`` design matrix, columns ordered ``[S..., X...]``.
        **params: Overrides of :data:`DEFAULT_PARAMS`.

    Returns:
        :class:`BenchmarkResult` with the raw ``W`` in paper orientation
        (``W[i, j] != 0`` means ``i -> j``).
    """
    from dagma.linear import DagmaLinear  # local import: optional dependency

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)      # thresholding happens in postprocess

    X = np.asarray(X, dtype=float)
    s = merged["s"]
    s = [float(v) for v in s] if isinstance(s, (list, tuple)) else float(s)

    model = DagmaLinear(loss_type=str(merged["loss_type"]))

    start = time.perf_counter()
    W = model.fit(
        X,
        lambda1=float(merged["lambda1"]),
        w_threshold=0.0,
        T=int(merged["T"]),
        mu_init=float(merged["mu_init"]),
        mu_factor=float(merged["mu_factor"]),
        s=s,
        warm_iter=int(merged["warm_iter"]),
        max_iter=int(merged["max_iter"]),
        lr=float(merged["lr"]),
        beta_1=float(merged["beta_1"]),
        beta_2=float(merged["beta_2"]),
    )
    seconds = time.perf_counter() - start

    return BenchmarkResult(
        W=np.asarray(W, dtype=float),
        seconds=seconds,
        params={**merged, "s": s},
        is_binary=False,
        extra={"implementation": "dagma.linear.DagmaLinear"},
    )
