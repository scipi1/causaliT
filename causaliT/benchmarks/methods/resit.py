"""
RESIT, regression with subsequent independence test (Peters et al., JMLR 2014).

Closest classical relative of causaliT's order-factorized design: the order
is found greedily by removing, at each step, the node whose regression
residual is most independent (HSIC) of the remaining nodes; edges are then
pruned by the same residual-independence test.

Implementation: ``lingam.RESIT`` (``pip install lingam``), maintained by the
LiNGAM authors.  The original code is MATLAB; ``lingam`` is the Python
reference used in the literature.  ``lingam`` pins an old scipy that does not
build on Python 3.13; install it with ``pip install --no-deps lingam`` plus
its runtime deps (pygam, psy, semopy, statsmodels, graphviz).

Defaults: ``alpha=0.01`` (package default for the HSIC test).  The regressor
is the free part of RESIT; the paper uses GP regression.  We use sklearn's
``GaussianProcessRegressor`` with an RBF + white-noise kernel, matching the
paper's choice, instead of an arbitrary tree model.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

DEFAULT_PARAMS: Dict[str, Any] = {
    "alpha": 0.01,
    "regressor": "gp",
}


def _make_regressor(name: str, seed: int):
    """Regressor used inside RESIT (``gp`` = paper choice, ``rf`` = faster)."""
    if name == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        return GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), normalize_y=True, random_state=seed
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(max_depth=4, random_state=seed)
    raise ValueError(f"Unknown RESIT regressor '{name}'; use 'gp' or 'rf'.")


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Run RESIT on *X*.

    Returns:
        Binary DAG in paper orientation and the estimated causal order.
        ``lingam`` stores ``B[child, parent]``, hence the transpose.
    """
    import lingam

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)
    seed = int(merged.pop("seed", 0))

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    model = lingam.RESIT(
        regressor=_make_regressor(str(merged["regressor"]), seed),
        random_state=seed,
        alpha=float(merged["alpha"]),
    )
    start = time.perf_counter()
    model.fit(X)
    seconds = time.perf_counter() - start

    W = (np.asarray(model.adjacency_matrix_) != 0).astype(float).T
    return BenchmarkResult(
        W=W,
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=True,
        output_type="dag",
        order=list(model.causal_order_),
        extra={"implementation": "lingam.RESIT"},
    )
