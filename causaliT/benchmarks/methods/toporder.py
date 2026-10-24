"""
Order-based ANM methods from ``dodiscover.toporder``: CAM, SCORE, DAS, NoGAM.

``dodiscover`` (py-why, MIT) contains the implementations contributed by the
SCORE / DAS / NoGAM authors (Montagna, Rolland et al.), with a pure-Python CAM
pruning step (GAM p-values via ``pygam``).  The papers pruned with the R
``CAM`` package through ``cdt``; we stay Python-only, so pruned graphs can
differ slightly from the published numbers.  The ORDER is unaffected.

dodiscover is no longer on PyPI; install from GitHub without its pins::

    pip install --no-deps git+https://github.com/py-why/dodiscover.git

Every method returns a binary DAG plus the estimated order.  ``DEFAULT_PARAMS``
are the package defaults, which equal the papers' (alpha=0.05, 10 splines of
degree 3, eta_G = eta_H = 1e-3).
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

#: Package defaults per algorithm (forwarded verbatim to the constructor).
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "CAM": {"alpha": 0.05, "n_splines": 10, "splines_degree": 3, "pns": False},
    "SCORE": {"eta_G": 0.001, "eta_H": 0.001, "alpha": 0.05, "n_splines": 10,
              "splines_degree": 3, "estimate_variance": False, "pns": False},
    "DAS": {"eta_G": 0.001, "eta_H": 0.001, "alpha": 0.05, "das_cutoff": None,
            "n_splines": 10, "splines_degree": 3, "min_parents": 5, "max_parents": 20},
    "NoGAM": {"n_crossval": 5, "ridge_alpha": 0.01, "ridge_gamma": 0.1,
              "eta_G": 0.001, "eta_H": 0.001, "alpha": 0.05, "n_splines": 10,
              "splines_degree": 3, "pns": False},
}

#: Read by ``base.default_params`` (one entry per fit callable).
DEFAULT_PARAMS_BY_FIT: Dict[str, Dict[str, Any]] = {
    "fit_cam": DEFAULTS["CAM"],
    "fit_score": DEFAULTS["SCORE"],
    "fit_das": DEFAULTS["DAS"],
    "fit_nogam": DEFAULTS["NoGAM"],
}


def _fit(algorithm: str, X: np.ndarray, params: Dict[str, Any]) -> BenchmarkResult:
    """Run one ``dodiscover.toporder`` algorithm and convert its output."""
    import networkx as nx
    import pandas as pd
    from dodiscover import make_context
    from dodiscover import toporder

    merged = {**DEFAULTS[algorithm], **params}
    merged.pop("w_threshold", None)
    seed = int(merged.pop("seed", 0))
    np.random.seed(seed)  # GAM / ridge CV use the global numpy RNG

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    columns = [str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)

    model = getattr(toporder, algorithm)(**merged)
    start = time.perf_counter()
    model.learn_graph(df, make_context().variables(data=df).build())
    seconds = time.perf_counter() - start

    W = nx.to_numpy_array(model.graph_, nodelist=columns)  # [parent, child]
    return BenchmarkResult(
        W=W,
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=True,
        output_type="dag",
        order=list(model.order_),
        extra={"implementation": f"dodiscover.toporder.{algorithm}",
               "pruning": "dodiscover CAMPruning (Python, pygam)"},
    )


def fit_cam(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """CAM (Buhlmann, Peters & Ernest, 2014)."""
    return _fit("CAM", X, params)


def fit_score(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """SCORE (Rolland et al., 2022)."""
    return _fit("SCORE", X, params)


def fit_das(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """DAS (Montagna et al., 2023b)."""
    return _fit("DAS", X, params)


def fit_nogam(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """NoGAM (Montagna et al., 2023a)."""
    return _fit("NoGAM", X, params)
