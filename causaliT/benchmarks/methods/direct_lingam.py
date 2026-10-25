"""
DirectLiNGAM (Shimizu et al., JMLR 2011): linear non-Gaussian acyclic model.

The reference method for LINEAR SCMs with NON-GAUSSIAN noise, where the DAG
is fully identifiable (Shimizu et al., 2006).  The causal order is found
directly, one exogenous variable at a time, by a pairwise likelihood-ratio
independence measure; edge weights are then estimated by adaptive Lasso
regression on the predecessors in that order.

Implementation: ``lingam.DirectLiNGAM`` (``pip install --no-deps lingam``,
same package and install as RESIT), maintained by the LiNGAM authors.

Defaults: ``measure='pwling'`` (package default, the paper's measure).  The
algorithm is deterministic given the data, so one init seed suffices; the
seed is still forwarded as ``random_state`` for completeness.

Output: weighted ``W`` (adaptive-Lasso coefficients, zeros already pruned)
plus the estimated causal order.  Scored like every weighted method:
``|W| >= w_threshold`` for SHD-family metrics, ``|W|`` for AUPRC.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

DEFAULT_PARAMS: Dict[str, Any] = {
    "measure": "pwling",
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Run DirectLiNGAM on *X*.

    Returns:
        Weighted adjacency in paper orientation (``lingam`` stores
        ``B[child, parent]``, hence the transpose) and the causal order.
    """
    import lingam

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)   # applied by the runner, not here
    seed = int(merged.pop("seed", 0))

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    model = lingam.DirectLiNGAM(random_state=seed, measure=str(merged["measure"]))
    start = time.perf_counter()
    model.fit(X)
    seconds = time.perf_counter() - start

    W = np.asarray(model.adjacency_matrix_, dtype=float).T
    return BenchmarkResult(
        W=W,
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=False,
        output_type="weighted",
        order=list(model.causal_order_),
        extra={"implementation": "lingam.DirectLiNGAM"},
    )
