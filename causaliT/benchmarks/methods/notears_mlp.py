"""
NOTEARS, nonlinear / per-node MLP (Zheng et al., AISTATS 2020).

Thin wrapper around the vendored ``NotearsMLP`` + ``notears_nonlinear``.

Architecture: ``dims=[d, H, 1]`` with a **fixed** ``H = 10``.  That is the value
the paper uses for every graph size it reports (d = 10 ... 100), because the MLP
is applied *per node*: the model already scales with ``d`` through the number of
locally-connected units, so the hidden width does not need to grow.  Keeping it
fixed also keeps the benchmark honest - no capacity tuning against the
ground-truth DAG.

Paper defaults: ``lambda1=0.01`` (L1 on the first layer), ``lambda2=0.01``
(L2 weight decay), ``max_iter=100``, ``h_tol=1e-8``, ``rho_max=1e16``.

Two mechanical details of the reference solver are handled here:

* it needs ``torch.set_default_dtype(torch.double)`` (``LBFGSBScipy`` hands the
  parameters to SciPy as float64); the wrapper sets and restores the global dtype
  so the rest of the process is unaffected;
* it is CPU-only via SciPy's L-BFGS-B, so no device handling is required.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

#: Paper hyperparameters.  ``hidden_units`` maps to ``dims=[d, hidden_units, 1]``.
DEFAULT_PARAMS: Dict[str, Any] = {
    "lambda1": 0.01,
    "lambda2": 0.01,
    "hidden_units": 10,
    "bias": True,
    "max_iter": 100,
    "h_tol": 1e-8,
    "rho_max": 1e16,
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Fit nonlinear NOTEARS (per-node MLP) on *X*.

    Args:
        X: ``(n_samples, N)`` design matrix, columns ordered ``[S..., X...]``.
        **params: Overrides of :data:`DEFAULT_PARAMS`.  ``seed`` is accepted and
            used to seed torch so the MLP initialisation is reproducible.

    Returns:
        :class:`BenchmarkResult` with the raw ``W`` in paper orientation.  Note
        that ``NotearsMLP.fc1_to_adj`` returns edge *magnitudes* (the L2 norm of
        the first-layer weights per input), so ``W`` is non-negative.
    """
    import torch  # local import: keeps torch out of the package import path

    from causaliT.benchmarks.vendor.notears.nonlinear import (
        NotearsMLP,
        notears_nonlinear,
    )

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)   # thresholding happens in postprocess
    seed = merged.pop("seed", 0)

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    d = int(X.shape[1])
    hidden = int(merged["hidden_units"])

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        torch.manual_seed(int(seed))
        model = NotearsMLP(dims=[d, hidden, 1], bias=bool(merged["bias"]))
        start = time.perf_counter()
        W = notears_nonlinear(
            model,
            X,
            lambda1=float(merged["lambda1"]),
            lambda2=float(merged["lambda2"]),
            max_iter=int(merged["max_iter"]),
            h_tol=float(merged["h_tol"]),
            rho_max=float(merged["rho_max"]),
            w_threshold=0.0,
        )
        seconds = time.perf_counter() - start
    finally:
        torch.set_default_dtype(previous_dtype)

    return BenchmarkResult(
        W=np.asarray(W, dtype=float),
        seconds=seconds,
        params={**merged, "seed": seed, "dims": [d, hidden, 1]},
        is_binary=False,
        extra={
            "implementation": "vendored notears.nonlinear.NotearsMLP",
            "dims": [d, hidden, 1],
        },
    )
