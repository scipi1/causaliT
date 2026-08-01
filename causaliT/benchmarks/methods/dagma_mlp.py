"""
DAGMA, nonlinear / per-node MLP (Bello, Aragam, Ravikumar, NeurIPS 2022).

Uses the authors' ``dagma`` package: ``dagma.nonlinear.DagmaMLP`` (the model)
plus ``dagma.nonlinear.DagmaNonlinear`` (the solver).

Architecture: ``dims=[d, H, 1]`` with a **fixed** ``H = 10``, the value used in
the package's documented example and in the paper's nonlinear experiments.  As
in NOTEARS-MLP, the network is per node, so capacity already grows with ``d``
and the hidden width stays constant - see ``docs/documentation/BENCHMARKS.md``.

Paper / package defaults: ``lambda1=0.02``, ``lambda2=0.005``, ``T=4``,
``mu_init=0.1``, ``mu_factor=0.1``, ``s=1.0``, ``warm_iter=5e4``,
``max_iter=8e4``, ``lr=2e-4``.

``DagmaMLP`` requires double precision, which the wrapper sets and restores
globally, and runs on CPU by default; ``device`` can be overridden.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

#: Paper / package defaults for the nonlinear model.
DEFAULT_PARAMS: Dict[str, Any] = {
    "lambda1": 0.02,
    "lambda2": 0.005,
    "hidden_units": 10,
    "bias": True,
    "T": 4,
    "mu_init": 0.1,
    "mu_factor": 0.1,
    "s": 1.0,
    "warm_iter": 50000,
    "max_iter": 80000,
    "lr": 2e-4,
    "device": "cpu",
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Fit nonlinear DAGMA (per-node MLP) on *X*.

    Args:
        X: ``(n_samples, N)`` design matrix, columns ordered ``[S..., X...]``.
        **params: Overrides of :data:`DEFAULT_PARAMS`.  ``seed`` seeds torch for
            reproducible MLP initialisation.

    Returns:
        :class:`BenchmarkResult` with the raw ``W`` in paper orientation.  Like
        NOTEARS-MLP, the nonlinear model reports edge *magnitudes*, so ``W`` is
        non-negative.
    """
    import torch  # local import: optional dependency chain

    from dagma.nonlinear import DagmaMLP, DagmaNonlinear

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)      # thresholding happens in postprocess
    seed = merged.pop("seed", 0)

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    d = int(X.shape[1])
    hidden = int(merged["hidden_units"])
    device = str(merged["device"])

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        torch.manual_seed(int(seed))
        eq_model = DagmaMLP(
            dims=[d, hidden, 1], bias=bool(merged["bias"]), dtype=torch.double
        ).to(device)
        model = DagmaNonlinear(eq_model, dtype=torch.double)

        X_torch = torch.from_numpy(X).to(device)

        start = time.perf_counter()
        W = model.fit(
            X_torch,
            lambda1=float(merged["lambda1"]),
            lambda2=float(merged["lambda2"]),
            T=int(merged["T"]),
            mu_init=float(merged["mu_init"]),
            mu_factor=float(merged["mu_factor"]),
            s=float(merged["s"]),
            warm_iter=int(merged["warm_iter"]),
            max_iter=int(merged["max_iter"]),
            lr=float(merged["lr"]),
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
            "implementation": "dagma.nonlinear.DagmaMLP + DagmaNonlinear",
            "dims": [d, hidden, 1],
        },
    )
