"""
DAGuerreotype (Zantedeschi et al., ICLR 2023): DAG learning on the permutahedron.

Learns a node score vector ``theta``; SparseMAP over the permutahedron turns it
into a sparse distribution over orders, each complete DAG is sparsified and fit.
This is the closest available relative of causaliT's order-factorized model
(learned order x edges given the order), trained by likelihood rather than
residual independence.

**External install, not vendored.**  The code (BSD-3) needs two compiled
extensions (``lp-sparsemap`` + a Cython/pybind11 module) and is built per
machine, so it cannot be copied as plain Python.  Build it on Linux / the
cluster following the repository's ``linux-install.sh``
(https://github.com/vzantedeschi/DAGuerreotype, commit ``47fc2ae``); ``wandb``
must be importable (it runs in disabled mode).  On Windows it needs MSVC
Build Tools and is not supported here.  The method is reported unavailable
when ``daguerreo`` cannot be imported.

Defaults are the repository's argparse defaults for the bilevel model with
SparseMAP structure, L0 Bernoulli-STE sparsifier and linear equations.  Use
``equations: nonlinear`` for nonlinear data (one hidden layer, ``hidden=50``).
"""

import time
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

DEFAULT_PARAMS: Dict[str, Any] = {
    "joint": False,
    "structure": "sp_map",
    "sparsifier": "l0_ber_ste",
    "equations": "linear",
    "loss": "nll_ev",
    "hidden": 50,
    "optimizer": "adam",
    "lr": 0.1,
    "num_epochs": 5000,
    "es_tol": 100,
    "es_delta": 1e-4,
    "pruning_reg": 0.001,
    "l2_theta": 0.0005,
    "l2_eq": 0.0005,
    "lr_theta": 0.1,
    "eq_optimizer": "sgd",
    "es_tol_inner": 10,
    "es_delta_inner": 1e-4,
    "num_inner_iters": 200,
    "init_theta": "zeros",
    "smap_init": False,
    "smap_iter_k": 100,
    "smax_max_k": 100,
    "nogpu": True,
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """
    Fit DAGuerreotype on *X*.

    Returns:
        Binary MAP DAG in paper orientation (the package's ``W[parent, child]``)
        and the MAP order (ascending ``theta``, see ``structures.map``).
    """
    import torch
    import wandb
    from daguerreo import utils as dg_utils
    from daguerreo.models import Daguerro

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)
    seed = int(merged.pop("seed", 0))
    args = SimpleNamespace(**merged)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)  # as in daguerreo.run_model
    wandb.init(mode="disabled")
    try:
        dg_utils.init_seeds(seed=seed)
        X_torch = torch.from_numpy(np.ascontiguousarray(np.asarray(X, dtype=float)))
        model = Daguerro.initialize(X_torch, args, args.joint)
        model, X_torch = dg_utils.maybe_gpu(args, model, X_torch)
        loss = dg_utils.AVAILABLE[args.loss]

        start = time.perf_counter()
        model(X_torch, loss, args)
        model.eval()
        _, dags = model(X_torch, loss, args)
        seconds = time.perf_counter() - start

        W = (dags[0].detach().cpu().numpy() != 0).astype(float)
        order = torch.argsort(model.structure.theta.view(-1)).cpu().numpy().tolist()
    finally:
        torch.set_default_dtype(previous_dtype)

    return BenchmarkResult(
        W=W,
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=True,
        output_type="dag",
        order=order,
        extra={"implementation": "daguerreo (vzantedeschi/DAGuerreotype 47fc2ae)"},
    )
