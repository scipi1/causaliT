"""
CaPS (Xu et al., NeurIPS 2024): order search for mixed linear / nonlinear ANMs.

Pipeline, as in the authors' ``train_order.py``:

1. ``compute_top_order``: leaf-by-leaf removal with the Stein Hessian
   (Lemma 1 of the paper), plus a ``parents_score`` matrix;
2. pre-pruning of the complete DAG with ``parents_score`` (``lambda1``);
3. CAM pruning with cutoff 0.001;
4. edge supplement from ``parents_score`` (``lambda2``), keeping acyclicity.

Steps 1, 2 and 4 are the vendored authors' code (``vendor/caps``).  Step 3 is
dodiscover's Python ``CAMPruning`` instead of the authors' R script; see
``vendor/caps/PROVENANCE.md``.  Defaults are the repository's argparse
defaults (``pre_pruning=True``, ``add_edge=True``, ``lambda1=lambda2=50``,
``eta_G=eta_H=1e-3``).  The Stein estimator is O(n^2) in memory, so
``max_samples`` in the benchmark config is worth setting for large n.
"""

import time
from typing import Any, Dict

import numpy as np

from causaliT.benchmarks.base import BenchmarkResult

DEFAULT_PARAMS: Dict[str, Any] = {
    "pre_pruning": True,
    "add_edge": True,
    "lambda1": 50.0,
    "lambda2": 50.0,
    "eta_G": 0.001,
    "eta_H": 0.001,
    "cutoff": 0.001,
    "n_splines": 10,
    "device": "cpu",
}


def fit(X: np.ndarray, **params: Any) -> BenchmarkResult:
    """Run CaPS on *X*; returns a binary DAG (paper orientation) and the order."""
    import networkx as nx
    import torch
    from dodiscover.toporder._base import CAMPruning

    from causaliT.benchmarks.vendor.caps.caps import (
        add_edge_with_parents_score,
        compute_top_order,
        full_DAG,
        pre_pruning_with_parents_score,
    )

    merged = {**DEFAULT_PARAMS, **params}
    merged.pop("w_threshold", None)
    seed = int(merged.pop("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    start = time.perf_counter()

    order, parents_score = compute_top_order(
        torch.Tensor(X), eta_G=float(merged["eta_G"]), eta_H=float(merged["eta_H"]),
        dispersion="mean", device=str(merged["device"]),
    )
    order = [int(i) for i in order]

    init_dag = full_DAG(order)
    if merged["pre_pruning"]:
        init_dag = pre_pruning_with_parents_score(init_dag, parents_score, float(merged["lambda1"]))

    # dodiscover recovers the order from the row sums of a COMPLETE DAG, which a
    # pre-pruned DAG is not; restricting a complete DAG keeps both pieces right.
    pruner = CAMPruning(alpha=float(merged["cutoff"]), n_splines=int(merged["n_splines"]))
    complete = full_DAG(order)
    allowed = (init_dag != 0)
    dag = pruner.prune(X, complete, nx.DiGraph(), _excluded(complete, allowed))

    if merged["add_edge"]:
        dag = add_edge_with_parents_score(dag.astype(np.int32), parents_score, float(merged["lambda2"]))
    seconds = time.perf_counter() - start

    return BenchmarkResult(
        W=np.asarray(dag, dtype=float),
        seconds=seconds,
        params={**merged, "seed": seed},
        is_binary=True,
        output_type="dag",
        order=order,
        extra={"implementation": "vendor/caps (E2real/CaPS 325a4ff) + dodiscover CAMPruning",
               "pruning": "Python CAMPruning, not the authors' R script"},
    )


def _excluded(complete: np.ndarray, allowed: np.ndarray):
    """Graph of complete-DAG edges removed by pre-pruning (passed as excluded)."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(complete.shape[0]))
    rows, cols = np.nonzero((complete != 0) & ~allowed)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G
