"""
Full-graph structure-learning metrics shared by every benchmark method.

All matrices here use **paper orientation** (``A[i, j] != 0`` means ``i -> j``)
and cover the whole ``(N, N)`` graph, unlike the block-wise ``dag_metrics``
keys (``standard_shd_cross`` ...), which are kept unchanged.

Metric groups (see docs/documentation/BENCHMARKS.md):

=================  =========================================================
graph              shd, shd_normalized, precision, recall, f1, skeleton_f1,
                   edge_ratio, is_dag
interventional     sid, sid_normalized          (gadjid; DAG predictions only)
order              order_divergence, order_accuracy  (methods with an order)
probabilistic      auroc, auprc                 (any edge score)
                   brier, ece                   (probability outputs only)
                   expected_shd, acyclic_fraction (graph samples only)
source             source_recall, source_auroc  (any edge score)
data diagnostics   var_sortability, r2_sortability  (per dataset, CausalDisco)
=================  =========================================================

A metric that does not apply is reported as ``None``, never imputed.
Third-party imports are lazy, so a missing package disables only its metrics.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

#: Number of equal-width bins of the expected calibration error.
ECE_BINS = 10


def _binary(A: np.ndarray) -> np.ndarray:
    """Off-diagonal 0/1 copy of *A*."""
    B = (np.asarray(A) != 0).astype(np.int8)
    np.fill_diagonal(B, 0)
    return B


def _off_diagonal(A: np.ndarray) -> np.ndarray:
    """Flattened off-diagonal entries of a square matrix."""
    A = np.asarray(A)
    return A[~np.eye(A.shape[0], dtype=bool)]


def _positions(order: Sequence[int]) -> np.ndarray:
    """``pos[node]`` = index of *node* in *order*."""
    order = np.asarray(order, dtype=int)
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))
    return pos


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_acyclic(A: np.ndarray) -> bool:
    """True when the directed graph of the non-zero entries has no cycle."""
    import networkx as nx

    return bool(nx.is_directed_acyclic_graph(nx.DiGraph(_binary(A))))


# =============================================================================
# Graph metrics
# =============================================================================

def shd(true: np.ndarray, pred: np.ndarray) -> int:
    """
    Structural Hamming distance over unordered pairs.

    Each pair ``{i, j}`` contributes 1 when its state (none, ``i->j``, ``j->i``,
    both) differs.  A reversed edge costs 1 (CDT / gCastle / CaPS convention);
    an undirected CPDAG edge (both directions) costs 1 against a true edge.
    """
    T, P = _binary(true), _binary(pred)
    diff = (T != P) | (T.T != P.T)
    return int(np.triu(diff, k=1).sum())


def edge_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """Directed and skeleton recovery metrics of a binary prediction."""
    T, P = _binary(true), _binary(pred)
    n_true, n_pred = int(T.sum()), int(P.sum())
    tp = int((T & P).sum())
    precision = tp / n_pred if n_pred else None
    recall = tp / n_true if n_true else None

    ST = np.triu(T | T.T, k=1)
    SP = np.triu(P | P.T, k=1)
    s_tp = int((ST & SP).sum())
    s_prec = s_tp / int(SP.sum()) if SP.sum() else None
    s_rec = s_tp / int(ST.sum()) if ST.sum() else None

    dist = shd(T, P)
    return {
        "shd": dist,
        "shd_normalized": dist / n_true if n_true else None,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "skeleton_f1": _f1(s_prec, s_rec),
        "edge_ratio": n_pred / n_true if n_true else None,
        "n_edges_pred": n_pred,
        "n_edges_true": n_true,
        "is_dag": is_acyclic(P),
    }



# =============================================================================
# Interventional metric
# =============================================================================

def sid(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """
    Structural intervention distance (Peters & Buhlmann, 2015) via ``gadjid``.

    DAG predictions only: CPDAGs (PC) and cyclic graphs get ``None`` plus the
    reason in ``sid_note``, rather than the score of an arbitrary DAG.
    """
    out: Dict[str, Any] = {"sid": None, "sid_normalized": None, "sid_note": None}
    P = _binary(pred)
    if not is_acyclic(P):
        out["sid_note"] = "prediction is not a DAG"
        return out
    try:
        import gadjid
    except ImportError:
        out["sid_note"] = "gadjid not installed"
        return out
    normalized, count = gadjid.sid(_binary(true), P, edge_direction="from row to column")
    out["sid"] = int(count)
    out["sid_normalized"] = float(normalized)
    return out


# =============================================================================
# Order metrics
# =============================================================================

def order_divergence(true: np.ndarray, order: Sequence[int]) -> int:
    """
    Number of true edges pointing backwards in *order* (Rolland et al., 2022).

    Zero means the true graph is a subgraph of the complete DAG of the order,
    i.e. a perfect pruning step could still recover it.
    """
    pos = _positions(order)
    parents, children = np.nonzero(_binary(true))
    return int((pos[parents] > pos[children]).sum())


def order_accuracy(true: np.ndarray, order: Sequence[int]) -> Optional[float]:
    """
    Fraction of true ancestor -> descendant pairs placed in the right order.

    Kendall-tau analogue restricted to comparable pairs (pairs without a
    directed path may appear in any order).  ``None`` for an edgeless graph.
    """
    import networkx as nx

    reach = nx.transitive_closure_dag(nx.DiGraph(_binary(true)))
    pairs = [(a, d) for a, d in reach.edges() if a != d]
    if not pairs:
        return None
    pos = _positions(order)
    return float(np.mean([pos[a] < pos[d] for a, d in pairs]))


# =============================================================================
# Probabilistic metrics
# =============================================================================

def score_metrics(true: np.ndarray, scores: np.ndarray, probabilistic: bool) -> Dict[str, Any]:
    """
    Threshold-free metrics of off-diagonal edge scores.

    Args:
        true: Binary true DAG.
        scores: Any monotone edge confidence (``|W|`` works).
        probabilistic: *scores* are probabilities; enables ``brier`` / ``ece``.
    """
    y = _off_diagonal(_binary(true)).astype(int)
    s = _off_diagonal(np.abs(np.asarray(scores, dtype=float)))
    out: Dict[str, Any] = {"auroc": None, "auprc": None, "brier": None, "ece": None}

    if 0 < y.sum() < y.size:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))

    if probabilistic:
        p = np.clip(s, 0.0, 1.0)
        out["brier"] = float(np.mean((p - y) ** 2))
        bins = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
        ece = 0.0
        for b in range(ECE_BINS):
            sel = bins == b
            if sel.any():
                ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
        out["ece"] = float(ece)
    return out


# =============================================================================
# Source metrics
# =============================================================================

def source_metrics(true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """
    Recovery of the source (parentless) nodes from incoming edge mass.

    True sources are the nodes without parents in *true* (the S variables of
    causaliT datasets), so the result does not depend on node ordering.  A node
    looks like a source when its incoming score ``sum_i |scores[i, node]|`` is
    low (same idea as the models' ``dag/source_recall``).

    * ``source_recall`` - recall@L_S: fraction of true sources among the L_S
      nodes with the lowest incoming score.  Ties at the cut-off get
      fractional credit (expected recall over random tie breaking), so
      discrete outputs are not favoured by node order.  Chance ~ L_S / N.
    * ``source_auroc``  - AUROC of true-source labels vs the negated incoming
      score (ties as average ranks; 0.5 = chance).

    Both are ``None`` when all or no nodes are sources.
    """
    out: Dict[str, Any] = {"source_recall": None, "source_auroc": None}
    T = _binary(true)
    y = (T.sum(axis=0) == 0).astype(int)          # paper orientation: column = child
    L_S = int(y.sum())
    if not 0 < L_S < len(y):
        return out
    S = np.abs(np.asarray(scores, dtype=float))
    np.fill_diagonal(S, 0.0)
    inc = S.sum(axis=0)

    cut = np.sort(inc)[L_S - 1]                   # score of the L_S-th lowest
    below, tied = inc < cut, np.isclose(inc, cut)
    below &= ~tied
    n_free = L_S - int(below.sum())               # slots filled from the tie
    hits = y[below].sum() + n_free * y[tied].sum() / tied.sum()
    out["source_recall"] = float(hits / L_S)

    from sklearn.metrics import roc_auc_score

    out["source_auroc"] = float(roc_auc_score(y, -inc))
    return out


def sample_metrics(true: np.ndarray, samples: np.ndarray) -> Dict[str, Any]:
    """Expected SHD and fraction of acyclic graphs over ``(S, N, N)`` samples."""
    samples = np.asarray(samples)
    return {
        "expected_shd": float(np.mean([shd(true, g) for g in samples])),
        "acyclic_fraction": float(np.mean([is_acyclic(g) for g in samples])),
        "n_graph_samples": int(len(samples)),
    }


# =============================================================================
# Data diagnostics
# =============================================================================

def data_diagnostics(X: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    """
    Var- and R2-sortability of a dataset (Reisach et al., 2021, 2023).

    Values near 1 mean the true order can be read off marginal variances (or
    R2) alone, so trivial baselines look strong.  Compute on the RAW data:
    standardisation makes var-sortability uninformative by construction.
    """
    out: Dict[str, Any] = {"var_sortability": None, "r2_sortability": None}
    W = _binary(true).astype(float)
    if W.sum() == 0:
        return out
    try:
        from CausalDisco.analytics import r2_sortability, var_sortability
    except ImportError:
        out["note"] = "CausalDisco not installed"
        return out
    X = np.asarray(X, dtype=float)
    out["var_sortability"] = float(var_sortability(X, W))
    out["r2_sortability"] = float(r2_sortability(X, W))
    return out


# =============================================================================
# Aggregation
# =============================================================================

def compute_structure_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    probabilistic: bool = False,
    order: Optional[Sequence[int]] = None,
    graph_samples: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Every applicable metric for one fit; non-applicable ones are ``None``.

    Args:
        true: Binary true DAG, paper orientation.
        pred: Binary predicted graph (DAG or CPDAG), paper orientation.
        scores: Edge scores for the threshold-free metrics (``pred`` if None).
        probabilistic: Whether *scores* are probabilities.
        order: Estimated causal order, most upstream first.
        graph_samples: ``(S, N, N)`` sampled graphs.
    """
    metrics = edge_metrics(true, pred)
    metrics.update(sid(true, pred))
    metrics["order_divergence"] = None if order is None else order_divergence(true, order)
    metrics["order_accuracy"] = None if order is None else order_accuracy(true, order)
    metrics.update(score_metrics(true, pred if scores is None else scores, probabilistic))
    metrics.update(source_metrics(true, pred if scores is None else scores))
    if graph_samples is None:
        metrics.update({"expected_shd": None, "acyclic_fraction": None, "n_graph_samples": None})
    else:
        metrics.update(sample_metrics(true, graph_samples))
    return metrics


def aggregate(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Mean and std over seeds of every numeric metric (``None`` if never set)."""
    keys = sorted({k for m in per_seed.values() for k in m})
    mean: Dict[str, Any] = {}
    std: Dict[str, Any] = {}
    for key in keys:
        vals = [
            float(m[key]) for m in per_seed.values()
            if isinstance(m.get(key), (int, float, np.integer, np.floating))
        ]
        mean[key] = float(np.mean(vals)) if vals else None
        std[key] = float(np.std(vals)) if vals else None
    return {"mean": mean, "std": std}


__all__ = [
    "ECE_BINS",
    "is_acyclic",
    "shd",
    "edge_metrics",
    "sid",
    "order_divergence",
    "order_accuracy",
    "score_metrics",
    "source_metrics",
    "sample_metrics",
    "data_diagnostics",
    "compute_structure_metrics",
    "aggregate",
]

