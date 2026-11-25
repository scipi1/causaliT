"""
All metrics of causaliT, as pure functions (numpy / torch in, numbers out).

Used by training callbacks, the forecaster, evaluation scripts and the
benchmark runner.  No Lightning, no config, no file I/O here.

Sections:
    1. Full-graph structure metrics   (paper orientation, benchmarks)
    2. Attention-block DAG metrics    (model orientation, callbacks + eval)
    3. Markov equivalence class (MEC) (model orientation)
    4. Causal order                   (model orientation)
    5. Prediction (regression) metrics
    6. Interventional (ATE) metrics
    7. Attention diagnostics

Orientation conventions (they differ by section):
    * paper orientation:  A[i, j] != 0  means  i -> j   (section 1)
    * model orientation:  A[i, j] != 0  means  j -> i   (sections 2-4):
      rows = children / attention queries, columns = parents / keys.
      Attention matrices and the dataset ``*_att_mask.csv`` files use it.

Coding Assistant:
    * Add new metrics HERE, never inline in callbacks / forecasters / scripts.
    * Keep functions pure; loading true DAGs lives in
      ``evaluation/eval_funs/helpers/eval_utils.py``.
    * Third-party imports (sklearn, networkx, gadjid, CausalDisco) are lazy so
      a missing package only disables its metric.
    * To switch orientation, transpose the input; do not add flags.
    * Inapplicable metrics return ``None`` (section 1) or NaN (others).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

#: Number of equal-width bins of the expected calibration error.
ECE_BINS = 10


# =============================================================================
# Internal helpers
# =============================================================================

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
    """Harmonic mean; None if an input is None, 0 if both are 0."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# 1. Full-graph structure metrics (paper orientation: A[i, j] = i -> j)
# =============================================================================

def is_acyclic(A: np.ndarray) -> bool:
    """True when the directed graph of the non-zero entries has no cycle."""
    import networkx as nx

    return bool(nx.is_directed_acyclic_graph(nx.DiGraph(_binary(A))))


def shd(true: np.ndarray, pred: np.ndarray) -> int:
    """
    Structural Hamming distance over unordered pairs.

    A pair costs 1 when its state (none, i->j, j->i, both) differs, so a
    reversed edge costs 1 (CDT / gCastle / CaPS convention).
    """
    T, P = _binary(true), _binary(pred)
    diff = (T != P) | (T.T != P.T)
    return int(np.triu(diff, k=1).sum())


def edge_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """SHD, directed precision / recall / F1, skeleton F1 and edge counts."""
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


def sid(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """
    Structural intervention distance (Peters & Buhlmann, 2015) via ``gadjid``.

    DAG predictions only: CPDAGs / cyclic graphs give ``None`` plus the
    reason in ``sid_note``.
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


def score_metrics(true: np.ndarray, scores: np.ndarray, probabilistic: bool) -> Dict[str, Any]:
    """
    Threshold-free metrics of off-diagonal edge scores.

    AUROC / AUPRC for any monotone score (``|W|`` works); Brier and ECE only
    when *probabilistic* (scores are probabilities).
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


def source_metrics(true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """
    Recovery of the parentless nodes from incoming edge mass.

    * ``source_recall``: fraction of true sources among the L_S nodes with
      the lowest incoming score (ties get fractional credit). Chance ~ L_S/N.
    * ``source_auroc``: AUROC of source labels vs negated incoming score.

    Both ``None`` when all or no nodes are sources.
    """
    out: Dict[str, Any] = {"source_recall": None, "source_auroc": None}
    T = _binary(true)
    y = (T.sum(axis=0) == 0).astype(int)          # column = child
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
    out["source_auroc"] = auroc(-inc, y)
    return out


def sample_metrics(true: np.ndarray, samples: np.ndarray) -> Dict[str, Any]:
    """Expected SHD and acyclic fraction over ``(S, N, N)`` graph samples."""
    samples = np.asarray(samples)
    return {
        "expected_shd": float(np.mean([shd(true, g) for g in samples])),
        "acyclic_fraction": float(np.mean([is_acyclic(g) for g in samples])),
        "n_graph_samples": int(len(samples)),
    }


def data_diagnostics(X: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    """
    Var- and R2-sortability of a dataset (Reisach et al., 2021, 2023).

    Near 1 = the true order is readable from marginal variances (or R2).
    Compute on RAW data: standardisation makes var-sortability meaningless.
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


def structure_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    probabilistic: bool = False,
    order: Optional[Sequence[int]] = None,
    graph_samples: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Every applicable full-graph metric for one fit (benchmark report).

    Args:
        true: Binary true DAG.
        pred: Binary predicted graph (DAG or CPDAG).
        scores: Edge scores for threshold-free metrics (``pred`` if None).
        probabilistic: Whether *scores* are probabilities.
        order: Estimated causal order, most upstream first.
        graph_samples: ``(S, N, N)`` sampled graphs.
    """
    scores = pred if scores is None else scores
    metrics = edge_metrics(true, pred)
    metrics.update(sid(true, pred))
    metrics["order_divergence"] = None if order is None else order_divergence(true, order)
    metrics["order_accuracy"] = None if order is None else order_accuracy(true, order)
    metrics.update(score_metrics(true, scores, probabilistic))
    metrics.update(source_metrics(true, scores))
    if graph_samples is None:
        metrics.update({"expected_shd": None, "acyclic_fraction": None, "n_graph_samples": None})
    else:
        metrics.update(sample_metrics(true, graph_samples))
    return metrics


def aggregate_seeds(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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



# =============================================================================
# 2. Attention-block DAG metrics (model orientation: A[i, j] = j -> i)
# =============================================================================
# Blocks may be rectangular (cross S->X, shape (L_X, L_S)) or square
# (self X->X, or the full homogeneous (N, N) posterior).

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Rank AUROC via the Mann-Whitney U identity (ties averaged).

    Equals ``sklearn.roc_auc_score`` for binary labels; NaN if a class is
    empty.  Inputs are flattened.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    sv = vals[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def soft_hamming(learned: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute difference between a soft adjacency in [0, 1] and the truth."""
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")
    return float(np.mean(np.abs(learned - true)))


def standard_shd(
    learned: np.ndarray,
    true: np.ndarray,
    threshold: float = 0.5,
    is_cross: bool = False,
) -> Dict[str, Any]:
    """
    Standard SHD = missing + extra + reversed, after binarising at *threshold*.

    Reversals are counted only on square blocks (``is_cross=False``); each
    costs 1 (GraN-DAG convention).  Also returns tp / tn and TPR, FDR,
    precision in PERCENT.

    Coding Assistant: missing / extra are cell-wise, so one reversed edge
    costs 3 here (missing + extra + reversed) but 1 in :func:`shd`.  Kept
    for continuity with past reports; do not change without renaming.
    """
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")

    pred = (learned >= threshold).astype(int)
    tb = true.astype(int)
    missing = int(((tb == 1) & (pred == 0)).sum())
    extra = int(((tb == 0) & (pred == 1)).sum())

    reversed_edges = 0
    if not is_cross and pred.shape[0] == pred.shape[1]:
        rev = (pred == 1) & (pred.T == 0) & (tb == 0) & (tb.T == 1)
        reversed_edges = int(np.triu(rev | rev.T, k=1).sum())

    tp = int(((tb == 1) & (pred == 1)).sum())
    tn = int(((tb == 0) & (pred == 0)).sum())
    nan = float("nan")
    return {
        "shd": missing + extra + reversed_edges,
        "missing": missing,
        "extra": extra,
        "reversed": reversed_edges,
        "n_true_edges": int(tb.sum()),
        "n_pred_edges": int(pred.sum()),
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "tpr": 100.0 * tp / (tp + missing) if (tp + missing) else nan,
        "fdr": 100.0 * extra / (extra + tp) if (extra + tp) else nan,
        "precision": 100.0 * tp / (tp + extra) if (tp + extra) else nan,
    }


def zeroness_metrics(learned: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    """
    Edge / non-edge separation of a soft adjacency.

    Returns mean / max |value| on non-edges, mean / min value on edges, their
    ``contrast`` (mean_edge - mean_nonedge, higher = better) and counts.
    """
    if learned.shape != true.shape:
        raise ValueError(f"Shape mismatch: learned {learned.shape} vs true {true.shape}")
    edge = true.astype(bool)
    non = np.abs(learned[~edge])
    on = learned[edge]
    nan = float("nan")
    mean_non = float(non.mean()) if non.size else nan
    mean_on = float(on.mean()) if on.size else nan
    return {
        "mean_nonedge": mean_non,
        "max_nonedge": float(non.max()) if non.size else nan,
        "mean_edge": mean_on,
        "min_edge": float(on.min()) if on.size else nan,
        "contrast": mean_on - mean_non,
        "n_edges": int(edge.sum()),
        "n_nonedges": int((~edge).sum()),
    }


def dag_confidence(fold_dags: List[np.ndarray]) -> float:
    """
    Agreement of soft DAGs across folds: ``1 - 2 * mean(std over folds)``.

    1 = identical folds, 0 = maximal disagreement; 1.0 for fewer than 2 folds.
    """
    if len(fold_dags) < 2:
        return 1.0
    conf = 1.0 - 2.0 * np.mean(np.std(np.stack(fold_dags, axis=0), axis=0))
    return float(np.clip(conf, 0.0, 1.0))



def _closure(g: np.ndarray) -> np.ndarray:
    """Transitive closure of a boolean square relation (fixpoint)."""
    tc = g.copy()
    for _ in range(g.shape[0]):
        new = tc | (tc @ g)
        if (new == tc).all():
            break
        tc = new
    return tc


def ancestor_not_parent_mask(true: np.ndarray) -> np.ndarray:
    """
    Cells (i, j) where j is a transitive ancestor of i but not a parent.

    All-False for non-square input.
    """
    g = np.asarray(true) > 0.5
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        return np.zeros_like(g)
    return _closure(g) & ~g


def descendant_mask(true: np.ndarray) -> np.ndarray:
    """
    Cells (i, j) where j is a descendant of i (children included).

    This is the anti-causal channel.  All-False for non-square input.
    """
    g = np.asarray(true) > 0.5
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        return np.zeros_like(g)
    return _closure(g).T.copy()


def causal_class_masks(true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Partition of a block into causal classes for diagnostics.

    Square blocks: parents / ancestors (not parents) / descendants / others
    (diagonal excluded).  Rectangular blocks: parents / others.
    """
    true = np.asarray(true)
    parents = true > 0.5
    if true.shape[0] != true.shape[1]:
        return {"parents": parents, "others": ~parents}
    anc = ancestor_not_parent_mask(true)
    dsc = descendant_mask(true)
    others = ~(parents | anc | dsc) & ~np.eye(true.shape[0], dtype=bool)
    return {"parents": parents, "ancestors": anc, "descendants": dsc, "others": others}


def attention_mass_metrics(att: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Threshold-free quality of one attention block against the true mask.

    Keys (missing when not defined):
        auroc, contrast           -- ranking / mean gap edges vs non-edges
        mass_on_{edges,parents}   -- share of attention mass on true edges
        mass_on_ancestors         -- share on transitive ancestors (square)
        parent_vs_ancestor_contrast
        mass_on_descendants       -- share on descendants (square)
        mass_on_others            -- remainder, diagonal excluded

    Empty dict if the block has no edge or no non-edge.
    """
    att = np.asarray(att, dtype=float)
    s, y = att.ravel(), np.asarray(true).ravel()
    on, off = s[y > 0.5], s[y <= 0.5]
    if on.size == 0 or off.size == 0:
        return {}
    out = {"auroc": auroc(s, y), "contrast": float(on.mean() - off.mean())}

    total = float(s.sum())
    ok = total > 1e-12
    if ok:
        out["mass_on_edges"] = float(on.sum() / total)
        out["mass_on_parents"] = out["mass_on_edges"]

    anc = ancestor_not_parent_mask(true).ravel()
    anc_mass = 0.0
    if anc.any():
        sa = s[anc]
        anc_mass = float(sa.sum())
        if ok:
            out["mass_on_ancestors"] = anc_mass / total
        out["parent_vs_ancestor_contrast"] = float(on.mean() - sa.mean())

    dsc = descendant_mask(true).ravel()
    dsc_mass = 0.0
    if dsc.any():
        dsc_mass = float(s[dsc].sum())
        if ok:
            out["mass_on_descendants"] = dsc_mass / total

    if ok:
        others = total - float(on.sum()) - anc_mass - dsc_mass
        if att.shape[0] == att.shape[1]:
            others -= float(np.diag(att).sum())
        out["mass_on_others"] = others / total
    return out


def class_means(values: np.ndarray, true: np.ndarray) -> Dict[str, Tuple[float, int]]:
    """
    NaN-safe mean of *values* over each class of :func:`causal_class_masks`.

    Returns ``{class: (mean, n_cells)}``; classes without finite cells are
    omitted.  Used for the raw pair-HSIC diagnostics (``hsic_class/*``).
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = {}
    for cls, m in causal_class_masks(true).items():
        sel = m & finite
        if sel.any():
            out[cls] = (float(values[sel].mean()), int(sel.sum()))
    return out



# =============================================================================
# 3. Markov equivalence class (model orientation: A[i, j] = j -> i)
# =============================================================================

def dag_to_skeleton(dag: np.ndarray) -> set:
    """Undirected edges ``{frozenset({i, j})}`` of a (soft) adjacency (> 0)."""
    n = dag.shape[0]
    return {frozenset({i, j}) for i in range(n) for j in range(i + 1, n)
            if dag[i, j] > 0 or dag[j, i] > 0}


def find_v_structures(dag: np.ndarray) -> set:
    """V-structures p1 -> c <- p2 (p1, p2 non-adjacent) as ``(c, min, max)``."""
    n = dag.shape[0]
    skel = dag_to_skeleton(dag)
    out = set()
    for c in range(n):
        parents = [j for j in range(n) if dag[c, j] > 0]
        for k, p1 in enumerate(parents):
            for p2 in parents[k + 1:]:
                if frozenset({p1, p2}) not in skel:
                    out.add((c, min(p1, p2), max(p1, p2)))
    return out


def soft_skeleton_distance(learned: np.ndarray, true_skeleton: set) -> Tuple[float, Dict[str, float]]:
    """
    ``1 - F1`` of a soft skeleton; edge strength = max(A[i, j], A[j, i]).

    Recall = mean strength on true pairs, precision = 1 - mean on non-pairs.
    """
    n = learned.shape[0]
    pairs = {frozenset({i, j}) for i in range(n) for j in range(i + 1, n)}

    def strength(e):
        i, j = tuple(e)
        return max(learned[i, j], learned[j, i])

    recall = float(np.mean([strength(e) for e in true_skeleton])) if true_skeleton else 1.0
    false = pairs - true_skeleton
    precision = 1.0 - float(np.mean([strength(e) for e in false])) if false else 1.0
    f1 = _f1(recall, precision)
    return 1.0 - f1, {
        "recall": recall, "precision": precision, "f1": f1,
        "n_true_edges": len(true_skeleton), "n_possible_edges": len(pairs),
    }


def soft_v_structure_distance(
    learned: np.ndarray, true_v: set, true_skeleton: set,
) -> Tuple[float, Dict[str, float]]:
    """
    ``1 - F1`` of soft v-structure recovery.

    Recall: per true v-structure, min(parent edges) * (1 - parent adjacency).
    Precision: 1 - mean strength of spurious candidates (edges > 0.3).
    *true_skeleton* is unused, kept for signature symmetry.
    """
    n = learned.shape[0]
    if true_v:
        recall = float(np.mean([
            min(learned[c, p1], learned[c, p2])
            * (1.0 - max(learned[p1, p2], learned[p2, p1]))
            for (c, p1, p2) in true_v
        ]))
    else:
        recall = 1.0

    spurious = []
    for c in range(n):
        cand = [(j, learned[c, j]) for j in range(n) if j != c and learned[c, j] > 0.3]
        for k, (p1, s1) in enumerate(cand):
            for p2, s2 in cand[k + 1:]:
                if (c, min(p1, p2), max(p1, p2)) not in true_v:
                    adj = max(learned[p1, p2], learned[p2, p1])
                    spurious.append(min(s1, s2) * (1.0 - adj))
    precision = max(0.0, 1.0 - float(np.mean(spurious))) if spurious else 1.0
    f1 = _f1(recall, precision)
    return 1.0 - f1, {
        "recall": recall, "precision": precision, "f1": f1,
        "n_true_v_structures": len(true_v), "n_spurious_candidates": len(spurious),
    }


def mec_distance(learned: np.ndarray, true_dag: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """Soft MEC distance = mean of skeleton and v-structure distances (0 = same MEC)."""
    skel = dag_to_skeleton(true_dag)
    v = find_v_structures(true_dag)
    sd, sdet = soft_skeleton_distance(learned, skel)
    vd, vdet = soft_v_structure_distance(learned, v, skel)
    dist = (sd + vd) / 2.0
    return dist, {
        "mec_distance": dist,
        "skeleton_distance": sd,
        "skeleton_recall": sdet["recall"],
        "skeleton_precision": sdet["precision"],
        "skeleton_f1": sdet["f1"],
        "v_structure_distance": vd,
        "v_structure_recall": vdet["recall"],
        "v_structure_precision": vdet["precision"],
        "v_structure_f1": vdet["f1"],
        "n_true_skeleton_edges": sdet["n_true_edges"],
        "n_true_v_structures": vdet["n_true_v_structures"],
    }


def mec_membership(
    learned: np.ndarray, true_dag: np.ndarray, threshold: float = 0.5,
) -> Tuple[bool, Dict[str, Any]]:
    """True if the binarised *learned* has the true skeleton and v-structures."""
    lb = (learned >= threshold).astype(float)
    ts, ls = dag_to_skeleton(true_dag), dag_to_skeleton(lb)
    tv, lv = find_v_structures(true_dag), find_v_structures(lb)
    in_mec = ts == ls and tv == lv
    return in_mec, {
        "in_mec": in_mec,
        "threshold": threshold,
        "skeleton_match": ts == ls,
        "v_structure_match": tv == lv,
        "n_true_skeleton_edges": len(ts),
        "n_learned_skeleton_edges": len(ls),
        "n_missing_edges": len(ts - ls),
        "n_extra_edges": len(ls - ts),
        "missing_edges": [tuple(e) for e in ts - ls],
        "extra_edges": [tuple(e) for e in ls - ts],
        "n_true_v_structures": len(tv),
        "n_learned_v_structures": len(lv),
        "n_missing_v_structures": len(tv - lv),
        "n_extra_v_structures": len(lv - tv),
        "missing_v_structures": list(tv - lv),
        "extra_v_structures": list(lv - tv),
    }


def mec_threshold(learned: np.ndarray, true_dag: np.ndarray) -> Tuple[Optional[float], bool]:
    """
    Largest binarisation threshold at which *learned* is in the true MEC.

    Higher = more discriminative scores.  ``(None, False)`` if none works.
    """
    top = float(np.max(learned)) if learned.size else 0.0
    eps = max(top * 1e-9, 1e-12)
    cands = np.unique(np.concatenate([[0.0], learned.ravel(), [top + eps]]))
    for theta in cands[::-1]:
        if mec_membership(learned, true_dag, threshold=float(theta))[0]:
            return float(theta), True
    return None, False



# =============================================================================
# 4. Causal order
# =============================================================================

def order_divergence(true: np.ndarray, order: Sequence[int]) -> int:
    """
    Number of true edges pointing backwards in *order* (Rolland et al., 2022).

    Paper orientation.  0 = the true graph fits the complete DAG of *order*.
    """
    pos = _positions(order)
    parents, children = np.nonzero(_binary(true))
    return int((pos[parents] > pos[children]).sum())


def order_accuracy(true: np.ndarray, order: Sequence[int]) -> Optional[float]:
    """
    Fraction of true ancestor -> descendant pairs placed in the right order.

    Paper orientation.  ``None`` for an edgeless graph.
    """
    import networkx as nx

    reach = nx.transitive_closure_dag(nx.DiGraph(_binary(true)))
    pairs = [(a, d) for a, d in reach.edges() if a != d]
    if not pairs:
        return None
    pos = _positions(order)
    return float(np.mean([pos[a] < pos[d] for a, d in pairs]))


def order_agreement(upstreamness: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    """
    Agreement of a per-node upstreamness score with the true causal order.

    Model orientation (``true[i, j] = 1`` means j -> i); larger score = more
    upstream.  Returns ``(pair_acc, kendall)``:

    * pair_acc: fraction of true ancestor pairs scored in the right order
      (ties 1/2); 0.5 = chance.
    * kendall: concordance with the true topological depth over pairs of
      distinct depth; +1 = perfect, 0 = none.
    """
    g = np.asarray(true) > 0.5
    n = g.shape[0]
    anc = _closure(g)
    np.fill_diagonal(anc, False)

    depth = np.zeros(n, dtype=float)              # longest path from a source
    for _ in range(n):
        new = depth.copy()
        for i in range(n):
            pa = np.where(g[i])[0]
            if pa.size:
                new[i] = max(new[i], depth[pa].max() + 1)
        if np.array_equal(new, depth):
            break
        depth = new

    u = np.asarray(upstreamness, dtype=float).ravel()
    ii, jj = np.where(anc)
    if ii.size:
        d = u[jj] - u[ii]
        pair_acc = float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / ii.size)
    else:
        pair_acc = float("nan")

    conc = disc = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            if depth[a] == depth[b]:
                continue
            true_sign = 1.0 if depth[a] < depth[b] else -1.0
            s = np.sign(u[a] - u[b])
            if s == 0:
                conc += 0.5
                disc += 0.5
            elif s == true_sign:
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    kendall = float((conc - disc) / tot) if tot else float("nan")
    return pair_acc, kendall



# =============================================================================
# 5. Prediction (regression) metrics
# =============================================================================
# Coding Assistant: the forecaster keeps stateful ``torchmetrics.R2Score``
# objects for epoch-level logging (Lightning accumulates them across
# batches).  The functions below are the stateless equivalents for one
# tensor pair; use them for ad-hoc logging, evaluation and tests.

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements."""
    return (pred - target).square().mean()


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all elements."""
    return (pred - target).abs().mean()


def r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pooled coefficient of determination over all elements (1 = perfect)."""
    t = target.reshape(-1)
    sst = (t - t.mean()).square().sum().clamp_min(1e-12)
    return 1 - (pred.reshape(-1) - t).square().sum() / sst


def per_node_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE per node over the batch: ``(B, N) -> (N,)``."""
    return (pred - target).square().mean(dim=0)


#: Per-feature metrics on ``(n, D)`` tensors, each returning ``(D,)``.
PREDICTION_METRICS = {
    "r2": lambda p, t: 1 - (p - t).square().sum(0)
    / (t - t.mean(0)).square().sum(0).clamp_min(1e-12),
    "mse": lambda p, t: (p - t).square().mean(0),
    "mae": lambda p, t: (p - t).abs().mean(0),
    "rmse": lambda p, t: (p - t).square().mean(0).sqrt(),
    "mape": lambda p, t: ((p - t).abs() / t.abs().clamp_min(1.17e-6)).mean(0),
}
DEFAULT_PREDICTION_METRICS = ["r2", "mse", "mae", "rmse"]


def _as_3d(y_pred: torch.Tensor, y_true: torch.Tensor, feat_idx: Optional[int]):
    """Squeeze, select the target feature and reshape both to ``(B, L, D)``."""
    if y_pred.ndim >= 2 and y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_true.ndim >= 2 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
    if y_pred.shape != y_true.shape and y_true.ndim == y_pred.ndim + 1:
        if feat_idx is None:
            raise ValueError(
                f"Predictions {tuple(y_pred.shape)} are univariate but targets "
                f"{tuple(y_true.shape)} are not; pass target_feature_idx."
            )
        y_true = y_true[..., feat_idx]
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}")
    if y_pred.ndim not in (1, 2, 3):
        raise ValueError(f"Expected 1D-3D tensors, got {tuple(y_pred.shape)}")
    while y_pred.ndim < 3:
        y_pred, y_true = y_pred.unsqueeze(-1), y_true.unsqueeze(-1)
    return y_pred, y_true


def _metric_values(yp: torch.Tensor, yt: torch.Tensor, names: List[str]) -> Dict[str, torch.Tensor]:
    """Per-feature values on ``(n, D)``; NaN when undefined (no data, r2 with n < 2)."""
    nan = torch.full((yt.shape[-1],), float("nan"))
    return {
        n: nan if yt.shape[0] == 0 or (n == "r2" and yt.shape[0] < 2)
        else PREDICTION_METRICS[n](yp, yt).detach().cpu()
        for n in names
    }



@torch.no_grad()
def prediction_metrics(
    results: Any,
    feature_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    mask: Optional[torch.Tensor] = None,
    per_sample: bool = True,
    target_feature_idx: Optional[int] = None,
) -> pd.DataFrame:
    """
    Regression metrics of a ``PredictionResult`` as a DataFrame.

    Args:
        results: Object with ``outputs``, ``targets`` and ``metadata``.
        feature_names: Labels of the D features (default ``f0, f1, ...``).
        metrics: Subset of :data:`PREDICTION_METRICS` (default r2/mse/mae/rmse).
        mask: ``(B, L)`` validity mask; default = targets not all-NaN.
        per_sample: One row per sample (True) or pooled (False).
        target_feature_idx: Target feature when predictions are univariate;
            default from ``metadata['val_idx']``.

    Returns:
        Columns ``index`` (per-sample), ``feature`` (D > 1), and one
        UPPERCASE column per metric.
    """
    names = metrics or DEFAULT_PREDICTION_METRICS
    unknown = set(names) - set(PREDICTION_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; available {list(PREDICTION_METRICS)}")

    y_true = torch.as_tensor(results.targets)
    y_pred = torch.as_tensor(results.outputs).to(y_true.device)
    meta = getattr(results, "metadata", None) or {}
    if target_feature_idx is None and y_pred.shape != y_true.shape:
        target_feature_idx = meta.get("val_idx")
        if target_feature_idx is None:
            target_feature_idx = meta.get("config", {}).get("data", {}).get("val_idx")
    y_pred, y_true = _as_3d(y_pred.float(), y_true.float(), target_feature_idx)
    B, L, D = y_true.shape

    if feature_names is not None and D > 1 and len(feature_names) != D:
        raise ValueError(f"feature_names length {len(feature_names)} must equal D={D}")
    labels = (feature_names or [f"f{d}" for d in range(D)]) if D > 1 else None

    if mask is None:
        mask = ~torch.isnan(y_true).all(dim=-1)
    elif tuple(mask.shape) != (B, L):
        raise ValueError(f"mask must be (B, L); got {tuple(mask.shape)}")
    mask = mask.to(y_true.device).bool()

    def rows(vals, base):
        if labels is None:
            return [{**base, **{n.upper(): float(v[0]) for n, v in vals.items()}}]
        return [{**base, "feature": f, **{n.upper(): float(v[d]) for n, v in vals.items()}}
                for d, f in enumerate(labels)]

    if not per_sample:
        return pd.DataFrame(rows(_metric_values(y_pred[mask], y_true[mask], names), {}))
    out = []
    for b in range(B):
        out.extend(rows(_metric_values(y_pred[b, mask[b]], y_true[b, mask[b]], names),
                        {"index": b}))
    return pd.DataFrame(out)


def compare_predictions(
    *results: Any,
    labels: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """:func:`prediction_metrics` of several results, stacked with a ``model`` column."""
    if not results:
        raise ValueError("At least one PredictionResult required")
    if labels is not None and len(labels) != len(results):
        raise ValueError(f"{len(labels)} labels for {len(results)} results")
    frames = []
    for i, res in enumerate(results):
        df = prediction_metrics(res, feature_names=feature_names, metrics=metrics)
        df.insert(0, "model", labels[i] if labels else f"Model_{i}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_metrics(
    df: pd.DataFrame,
    groupby: Union[str, List[str]] = "feature",
    agg_funcs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate the UPPERCASE metric columns of a metrics DataFrame by *groupby*."""
    cols = [c for c in df.columns if c.isupper() and c not in ("INDEX", "FEATURE", "MODEL")]
    if not cols:
        raise ValueError("No metric columns found in DataFrame")
    return df.groupby(groupby)[cols].agg(agg_funcs or ["mean", "std", "min", "max"])



# =============================================================================
# 6. Interventional (ATE) metrics
# =============================================================================

def ate_errors(
    treated_mean: float, baseline_mean: float, true_ate: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    Model ATE = treated_mean - baseline_mean, and its error vs *true_ate*.

    Inputs must be in RAW (de-normalised) units.  ``abs_error`` is None
    without ground truth; ``rel_error`` is None when |true_ate| <= 1e-10.
    """
    model_ate = treated_mean - baseline_mean
    abs_error = abs(model_ate - true_ate) if true_ate is not None else None
    rel_error = (abs_error / abs(true_ate)
                 if (abs_error is not None and abs(true_ate) > 1e-10) else None)
    return {"model_ate": model_ate, "abs_error": abs_error, "rel_error": rel_error}


# =============================================================================
# 7. Attention diagnostics
# =============================================================================

def attention_entropy(att: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Shannon entropy [nats] over the last (key) dimension.

    ``(..., L, S) -> (..., L)``.  Weights are clamped to >= eps so
    non-softmax attention cannot give negative entropy.
    """
    p = att.clamp(min=eps)
    return torch.nan_to_num(-(p * p.log()).sum(dim=-1), nan=0.0)


def row_entropy_stats(weights: torch.Tensor) -> Dict[str, float]:
    """
    Entropy statistics of row-wise pair weights (detached).

    Rows are L1-renormalised first, so the stats hold for non-stochastic
    weightings too.  Returns mean / min / max entropy [nats], ``norm_mean``
    (H / ln K, 1 = uniform), ``eff_competitors_mean`` (exp H) and the raw
    ``row_mass_mean``.  All 0.0 when no row has mass.
    """
    w = weights.detach()
    row_mass = w.sum(dim=1)
    live = row_mass > 0
    if not bool(live.any()):
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "norm_mean": 0.0,
                "eff_competitors_mean": 0.0, "row_mass_mean": 0.0}
    w_live = w[live] / row_mass[live].unsqueeze(1)
    h = -(w_live * torch.log(w_live.clamp_min(1e-12))).sum(dim=1)
    k = (w_live > 0).sum(dim=1).clamp_min(1).to(h.dtype)
    norm = h / torch.log(k.clamp_min(2.0))        # K=1 row -> H=0; guard ln(1)
    return {
        "mean": float(h.mean()),
        "min": float(h.min()),
        "max": float(h.max()),
        "norm_mean": float(norm.mean()),
        "eff_competitors_mean": float(torch.exp(h).mean()),
        "row_mass_mean": float(row_mass[live].mean()),
    }

