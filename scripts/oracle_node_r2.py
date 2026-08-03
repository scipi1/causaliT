"""Per-node ORACLE R2 reference for a generated SCM dataset.

Why this exists
---------------
``val_x_r2`` logged during training is pooled over every node.  In homogeneous
mode that pool includes the exogenous S rows, whose causally-correct R2 is ~0, so
the metric has a ceiling far below 1 and "R2 < 0.5 => weak fit" is meaningless.
The forecaster now logs ``x_r2_endo`` / ``x_r2_macro`` / ``x_r2_src`` instead
(see ``AttentionSelectorForecaster._log_r2_variants``), but a corrected R2 still
needs something to be compared against.  This script produces that reference:

  r2_parents     R2 of node i from its TRUE parents only.  This is what a
                 perfectly-fit CAUSAL model would reach.  ``r2_model << this``
                 => genuine underfit.
  r2_all_others  R2 of node i from ALL other nodes.  In homogeneous mode every
                 node is reconstructed from the other N-1 nodes, so this is the
                 ACHIEVABLE ceiling -- it is > r2_parents whenever descendants
                 or ancestors carry extra information.  ``r2_model`` between the
                 two => the fit is using non-parents (anti-causal / ancestor
                 leakage), which is exactly what ``x_r2_src`` also flags.
  r2_marginal    R2 of node i from ONE candidate at a time.  A parent that is
                 marginally weaker than a non-parent (e.g. a grandparent that is
                 a near-substitute) is a data-level explanation for a wrong edge
                 that no amount of mechanism tuning will fix.

The fit basis is the same additive degree-5 standardised-power basis used by
``scripts/build_scm_equal.py``, so the numbers are directly comparable with
``data/<ds>/equal_strength_report.json`` where that file exists.

Batch-key-dropout (BKD) mode  (``--bkd_p``)
-------------------------------------------
Without BKD the reconstruction loss values a parent by its CONDITIONAL marginal
contribution, i.e. "what does j add GIVEN every other candidate is present".
That number is ~0 for a weak parent and ~0 for a redundant non-parent alike.
BKD drops key columns with probability ``p`` each step, which changes the
objective to the EXPECTED R2 over random candidate subsets; a candidate is then
valued by its marginal contribution AVERAGED over subsets (a Bernoulli-sampled
Shapley value).  ``--bkd_p`` computes both numbers per candidate, so the effect
of switching BKD on can be PREDICTED from the data before training anything:

  marg_cond  R2(all) - R2(all \\ j)                      what training sees now
  marg_bkd   E_{S ~ Bern(1-p)}[R2(S + j) - R2(S)]        what BKD would see
  promotion  marg_bkd / marg_cond                        > 1 => BKD favours j

The catch this exposes: a REDUNDANT non-parent (a near-substitute for a true
parent) also gets promoted, because in the ~p fraction of steps where the true
parent is dropped the substitute is the best predictor left.  So the same knob
that should recover a weak parent can manufacture a spurious edge, and the two
effects are visible in different rows of the table.

Usage
-----
    python scripts/oracle_node_r2.py --dataset scm_equal
    python scripts/oracle_node_r2.py --dataset scm3_continuous --split test
    python scripts/oracle_node_r2.py --dataset scm3_continuous --bkd_p 0.2

Writes ``data/<dataset>/oracle_node_r2.json`` (and
``bkd_marginal_report_p<p>.json`` in BKD mode) and prints a summary table.
"""


from __future__ import annotations

import argparse
import json
from os.path import abspath, dirname, exists, join
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
DEFAULT_DEGREE = 5


# --------------------------------------------------------------------------- #
# Data / DAG loading
# --------------------------------------------------------------------------- #
def load_true_dag(data_dir: str) -> tuple[List[str], np.ndarray]:
    """Read ``dag_adj_mask.csv`` -> (node labels, adjacency with A[i, j] = j -> i)."""
    path = join(data_dir, "dag_adj_mask.csv")
    if not exists(path):
        raise FileNotFoundError(f"missing true DAG: {path}")
    rows = [ln.split(",") for ln in open(path, encoding="utf-8").read().splitlines() if ln]
    labels = [c.strip() for c in rows[0][1:]]
    adj = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    if adj.shape != (len(labels), len(labels)):
        raise ValueError(f"non-square DAG {adj.shape} for {len(labels)} labels")
    return labels, adj


def load_node_values(data_dir: str, split: str, labels: List[str]) -> np.ndarray:
    """Return the ``(n_samples, N)`` VALUE matrix in ``[S ; X]`` node order.

    The exporter stores ``(n, n_var, n_feature)`` blocks with the value channel
    index recorded in ``dataset_metadata.json``.  The S and X blocks are
    concatenated in declaration order, which is the same order as the DAG CSV
    header -- asserted below, because a silent column permutation would relabel
    every parent in the report.
    """
    meta_path = join(data_dir, "dataset_metadata.json")
    val_idx, var_idx = 0, None
    if exists(meta_path):
        fi = json.load(open(meta_path, encoding="utf-8")).get("feature_indices", {})
        val_idx = int(fi.get("value", 0))
        var_idx = fi.get("variable")

    npz = join(data_dir, f"ds_{split}.npz")
    if not exists(npz):
        raise FileNotFoundError(f"missing split: {npz}")
    d = np.load(npz)
    s_raw = np.asarray(d["s"], dtype=np.float64)
    x_raw = np.asarray(d["x"], dtype=np.float64)

    if s_raw.shape[1] + x_raw.shape[1] != len(labels):
        raise ValueError(
            f"S({s_raw.shape[1]}) + X({x_raw.shape[1]}) != {len(labels)} DAG nodes"
        )
    if var_idx is not None:
        # Each block is re-indexed locally by the exporter, so we can only check
        # that the ids are ASCENDING -- which does prove declaration order.
        for name, raw in (("S", s_raw), ("X", x_raw)):
            ids = raw[0, :, int(var_idx)].astype(int).tolist()
            if ids != sorted(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"{name} columns are not in declaration order: {ids}")

    return np.concatenate(
        [s_raw[:, :, val_idx], x_raw[:, :, val_idx]], axis=1
    )                                                   # (n, N)


# --------------------------------------------------------------------------- #
# Fit helpers (identical basis to scripts/build_scm_equal.py)
# --------------------------------------------------------------------------- #
def _basis(col: np.ndarray, degree: int = DEFAULT_DEGREE) -> np.ndarray:
    z = (col - col.mean()) / (col.std() + 1e-12)
    return np.stack([z ** k for k in range(1, degree + 1)], axis=1)


def _r2(y: np.ndarray, design: Optional[np.ndarray]) -> float:
    if design is None or design.shape[1] == 0:
        return 0.0
    A = np.concatenate([np.ones((len(y), 1)), design], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - float(resid.var()) / float(y.var() + 1e-12)


def _design(values: np.ndarray, cols: List[int], degree: int) -> Optional[np.ndarray]:
    if not cols:
        return None
    return np.concatenate([_basis(values[:, c], degree) for c in cols], axis=1)


# --------------------------------------------------------------------------- #
# Fast subset R2 (Gram matrix) -- needed for the BKD expectation
# --------------------------------------------------------------------------- #
class SubsetR2:
    """R2 of any node from any SUBSET of candidates, via one cached Gram matrix.

    The BKD expectation needs thousands of fits (candidates x Monte-Carlo draws),
    which is only tractable if the data are touched once.  With CENTRED columns
    and a centred target, OLS needs no intercept and

        SS_explained = b_sub^T (G_sub^-1 b_sub),   b = D^T y,  G = D^T D

    so every subset fit is a small ``len(cols)``-by-``len(cols)`` solve on cached
    matrices instead of a pass over the samples.  A tiny ridge keeps the solve
    stable when two candidates are near-collinear (exactly the substitute case
    this analysis is about), at a negligible cost in R2.
    """

    def __init__(self, values: np.ndarray, degree: int = DEFAULT_DEGREE,
                 ridge: float = 1e-8):
        self.degree = degree
        self.n_nodes = values.shape[1]
        blocks = [_basis(values[:, j], degree) for j in range(self.n_nodes)]
        D = np.concatenate(blocks, axis=1)
        D = D - D.mean(axis=0)
        Y = values - values.mean(axis=0)
        self.G = D.T @ D
        self.B = D.T @ Y                                  # (N*degree, N)
        self.yy = (Y ** 2).sum(axis=0)                     # (N,)
        self.ridge = ridge * float(np.trace(self.G)) / self.G.shape[0]

    def _cols(self, cand: List[int]) -> np.ndarray:
        return np.concatenate(
            [np.arange(j * self.degree, (j + 1) * self.degree) for j in cand]
        ) if cand else np.empty(0, dtype=int)

    def r2(self, node: int, cand: List[int]) -> float:
        c = self._cols(cand)
        if c.size == 0:
            return 0.0
        G = self.G[np.ix_(c, c)] + self.ridge * np.eye(c.size)
        b = self.B[c, node]
        return float(b @ np.linalg.solve(G, b) / self.yy[node])


def bkd_marginal_report(
    labels: List[str],
    adj: np.ndarray,
    values: np.ndarray,
    p: float,
    n_draws: int = 128,
    degree: int = DEFAULT_DEGREE,
    seed: int = 0,
    max_samples: int = 20_000,
) -> Dict[str, dict]:
    """Conditional vs BKD-averaged marginal contribution of every candidate.

    ``marg_cond`` is what the current objective rewards, ``marg_bkd`` is what an
    objective with key-dropout ``p`` rewards; see the module docstring.  The SAME
    Monte-Carlo masks are reused for the ``S + j`` and ``S`` fits of a candidate
    (a paired / common-random-numbers estimator), so the DIFFERENCE is far more
    accurate than either term and small contributions stay resolvable.
    """
    if len(values) > max_samples:
        values = values[:max_samples]
    sub = SubsetR2(values, degree)
    rng = np.random.default_rng(seed)
    keep_masks = rng.random((n_draws, len(labels))) >= p     # True = key kept

    out: Dict[str, dict] = {}
    for i, child in enumerate(labels):
        cands = [j for j in range(len(labels)) if j != i]
        parents = [j for j in range(len(labels)) if adj[i, j] > 0]
        r2_all = sub.r2(i, cands)
        r2_exp = float(np.mean([
            sub.r2(i, [j for j in cands if m[j]]) for m in keep_masks
        ]))

        marg_cond, marg_bkd = {}, {}
        for j in cands:
            others = [c for c in cands if c != j]
            marg_cond[labels[j]] = r2_all - sub.r2(i, others)
            acc = 0.0
            for m in keep_masks:
                base = [c for c in others if m[c]]
                acc += sub.r2(i, base + [j]) - sub.r2(i, base)
            marg_bkd[labels[j]] = acc / n_draws

        out[child] = {
            "parents": [labels[j] for j in parents],
            "r2_all_candidates": round(r2_all, 4),
            "r2_expected_under_bkd": round(r2_exp, 4),
            "marg_cond": {k: round(v, 5) for k, v in marg_cond.items()},
            "marg_bkd": {k: round(v, 5) for k, v in marg_bkd.items()},
            "promotion": {
                k: (round(marg_bkd[k] / marg_cond[k], 2)
                    if marg_cond[k] > 1e-6 else None)
                for k in marg_cond
            },
        }
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(
    labels: List[str],
    adj: np.ndarray,
    values: np.ndarray,
    degree: int = DEFAULT_DEGREE,
    max_samples: int = 20_000,
) -> Dict[str, dict]:
    """Per-node ``r2_parents`` / ``r2_all_others`` / ``r2_marginal``."""

    if len(values) > max_samples:
        # The fits are least-squares on a small basis; 20k rows already give a
        # R2 stable to ~1e-3 and keep the all-others fit fast.
        values = values[:max_samples]

    out: Dict[str, dict] = {}
    for i, child in enumerate(labels):
        parents = [j for j in range(len(labels)) if adj[i, j] > 0]
        others = [j for j in range(len(labels)) if j != i]
        y = values[:, i]
        out[child] = {
            "parents": [labels[j] for j in parents],
            "r2_parents": round(_r2(y, _design(values, parents, degree)), 4),
            "r2_all_others": round(_r2(y, _design(values, others, degree)), 4),
            "r2_marginal": {
                labels[j]: round(_r2(y, _design(values, [j], degree)), 4)
                for j in others
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="dataset folder under data/")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    ap.add_argument("--data_root", default=join(PROJECT_ROOT, "data"))
    ap.add_argument("--bkd_p", type=float, default=None,
                    help="also report conditional vs BKD-averaged marginal "
                         "contributions at this key-dropout probability")
    ap.add_argument("--bkd_draws", type=int, default=128,
                    help="Monte-Carlo mask draws for --bkd_p (paired estimator)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()


    data_dir = join(args.data_root, args.dataset)
    labels, adj = load_true_dag(data_dir)
    values = load_node_values(data_dir, args.split, labels)
    report = build_report(labels, adj, values, degree=args.degree)

    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "basis": f"additive standardised powers, degree {args.degree}",
        "n_samples_used": int(min(len(values), 20_000)),
        "nodes": report,
        "_how_to_read": (
            "r2_parents = what a perfectly-fit CAUSAL model reaches; "
            "r2_all_others = the ceiling actually available in homogeneous mode "
            "(every node is reconstructed from the other N-1); a model R2 above "
            "r2_parents means non-parents (descendants/ancestors) are being used. "
            "r2_marginal ranks single candidates: a non-parent above a true parent "
            "is a DATA-level explanation for a wrong edge."
        ),
    }
    out_path = join(data_dir, "oracle_node_r2.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"dataset={args.dataset} split={args.split} nodes={len(labels)}")
    print(f"{'node':>6} {'parents':<18} {'r2_parents':>11} {'r2_all_others':>14}"
          f"  best non-parent (marginal)")
    for node, rec in report.items():
        pset = set(rec["parents"])
        nonpar = {k: v for k, v in rec["r2_marginal"].items() if k not in pset}
        best = max(nonpar.items(), key=lambda kv: kv[1]) if nonpar else ("-", 0.0)
        weakest_par = (
            min(((p, rec["r2_marginal"][p]) for p in pset), key=lambda kv: kv[1])
            if pset else ("-", 0.0)
        )
        flag = " <-- outranks weakest parent" if best[1] > weakest_par[1] else ""
        print(
            f"{node:>6} {','.join(rec['parents']) or '-':<18} "
            f"{rec['r2_parents']:>11.4f} {rec['r2_all_others']:>14.4f}"
            f"  {best[0]}={best[1]:.4f}{flag}"
        )
    print(f"\nwritten: {out_path}")

    if args.bkd_p is not None:
        _run_bkd_mode(args, data_dir, labels, adj, values)


def _run_bkd_mode(args, data_dir: str, labels, adj, values) -> None:
    """Print + persist the conditional-vs-BKD marginal contribution comparison."""
    p = float(args.bkd_p)
    bkd = bkd_marginal_report(
        labels, adj, values, p=p, n_draws=args.bkd_draws,
        degree=args.degree, seed=args.seed,
    )
    payload = {
        "dataset": args.dataset,
        "split": args.split,
        "bkd_p": p,
        "n_draws": args.bkd_draws,
        "seed": args.seed,
        "nodes": bkd,
        "_how_to_read": (
            "marg_cond is the marginal R2 of candidate j GIVEN all other "
            "candidates -- what the current objective rewards. marg_bkd is the "
            "same quantity AVERAGED over Bernoulli(1-p) candidate subsets -- what "
            "an objective with batch key dropout p rewards. promotion = "
            "marg_bkd / marg_cond: > 1 means BKD makes j more attractive. A TRUE "
            "PARENT with promotion >> 1 is the intended effect (a weak, "
            "non-redundant parent becomes necessary when its co-parent is "
            "dropped). A NON-PARENT with promotion >> 1 is the cost: it is a "
            "substitute for a true parent and BKD manufactures the spurious edge."
        ),
    }
    out = join(data_dir, f"bkd_marginal_report_p{p:g}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n=== BKD objective prediction (p={p:g}, {args.bkd_draws} draws) ===")
    print("marg_cond = marginal R2 given ALL others | marg_bkd = averaged over "
          "Bernoulli(1-p) subsets")
    for node, rec in bkd.items():
        pset = set(rec["parents"])
        if not pset:
            continue                       # a source has no parents to promote
        print(f"\n{node} <- {','.join(rec['parents'])}   "
              f"R2(all)={rec['r2_all_candidates']:.4f}  "
              f"E[R2] under BKD={rec['r2_expected_under_bkd']:.4f}")
        # True parents first, then the non-parents BKD promotes the most.
        ranked = sorted(rec["marg_bkd"].items(), key=lambda kv: -kv[1])
        for cand, mb in ranked:
            mc = rec["marg_cond"][cand]
            role = "PARENT    " if cand in pset else "non-parent"
            if abs(mb) < 1e-4 and abs(mc) < 1e-4:
                continue                   # irrelevant candidate, keep it quiet
            promo = rec["promotion"][cand]
            promo_s = "   inf" if promo is None else f"{promo:6.2f}"
            print(f"   {role} {cand:>4}  marg_cond={mc: .5f}  "
                  f"marg_bkd={mb: .5f}  promotion={promo_s}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":

    main()
