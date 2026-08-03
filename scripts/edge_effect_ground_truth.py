"""Ground-truth causal effect of every edge of a dataset's true DAG.

Why
---
Structure metrics (SHD, F1) count every edge equally. That is misleading when the
true edges differ in causal strength by orders of magnitude: on `scm3` the edge
`S5 -> X4` has an average causal effect 24x smaller than `S4 -> X4`, because S5
enters only through the product `e*S5*X2` with `E[X2] ~ 0`. An additive
aggregator `sum_j A_ij V(x_j)` cannot represent such a term at all, so dropping
that edge is the correct answer, not a failure.

This script writes the numbers that let a paper SAY that instead of assuming it,
and reports effect-weighted recall/precision so a missed near-zero-effect edge
costs almost nothing while a missed strong edge costs full.

Usage
-----
    python scripts/edge_effect_ground_truth.py --dataset ds_scm3_continuous
    python scripts/edge_effect_ground_truth.py --dataset ds_scm3_continuous \
        --learned-dag path/to/learned_adjacency.csv

Output
------
    data/<dataset>/edge_effect_ground_truth.json
"""
from __future__ import annotations

import argparse
import json
from os import makedirs
from os.path import join
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from causaliT.paths import ROOT_DIR
from scm_ds.datasets import get_dataset


def data_dir_for(dataset: str) -> str:
    """`ds_scm3_continuous` -> `data/scm3_continuous` (the `ds_` prefix is the
    registry key, the folder drops it)."""
    folder = dataset[3:] if dataset.startswith("ds_") else dataset
    return join(ROOT_DIR, "data", folder)


def weighted_structure_scores(
    edges: List[Dict], learned: Optional[pd.DataFrame],
) -> Optional[Dict[str, float]]:
    """Effect-weighted recall / precision.

    `weighted_recall = sum(effect_std over RECOVERED true edges) /
                       sum(effect_std over ALL true edges)`

    so failing to recover an edge with `effect_std ~ 0` barely moves the score,
    whereas plain recall would punish it exactly like a strong edge.
    `learned` is an adjacency DataFrame with 1 where an edge is present, indexed
    [child, parent] (the convention of `dag_adj_mask.csv`).
    """
    if learned is None:
        return None

    def present(parent: str, child: str) -> bool:
        if child not in learned.index or parent not in learned.columns:
            return False
        return bool(learned.loc[child, parent])

    tot = sum(abs(e["effect_std"]) for e in edges)
    hit = sum(abs(e["effect_std"]) for e in edges if present(e["parent"], e["child"]))
    n_true = len(edges)
    n_hit = sum(1 for e in edges if present(e["parent"], e["child"]))
    n_learned = int(np.asarray(learned.to_numpy() != 0).sum())

    return {
        "recall_plain": n_hit / n_true if n_true else float("nan"),
        "recall_weighted": hit / tot if tot > 0 else float("nan"),
        "precision_plain": n_hit / n_learned if n_learned else float("nan"),
        "n_true_edges": n_true,
        "n_learned_edges": n_learned,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="registry key, e.g. ds_scm3_continuous")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-grid", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negligible-effect", type=float, default=0.02,
                    help="effect_std below which an edge has no average effect")
    ap.add_argument("--modifier-ratio", type=float, default=5.0,
                    help="modifier/effect_std above which an edge is a pure moderator")
    ap.add_argument("--learned-dag", default=None,
                    help="optional adjacency csv (rows=child, cols=parent) to score")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    report = ds.compute_edge_effect_ground_truth(
        n_grid=args.n_grid,
        n_samples=args.n_samples,
        seed=args.seed,
        negligible_effect=args.negligible_effect,
        modifier_ratio=args.modifier_ratio,
    )

    edges = sorted(report["edges"], key=lambda e: -abs(e["effect_std"]))

    print(f"\n=== ground-truth edge effects: {args.dataset} ===")
    print("effect_std = std of E[child|do(parent=v)] over the do-grid, in sd(child) units")
    print("ate_direct = co-parents frozen at their mean (the edge's own claim)")
    print("modifier   = how much the parent changes a CO-PARENT's effect\n")
    print(f"{'edge':>12} {'effect_std':>11} {'ate_total':>10} {'ate_direct':>11} "
          f"{'modifier':>9}  label")
    for e in edges:
        print(f"{e['edge']:>12} {e['effect_std']:>11.5f} {e['ate_total']:>10.4f} "
              f"{e['ate_direct']:>11.4f} {e['modifier']:>9.4f}  {e['label']}")

    strong = [e for e in edges if e["label"] == "strong"]
    mods = [e for e in edges if e["label"] == "modifier_only"]
    weak = [e for e in edges if e["label"] == "weak"]
    print(f"\n{len(strong)} strong, {len(mods)} modifier_only, {len(weak)} weak "
          f"(of {len(edges)} true edges)")
    if mods:
        print("  modifier_only edges have ~ZERO average causal effect: an additive")
        print("  aggregator cannot represent them, so omitting them is correct.")
        for e in mods:
            print(f"    {e['edge']}: effect_std={e['effect_std']:.5f}, "
                  f"modifier={e['modifier']:.4f} via {e['modifier_per_coparent']}")

    anc = sorted(report["ancestor_pairs"], key=lambda a: -abs(a["effect_std"]))[:8]
    if anc:
        print("\nNon-edge ancestor pairs with the largest TOTAL effect "
              "(why a 'spurious' edge can look attractive):")
        for a in anc:
            print(f"    {a['pair']:>12} effect_std={a['effect_std']:.5f} "
                  f"ate_total={a['ate_total']:+.4f}")

    learned = None
    if args.learned_dag:
        learned = pd.read_csv(args.learned_dag, index_col=0)
    scores = weighted_structure_scores(edges, learned)
    if scores:
        report["weighted_scores"] = scores
        print("\n=== effect-weighted structure scores ===")
        for k, v in scores.items():
            print(f"    {k:>18}: {v}")

    report["edges"] = edges
    out_dir = data_dir_for(args.dataset)
    makedirs(out_dir, exist_ok=True)
    out = join(out_dir, "edge_effect_ground_truth.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
