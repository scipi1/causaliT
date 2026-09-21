"""Backfill ``sortability.json`` for an existing dataset folder.

Computes var-sortability (Reisach et al. 2021) and R2-sortability (Reisach
et al. 2023) on the STORED design matrix (ds.npz / ds_train.npz), using the
full DAG adjacency (dag_adj_mask.csv).  Raw (pre-normalization) values are
not recoverable for existing folders, so only the stored-values metrics are
written; regenerate the dataset to get the ``*_raw`` entries.

Usage:
    python scripts/compute_sortability.py data/random_n20_k4_er_nonlinear_gaussian_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scm_ds.sortability import r2_sortability, var_sortability


def compute_sortability(dataset_dir: Path) -> dict:
    npz_name = "ds.npz" if (dataset_dir / "ds.npz").exists() else "ds_train.npz"
    data = np.load(dataset_dir / npz_name)

    adj = pd.read_csv(dataset_dir / "dag_adj_mask.csv", index_col=0)
    labels = list(adj.columns)
    # dag_adj_mask.csv is stored in the (possibly permuted) SCM label order,
    # which need NOT match the array column order: map each label to its
    # array column through the 1-based vars maps (padding_idx 0).
    sv_map = json.load(open(dataset_dir / "source_vars_map.json"))
    iv_map = json.load(open(dataset_dir / "input_vars_map.json"))
    cols = []
    for lab in labels:
        if lab in sv_map:
            cols.append(np.asarray(data["s"])[:, sv_map[lab] - 1, 0])
        else:
            cols.append(np.asarray(data["x"])[:, iv_map[lab] - 1, 0])
    X = np.column_stack(cols)  # (n, d), aligned with the adjacency order

    assert adj.shape[0] == adj.shape[1] == X.shape[1], (
        f"adjacency {adj.shape} vs values {X.shape}"
    )
    A = adj.values != 0  # [child, parent]: A[i, j] == 1 means j -> i
    # Sanity: rows of root nodes must be all-zero in [child, parent] layout.
    root_rows = [i for i, lab in enumerate(labels) if lab in sv_map]
    assert (A[root_rows] == 0).all(), (
        "source rows are not all zero -- dag_adj_mask is not [child, parent]"
    )
    W = A.T  # CausalDisco convention: W[i, j] = edge i -> j

    var_tol = 1e-6 * float(np.var(X, axis=0, ddof=1).mean())
    out = {
        "definition": (
            "path-weighted order alignment over directed paths "
            "(CausalDisco order_alignment_paths; ties count 1/2)"
        ),
        "references": [
            "Reisach et al. 2021 (arXiv:2102.13647) var-sortability",
            "Reisach et al. 2023 (arXiv:2303.18211) R2-sortability",
        ],
        "computed_offline": True,
        "values": npz_name,
        "n_samples": int(X.shape[0]),
        "varsortability": var_sortability(X, W, tol=var_tol),
        "r2_sortability": r2_sortability(X, W),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for d in args.dataset_dirs:
        out = compute_sortability(d)
        with open(d / "sortability.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(
            f"{d.name}: varsortability={out['varsortability']:.4f} "
            f"r2_sortability={out['r2_sortability']:.4f} "
            f"(n={out['n_samples']}, {out['values']})"
        )


if __name__ == "__main__":
    main()
