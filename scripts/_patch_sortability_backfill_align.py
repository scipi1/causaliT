"""Fix compute_sortability.py: align array columns to adjacency labels.

dag_adj_mask.csv is stored in the (permuted) SCM label order, while the
array columns follow the sequential X1..X18 order.  Map via the vars maps.
"""
from pathlib import Path

P = Path("scripts/compute_sortability.py")
t = P.read_text(encoding="utf-8")

old = """    npz_name = "ds.npz" if (dataset_dir / "ds.npz").exists() else "ds_train.npz"
    data = np.load(dataset_dir / npz_name)
    parts = []
    if "s" in data.files:
        parts.append(np.asarray(data["s"])[:, :, 0])
    parts.append(np.asarray(data["x"])[:, :, 0])
    X = np.concatenate(parts, axis=1)  # (n, d): [sources; inputs]

    adj = pd.read_csv(dataset_dir / "dag_adj_mask.csv", index_col=0)
    assert adj.shape[0] == adj.shape[1] == X.shape[1], (
        f"adjacency {adj.shape} vs values {X.shape}"
    )
    A = adj.values != 0  # [child, parent]: A[i, j] == 1 means j -> i
    # Sanity: rows of root nodes must be all-zero in [child, parent] layout.
    assert (A[: data["s"].shape[1]] == 0).all(), (
        "source rows are not all zero -- dag_adj_mask is not [child, parent]"
    )
    W = A.T  # CausalDisco convention: W[i, j] = edge i -> j
"""

new = """    npz_name = "ds.npz" if (dataset_dir / "ds.npz").exists() else "ds_train.npz"
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
"""
assert t.count(old) == 1
P.write_text(t.replace(old, new), encoding="utf-8")

import ast
ast.parse(P.read_text(encoding="utf-8"))
print("backfill alignment fixed, syntax OK")
