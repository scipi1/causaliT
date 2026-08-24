"""Produce the n=20 dataset used by the LARGER_DAGS experiments, and verify it.

Regenerates ``random_n20_k4_er_nonlinear_gaussian_s1`` from the stored
``dag_recipe.json`` (same seed -> same DAG and same samples) into the shared
``experiments/6_INVESTIGATIONS/LARGER_DAGS/datasets/`` root, then checks the
DAG adjacency and the sample arrays against the baseline run's copy.

Run:  python scripts/produce_n20_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scm_ds.random_scm import RandomSCMConfig
from causaliT.euler_sweep.euler_sweep.dag_provider import (
    generate_dag_dataset,
    _random_scm_field_names,
)

BASELINE_DIR = Path(
    "experiments/6_INVESTIGATIONS/LARGER_DAGS/baseline/random_n20_k4_er_nonlinear_gaussian_s1"
)
DATA_ROOT = Path("experiments/6_INVESTIGATIONS/LARGER_DAGS/datasets")


def main():
    recipe = json.load(open(BASELINE_DIR / "dag_recipe.json"))
    known = _random_scm_field_names()
    scm_fields = {k: v for k, v in recipe["random_scm_config"].items()
                  if k in known}
    scm_fields.pop("name", None)   # the folder name is re-derived identically
    cfg = RandomSCMConfig(**scm_fields)
    gen_kwargs = dict(recipe["generation"])

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    name = generate_dag_dataset(cfg, DATA_ROOT, gen_kwargs)
    print(f"produced: {name} -> {DATA_ROOT / name}")

    # ---- Verify against the baseline copy (same seed -> identical) -----------
    new_adj = pd.read_csv(DATA_ROOT / name / "dag_adj_mask.csv", index_col=0)
    base_adj = pd.read_csv(BASELINE_DIR / "dag_adj_mask.csv", index_col=0)
    new_adj = new_adj.reindex(index=base_adj.index, columns=base_adj.columns)
    assert (new_adj.values == base_adj.values).all(), "DAG adjacency mismatch!"
    print("DAG adjacency matches the baseline: True")

    new_ds = np.load(DATA_ROOT / name / "ds.npz")
    base_ds = np.load(BASELINE_DIR / "ds.npz")
    for key in base_ds.files:
        assert np.allclose(new_ds[key], base_ds[key]), f"sample array '{key}' mismatch!"
    print(f"sample arrays match the baseline: True ({base_ds.files})")
    print(f"dataset ready at {DATA_ROOT / name}")


if __name__ == "__main__":
    main()
