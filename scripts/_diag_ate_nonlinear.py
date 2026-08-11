"""Diagnose why svfa trails vanilla on the nonlinear ATE datasets.

Read-only analysis over the regenerated artifacts (roll-out ATE eval). Splits
the gap into four candidate drivers:
  1. mechanism fit quality      (test_x_mae / test_x_r2_macro, kfold_summary.json)
  2. roll-out convergence       (rollout_delta, predictions_mc.csv) - cyclic graphs
  3. learned X->X structure     (shd_self / soft_hamming_self, dag_metrics.json)
  4. gap localisation           (per-cell ATE error, ID vs OOD x direct/indirect)
  5. residual-pool scale        (pool std vs vanilla; inflated => model error, not noise)

Usage:  python scripts/_diag_ate_nonlinear.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/7_PUBLISH/ATE/results")
ARMS = {"vanilla_9648449": "vanilla", "svfa_9425659": "svfa", "cheater_9427393": "cheater"}


def load_all():
    ate, struct, train, roll = [], [], [], []
    for arm_dir, arm in ARMS.items():
        base = RESULTS / arm_dir / "groups"
        for run in sorted(base.glob("*/sweeper/runs/combinations/*")):
            ds = next((p for p in run.parts if p.startswith("ds_scm")), None)
            if ds is None:
                continue
            f_ate = run / "eval/eval_ate_mc/files/ate_metrics_mc.csv"
            f_pred = run / "eval/eval_ate_mc/files/predictions_mc.csv"
            f_dag = run / "eval/eval_attention_scores/files/dag_metrics.json"
            f_kf = run / "kfold_summary.json"
            if f_ate.is_file():
                d = pd.read_csv(f_ate)
                d["arm"], d["dataset"] = arm, ds
                ate.append(d)
            if f_pred.is_file():
                p = pd.read_csv(f_pred)
                roll.append({"arm": arm, "dataset": ds,
                             "rollout_delta_max": p["rollout_delta"].max(),
                             "frac_nonconverged": (p["rollout_delta"] > 1e-6).mean()})
            if f_dag.is_file():
                j = json.load(open(f_dag))
                struct.append({"arm": arm, "dataset": ds,
                               "shd_self": (j.get("standard_shd_self") or {}).get("mean"),
                               "shd_cross": (j.get("standard_shd_cross") or {}).get("mean"),
                               "soft_h_self": (j.get("soft_hamming_self") or {}).get("mean")})
            if f_kf.is_file():
                m = json.load(open(f_kf)).get("fold_results", {}).get("0", {}).get("metrics", {})
                train.append({"arm": arm, "dataset": ds,
                              "test_x_mae": m.get("test_x_mae"),
                              "test_x_r2_macro": m.get("test_x_r2_macro")})
    return (pd.concat(ate), pd.DataFrame(roll), pd.DataFrame(struct), pd.DataFrame(train))


def main():
    df_ate, df_roll, df_struct, df_train = load_all()

    pd.set_option("display.width", 160)
    print("=" * 78)
    print("1. MECHANISM FIT (mean over seeds) - the rollout amplifies fit error")
    print("=" * 78)
    print(df_train.groupby(["arm", "dataset"])[["test_x_mae", "test_x_r2_macro"]]
          .mean().round(4).to_string())

    print("\n" + "=" * 78)
    print("2. ROLL-OUT CONVERGENCE (rollout_delta) - >0 flags cyclic learned graph")
    print("=" * 78)
    print(df_roll.groupby(["arm", "dataset"])
          [["rollout_delta_max", "frac_nonconverged"]].mean().round(4).to_string())

    print("\n" + "=" * 78)
    print("3. LEARNED X->X STRUCTURE (self block) - propagation needs these edges")
    print("=" * 78)
    print(df_struct.groupby(["arm", "dataset"])[["shd_self", "soft_h_self"]]
          .mean().round(3).to_string())

    print("\n" + "=" * 78)
    print("4. GAP LOCALISATION - mean abs_error by arm x dataset x path x regime")
    print("=" * 78)
    # Rebuild path/dist groups using the GT masks (same logic as the notebook).
    cross_csv = next(RESULTS.glob("*/groups/*/datasets/*/dec1_cross_att_mask.csv"))
    cross_gt = pd.read_csv(cross_csv, index_col=0)
    self_gt = pd.read_csv(cross_csv.with_name("dec1_self_att_mask.csv"), index_col=0)
    direct = {s: [x for x in cross_gt.index if cross_gt.loc[x, s]] for s in cross_gt.columns}
    x_children = {x: [t for t in self_gt.index if self_gt.loc[t, x]] for x in self_gt.columns}

    def descendants(s):
        seen, stack = set(), list(direct.get(s, []))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(x_children.get(x, []))
        return seen

    desc = {s: descendants(s) for s in cross_gt.columns}

    def path_type(row):
        src = row["intervention"].split("=")[0]
        if row["variable"] not in desc.get(src, []):
            return "zero"
        return "direct" if row["variable"] in direct.get(src, []) else "indirect"

    df_ate["path_type"] = df_ate.apply(path_type, axis=1)
    df_ate["dist"] = np.where(
        df_ate["intervention"].str.extract(r"=(-?[\d.]+)")[0].astype(float).abs() > 1.0,
        "OOD", "ID")
    sub = df_ate[df_ate.path_type.isin(["direct", "indirect"])]
    piv = (sub.groupby(["dataset", "arm", "path_type", "dist"])["abs_error"]
           .mean().round(3).reset_index())
    for ds in sorted(piv.dataset.unique()):
        print(f"--- {ds} ---")
        print(piv[piv.dataset == ds]
              .pivot_table(index=["path_type", "dist"], columns="arm",
                           values="abs_error")
              .round(3).to_string())

    print("\n" + "=" * 78)
    print("5. RESIDUAL-POOL SCALE (one run per arm x dataset)")
    print("    inflated vs vanilla => pool carries model error, not just noise")
    print("=" * 78)
    # Cheap proxy that avoids loading checkpoints: teacher-forced residual std
    # is not stored, so estimate from the pool rebuild only for one run each.
    from causaliT.evaluation.eval_funs.eval_interventions import (
        _build_residual_pool, _find_train_npz)
    from causaliT.evaluation.eval_funs.helpers.datadir import resolve_datadir
    from causaliT.evaluation.predict import create_predictor
    from causaliT.training.experiment_control import update_config
    from causaliT.evaluation.eval_funs.helpers.eval_lib import (
        find_config_file, find_best_or_last_checkpoint)
    from omegaconf import OmegaConf
    import torch

    for arm_dir, arm in ARMS.items():
        for ds_dir in sorted((RESULTS / arm_dir / "groups").iterdir()):
            ds = ds_dir.name
            runs = sorted(ds_dir.glob("sweeper/runs/combinations/*"))
            runs = [r for r in runs if (r / "k_0/checkpoints").is_dir()]
            if not runs:
                continue
            run = runs[0]
            datadir = ds_dir / "datasets"
            try:
                cfg = update_config(OmegaConf.load(find_config_file(str(run))))
                ck = find_best_or_last_checkpoint(str(run / "k_0/checkpoints"), "last")
                pred = create_predictor(cfg, ck, str(datadir))
                pred.model.eval()
                npz = _find_train_npz(str(datadir), ds)
                import json as _json
                svm = _json.load(open(datadir / ds / "source_vars_map.json"))
                ivm = _json.load(open(datadir / ds / "input_vars_map.json"))
                meta = _json.load(open(datadir / ds / "dataset_metadata.json"))
                pool = _build_residual_pool(
                    pred.model, npz, svm, ivm,
                    meta["variable_info"]["source_labels"],
                    meta["variable_info"]["input_labels"],
                    torch.device("cpu"))
                print(f"  {arm:8s} {ds}: pool std per node = "
                      f"{np.round(pool.std(0).numpy(), 4)}")
            except Exception as exc:
                print(f"  {arm:8s} {ds}: FAILED ({exc})")


if __name__ == "__main__":
    main()
