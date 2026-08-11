"""Smoke validation of the interventional roll-out on a trained cheater run.

Non-destructive: calls ``run_mc_predictions`` directly (writes no eval files)
with a small MC sample and prints the recovered ATE for the indirect paths
S3->X2->X4 and S3->X2->X5 on ds_scm1, which the legacy one-shot harness forced
to exactly 0.  Also runs the deterministic variant (noise="none") so the A/B
gap of ATE_INTERVENTIONAL_ROLLOUT.md Sec. 8 can be read off.

Usage:  python scripts/_validate_causal_predict.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.evaluation.eval_funs.eval_interventions import (
    run_mc_predictions,
    get_scm_for_dataset,
    load_ate_ground_truth,
    load_normalization_stats,
)
from causaliT.evaluation.eval_funs.helpers.eval_utils import load_dataset_metadata

RUN = (
    "experiments/7_PUBLISH/ATE/results/cheater_9427393/groups/"
    "ds_scm1_continuous/sweeper/runs/combinations/"
    "cheater_ds_scm1_continuous_dag_0_model_1"
)
DATASET = "ds_scm1_continuous"
N_SAMPLES = 4000  # small: smoke test only


def _datadir(run: str) -> str:
    # Walk upward from the run folder to the directory that holds the
    # ``datasets/<DATASET>`` subfolder (the run's data root).
    for parent in Path(run).parents:
        cand = parent / "datasets"
        if (cand / DATASET).is_dir():
            return str(cand)
    raise FileNotFoundError(f"No datasets/{DATASET} found above {run}")


def main() -> None:
    datadir = _datadir(RUN)
    metadata = load_dataset_metadata(datadir, DATASET)
    scm_dataset = get_scm_for_dataset(DATASET, datadir_path=datadir)
    ate_gt = load_ate_ground_truth(datadir, DATASET)
    norm_stats = load_normalization_stats(datadir, DATASET)

    source_labels = metadata["variable_info"]["source_labels"]
    input_labels = metadata["variable_info"]["input_labels"]

    # Mirror eval_ate_mc: separate per-family maps when present, else shared.
    import json
    from os.path import join, exists
    svm_path = join(datadir, DATASET, "source_vars_map.json")
    ivm_path = join(datadir, DATASET, "input_vars_map.json")
    if exists(svm_path) and exists(ivm_path):
        source_vars_map = json.load(open(svm_path))
        input_vars_map = json.load(open(ivm_path))
    else:
        var_idx_map = metadata.get("variable_index_map", {})
        source_vars_map = {k: v for k, v in var_idx_map.items() if k in source_labels}
        input_vars_map = {k: v for k, v in var_idx_map.items() if k in input_labels}

    # Only the S3 intervention carries the multi-hop effects of interest.
    intervention_config = {"S3": [-0.5, 1.5]}
    true = ate_gt["monte_carlo"]["ate"]

    for noise in ("none", "residual"):
        df = run_mc_predictions(
            experiment_path=RUN,
            scm_dataset=scm_dataset,
            intervention_config=intervention_config,
            norm_stats=norm_stats,
            source_vars_map=source_vars_map,
            input_vars_map=input_vars_map,
            source_labels=source_labels,
            input_labels=input_labels,
            n_samples=N_SAMPLES,
            seed=42,
            checkpoint_type="last",
            propagation="rollout",
            noise=noise,
            datadir_path=datadir,
        )

        def mean_norm(label, var):
            m = df[(df.intervention == label) & (df.variable == var)]
            return float(m["pred_feat_0"].iloc[0])

        istats = norm_stats["input"]
        rng = istats["max"] - istats["min"]

        print(f"\n=== noise = {noise} ===")
        for label in ("S3=-0.5", "S3=1.5"):
            base = f"S3=0_baseline"
            for var in ("X2", "X4", "X5"):
                model_ate = (mean_norm(label, var) - mean_norm(base, var)) * rng
                delta = float(
                    df[(df.intervention == label) & (df.variable == var)][
                        "rollout_delta"
                    ].iloc[0]
                )
                print(
                    f"  {label} {var}: model_ate={model_ate:+.3f}  "
                    f"true_ate={true[label][var]:+.3f}  rollout_delta={delta:.2e}"
                )


if __name__ == "__main__":
    main()
