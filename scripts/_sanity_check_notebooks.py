"""Quick sanity check: verify all 12 sweep data files are accessible from notebook working dir."""
import os, json
import numpy as np
import pandas as pd
from os.path import join, exists

os.chdir("experiments/2_ARCH_STUDY/BENCHMARKS")

RESULTS_DIR  = "results"
EVAL_SUBPATH = "eval/eval_seed_sweep/files"

MODELS = [
    {"model": "Baseline",        "dataset": "scm1", "sweep": "SWEEP_seed_bs_scm1c_586027"},
    {"model": "Baseline",        "dataset": "scm2", "sweep": "SWEEP_seed_bs_scm2c_586024"},
    {"model": "Baseline",        "dataset": "scm3", "sweep": "SWEEP_seed_bs_scm3c_586021"},
    {"model": "Causal Baseline", "dataset": "scm1", "sweep": "SWEEP_seed_bs_hard_scm1c_586013"},
    {"model": "Causal Baseline", "dataset": "scm2", "sweep": "SWEEP_seed_bs_hard_scm2c_586014"},
    {"model": "Causal Baseline", "dataset": "scm3", "sweep": "SWEEP_seed_bs_hard_scm3c_586018"},
    {"model": "NaRes+CC",        "dataset": "scm1", "sweep": "SWEEP_nares_Toe_CC_scm1c_585726"},
    {"model": "NaRes+CC",        "dataset": "scm2", "sweep": "SWEEP_nares_Toe_CC_scm2c_585724"},
    {"model": "NaRes+CC",        "dataset": "scm3", "sweep": "SWEEP_nares_Toe_CC_scm3c_585720"},
    {"model": "Single+CC",       "dataset": "scm1", "sweep": "SWEEP_seed_single_Toe_CC_scm1c_1758745"},
    {"model": "Single+CC",       "dataset": "scm2", "sweep": "SWEEP_seed_single_Toe_CC_scm2c_1758754"},
    {"model": "Single+CC",       "dataset": "scm3", "sweep": "SWEEP_seed_single_Toe_CC_scm3c_1758760"},
]

ate_dfs, exp_dfs = [], []
errors = []

for m in MODELS:
    files_dir = join(RESULTS_DIR, m["sweep"], EVAL_SUBPATH)
    for fname in ["ate_summary.csv", "experiment_summary.csv", "aggregate_dag.json"]:
        p = join(files_dir, fname)
        if not exists(p):
            errors.append(f"MISSING: {p}")

    ate_path = join(files_dir, "ate_summary.csv")
    exp_path = join(files_dir, "experiment_summary.csv")
    if exists(ate_path):
        df = pd.read_csv(ate_path)
        df["model"] = m["model"]
        df["dataset"] = m["dataset"]
        ate_dfs.append(df)
    if exists(exp_path):
        df = pd.read_csv(exp_path)
        df["model"] = m["model"]
        df["dataset"] = m["dataset"]
        exp_dfs.append(df)

if errors:
    print("ERRORS:")
    for e in errors:
        print(" ", e)
else:
    print("All files found!")

df_ate = pd.concat(ate_dfs, ignore_index=True)
df_exp = pd.concat(exp_dfs, ignore_index=True)

print(f"ATE rows loaded: {len(df_ate)}")
print(f"Exp rows loaded: {len(df_exp)}")
print(f"Models in ATE: {df_ate['model'].unique().tolist()}")
print(f"Datasets: {df_ate['dataset'].unique().tolist()}")
print(f"Interventions: {sorted(df_ate['intervention'].unique())}")
print(f"Variables: {sorted(df_ate['variable'].unique())}")
print("\nSample ATE rows (Baseline, scm1):")
sample = df_ate[(df_ate["model"] == "Baseline") & (df_ate["dataset"] == "scm1")]
print(sample[["intervention", "variable", "abs_error_mean", "abs_error_std"]].to_string(index=False))
