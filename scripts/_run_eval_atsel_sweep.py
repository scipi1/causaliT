"""
Run eval_seed_sweep for all three 1_ATT_SELECT sweep experiments so that
the aggregated summary files (experiment_summary.csv, dag_summary.csv/json)
are available for the evaluation notebook.

Usage:
    python scripts/_run_eval_atsel_sweep.py
"""

import os
import sys
import traceback

from causaliT.evaluation.eval_funs.eval_seed_sweep import eval_seed_sweep

BASE = os.path.join("experiments", "4_CAUSAL_DISCOVERY", "1_ATT_SELECT", "results")

sweep_folders = sorted([
    os.path.join(BASE, d)
    for d in os.listdir(BASE)
    if os.path.isdir(os.path.join(BASE, d))
])

print(f"Found {len(sweep_folders)} sweep(s):\n  " + "\n  ".join(sweep_folders) + "\n")

ok, failed = [], []
for sweep in sweep_folders:
    print(f"\n{'='*60}")
    print(f"eval_seed_sweep: {sweep}")
    print('='*60)
    try:
        df_ate, df_exp = eval_seed_sweep(sweep)
        print(f"\n  Experiment summary ({len(df_exp)} row(s)):")
        print(df_exp.to_string(index=False))
        ok.append(sweep)
    except Exception:
        print(f"  ✗ FAILED:")
        traceback.print_exc()
        failed.append(sweep)

print(f"\n{'='*60}")
print(f"Done: {len(ok)} succeeded, {len(failed)} failed.")
if failed:
    print("\nFailed sweeps:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
