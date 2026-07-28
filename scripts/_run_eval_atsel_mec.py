"""
Re-run eval_attention_selector_scores for all 1_ATT_SELECT seed experiments
so that the updated dag_metrics.json includes MEC metrics.

Usage:
    python scripts/_run_eval_atsel_mec.py
"""

import os
import sys
import traceback

from causaliT.evaluation.eval_funs._OLD.eval_attention_selector import (
    eval_attention_selector_scores,
)

BASE = os.path.join("experiments", "4_CAUSAL_DISCOVERY", "1_ATT_SELECT", "results")

sweep_folders = sorted([
    d for d in os.listdir(BASE)
    if os.path.isdir(os.path.join(BASE, d))
])

all_experiments = []
for sw in sweep_folders:
    combos_dir = os.path.join(BASE, sw, "sweeper", "runs", "combinations")
    if not os.path.isdir(combos_dir):
        print(f"[WARN] combinations dir not found: {combos_dir}")
        continue
    for c in sorted(os.listdir(combos_dir)):
        exp_path = os.path.join(combos_dir, c)
        if os.path.isdir(exp_path):
            all_experiments.append(exp_path)

print(f"Found {len(all_experiments)} experiments to evaluate.\n")

ok, failed = [], []
for i, exp in enumerate(all_experiments, 1):
    print(f"\n[{i}/{len(all_experiments)}] {exp}")
    try:
        eval_attention_selector_scores(exp, show_plots=False)
        ok.append(exp)
    except Exception:
        print(f"  ✗ FAILED:")
        traceback.print_exc()
        failed.append(exp)

print(f"\n{'='*60}")
print(f"Done: {len(ok)} succeeded, {len(failed)} failed.")
if failed:
    print("\nFailed experiments:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
