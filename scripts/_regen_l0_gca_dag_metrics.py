"""
Regenerate stage_2 DAG metrics + learned_dag_edges for the L0_reg_GCA_scm3c
sweep using the (now default) LAST checkpoint.

Root cause being fixed: eval_attention_selector_scores previously loaded the
``best_causal`` checkpoint (min val_hsic_reg), which for the two-stage L0
protocol lands at the START of the structural stage — before the L0 penalty has
pruned the structure gate. The retrieved DAG therefore looked identical across
lambda_l0. infer_checkpoint_type now defaults to "last", so re-running the eval
on the existing checkpoints regenerates the figures from the pruned end-state.

Usage:
    python scripts/_regen_l0_gca_dag_metrics.py
"""

import os
import sys
import traceback

from causaliT.evaluation.eval_funs.eval_attention_selector import (
    eval_attention_selector_scores,
)

SWEEP_ROOT = os.path.join(
    "experiments", "2_ARCH_STUDY", "L0_REG", "L0_reg_GCA_scm3c_6181523",
    "sweeper", "runs", "combinations",
)
STAGE_2 = os.path.join("anm_stages", "01_stage_2")


def main():
    combos = sorted(
        d for d in os.listdir(SWEEP_ROOT)
        if os.path.isdir(os.path.join(SWEEP_ROOT, d))
    )
    print(f"Found {len(combos)} combos under {SWEEP_ROOT}")

    ok, skipped, failed = [], [], []
    for i, combo in enumerate(combos, 1):
        stage_dir = os.path.join(SWEEP_ROOT, combo, STAGE_2)
        if not os.path.isdir(stage_dir):
            skipped.append(combo)
            continue
        print(f"\n[{i}/{len(combos)}] eval_attention_selector_scores: {combo}")
        try:
            eval_attention_selector_scores(stage_dir, show_plots=False)
            ok.append(combo)
        except Exception:
            print(f"  FAILED: {combo}")
            traceback.print_exc()
            failed.append(combo)

    print(f"\n{'='*60}")
    print(f"Done: {len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed.")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
