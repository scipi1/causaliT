"""Re-run the ATE evaluation (roll-out estimator) for the three ATE arms.

Evaluation ONLY - no retraining.  Iterates over every seed run folder under
``experiments/7_PUBLISH/ATE/results/<arm>_<id>/groups/<dataset>/.../combinations/<run>``
and calls ``eval_ate_mc`` with the interventional roll-out (variant B by
default), overwriting ``eval/eval_ate_mc/files/ate_metrics_mc.{csv,json}``.

The legacy one-shot harness forced every indirect effect to exactly 0; this
regenerates those numbers.  See docs/documentation/ATE_INTERVENTIONAL_ROLLOUT.md.

Usage:
    python scripts/_rerun_ate_eval_rollout.py                # all arms, all runs
    python scripts/_rerun_ate_eval_rollout.py --arm cheater  # one arm
    python scripts/_rerun_ate_eval_rollout.py --noise none   # variant A
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.evaluation.eval_funs.eval_interventions import eval_ate_mc

RESULTS = project_root / "experiments" / "7_PUBLISH" / "ATE" / "results"


def find_runs(arm_filter=None):
    """Yield every run folder that contains a config*.yaml (i.e. a trained run)."""
    for arm_dir in sorted(RESULTS.iterdir()):
        if not arm_dir.is_dir():
            continue
        if arm_filter and not arm_dir.name.startswith(arm_filter):
            continue
        combos = arm_dir / "groups"
        if not combos.is_dir():
            continue
        for cfg in combos.rglob("config*.yaml"):
            run_dir = cfg.parent
            # Only seed runs (they carry k_0/), not sweep/aggregate folders.
            if (run_dir / "k_0").is_dir():
                yield run_dir


def run_datadir(run_dir: Path) -> str:
    """The run-local ``.../groups/<dataset>/datasets`` folder.

    The config's ``data_root`` is a stale cluster path on relocated runs, so we
    resolve the datasets folder by walking up from the run folder.  This is what
    makes a LOCAL re-evaluation possible.
    """
    for parent in run_dir.parents:
        cand = parent / "datasets"
        if cand.is_dir() and any(cand.iterdir()):
            return str(cand)
    raise FileNotFoundError(f"No datasets/ folder found above {run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="e.g. svfa | vanilla | cheater")
    ap.add_argument("--noise", default="residual", choices=["residual", "none"])
    ap.add_argument("--n_samples", type=int, default=50000)
    args = ap.parse_args()

    runs = list(find_runs(args.arm))
    print(f"Found {len(runs)} run(s) to re-evaluate (noise={args.noise}).")
    n_ok = 0
    for run_dir in runs:
        print(f"\n=== {run_dir.name} ===")
        try:
            eval_ate_mc(
                str(run_dir),
                n_samples=args.n_samples,
                propagation="rollout",
                noise=args.noise,
                datadir_path=run_datadir(run_dir),
            )
            n_ok += 1
        except Exception as exc:  # keep going; report at the end
            print(f"  [FAIL] {run_dir.name}: {exc}")
    print(f"\nDone: {n_ok}/{len(runs)} runs re-evaluated.")


if __name__ == "__main__":
    main()
