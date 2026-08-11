"""Recover the cheater arm: regenerate pruned datasets and re-run evaluations.

Step 1 (regen):  regenerate the three pruned datasets from scm_recipe.json and
                 verify determinism by diffing the light artifacts that survived
                 pruning (normalization.json, ate_ground_truth.json) against the
                 freshly regenerated ones.
Step 2 (eval):   re-run eval_attention_scores + eval_interventions on the 15
                 cheater runs with a temporarily relocated data_root.

Usage: python _recover_cheater.py regen|eval
"""
import filecmp
import json
import re
import shutil
import sys
import time
from pathlib import Path

from causaliT.euler_sweep.euler_sweep.data_source import regenerate_from_scm_recipe

CHEATER = Path("experiments/7_PUBLISH/ATE/results/cheater_9427393")
DATASETS = ["ds_scm1_continuous", "ds_scm2_continuous", "ds_scm3_continuous"]
BAK = Path("_cheater_recovery_backup")
LIGHT = ["normalization.json", "ate_ground_truth.json"]
RUNS_GLOB = "groups/*/sweeper/runs/combinations/cheater_*"


def regen():
    BAK.mkdir(exist_ok=True)
    for ds in DATASETS:
        ds_dir = CHEATER / "groups" / ds / "datasets" / ds
        print(f"\n=== {ds} ===")
        for f in LIGHT:
            src = ds_dir / f
            if src.exists() and not (BAK / f"{ds}_{f}").exists():
                shutil.copy2(src, BAK / f"{ds}_{f}")
        regenerate_from_scm_recipe(str(ds_dir))
        for f in LIGHT:
            new = ds_dir / f
            old = BAK / f"{ds}_{f}"
            if not old.exists():
                print(f"  {f}: no backup to compare")
                continue
            same = filecmp.cmp(new, old, shallow=False)
            print(f"  {f}: {'IDENTICAL' if same else 'DIFFERS!'}")
            if not same:
                a = json.loads(new.read_text())
                b = json.loads(old.read_text())
                ka, kb = set(a), set(b)
                print(f"    keys new-old: {sorted(ka - kb)}, old-new: {sorted(kb - ka)}")
        arrays = sorted(p.name for p in ds_dir.glob("*.npz"))
        print(f"  arrays: {arrays}")


def _find_runs():
    return sorted(p for p in CHEATER.glob(RUNS_GLOB) if p.is_dir())


def _patch_config(run_dir: Path) -> str:
    """Point data.data_root at the local datasets dir; return original text."""
    cfg = run_dir / "config.yaml"
    text = cfg.read_text(encoding="utf-8")
    ds = run_dir.parents[3].name  # groups/<ds>/sweeper/runs/combinations/<run>
    local_root = (CHEATER / "groups" / ds / "datasets").resolve().as_posix()
    new, n = re.subn(r"(?m)^(\s*data_root:\s*).*$", rf"\g<1>{local_root}", text)
    assert n == 1, f"expected exactly one data_root in {cfg}, got {n}"
    cfg.write_text(new, encoding="utf-8")
    return text


def _restore_config(run_dir: Path, original: str):
    (run_dir / "config.yaml").write_text(original, encoding="utf-8")


def eval_runs(only: str = None):
    from causaliT.evaluation.eval_funs.eval_attention import eval_attention_scores
    from causaliT.evaluation.eval_funs.eval_interventions import eval_interventions

    runs = _find_runs()
    if only:
        runs = [r for r in runs if r.name == only]
    assert runs, "no runs matched"
    print(f"evaluating {len(runs)} cheater run(s)")

    originals = {}
    try:
        for run in runs:
            originals[run] = _patch_config(run)
        for i, run in enumerate(runs, 1):
            t0 = time.time()
            print(f"\n[{i}/{len(runs)}] {run.name}")
            try:
                eval_attention_scores(str(run), show_plots=False)
            except Exception as exc:
                print(f"  [FAIL] eval_attention_scores: {exc}")
            try:
                eval_interventions(str(run), show_plots=False)
            except Exception as exc:
                print(f"  [FAIL] eval_interventions: {exc}")
            ate = run / "eval/eval_ate_mc/files/ate_metrics_mc.csv"
            dag = run / "eval/eval_attention_scores/files/dag_metrics.json"
            print(f"  ate_csv={ate.exists()} dag_json={dag.exists()} "
                  f"({time.time() - t0:.0f}s)")
    finally:
        for run, text in originals.items():
            _restore_config(run, text)
        print("\nconfig.yaml files restored to cluster data_root")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "regen":
        regen()
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_runs(only=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("usage: python _recover_cheater.py regen|eval [run_name]")
