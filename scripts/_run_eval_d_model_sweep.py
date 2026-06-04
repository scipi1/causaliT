"""Run eval_seed_sweep on all sweep folders in BENCHMARKS/results."""
import os
import sys

sys.path.insert(0, r"c:\Users\ScipioneFrancesco\Documents\Projects\causaliT")

from causaliT.evaluation.eval_funs.eval_seed_sweep import eval_seed_sweep

results_dir = r"experiments/2_ARCH_STUDY/BENCHMARKS/results"
sweep_folders = sorted([
    d for d in os.listdir(results_dir)
    if os.path.isdir(os.path.join(results_dir, d))
])

print(f"Found {len(sweep_folders)} sweep folders")
print()

errors = []
for folder in sweep_folders:
    sweep_path = os.path.join(results_dir, folder)
    print("=" * 60)
    print(f"Processing: {folder}")
    print("=" * 60)
    try:
        df_ate, df_exp = eval_seed_sweep(sweep_path)
        print(f"SUCCESS: {len(df_ate)} ATE rows, experiment summary written")
    except Exception as e:
        print(f"ERROR: {e}")
        errors.append((folder, str(e)))
    print()

if errors:
    print(f"ERRORS ({len(errors)}):")
    for f, e in errors:
        print(f"  {f}: {e}")
else:
    print("All sweeps processed successfully!")
