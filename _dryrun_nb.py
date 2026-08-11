"""Dry-run the code cells of the ATE evaluation notebook (headless)."""
import builtins
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import nbformat as nbf  # noqa: E402

NB = os.path.abspath("experiments/7_PUBLISH/ATE/results/evaluate_ate_results.ipynb")
os.chdir(os.path.dirname(NB))

builtins.display = lambda *a, **k: [print(str(x)[:600]) for x in a]

nb = nbf.read(NB, as_version=4)
env = {"__name__": "__main__"}
with warnings.catch_warnings():
    warnings.simplefilter("always")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        print(f"\n===== cell {i} =====")
        try:
            exec(compile(cell.source, f"<cell {i}>", "exec"), env)
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
print("\nALL CELLS OK")
