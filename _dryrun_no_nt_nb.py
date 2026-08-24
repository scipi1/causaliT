# Temporary dry-run of analyze_no_NT_hsic_signal.ipynb code cells (Agg backend).
# DELETE after use (see .clinerules/long-test-shell-commands.md).
import json
import matplotlib
matplotlib.use("Agg")

nb = json.load(open(
    r"experiments/6_INVESTIGATIONS/LARGER_DAGS/analyze_no_NT_hsic_signal.ipynb",
    encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
print(f"{len(cells)} code cells")

ns = {"__name__": "__main__"}
for i, src in enumerate(cells):
    print(f"\n===== cell {i} =====", flush=True)
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print("\nALL CELLS OK")
