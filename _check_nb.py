"""Compile-check every code cell of the generated notebook (no execution)."""
import nbformat

nb = nbformat.read(
    r"experiments/7_PUBLISH/ATE/results/evaluate_ate_results.ipynb", as_version=4
)
ok = True
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    try:
        compile(cell.source, f"<cell {i}>", "exec")
    except SyntaxError as e:
        ok = False
        print(f"cell {i}: SYNTAX ERROR: {e}")
print("all code cells compile" if ok else "FAILURES FOUND")
