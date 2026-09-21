"""Add variance tolerance to varsortability calls (ties after z-scoring)."""
from pathlib import Path

# --- scm_ds/scm.py ---
p = Path("scm_ds/scm.py")
t = p.read_text(encoding="utf-8")

old = '"varsortability": var_sortability(X_stored, W_full),'
new = '"varsortability": var_sortability(\n'
new += '                    X_stored, W_full, tol=_var_tol(X_stored)),'
assert t.count(old) == 1
t = t.replace(old, new)

old = '"varsortability_raw": var_sortability(X_raw, W_full),'
new = '"varsortability_raw": var_sortability(X_raw, W_full, tol=_var_tol(X_raw)),'
assert t.count(old) == 1
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")

# --- scripts/compute_sortability.py ---
p = Path("scripts/compute_sortability.py")
t = p.read_text(encoding="utf-8")
old = "    out = {"
assert t.count(old) == 1
t = t.replace(old, "    var_tol = 1e-6 * float(np.var(X, axis=0, ddof=1).mean())\n    out = {")
old = '"varsortability": var_sortability(X, W),'
assert t.count(old) == 1
t = t.replace(old, '"varsortability": var_sortability(X, W, tol=var_tol),')
p.write_text(t, encoding="utf-8")

import ast
for f in ("scm_ds/scm.py", "scripts/compute_sortability.py"):
    ast.parse(Path(f).read_text(encoding="utf-8"))
print("tol patched, syntax OK")
