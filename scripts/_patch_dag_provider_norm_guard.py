"""Patch dag_provider.ensure_dag_dataset: warn on normalize_method mismatch.

The dataset folder name does NOT encode ``normalize_method``, so reusing a
materialized dataset generated with a different normalization would silently
train on the wrong scaling.  Emit a loud warning in that case.
"""
from pathlib import Path

P = Path("causaliT/euler_sweep/euler_sweep/dag_provider.py")
t = P.read_text(encoding="utf-8")

old = """    if not force and is_materialized(dataset_dir) and has_arrays(dataset_dir):
        if verbose:
            print(f"  [dag_provider] reusing {name}")
        return name
"""
new = """    if not force and is_materialized(dataset_dir) and has_arrays(dataset_dir):
        # The folder name does not encode ``normalize_method``: a materialized
        # dataset may have been generated with a DIFFERENT normalization than
        # the one requested now.  Warn loudly instead of silently training on
        # the wrong scaling.
        requested = (gen_kwargs or {}).get("normalize_method", None)
        recipe = read_recipe(dataset_dir)
        stored = (
            (recipe or {}).get("generation", {}).get("normalize_method", None)
        )
        if requested is not None and stored is not None and requested != stored:
            print(
                f"  [dag_provider] WARNING: {name} was generated with "
                f"normalize_method={stored!r} but {requested!r} was requested. "
                "Reusing the STORED arrays; regenerate (force=True or a "
                "distinct cfg.name) if the new normalization is intended."
            )
        if verbose:
            print(f"  [dag_provider] reusing {name}")
        return name
"""
assert t.count(old) == 1, f"anchor found {t.count(old)}x"
P.write_text(t.replace(old, new), encoding="utf-8")
print("dag_provider patched")
