"""
Vendored NOTEARS reference implementation (Zheng et al.).

See ``PROVENANCE.md`` for the exact upstream commit and the list of edits.
The code is kept as close to the original as possible so the benchmark is a
faithful reproduction of the paper's method; only the intra-package imports
were rewritten (``from notears.x import`` -> ``from .x import``) and the
data-simulation ``utils.py`` was dropped (causaliT generates its own data via
``scm_ds``).

Public entry points used by ``causaliT.benchmarks.methods``:
    ``linear.notears_linear``           - linear NOTEARS
    ``nonlinear.NotearsMLP``            - per-node MLP (dims=[d, H, 1])
    ``nonlinear.notears_nonlinear``     - augmented-Lagrangian solver for the MLP
"""
