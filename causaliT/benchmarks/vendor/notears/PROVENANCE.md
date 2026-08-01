# NOTEARS - vendored source

Upstream: https://github.com/xunzheng/notears (branch `master`)
Papers:
- Zheng, Aragam, Ravikumar, Xing. *DAGs with NO TEARS: Continuous Optimization
  for Structure Learning.* NeurIPS 2018. (linear)
- Zheng, Dan, Aragam, Ravikumar, Xing. *Learning Sparse Nonparametric DAGs.*
  AISTATS 2020. (nonlinear / MLP)

License: Apache-2.0, see `LICENSE` (copied verbatim from the upstream repo).

## Why vendored instead of pip-installed

The upstream project is not published on PyPI, and its `utils.py` pulls in
`igraph`, a heavy dependency needed only for the paper's own data simulation.
Vendoring the four solver files keeps the benchmark faithful and the
dependency footprint small.

## Files copied verbatim

| file | purpose |
|---|---|
| `linear.py` | `notears_linear` - linear NOTEARS (augmented Lagrangian, L-BFGS-B) |
| `nonlinear.py` | `NotearsMLP` + `notears_nonlinear` - per-node MLP variant |
| `locally_connected.py` | `LocallyConnected` layer used by `NotearsMLP` |
| `lbfgsb_scipy.py` | `LBFGSBScipy` optimizer wrapper (bound-constrained L-BFGS-B) |
| `trace_expm.py` | `trace_expm` autograd function for the acyclicity term |
| `LICENSE` | Apache-2.0 license text |

## Local modifications

1. `nonlinear.py`: absolute intra-package imports rewritten to relative ones
   (`from notears.locally_connected import ...` -> `from .locally_connected import ...`).
   Nothing else was touched; in particular the solver, the objective and all
   default arguments are unchanged.
2. `utils.py` was **not** vendored (paper data simulator, requires `igraph`).
   Its only reference is inside `nonlinear.main()`, the upstream demo, which is
   never called by causaliT.

## Conventions worth remembering

- Both entry points return `W` with `W[i, j] != 0` meaning **i -> j**
  (`X = X W + noise`), i.e. rows are parents.  causaliT's canonical adjacency is
  the transpose (rows = child).  The conversion lives in
  `causaliT/benchmarks/postprocess.py`.
- Both apply `w_threshold` (default 0.3) internally before returning, so the
  benchmark wrappers pass `w_threshold=0.0` and threshold afterwards; this way
  the raw `|W|` is preserved in `benchmark_run.json` and any threshold can be
  re-scored offline without refitting.
- `notears_nonlinear` requires double precision
  (`torch.set_default_dtype(torch.double)`), which the wrapper sets and restores.
