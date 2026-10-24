# Provenance: CaPS

- Paper: Xu et al., "Ordering-Based Causal Discovery for Linear and Nonlinear
  Relations", NeurIPS 2024.
- Repository: https://github.com/E2real/CaPS
- Commit: `325a4ffffab0f9cd14db66dc404b422bc976d916` (2024-09-29)
- Licence: MIT (copied to `LICENSE`).

## What is vendored

`caps.py` contains, unchanged in substance:

| Function | Original file |
|---|---|
| `Stein_hess`, `compute_top_order` | `train_order.py` (nested in `train_test`) |
| `full_DAG`, `get_parents_score`, `pre_pruning_with_parents_score`, `add_edge_with_parents_score`, `is_acyclic` | `utils.py` |

## Changes

1. Nested functions lifted to module level; `device` became a parameter.
2. A debug `print(parents_score)` removed.
3. **Pruning backend.** The original prunes with R (`cdt.launch_R_script` on
   `pruning_R_files/cam_pruning.R`, `selGam`, `cutOffPVal = 0.001`,
   `numBasisFcts = 10`). causaliT is Python-only, so
   `causaliT/benchmarks/methods/caps.py` calls `dodiscover`'s `CAMPruning`
   with `alpha = 0.001` and 10 splines. Same test (GAM p-values per parent),
   different implementation (`pygam` vs `mgcv`): pruned graphs can differ
   slightly from the paper. Order search and parent scores are unaffected.

Not vendored: data simulation, the Sachs data, `cdt`/`castle` metrics.
