# Ground-truth causal effect per edge

## Why this exists

SHD and edge-F1 count every edge of the true DAG equally. That is only fair if the
edges carry comparable causal signal. On `scm3_continuous` they do not: the edges differ
by a factor ~26 in average causal effect, and one of them has *no* average effect at all.
So "the model does not recover `S5 -> X4`" was being scored as a failure when it is the
correct answer for the model class we use.

This document defines the estimands, reports the measured values, and states the claim
that the numbers do (and do not) license.

Code: `SCMDataset.compute_edge_effect_ground_truth` (`scm_ds/scm.py`),
CLI `scripts/edge_effect_ground_truth.py`, tests `tests/test_edge_effect_ground_truth.py`.

## Estimands

For an edge `j -> i`, with `lo_j`, `hi_j` the 5%/95% observational quantiles of `j` and a
grid of `n_grid` do-values between them (all by Monte Carlo `do()`):

| quantity | definition | what it answers |
|---|---|---|
| `ate_total` | `E[i \| do(j=hi)] - E[i \| do(j=lo)]` | total effect along **all** directed paths |
| `ate_direct` | same, with the other parents of `i` frozen at their observational mean | the **controlled direct effect**, i.e. what the single edge claims |
| `effect_std` | `std_v( E[i \| do(j=v)] ) / sd(i)` | scale-free strength, independent of the lo/hi choice |
| `modifier` | for each co-parent `k`, the spread over `s` of `E[i\|do(k=hi,j=s)] - E[i\|do(k=lo,j=s)]`, over `sd(i)`; max over `k` | how much `j` changes the effect **of another parent** |

`effect_std` rather than `ate_total` is the strength measure because an **even**
mechanism cancels at symmetric endpoints: `X2 = X1^2` gives `ate_total ~ 0` for a
genuinely strong edge. That is a regression test
(`test_effect_std_survives_symmetric_mechanisms`), not a hypothetical: on `scm3`,
`S2 -> X1` has `ate_total = 0.007` and `effect_std = 0.987`.

Labels (thresholds are recorded in the output JSON, so any figure can be re-derived):
- `strong` — `effect_std >= negligible_effect` (default 0.02).
- `modifier_only` — negligible average effect but `modifier > 5 * effect_std`.
- `weak` — negligible average effect and no modification.

## Measured: `ds_scm3_continuous`

| edge | `effect_std` | `ate_direct` | `modifier` | label |
|---|---|---|---|---|
| `S2->X1` | 0.98694 | 0.0069 | 0.0000 | strong |
| `S3->X2` | 0.98489 | 0.6046 | 0.0000 | strong |
| `S3->X3` | 0.94228 | 0.7345 | 0.0000 | strong |
| `X1->X5` | 0.86744 | 1.8073 | 0.0000 | strong |
| `X2->X5` | 0.43097 | -0.0522 | 0.0000 | strong |
| `S4->X4` | 0.38511 | 0.0004 | 0.0000 | strong |
| `X2->X4` | 0.36109 | -0.0299 | 7.0619 | strong |
| **`S5->X4`** | **0.01490** | 0.0285 | **7.0619** | **modifier_only** |

`S5 -> X4` is the **only** edge with no average causal effect: `effect_std` is 26x below
`S4 -> X4` and 58x below the strongest edge. Its `modifier` score is attributed entirely
to `X2` (7.06) and is exactly zero for `S4` (3.6e-16) — which is precisely the generative
term `e * S5 * X2`. The mechanism is visible directly:

```
E[X4 | do(X2=x, S5=s)]      x=-1      x=0      x=+1
      s = -1              +2.0631  +0.2631  -0.3369     <- X2 effect DECREASING
      s =  0              +0.8631  +0.2631  +0.8631     <- symmetric
      s = +1              -0.3369  +0.2631  +2.0631     <- X2 effect INCREASING
```

S5 flips the **sign** of X2's effect on X4 while having no effect of its own.

## Measured: `ds_scm_equal` (the control)

All 8 edges are `strong`, and the two parents of X4 are equal to four decimals:
`S4 -> X4` = 0.66918, `S5 -> X4` = 0.66914. This is an independent confirmation that
`scm_equal` is built as intended, and it explains why X4 recovers **both** parents there
while it recovers only one on `scm3`.

## What the numbers license

**Claim (supported).** The edges our selector omits are those with negligible *average*
causal effect — 26x below the recovered edges on `scm3`, while on the equal-strength
control (all edges 0.67) the same configuration recovers both parents of X4. The earlier
`Q_NORM` "in-degree-1 bias" reading of the negative arms is therefore wrong: the query
budget was not the binding constraint.

**Limitation (must be stated, not hidden).** `S5 -> X4` is not irrelevant, it is an
**effect modifier**: zero main effect, large interaction. An additive aggregator
`sum_j A_ij V(x_j)` cannot represent such a term, and no average-effect estimand credits
it. Detecting moderators requires a non-additive aggregator and is left as future work.
Reporting `modifier` alongside `effect_std` is what keeps the first claim honest.

## Effect-weighted structure scores

`scripts/edge_effect_ground_truth.py --learned-dag <adjacency.csv>` reports

```
recall_weighted = sum(effect_std over RECOVERED true edges) / sum(effect_std over ALL true edges)
```

so missing an `effect_std ~ 0` edge barely moves the score while missing a strong edge
costs full. On `scm3`, a model that recovers all 7 strong edges and misses `S5 -> X4`
scores `recall_plain = 0.875` but `recall_weighted = 0.997`.

## Reproduce

```
python scripts/edge_effect_ground_truth.py --dataset ds_scm3_continuous
python scripts/edge_effect_ground_truth.py --dataset ds_scm_equal
python -m pytest tests/test_edge_effect_ground_truth.py -q
```

Output: `data/<dataset>/edge_effect_ground_truth.json`.
