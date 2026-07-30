# `query_fanin_scale`: the directional budget, the in-degree tax, and how to set it

**Status:** derivation + measurement, 2026-07-29
**Evidence:** `experiments/6_INVESTIGATIONS/Q_NORM/results/baseline_learn_centroid_init_1_8953226/`
(notebook `investigate_S3_X4_spurious_barrier.ipynb`, Sections 1-4, plus direct
checkpoint probes)
**Arms produced:** `Q_NORM/centroid_init_fanin_69` (F = 68.69),
`Q_NORM/centroid_init_fanin_109` (F = 108.66)
**Tool:** `scripts/query_fanin_capacity.py` (recomputes every number below)

---

## 1. Setting

With `remove_query_projection=true`, `remove_key_projection=true`,
`struct_embedding_type=orthogonal_fixed`, `shared_query=true`, `shared_key=true`
and `normalize_query=true`, the structure logit of edge *parent j -> child i* is

```
log_alpha_ij = M_i * cos(q_hat_i, k_j) * sqrt(F)      F = query_fanin_scale
```

where `q_hat_i` is the unit query of child `i`, `k_j` the (exactly orthonormal —
verified, `|K K^T - I| ~ 1e-7`) key of node `j`, and `M_i = exp(log_scale_i)` the
learnable per-node norm budget. The Hard-Concrete gate then gives

```
la      = log_alpha - T                      T = init_edge_offset = ln 3
P(edge) = sigmoid(la - beta*ln(-gamma/zeta)) = sigmoid(la)   (stretch = 0 here)
z       = clamp(sigmoid(la/beta)*(zeta - gamma) + gamma, 0, 1)
```

with `beta = init_tau = 0.5`, `gamma = -1.1`, `zeta = 1.1`.

## 2. The budget is `= 1`, not `<= 1`

The docs so far described `sum_j cos^2(q_hat_i, k_j) <= 1` as a *ceiling*. It is
in fact an **identity** as soon as the query lies in `span(K)`:

* the score gradient is `d(score)/dq_i = sum_j (dL/dscore_ij) * k_j`, i.e. it lives
  in `span(K)` — the optimiser can never push the query *out* of the key span;
* `query_centroid_init=true` **starts** the query inside `span(K)`.

Measured on the run: concentration `= 0.216` at random init, then `1.0000` from
epoch 9 to the end. So **there is never spare budget**: every unit of `cos^2` given
to one key is taken from another. `M_i` does not relieve this — it multiplies the
whole row uniformly (final values 1.10-1.27, `query_norm_target = 1.0`).

## 3. The in-degree tax (closed form)

A non-parent is *not* free: at `cos = 0` its posterior is `sigmoid(-T) = 0.25`, so
suppressing it also costs budget. For a row with `m` parents at `P_on = sigmoid(d)`
and `n - m` non-parents at `P_off = sigmoid(-d)` (same margin `d`, `n = L_S + L_X`):

```
F(m, d) = [ m*(T + d)^2 + (n - m)*(T - d)^2 ] / M_i^2
```

Inverting for the best affordable margin gives the **capacity table** (`n = 10`,
`M = 1`, `T = ln 3`):

| F | m=1 | m=2 | m=3 | m=4 | centroid `z` / `P_cross` |
|---|---|---|---|---|---|
| **12.07** (baseline) | 0.853/0.147 | 0.789/0.211 | 0.707/0.293 | 0.608/0.392 | **0.000 / 0.500** |
| 20.00 | 0.894/0.106 | 0.854/0.146 | 0.807/0.193 | 0.757/0.243 | 0.336 / 0.578 |
| 41.00 | 0.942/0.058 | 0.923/0.077 | 0.900/0.100 | 0.874/0.126 | 0.802 / 0.716 |
| **68.69** (arm 1) | 0.968/0.032 | 0.958/0.042 | 0.946/0.054 | 0.931/0.069 | **1.000 / 0.821** |
| **108.66** (arm 2, exact 108.63) | 0.984/0.016 | 0.979/0.021 | 0.973/0.027 | 0.966/0.034 | **1.000 / 0.900** |
| 163.50 | 0.992/0.008 | 0.990/0.010 | 0.987/0.013 | 0.984/0.016 | 1.000 / 0.950 |

**Read the first row.** At `F = 12.07` a *correct* 3-parent row can only reach
`0.71 / 0.29`, while a *wrong* 1-parent row reaches `0.85 / 0.15`. Under a loss
that rewards confident, low-residual-dependence rows, the wrong sparse row wins.
That is the mechanism, and it is a *capacity* statement, not an optimiser one.

## 4. Why 12.07 was doubly unlucky

`12.07 = n * (ln 3)^2 = 10 * 1.0986^2`. Hence at the centroid

```
log_alpha = sqrt(F/n) = 1.0986 = ln 3 = T   =>   la = 0
```

* `P(edge) = sigmoid(0) = 0.500` — measured exactly at epoch 0 for every cross edge;
* the **deterministic** gate value is `z = 2.2*sigmoid(0/0.5) - 1.1 = 0.0`.

So at evaluation the centroid initialisation passes **nothing**, and in training
only `E[z] ~ 1/3`. The whole reconstruct phase fits the function through a
half-dead gate. The centroid init was, in effect, cancelled by the fan-in scale.

## 5. Measured confirmation

* X4 (true parents S4, S5, X2) ends at `S4 = 0.811`, others `0.108-0.174`. The
  `m = 1` optimum at `F = 12.07`, `M = 1.098` predicts `0.869 / 0.131` — the run
  landed on the best row it could **buy**. Nothing was left on the table.
* Section 3's direction sweep toward the true-parent centroid never turns on more
  than one true parent at any `a`, at any of the three anchors — consistent with
  "the multi-parent row is not affordable", not with "a barrier hides it".
* Section 4's budget sweep (direction fixed, `M_i` swept to 10) also never opens a
  second parent: `M` rescales the row, so it sharpens the winner instead. Both
  hypotheses in the notebook were therefore *incomplete*: the binding constraint is
  `F`, which sets how much `cos^2` each decision costs.

## 6. Choosing F

Two natural targets, both of the form `F = n * x^2` where `x` is the desired
`log_alpha` at the centroid (`log_alpha(centroid) = M * (1/sqrt(n)) * sqrt(F)`):

**(a) Gate at its maximum — `F = 68.69`. RECOMMENDED.**
```
z = 1  <=>  la >= beta*ln((1 - gamma)/(zeta - 1)) = 0.5*ln(2.1/0.1) = 1.5223
       <=>  x_sat = 1.5223 + ln 3 = 2.6209
F = n * x_sat^2 = 10 * 2.6209^2 = 68.69
```
This is the exact formalisation of "the gate value at the centroid should be the
maximum, i.e. 1". Every candidate parent starts at **full** weight, so reconstruct
learns the function of the complete parent set before structure prunes, and a
3-parent row is afterwards affordable at 0.95/0.05.

**(b) Posterior target — `F = 108.66`.** `P_cross(centroid) = 0.90` at
`x = ln 9 + ln 3 = 3.2958`, i.e. `F = 108.63`, rounded up to **108.66** in the
arm (`P = 0.9001`). Clears the clamp with margin; the in-degree penalty
almost vanishes (m=1 0.984 vs m=3 0.973), at the price of a heavier pruning burden.

Both keep the budget *binding* (at the `m = 3` optimum, parents consume 0.68 of the
budget at `F = 68.69` and 0.60 at `F = 108.66`, the rest buying suppression) and
both **increase** the gradient scale at the decision point,
`dP/dcos = M*sqrt(F)*P(1-P) = sqrt(F)/4`: 0.87 -> 2.07 -> 2.61. There is no
vanishing-gradient trade-off in this direction.

> **F scales with the node count**: `F = n * x^2`, `n = L_S + L_X`. Do **not** copy
> 68.69 to a dataset with a different number of nodes — recompute with
> `scripts/query_fanin_capacity.py`.

## 7. Risks to monitor

1. **Block rows.** HSIC never penalises *extra* parents, so an all-on centroid row
   is HSIC-optimal. The only prune forces are L0 (`lambda_l0 = 1e-6` -> a full
   10-wide row costs ~4e-5, two orders below the HSIC differences measured in the
   barrier sweep) and NOTEARS on the X->X part (`kappa = 1e-3` -> an all-on 5x5
   block costs ~5e-2, genuinely strong). Watch `l0_penalty` / mean in-degree in the
   first structure epochs; if rows do not deflate, the follow-up arm is
   `lambda_l0 = 1e-5`.
2. **Cross-vs-self init balance drifts.** `init_edge_offset = ln 3` equalised the
   cross gate (0.25) with the directed self edge (0.5 * 0.5) *at* `log_alpha = 0`.
   The general matched value is `offset = ln(2 + e^x)` — 2.757 at `F = 68.69`,
   3.368 at `F = 108.66`. We deliberately **keep** `ln 3`: raising the offset also
   raises `T` in the capacity formula and gives the budget back (m=3 would fall to
   0.86/0.14). The new starts are cross 0.821 vs self-directed 0.466 (arm 1) and
   0.900 vs 0.482 (arm 2). If spurious S->X edges reappear, the *next* arm is the
   matched offset — not a smaller F.
3. **`M_i` should relax toward 1.0.** With enough F, nodes no longer need to
   over-spend; `M_i` staying above ~1.2 would mean F is still too small.
