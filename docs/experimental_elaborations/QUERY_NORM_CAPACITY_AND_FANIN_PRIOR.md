# Query-norm capacity Z and the fan-in prior

Status: DESIGN (no code written yet). This document fixes the equations and the
config surface before implementation.

Scope. It (i) re-derives the query normalisation in the paper notation, with the
normalisation constant written `Z`, (ii) proves that `Z` is nothing other than a
*fan-in capacity measured in edges*, (iii) shows that the current `auto` rule
sets that capacity to `N` (i.e. no cap at all), (iv) specifies how to declare a
smaller capacity as a **prior on the in-degree** without killing the
initialisation, and (v) derives how that prior should be set for the random
ER-`k` DAGs produced by `scm_ds/random_scm.py`.

Supersedes the notation of `QUERY_FANIN_SCALE_BUDGET.md` (which writes the same
constant as `F = query_fanin_scale`). Mapping: `F = Z^-2`, see Section 12.

---

## 1. Notation and the forward path

| symbol | meaning | code |
|---|---|---|
| `N` | number of candidate parents (structural keys) | `n_keys` |
| `i`, `j` | child (query row) index, candidate parent (key) index | |
| `k_j` | structural key of candidate parent `j` | |
| `q_i` | raw structural query of child `i` | |
| `u_i` | unit query direction, `u_i = q_i / ||q_i||` | `F.normalize(query)` |
| `M_i` | learnable per-node query norm, `M_i = exp(s_i) > 0` | `exp(query_norm_log_scale)` |
| `Z` | score normalisation constant (a divisor) | `Z = 1/sqrt(query_fanin_scale)` |
| `c_ij` | directional alignment `<u_i, k_j>` (a cosine) | |
| `l_ij` | structural logit of edge `j -> i` | `scores` |
| `T` | additive gate offset | `init_edge_offset` |
| `beta`, `gamma`, `zeta` | Hard-Concrete temperature and stretch interval | `init_tau`, `init_gamma`, `init_zeta` |
| `p` | a target edge posterior | `query_centroid_max_p` |
| `K` | **capacity**: number of parents affordable at posterior `p` | new |
| `q` | prior (declared) in-degree, i.e. the target capacity | `max_in_edges` |

The structural keys are orthonormal by construction (orthonormal frame /
orthogonal key projection):

```
<k_a, k_b> = delta_ab                                                        (0)
```

`normalize_query=True` replaces the standard scaled dot product by

```
l_ij = <M_i u_i , k_j> / Z = M_i c_ij / Z                                    (1)
```

Two deliberate differences from vanilla attention: the query norm `||q_i||` is
**deleted** (replaced by the penalised scalar `M_i`), and the divisor is a fixed
`Z` rather than `sqrt(E)`.

The Hard-Concrete gate turns the logit into an edge posterior. With the
stretch constant

```
c = beta * ln( -gamma / zeta )                                               (2a)
P_ij = P(z_ij > 0) = sigmoid( l_ij - T - c )                                 (2b)
```

and the deterministic (eval-time) gate

```
z_ij = clamp( sigmoid( (l_ij - T)/beta ) * (zeta - gamma) + gamma , 0 , 1 )  (2c)
```

**Default constants** (`query_norm.py`): `beta = 0.5`, `gamma = -1.1`,
`zeta = 1.1`, hence `-gamma/zeta = 1` and

```
c = 0            (exactly, for the default gamma = -zeta)                    (2d)
```

so `P_ij = sigmoid(l_ij - T)`. Two thresholds of (2c) are used repeatedly:

```
z_ij > 0   <=>  l_ij > T + beta*ln( -gamma/zeta ) = T + c       (= T here)    (2e)
z_ij = 1   <=>  l_ij >= T + beta*ln( (1-gamma)/(zeta-1) ) = T + 0.5*ln(21)
                                                        = T + 1.5223         (2f)
```

The `z > 0` threshold in (2e) is exactly the `P_ij >= 0.5` threshold of (2b):
the deterministic gate opens precisely when the edge posterior crosses one half.
And (2f) explains the recurring magic default `query_centroid_max_p = 0.8209`:
`sigmoid(0.5*ln 21) = sigmoid(1.5223) = 0.82089`, i.e. *"the posterior at which
the deterministic gate saturates"*. `DEFAULT_CENTROID_MAX_P = 0.9` is slightly
above it.

Every number in this document was verified numerically against (1)-(3) and (2c).
They are quoted for `p = 0.8209` as it appears in the configs, which is the
rounded form of `sigmoid(0.5*ln 21)` - hence e.g. `x = 1.5225` below rather than
the exact `1.5223`.

### Threshold logit

Define, for a target posterior `p`,

```
x(p) = logit(p) + T + c                so that   P_ij >= p  <=>  l_ij >= x(p) (3)
```

Numbers used throughout (defaults, `c = 0`):

| mode | `T` | `p` | `x(p)` |
|---|---|---|---|
| split (`init_edge_offset = ln 3`) | 1.0986 | 0.8209 | 2.6211 |
| split | 1.0986 | 0.9 | 3.2958 |
| homogeneous (`T` dropped) | 0 | 0.8209 | 1.5225 |
| homogeneous | 0 | 0.9 | 2.1972 |

---

## 2. The budget identity

Let `P_K` be the orthogonal projector onto `span(k_1..k_N)`. By (0),

```
sum_{j=1..N} c_ij^2 = || P_K u_i ||^2 <= 1                                   (4a)
```

with equality iff `u_i in span(K)`. Equality holds in practice because
(i) `query_centroid_init` puts `q_i` in `span(K)` at epoch 0, and (ii) the
gradient of any score-based loss w.r.t. `q_i` is a combination of the `k_j`, so
the iterates stay in that span. Q_NORM observation O2 measures
`sum_j c_ij^2 = 1.0000` from roughly epoch 9 onwards. We therefore use

```
sum_{j=1..N} c_ij^2 = 1                                                      (4b)
```

This is the crux: **the directional budget of a query row is conserved.** A row
cannot align with many keys at once; it can only redistribute a fixed unit of
squared cosine. The only quantity that can inflate every logit of a row at once
is `M_i` (a scale, not a redistribution) - which is why `M_i` is penalised.

---

## 3. Capacity theorem: `Z` is a fan-in cap measured in edges

**Theorem.** Fix a row `i` and a target posterior `p`, and let
`A_i(p) = { j : P_ij >= p }`. Then

```
|A_i(p)| <= K_i(p) := M_i^2 / ( Z^2 * x(p)^2 )                               (5)
```

and the bound is attained for every integer `K <= K_i(p)`.

*Proof.* By (1) and (3), `j in A_i(p)` iff `M_i c_ij / Z >= x(p)` iff
`c_ij >= Z x(p) / M_i =: c*`. Since `c* > 0` (x(p) > 0 is enforced by the
resolver) each such `j` contributes at least `c*^2` to the sum in (4b), hence
`|A_i(p)| * c*^2 <= 1`, i.e. `|A_i(p)| <= 1/c*^2 = M_i^2/(Z^2 x(p)^2)`.
Attainment: for any parent set `S` with `|S| = K <= K_i(p)` take

```
u_i = (1/sqrt(K)) * sum_{j in S} k_j                                         (5a)
```

which is a unit vector with `c_ij = 1/sqrt(K) >= c*` for `j in S` and `c_ij = 0`
otherwise. QED

**Corollary (reparametrisation).** `Z` and the capacity are in bijection:

```
Z(K) = 1 / ( x(p) * sqrt(K) )        K(Z) = 1 / ( Z^2 x(p)^2 )               (6a)
F(K) = Z^-2 = K * x(p)^2                                                     (6b)
```

So `query_fanin_scale` carries **no information beyond a number of edges**.
"`F = 68.69`" and "`capacity = 10 parents at posterior 0.8209 in split mode`"
are the same statement. All configuration should be done in edge units.

**Remark (what the row looks like at capacity).** At (5a) the non-parents sit at
`c_ij = 0`, i.e. `P_ij = sigmoid(-T-c)` - 0.25 in split mode, 0.5 in homogeneous
mode - and `z_ij = 0` by (2e) whenever `T >= 0`. So capacity `K` buys
"`K` parents at posterior `p`, everything else at the neutral floor". Pushing
non-parents *below* the floor requires `c_ij < 0`, which spends budget from
(4b) and therefore *reduces* the number of parents affordable at `p`. The
capacity in (5) is thus an upper bound on a sharp row, not a promise.

---

## 4. What the current `auto` rule does

`resolve_query_fanin_scale` / `query_fanin_scale_from_centroid_p` choose `Z` so
that a query at the centroid of **all** `N` keys gives **each** of the `N`
candidates posterior `p`, at `M_i = 1`. At that centroid `c_ij = 1/sqrt(N)`, so
`l_ij = M_i x(p) sqrt(K/N)`-style algebra gives

```
Z_0 = 1 / ( x(p) * sqrt(N) )       [ F = N * x(p)^2 ]                        (7a)
K_i(p) = M_i^2 * N                                                           (7b)
```

By (7b) with `M_i = 1` the capacity is exactly `N`:

> The current automatic rule grants every row enough capacity to hold **all**
> `N` candidates at posterior `p`. It imposes no fan-in constraint whatsoever.

This was intentional (the centroid init must start "all-on" so the value stream
can reconstruct from all parents), but it has a structural consequence: the
all-on block row - the shortcut the whole `normalize_query` machinery exists to
remove - is exactly affordable, at full confidence, by construction. And it
gets worse with `N`, because `F` grows linearly:

| `N` | `F = N x^2` (split, p=0.8209) | `1/Z = sqrt(F)` | `dP/dc` at the threshold = `sqrt(F)/4` |
|---|---|---|---|
| 10 | 68.7 | 8.3 | 2.1 |
| 50 | 343.5 | 18.5 | 4.6 |
| 100 | 687.0 | 26.2 | 6.6 |
| 400 | 2748.0 | 52.4 | 13.1 |

At `N = 400` a single row can hold 400 parents at `P = 0.82`, and one unit of
cosine moves the posterior 13x faster than at `N = 10` while `lambda_l0` prices
a full row at `~1.6e-3`. The hypothesis of this document is that this is a
material handicap for DAG retrieval at large `N`, and that it is invisible at
`N = 10` (where Q_NORM/T1 indeed found `F` was not the binding constraint).

---

## 5. Two places a capacity can live - and they are not interchangeable

By (5) the capacity of a row is `M_i^2 / (Z^2 x^2)`. There are exactly two
knobs, with different semantics:

| | knob | scope | learnable | when it acts |
|---|---|---|---|---|
| initial capacity | `Z` | global, all rows | no | epoch 0 onwards, in the forward pass |
| prior capacity | `mu` (target of the penalty on `M_i`) | per row | yes, buy-out-able | whenever the structural loss is active |

### Route (a) - hard prior: shrink `Z`. Rejected as default.

Setting `Z_q = 1/(x sqrt(q))` (`F = q x^2`) gives capacity `q` directly. But the
initialisation is still the all-`N` centroid, where `c_ij = 1/sqrt(N)`, so

```
l_init = x(p) * sqrt(q/N)                                                    (8a)
P_init = sigmoid( x(p) sqrt(q/N) - T - c )                                   (8b)
z_init > 0  <=>  x(p) sqrt(q/N) > T          (by (2e))                       (8c)
```

The init signal decays like `sqrt(q/N)`. Concretely, split mode, `p = 0.8209`
(`x = 2.6211`, `T = 1.0986`), `q = 10`:

| `N` | `l_init` | `P_init` | `z_init` |
|---|---|---|---|
| 10 | 2.621 | 0.821 | 1.00 |
| 50 | 1.172 | 0.518 | 0.08 |
| 100 | 0.829 | 0.433 | **0** |
| 400 | 0.414 | 0.335 | **0** |

By (8c) the init gate survives only while `q/N > (T/x)^2 = 0.1757`, i.e.
`q > 70` at `N = 400`. Below that the deterministic gate passes **nothing** at
initialisation - the same failure mode as the `F = 12.07` incident (Section 4 of
`QUERY_FANIN_SCALE_BUDGET.md`). In homogeneous mode `T = 0` and (8c) is always
satisfied, but the init still loses its `z = 1` saturation. This is why route
(a) is kept only as an ablation, behind an explicit guard.

### Route (b) - annealed prior: keep `Z = Z_0`, move the penalty target. Default.

Keep (7a), so **epoch 0 is bit-identical to today**, and impose the capacity
through the existing structural penalty
`lambda_query_norm * sum_i relu(M_i - mu)^2`. By (7b), capacity `K` corresponds
to

```
mu(K) = sqrt( K / N )                                                        (9)
```

and the present code pins `mu = 1`, i.e. `K = N` - the penalty exists but
prices no fan-in at all. Declaring a prior in-degree `q` means annealing the
capacity from `N` down to `q`:

```
K(t)  = N + (q - N) * rho(t),     rho: 0 -> 1 monotone                      (10a)
mu(t) = sqrt( K(t) / N )                                                    (10b)
R(t)  = lambda_query_norm * sum_i relu( M_i - mu(t) )^2                     (10c)
```

`rho(0) = 0` gives `mu = 1` exactly (today's behaviour), so the squeeze starts
from the un-modified state and no init is ever destroyed. Linear-in-`K` is
chosen (rather than linear in `mu`) so that the schedule is interpretable in
edges.

**Schedule clock.** `rho` is measured in *structure-phase* epochs counted from
an anchor installed at the FIRST structure phase, not from the global epoch.
Under adaptive training everything runs in a single `fit()`, so `current_epoch`
is global and a raw window is expired by the (long) reconstruct phases - this is
exactly the lesson recorded in the `query_norm.py` header ("the budget saturates
much LATER than any preset window") and the fix already used for
`_descendant_warmup_anchor` in `adaptive_trainer.py`. The penalty acts on the
structural stream (`query_norm_log_scale` is routed as a structural parameter),
so structural time is the correct clock.

---

## 6. The price of exceeding the prior

A row that wants `P` parents at posterior `p` needs, by (5) and (7b),
`M_i >= sqrt(P/N)`. Substituting into (10c) with `mu = sqrt(q/N)` at the end of
the anneal, its penalty is

```
R_i(P) = ( lambda_query_norm / N ) * ( sqrt(P) - sqrt(q) )^2 ,   P > q       (11)
dR_i/dP = ( lambda_query_norm / N ) * ( 1 - sqrt(q/P) )                     (12a)
d2R_i/dP2 = ( lambda_query_norm / (2N) ) * sqrt(q) * P^(-3/2) > 0           (12b)
```

Properties, all desirable for a *prior* (as opposed to a constraint):

* `R_i(q) = 0` and `dR_i/dP = 0` at `P = q`: the prior is free up to the
  declared in-degree, so a correctly-specified prior costs nothing;
* convex and increasing in `P` by (12): the marginal price of extra parents
  rises smoothly towards `lambda/N`, so the prior hardens rather than clips;
* strictly per-row, so a genuinely high-in-degree node can buy its parents if
  the structural loss pays more than (11) - which is exactly the ER situation,
  where the last topological nodes legitimately exceed the mean in-degree.

**Caveat - `lambda_query_norm` does not transfer across `N`.** (11) carries an
explicit `1/N`. At `N = 400`, `q = 10`, going from 10 to 15 parents costs
`1.26e-3 * lambda_query_norm`, i.e. `1.3e-6` at the current default `1e-3` -
nothing. Meanwhile the *pull-down* term at the start of the squeeze is
`relu(1 - mu)^2 = (1 - 0.1581)^2 = 0.709` per row, i.e. `0.284` summed over 400
rows at the same `lambda` - large. The penalty is therefore strongly asymmetric
in `N`: it pushes `M_i` down hard and prices over-spend not at all. Two
consequences for the implementation:

1. `lambda_query_norm` must be an explicit axis of the experiment, not
   inherited;
2. consider reporting (and possibly using) the mean-normalised form
   `R = lambda * mean_i relu(M_i - mu)^2` so the magnitude is `N`-independent.
   This is a behaviour change for existing configs and must be a flag, not a
   silent fix.

**Caveat - `M_i` is a scale, not a redistributor (Q_NORM O6).** Lowering `M_i`
deflates every logit of the row uniformly; it does not preferentially delete the
weakest edge. The prior becomes a *fan-in* prior only indirectly: with a smaller
`M_i`, keeping any edge above `x(p)` requires a larger `c_ij`, and by (4b)
larger cosines can only be bought by concentrating the budget on fewer keys.
That indirection is the mechanism claim under test, and it is the reason the
squeeze must be gradual. See the falsifier in Section 10.

---

## 7. Choosing `max_in_edges` for random ER-`k` DAGs

### What the sampler actually controls

`scm_ds/random_scm.py::_sample_dag` draws a random topological order, then picks

```
m = round( degree * n_nodes )   edges uniformly WITHOUT replacement
from the  n_slots = C(n_nodes, 2)  forward pairs.
```

So `degree` fixes the **total** edge count; in-degree, out-degree and depth are
emergent. In particular `degree` is a **mean** in-degree, not a maximum:

```
E[ in-degree averaged over nodes ] = m / n = degree                         (13a)
```

### In-degree law

Indexing nodes by topological position `i = 0..n-1`, node `i` has `i` admissible
parents, each present with probability `p_e = m / C(n,2) = 2*degree/(n-1)`
(exactly hypergeometric because the draw is without replacement; Binomial is the
standard approximation, exact in the `n -> inf` limit):

```
D_i ~ Binomial( i , p_e ),      p_e = 2*degree/(n-1)                        (13b)
E[D_i] = i * p_e   ->   E[D_{n-1}] = 2*degree                              (13c)
```

Averaging (13b) over positions gives (13a), and the root fraction is
`P(D = 0) = (1/n) sum_i (1-p_e)^i`, which is precisely the existing
`expected_er_roots`. The **pooled** in-degree CDF and its quantile are

```
P(D <= d) = (1/n) * sum_{i=0}^{n-1} P( Bin(i, p_e) <= d )                   (14a)
Q(alpha)  = min { d : P(D <= d) >= alpha }                                  (14b)
```

`Q(alpha)` is the recommended estimator for `max_in_edges`: "the prior covers a
fraction `alpha` of the nodes". `er_indegree_quantile(n_nodes, degree, alpha)`
will implement (14). Evaluated, and compared with the `p95 in` column of the
Monte Carlo below (which estimates the same quantity up to the discreteness of
`d` and the difference between a pooled quantile and the mean of per-graph
quantiles):

| n | deg | `Q(0.95)` from (14) | MC `p95 in` | diff |
|---|---|---|---|---|
| 10 | 4 | 8 | 7.71 | +0.29 |
| 50 | 4 | 9 | 9.10 | -0.10 |
| 100 | 4 | 10 | 9.33 | +0.67 |
| 400 | 1 | 3 | 3.03 | -0.03 |
| 400 | 2 | 5 | 5.34 | -0.34 |
| 400 | 4 | 10 | 9.56 | +0.44 |

All within one edge, so (14) is a usable closed form and the Binomial
approximation to the without-replacement draw is harmless at these sizes.

### Monte-Carlo verification (300 DAGs per row, exact sampler rule)

| n | deg | m | slots | fill % | mean in | p95 in | max in | depth | roots % |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 10 | 45 | 22.2 | 1.00 | 2.66 | 3.08 | 2.9 | 40.3 |
| 20 | 1 | 20 | 190 | 10.5 | 1.00 | 2.81 | 3.64 | 3.5 | 41.8 |
| 50 | 1 | 50 | 1225 | 4.1 | 1.00 | 3.11 | 4.38 | 4.5 | 42.6 |
| 100 | 1 | 100 | 4950 | 2.0 | 1.00 | 3.11 | 4.87 | 5.0 | 42.9 |
| 400 | 1 | 400 | 79800 | 0.5 | 1.00 | 3.03 | 6.01 | 6.3 | 43.1 |
| 10 | 2 | 20 | 45 | 44.4 | 2.00 | 4.42 | 4.91 | 4.8 | 21.9 |
| 20 | 2 | 40 | 190 | 21.1 | 2.00 | 4.84 | 5.83 | 5.9 | 22.8 |
| 50 | 2 | 100 | 1225 | 8.2 | 2.00 | 5.21 | 6.96 | 7.1 | 23.6 |
| 100 | 2 | 200 | 4950 | 4.0 | 2.00 | 5.37 | 7.67 | 8.3 | 24.2 |
| 400 | 2 | 800 | 79800 | 1.0 | 2.00 | 5.34 | 9.16 | 10.0 | 24.5 |
| 10 | 4 | 40 | 45 | **88.9** | 4.00 | 7.71 | 8.18 | 8.1 | 11.3 |
| 20 | 4 | 80 | 190 | 42.1 | 4.00 | 8.46 | 9.79 | 9.6 | 11.8 |
| 50 | 4 | 200 | 1225 | 16.3 | 4.00 | 9.10 | 11.44 | 12.2 | 11.8 |
| 100 | 4 | 400 | 4950 | 8.1 | 4.00 | 9.33 | 12.85 | 14.0 | 12.3 |
| 400 | 4 | 1600 | 79800 | 2.0 | 4.00 | 9.56 | **14.61** | **17.6** | 12.5 |

The `roots %` column reproduces `expected_er_roots` (43 / 24 / 12 % for
ER-1/2/4), and `mean in` reproduces (13a) exactly, so the law (13) is verified
against the sampler.

### Three conclusions

1. **ER-4 does not cap the in-degree at 4.** The mean is 4 by construction, but
   the maximum is 8.2 at `n = 10` and 14.6 at `n = 400`. Setting
   `max_in_edges = degree` would starve the downstream nodes, whose expected
   in-degree is already `2*degree` by (13c). Use `Q(0.95)` from (14) - about
   `9.6` for ER-4 - or the realised maximum for an oracle arm.
2. **`degree` says nothing about depth.** ER-4 has longest path 8 (`n = 10`) to
   17.6 (`n = 400`), not 4.
3. **ER-4 at `n = 10` fills 88.9 % of all admissible forward slots** - it is
   nearly the complete DAG on 10 nodes, so `Q(0.95) = 8 = N - 2` and any
   fan-in prior there is close to vacuous. This is an independent reason to
   expect the effect only at larger `n`, and a reason not to read a null result
   at `n = 10` as a refutation.

### Prior strength at a glance

Split mode, `p = 0.8209`, `q = Q(0.95)` from the table:

| n | deg | `q` | `mu = sqrt(q/N)` | logit shrink `1/mu` | `R_i` at `P = q + 5` (`/lambda`) |
|---|---|---|---|---|---|
| 50 | 4 | 9.1 | 0.427 | 2.3 | 1.09e-2 |
| 100 | 4 | 9.3 | 0.305 | 3.3 | 5.36e-3 |
| 400 | 4 | 9.6 | 0.155 | 6.5 | 1.31e-3 |
| 400 | 2 | 5.3 | 0.115 | 8.7 | 2.06e-3 |

The last column is the whole `lambda`-calibration problem in one place: the same
five extra parents cost 8x less at `n = 400` than at `n = 50`.

---

## 8. Config surface

No `F`, no `Z`: everything in edges.

```yaml
experiment:
  normalize_query: true
  query_centroid_max_p: 0.8209   # the posterior p in the capacity definition (2f)
  query_capacity_init: auto      # capacity at M_i = 1, in EDGES; auto = n_keys -> sets Z, eq (6a)
  max_in_edges: null             # prior capacity in EDGES; null = disabled (= n_keys) -> sets mu, eq (9)
  query_fanin_scale: auto        # DERIVED = Z^-2. An explicit float still wins (legacy configs).
training:
  lambda_query_norm: 1.0e-3      # must be re-calibrated with N, see Section 6
  max_in_edges_anneal_epochs: 0  # STRUCTURE epochs to squeeze n_keys -> max_in_edges (0 = immediate)
  max_in_edges_anneal_idle_epochs: 0
```

Resolution order (in `populate_seq_lengths_from_dataset`, where `n_keys` is
known):

1. explicit numeric `query_fanin_scale` -> honoured verbatim, everything below
   is skipped (byte-identical legacy behaviour);
2. else `K_init = query_capacity_init` (`auto -> n_keys`), and
   `query_fanin_scale = F(K_init) = K_init * x(p)^2` by (6b);
3. `q = max_in_edges` (`null -> n_keys`), `mu_end = sqrt(q/n_keys)` by (9);
4. validate `1 <= K_init <= n_keys`, `1 <= q <= n_keys`;
5. **init-gate guard**: compute `l_init`, `P_init`, `z_init` from (8) with
   `K = K_init`; if `z_init == 0` while `query_centroid_init` is on, raise with
   the numbers and the two fixes (raise `K_init`, or drop `T`).

Startup log line (readable, replaces the raw `F`):

```
[query-norm] n_keys=400 | capacity_init=400 edges | max_in_edges=10 edges
             Z=0.01908 (F=2748.0) | x(p)=2.6211 @ p=0.8209 | mu_end=0.1581
             P_init=0.8209 z_init=1.000 | anneal=30 struct epochs
```

Configurations of interest:

| arm | `query_capacity_init` | `max_in_edges` | effect |
|---|---|---|---|
| control (today) | `auto` | `null` | bit-identical to current behaviour |
| annealed prior | `auto` | `Q(0.95)` | route (b) |
| oracle | `auto` | realised max in-degree | route (b), upper bound on the gain |
| hard prior (ablation) | `Q(0.95)` | `Q(0.95)` | route (a); guard may refuse it |

---

## 9. Interaction with the rest of the system

* **`query_centroid_init`** - unchanged under route (b) by construction, since
  `mu(0) = 1` and `Z = Z_0`. Under route (a) it is the thing that breaks (8).
* **homogeneous mode** - `init_edge_offset` is dropped (`T = 0`), so `x(p)` is
  smaller (1.5225 instead of 2.6211), `F` is smaller by `(1.5225/2.6211)^2 =
  0.337`, and the init gate can never die by (2e). All equations hold with
  `T = 0`; `n_keys = N` for both routes.
* **L0 / `lambda_l0`** - complementary: L0 prices *posterior mass*, the fan-in
  prior prices *how many edges can be confident at once*. The prior costs
  nothing at eval time and does not shift the L0 threshold.
* **gradient routing** - `query_norm_log_scale` is a structural parameter, so
  the squeeze acts on the structural stream only. Unchanged.
* **`shared_query`** - a single `log_scale` shared by cross and self blocks must
  be de-duplicated by parameter id when the target is written, symmetrically to
  `collect_query_norm_penalty`.
* **adaptive phases** - anchor as in Section 5; the target must be re-applied on
  every phase switch (a cheap idempotent write in `on_train_epoch_start`).
* **sweeps** - `max_in_edges` is a *prior*, therefore an OPT-IN extra arm, never
  a silent default: the main arms must not receive an inductive bias the
  baselines lack. In the dagsweep it can be derived from the `dag.degree`
  *generation setting* via (14); deriving it from the realised graph is
  ground-truth leakage and is admissible only in an arm explicitly named
  `oracle`.
* **`validate_dimensions` / `fanin_saturating`** - both currently rewrite `F`
  from `n_keys`; they must be made capacity-aware so they cannot silently
  restore `K = N` and delete the feature.

---

## 10. Predictions and falsifier (recorded before the runs)

Primary prediction: under route (b) with `q = Q(0.95)`, DAG-retrieval metrics
(SHD, `precision_cross`, `tpr`) improve, and the improvement **grows with `N`**,
being ~0 at `n = 10` (Section 7, conclusion 3) and largest at `n = 400`.

Mechanism prediction: `mean learned in-degree` falls towards `q` while
`M_i` settles near `mu_end` for most rows and above it for the genuinely
high-in-degree (late topological) rows - i.e. the buy-out in (11) is exercised
selectively.

Falsifiers:

* recall/`tpr` falls roughly uniformly across rows while `M_i -> mu` for all
  rows: the squeeze is *deflating* rows rather than pruning them (the O6 risk of
  Section 6), and the mechanism claim is refuted;
* `M_i` stays at 1 and nothing moves: `lambda_query_norm` is too small
  (Section 6, caveat 1) - re-run on the `lambda` axis before concluding;
* the effect does not scale with `N`: the capacity argument of Section 4 is not
  the binding constraint, and the feature should be abandoned rather than tuned.

New diagnostics required to evaluate the above: `query_norm/target_mu`,
`query_norm/cap_target_edges` (= `K(t)` from (10a)),
`query_norm/cap_actual_edges` (= `mean_i M_i^2 * N`), and the learned in-degree
distribution alongside the existing `mean_M` / `max_M`.

---

## 11. Symbol / code mapping

| paper | code | notes |
|---|---|---|
| `Z` | `1/sqrt(experiment.query_fanin_scale)` | divisor, eq (1) |
| `F` | `experiment.query_fanin_scale` | `F = Z^-2`, eq (6b) |
| `M_i` | `exp(query_norm_log_scale[i])` | structural parameter |
| `mu` | `module.query_norm_target` | plain float, read every step |
| `K_init` | `experiment.query_capacity_init` | new, in edges |
| `q` | `experiment.max_in_edges` | new, in edges |
| `p` | `experiment.query_centroid_max_p` | default 0.9; 0.8209 = gate saturation |
| `T` | `experiment.init_edge_offset` | dropped in homogeneous mode |
| `beta, gamma, zeta` | `init_tau, init_gamma, init_zeta` | 0.5, -1.1, 1.1 -> `c = 0` |
| `x(p)` | (new helper) | eq (3) |

`query_fanin_scale` is kept as the code symbol (no rename) so that every
existing config, checkpoint and sweep rule keeps working; it becomes a *derived*
quantity that users are not expected to set.

---

## 12. Open questions

1. Should the penalty be normalised by `N` (Section 6, caveat 1)? It makes
   `lambda_query_norm` transferable across `n_nodes` but changes existing runs.
   Proposal: add `query_norm_penalty_reduction: sum | mean`, default `sum`.
2. Is `rho(t)` linear enough? A cosine or exponential-in-`mu` squeeze may be
   gentler at the end. Proposal: implement linear-in-`K`, keep the shape behind
   a single enum if the linear version deflates rows.
3. Should route (a) instead anneal `Z` itself (capacity `N -> q` in the forward
   pass)? It avoids the penalty entirely but rescales all rows globally and
   fights the already-learned `M_i`; not planned.
4. `Q(alpha)`: is `alpha = 0.95` the right convention, or should the prior be
   the expected maximum in-degree (`~14.6` for ER-4 at `n = 400`)? The oracle
   arm answers this empirically.
