# Anti-Adjacency (Causal / Anti-Causal Log-Ratio) Loss

Structural objective of `OrderFactorizedSelfAttention` that replaces the plain
residual HSIC when `training.anti_adjacency_lambda > 0`.
Code: `AttentionSelectorForecaster._anti_adjacency_hsic`
(`causaliT/training/forecasters/attention_selector_forecaster.py`).
Model background: `docs/ideas/ORDER_FACTORIZED_VISION.md`.

## 1. Ingredients

| Symbol | Meaning | Code | Gradient |
|---|---|---|---|
| $r \in \mathbb{R}^N$ | node order scores (larger = earlier) | `order_scores` | **trained** (only structural parameter) |
| $P$ | relaxed permutation, $P = \mathrm{Sinkhorn}((\Theta(r) + s\,\mathcal{G})/\tau)$ | `P` | through $r$ |
| $M$ | soft precedence, $M = P T P^\top$, $M_{ii}=0$, $M_{ij} \approx \Pr(i \prec j)$ | `_last_M` | **attached** |
| $\pi_{ij}$ | symmetric existence probability | `_last_pi` | **detached** |
| $G$ | gate applied in the structure forwards (Hard-Concrete, relaxed path): the structural selector $G_S$ if `toggle_on_struct_att_selector`, otherwise the reconstruction selector $G_R$ | `G` | $G_S$ (`struct_gate_logits`): **trained** (structural). $G_R$ (`query_embed`): carries grad, but always owned by the reconstruction optimizer |
| $\theta_R$ | reconstruction parameters | - | **frozen** in structure phases |
| $\nu$ | permutation-null mean of the current phase (1 before calibration) | `hsic_null_stats["mean"]` | constant |
| $\epsilon$ | log floor (> 0) | `anti_adjacency_eps` | constant |
| $\lambda_a$ | anti-causal weight (> 0 activates the loss) | `anti_adjacency_lambda` | constant |

$T$ is the strict upper-triangular all-ones matrix, $s$ = `order_noise_scale`,
$\tau$ = `order_tau`, $\mathcal{G}$ the per-step Gumbel noise (shared by all
forwards of a step).

## 2. Two paired forwards

Same batch, same noise, same $G$; only the orientation of the order differs
(`set_orientation`):

$$
Z^{c} = G \odot M, \qquad Z^{a} = G \odot M^\top
$$

- causal pass: each node regresses on its selected EARLIER nodes,
  residual $r^{c}_t = x_t - \hat{x}^{c}_t$;
- anti-causal pass: each node regresses on its selected LATER nodes,
  residual $r^{a}_t = x_t - \hat{x}^{a}_t$.

The causal residuals are reused from the training step (cross-fit fold when
`hsic_cross_fit` is active); the anti-causal pass is one extra forward.
Per-pair HSIC matrices (estimator per `hsic_mode`, bandwidth adaptive or
phase-frozen):

$$
H^{c}_{ts} = \mathrm{HSIC}(x_s, r^{c}_t), \qquad
H^{a}_{ts} = \mathrm{HSIC}(x_s, r^{a}_t)
$$

## 3. Order-weighted scores

For each unordered pair $e = \{i, j\}$:

$$
C_e = M_{ij}\,H^{c}_{ji} + M_{ji}\,H^{c}_{ij}
\qquad \text{(later causal residual vs earlier node: should be LOW)}
$$

$$
A_e = M_{ij}\,H^{a}_{ij} + M_{ji}\,H^{a}_{ji}
\qquad \text{(earlier anti-causal residual vs later node: should be HIGH)}
$$

Both are clamped at $\ge 0$. $C$ softly weights the PREDECESSORS of each
residual: a descendant $k$ of $j$ enters $C$ with weight $M_{kj}$, small only
when the order is correct and sharp. $A$ softly weights the SUCCESSORS.

## 4. Loss

$$
\mathcal{L}_{\mathrm{AA}} =
\frac{\sum_{i<j} w_{ij}\left[\log\!\left(\epsilon + \frac{C_{ij}}{\nu}\right)
 - \lambda_a \log\!\left(\epsilon + \frac{A_{ij}}{\nu}\right)\right]}
{\sum_{i<j} w_{ij}},
\qquad w_{ij} = \mathrm{sg}(\pi_{ij})
$$

$\mathrm{sg}(\cdot)$ = stop-gradient. The value can be negative. Logged as
`{stage}_hsic_causal` ($\bar C$) and `{stage}_hsic_anticausal` ($\bar A$),
both $w$-weighted means.

## 5. Gradient flow

Only $r$ (`order_scores`) is updated, via two paths:

1. **Weighting path:** $\partial \mathcal{L}/\partial M$ through the
   coefficients of $C$ and $A$. Per pair it compares the two orientations,
   e.g. a large $H^{c}_{jk}$ (residual of $j$ vs its child $k$) pushes
   $M_{kj}$ down, i.e. $j$ before $k$.
2. **Residual path:** $r^{c}, r^{a}$ depend on $M$ through $Z^{c}, Z^{a}$
   (relaxed order path in structure phases): the order changes which inputs
   each regression sees.

Detached / not updated by this loss:

- $\pi$ (pair weights $w$);
- $\theta_R$: frozen. With `structural_grad: hsic_unrolled` it enters only
  through the DARTS virtual refit
  $\theta_R' = \theta_R - \eta \nabla_{\theta_R} \mathcal{L}_{\mathrm{recon}}$
  ($\eta$ = `unrolled.inner_lr`, $\mathcal{L}_{\mathrm{recon}}$ computed with the
  causal orientation); the gradient is written into structural params only;
- $\nu$ and the HSIC bandwidths.

Consequence: the loss cannot be lowered by degrading the regressor (e.g.
inflating $A$ through a poor anti-causal fit), nor by switching edges off
($w$ is detached and normalized).

## 6. Pair mass and partial collapse

With row-stochastic $P$ and $T + T^\top = \mathbf{1}\mathbf{1}^\top - I$:

$$
M_{ij} + M_{ji} = 1 - \langle P_{i\cdot}, P_{j\cdot} \rangle \in \left[\tfrac12, 1\right]
\quad (i \neq j)
$$

- hard permutation: exactly 1, each pair term is a convex combination of the
  two orientations, so the trivial "all weights zero" solution does not exist;
- flat order (`order_r_init: flat`): $1 - 1/N$;
- lower bound $1/2$ from column-stochasticity ($p_{ik} + p_{jk} \le 1$).

Residual risk: normalization uses $w$, not $M$, so the order can shrink $C_e$
by up to a factor 2 by keeping $i, j$ positionally tied. $A_e$ shrinks by the
same factor, so the incentive roughly cancels for $\lambda_a \ge 1$ and is
net positive for $\lambda_a < 1$ (works against sharpening the order).
Derived from the code, not measured.

## 7. Interaction with other options

- `hsic_aggregation` must be `plain`.
- Requires `self_attention_type: OrderFactorizedSelfAttention` and
  `homogeneous_nodes: true`.
- `hsic_exclude_descendants: true` is rejected at init. The order weighting
  already down-weights descendants softly in $C$ (exactly only for a hard
  order). A graph-based hard descendant mask must never be applied to $A$
  (it would delete the anti-causal signal); a C-only mask is conceivable but
  not implemented. The `hsic_descendant_*` keys are ignored.
- Validation stage computes the loss as a no-grad probe.

## 8. Two selectors (reconstruction / structural)

`OrderFactorizedSelfAttention` has two symmetric selectors with fixed owners,
toggled in `model.kwargs` (they replace the removed
`training.variable_selection_objective` and `order_existence_mode`). They
NEVER multiply: each forward applies exactly one of them.

| Forward | Gate applied |
|---|---|
| reconstruction (warmup / reconstruct) | $G_R$ (times $M_{hard}$ if `use_order_in_recon`) |
| structure (both orientations) | $G_S \odot M$ if `toggle_on_struct_att_selector`, else $G_R \odot M$ |

| `toggle_on_recon_att_selector` | `toggle_on_struct_att_selector` | Behaviour |
|---|---|---|
| true (default) | false (default) | $G_R$ in all forwards; in structure it is a plain weight (former `variable_selection_objective: recon`) |
| true | true | $G_R$ in reconstruction, trainable $G_S$ in structure |
| false | true | all-ones in reconstruction, $G_S$ in structure |
| false | false | order only, full DAG of the order (former `order_existence_mode: full`) |

- $G_R$: Hard-Concrete from the query/key descriptors, parameters
  (`query_embed`) always in the reconstruction group.
- $G_S$: free symmetric logits `struct_gate_logits`, always in the
  structural group (`STRUCTURAL_PATTERNS`), initial logit
  `struct_att_selector_init_logit` (default 2.0, $\pi_S \approx 0.88$). No L0
  on $G_S$ yet: HSIC alone drives it.
- `struct_att_selector_init: recon` replaces the constant init by ONE copy
  of the $G_R$ existence logits ($S_{sym} - \text{edge\_offset}$, so
  $\pi_S = \pi_R$) at the first forward of the first structure phase. The
  first structure step then equals the $G_S$-off run; afterwards $G_S$ is
  trained by the structural loss only while $G_R$ keeps following
  reconstruction. The persistent buffer `struct_selector_synced` prevents a
  second copy (also after a checkpoint reload). Observed with the uniform
  init (`adaptive_aa_ss`): $G_S$ barely moves and a dense structure forward
  gives no order signal ($C \approx A$).
- Learned graph (posterior, exclusion graph, `_last_pi`, DAG metrics):
  $\pi_S \odot M$ when $G_S$ is on, $\pi_R \odot M$ otherwise. The AA pair
  weights $w = \mathrm{sg}(\pi)$ follow the same rule.
- Legacy configs / checkpoints: `variable_selection_objective: recon` and
  `order_existence_mode: learned|full` load with a deprecation warning;
  `variable_selection_objective: struct` is rejected.
