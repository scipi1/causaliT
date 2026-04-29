# Edge decisiveness — iter_11 vs `e1_self_only_hsic`

This note documents why the iter_11 attention weights look **undecisive**
(probabilities clustered around 0.5) while the `e1_self_only_hsic` test in
`experiments/tests/hsic_parents/results/e1_self_only_hsic_64894415/`
produces **sharply decisive** edges, and what changes are needed to
replicate (or improve) the latter at paper scale. The analysis is based on
a side-by-side reading of the two YAMLs and of the four affected modules:

- `causaliT/core/modules/attention.py`
  (`ToeplitzAttention`, `CausalCrossAttention`, `SigmoidCrossAttention`,
  `AttentionLayer`)
- `causaliT/core/architectures/single_causal/model.py`
- `causaliT/training/forecasters/single_causal_forecaster.py`
  (gradient routing, weight decay split)
- `causaliT/training/score_sparsity_cv.py` (CV stage — to be deprecated)

---

## 1. Config diff that matters

The two configs share the same architecture (`SingleCausalLayer`,
`ToeplitzAttention` self / `CausalCrossAttention` cross, `factorization=svfa`,
`d_model=48`, single head, 2 decoder layers). The differences that
control **edge decisiveness** are concentrated in three blocks:

| Knob | `e1_self_only_hsic` (decisive) | `iter_11` (undecisive) | Effect on $\sigma(s/\tau)$ |
|---|---|---|---|
| `training.weight_decay` (reconstruction) | **0.01** | 0.01 | — |
| `training.structural_weight_decay` | **null** (= same wd, but only on Q,K,φ) | 0.1 (10× stronger) | shrinks $\\|Q\\|, \\|K\\|$ → shrinks $s = QK^\top/\sqrt{d_qk}$ → drives $\sigma(s/\tau)$ to 0.5 |
| `training.lr` (reconstruction) | 0.001 | 0.001 | — |
| `training.structural_lr` | **3e-4** | 3e-5 (10× weaker) | the structural side cannot escape the diffuse basin enforced by wd |
| `training.lambda_hsic_self` | **1.0** | 1.0 | — |
| `training.lambda_hsic_cross` | **0.0** (S dropped from HSIC) | 1.0 | HSIC(S, res) is non-identifying (see hypothesis in the YAML) |
| `training.lambda_self_score_sparse` | **0.0** | $\lambda_{\text{cv}} \cdot 0.5$ | extra L1 on $P_{\text{edge}}$ pulls scores to 0 → gate $\sigma(S/\tau) \to 0.5$ |
| `training.lambda_cross_score_sparse` | **0.0** | $\lambda_{\text{cv}}$ | same on cross |
| `training.kappa` (residual contrast) | **0.0** | non-zero in iter_11 | another regularizer competing with HSIC |
| `training.use_tau_annealing` | true but **start = end = 4.0** | start 3.0 → end 1.0 (full run) | no annealing; tau is just a constant |
| `training.use_score_sparsity_cv` | **false** | true (3-fold, 60 epochs/fold) | each fold runs a fresh head from scratch; the CV picks $\lambda$ that *minimises HSIC*, and that minimum is achieved with diffuse attention because diffuse attention averages-out the residual |
| `staged_training.use_calibration` / `use_causal_init` | false | true | extra regularisers re-balancing whatever is still moving |
| `experiment.max_epochs` | **500** (single fold) | 200 main + 60·3 CV | iter_11 spends most of its budget exploring λ; the production run on the chosen λ is short |

Everything else (gradient routing, `freeze_tau_during_anneal`, hard masks
off, learned attention, single seed) is identical.

---

## 2. Why iter_11 sits at $\approx 0.5$

The Toeplitz / sigmoid attentions all compute, per edge,

$$
\text{att}_{ij} = \sigma(S_{ij}/\tau)\,\sigma(A_{ij}/\tau)
\quad\text{with } S = (QK^\top + KQ^\top)/\!\sqrt 2,\;A = (QK^\top - KQ^\top)/\!\sqrt 2
$$

The **only thing that drives a sigmoid away from 0.5 is a large pre-activation**
$S/\tau$ (or $A/\tau$). Its variance is, to first order,

$$
\operatorname{Var}(S_{ij}) \;\approx\; \operatorname{Var}(QK^\top)
\;\approx\; \frac{\\|q\\|^2\,\\|k\\|^2}{d_{qk}}.
$$

Two iter_11 ingredients work together to drag $\\|q\\|$ and $\\|k\\|$ to zero:

1. **`structural_weight_decay = 0.1`** is applied by AdamW directly on the
   Q-projection, K-projection (and φ if present), at *every* step of the
   structural sub-optimizer. With AdamW the post-step decay factor is
   $1 - \eta\cdot \mathrm{wd}$, i.e. with $\eta_{\text{struct}} = 3\!\times\!10^{-5}$
   and $\mathrm{wd}=0.1$, $\\|W\\|$ shrinks by about $3\!\times\!10^{-6}$
   per step on top of the gradient pull toward zero coming from the L1 on
   $P_{\text{edge}}$. Over 200 epochs $\times \sim$200 batches that is a
   geometric decay of order $\exp(-0.12)\approx 0.89$ — small in absolute
   terms but *much* larger than the geometric decay seen on the
   reconstruction side, where the same wd is paired with a $10\times$
   larger learning rate (so the relative shrinkage per *useful* step is
   the same, but the structural side is starving for signal).

2. **`structural_lr = 3e-5`** is $10\times$ smaller than the reconstruction
   lr. Combined with the explicit L1 on $P_{\text{edge}}=\sigma(S/\tau)$
   coming from `lambda_self_score_sparse` (and on the cross score from
   `lambda_cross_score_sparse`), the only stable point for the structural
   parameters is precisely the one where $\\|q\\|=\\|k\\|=0$ and every
   sigmoid sits at 0.5. The HSIC(self) gradient that *would* push $S$
   away from 0.5 has to compete against (a) wd, (b) L1 on $P_{\text{edge}}$,
   (c) the smaller lr — and it loses.

The visible signature is exactly what we see in
`nb_eval_att_scores.ipynb`: rows of $\sim\!0.5$ everywhere, with a tiny
$\pm$ bias from $\sigma(A/\tau)$ — the antisymmetric direction picks up
*some* signal because tanh-like terms are not regularized to zero, but
the gate $\sigma(S/\tau)$ is.

The CV makes this worse, not better. `_build_cv_config` in
`causaliT/training/score_sparsity_cv.py` runs each candidate $\lambda$ in a
fresh fold for 60 epochs *with the same wd / lr ratio* — i.e. each fold
starts from a freshly diffuse attention, the HSIC(self) loss is then
*lower* on that diffuse attention (because residuals are S-mixed at the
floor), and the “min_hsic” selection rule picks the most-shrinking
$\lambda$. The CV is therefore biased toward the undecisive regime.

---

## 3. Why `e1_self_only_hsic` stays sharp

The same algebra now favours the structural side:

- `structural_weight_decay = null` means the structural sub-optimizer
  inherits the **reconstruction** wd (0.01). Crucially, AdamW's wd term is
  computed against `structural_lr`. With `structural_lr = 3e-4` and
  `wd = 0.01` the per-step contraction is $\eta\cdot \mathrm{wd} = 3\!\times\!10^{-6}$,
  i.e. **40$\times$ smaller** than iter_11. $\\|q\\|, \\|k\\|$ are free
  to grow.
- `lambda_self_score_sparse = 0`, `lambda_cross_score_sparse = 0`,
  `lambda_decisive = 0`, `lambda_group_l1 = 0`, `lambda_kl = 0` — the only
  structural objective is HSIC(self), which has a non-trivial minimum
  whenever the residual `res_X` is independent of `X`. That minimum
  *requires* a sharp attention pattern: with a diffuse attention every
  child mixes its parents and `res_X` retains a lot of `X` information.
- `lambda_hsic_cross = 0` removes the non-identifying HSIC(S, res) term.
- 500 epochs in a single fold give the structural side time to escape the
  initial diffuse basin.

Result: $\sigma(S/\tau)$ saturates at $\approx 1$ on true parents and at
$\approx 0$ on others, the antisymmetric $\sigma(A/\tau)$ picks the
direction, and the product is decisive.

The post-training metric in `kfold_summary.json` confirms:

- `val_x_r2 = 0.84`, `test_x_r2 = 0.84` — the reconstruction is fine
  *despite* the absence of any sparsity prior, because HSIC(self) alone
  is enough to shape the attention.
- `val_self_score_sparse = 0.26`, `val_cross_score_sparse = 0.025` —
  the *natural* sparsity (just the L1 norm of $P_{\text{edge}}$) is
  already small without any explicit penalty.

---

## 4. Proposed iter_12: replicate, then improve

The minimal replication of `e1_self_only_hsic` at paper scale (10 seeds,
3 SCMs) is to drop the CV, drop the L1 on the score, drop HSIC-cross, and
unify the wd/lr ratios. Concretely, in the next-iteration config:

```yaml
training:
  # --- learning rates ---
  lr: 1.0e-3
  structural_lr: 3.0e-4          # was 3.0e-5 in iter_11 (×10 weaker)
  use_scheduler: false

  # --- weight decay: the SAME on reconstruction and structure ---
  weight_decay: 1.0e-2
  structural_weight_decay: null  # fall back to weight_decay

  # --- structural objective: HSIC(self) only ---
  lambda_hsic_self: 1.0
  lambda_hsic_cross: 0.0
  kappa: 0.0

  # --- NO score-sparsity, NO group-L1, NO decisiveness L1, NO KL ---
  lambda_self_score_sparse: 0.0
  lambda_cross_score_sparse: 0.0
  lambda_group_l1: 0.0
  lambda_decisive: 0.0
  lambda_decisive_cross: 0.0
  lambda_kl: 0.0

  # --- tau is a constant ---
  use_tau_annealing: true
  tau_anneal_start: 3.0
  tau_anneal_end: 3.0
  tau_anneal_idle_epochs: 0
  tau_anneal_transient_epochs: 0
  freeze_tau_during_anneal: true

  # --- gradient routing on (iter_6 dynamics) ---
  use_gradient_routing: true

  # --- single-stage training ---
  max_epochs: 500

staged_training:
  use_calibration: false
  use_causal_init: false
  use_score_sparsity_cv: false        # *** CV dropped; faster too ***
```

This is **byte-equivalent** to `e1_self_only_hsic` on the regularizer
front; the only differences are dataset, seed sweep, and the bookkeeping
needed to log per-seed HSIC / SHD.

### What we keep from iter_11

- iter_10’s ToeplitzAttention with the **`/sqrt(2)` variance-preserving
  split** and **constant `tau = 3.0`** (already in
  `causaliT/core/modules/attention.py`).
- `init_tau` plumbed through `single_causal/model.py::_attn` — settable
  per-experiment without changing code.
- The new SVFA shared-DAG / multi-head-V mode (see §5) — lets us go to
  multi-head V *without* multiplying the structural parameters.

### Improvements over `e1_self_only_hsic`

1. **Multi-seed sweep** — `e1_self_only_hsic` is a 1-fold/1-seed
   diagnostic; `iter_12` runs ten seeds via `sweeper/sweep.yaml` so the
   decisiveness claim is reproducible.
2. **Multi-head V via shared-DAG attention** — in
   `AttentionLayer.__init__` a new flag `shared_dag_across_heads: bool =
   True` makes Q/K and the attention score single-head while V keeps
   `n_heads` independent channels. This preserves the score-imposed
   sparsity (the per-head value channels are linearly mixed by
   `out_projection`, which cannot synthesise an edge that the score has
   zeroed out) while giving the reconstruction more capacity. At
   `n_heads = 1` the mode is byte-identical to single-head.
3. **`d_model = 64`** (vs 48) for the larger SCMs — capacity headroom on
   reconstruction without growing the DAG mask. Optional; iter_12 keeps
   48 by default to stay comparable with iter_9–11.
4. **No CV** — the structural sparsity is *implicit* in HSIC(self), so
   the CV stage is now waste. Dropping it saves $3\times 60 = 180$
   epochs per arm and removes the bias documented in §2.

---

## 5. Code changes required (already implemented)

The new flag and shared-DAG mode are wired through three files:

- `causaliT/core/modules/attention.py`:
  `AttentionLayer` gains `shared_dag_across_heads: bool = True`. When
  `True`, Q and K are single-head (`d_qk`) while V is per-head
  (`d_v · n_heads`). Inside the inner attention, a new branch detects
  `query.dim() == 3 and value.dim() == 4` and computes
  `V = einsum("bls,bshd->blhd", A, value)` so the same `(B, L, S)` score
  is broadcast across the V-head dimension. The output projection
  flattens `(B, L, H, d_v) → (B, L, H·d_v)` and mixes heads back to
  `d_model` — a **linear** post-attention map that cannot manufacture an
  edge that the score has set to zero.
  Implemented for `ToeplitzAttention`, `CausalCrossAttention`,
  `SigmoidCrossAttention`. `LieAttention`, `ToeplitzLieAttention`,
  `PhiSoftMax`, `ScaledDotAttention*` keep the legacy per-head behaviour.
  Reverting to the legacy mode is a one-line change
  (`shared_dag_across_heads: false`).
- `causaliT/core/architectures/single_causal/model.py`:
  `SingleCausalLayer.__init__` accepts `shared_dag_across_heads` (default
  `True`) and threads it through `attn_shared_kwargs` and `_attn`.
- `causaliT/core/architectures/noise_aware/model.py`:
  same flag added to `NoiseAwareSingleCausalLayer._attn` for parity.

The activation temperature is **already** a constant scalar (not a
parameter, not a buffer) since iter_10. Tau annealing is therefore a
no-op for these modules — `start = end = 3.0` is enough to disable it,
no code change needed.

---

## 6. Sanity-check checklist for iter_12

After the first run inspect the following — they should all hold:

1. `val_x_r2 ≥ 0.80` on each SCM (reconstruction is not sacrificed).
2. `val_self_score_sparse ≤ 0.30` *without* an L1 penalty (the natural
   sparsity from HSIC(self) alone).
3. The `nb_eval_att_scores.ipynb` heatmaps show edge probabilities
   bimodal at $\\{0, 1\\}$, **not** unimodal at 0.5.
4. `kfold_summary.json::val_hsic_self` is in the same ballpark as
   `e1_self_only_hsic_64894415` (≈ $1.5\!\times\!10^{-3}$). If it is
   $10\times$ smaller, the structural side has collapsed back to
   diffuse — most likely a wd/lr regression.
5. SHD (`eval_seed_sweep/aggregate_dag.json`) shows clear preference for
   true edges across seeds — i.e. the decisiveness is not seed-dependent.

If (1) and (2) hold but the heatmaps still look diffuse, the next thing
to try is to **lower `init_tau` to 1.5** (sharper sigmoid) and/or to set
`weight_decay = 0` for the structural sub-optimizer only via
`structural_weight_decay: 0.0`. Both knobs are already exposed and only
require a config change.
