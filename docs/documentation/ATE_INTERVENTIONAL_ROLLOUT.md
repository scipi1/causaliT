# ATE by Interventional Roll-out

Status: IMPLEMENTED. Estimator used by `eval_interventions` for the
AttentionSelector family. Supersedes the one-shot estimator of
`ATE_EVALUATION.md` (still correct on normalization, baselines, file formats).

---

## 1. Problem

Two independent defects made every **mediated** (indirect) effect come out
**exactly zero**. Evidence (`cheater/ds_scm1/model_1`, an arm handed the TRUE
DAG): every direct effect accurate to a few %, every indirect effect
`model_ate = 0.000000` with `abs_error = |true_ate|`.

| intervention | target | model_ate | true_ate | path |
|---|---|---|---|---|
| `S3=1.5` | `X2` | 1.419 | 1.50 | direct |
| `S3=1.5` | `X4` | **0.000** | 1.80 | indirect `S3->X2->X4` |
| `S3=1.5` | `X5` | **0.000** | 0.90 | indirect `S3->X2->X5` |

**Defect 1 - harness bug (Informer-style blanking misread).** The model blanks
the X value column **only for the query path**; the key/value path
(`x_actual`) must carry real parent values. The old harness passed an all-zero
X tensor *and relied on blanking to hide it* - but blanking only touches the
query clone, so the zero tensor became the **values of every X parent**,
identical in treated and baseline runs, and every mediated effect cancelled in
the difference.

**Defect 2 - estimand (overcontrol / mediator bias).** Even with real X, a
single forward pass computes $f_i(\mathrm{pa}(i)) = \mathbb{E}[X_i \mid
\mathrm{pa}(i)]$. Conditioning on a realised mediator $X_2$ while varying $X_1$
measures the **controlled direct effect**, not the total effect; for
$X_1{\to}X_2{\to}X_3$ the direct effect is genuinely $0$. The total effect
requires **propagating** the intervention through the learned equations, not
regressing once.

> Why the usual "regress and difference" recipe does not apply: applied
> benchmarks (TARNet/CFRNet, AIPW) compute $\mathbb{E}[\mu_1(W)-\mu_0(W)]$ from
> one regression, valid only because $W$ is restricted to **pre-treatment**
> covariates. Our model regresses each node on all others, so mediators are in
> the conditioning set and the shortcut is unavailable.

The fix is the standard one for "learn an SCM, then answer do-queries" (DECI,
causal normalizing flows, VACA; classically Robins' parametric g-formula):
treat the trained model as structural equations, **mutilate** at the
intervened node, **propagate**, **Monte-Carlo average**.

---

## 2. Estimand and algorithm

Sources $S=(S_1..S_{L_S})$ exogenous, intermediates $X=(X_1..X_{L_X})$. The
model gives an amortised conditional mean per node; under the additive-noise
assumption encoded by the reconstruction loss,

$$X_i = f_i(\mathrm{pa}(i)) + e_i, \qquad e_i \perp \mathrm{pa}(i),\ \mathbb{E}[e_i]=0 .$$

Target (DECI eq. 2; matches `compute_ate_ground_truth` / `ate_total`):

$$\mathrm{ATE}(a,b) = \mathbb{E}\!\left[X_Y \mid do(D{=}a)\right] - \mathbb{E}\!\left[X_Y \mid do(D{=}b)\right].$$

**Roll-out (generative forward).** Per Monte Carlo draw $m$: sample the
intervened sources $S^{(m)}$, initialise $X^{(0)}$, and iterate

$$X^{(k)}_i = f_i\!\left(S^{(m)}, X^{(k-1)}\right) + \mathbb{1}[\,i \notin D\,]\, e_i^{(m)}, \qquad X^{(k)}_j = d_j \ \ (j \in D),$$

then

$$\widehat{\mathbb{E}}[X \mid do(D)] = \tfrac{1}{M} \textstyle\sum_m X^{(m,K)} .$$

Only the **last** converged iterate $X^{(m,K)}$ of each draw is kept.

* **Clamping = mutilation.** Overwriting the do-set every round is exactly
  $G_{do}$: the clamped node no longer depends on its parents, downstream nodes
  still see its value as key/value. `clamp = {}` is the observational
  (generative) run.
* **Convergence.** Each round resolves one more topological layer, so a DAG on
  $L_X$ nodes stops **exactly** within $K = L_X$ rounds (early stop when
  $\Delta = \max_i |X^{(k)}_i - X^{(k-1)}_i| = 0$). A non-zero final $\Delta$
  flags a **cyclic** learned graph, whose interventional semantics are
  undefined - a reported diagnostic, not an assertion.
* **Common random numbers.** Same $S$ draws and same residual draws for
  treated and baseline; the ATE is a difference of two correlated means, so
  CRN removes most of the MC variance.

---

## 3. Variant A vs variant B

They differ only in the noise term $e_i^{(m)}$ of the update rule.

**A - deterministic mean roll-out** (`noise="none"`, $e_i^{(m)} \equiv 0$).
Propagates conditional means: $\widehat{X}_4 = f_4(S, f_2(S_3))$. Drops the
mediator's residual variance, so by a 2nd-order expansion
$\mathbb{E}[f_4(m_2+e_2)] - f_4(m_2) \approx \tfrac12 f_4''(m_2)\,\mathrm{Var}(e_2)$,
and the **bias of the ATE** is

$$\mathrm{bias}_{ATE} \approx \tfrac12\,\mathrm{Var}(e_2)\,\big[\, f_4''\!\left(m_2(a)\right) - f_4''\!\left(m_2(b)\right) \big],$$

**zero whenever the mechanism has constant curvature in the mediator** (any
linear or quadratic mediator term).

**B - residual-bootstrap ancestral sampling** (`noise="residual"`, default).
Restores the dropped noise: the per-node residual pool
$e_i^{(n)} = x_i^{(n)} - f_i(\mathrm{pa}(i)^{(n)})$ is collected teacher-forced
on the training split; one residual vector $e^{(m)}$ is drawn per draw $m$ and
**re-added every round**, so mediator noise reaches nonlinear children. Yields
draws from the model's interventional *distribution* (not just its mean), hence
also variances / quantiles / CATE.

**B is the sound one in general** (no approximation to defend, matches DECI's
$z \sim p_z$ and simulation g-computation). Its assumptions: additive,
parent-independent noise (already implied by the reconstruction loss; check
with `eval_anm_residual_hsic`), and a pool from data the model did not overfit.

**On our benchmark the two agree.** Only `X4, X5` have X-parents
(`ds_scm2/3`): the product term $e\,S_5 X_2$ is *linear* in the mediator
(contributes nothing), the quadratic terms $f X_2^2,\ h X_2^2$ add a constant
$f\,\mathrm{Var}(e_2)$ that cancels in the ATE difference, and only
$g\tanh(X_1)$ fails to cancel - bounded by $\tfrac12 \max|\tanh''|
\mathrm{Var}(e_1)|g| \approx 0.004|g|$, three orders of magnitude below the
effects ($0.5$-$1.8$). `ds_scm1` is linear in its mediators, so A is exact
there. **The SCMs are left unchanged** - the bias never came from the product
term, and weakening a benchmark to suit an estimator is not defensible.

**Recommendation adopted: B is the default, A kept as `noise="none"`.** Their
agreement is a one-line robustness statement; a disagreement would reveal extra
curvature in the *learned* mechanisms.

---

## 4. Code map

| piece | location | role |
|---|---|---|
| `causal_predict` | `causaliT/training/forecasters/attention_selector_forecaster.py` | generative forward; the fixed-point iteration, clamp (mutilation), residual re-add. Returns `(x_final, n_iter_used, rollout_delta)`. |
| `_build_residual_pool` | `causaliT/evaluation/eval_funs/eval_interventions.py` | one teacher-forced pass over the train npz -> residual pool `(n_train, L_X)`, normalized space. |
| `_find_train_npz` | same | locates `ds_train.npz` / `ds.npz`. |
| `run_mc_predictions` | same | dispatch (`isinstance` AttentionSelector -> roll-out, else legacy one-shot), per-chunk CRN generator, records `rollout_delta` / `rollout_iters`. New args `propagation`, `noise`, `datadir_path`. |
| `eval_ate_mc` | same | passes `propagation="rollout"`, `noise="residual"` by default; cache invalidated via required `rollout_delta` column. |

Interfaces:

```python
forecaster.causal_predict(
    data_source,                 # (B, L_S, 2) normalized, already intervened if S-side
    x_init,                      # (B, L_X, 2) initial state; index column preserved
    clamp={pos: value},          # do(X_pos = value); {} = observational run
    residual_pool=pool_or_None,  # (N_pool, L_X); None -> variant A
    n_iter=None,                 # default L_X; early stop on Delta == 0
    generator=None,              # CRN: same seed for treated & baseline
)  # -> (x_final, n_iter_used, rollout_delta)
```

New columns in `predictions_mc.csv`: `rollout_delta` (convergence; ~0 for an
acyclic learned graph, non-zero flags a cycle), `rollout_iters`. Function name
and output paths (`eval/eval_ate_mc/files/ate_metrics_mc.{csv,json}`) are
unchanged, so `eval_seed_sweep` and the results notebook keep working.

### do(X_i) interventions

`do(X_1 = x)` needs no special model-side handling - the roll-out clamps
whatever node it is given (normalize `x` with `norm_stats["input"]`). Ground
truth: `compute_interventional_expectation` is already node-agnostic, and
`get_scm_for_dataset()` rebuilds a live SCM at eval time, so X-node ground
truth can be computed on the fly - **no dataset regeneration**.

---

## 5. Validation

* `tests/test_atsel_causal_predict.py` (10 tests, all passing): linear chain
  $X_1{\to}X_2{\to}X_3$ - roll-out recovers the total effect while one-shot
  returns 0; clamped slots never move; acyclic graphs converge with
  $\Delta{=}0$ within $L_X$ rounds; a cyclic stub does not converge
  (diagnostic fires); a no-X->X-edge DAG reproduces the one-shot result
  exactly; variant B recovers the Jensen term $\mathbb{E}[(m+e)^2] = m^2 +
  \mathrm{Var}(e)$ that variant A drops; CRN reproducibility.
* `scripts/_validate_causal_predict.py` (non-destructive, real cheater
  checkpoint, `ds_scm1`): indirect ATE recovered, roll-out converged exactly.

  | intervention | target | legacy (one-shot) | roll-out | true |
  |---|---|---|---|---|
  | `S3=1.5` | `X4` | 0.000 | **1.69** | 1.80 |
  | `S3=1.5` | `X5` | 0.000 | **0.83** | 0.90 |
  | `S3=-0.5` | `X4` | 0.000 | **-0.61** | -0.60 |

  Variants A and B agree to ~0.001 here (linear mediators), matching Sec. 3.

---

## 6. Impact on published numbers

* All previously reported **indirect** ATE numbers (all arms) are artefacts
  (`model_ate = 0`, `abs_error = |true_ate|`) and must be regenerated.
* **Direct** numbers are unaffected where no X-parent exists and change only
  slightly otherwise.
* Only the **evaluation** needs re-running - no retraining. The notebook's
  `direct/indirect/zero` categories become meaningful for the first time.

---

## 7. References

1. Geffner et al. (2022). *Deep End-to-end Causal Inference.*
   arXiv:2202.02195 - sec. 3.3: mutilated-graph + topological-order simulation
   + Monte Carlo ATE estimator (our roll-out); ATE/CATE definitions (eqs. 2-3).
2. Javaloy, Sanchez-Martin, Valera (2023). *Causal Normalizing Flows.*
   NeurIPS 2023 - do-operator in exogenous space; same estimand,
   architecture-specific mechanics (justifies our fixed-point form).
3. Sanchez-Martin, Rateike, Valera (2022). *VACA.* AAAI 2022 - interventional
   queries by message passing over the intervened graph.
4. Robins (1986). *A new approach to causal inference...* Math. Modelling 7 -
   the parametric g-formula: A is the plug-in form, B the simulation form with
   residual resampling.
5. Rosenbaum (1984). JRSS A 147(5) - overcontrol/mediator bias (defect 2).
6. Cole & Hernan (2002). *Fallibility in estimating direct effects.* Int. J.
   Epidemiol. 31(1) - direct vs total effect.
7. Pearl (2009). *Causality*, 2nd ed., ch. 3 - truncated factorization /
   g-formula; controlled direct effect.
8. Shalit, Johansson, Sontag (2017). *Estimating Individual Treatment Effect.*
   ICML - single-regression ATE, valid because the conditioning set is
   pre-treatment covariates only (sec. 1).

## 8. Related documents

* `docs/documentation/ATE_EVALUATION.md` - normalization, baseline convention
  `do(S=0)`, file formats. Current except for the estimator itself.
* `docs/experimental_elaborations/EDGE_EFFECT_GROUND_TRUTH.md` - `ate_total` vs
  `ate_direct` on the ground-truth side.
* `experiments/7_PUBLISH/ATE/README.md` - the three arms and what they bracket.
* `docs/ideas/PARTIAL_ANM_REGRESSION.md`, `eval_anm_residual_hsic` - the
  residual-independence check variant B relies on (sec. 3).
