# Conditional HSIC via Per-Edge Leave-One-Out: Counter-Proposal

Companion to `FROM_MARGINAL_TO_CONDITIONAL.md`. Same goal (edge weights modulated by a
conditional, Bayesian-style multiplier), same Bayes architecture — but with a corrected
statistic and a corrected likelihood.

---

## 1. The statistic: per-edge LOO contrast

For each candidate edge `i → j`, with the model frozen:

- `ε⁺ = X_j − f⁺(...)`: residual from the current forward pass (edge at weight `P_m`). **Free** — already computed for the MSE loss.
- `ε^{-i} = X_j − f^{-i}(...)`: residual from one masked forward pass with edge `i→j` forced to zero.

$$\Delta_{i \rightarrow j} = \text{nHSIC}(X_i,\ \epsilon_j^{-i}) \;-\; \text{nHSIC}(X_i,\ \epsilon_j^{+})$$

**The first argument is identical on both sides** (same variable, same kernel, same
bandwidth). Only the residual changes, so the difference is attributable to edge `i→j`
alone. The conditioning on the other parents enters **through the model**: `f^{-i}` has
already absorbed everything the remaining parents explain, so
`HSIC(X_i, ε^{-i})` asks "is there anything left that *only* X_i predicts?" — the
conditional-independence question.

Sign semantics:

| Δ | Meaning | Action |
|---|---------|--------|
| `≫ 0` | cutting the edge reintroduces dependence on X_i | load-bearing → γ → 1 |
| `≈ 0` | remaining parents cover X_i's contribution | redundant → stay at prior, L0 prunes |
| `< 0` | edge injects dependence | harmful → γ → 0 |

Why not the joint parent matrix as first argument (the original document's choice)?

1. **Comparability:** `HSIC(X_parents, ·)` and `HSIC(X_parents∖{i}, ·)` live on different
   kernel spaces (different dimension, different median-heuristic bandwidth), and the
   Bayes posterior depends only on their difference (see §3) — so the artifact lands
   directly in the multiplier.
2. **Attribution:** a joint kernel returns one number per node j; it cannot say *which*
   edge carries the dependence. In the redundant-pair case it is blind by construction
   (the surviving twin keeps the joint HSIC high/low identically).
3. **Power:** HSIC power decays with kernel dimension; 1-D kernels per edge keep every
   test at maximum power and mutually comparable.

## 2. What Δ *is* (the passage that was unclear)

HSIC estimates a concrete functional:

$$\text{HSIC}(U,V) \xrightarrow{B\to\infty} \|C_{UV}\|_{HS}^2 = \text{MMD}^2(P_{UV},\, P_U \otimes P_V)$$

— the squared Hilbert–Schmidt norm of the cross-covariance operator, i.e. the squared
distance between the joint distribution and the product of marginals (0 iff independent).

So Δ is a **contrast of two dependence measures**:

$$\Delta_{i\rightarrow j} = \text{MMD}^2(P_{X_i \epsilon^{-i}}, P_{X_i}P_{\epsilon^{-i}}) - \text{MMD}^2(P_{X_i \epsilon^{+}}, P_{X_i}P_{\epsilon^{+}})$$

Its status:

- **Not a loss.** Nothing is minimized; it is a diagnostic computed under `no_grad`.
- **Not a log-likelihood.** It has units of squared kernel-space distance, no absolute
  scale, and a null distribution that shifts with the data — which is exactly why
  `exp(−λH)` cannot serve as a likelihood.
- **A test statistic with a known null distribution.** Under independence, HSIC is a
  weighted sum of χ² variables, well approximated by a gamma distribution whose moments
  are computable from the kernel matrices. This is the bridge to Bayes (§3).

## 3. The Bayesian update is kept — the likelihood is repaired

The original document plugs `exp(−λH)` into Bayes' rule as the "likelihood of the edge
being on". Observe what that functional form *is*: an **exponential density in H** with
rate λ. The document is implicitly approximating

$$L(\text{edge on}) = p(\text{observe } H \mid \text{independence}) \approx \lambda e^{-\lambda H}$$

i.e. guessing the null density of the HSIC statistic as a one-parameter exponential and
curve-fitting λ at runtime (the whole EMA/`μ_HSIC`/clipping machinery).

The principled version of *the same move*: use the actual gamma null density,
moment-matched per edge, per step, from the kernel matrices:

$$L(\text{edge on}) = q_0(H^+;\alpha,\beta), \qquad L(\text{edge off}) = q_0(H^{-i};\alpha,\beta)$$

$$\boxed{\;\gamma_{i\rightarrow j} = \frac{q_0(H^+)\,P_m}{q_0(H^+)\,P_m + q_0(H^{-i})\,(1-P_m)}\;}$$

Identical formula to the original document — but now:

- the "likelihood" is a real density under a real model (independence);
- its parameters come from the kernel matrices, **no λ, no EMA, no clipping**;
- both terms share one kernel space, so the likelihood ratio is meaningful;
- γ is **detached** (LOO passes run under `no_grad`).

Variant: use tail probabilities (p-values) instead of density evaluations — more robust
in the tails, slightly further from the original formula. Density version preferred for
fidelity to the Bayes picture.

Honest caveat: this is a *one-sided* likelihood (we model `p(H | independent)` but not
`p(H | dependent)`, which has no fixed form). Standard for test-based Bayes factors.

## 4. Corrections to the original document

1. **"Pruning Redundant Nodes" claim is wrong.** If A is redundant, `H^{-A} ≈ H^+`, so
   the likelihood ratio ≈ 1 and γ = 1 — the posterior returns the *prior* `P_m`, no
   pruning signal. Pruning belongs to the L0 sparsity prior; γ only *modulates*.
2. **Joint-matrix HSIC comparison invalid** — replaced by the per-edge contrast (§1).
3. **λ/EMA/clipping (Phase 2, Section 3) removed** — replaced by moment-matched gamma
   calibration (§3).
4. **"Collider" framing dropped.** Residual-HSIC cannot orient v-structures
   (Markov-equivalent DAGs score identically). What the LOO contrast protects is
   *synergistic* parents (e.g. XOR): cutting one floods the residual with dependence.
5. γ must be stated as **detached**; the MLP-encoder framing maps onto our
   attention-selector as: LOO = re-run forward with the edge's gate logit masked.

## 5. Integration plan (attention_selector_forecaster)

- Keep the existing per-pair marginal HSIC regularizer (with descendant mask) as the
  differentiable gradient signal.
- Add γ as a **detached multiplicative gate on the per-edge HSIC loss terms**
  (replacing `use_attention_weighted_hsic`'s raw attention weighting). Forward pass and
  MSE stream untouched.
- LOO cost control: full LOO on small graphs first (exact reference); later top-k
  shortlist by current `P_m`, staggered LOO, or batched gate-masking.

## 6. Validation experiment (before trainer integration)

Standalone script on a small known SCM (chain + redundant copy + XOR pair), frozen
trained model: compute Δ per edge and verify the sign table in §1 empirically. Reuses
`causaliT/utils/hsic_utils.py`.
