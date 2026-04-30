# Attention Magnitude Balance: Self vs. Cross

**Status**: documented in iter_10 after the magnitude gap observed in
`experiments/single/calitrain/iter_9/single_Toeplitz_Sigmoid_svfa_scm{1,2,3}c`.

## TL;DR

> In iter_9, the **self-attention** scores produced by `ToeplitzAttention` were
> ~one order of magnitude smaller than the **cross-attention** scores produced
> by `SigmoidCrossAttention`. The cause was **not** a missing $1/\sqrt{d_{qk}}$
> normaliser — that already lives at the dot-product level and is shared by
> both branches. The gap had two sources, one *learned* and one *structural*:
>
> 1. **Learned** (~10× factor): cross-attention had an extra `log_gain`
>    Parameter (init ≈ 10) and a redundant `log_tau_act` slope inside the
>    sigmoid; self-attention had neither. Removed in iter_10.
> 2. **Structural** (≤ 4× ceiling factor): cross uses **one** sigmoid,
>    self uses a **product of two** sigmoids ($\sigma(S/\tau)\cdot\sigma(A/\tau)$
>    from the Toeplitz decomposition). For the two branches to deliver the same
>    attention probability $p$, **the self-attention $W_q$ and $W_k$ projections
>    must produce a *more aligned* $QK^\top$ than the cross-attention ones**.
>    This part is *intrinsic to the Toeplitz factorisation* and is **not**
>    removed in iter_10 — it is a design choice that enforces
>    $P(i\!\to\!j) + P(j\!\to\!i) \le 1$ in self-attention (Prop. P2(a) of the
>    paper template). iter_10 only **partially compensates** by zeroing the
>    learned amplifier and putting both branches on the same constant
>    temperature.

## 1. The two branches, side by side

Let $s_{ij} = (Q_i K_j^\top)/\sqrt{d_{qk}}$ be the scaled dot-product score
(common to both branches: this is the Vaswani normaliser, applied identically
in cross and self via `ScaledDotAttention`'s `scale = d_qk ** -0.5`).

| Branch | Module | Activation (iter_10) | Range | Mid-point requirement |
|---|---|---|---|---|
| **Cross** (S → X) | `SigmoidCrossAttention` | $\displaystyle\sigma\!\left(\frac{s_{ij}}{\tau}\right)$ | $(0, 1)$ | $p=\tfrac12 \;\Leftrightarrow\; s/\tau = 0$ |
| **Self** (X → X) | `ToeplitzAttention` | $\displaystyle\sigma\!\left(\frac{S_{ij}}{\tau}\right)\cdot\sigma\!\left(\frac{A_{ij}}{\tau}\right)$ | $(0, 1)$ but $\le \tfrac14$ at the symmetric mid-point | $p=\tfrac12 \;\Leftrightarrow\; \sigma(S/\tau)\cdot\sigma(A/\tau) = \tfrac12$ |

where, from the Toeplitz decomposition of `scores = QK^\top / \sqrt{d_{qk}}`:

$$
S = \tfrac{1}{2}(\text{scores} + \text{scores}^{\!\top}), \qquad
A = \tfrac{1}{2}(\text{scores} - \text{scores}^{\!\top}).
$$

See `docs/TOEPLITZ_DECOMPOSITION.md` for the full derivation.

## 2. Why self needs better Q–K alignment than cross

To produce the **same** attention probability $p$:

- **Cross** needs $s_{ij}/\tau = \sigma^{-1}(p)$.
- **Self** needs *both* $S_{ij}/\tau$ *and* $A_{ij}/\tau$ such that
  $\sigma(S/\tau)\cdot\sigma(A/\tau) = p$.

Two illustrative working points (with $\tau = 3.0$, the iter_10 default):

| Target $p$ | Cross requires | Self requires (symmetric case $S/\tau = A/\tau = x$, i.e. $\sigma(x)^2 = p$) |
|---|---|---|
| 0.10 | $s/\tau \approx -2.20$ | $S/\tau, A/\tau \approx -0.77$ each |
| 0.25 | $s/\tau \approx -1.10$ | $S/\tau, A/\tau \approx \phantom{+}0$ each |
| **0.50** | $s/\tau = 0$ | $S/\tau, A/\tau \approx +0.88$ each |
| 0.75 | $s/\tau \approx +1.10$ | $S/\tau, A/\tau \approx +1.87$ each |

(Numerically: self needs $x = \mathrm{logit}(\sqrt{p})$ while cross needs
$x = \mathrm{logit}(p)$.)

The asymmetry is sharpest in the upper half: at $p=0.5$, **self** needs the
symmetric *and* antisymmetric components of the Toeplitz-decomposed
$QK^\top$ to each reach $\sim 0.88\,\tau$, while **cross** only needs the
single score to be at zero. Equivalently, **the self branch's "neutral"
operating point ($s = 0$) sits at $p = 0.25$, not at $p = 0.5$.** The same
"raw alignment" $QK^\top$ yields a strictly smaller attention weight in
the self branch.


This is **not** a $1/\sqrt{d_{qk}}$ effect — Vaswani's normaliser is already
applied to `scores` in both branches.  It is a **second** normalisation,
implicit in the choice of activation, and it acts at the *output* of the
attention scoring rather than at the dot-product variance.

## 3. iter_9: a learned amplifier on top of the structural gap

Until iter_9, `CausalCrossAttention` and `SigmoidCrossAttention` carried two
extra learnable parameters that self-attention did not have:

```python
# iter_9 cross-attention forward (excerpt)
gain   = exp(log_gain).clamp(1e-3, max_gain)   # init ~ 10, cap = 100
slope  = scores / exp(log_tau_act)             # learnable inverse-tau
out    = gain * activation(slope / tau)        # extra ×gain after the sigmoid
```

Self-attention (`ToeplitzAttention`) had no `log_gain`, no `max_gain`, and a
single `log_tau` instead of a separate `log_tau_act`. Effectively:

$$
\text{att}^{\text{cross}} \;\approx\; g \cdot \sigma\!\bigl(s/\tau\bigr) \quad\text{with } g\!\sim\!10,
\qquad
\text{att}^{\text{self}} \;=\; \sigma(S/\tau)\!\cdot\!\sigma(A/\tau).
$$

So in iter_9 the cross branch was **deterministically amplified by a
learned gain initialised at ~10×** *on top of* the structural double-sigmoid
penalty already paid by self. Together these explain the ~10× magnitude
gap observed in `score_tensor_for_sparsity` / `phi` between the two paths
in `iter_9/single_Toeplitz_Sigmoid_svfa_scm{1,2,3}c`.

## 4. iter_10: partial compensation

Implemented in `causaliT/core/modules/attention.py`:

1. **Dropped** `log_gain`, `max_gain`, and `log_tau_act` from
   `CausalCrossAttention` and `SigmoidCrossAttention`. The gain × inverse-tau
   collapsed to one redundant slope, so we kept only one constant.
2. **Dropped** the learnable `log_tau` Parameter from `ToeplitzAttention`.
3. Both branches now use a **single non-learnable** `self.tau = init_tau`
   (default `3.0`) — same value, same type, same constancy. Threaded
   through `SingleCausalLayer → attn_shared_kwargs → _attn → AttentionLayer →
   {ToeplitzAttention, CausalCrossAttention, SigmoidCrossAttention}`.
4. Removed `"log_tau_act"` and `"log_tau"` from the
   `tau_param_names = ("log_tau_gate", "log_tau_dir")` annealer tuples in
   `single_causal_forecaster.py` and `noise_aware_forecaster.py`. The new
   `tau` is a Python float and is therefore **never** touched by the unified
   tau annealer.
5. Added a `_register_load_state_dict_pre_hook` on each affected module that
   silently drops `log_gain`, `log_tau_act`, `log_tau`, `max_gain` keys at
   `state_dict` load time, so legacy iter_9 checkpoints still load.

Net effect on the magnitude gap:

|   | iter_9 | iter_10 |
|---|---|---|
| Learned cross-side amplifier | ≈ 10× | **gone** (by construction) |
| Structural double-sigmoid penalty (self) | yes | **still there** (by design) |
| `tau` consistency between branches | each had its own learnable form | **shared constant `init_tau=3.0`** |
| Total expected magnitude ratio cross / self at $p\!\sim\!0.25$ | ~ 10–15× | ~ 1–2× |

## 5. What we did *not* fix, and why

The structural factor of up to ~4× from the product-of-sigmoids in self is
a **design feature** of the Toeplitz decomposition, not a bug:

- It is what makes $P(i\!\to\!j) + P(j\!\to\!i) \le 1$ realisable
  ($\sigma(\gamma)\cdot\sigma(\phi) + \sigma(\gamma)\cdot\sigma(-\phi)
  = \sigma(\gamma) \le 1$, see Prop. P2(a) in `docs/PAPER_TEMPLATE.md`).
- It is what allows $P(i\!\to\!j) = P(j\!\to\!i) = 0$ (gate closed),
  the property the paper relies on to claim self-attention can represent the
  *absence* of an edge. SoftMax cannot. A single-sigmoid replacement
  ($\sigma(\text{combined}/\tau)$) would also lose this guarantee.

So we keep the asymmetry and instead **document its calibration consequence**
for downstream regularisers:

- **L1 score sparsity** (`lambda_self_score_sparse`,
  `lambda_cross_score_sparse`) sees a smaller mean magnitude on the self
  side. Symmetric $\lambda$'s therefore over-regularise self relative to
  cross.
- **HSIC** is computed on residuals, not on attention weights, so it is
  largely insensitive to this gap. Nevertheless the
  `staged_training` calibration (see `docs/STAGED_TRAINING.md`) already
  tracks `lambda_hsic_cross` and `lambda_hsic_self` *separately* via two
  multipliers — and the magnitude argument here is a strong reason
  *not* to collapse them into one.

## 6. Practical consequences

- **Tuning sparsity.** Expect to need
  `lambda_self_score_sparse < lambda_cross_score_sparse` (often by a factor
  of 2–4×) to obtain comparable per-branch edge density. This is the
  iter_10 prescription; it does not need to be re-tuned across SCM variants
  because the ratio comes from the activation algebra, not from the data.

  **iter_11 knob.** A single config key formalises this rule:

  ```yaml
  staged_training:
    lambda_self_to_cross_score_ratio: 0.5   # default 1.0 (legacy)
  ```

  - In the score-sparsity CV (`score_sparsity_cv._build_cv_config`) each
    candidate λ is interpreted as **λ_cross**, and λ_self is set to
    `ratio · λ_cross`.
  - In `configure_main_training_from_staged`, the same scaling is applied
    when the CV-selected `lambda_score_suggested` is propagated to
    `training.lambda_{cross,self}_score_sparse`.
  - With `ratio = 1.0` the behaviour is byte-identical to iter_10.
  - With `ratio = 0.5` the asymmetric L1 weights compensate for the
    structural factor of (≤) 2 between the two branches' average
    magnitudes at typical operating points (Section 2 above).

- **Reading attention plots.** A self-attention heat-map maxing out around
  0.25 at uniform $S/A$ is *not* under-trained — it is at the symmetric
  mid-point of the Toeplitz factorisation. A cross-attention heat-map at
  the same input would be at 0.5.
- **Comparing magnitudes between branches.** Plot
  $\sigma(S/\tau)$ alone (the Toeplitz "gate") instead of the full
  product when comparing structural strength to cross-attention's
  $\sigma(s/\tau)$. This recovers a 1-to-1 magnitude comparison.

## 7. References inside the repo

- `causaliT/core/modules/attention.py` — `CausalCrossAttention`,
  `SigmoidCrossAttention`, `ToeplitzAttention`, `_LEGACY_ATTENTION_KEYS`,
  `_drop_legacy_attention_keys`, `init_tau` plumbing in `AttentionLayer`.
- `causaliT/core/architectures/single_causal/model.py` — `init_tau`
  threaded through `attn_shared_kwargs` and `_attn`.
- `causaliT/training/forecasters/{single_causal,noise_aware}_forecaster.py`
  — `tau_param_names = ("log_tau_gate", "log_tau_dir")`, the unified tau
  annealer no longer touches the constant taus.
- `docs/TOEPLITZ_DECOMPOSITION.md` — derivation of $\sigma(S/\tau)\cdot
  \sigma(A/\tau)$ and the $P(i\!\to\!j)+P(j\!\to\!i)\le 1$ property.
- `docs/SVFA_ATTENTION.md` — the $W_q, W_k$ structure-only projections
  whose alignment is the quantity discussed in §2.
- `docs/STAGED_TRAINING.md` — calibration stage that tracks
  `lambda_hsic_cross` and `lambda_hsic_self` separately.
- `experiments/single/calitrain/iter_9/single_Toeplitz_Sigmoid_svfa_scm{1,2,3}c`
  — the experiments that surfaced the magnitude gap.
