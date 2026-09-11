# The centroid-commit subset score: vMF-MAP readout, margin semantics, and the dense-set bias

**Status:** derivation + post-mortem, 2026-09-01
**Evidence:** `experiments/6_INVESTIGATIONS/BILEVEL_GATE/results/gate_d20_12302718/`
(1191 bilevel probes, 3 accepts, committed subset sizes $|S| \approx 18.5$ of 19,
SHD increasing; see Sections 5 and 8)
**Arms produced:** `BILEVEL_GATE/gate_bkd_d20`, `BILEVEL_GATE/gate_bkd_d20_smoke`
(run-level BKD curriculum arm; Section 8)
**Code:** `causaliT/training/centroid_commit.py` (score, margin, shadow, taboos),
`causaliT/training/bilevel_probe.py` (paired refit gate),
`causaliT/training/forecasters/attention_selector_forecaster.py` (wiring, ~L700,
~L2530, ~L2760). Companion design doc: `docs/ideas/BILEVEL_CENTROID_COMMIT.md`.

---

## 1. Setting and notation

Each node $i$ of the $N$-node graph ($N = N_S + N_X$; sources first) owns a
**structural query** $q_i \in \mathbb{R}^d$. The key frame
$K = [k_1, \dots, k_N]^\top \in \mathbb{R}^{N \times d}$ is **frozen and
orthonormal** ($K K^\top = I$, verified to $\sim 10^{-7}$ in
`QUERY_FANIN_SCALE_BUDGET.md`). Under centroid commit, $q_i$ is not free: it is
quantized to the normalized centroid of a key subset $S_i \subseteq [N] \setminus \{i\}$,

$$
q_i \;=\; c(S_i), \qquad c(S) := \frac{\bar k_S}{\lVert \bar k_S \rVert},
\qquad \bar k_S := \tfrac{1}{|S|} \textstyle\sum_{j \in S} k_j ,
$$

with $c(\varnothing) = 0$ (the source hypothesis). Between commits $q_i$ is
frozen; evidence accumulates in a per-node **shadow** $s_i$ (Section 4). The
commit decision is driven entirely by the **subset score** defined next.

Define the **alignments** of a direction $q$ with the key frame:

$$
a_j(q) \;=\; \hat q \cdot k_j, \qquad \hat q := q / \lVert q \rVert .
$$

## 2. The subset score

### 2.1 Definition and the orthonormality identity

$$
\boxed{\;\mathrm{score}(S; q) \;=\; \frac{\sum_{j \in S} a_j(q)}{\sqrt{|S|}} \;-\; \rho\, |S|\;}
$$

(`subset_score`, `centroid_commit.py:55`), with $\mathrm{score}(\varnothing) = 0$
and $\rho$ = `prior_rho` (per-key prior penalty, default $0$).

Because the keys are orthonormal, $\lVert \bar k_S \rVert = 1/\sqrt{|S|}$, hence

$$
\cos\big(q,\, c(S)\big) \;=\; \frac{\sum_{j \in S} a_j(q)}{\sqrt{|S|}} :
$$

the first term *is* the cosine alignment with the subset centroid. The identity
fails for non-orthonormal keys — the whole construction leans on the frozen
frame.

### 2.2 Probabilistic reading: profile likelihood under a vMF model

Regard the observed direction $\hat q$ as a von Mises–Fisher draw around the
true centroid direction, $\hat q \sim \mathrm{vMF}(\kappa\, c(S))$ on
$\mathbb{S}^{d-1}$, with log-likelihood (up to constants)
$\ell(S, \kappa) = \kappa\, \cos(\hat q, c(S)) - \log Z_d(\kappa)$. Profiling
out the concentration $\kappa$ at fixed $S$ — the MLE
$\hat\kappa$ solves $A_d(\hat\kappa) = \cos(\hat q, c(S))$ and is monotone in
the cosine — makes the profile log-likelihood an increasing function of
$\cos(\hat q, c(S))$ alone. Comparing subsets by their profile likelihood is
therefore exactly comparing the first term of the score. The penalty
$\rho |S|$ is a prior $\log p(S) = -\rho |S| + \mathrm{const}$ (per-key
log-odds), turning the ML readout into a **MAP** readout. This is the entire
theoretical content of `prior_rho`: it is not a regularizer on the model — it
acts only on the controller's readout geometry (Section 5).

### 2.3 Exact argmax in $O(N \log N)$

A naive MAP search costs $2^N$. With orthonormal keys the optimal size-$m$
subset is always the **top-$m$ keys by alignment** (exchange argument: swapping
any member for a higher-alignment non-member increases the numerator at fixed
$m$), so

$$
S^*(q) \;=\; \arg\max_{m \in \{0, \dots, N\}} \Big[ \frac{\mathrm{csum}(m)}{\sqrt{m}} - \rho m \Big],
\qquad \mathrm{csum}(m) = \textstyle\sum_{j \le m} a_{(j)} ,
$$

with $a_{(1)} \ge a_{(2)} \ge \dots$ the sorted alignments, $m = 0$ the empty
subset (score 0), and the node's own key excluded ($a_i = -\infty$: no
self-loops). Implemented in `best_subset` (`centroid_commit.py:69`); the
taboo-aware variant `best_subset_excluding` (`:103`) skips forbidden subsets
while scanning $m$ (exactness is preserved: the optimal admissible size-$m$
subset is still the top-$m$ keys, and sizes are enumerated exhaustively).

## 3. The dense-set bias

The score's argmax has a structural bias toward large subsets under diffuse
evidence. Adding the next-best key (alignment $a$) to a size-$m$ subset
improves the ML score iff

$$
\frac{S_m + a}{\sqrt{m+1}} > \frac{S_m}{\sqrt{m}}
\quad\Longleftrightarrow\quad
a \;>\; \mathrm{score}_m \cdot \big(\sqrt{m+1} - \sqrt{m}\big)
\;\approx\; \frac{\mathrm{score}_m}{2\sqrt{m}} ,
$$

where $S_m = \mathrm{csum}(m)$ and $\mathrm{score}_m = S_m/\sqrt{m}$. The entry
bar **decays like $m^{-1/2}$**: at $m = 18$ a key needs only
$a > \mathrm{score}/8.5$ to be absorbed. Any mildly diffuse shadow — small
positive alignment with many keys, exactly what a first-order HSIC gradient
produces at a near-full centroid — is read out as a near-complete parent set.
With $\rho > 0$ the condition becomes

$$
a \;>\; \mathrm{score}_m \big(\sqrt{m+1} - \sqrt{m}\big) + \rho
\;\approx\; \frac{\mathrm{score}_m}{2\sqrt{m}} + \rho ,
$$

i.e. each key must clear a **fixed evidence bar** $\rho$ regardless of $m$.
This is the mechanism behind the `gate_d20_12302718` failure (Section 8):
`prior_rho = 0` + diluted first-order evidence $\Rightarrow$ committed subsets
of size 18–19 out of 19.

## 4. Shadow dynamics and candidacy

Between commits the forward query is frozen at $q_i = c(S_i)$; the shadow
integrates the (HSIC-sourced) structural gradient row $g_i$ with leak $\beta$
(`evidence_leak`) and rate $\eta_e$ (`evidence_lr`):

$$
s_i \;\leftarrow\; q_i \;+\; \beta\,(s_i - q_i) \;-\; \eta_e\, g_i
$$

(`step()`, `centroid_commit.py:327`). $s_i - q_i$ is the leaked accumulated
evidence; on commit the shadow re-centres ($s_i \leftarrow c(S')$) and $M_i$
(the per-node norm budget) is optionally reset (`reset_m_on_commit`).

**Candidacy.** Node $i$ becomes a candidate when the shadow's MAP subset
disagrees with the committed one:

$$
S^*(s_i) \;\neq\; S_i
\qquad\text{(taboo-aware: } S^* \text{ is the best non-taboo subset).}
$$

Candidacy is a **snapshot**, recomputed every step from the live shadow; a
pending candidacy is superseded if the shadow drifts elsewhere before being
probed.

## 5. The margin: definition and three uses

For candidate subset $S'$ against incumbent $S$, both scored on the **shadow's
own alignments** $a(s_i)$:

$$
\boxed{\;\Delta_{\mathrm{score}} \;=\; \mathrm{score}\big(S';\, a(s_i)\big) \;-\; \mathrm{score}\big(S;\, a(s_i)\big)\;}
$$

(`centroid_commit.py:356`). Units: cosine gain (minus prior terms). Uses:

1. **Hysteresis** — candidacy requires $\Delta_{\mathrm{score}} \ge$
   `commit_margin` (`:360`). At `commit_margin = 0` any positive gain
   triggers; this is the knob that would throttle jitter-driven candidacies
   (in `gate_d20_12302718` candidacies arose on ~every structural step).
2. **Winner-take-all priority** — with `winner_take_all: true`, at most one
   node is probed per step: the eligible node with the largest
   $\Delta_{\mathrm{score}}$ (`:373`). The margin is the resource-allocation
   key for the probe budget.
3. **Diagnostics** — logged per commit as `struct/commit_margin` and printed
   in the commit log line.

**The margin never reaches the gate.** The bilevel acceptance test uses a
different quantity with its own threshold (Section 6): the paired-refit HSIC
delta with `accept_margin`. The two "margins" are disconnected — different
quantities, different units, different stages:

| | `commit_margin` (Tier 1) | `accept_margin` (Tier 2) |
|---|---|---|
| quantity | $\Delta_{\mathrm{score}}$ on shadow alignments | $\Delta H_i$ after paired refit |
| units | cosine gain | HSIC row difference |
| stage | candidacy / probe priority | acceptance |
| `gate_d20_12302718` | 0.0 (everything proposes) | 0.0 (3/1191 accept) |

A useful open diagnostic: the margin of **rejected** candidacies is currently
not logged (only `struct/commit_margin` on commits), so we cannot yet test
whether Tier-1 evidence strength predicts Tier-2 acceptance.


## 6. The paired bilevel probe (acceptance test)

When node $i$ is PENDING with candidate $S'$, the gate runs a **paired refit**
(`bilevel_probe.py::paired_refit_probe`): from the *current* reconstruction
parameters $\theta_R$, refit the reconstruction arm for $k$ steps
(`k_inner`, Adam, `inner_lr`) on cached validation batches, once with the
incumbent query and once with $q_i \leftarrow c(S')$ written, then compare the
node-responsible HSIC rows:

$$
H_i^{\mathrm{cur}} = H_i\big(\mathrm{Refit}_k(\theta_R;\, q_i)\big), \qquad
H_i^{\mathrm{cand}} = H_i\big(\mathrm{Refit}_k(\theta_R;\, q_i \!\leftarrow\! c(S'))\big),
$$

$$
\text{accept} \;\iff\; H_i^{\mathrm{cand}} < H_i^{\mathrm{cur}} - \delta,
\qquad \delta = \texttt{accept\_margin}.
$$

Design invariants (all load-bearing):

* **Isolation** — both arms are deepcopies of the live forecaster (with
  graph-tensor sanitization; jobs 12281530/12294402); the live model is never
  mutated.
* **Pairing** — same $\theta_R$ start, same optimizer config, same torch seed
  per arm (identical stochastic-gate draws), same batch order.
* **Latched bandwidths** — the median-heuristic HSIC bandwidths are computed
  once from the pre-refit incumbent and reused for both arms and the final
  readout, so bandwidth jitter cannot enter the comparison.
* **Eval readout** — $H_i$ is measured in eval mode on a reserved validation
  batch (the last cached batch never feeds the refit).

**Reject** $\Rightarrow$ $S'$ is **tabooed** (soft: skipped inside
`best_subset_excluding`) and the shadow is *not* reset — evidence keeps
accumulating, so the node can diffuse out of the tabooed centroid; the taboo
lifts automatically once the shadow's raw projection is neither the committed
nor the tabooed subset (`centroid_commit.py:334–347`).

### 6.1 Why the probe runs with BKD off (design choice, principled)

BKD makes the *training-time* aggregation stochastic over key subsets
($A \leftarrow A \odot \mathrm{keep}$, one Bernoulli mask per step,
`gated_self_attention.py:626–636`). The gate, however, must certify a
**fixed** object: the ANM condition
$\varepsilon_i \perp \{s_j\}$ given the true parents is a property of the
deployed, mask-free model. Evaluating $H_i$ under a random mask would (a)
inflate the row whenever a true parent is dropped (the residual re-absorbs
dependence on it) and (b) make the acceptance decision itself a lottery over
masks. Hence `_prepare_copy` disables BKD on the probe copy
(`set_bkd_phase_active(False)`, `bilevel_probe.py:235–238`) and the pair mask
stashed by `_step` excludes the BKD factor — "a dropped-in-train key stays a
valid candidate". The refit also runs BKD-free, removing a variance source
from a paired estimator whose signal ($\Delta H_i \sim 10^{-3}$, Phase 0) sits
at the noise floor. This is a *deliberate asymmetry*: evidence is gathered
under the stochastic curriculum, certification is always against the full
dense adjacency.

### 6.2 Known biases of the probe

* **Incumbent co-adaptation.** $\theta_R$ is adapted to the incumbent queries;
  any write initially disrupts reconstruction, and $k = 30$ Adam steps may not
  fully re-adapt (Phase-0 caveat: "k=15/lr=1e-3 may underfit the new
  structure"). The probe therefore measures a *transient* cost, biasing
  $\Delta H_i$ positive — consistent with the systematically positive
  `gate_delta` ($\approx +7\mathrm{e}{-4}$) in `gate_d20_12302718`.
* **Constant-shift blindness.** The centered HSIC kernel is blind to
  prediction changes constant across samples (toy: $\Delta$pred $= -0.835$ on
  every sample moved $H_i$ by $\sim 10^{-9}$). A commit can improve MSE and be
  invisible to the gate; `struct/gate_dmse` is the tripwire.
* **Incumbent-mask approximation.** The pair mask (descendant $\times$
  LOO-$\gamma$) is the incumbent's; complementary keys stay downweighted —
  conservative, never inflating. (OFF in the `gate_*` arms, where the probe
  objective is exactly the training objective.)



## 7. Full per-step algorithm

Per-node state machine: `ACCUMULATING → PENDING → PROBING → COMMITTED |
REJECTED → ACCUMULATING`. All shadows update every step; only probes
serialize (winner-take-all).

$$
\begin{array}{l}
\textbf{repeat per structural training step}\\
\quad 1.\ \text{forward/backward; } \mathcal{L}_{\mathrm{rec}},\ \mathcal{L}_{\mathrm{out}} \text{ as usual}\\
\quad 2.\ \textbf{for all } i:\ s_i \leftarrow q_i + \beta(s_i - q_i) - \eta_e\, g_i \quad \text{(shadow update)}\\
\quad 3.\ \textbf{for all } i:\ \text{maintain taboos (lift if } S^*_{\mathrm{raw}}(s_i) \notin \{S_i\} \cup \mathcal{T}_i\text{)}\\
\quad 4.\ \mathcal{E} \leftarrow \{\, i : S^*(s_i) \neq S_i,\ \Delta_{\mathrm{score}}(i) \ge \texttt{commit\_margin},\ \mathrm{SNR}_i \ge \texttt{min\_snr} \,\}\\
\quad 5.\ \textbf{if } \texttt{winner\_take\_all}:\ \mathcal{E} \leftarrow \{\arg\max_{i \in \mathcal{E}} \Delta_{\mathrm{score}}(i)\}\\
\quad 6.\ \textbf{for } i \in \mathcal{E} \textbf{ (defer mode)}:\ \text{paired probe} \Rightarrow \Delta H_i\\
\quad 7.\ \quad \text{accept } (\Delta H_i < -\delta):\ q_i \leftarrow c(S'),\ s_i \leftarrow c(S'),\ M_i \text{ reset}\\
\quad 8.\ \quad \text{reject}:\ \mathcal{T}_i \leftarrow \mathcal{T}_i \cup \{S'\},\ s_i \text{ untouched}\\
\quad 9.\ \text{optimizer steps } (\theta_R \leftarrow \mathcal{L}_{\mathrm{rec}},\ \theta_S \leftarrow \text{structural terms})
\end{array}
$$

## 8. Post-mortem exhibit: `gate_d20_12302718` and the BKD-curriculum arm

The run (interrupted at epoch 1395/10000, i.e. ~6% of the terminal structure
phase) shows the dense-bias failure mode end to end:

* candidacies on ~every structural step ($\approx 2$/epoch at ~3 structural
  batches/epoch) — the bottleneck is **acceptance**, not proposal;
* **1191 probes, 3 accepted, 1188 rejected**; `gate_delta` mean $+8.8\mathrm{e}{-4}$
  (systematically positive: candidates genuinely worse under the probe, plus
  the incumbent-advantage bias of Section 6.2);
* committed subset size **18.95 → 17.85 of max 19** — the ML readout
  ($\rho = 0$) saturated near the full key set, exactly as Section 3 predicts;
  the 3 accepted (noise-level, $|\Delta| \approx 3\mathrm{e}{-4}$) commits moved
  *toward dense* → SHD increased;
* taboo churn: a rejected subset is re-proposed as $\pm 1$-key variants
  (shadow not reset, `best_subset_excluding` returns the next-best), burning
  the 1-probe/step budget.

**Causal chain:** first-order shadow (Phase 0: anti-correlated at frozen
$\theta_R$) + BKD 0.05 (decoder co-adapts to dense queries; per-row HSIC
evidence diluted over all 19 keys) $\Rightarrow$ dense/wrong candidacies
$\Rightarrow$ the gate (correctly, plus bias) rejects ~everything
$\Rightarrow$ rare accepts are noise $\Rightarrow$ SHD worsens.

**The `gate_bkd_d20` arm** changes exactly one factor — the BKD curriculum —
to attack the evidence dilution: warmup cosine spanning (nearly) $[0, 1]$
(`bkd_p_base 0.0`, `bkd_amplitude 1.0`, `bkd_cycles 2`, landing on $0.6$),
reconstruct constant $0.6$, structure **linear decay $0.6 \to 0.0$** over the
phase (global anneal clock compensated: `p0 = 0.667`,
`annealing_batches = 30000`; see the config header). The gate is unchanged:
probes always certify against the full dense adjacency (Section 6.1).
Success criteria vs `gate_d20_12302718`: lower committed $|S_i|$, a
sign-balanced `gate_delta` distribution, more accepted commits, and
non-increasing SHD (`evaluate_updates.ipynb` readouts).

## 9. Symbol → code index

| symbol | meaning | config key | code |
|---|---|---|---|
| $a_j(q)$ | key alignment $\hat q \cdot k_j$ | — | `best_subset`, `centroid_commit.py:87` |
| $\mathrm{score}(S;q)$ | subset score | `prior_rho` ($\rho$) | `subset_score`, `:55` |
| $S^*(q)$ | MAP subset | — | `best_subset` / `best_subset_excluding`, `:69/:103` |
| $s_i$ | evidence shadow | `evidence_lr` ($\eta_e$), `evidence_leak` ($\beta$) | `step()`, `:327` |
| $\Delta_{\mathrm{score}}$ | commit margin | `commit_margin` | `:356`; WTA `:373` |
| $c(S)$ | normalized centroid | — | `centroid_of`, `:145` |
| $\mathcal{T}_i$ | taboo sets | — | `:334–347`, `finalize` `:388` |
| $k$, $\delta$ | refit steps, accept threshold | `bilevel_gate.k_inner`, `accept_margin` | `paired_refit_probe` |
| $H_i$ | node HSIC row | — | `hsic_rows_eval`, `bilevel_probe.py:129` |
| BKD $p(t)$ | key-dropout rate | `adaptive_training.*.batch_key_dropout*` | `_current_bkd_p`, `gated_self_attention.py:256` |

## 10. References

* Liu, Simonyan, Yang — *DARTS: Differentiable Architecture Search*, ICLR 2019
  (bilevel formulation; the commit gate is its discrete, event-triggered
  analogue).
* Finn, Abbeel, Levine — *Model-Agnostic Meta-Learning*, ICML 2017; Franceschi
  et al., *Bilevel Programming for Hyperparameter Optimization*, ICML 2018
  (unrolled bilevel gradients; `shadow_source: hsic_unrolled`, Phase 3).
* Gretton et al. — *Measuring Statistical Dependence with Hilbert–Schmidt
  Norms*, ALT 2005; Gretton et al., *A Kernel Statistical Test of
  Independence*, NeurIPS 2007 (HSIC).
* Mardia & Jupp — *Directional Statistics* (vMF MLE); Banerjee et al.,
  *Clustering on the Unit Hypersphere using von Mises–Fisher Distributions*,
  JMLR 2005 (centroid-cosine identities behind Section 2).
* Louizos, Welling, Kingma — *Learning Sparse Neural Networks through $L_0$
  Regularization*, ICLR 2018 (Hard-Concrete gates in the attention posterior).
* Zheng et al. — *DAGs with NO TEARS*, NeurIPS 2018 (acyclicity diagnostics).
