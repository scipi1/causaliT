### Bilevel-Gated Centroid Commits (discrete DARTS)

> The subset score, margin semantics, dense-set bias, and probe design
> invariants are derived in detail in
> `docs/experimental_elaborations/CENTROID_COMMIT_SCORE.md` (includes the
> `gate_d20_12302718` post-mortem and the `gate_bkd_d20` BKD-curriculum arm).


#### Motivation

Structural queries are optimized by a HSIC-independence gradient evaluated at **frozen reconstruction parameters** $\theta_R$. But the informative landscape is HSIC on the **response manifold** $\theta_R^*(\theta_S)$: the oracle experiment shows the true adjacency minimizes HSIC only *after* the reconstruction is refit. At 20 nodes the effect is further diluted by the global mean over $\sim N^2$ (source, residual) pairs: one node's correct move touches $\sim 1/N$ of the averaged terms.

#### Bilevel formulation (DARTS mapping)

Our model is a DARTS with a single operation (identity): the per-node structural query $q_i$ selects how many identity gates (keys) are open. Parameters split as

- $\theta_S$ — structural: query/key projections, structure embeddings, free query embeddings;
- $\theta_R$ — reconstruction: value/out projections, FFN, forecaster head.

$$
\min_{\theta_S}\; \mathcal{L}_{\text{out}}\big(\theta_S, \theta_R^*(\theta_S)\big)
\quad \text{s.t.} \quad
\theta_R^*(\theta_S) = \arg\min_{\theta_R}\; \mathcal{L}_{\text{rec}}(\theta_S, \theta_R),
$$

with $\mathcal{L}_{\text{rec}}$ the MSE and $\mathcal{L}_{\text{out}}$ the HSIC objective. The current gradient-routing scheme is the **first-order** approximation (frozen $\theta_R$); this proposal evaluates and exploits the **unrolled** objective.

#### Per-node responsibility

Node $i$'s query affects only `residual_i` (row-wise attention). The correct acceptance criterion is the node's own HSIC row:

$$
H_i(\theta_S, \theta_R) \;=\; \frac{1}{|\mathcal{J}_i|} \sum_{j \in \mathcal{J}_i} \operatorname{HSIC}(s_j,\; \varepsilon_i),
$$

where $\mathcal{J}_i$ is the (masked) source set. Under the true parents $\varepsilon_i = \epsilon_i \perp$ all sources (ANM condition). The global mean $\frac{1}{N}\sum_i H_i$ is retained only as a diagnostic.

#### Tier 1 — second-order shadow evidence (every step, all nodes in parallel)

The query shadow integrates the **destination-state gradient** (DARTS second-order, finite-difference Hessian). With virtual refit $\theta_R' = \theta_R - \eta \nabla_{\theta_R}\mathcal{L}_{\text{rec}}$ and $v = \nabla_{\theta_R'} \mathcal{L}_{\text{out}}$:

$$
g_i \;=\; \nabla_{q_i}\mathcal{L}_{\text{out}}(\theta_S, \theta_R')
\;-\; \eta\, \underbrace{\frac{\nabla_{q_i}\mathcal{L}_{\text{rec}}(\theta_R + \epsilon v) - \nabla_{q_i}\mathcal{L}_{\text{rec}}(\theta_R - \epsilon v)}{2\epsilon}}_{\nabla^2_{q_i,\theta_R}\mathcal{L}_{\text{rec}}\, v}
$$

One batched special forward yields $g$ for **all** rows at once (3–4 extra backwards total, not per node). HSIC bandwidths are frozen across all perturbed passes. Evidence accumulates as before (leak $\beta$, rate $\eta_e$):

$$
\text{shadow}_i \leftarrow q_i^{\text{committed}} + \beta\,(\text{shadow}_i - q_i^{\text{committed}}) - \eta_e\, g_i
$$

#### Tier 2 — paired bilevel commit gate (event-triggered)

When node $i$'s shadow crosses into a new centroid $c(S')$, do **not** commit. Probe on **validation** data with a paired refit from the same $\theta_R$:

$$
H_i^{\text{cur}} = H_i\big(\text{Refit}_k(\theta_R;\, q_i)\big), \qquad
H_i^{\text{cand}} = H_i\big(\text{Refit}_k(\theta_R;\, q_i \leftarrow c(S'))\big)
$$

**Accept** iff $H_i^{\text{cand}} < H_i^{\text{cur}} - \delta$; **reject** $\Rightarrow$ shadow not reset, $S'$ tabooed (soft, inside `best_subset`); the taboo lifts automatically when the shadow's projection moves elsewhere (re-entry stays possible).

#### Orchestration

Per-node state machine under one controller: `ACCUMULATING → PENDING → PROBING → COMMITTED | REJECTED → ACCUMULATING`. Shadows of all nodes (including PENDING ones) update every step; only probes serialize. Candidacy is a **snapshot** (superseded if the shadow drifts to another centroid before probing).

Simultaneous pendings: default `wta` (one probe/step, margin-ordered — already implemented). Optional `joint`: single refit with all pending queries, per-row acceptance readout, mixed verdicts re-probed alone before commit.

#### Algorithm

$$
\begin{array}{l}
\textbf{repeat per training step}\\
\quad 1.\ \text{forward};\ \mathcal{L}_{\text{rec}},\ \mathcal{L}_{\text{out}} \text{ as usual}\\
\quad 2.\ \text{Tier-1: compute } g \text{ (unrolled, batched); update all shadows}\\
\quad 3.\ \text{mark nodes whose shadow's best non-taboo subset} \neq \text{committed as PENDING}\\
\quad 4.\ \text{pick pending set } \mathcal{P} \text{ per policy (wta | joint)}\\
\quad 5.\ \textbf{for } i \in \mathcal{P}: \text{paired probe} \Rightarrow H_i^{\text{cur}}, H_i^{\text{cand}}\\
\quad 6.\ \quad \text{accept: } q_i \leftarrow c(S'), \text{ shadow re-centred, } M_i \text{ reset}\\
\quad 7.\ \quad \text{reject: taboo } S', \text{ shadow untouched}\\
\quad 8.\ \text{optimizer steps (}\theta_R \leftarrow \mathcal{L}_{\text{rec}},\ \theta_S \leftarrow \text{structural terms)}\\
\end{array}
$$

#### Config sketch

```yaml
training:
  centroid_commit:
    enabled: true
    shadow_source: hsic_unrolled   # new; hsic | structural | hsic_unrolled
    unrolled:
      inner_lr: null               # null = recon lr
      fd_epsilon: 0.01
      every: 1                     # cadence (m = every m steps)
    bilevel_gate:
      enabled: true
      k_inner: 20                  # refit steps per probe
      inner_lr: null
      max_val_batches: 8
      accept_margin: 0.0           # delta
      simultaneous: wta            # wta | joint
```

#### Implementation phases (each default-off, independently tested)

0. **Diagnostics**: per-row HSIC logging; per-row teleportation test (frozen vs $k$-refit). *Go/no-go.*
1. `causaliT/training/bilevel_probe.py`: paired refit on **deepcopy**, fresh inner optimizer, eval-mode, latched bandwidths. Tests: no live-state mutation (param hash), determinism, identity-candidate invariance.
2. Gated `CentroidCommitController` (state machine, taboos, snapshot candidacy, controller `state_dict`). Tests: rejection preserves shadow; taboo blocks/lifts; brute-force check of tabooed `best_subset`.
3. `hsic_unrolled` shadow source (FD Hessian, ordered before the dual `manual_backward`). Tests: FD vs `create_graph=True` autodiff on a linear SCM; cosine-to-oracle improvement on the d20 fixture.
4. Experiment ladder on d20: baseline → +rows → +gate → +unrolled shadow.

**Known bug traps**: live-graph ordering in `training_step`; probe state leakage; HSIC bandwidth jitter in paired comparisons; taboo-subset logic in the top-$m$ argmax; silent no-ops (every gate decision logged: `struct/gate_{accepted,rejected,tabooed}`).

#### Caveat discovered in Phase 1: constant-shift blindness

The HSIC centered kernel is exactly blind to prediction changes that are constant across samples. In a toy with independent random values, a query-row write changed a node's prediction by $-0.835$ on *every* sample ($\text{std} \approx 10^{-7}$) and the HSIC row moved by $\sim 10^{-9}$. Implication: a commit can improve MSE and still be invisible to the gate ($\Delta H_i \approx 0$). Real (dependent) data is not in this degenerate regime — the Phase-0 teleportation test showed clear per-row signal — but near-collinear value embeddings could partially hide moves. Mitigation if observed: add $\Delta\text{MSE}_i$ as a gate diagnostic (or secondary criterion).

#### LOO / descendant-mask compatibility

The pair weights in `_step` are `descendant_mask × BKD_keep × loo_gamma` (all detached). For both bilevel tiers, the mask is treated as a property of the **current committed structure**, fixed within the step: `_step` stashes `_last_probe_pair_mask = descendant × loo_gamma` (**without** the BKD factor — probes and virtual passes run with BKD off, so a dropped-in-train key stays a valid candidate), and (a) the probe's `H_i` rows and (b) the unrolled shadow's virtual HSIC both use it. Consequences:

- Objective consistency: proposal, acceptance, and training measure the same masked HSIC.
- LOO/descendant state is never recomputed inside probes or virtual passes (no `_maybe_update_loo_gamma` calls there); the detached incumbent mask is reused for BOTH arms, keeping the comparison paired.
- Known approximation: the candidate is scored under the **incumbent's** γ/descendant sets. This is conservative — a key that is conditionally redundant under the incumbent but load-bearing in the candidate (complementarity) stays downweighted, so the gate may under-accept synergistic moves, but it never inflates a bad one. The `gate_d20` run has both LOO and descendant exclusion OFF, so there the probe objective is exactly the training objective. A per-arm recomputed-γ probe variant to measure this discrepancy is a Phase-4 diagnostic option.

#### Status

- **Phase 0 — done** (2026-08-31): `hsic_row_means` + `return_matrix` in `hsic_utils.py`; `training.log_hsic_rows` per-row logging in the forecaster; per-row teleportation diagnostic. Premise confirmed (results below).
- **Phase 1 — done** (2026-08-31): `causaliT/training/bilevel_probe.py` (paired refit, deepcopy isolation with graph-tensor sanitization, inherited inner optimizer, latched bandwidths, seeded arms). 10 tests in `tests/test_bilevel_probe.py`.
- **Phase 2 — done** (2026-08-31): `best_subset_excluding` (exact taboo-aware argmax), `CentroidCommitController.step(defer=True)` / `finalize(accept|reject)` with soft taboos that auto-lift, controller `state_dict` persisted via `on_save/load_checkpoint`, forecaster wiring (`_run_bilevel_gate`, val-batch cache in `validation_step`, gate metrics `struct/gate_{accepted,rejected,delta,taboos,deferred}`), config block `training.centroid_commit.bilevel_gate` (default off). 13 tests in `tests/test_centroid_commit_bilevel.py`. Note: the controller requires a query table per node — i.e. `homogeneous_nodes: true` (pre-existing constraint, unchanged by this work).
- **Phase 3 — done** (2026-08-31): `shadow_source: hsic_unrolled` in the forecaster — DARTS second-order destination-state shadow gradient via `torch.func.functional_call` parameter substitution (NO in-place swaps: they bump param version counters and invalidate the retained main graph — caught by tests), BKD off in lean passes, all lean passes RNG-paired via `fork_rng` (unpaired gate draws swamp the FD numerator: cos 1.0 paired vs −0.35 unpaired), frozen-bandwidth reuse, cadence `unrolled.every`. Config block `centroid_commit.unrolled`. 5 tests in `tests/test_unrolled_shadow.py` (FD vs exact per-element Jacobian reference — note the STE makes the *symmetric* mixed partial ~0, the correct reference is the directional derivative of the surrogate gradient field; no-mutation; determinism after a one-time ~1e-9 warm-up call; cadence; end-to-end `training_step` smoke with dual optimizers).
- **Configs**: `experiments/6_INVESTIGATIONS/BILEVEL_GATE/gate_d20` (gate, first-order shadow) and `gate_unrolled_d20` (gate + unrolled shadow); identical otherwise. `gate_d20_smoke` is a 120-epoch crash-hunting variant (warmup 20) for cluster smoke tests.
- **Bugfix (job 12281530)**: the first gated commit after warmup crashed in `_prepare_copy` — `deepcopy` rejects non-leaf tensors, and `_last_loss_components` (a dict of graph-carrying tensors set by `_step`) escaped the module-level sanitizer. The sanitizer now recurses into attrs/dicts/lists/tuples across the forecaster and all modules (restore in `finally`). Regression tests in `tests/test_bilevel_probe.py::TestProbeAfterTrainingStep` replicate the exact crash condition.
- **Bugfix (job 12294402, GPU-only)**: `nn.Module._apply` (`.cuda()`) replaces buffers with op results — non-leaf with `requires_grad` preserved — so the commit **shadow buffer became non-leaf on GPU**, breaking deepcopy (it lives in `_buffers`, previously skipped by the sanitizer) and silently starving `shadow.grad` (the warnings in both cluster logs; harmless only because the `hsic` source uses the `cc_grads` override). Fixed at the root with a `FreeQueryEmbedding._apply` override that re-leafs the shadow after any move, plus `_buffers` coverage in the probe sanitizer. Also fixed downstream: the probe copy inherits the phase's `requires_grad` masks (θ_R frozen in structure phases), which emptied the inner optimizer — the copy is now unfrozen (only the recon group is optimized). Verified end-to-end by running the real adaptive trainer locally (micro smoke): gate fired, probes accepted/rejected, taboos accumulated, run completed.

#### Phase 0 results (2026-08-31, `HSIC_OPT_2/diagnostics/teleport_per_row.py`)

Checkpoint `epoch=159` of `bkd_warmup_06_global_nonorm_hsicbkd_d20_loogamma_12082167`, 6 endogenous nodes × 3 repeats, $k=15$ refit steps on a deepcopy, paired incumbent baseline, per-row HSIC:

| signal | true-parent teleport | wrong-parent teleport |
|---|---|---|
| $\Delta$ frozen (current training signal) | **−0.0145** | **−0.0383** (wrong decreases *more*!) |
| $\Delta$ refit − incumbent | **−0.0012** (78% negative) | **+0.0020** (39% negative) |

Two conclusions: (1) the frozen-$\theta_R$ per-row signal is not just noisy but **anti-correlated** — it prefers wrong parent sets; the bilevel premise is confirmed at row level. (2) After a short refit the ordering flips correctly: true-parent teleports reduce the node's HSIC row while wrong ones increase it. Caveats: X12 is a counterexample (wrong refit decreases more), X1 true-teleport stays positive vs incumbent — k=15/lr=1e-3 may underfit the new structure; probe hyperparameters need care in Phase 1.
