# ORIENTED_RECON_SAMPLING

Status: agreed design, not implemented.  Test arm planned against
`experiments/6_INVESTIGATIONS/ORDER_FIRST/adaptive_base`.

## 1. Motivation

- The structural loss (`_anti_adjacency_hsic`, attention_selector_forecaster.py)
  calls ONE FiLM per-node regressor twice per step:
  - causal pass: context `G * M_hard` (predecessors only) -> residuals in C;
  - anti-causal pass: context `G * M_hard^T` (successors only) -> residuals in A.
  With `order_straight_through` and hard gates, both contexts are 0/1 rows
  containing ONE side only.
- Sources get their signal only from A (their causal pass = empty set, not
  used in C).  Source recall must not regress.
- Reconstruction (`use_order_in_recon: false` + uniform BKD 0.5) trains FiLM
  on uniform random subsets, which rarely match the one-sided prefix /
  suffix subsets queried by the structural stage.  Capacity is spent on
  subsets the order will never query.

## 2. Design (agreed)

Train the regressor on the SAME subset distribution the structural stage
queries.  No orientation flag: FiLM already receives the applied row
(`last_applied_A`), so the subset is fully known to it.

Per reconstruction training step (warmup / reconstruct / final_reconstruct):
1. Draw the order noise as today (`resample`, one draw per step).
2. Build the hard order from `theta + recon_noise_scale * gumbel`
   (dedicated scale, constant 1.0, independent of the structural anneal).
3. Draw ONE orientation for the whole step: anticausal with p = 0.5,
   else causal (existing `set_order_orientation`).
4. Applied weight = `G * M_hard` (causal) or `G * M_hard^T` (anticausal),
   i.e. the existing `order_path="hard"` branch.  G is the active selector
   (G_R today; shared / G_S detached later), so the selector is accounted
   for automatically.
5. BKD: off in the test arm; kept compatible as an extra drop applied
   after the order mask (existing BKDMixin, unchanged).

Properties:
- Flat order init + noise 1.0 -> uniform random permutation: every node is
  in the subset with marginal p = 0.5 (as BKD 0.5), but subset sizes are
  uniform on 0..N-1 and include the empty / full sets queried by the
  structural stage (sources, sinks).
- No leakage: the recon order mask is detached; MSE trains only recon
  params and G_R.
- The 50/50 orientation mix trains both regressors in one shared network
  (the "OR" condition, not the "AND" of mixed contexts).
- Sources: the anti-causal half trains exactly the regressor used by A.

## 3. Code changes

### 3.1 OrderFactorizedSelfAttention (core/modules/order_factorized_attention.py)
- New attribute `recon_noise_scale` (float, default None = use
  `noise_scale`, backward compatible).
- In `forward`: if `self.training and self.order_path == "hard" and
  self.recon_noise_scale is not None`, compute a recon `M_hard` from
  `theta + recon_noise_scale * noise["gumbel"]`; otherwise keep today's
  `perturbed`.  Relaxed path and `noise_scale` untouched (the structural
  anneal, `_update_order_temperature` in adaptive_trainer.py, keeps
  writing `noise_scale` only; today it is global and phase-agnostic).
- New setter `set_recon_noise_scale(value)`.
- No dedicated Sinkhorn: on the hard path tau does not change the argmax,
  so the Gumbel scale is the only knob.
- The posterior / exclusion graph (`score_tensor_for_sparsity`) keeps the
  structural `M_hard`; the recon `M_hard` is used ONLY in
  `Z = G * M_app_hard`, so diagnostics are unchanged.

### 3.2 Model wrapper (core/architectures/attention_selector/model.py)
- Pass-through `set_recon_noise_scale` next to `set_order_temperature`
  (~line 3086), returning bool like the others.

### 3.3 Adaptive trainer (training/adaptive_trainer.py)
- New config block (global default, overridable per warmup / reconstruct /
  final block, parsed like `recon_gates`):

      adaptive_training:
        recon_order_sampling:
          enabled: false        # default: current behaviour
          p_anticausal: 0.5     # per-step orientation probability
          noise_scale: 1.0      # constant recon Gumbel scale

- Validation: error if `enabled` and `use_order_in_recon` are both true
  (distinct methods).
- `_apply_recon_forward_cfg`: for non-structure phases with `enabled`,
  call `set_order_path("hard")` and `set_recon_noise_scale(noise_scale)`.
  On structure, orientation stays causal (the structural code already
  resets it after the anti-causal pass); recon noise is unused.
- Log `recon_order_sampling` (0/1) and the per-epoch fraction of
  anticausal steps.

### 3.4 Forecaster (training/forecasters/attention_selector_forecaster.py)
- At the start of each reconstruction training step (where `resample` is
  called; location to confirm), if sampling is enabled: draw the
  orientation from a dedicated torch.Generator (seeded from
  `training.seed`) and call `model.set_order_orientation(...)`; reset to
  "causal" after the forward (try/finally, as in `_order_residuals`).
- Validation / eval forwards: causal orientation (deterministic).
  Optional later: val MSE for both orientations.
- Phase awareness: setter `set_recon_order_sampling(enabled, p)` called by
  the trainer in `_apply_recon_forward_cfg`.

### 3.5 Unchanged
`_anti_adjacency_hsic`, FiLM / PerNodeMLPHead, `_adjacency_context`, the
structural anneal, BKDMixin.

## 4. Test arm

`experiments/6_INVESTIGATIONS/ORDER_FIRST/adaptive_orient_recon/config.yaml`
= copy of `adaptive_base` with:
- `warmup.batch_key_dropout: null` (BKD off);
- `adaptive_training.recon_order_sampling: {enabled: true,
  p_anticausal: 0.5, noise_scale: 1.0}`;
- `use_order_in_recon: false` (unchanged; the new key selects the method).
Everything else identical (seed 1, data_seed 1, budget 5000).
Add the arm row to the ORDER_FIRST README.

## 5. Evaluation (vs adaptive_base)

- source recall / source R2 (must not regress);
- Kendall tau of the learned order;
- train C, A and HSIC vs permutation null (`hsic/null_mean`);
- order-conditioned drift-probe MSE (expected to drop);
- G_R density (now trained on one-sided contexts only);
- order oscillation between cycles.

## 6. Unit tests (tests/)

- `recon_noise_scale=None` reproduces current outputs exactly.
- Sampling on: anticausal applied row equals the transposed-order mask;
  the other side has zero weight.
- Structural `M_hard` / exclusion graph unaffected by `recon_noise_scale`.
- Orientation frequency approx p_anticausal over many seeded steps.
- Config error when `use_order_in_recon` and the new key are both on.

## 7. Future extensions (not in the first arm)

- Position-dependent orientation p (e.g. 0.9 sources, 0.5 middle, 0.1
  sinks, clamped to [0.1, 0.9]); needs per-row orientation or per-node
  loss weighting, since orientation is global per step.
- Re-enable BKD as extra regularization.
- Recon noise schedule tied to the spread of theta.
- Masked softmax / entmax selector; G_S + L0 on top.

## 8. Risks

- Self-reinforcement: the regressor fits the current order best.
  Mitigated by constant recon noise 1.0 (never annealed).
- Effective order sharpness grows with the spread of theta at constant
  noise.
- One orientation per step: diversity only across steps (accepted).

