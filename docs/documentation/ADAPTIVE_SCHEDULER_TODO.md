# Adaptive scheduler: open items

Status after the "declarative schedule + joint phase" work on
`causaliT/training/adaptive_trainer.py`.

## Done

- Declarative `adaptive_training.schedule: [warmup, joint, alternate,
  final_reconstruct]` (legacy keys still translated by `resolve_schedule`).
- `joint` phase: both parameter groups train in one step; the structural
  gradient uses the pre-step regressor.  With `data_split_ratio` the MSE uses
  the reconstruct split and the HSIC residual a batch from the structure split
  (`dm.stage_split_dataloader` + `forecaster.set_joint_structure_loader`).
  `hsic_unrolled` is rejected.  `lambda_l0` defaults to 0 in joint.
- Forecaster fix: reconstruction params never keep a structural gradient
  (relevant now that both groups can be trainable at once).
- Single config resolver `_phase_cfg`; warmup now inherits the reconstruct
  block for dir_bias / descendant / LOO / bandwidth knobs (BKD and dense gate
  keep the warmup block alone).
- Loss weights are applied in EVERY phase on top of the training-level values
  (structure lambdas no longer leak into reconstruct phases).
- Resume no longer repeats one-off entry actions (edge-offset step, optimizer
  reset, order init, drift baseline, null estimate).
- `on_train_end` records the last phase (`reason=global_budget`).
- Order initialization accepts the first structure-training entry after warmup
  (structure or joint).
- Sign-safe HSIC plateau check (`_improved`), return type, ASCII docstrings.

## To do

1. **Joint for other forecasters.**  NO
2. **Joint + HSIC fold B.**  With `data_split_ratio` the joint side loader
   replaces fold B (`training.hsic_cross_fit`).  If both are wanted, decide
   whether fold B should be taken INSIDE the structure split.
3. **Joint + swap_splits.**  Joint is never swapped (it runs before cycle 1).
   Swapping per N joint epochs could be added via `_resolve_split_key`.
4. **Joint-phase cost.**  Each joint step runs two forwards (recon batch +
   structure batch).  Consider a lower structural LR (two-timescale) and
   measure wall-clock vs. the sequential warmup + structure phases.
5. **DARTS stage.**  Make `training.structural_grad` a per-phase key (e.g.
   `structure.structural_grad: hsic_unrolled`) so the last stage can use the
   unrolled update while joint stays first-order.  Needs the forecaster to
   accept a runtime switch (the probe-mask stash in `_step` depends on it).
6. **Order-stability exit for joint.**  Add a trigger on the stability of the
   learned order (e.g. Kendall tau between consecutive validation epochs of
   the Sinkhorn mean permutation) as an alternative to the HSIC plateau.
7. **Drift probe cost.**  `_measure_structure_monitor` runs a full pass over
   fold B at every structure validation epoch; add `drift_probe_every_n`.
8. **Ramps at validation time only.**  gain-lambda / order-tau / BKD logging
   update in `on_validation_epoch_end` after the monitor check; move them to
   `on_train_epoch_start` so `check_val_every_n_epoch > 1` does not step them.
9. **Optimizer reset granularity.**  `reset_optimizer_state_on_switch` clears
   both optimizers; clear only the one of the group being unfrozen.
10. **Cycle definition with joint.**  Joint does not increment
    `_cycle_count` (BKD ladders / swap parity / L0-kappa gates see it as
    rung 0).  Revisit if the ladder should advance during joint.
11. **Pre-existing test failure.**
    `tests/test_atsel_gain_softmax.py::TestGainPhaseController::
    test_ramp_interpolates_and_saturates` fails because its fake module has no
    `modules()` (needed by `_apply_open_gate_cfg`, older than this change).
    Add `modules()` to the fake.
12. **End-to-end check.**  Run one short real `adaptive_trainer` job with
    `schedule: [warmup, joint, alternate]` and `data_split_ratio: 0.5`, and
    the routing/gradient test subset (`pytest tests -k "routing or gradient"`,
    128 tests).
