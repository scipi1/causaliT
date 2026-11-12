# Euler workflow in causaliT: wiring status

Copied from `Projects/euler_workflow` (reference: its README.md and
`scripts/euler_instructions.txt`). Core and built-in plugins are unchanged;
the worker module resolves to `causaliT.euler_funs.worker` automatically.

## Done
- `causaliT/euler_funs/` (core, plugins training/sweep/dagsweep, lightning_utils)
- Optuna removed from euler_funs (plugin, launcher options). No HPO here.
- `train_fns.py`: `train` (k-fold `trainer`) and `adaptive_train`
  (`adaptive_trainer`); resume from `<save_dir>/k_<fold>/resume/last.ckpt`,
  pause at epoch end, raise TaskPaused.
- `trainer()` got `folds`, `fold_hooks`, `after_fold`; `adaptive_trainer()` got
  `extra_callbacks`, `resume_ckpt`. Defaults keep the old behaviour.
- `PhaseController.state_dict/load_state_dict`: phase/cycle state is stored in
  the checkpoint and re-applied in `on_train_start` on resume. The dropout
  pre-flight choice is saved to `dropout_selection_choice.json` and reused.
- `plugins/dagsweep.py`: one task per (group, dag_seed, resample, model_seed);
  data generated in the run at prepare; base config only (enabled optuna block
  is rejected); finalize writes `groups/index.json` and prunes ds arrays.
  Reuses Optuna-free helpers of `causaliT/euler_sweep`.
- `scripts/euler.sh`: FUNCTION training | sweep | dagsweep, TRAIN_FN set.
- Pre-flight check (CPU, seconds): plugin `check_task` (+ optional
  `check_selection`) via `train_fns.check` = update_config, dataset metadata,
  seq lengths, adaptive requirements, one model build. Runs in `launch`
  before submitting (`--no-check`, CHECK=0 in euler.sh) and as
  `cli check --run-dir <run> [--all]`. dagsweep checks one task per group.
- Fixed after the first Euler run: update_config needs OmegaConf (not dict);
  DATA_DIR is $PROJ_HOME/data (was data/input).
- `tests/test_euler_funs.py` (fast, no training).

## To do
1. Pilot on Euler: one short training (fold mode), one sweep, one dagsweep
   with a forced pause (small walltime) to check resume, incl. adaptive.
2. Optional: k-fold/adaptive summaries in `finalize`, `export_rewrite`.
3. After the pilot: retire causaliT/euler_optuna, causaliT/euler_sweep (move
   the helpers used by plugins/dagsweep.py first) and the old
   scripts/*_parallel*.sh, *_scratch_job*.sh; remove optuna from the codebase.
4. Copy `tests/test_workflow.py` from euler_workflow and adapt imports.
