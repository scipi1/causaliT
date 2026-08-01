# Training summary and the runtime metric

Every path that fits a model writes one file at the run root:

```
<run_dir>/training_summary.json
```

The causal-attention trainers and the benchmark methods (NOTEARS, DAGMA, PC)
write the same schema, so a run's training record is readable without knowing
which method produced it.

## What belongs here, and what does not

| Goes in `training_summary.json` | Goes in `eval/` |
|---|---|
| runtime, epochs, early stopping | SHD, precision, recall, MEC |
| final train/val/test loss, MAE, R2, HSIC | skeleton / v-structure metrics |
| trainable parameters, device | learned DAG edges, attention scores |

Structural evaluation is produced *after* training by the evaluation functions
and must not be written into this file. Runtime is a **training** metric, so it
lives here and not in `dag_metrics.json`.

## The "fit" abstraction

A *fit* is one optimisation run that produced one estimate:

- causal-attention models: one cross-validation fold (`k_0`, `k_1`, ...)
- benchmark methods: one seed (`seed_0`, `seed_1`, ...)

Treating them as the same object is what makes the comparison possible: runtime
and structural metrics are then aggregated the same way, as mean +- std over the
repetition axis.

## Schema (version 1)

```jsonc
{
  "schema_version": 1,
  "run": {
    "kind": "model" | "benchmark",
    "method": "atsel" | "notears_linear" | ...,
    "dataset": "scm3c",
    "save_dir": "...",
    "timestamp": "2026-08-01T11:20:00"
  },
  "environment": {"device": "cuda:...", "n_threads": 8, "python": "3.11.5", "torch": "2.1.0"},
  "n_fits": 3,
  "fits": [
    {
      "id": "k_0",
      "method": "atsel",
      "seconds": 812.4,             // wall-clock of the fit only
      "epochs_run": 140,
      "max_epochs": 200,
      "stopped_early": true,
      "avg_time_per_epoch": 5.8,
      "trainable_params": 184320,
      "converged": null,            // benchmarks only, where exposed
      "iterations": null,           // benchmarks only, where exposed
      "checkpoint": "k_0/checkpoints/best_checkpoint.ckpt",
      "metrics": {"val_x_mae": 0.031, "test_r2": 0.94, "val_hsic_reg": 0.0004}
    }
  ],
  "statistics": {"atsel": {"seconds": {"mean": 800.1, "std": 12.0, "min": 780.0, "max": 815.0, "n": 3}}},
  "best_fit": {"id": "k_1", "selection_criterion": "val_hsic_reg", "...": "..."}
}
```

### Conventions

- **Missing means missing.** A benchmark has no `epochs_run` and no `test_r2`;
  those keys are absent, never `0.0`. Consumers should carry the absence
  through as `NaN`.
- **JSON-native at write time.** Tensors and numpy scalars are converted when
  written, so no `"tensor(0.0005)"` strings ever reach the file and no repair
  pass is needed afterwards.
- **`statistics` is per method**, so a benchmark run fitting several methods in
  one folder does not average them together.
- The file is rewritten after every fit, so an interrupted run still leaves a
  valid summary of the fits that completed.

## The runtime metric

**Definition.** `seconds` is the wall-clock time of the training/optimisation
call that produced the scored estimate. It excludes dataset generation,
evaluation and checkpoint I/O.

- Benchmarks: the method's fit call.
- Model: total training time of the fold. For staged trainers
  (`anm_alternating_trainer`, `staged_trainer`) it is the **sum over stages**,
  since all stages are needed to produce the final estimate.

**Reported as** mean +- std over seeds, in the same row as SHD / precision /
recall - runtime is only meaningful next to the quality it bought.

**Total, not time-to-best.** With early stopping the scored checkpoint existed
at epoch `E_best`, but the clock ran to `E_best + patience`. The reported number
is the total, because you cannot know `E_best` without paying the patience. The
baselines are symmetric: their time includes the iterations after the returned
estimate stopped improving.

### Two caveats that must be disclosed, not hidden

1. **Censoring.** If `epochs_run == max_epochs` (or a baseline hit its
   `max_iter`), the runtime reports the *budget*, not the method's convergence.
   Such cells should be marked as censored (e.g. `>1200 s`) in any table. This
   is why `max_epochs` / `stopped_early` / `converged` are mandatory fields
   rather than optional extras.
2. **Hardware.** The model trains on GPU; the benchmark methods are
   single-process CPU fits. Absolute seconds across the two is partly a
   hardware comparison, so the `environment` block must be reported alongside.
   The *slope* of runtime against the number of edges is the hardware-robust
   part of the claim.

### Tuning cost

Optuna trials each run through the normal training path, so every trial folder
carries its own `training_summary.json`. The tuning cost is therefore
recoverable, but is **not** part of the headline runtime, which is the cost of
the declared configuration. If reported, it belongs in a separate
`tuning_seconds` column with a footnote, since the baselines have zero tuning
cost by protocol.

## Reading a summary

```python
from causaliT.training.training_summary import load_training_summary, get_statistic

summary = load_training_summary(run_dir)      # None if the folder is not a run
mean_seconds = get_statistic(summary, "seconds", "mean")
```

`load_training_summary` transparently falls back to the legacy
`kfold_summary.json` of previously finished experiments, translating it into
the current schema. Fields the old format never had (device, `epochs_run`,
`stopped_early`, ...) stay absent, so an old run is visibly less informative
rather than silently looking complete.

## Detecting a run

`causaliT.evaluation.eval_sweeps.is_trained_run` accepts **either** marker
(`training_summary.json` or `kfold_summary.json`). This matters more than it
looks: `get_df_recursive` keeps descending while the predicate is `False`, so a
detector that recognised only one format would walk straight past valid runs and
return an **empty DataFrame with no error**.

Benchmark runs are now detected as trained runs too. Since they have no
checkpoint, `eval_models_bottom_action` skips them explicitly on
`run.kind == "benchmark"` rather than failing on a missing checkpoint.
