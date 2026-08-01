# Dimensioning the model to the DAG: the `dagsweep` Optuna phase

How phase 1 of `causaliT.euler_sweep.euler_sweep.cli dagsweep` works, what it
optimises and why.  Code: `causaliT/euler_sweep/euler_sweep/search_space.py`
(pure, unit-tested functions) driven from `opt_train_sweep.py`.

## The objective of the search

A scaling benchmark is only fair if **every DAG size gets a model of
proportional capacity**.  If a fixed 32-dim model is used at 6 nodes and at 400
nodes, a degrading SHD tells us nothing about the method: the large graph simply
had no capacity.  The search therefore answers ONE question per DAG size:

> what is the smallest model that can still reconstruct the mechanisms of a DAG
> of this size, and at what learning rate?

Everything else (structure learning quality) is then measured in phase 2 with
that model held fixed.

## What is tuned (and what is not)

| Parameter | Why |
|---|---|
| `experiment.d_model_set` | capacity, **and** a hard feasibility constraint (below) |
| `experiment.n_heads` | how the width is split; larger DAGs have wider fan-in, so the useful head count changes with size |
| `training.lr` | must co-adapt with width/heads; otherwise a "too small" verdict is really "badly optimised" |

Not tuned: depth, dropout, `d_ff_mult`, and every structural lambda
(`lambda_hsic`, `lambda_l0`, `kappa`, ...).  Tuning the lambdas would tune the
answer the benchmark is supposed to measure.

Two fields are **derived, not searched** (`size_derived` in `dagsweep.yaml`) —
they are functions of the node count, so leaving them fixed silently changes the
experiment across sizes:

* `experiment.batch_size` — from an activation budget (see below);
* `experiment.query_fanin_scale` — `F = n_keys * x_sat^2`
  (`docs/experimental_elaborations/QUERY_FANIN_SCALE_BUDGET.md`).

## 1. Adaptive width range

```
choices = aligned multiples of `align` in [n_keys, size_mult * n_keys],
          endpoints pinned, at most `max_choices` entries
```

* **Lower bound = `n_keys`, non-negotiable.**  With
  `struct_embedding_type: orthogonal_fixed` the structural frame needs one
  dimension per node; `d_model < n_keys` cannot represent an orthogonal key set
  at all (and the run dies inside model construction).
* **Upper bound = `size_mult * n_keys`** (default 2x): "at most twice the strict
  minimum" — enough head-room to show the knee, small enough to stay honest.
* **`max_choices` (default 8)** caps the candidate count, so the search cost does
  NOT grow with the DAG: 6 nodes and 400 nodes both get <= 8 candidates.
* **`align` (default 8)** must be a multiple of the largest searched `n_heads`,
  which guarantees `d_model % n_heads == 0` for every combination.

Examples (`align=8, size_mult=2, max_choices=8`):

| `n_keys` | choices |
|---|---|
| 6 | 8, 16 |
| 10 | 16, 24 |
| 50 | 56, 64, 72, 80, 88, 96, 104 |
| 400 | 400, 456, 512, 568, 632, 688, 744, 800 |

(`n_keys = 10` gives `F = 68.7`, reproducing the value derived by hand in the
`6_INVESTIGATIONS` arms — a useful cross-check of the derived rule.)


## 2. Reconstruction-only trial protocol

`optuna.protocol: reconstruction` rewrites every trial config to:

* `use_gradient_routing: false`, `k_fold: 1`;
* **all** structural weights zero — `lambda_hsic`, `lambda_l0`, `kappa`,
  `lambda_query_norm`, the score-sparsity/noise-prior family, and the adaptive
  trainer's private `adaptive_training.structure.lambda_l0`;
* early stopping on `val_x_mae` (a trial ends at its own plateau);
* **no evaluations** — `evaluation.functions: []`, plus
  `adaptive_training.eval_dag: false` and
  `adaptive_training.run_final_evaluations: false`.


Why: `val_x_mae` is only a clean **capacity** signal if nothing else can lower
it.  With HSIC/L0 active, a good value could equally mean "structure was
ignored".  And a model that cannot reconstruct cannot do structure either — HSIC
itself is computed on residuals — so reconstruction is the right necessary
condition to size on.  The current model settings (frozen orthogonal key
embeddings, no `W_q`/`W_K`, free query embedding) are kept exactly as in the
evaluation config; only the losses change.

### Why no evaluations during the search

A trial is scored on `val_x_mae` alone, so nothing ever reads the evaluation
artefacts — and they are meaningless here anyway, because with every structural
lambda at zero the DAG never trains.  `eval_interventions` is the expensive one
(it rebuilds the SCM from `dag_recipe.json` and runs a Monte-Carlo intervention
per node), and its cost grows with the node count, i.e. exactly where the search
is already most expensive.

There are **three** independent paths and the protocol closes all of them:

| Path | Setting | When it fires |
|---|---|---|
| shared post-training suite | `evaluation.functions: []` | end of run (both trainers) |
| adaptive post-training suite | `adaptive_training.run_final_evaluations: false` | end of run |
| adaptive in-fit DAG diagnostics | `adaptive_training.eval_dag: false` | **at every phase switch** |

> **Trap: `evaluation.functions: null` means "run ALL evaluations", not "run
> none".**  Both `trainer._run_post_training_evaluations` and
> `run_evaluations_from_config` treat `functions is None` as the
> "no list given, run the defaults" sentinel.  The value that disables the suite
> is an **empty list**.  `tests/test_dagsweep_search_space.py` asserts this
> explicitly, so it cannot be "simplified" back to `null`.

The adaptive keys are written only when an `adaptive_training` block already
exists, so a `standard`-trainer config is not given a spurious section.
**Phase 2 is deliberately untouched**: the evaluation seeds keep running
`eval_attention_scores` + `eval_interventions`, because SHD/MEC and ATE *are* the
benchmark result.  The overrides apply to trial configs only.


## 3. Parsimonious selection (the knee, not the argmin)

Reconstruction error is near-monotone in capacity, so `argmin` would always
return the largest width in the range — the adaptive range would buy nothing and
every size would be over-parameterised.  Default `selection.mode: parsimonious`
returns the **smallest model whose metric is within `tol` (relative) of the best
observed** (`tol: 0.02` = "give up at most 2% reconstruction quality for a
smaller model").  Model size is the real `trainable_params` count reported by the
trainer, so 1-head and 4-head models are compared on actual size.

`best_trial.yaml` records the chosen trial, the raw argmin, and the full
capacity/metric `curve` — that curve IS the "required capacity vs DAG size"
result of the scaling study.

## 4. Batch size from an activation budget

Peak activation memory of a block scales as `B * N * H * (N + d)` (value stream
plus attention maps), so

```
B = C / (N * H * (N + d_ref)),   snapped down to a power of two, clamped
```

with `d_ref = d_ref_mult * n_keys` (the **ceiling** of the width range, not the
sampled width).  Using the ceiling makes the batch a pure function of the DAG
size: identical for every trial of a group and for the evaluation runs, so the
tuned learning rate remains valid.

`C` is the single device-specific constant.  Measure it once per machine:

```bash
python -m causaliT.euler_sweep.euler_sweep.cli calibrate-batch-budget
```

It writes `~/.causalit/activation_budget.json` (or `$CAUSALIT_CACHE_DIR`), keyed
by GPU name; `C: auto` in `dagsweep.yaml` picks it up.  After an OOM, re-run with
`--safety 0.2`: the batch shrinks at every size at once, which keeps sizes
comparable.

## Self-repairing dimension rules

Before each run (and before staging the search config) `validate_dimensions`
checks and, by default, **fixes** the config instead of aborting an hours-long
run — every rule has a unique right answer, and the repair is logged:

1. `d_model >= n_keys` -> raise the width to the next aligned value;
2. `d_model % n_heads == 0` -> raise the width (never lower it);
3. `d_qk * n_heads_struct == d_model` when `remove_query/key_projection` -> write
   `experiment.d_qk` explicitly;
4. `query_fanin_scale ~= n_keys * x_sat^2` -> write the saturating value.

Pass `repair=False` for strict mode (used by the tests).

## Anti-leakage and cost

* The study runs on a dedicated `optuna.opt_seed` DAG, which may not appear in
  `dag_seeds` (the loader rejects the overlap) — tuning never sees an evaluated
  graph.
* One study per group, reused by every seed:
  `sizes x n_trials + sizes x dag_seeds x model_seeds` runs instead of
  `sizes x seeds x n_trials`.
* `best_trial.yaml` short-circuits the phase, so an interrupted sweep never
  re-tunes a finished group (`--force_optuna` to redo, `--skip_optuna` to reuse).

## Configuring it

`optuna_settings.yaml` (search space + budget + selection) and the `optuna` /
`size_derived` blocks of `dagsweep.yaml`; a fully commented pair lives in
`experiments/0_TESTS/FUN_dagsweep/`.  Parameter names are **dotted config
paths** and are used verbatim as the Optuna parameter names, which is what makes
`best_trial.yaml` applicable to any config with `OmegaConf.update` — and what
makes adding a hyper-parameter for another benchmark model a YAML edit rather
than a code change.

## Applying this to other benchmark models

1. keep `protocol: reconstruction` (it only zeroes losses the model may not have);
2. replace the `search_space` entries with that model's capacity knobs —
   `adaptive_width` is available for any width-like field, otherwise use
   `categorical` / `int` / `float`;
3. drop the `size_derived` rules that do not apply (`fanin_saturating` is
   specific to the Hard-Concrete gate; `activation_budget` is generic);
4. keep `selection.mode: parsimonious` so all methods are sized by the same
   "smallest sufficient model" rule.

### External baselines (NOTEARS, DAGMA, PC) are not tuned at all

The structure-learning baselines take a different route on purpose: they run with
fixed paper hyperparameters, so they have no search space and skip Optuna
entirely.  Point the trainer at them and the sweep fits them on exactly the same
generated DAGs and datasets as the models:

```yaml
training:
  trainer: benchmark      # -> cli.benchmark_function_for_sweep
benchmark:
  methods: [notears_linear, dagma_linear, pc]
  seeds: [0, 1, 2]
```

Each method writes its own `eval/eval_benchmark_<method>/` folder with the
standard `dag_metrics.json`, so `eval_seed_sweep` and the analysis notebooks
aggregate baselines and models with the same code.  Details and the reasoning
behind the fixed hyperparameters: `BENCHMARKS.md`.

## Troubleshooting

**`ValueError: No completed trial: cannot select hyper-parameters.`**
The search space is not the problem: EVERY trial crashed during training, and the
selection step is only the first place that notices.  The message now lists the
recorded reason of the last failed trials (`trial N {params}: RuntimeError ...`),
which is the line to act on.  Per-trial logs live in
`groups/<group>/optuna/run_<n>/`.

Known cause, fixed: `DataLoader worker (pid(s) ...) exited unexpectedly`.  The
trainer used to hardcode `num_workers=20` off-cluster; on Windows workers are
spawned (not forked), so 20 processes each re-imported torch and re-copied the
dataset and the pool died at the first batch.  `trainer.resolve_num_workers` now
returns 0 on Windows, 1 on the cluster, and `min(8, cpu_count // 2)` otherwise;
`training.num_workers` overrides it if a machine needs something else.

## Running it in parallel on the cluster

`--cluster` turns the same spec into a chained SLURM job graph instead of one
long process (add `--sequential` to force the linear run on a cluster node):

```
prep (CPU)            generate every DAG, stage the group configs, create the
  |                   study DBs, write dagsweep/plan.json
trials[0..T-1%C]      one array task = ONE Optuna trial          (1 GPU each)
  |  afterany
select (CPU)          per group: parsimonious pick -> best_trial.yaml
  |  afterok
train[0..R-1%C]       one array task = ONE (dag_seed, model_seed) run (1 GPU)
  |  afterany
cleanup (CPU)         prune ds*.npz, write the planned-vs-reached report
```

`T = groups x n_trials`, `R = groups x |dag_seeds| x |model_seeds|`, both capped
at `--max_concurrent_jobs` concurrent tasks.

Why it is shaped like this:

* **`select` is a global barrier.**  The sweep is two-phase by construction (a
  run must train the *selected* model), and one barrier for all groups is the
  cheapest correct option: no group can start training on a study that is still
  running.  Trials and runs are parallel *within* their phase, which is where
  essentially all the GPU time is.
* **Every DAG is generated in `prep`, never in a task.**  Model seeds share a
  DAG on purpose (that is what makes edge stability attributable to the model),
  so two concurrent tasks would otherwise race on the same dataset folder.  The
  price is that pruning moves to `cleanup`: peak disk holds every dataset of the
  sweep, so `delete_dataset` frees space at the end rather than after each run.
* **`select` never raises.**  The train array depends on it with `afterok`, so a
  group whose study produced no usable trial is recorded as failed and only *its*
  runs are refused — the other groups still train.
* **A run refuses to train untuned when tuning was requested.**  Otherwise a
  failed study would silently yield results indistinguishable from tuned ones.
  `--skip_optuna` (or `optuna.enabled: false`) is the explicit untuned arm.
* **One progress file per item** (`dagsweep/progress/*.json`): array tasks never
  write the same file, so nothing is lost when the walltime kills a task and a
  re-submission simply resumes (an already-`ok` run is skipped).

As everywhere else in the project, the entry point is a **template shell script
holding all the options**, submitted with `sbatch` (edit the variables at its
top: experiment id, concurrency, walltime, venv, scratch folder):

```bash
sbatch scripts/dagsweep_parallel.sh
```

That job only plans and submits (seconds), then exits; the five stage scripts it
generates live in `<run folder>/dagsweep/scripts/` and the logs in
`<run folder>/dagsweep/slurm_logs/`.  It wraps the CLI, so the same thing can be
done by hand on a login node:

```bash
# submit the chain
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 7_SCALING/atsel_nodes --cluster --max_concurrent_jobs 10 --walltime 24:00:00

# write the five job scripts WITHOUT submitting (inspect them first)
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 7_SCALING/atsel_nodes --cluster --dry_run

# planned vs reached, per group (works while the chain is still running)
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep-status --exp_id 7_SCALING/atsel_nodes
```

`--scratch_path` (set in the template) copies only the spec files
(`config*.yaml`, `dagsweep*.yaml`, `optuna*.yaml`) to scratch and runs there, so
datasets, checkpoints and state are *born* outside `$HOME`.  Any single stage can
be re-submitted by hand (`sbatch <run folder>/dagsweep/scripts/train.sh`) after a
partial failure.

Two things to size before a big sweep: run `calibrate-batch-budget` on a node of
the target partition (otherwise `C: auto` falls back to a conservative default),
and set `--walltime` to the cost of ONE task, not of the whole sweep.

## Tests

`tests/test_dagsweep_search_space.py` — width feasibility/boundedness, dotted
round-trip into a config, protocol zeroing, batch-rule monotonicity and
width-independence, the `F` closed form, the four repair rules, and the knee
selection (including the argmin fallback and maximised metrics).

`tests/test_dagsweep_parallel.py` — the parallel plan and state machine: array
sizes derived from the spec alone (no dataset), one task per item, the
fault-isolating `select`, the refusal to train untuned, model seeds sharing one
dataset and one split, resume/force semantics, cleanup-time pruning, the
planned-vs-reached rollup and the submitted dependency chain.


