# euler_optuna — Quick Start

Hyperparameter optimisation for causaliT causal transformer models using Optuna.

---

## Objective

Ensure **fair comparison** between model architectures by giving each architecture enough capacity to perform its best reconstruction on each dataset. Models differ structurally (SingleCausalLayer vs NoiseAwareSingleCausalLayer vs StageCausaliT), so identical parameter counts do not imply identical expressivity. The optimisation search finds the best combination of architectural capacity parameters (and learning rate) that minimises **validation reconstruction loss** (`val_x_mae`) for each (model × dataset) pair.

---

## Design Decisions

### Why validation loss (not causal metrics)?

Optuna is run **before** causal training. The goal is to ensure each model has enough representational capacity for the dataset, so we optimise pure reconstruction ability. Causal penalties (L1, HSIC, notears, gradient routing) are all disabled during the search — only `val_x_mae` from the best checkpoint is returned to Optuna.

### Why not k-fold cross-validation?

With k-fold, each trial runs k × more epochs, which multiplies wall-clock time by k on a finite GPU budget. For capacity search (not final model selection), a single 80/20 split with early stopping is sufficient and much more practical. The noise from a single split is well managed by running enough Optuna trials (typically 50).

### What is frozen during the search?

The **optimization protocol** (applied automatically per model) freezes all causal/structural learning mechanisms so they do not interfere with reconstruction quality:
- `lambda_l1`, `lambda_hsic`, `lambda_notears`, `lambda_group_l1`: all set to 0
- `hard_mask_files`: cleared (no gradient routing constraint)
- `k_fold`: fixed to 1 (single 80/20 split)
- Early stopping: `patience=10` on `val_x_mae` (min_delta=1e-4)
- `best=True`: metrics are taken from the **best reconstruction checkpoint**, not the final epoch

This means: we find the architecture that is best at reconstruction **without any structural bias** — which sets the upper bound for what causal training can achieve.

### Learning rate included?

Yes. LR is sampled on a log-uniform scale [1e-4, 1e-3]. Architectural capacity and LR interact: a larger model may need a different LR schedule to converge within the epoch budget, so joint optimisation is necessary for a fair comparison.

### Models with special protocols (heavy sparsity conditions)?

Models like `NoiseAwareSingleCausalLayer` that use gradient routing / batch-key dropout are optimised in their **best-case scenario**: all structural constraints disabled, dense attention. The rationale is that we want to know if the architecture has enough expressivity for the data — not whether it can learn under adversarial sparsity conditions. Capacity search gives the "ceiling"; causal training then constrains the model from there.

---

## Prerequisites

```bash
pip install optuna
# or, if it's been added to requirements.txt already:
pip install -r requirements.txt
```

Optionally, place an `optuna_settings.yaml` file in your experiment directory
to configure the study (see the template at the bottom of this page).

---

## Supported Models

| `model_object` in config | Params optimised |
|--------------------------|-----------------|
| `proT` | `d_model_set`, `e_layers`, `d_layers`, `n_heads`, `lr`, `dropout` |
| `StageCausaliT` | `d_model_set`, `d1_layers`, `d2_layers`, `n_heads`, `lr`, `dropout` |
| `SingleCausalLayer` / `SingleCausalLayerRes` | `d_model_set`, `dec_layers`, `n_heads`, `lr`, `dropout` |
| `NoiseAwareSingleCausalLayer` / `NoiseAwareSingleCausalLayerRes` | `d_model_set`, `dec_layers`, `n_heads`, `lr`, `dropout` |

`d_ff` and `d_qk` are **not** sampled — they are derived automatically from
`d_model_set` via the `d_ff_mult` / `d_qk_mult` multipliers in `update_config()`.

---

## Workflow

### 1 — Create the study

```bash
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \
    --exp_id 3_OPT_STUDY/my_experiment \
    --study_name capacity_study \
    --mode create
```

The study database is created at `experiments/3_OPT_STUDY/my_experiment/optuna/study.db`.
An `optuna_protocol.json` is also saved alongside the DB, documenting exactly which
config overrides were applied for this search.

The default metric is `val_x_mae_mean` (best-checkpoint MAE).
To use a different metric, pass `--optimization_metric <name>`.

---

### 2 — Run optimisation

#### Sequential (local or single cluster job)

```bash
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \
    --exp_id 3_OPT_STUDY/my_experiment \
    --study_name capacity_study \
    --mode resume
```

#### Parallel on SLURM cluster

```bash
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \
    --exp_id 3_OPT_STUDY/my_experiment \
    --study_name capacity_study \
    --mode resume \
    --parallel --cluster \
    --n_trials 50 \
    --max_concurrent_jobs 8 \
    --scratch_path $SCRATCH/my_experiment
```

This generates and submits a SLURM job array where each task runs one trial.
Each trial requests 1 GPU independently — much more scheduler-friendly than
allocating multiple GPUs in a single job.

---

### 3 — View results

```bash
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \
    --exp_id 3_OPT_STUDY/my_experiment \
    --study_name capacity_study \
    --mode summary
```

Results are saved to `experiments/3_OPT_STUDY/my_experiment/best_trial.yaml`.

---

## Directory structure after a study

```
experiments/3_OPT_STUDY/my_experiment/
├── config_*.yaml               # Your experiment config
├── optuna_settings.yaml        # (optional) n_trials, sampler, pruner
├── best_trial.yaml             # Best trial summary (created by --mode summary)
└── optuna/
    ├── study.db                # SQLite database
    ├── optuna_protocol.json    # Documents which config overrides were applied
    ├── run_0/                  # Trial 0 config + outputs
    │   └── config.yaml
    ├── run_1/
    │   └── config.yaml
    └── slurm_logs/             # SLURM stdout/stderr (parallel mode only)
```

---

## CLI reference

```
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--exp_id` | required | Path inside `experiments/` |
| `--study_name` | `capacity_study` | Optuna study name |
| `--mode` | required | `create` / `resume` / `summary` |
| `--optimization_metric` | `val_x_mae_mean` | Metric column name + `_mean` suffix |
| `--optimization_direction` | `minimize` | `minimize` or `maximize` |
| `--sampling_profile` | `baseline` | Sampling bounds profile |
| `--parallel` | `False` | Use SLURM job arrays (requires `--cluster`) |
| `--n_trials` | `50` | Total trials (parallel mode) |
| `--max_concurrent_jobs` | `6` | Max concurrent SLURM jobs |
| `--walltime` | `5-00:00:00` | SLURM walltime per trial |
| `--gpu_type` | `rtx_4090` | SLURM GPU type |
| `--mem_per_cpu` | `23g` | SLURM memory per CPU |
| `--scratch_path` | `None` | SCRATCH directory (cluster) |

---

## optuna_settings.yaml template

Copy to your experiment directory to override study defaults:

```yaml
# Total number of trials
n_trials: 50

# Optimisation direction (overrides --optimization_direction)
direction: "minimize"

# Sampler: "sobol" (quasi-random exploration) or "tpe" (Bayesian)
sampler:
  name: "sobol"

# Pruner: "none", "median", or "hyperband"
pruner: "none"
pruner_warmup: 5   # warmup steps before median pruning starts
```

---

## Sampling bounds (baseline profile)

| Parameter | Range | Step |
|-----------|-------|------|
| `d_model_set` | 16 – 128 | 16 |
| `n_heads` | {1, 2, 4} | categorical |
| `dec_layers` / `d1_layers` / `d2_layers` / `e_layers` / `d_layers` | 1 – 4 | 1 |
| `dropout` | 0.0 – 0.3 | 0.1 |
| `lr` | 1e-4 – 1e-3 | log scale |

Adjust bounds by editing `BASELINE_SAMPLING_BOUNDS` in
`causaliT/euler_optuna/euler_optuna/cli.py`.

---

## Resuming / adding more trials

Simply re-run `--mode resume`.  Optuna loads the existing study and continues
from where it left off.  To increase the trial budget, either:

- Update `n_trials` in `optuna_settings.yaml` before resuming, or
- Pass `--n_trials <new_total>` in parallel mode.

---

## Known issues / notes

### Multi-head attention with SVFA (ScaledDotSoftmax)

The SVFA architecture uses **separate head counts** for Q/K (structure) and V (value).
When `n_heads_struct = 1` and `n_heads_value > 1`, the Q/K tensors are 3-D while V is
4-D. `ScaledDotSoftmax` handles this correctly via the mixed-head einsum path
(`"bls,bshd->blhd"`), added in `causaliT/core/modules/attention.py`. The same fix
also applies to `CausalCrossAttention`, `SigmoidCrossAttention`, and
`ToeplitzAttention`.


