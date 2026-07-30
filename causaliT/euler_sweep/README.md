# Euler Sweep - Parameter Sweeps for causaliT

Run systematic parameter sweeps locally or on cluster (SLURM).

## Quick Start

### 1. Create Experiment Structure

```
experiments/<exp_id>/
├── config.yaml          # Your experiment config
└── sweeper/
    └── sweep.yaml       # Parameters to sweep
```

### 2. Define sweep.yaml

**Important**: Sweep parameters must be **2 levels deep** (category.parameter).

```yaml
# ✅ CORRECT - 2 levels deep
training:
  lambda_l1_cross_scores: [0.0, 0.01, 0.05, 0.1]
  learning_rate: [0.001, 0.0001]

model:
  hidden_dim: [64, 128, 256]

# ❌ WRONG - 3+ levels (nested under kwargs)
# model:
#   kwargs:
#     dec_self_attention_type: ["LieAttention", "PhiSoftMax"]
```

### 3. Run Sweep

**Local (sequential):**
```bash
python -m causaliT.euler_sweep.euler_sweep.cli sweep \
    --exp_id single/scm6/my_sweep \
    --sweep_mode combination
```

**Cluster (parallel via SLURM):**
```bash
python -m causaliT.euler_sweep.euler_sweep.cli sweep \
    --exp_id single/scm6/my_sweep \
    --sweep_mode combination \
    --parallel \
    --cluster \
    --max_concurrent_jobs 10
```

## Sweep Modes

| Mode | Description | Example (2 params × 2 values each) |
|------|-------------|-----------------------------------|
| `independent` | One parameter at a time | 4 runs (2 + 2) |
| `combination` | All combinations (Cartesian product) | 4 runs (2 × 2) |

## Results Location

```
experiments/<exp_id>/
└── sweeper/
    ├── sweep.yaml
    └── runs/
        └── combinations/           # or sweeps/ for independent mode
            ├── combo_param1_val1_param2_val1/
            │   └── config.yaml
            └── combo_param1_val1_param2_val2/
                └── config.yaml
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--exp_id` | Experiment folder path (relative to `experiments/`) | Required |
| `--sweep_mode` | `independent` or `combination` | Required |
| `--parallel` | Use SLURM job arrays | `False` |
| `--cluster` | Running on cluster (required for `--parallel`) | `False` |
| `--max_concurrent_jobs` | Max parallel SLURM jobs | `6` |
| `--walltime` | SLURM walltime | `4:00:00` |

---

# DAG sweep (scaling studies)

`dagsweep` answers *"how does the model behave as the graph grows?"* without
paying for hyper-parameter optimisation on every single run.

### Why it exists

A `[10, 50, 100, 500]` node sweep with 10 seeds is 40 runs. Naively adding a
10-trial Optuna study per run gives 440 runs. The insight is that the right
hyper-parameters depend on the **DAG size**, not on which random DAG of that size
you happened to draw. So we group by size:

```
runs = n_sizes x n_trials      (tuning)   +   n_sizes x n_seeds   (evaluation)
     = 4 x 10 = 40                        +   4 x 10 = 40         = 80
```

Seeds are **members** of a group, never an axis of the Cartesian product - that
is the one thing plain `sweep` cannot express.

### Protocol

Per group (= one combination of `group_axes`, e.g. `n_nodes=50`):

1. Sample a **dedicated tuning DAG** using `optuna.opt_seed`.
2. Run **one** Optuna study on it -> `best_trial.yaml`.
3. For each `seeds` entry: sample that seed's own DAG, train with the group's
   best params, run the post-training evaluations.

`opt_seed` must not appear in `seeds` (enforced): tuning and evaluation DAGs stay
disjoint, so no hyper-parameter information leaks into the reported numbers.

### Experiment layout

```
experiments/<exp_id>/
├── config_atsel.yaml     # base config
├── optuna_atsel.yaml     # optional: search space (copied into each group)
└── dagsweep.yaml         # the sweep spec
```

### dagsweep.yaml

```yaml
group_axes:                 # Cartesian product -> one group each, one study each
  n_nodes: [10, 50, 100, 500]

seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # runs *within* every group

dag:                        # RandomSCMConfig fields + generation options
  degree: 2
  linearity: nonlinear
  noise: gaussian
  n_samples: 20000
  normalize_method: minmax

optuna:
  enabled: true
  opt_seed: 1000            # must NOT be in `seeds`
  metric: val_loss
  direction: minimize
  trainer: standard

training:
  trainer: standard         # standard | staged | anm | adaptive

dataset_derived:            # config field <- len(metadata field) of the sampled DAG
  experiment.n_source: variable_info.source_labels
  experiment.n_input:  variable_info.input_labels

delete_dataset: true        # prune ds.npz after each run (default)
```

Notes:

* Any unknown key under `dag:` raises - a typo like `n_node` would otherwise
  silently invalidate the whole sweep.
* Group axis values are exposed as `experiment.<axis>`, so the base config can
  interpolate them (e.g. `d_model: ${experiment.n_nodes}`).
* `dataset_derived` is required whenever the config carries variable counts:
  every group trains on a differently-sized DAG, so those fields must be
  refreshed from the sampled `dataset_metadata.json`.

### Run

```bash
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep \
    --exp_id 7_SCALING/atsel_nodes

# inspect the plan first
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep \
    --exp_id 7_SCALING/atsel_nodes --dry_run
```

| Option | Description |
|--------|-------------|
| `--exp_id` | Experiment folder (relative to `experiments/`) |
| `--dry_run` | Print groups/seeds and exit |
| `--skip_optuna` | Reuse existing `best_trial.yaml`, never tune |
| `--force_optuna` | Re-tune even if a study summary exists |
| `--keep_data` | Keep every `ds.npz` (debugging; expensive) |
| `--cluster` | Cluster-side settings (worker counts, etc.) |

### Results layout

```
experiments/<exp_id>/groups/n_nodes_50/
├── config_atsel.yaml        # staged config (tuning DAG)
├── best_trial.yaml          # the group's hyper-parameters
├── datasets/                # group-local data root
│   └── random_n50_k2_.../   # light artefacts kept, ds.npz pruned
│       ├── dataset_metadata.json
│       ├── dag_recipe.json  # everything needed to rebuild the arrays
│       └── ...
└── sweeper/runs/combinations/<exp_id>_n_nodes_50_seed_3/
```

### Disk & reproducibility

Sample arrays dominate disk usage (a 500-node dataset is large, and there are
`n_sizes x (1 + n_seeds)` of them), so `ds.npz` is **ephemeral**: generated
before a run, pruned after it. Everything an evaluation needs later
(`dataset_metadata.json`, masks, ATE ground truth, normalization) is kept.

Each dataset also carries `dag_recipe.json`, which pins the full
`RandomSCMConfig` plus generation kwargs. The arrays are a pure function of it,
so any dataset can be rebuilt bit-identically:

```bash
python -m causaliT.euler_sweep.euler_sweep.cli dagsweep-regen \
    --dataset_dir experiments/7_SCALING/atsel_nodes/groups/n_nodes_50/datasets/random_n50_k2_nonlinear_gaussian_s3
```

The recipe is also what lets `eval_ate_mc` reconstruct a live SCM for a sampled
DAG (sampled graphs are not in the static SCM registry).

### Robustness

* One failing run is logged and recorded as `failed`; the sweep continues.
* A group with an existing `best_trial.yaml` is not re-tuned, so an interrupted
  sweep can simply be relaunched.


