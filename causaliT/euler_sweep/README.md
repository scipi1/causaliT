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
