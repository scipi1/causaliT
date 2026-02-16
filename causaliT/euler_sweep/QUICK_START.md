# Euler Sweep - Quick Start Guide

## AI Integration Guide (Copy & Wire)

**For AI Assistants**: When a user asks you to "wire" euler_sweep into their project, follow these steps:

### What to Update in `euler_sweep/cli.py`:

1. **Import the user's training function** at the top of the file:
   ```python
   # Replace:
   # from sweeper import run_sequential_sweep, run_parallel_sweep
   
   # Add user's training function import:
   from user_project.training.trainer import trainer  # User's function
   ```

2. **Update ROOT_DIR** to match the project structure:
   ```python
   ROOT_DIR = dirname(dirname(abspath(__file__)))  # Adjust based on where euler_sweep/ is placed
   ```

3. **Update the training function wrapper** (`train_function_for_sweep` or `train_function_template`):
   ```python
   def train_function_for_sweep(config, save_dir, data_dir, cluster, **kwargs):
       # Call user's training function with their specific preprocessing
       return trainer(
           config=config,
           save_dir=save_dir,
           data_dir=data_dir,
           cluster=cluster,
           **kwargs
       )
   ```

4. **Update `train_fn` assignment** in the sweep command:
   ```python
   train_fn = train_function_for_sweep
   ```

5. **Update paths** in the sweep command:
   ```python
   # Experiment directory
   exp_dir = join(ROOT_DIR, "experiments", exp_id)  # Adjust folder name
   
   # Data directory
   data_dir = join(ROOT_DIR, "data", "input")  # Adjust path
   ```

6. **For parallel execution**, update `train_fn_module` and `train_fn_name`:
   ```python
   train_fn_module = "user_project.euler_sweep.cli"  # Module path to this file
   train_fn_name = "train_function_for_sweep"
   ```

### Example Prompt for AI Assistant:
```
Wire the euler_sweep module into my proT project:
- Copy euler_sweep/ to proT/proT/euler_sweep/
- My training function is `trainer` in `proT.training.trainer`
- It needs `update_config(config)` preprocessing before training
- Experiments are in `experiments/training/`
- Data is in `data/input/`
```

---

## What is Parameter Sweeping?

Parameter sweeping is a systematic way to explore how different hyperparameter values affect your model's performance. Unlike Optuna (which adaptively searches for optimal values), sweeps exhaustively test predetermined parameter combinations.

**Use sweeps when you want to:**
- Compare specific parameter values systematically
- Understand parameter sensitivities
- Test all combinations of a small parameter set
- Reproduce results with exact parameter grids

**Use Optuna when you want to:**
- Find optimal hyperparameters automatically
- Explore large search spaces efficiently
- Let the algorithm decide which values to try

---

## 3-Minute Setup

### 1. Copy Template Files

```bash
# Copy sweeper module to your project
cp -r euler_sweep/euler_sweep/ your_project/

# Or add to your project's imports
```

### 2. Create Experiment Directory

```bash
mkdir -p experiments/my_sweep_exp
mkdir -p experiments/my_sweep_exp/sweeper
```

### 3. Create Configuration Files

**config.yaml** - Define your model and training parameters (in experiment root):
```yaml
model:
  hidden_dim: 128
  num_layers: 3

training:
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 50
```

**sweeper/sweep.yaml** - Define what to sweep (in sweeper subdirectory):
```yaml
model:
  hidden_dim: [64, 128, 256]
  num_layers: [2, 3, 4]

training:
  learning_rate: [0.0001, 0.001, 0.01]
```

**Note**: The sweep configuration must be in `experiment_id/sweeper/sweep.yaml` (like Optuna's structure).

### 4. Customize CLI

Edit `euler_sweep/cli.py`:

```python
# Import your training function
from your_project.trainer import train_model

# Replace train_function_template
train_fn = train_model
```

### 5. Run Your Sweep

```bash
# Sequential independent sweep (14 runs: 3 + 3 + 3)
python euler_sweep/cli.py sweep --exp_id my_sweep_exp --sweep_mode independent

# Sequential combination sweep (27 runs: 3 × 3 × 3)
python euler_sweep/cli.py sweep --exp_id my_sweep_exp --sweep_mode combination

# Parallel combination sweep on cluster (requires --cluster)
python euler_sweep/cli.py sweep \
    --exp_id my_sweep_exp \
    --sweep_mode combination \
    --parallel \
    --cluster \
    --max_concurrent_jobs 10
```

**Important**: Parallel execution (`--parallel`) requires `--cluster` flag. Local machines cannot run SLURM job arrays.

---

## Sweep Modes Explained

### Independent Sweep
Tests **one parameter at a time**, keeping others at default:

```yaml
# Config defaults: hidden_dim=128, lr=0.001
# Sweep:
model:
  hidden_dim: [64, 256]
training:
  learning_rate: [0.0001, 0.01]
```

**Creates 4 runs:**
1. hidden_dim=64, lr=0.001 (default)
2. hidden_dim=256, lr=0.001 (default)  
3. hidden_dim=128 (default), lr=0.0001
4. hidden_dim=128 (default), lr=0.01

**Use for:** Initial exploration, baseline comparisons

---

### Combination Sweep
Tests **all possible combinations** (Cartesian product):

```yaml
# Same sweep as above
```

**Creates 4 runs:**
1. hidden_dim=64, lr=0.0001
2. hidden_dim=64, lr=0.01
3. hidden_dim=256, lr=0.0001
4. hidden_dim=256, lr=0.01

**Use for:** Finding optimal combinations, studying interactions

**Warning:** Combinations grow exponentially!
- 3 params × 3 values each = 3³ = 27 runs
- 5 params × 3 values each = 3⁵ = 243 runs

---

## Directory Structure

### Experiment Setup
```
experiments/my_sweep_exp/
├── config.yaml              # Main config (or config_*.yaml - flexible naming)
└── sweeper/
    └── sweep.yaml          # Sweep definition
```

### Independent Sweep Results
```
experiments/my_sweep_exp/
├── config.yaml
└── sweeper/
    ├── sweep.yaml
    └── runs/
        └── sweeps/
            ├── sweep_hidden_dim/
            │   ├── sweep_hidden_dim_64/
            │   │   ├── config.yaml
            │   │   └── results/
            │   └── sweep_hidden_dim_256/
            │       ├── config.yaml
            │       └── results/
            └── sweep_learning_rate/
                └── ...
```

### Combination Sweep Results
```
experiments/my_sweep_exp/
├── config.yaml
└── sweeper/
    ├── sweep.yaml
    └── runs/
        └── combinations/
            ├── combo_hidden_dim_64_learning_rate_0.0001/
            │   ├── config.yaml
            │   └── results/
            ├── combo_hidden_dim_64_learning_rate_0.01/
            │   ├── config.yaml
            │   └── results/
            └── ...
```

### Parallel Sweep (additional files)
```
experiments/my_sweep_exp/
└── sweeper/
    ├── sweep.yaml
    ├── combinations_data.json    # Metadata for all combinations
    ├── run_sweep_array.sh        # Generated SLURM script
    ├── job_id.txt               # Submitted job ID
    ├── slurm_logs/              # SLURM output logs
    └── runs/
        └── combinations/
            └── ...
```

---

## Common Commands

```bash
# Local sequential sweep
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination

# Cluster parallel sweep
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination \
    --parallel \
    --cluster \
    --scratch_path $SCRATCH/my_exp

# Customize SLURM resources
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination \
    --parallel \
    --max_concurrent_jobs 20 \
    --walltime "3-00:00:00" \
    --gpu_mem "48g"

# Dry run (generate scripts without submitting)
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination \
    --parallel \
    --submit_jobs False
```

---

## Best Practices

### 1. Start Small
```yaml
# Good: 2-3 values per parameter
model:
  hidden_dim: [128, 256]
training:
  learning_rate: [0.001, 0.01]
# Result: 2 × 2 = 4 combinations

# Bad: Too many values
model:
  hidden_dim: [32, 64, 128, 256, 512, 1024]
training:
  learning_rate: [0.00001, 0.0001, 0.001, 0.01, 0.1]
# Result: 6 × 5 = 30 combinations (too many for initial exploration)
```

### 2. Use Independent First
```bash
# Step 1: Independent sweep to identify important parameters
python cli.py sweep --exp_id exp1 --sweep_mode independent

# Step 2: Analyze results, narrow ranges

# Step 3: Combination sweep on promising ranges
python cli.py sweep --exp_id exp2 --sweep_mode combination
```

### 3. Organize Experiments
```bash
experiments/
├── lr_sweep/          # Focus on learning rates
├── arch_sweep/        # Focus on architecture
├── optimizer_sweep/   # Compare optimizers
└── final_sweep/       # Narrow combinations
```

### 4. Use Parallel for Large Sweeps
```bash
# Sequential: OK for < 10 combinations
python cli.py sweep --exp_id small_sweep --sweep_mode combination

# Parallel: Better for > 10 combinations
python cli.py sweep --exp_id large_sweep --sweep_mode combination --parallel
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Config file not found" | Create `experiments/exp_id/config.yaml` |
| "Sweeper directory not found" | Create `experiments/exp_id/sweeper/` directory |
| "Sweep file not found" | Create `experiments/exp_id/sweeper/sweep.yaml` |
| "Parameter not found in config" | Ensure sweep params exist in config.yaml |
| "Parallel execution requires cluster" | Add `--cluster` flag when using `--parallel` |
| "NotImplementedError: train_function_template" | Replace with your training function in cli.py |
| SLURM job fails | Check module loads and venv path in generated script |

---

## Full Documentation

See `SWEEP_SKELETON_README.md` for:
- Detailed architecture explanation
- Customization guide
- Cluster usage with SCRATCH
- Advanced examples
- Complete API reference

---

## Quick Reference

| Task | Command |
|------|---------|
| Sequential independent | `--sweep_mode independent` |
| Sequential combination | `--sweep_mode combination` |
| Parallel sweep | `--parallel --cluster` |
| Customize resources | `--max_concurrent_jobs N --walltime TIME` |
| Dry run | `--submit_jobs False` |
| Use scratch | `--scratch_path $SCRATCH/exp` |

---

**Ready to sweep?**

Start with a small independent sweep, analyze results, then scale up to combinations!
