# Euler Sweep - Parameter Sweep Template

Generic template for systematic parameter sweeps in ML experiments. Supports local and cluster execution with SLURM job arrays.

## Overview

Systematically explores hyperparameter spaces by exhaustively testing predetermined parameter combinations. Unlike adaptive methods (Optuna), sweeps test all specified values.

**Use for:** Parameter sensitivity analysis, systematic configuration comparison, reproducible grids, parameter interaction studies

**Avoid for:** Automatic optimization (use euler_optuna), very large search spaces (use euler_optuna)

## Quick Start

### Integration Method: Copy & Wire

The recommended way to integrate euler_sweep into your project:

1. **Copy the `euler_sweep/` folder** into your project
2. **Ask your AI assistant to wire it** with a prompt like:

```
Wire the euler_sweep module into my project:
- My training function is in `my_project/trainer.py` and called `train_model`
- My experiments directory is `experiments/`
- My data directory is `data/input/`
- Update the CLI to use my training function
```

The AI will update `euler_sweep/cli.py` to import your training function and set up the correct paths.

---

### Manual Setup

#### 1. Setup Your Experiment

```bash
# Create experiment directory structure
mkdir -p experiments/my_sweep/sweeper

# Copy template configs
cp configs/config_example.yaml experiments/my_sweep/config.yaml
cp configs/sweep_example.yaml experiments/my_sweep/sweeper/sweep.yaml

# Edit configs for your needs
```

**Note**: Like Optuna, sweep configurations are organized in a `sweeper/` subdirectory within your experiment folder.

#### 2. Customize the CLI

Edit `euler_sweep/cli.py`:
```python
# Import your training function
from your_project.trainer import train_model

# Replace template
train_fn = train_model
```

#### 3. Run Your Sweep

```bash
# Sequential sweep (local or cluster)
python euler_sweep/cli.py sweep --exp_id my_sweep --sweep_mode combination

# Parallel sweep (requires cluster with SLURM)
python euler_sweep/cli.py sweep \
    --exp_id my_sweep \
    --sweep_mode combination \
    --parallel \
    --cluster
```

**Important**: `--parallel` requires `--cluster`. Parallel execution uses SLURM job arrays which are only available on clusters.

## Documentation

- [QUICK_START.md](QUICK_START.md) - Setup guide with examples
- [SWEEP_SKELETON_README.md](SWEEP_SKELETON_README.md) - Complete technical documentation
- [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) - Changelog and cleanup notes

## Key Features

### Two Sweep Modes

**Independent Sweep** - Vary one parameter at a time
```yaml
# Sweep params: hidden_dim=[64, 128], lr=[0.001, 0.01]
# Creates: 4 runs (2 + 2)
```

**Combination Sweep** - Test all combinations (Cartesian product)
```yaml
# Sweep params: hidden_dim=[64, 128], lr=[0.001, 0.01]
# Creates: 4 runs (2 × 2)
```

### Two Execution Modes

**Sequential** - Run combinations one after another (local/cluster)
```bash
python cli.py sweep --exp_id my_exp --sweep_mode combination
```

**Parallel** - Use SLURM job arrays for cluster parallelization
```bash
python cli.py sweep --exp_id my_exp --sweep_mode combination --parallel --cluster
```

## Project Structure

```
euler_sweep/
├── euler_sweep/                    # Core sweep framework
│   ├── cli.py                      # Command-line interface
│   ├── sweeper.py                  # Sweep logic & execution
│   └── sweep_worker.py             # Worker for parallel jobs
├── configs/                        # Configuration templates
│   ├── config_example.yaml         # Model/training config template
│   └── sweep_example.yaml          # Sweep definition template (copy to exp/sweeper/)
├── examples/                       # Working examples
├── scripts/                        # Cluster submission scripts
├── README.md                       # This file
├── QUICK_START.md                  # Quick reference
└── SWEEP_SKELETON_README.md        # Full documentation

# Your experiment structure:
experiments/my_exp/
├── config.yaml                     # Main configuration
└── sweeper/
    └── sweep.yaml                  # Sweep definition
```

## Usage Examples

```bash
# Local sequential sweep
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode independent

# Cluster parallel sweep with custom resources
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination \
    --parallel \
    --cluster \
    --max_concurrent_jobs 20 \
    --walltime "2-00:00:00" \
    --gpu_mem "48g"

# Dry run (generate scripts without submitting)
python euler_sweep/cli.py sweep \
    --exp_id my_exp \
    --sweep_mode combination \
    --parallel \
    --submit_jobs False
```

## Example Workflow

```bash
# 1. Create experiment with configs
mkdir -p experiments/arch_search/sweeper
# Edit experiments/arch_search/config.yaml
# Edit experiments/arch_search/sweeper/sweep.yaml

# 2. Start with independent sweep (quick exploration)
python cli.py sweep --exp_id arch_search --sweep_mode independent

# 3. Analyze results, narrow parameter ranges

# 4. Run combination sweep on promising ranges
python cli.py sweep --exp_id arch_search_refined --sweep_mode combination --parallel
```

## Sweep vs Optuna

| Feature | Sweep (This Template) | Optuna (euler_optuna) |
|---------|----------------------|----------------------|
| **Search Strategy** | Exhaustive grid | Adaptive sampling |
| **Values Tested** | All specified values | Algorithm decides |
| **Best For** | Small, specific grids | Large search spaces |
| **Reproducibility** | Exact parameter grid | Varies by seed |
| **Efficiency** | Tests everything | Focuses on promising areas |
| **Control** | Full control | Algorithm-driven |

**Use Both!** Start with sweeps for initial exploration, then use Optuna to fine-tune promising regions.

## Customization

### 1. Training Function (Required)

Your training function signature:
```python
def train_model(config, save_dir, data_dir, cluster, **kwargs):
    """
    Args:
        config: OmegaConf configuration with all parameters
        save_dir: Path to save results
        data_dir: Path to data
        cluster: bool, whether on cluster
    
    Returns:
        Any (typically metrics dict)
    """
    # Your training logic
    pass
```

### 2. Configuration Files (Required)

**config.yaml** - Define all parameters (in experiment root):
```yaml
model:
  hidden_dim: 128
  num_layers: 3
training:
  learning_rate: 0.001
  batch_size: 64
```

**sweeper/sweep.yaml** - Define what to sweep (in sweeper subdirectory):
```yaml
model:
  hidden_dim: [64, 128, 256]
training:
  learning_rate: [0.0001, 0.001, 0.01]
```

**Directory structure**:
```
experiments/my_exp/
├── config.yaml
└── sweeper/
    └── sweep.yaml
```

### 3. Paths (Optional)

Update in `cli.py`:
```python
ROOT_DIR = "your/project/root"
exp_dir = join(ROOT_DIR, "experiments", exp_id)
data_dir = join(ROOT_DIR, "data")
```

## Results

### Directory Structure

**Independent Sweep:**
```
experiments/my_exp/
├── config.yaml            # or config_*.yaml (flexible naming)
└── sweeper/
    ├── sweep.yaml
    └── runs/
        └── sweeps/
            ├── sweep_hidden_dim/
            │   ├── sweep_hidden_dim_64/
            │   │   ├── config.yaml
            │   │   └── results/
            │   └── sweep_hidden_dim_128/
            │       └── ...
            └── sweep_learning_rate/
                └── ...
```

**Combination Sweep:**
```
experiments/my_exp/
├── config.yaml            # or config_*.yaml (flexible naming)
└── sweeper/
    ├── sweep.yaml
    └── runs/
        └── combinations/
            ├── combo_hidden_dim_64_learning_rate_0.0001/
            │   ├── config.yaml
            │   └── results/
            ├── combo_hidden_dim_64_learning_rate_0.001/
            │   └── ...
            └── ...
```

**Parallel Sweep (additional files):**
```
experiments/my_exp/
└── sweeper/
    ├── sweep.yaml
    ├── combinations_data.json    # Metadata for all combinations
    ├── run_sweep_array.sh        # Generated SLURM script
    ├── job_id.txt               # Submitted job ID
    ├── slurm_logs/              # SLURM output logs
    │   ├── sweep_<jobid>_0.out
    │   └── ...
    └── runs/
        └── combinations/
            └── ...
```

## Cluster Usage

Supports efficient cluster execution with SCRATCH storage. Config files stay in HOME (persistent), training runs in SCRATCH (fast I/O), SLURM array jobs run combinations in parallel.

```bash
python cli.py sweep --exp_id my_exp --sweep_mode combination --parallel --cluster --scratch_path $SCRATCH/my_exp
```

See SWEEP_SKELETON_README.md for details.

## Requirements

Python 3.7+, OmegaConf, Click, SLURM (for parallel cluster execution)

## Related Templates

[euler_optuna](../euler_optuna/) - Adaptive hyperparameter optimization

## License

MIT License - Customize and use as needed.

For detailed documentation, see [SWEEP_SKELETON_README.md](SWEEP_SKELETON_README.md).
