# Boilerplate Trainer Example - Parameter Sweeps

This is the **simplest possible example** showing how to integrate the sweep framework into your project.

## 🎯 Overview

This example demonstrates:
- ✅ Minimal boilerplate trainer with synthetic data
- ✅ Simple parameter sweep configuration
- ✅ Local sequential sweep execution
- ✅ Fast execution (~2-3 minutes total)
- ✅ Clear result organization

**Focus**: Local sequential sweeps only (no cluster, no parallel complexity)

## 📁 Files

```
boiler_trainer/
├── trainer.py           # Minimal training function with synthetic data
├── cli.py               # Simple CLI for running sweeps
├── config.yaml          # Fast configuration (10 epochs)
├── sweeper/
│   └── sweep.yaml      # Small sweep definition (8 combinations)
└── README.md           # This file
```

## ⚡ Quick Start

### 1. Set up your experiment

```bash
# Create experiment directory structure
mkdir -p experiments/boiler_sweep/sweeper

# Copy config files
cp examples/boiler_trainer/config.yaml experiments/boiler_sweep/
cp examples/boiler_trainer/sweeper/sweep.yaml experiments/boiler_sweep/sweeper/
```

### 2. Run independent sweep (6 runs, ~1-2 minutes)

```bash
cd examples/boiler_trainer
python cli.py sweep --exp_id boiler_sweep --sweep_mode independent
```

**What happens:**
- Tests each parameter independently
- Creates 6 runs: 2 hidden_dim + 2 num_layers + 2 lr values
- Results in `experiments/boiler_sweep/sweeps/`

### 3. Run combination sweep (8 runs, ~2-3 minutes)

```bash
python cli.py sweep --exp_id boiler_sweep --sweep_mode combination
```

**What happens:**
- Tests all combinations of parameters
- Creates 8 runs: 2 × 2 × 2 combinations
- Results in `experiments/boiler_sweep/combinations/`

## 📊 Understanding the Results

### Independent Sweep Directory Structure

```
experiments/boiler_sweep/
├── config.yaml
├── sweeper/
│   └── sweep.yaml
└── sweeps/
    ├── sweep_hidden_dim/
    │   ├── sweep_hidden_dim_64/
    │   │   ├── config.yaml
    │   │   └── results.txt
    │   └── sweep_hidden_dim_128/
    │       ├── config.yaml
    │       └── results.txt
    ├── sweep_num_layers/
    │   └── ...
    └── sweep_lr/
        └── ...
```

### Combination Sweep Directory Structure

```
experiments/boiler_sweep/
├── config.yaml
├── sweeper/
│   └── sweep.yaml
└── combinations/
    ├── combo_hidden_dim_64_num_layers_2_lr_0.001/
    │   ├── config.yaml
    │   └── results.txt
    ├── combo_hidden_dim_64_num_layers_2_lr_0.01/
    │   ├── config.yaml
    │   └── results.txt
    └── ... (6 more combinations)
```

## 🔍 Understanding the Code

### trainer.py

The `simple_trainer()` function:
1. Uses synthetic data (no external data needed!)
2. Simulates realistic training dynamics
3. Metrics depend on hyperparameters
4. Fast execution (10 epochs)
5. Returns validation and training losses

**Key feature**: Loss is influenced by hyperparameters, so you can see which combinations work better!

### config.yaml

Defines default parameter values:
```yaml
model:
  hidden_dim: 128      # Will be swept: [64, 128]
  num_layers: 3        # Will be swept: [2, 3]

training:
  num_epochs: 10       # Fast for quick testing
  lr: 0.001           # Will be swept: [0.001, 0.01]
  dropout: 0.1
```

### sweeper/sweep.yaml

Defines parameter ranges to sweep:
```yaml
model:
  hidden_dim: [64, 128]        # 2 values
  num_layers: [2, 3]           # 2 values

training:
  lr: [0.001, 0.01]           # 2 values
```

**Result**: 2 × 2 × 2 = 8 combinations (very manageable!)

### cli.py

Simple integration showing:
```python
# 1. Import sweep framework
from euler_sweep.sweeper import run_sequential_sweep

# 2. Import your trainer
from trainer import simple_trainer

# 3. Call the sweep
run_sequential_sweep(
    exp_dir=exp_dir,
    sweep_mode=sweep_mode,
    train_fn=simple_trainer,  # Pass your trainer here!
    data_dir=data_dir,
    cluster=False
)
```

**That's it!** The framework handles all the rest.

## 🎓 How Sweeps Work

### Independent Sweep

Tests one parameter at a time:
1. Vary `hidden_dim`: [64, 128] → 2 runs
2. Vary `num_layers`: [2, 3] → 2 runs  
3. Vary `lr`: [0.001, 0.01] → 2 runs

**Total**: 6 runs, keeping other params at default

**Use when**: 
- Initial exploration
- Identifying important parameters
- Quick comparisons

### Combination Sweep

Tests all combinations:
```
hidden_dim=64, num_layers=2, lr=0.001
hidden_dim=64, num_layers=2, lr=0.01
hidden_dim=64, num_layers=3, lr=0.001
hidden_dim=64, num_layers=3, lr=0.01
hidden_dim=128, num_layers=2, lr=0.001
hidden_dim=128, num_layers=2, lr=0.01
hidden_dim=128, num_layers=3, lr=0.001
hidden_dim=128, num_layers=3, lr=0.01
```

**Total**: 8 runs

**Use when**:
- Finding optimal combinations
- Studying parameter interactions
- Final hyperparameter selection

## 🔧 Customizing for Your Project

### 1. Replace the Trainer

Modify `trainer.py`:
```python
def your_trainer(config, save_dir, data_dir, experiment_tag, cluster, **kwargs):
    # Load your real data
    train_loader = load_data(data_dir, config.training.batch_size)
    
    # Build your real model
    model = YourModel(
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers
    )
    
    # Train your model
    for epoch in range(config.training.num_epochs):
        loss = train_epoch(model, train_loader, config.training.lr)
    
   # Save and return metrics
    return {"val_loss": val_losses, "train_loss": train_losses}
```

### 2. Update Configuration

Edit `config.yaml` with your parameters:
```yaml
model:
  your_model_param: value

training:
  your_training_param: value
```

### 3. Define Your Sweep

Edit `sweeper/sweep.yaml`:
```yaml
model:
  your_model_param: [value1, value2, value3]

training:
  your_training_param: [value1, value2]
```

### 4. Update the CLI

In `cli.py`, import your trainer:
```python
from your_trainer import your_training_function

# Then pass it to run_sequential_sweep
run_sequential_sweep(
    exp_dir=exp_dir,
    sweep_mode=sweep_mode,
    train_fn=your_training_function,  # Your function here!
    ...
)
```

## 💡 Tips

### Start Small
- ✅ Use 2-3 values per parameter initially
- ✅ Keep training fast (fewer epochs)
- ✅ Test with independent sweep first
- ✅ Then scale up to combinations

### Organize Your Sweeps
```bash
experiments/
├── lr_sweep/          # Focus on learning rates
├── arch_sweep/        # Focus on architecture
└── final_sweep/       # Best combinations
```

### Beware of Combinatorial Explosion
- 3 params × 3 values = 3³ = 27 runs ✅ manageable
- 5 params × 5 values = 5⁵ = 3125 runs ❌ too many!

## 🆚 Comparison with Optuna Example

| Aspect | Sweep (This) | Optuna |
|--------|-------------|--------|
| **Selection** | Exhaustive grid | Adaptive sampling |
| **Config files** | config + sweeper/sweep | config + optuna_settings |
| **Define params** | In YAML | In Python code |
| **Execution** | Direct sweep | Create/resume/summary |
| **Best for** | Small grids | Large spaces |

## 📚 Next Steps

1. **Try it!** Run both sweep modes and examine results
2. **Modify sweeper/sweep.yaml** - Try different parameter ranges
3. **Compare results** - Which combinations work best?
4. **Scale up** - Add more parameters to sweep
5. **Integrate** - Replace with your actual trainer

## ✨ What Makes This Example Special

- ✅ **No external dependencies** - Synthetic data only
- ✅ **Fast execution** - Complete sweeps in minutes
- ✅ **Simple code** - ~100 lines total
- ✅ **Clear results** - Organized directory structure
- ✅ **Ready to adapt** - Easy to replace with your trainer

## 🎯 Learning Objectives

After running this example, you'll understand:
1. How to structure experiments with `sweeper/` subdirectory
2. How sweep configurations map to parameter values
3. Difference between independent and combination sweeps
4. How to integrate the sweep framework with your code
5. How results are organized and stored

---

**Happy Sweeping! 🚀**

For more advanced usage (cluster, parallel), see the main documentation in `euler_sweep/README.md`.
