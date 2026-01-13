# ProT Repository Restructure - Documentation

## Overview

The ProT repository has been restructured to cleanly separate the core transformer model from training infrastructure. This improves maintainability, reusability, and scalability.

## New Directory Structure

```
proT/
├── core/                          # Pure transformer model (architecture only)
│   ├── __init__.py
│   ├── model.py                   # ProT class
│   ├── utils.py                   # Model utilities
│   └── modules/                   # Transformer building blocks
│       ├── __init__.py
│       ├── attention.py           # Attention mechanisms
│       ├── encoder.py             # Encoder layers
│       ├── decoder.py             # Decoder layers
│       ├── embedding.py           # Embedding composition
│       ├── embedding_layers.py    # Embedding layer types
│       └── extra_layers.py        # Normalization, etc.
│
├── training/                      # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                 # Training orchestration
│   ├── dataloader.py              # Data loading
│   ├── experiment_control.py      # Experiment sweeps
│   ├── forecasters/               # Lightning model wrappers
│   │   ├── __init__.py
│   │   └── transformer_forecaster.py  # Simplified with AdamW
│   └── callbacks/                 # Training callbacks
│       ├── __init__.py
│       ├── training_callbacks.py  # Universal callbacks
│       └── model_callbacks.py     # Model-specific monitoring
│
├── utils/                         # Shared utilities
│   ├── entropy_utils.py
│   └── check_GPU.py
│
├── config/                        # Configuration files
├── labels.py                      # Constants
└── __init__.py
```

## Key Changes

### 1. Core Model (`proT/core/`)
- **Independent model architecture** - can be imported separately
- All transformer components in `modules/`
- Clean imports: `from proT.core import ProT`
- No training dependencies

### 2. Training Infrastructure (`proT/training/`)

#### Forecasters
- **Simplified `TransformerForecaster`**:
  - Uses only AdamW optimizer
  - Optional learning rate scheduler
  - Clean forward/step methods
  - Removed complex manual optimization modes

#### Callbacks Split
- **`training_callbacks.py`**: Universal training utilities
  - PerRunManifest
  - Checkpoint management
  - Memory logging
  - Best checkpoint tracking
  - K-fold tracking
  - Early stopping

- **`model_callbacks.py`**: Model-specific monitoring
  - GradientLogger
  - LayerRowStats
  - MetricsAggregator

### 3. Old Files Moved to `TO_DELETE/`
These files are preserved for reference but replaced by new structure:
- `transformer_modules/` → `core/modules/`
- `callbacks.py` → `training/callbacks/`
- `forecaster.py` → `training/forecasters/transformer_forecaster.py`
- `proT_model.py` → `core/model.py`
- `trainer.py` → `training/trainer.py`
- `dataloader.py` → `training/dataloader.py`
- `experiment_control.py` → `training/experiment_control.py`
- `cli.py` → (to be reorganized if needed)

## Usage Examples

### Import Core Model Only
```python
from proT.core import ProT

# Create model instance
model = ProT(**model_config)
```

### Import Training Components
```python
from proT.training import TransformerForecaster, trainer, ProcessDataModule

# Use forecaster
forecaster = TransformerForecaster(config)

# Or run training
trainer(config, data_dir, save_dir, cluster=False)
```

### Import Callbacks
```python
from proT.training.callbacks import (
    BestCheckpointCallback,
    GradientLogger,
    LayerRowStats
)
```

## Benefits

1. **Reusability**: Core model can be used independently for:
   - Inference
   - Fine-tuning
   - Integration in other projects

2. **Maintainability**: Clear separation of concerns:
   - Model architecture in `core/`
   - Training logic in `training/`

3. **Scalability**: Easy to:
   - Add new model architectures
   - Implement different training strategies
   - Create custom callbacks

4. **Testing**: Components can be tested independently

5. **Benchmarking**: Use same training infrastructure with different models

## Configuration Notes

### Optimizer Configuration
The new `TransformerForecaster` uses simplified optimization:
- **AdamW optimizer** by default
- Learning rate: `config["training"]["base_lr"]` (default: 1e-4)
- Weight decay: `config["training"]["weight_decay"]` (default: 0.01)
- Optional scheduler: Set `config["training"]["use_scheduler"] = True`

### Removed Configuration Options
The following optimization modes (1-7) have been removed for simplicity:
- Complex multi-optimizer setups
- Manual optimization with optimizer switching
- Phase-based training schedules

If you need these features for benchmarking, they can be found in `TO_DELETE/forecaster.py`.

## Migration Guide

### For Existing Experiments
1. Configurations should work with minimal changes
2. Update `optimization` setting to use default AdamW (or remove)
3. Run a test training to verify

### For Notebooks
Update imports:
```python
# Old
from proT.proT_model import ProT

# New
from proT.core import ProT
```

### For Custom Training Scripts
```python
# Old
from proT.forecaster import TransformerForecaster

# New
from proT.training.forecasters import TransformerForecaster
```

## Testing Checklist

- [x] Core model imports work
- [x] Training runs successfully
- [ ] All callbacks function correctly
- [ ] Metrics are logged properly
- [ ] K-fold cross-validation works
- [ ] Checkpoints save/load correctly

## Next Steps

1. Test with your existing configurations
2. Verify all functionality works as expected
3. Delete `TO_DELETE/` folder once confident
4. Update any external scripts/notebooks that import from proT

## Questions or Issues?

If you encounter any problems with the new structure, refer to the old files in `TO_DELETE/` for comparison.
