# Evaluation Module - Prediction System

## Overview

The evaluation module provides a flexible, object-oriented prediction system for trained transformer models.

## Architecture

### Key Components

1. **PredictionResult**: A dataclass that encapsulates all prediction outputs
2. **BasePredictor**: Abstract base class defining the prediction interface
3. **TransformerPredictor**: Handles transformer-based forecasting models
4. **create_predictor()**: Factory function that automatically selects the correct predictor

## Quick Start

### Basic Usage

```python
from omegaconf import OmegaConf
from proT.evaluation import predict_test_from_ckpt

# Load configuration
config = OmegaConf.load("path/to/config.yaml")

# Run prediction
results = predict_test_from_ckpt(
    config=config,
    datadir_path="path/to/data",
    checkpoint_path="path/to/checkpoint.ckpt",
    dataset_label="test"
)

# Access results
print(f"Predictions shape: {results.outputs.shape}")
print(f"Targets shape: {results.targets.shape}")
print(f"Model type: {results.metadata['model_type']}")

# Check if attention weights are available (only for Transformer models)
if results.attention_weights is not None:
    cross_att = results.attention_weights['cross']
    enc_att = results.attention_weights['encoder']
    dec_att = results.attention_weights['decoder']
```

### Advanced Usage with Custom Predictor

```python
from proT.evaluation import create_predictor

# Create predictor (auto-selects correct type based on config)
predictor = create_predictor(config, checkpoint_path, datadir_path)

# Create data module
dm = predictor.create_data_module(cluster=False)

# Run prediction with custom settings
results = predictor.predict(
    dm=dm,
    dataset_label="test",
    debug_flag=False,
    show_trg_max_idx=50  # For Transformer models
)

# Access individual components
inputs = results.inputs
outputs = results.outputs
targets = results.targets
```

## PredictionResult Structure

```python
@dataclass
class PredictionResult:
    inputs: np.ndarray              # Input data (B x L x F)
    outputs: np.ndarray             # Model predictions (B x L) or (B x L x F')
    targets: np.ndarray             # Target data (B x L x F)
    attention_weights: Optional[Dict[str, np.ndarray]]  # Attention weights (Transformer only)
    metadata: Optional[Dict[str, Any]]  # Additional information
```

### Attention Weights Dictionary

For Transformer models, `attention_weights` contains:
- `'encoder'`: Encoder self-attention weights
- `'decoder'`: Decoder self-attention weights
- `'cross'`: Decoder cross-attention weights

## Supported Models

Currently supports transformer-based forecasting models (proT).

## Examples

### Example 1: Predicting on Test Set

```python
from omegaconf import OmegaConf
from proT.evaluation import predict_test_from_ckpt

config = OmegaConf.load("config.yaml")

results = predict_test_from_ckpt(
    config=config,
    datadir_path="../data/input",
    checkpoint_path="../experiments/model/checkpoint.ckpt",
    dataset_label="test"
)

print(f"Number of samples: {results.metadata['num_samples']}")
print(f"Model: {results.metadata['model_type']}")
```

### Example 2: Using External Dataset

```python
external_dataset = {
    "dataset": "custom_dataset",
    "filename_input": "X.npy",
    "filename_target": "Y.npy",
}

results = predict_test_from_ckpt(
    config=config,
    datadir_path="../data/input",
    checkpoint_path="../experiments/model/checkpoint.ckpt",
    external_dataset=external_dataset,
    dataset_label="all"
)
```

### Example 3: Analyzing Attention Patterns (Transformer Only)

```python
results = predict_test_from_ckpt(config, datadir_path, checkpoint_path)

if results.attention_weights is not None:
    import matplotlib.pyplot as plt
    
    # Plot cross-attention for first sample
    cross_att = results.attention_weights['cross'][0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cross_att, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Encoder Position')
    plt.ylabel('Decoder Position')
    plt.title('Cross-Attention Heatmap')
    plt.savefig('attention_heatmap.png')
```

### Example 4: Computing Metrics

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = predict_test_from_ckpt(config, datadir_path, checkpoint_path)

# Flatten arrays for metric computation
outputs_flat = results.outputs.flatten()
targets_flat = results.targets.flatten()

# Remove NaN values
mask = ~np.isnan(targets_flat)
outputs_clean = outputs_flat[mask]
targets_clean = targets_flat[mask]

# Compute metrics
mae = mean_absolute_error(targets_clean, outputs_clean)
rmse = np.sqrt(mean_squared_error(targets_clean, outputs_clean))
r2 = r2_score(targets_clean, outputs_clean)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

### Example 5: Batch Processing Multiple Checkpoints

```python
import os
from pathlib import Path

experiment_dir = Path("../experiments/model/")
checkpoints = sorted(experiment_dir.glob("k_*/checkpoints/*.ckpt"))

all_results = []
for ckpt in checkpoints:
    print(f"Processing {ckpt.name}...")
    results = predict_test_from_ckpt(
        config=config,
        datadir_path="../data/input",
        checkpoint_path=str(ckpt),
        dataset_label="test"
    )
    all_results.append(results)

# Aggregate results across checkpoints
mean_output = np.mean([r.outputs for r in all_results], axis=0)
std_output = np.std([r.outputs for r in all_results], axis=0)
```

## Extending the System

### Creating a Custom Prediction Strategy

To add custom prediction behaviors (e.g., input masking, noise injection):

```python
from proT.evaluation.predictors import BasePredictor

class MaskedInputPredictor(TransformerPredictor):
    def __init__(self, config, checkpoint_path, datadir_path, mask_ratio=0.2):
        super().__init__(config, checkpoint_path, datadir_path)
        self.mask_ratio = mask_ratio
    
    def _forward(self, X, trg, **kwargs):
        # Apply masking to input
        mask = torch.rand_like(X) < self.mask_ratio
        X_masked = X.clone()
        X_masked[mask] = 0
        
        # Call parent forward with masked input
        return super()._forward(X_masked, trg, **kwargs)
```

### Adding Post-Processing

```python
class PredictionAnalyzer:
    """Utility class for analyzing prediction results"""
    
    def __init__(self, results: PredictionResult):
        self.results = results
    
    def compute_metrics(self):
        """Compute common metrics"""
        # Implementation here
        pass
    
    def plot_predictions(self, sample_idx=0):
        """Plot predictions vs targets"""
        # Implementation here
        pass
    
    def analyze_attention(self):
        """Analyze attention patterns (Transformer only)"""
        if self.results.attention_weights is None:
            raise ValueError("No attention weights available")
        # Implementation here
        pass
```

## Backward Compatibility

The legacy `predict()` function is still available for backward compatibility with existing notebooks:

```python
from proT.evaluation import predict

# Old API (returns tuple)
input_array, output_array, target_array, cross_att, enc_att, dec_att = predict(
    model=model,
    dm=dm,
    dataset_label="test"
)
```

However, new code should use `predict_test_from_ckpt()` which returns a `PredictionResult` object.

## Migration Guide

### Migrating from Old API to New API

**Before:**
```python
input_array, output_array, target_array, cross_att, enc_att, dec_att = predict_test_from_ckpt(...)
```

**After:**
```python
results = predict_test_from_ckpt(...)
input_array = results.inputs
output_array = results.outputs
target_array = results.targets
cross_att = results.attention_weights['cross'] if results.attention_weights else None
enc_att = results.attention_weights['encoder'] if results.attention_weights else None
dec_att = results.attention_weights['decoder'] if results.attention_weights else None
```

## Best Practices

1. **Use the factory function**: Let `create_predictor()` handle model type selection
2. **Check attention availability**: Always check if `attention_weights` is not None before accessing
3. **Use metadata**: Leverage the metadata field for tracking prediction context
4. **Handle NaN values**: Remember to filter NaN values when computing metrics
5. **Batch processing**: For multiple checkpoints, process in batches to manage memory

## Troubleshooting

### Common Issues

**Issue**: `ValueError: Unknown model type`
- **Solution**: Ensure your config file has a valid `model.model_object` field

**Issue**: `RuntimeError: Model failed to load from checkpoint`
- **Solution**: Verify checkpoint path and ensure it matches the model architecture in config

**Issue**: `AttributeError: 'NoneType' object has no attribute 'cross'`
- **Solution**: Check if model has attention weights before accessing (baseline models return None)

## Future Enhancements

Potential additions to the prediction system:
- Prediction strategies module (masking, noise injection, etc.)
- Post-processing utilities (metrics calculator, visualization helpers)
- Ensemble prediction support
- Uncertainty quantification
- Real-time prediction streaming
