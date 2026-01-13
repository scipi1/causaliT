# Gradient Jacobian Logger

## Overview

The `GradientJacobianLogger` callback computes and saves Jacobian matrices during model training. It calculates the gradients of the output tensor with respect to the **input tensor before embedding**, allowing you to analyze how input perturbations affect model predictions.

## Configuration

Add these parameters to your config YAML file under the `training` section:

```yaml
training:
  log_jacobian: true              # Enable/disable Jacobian logging
  jacobian_every_n_epochs: 5      # Compute gradients every N epochs
```

## Output

When enabled, the callback will:

1. Compute Jacobians during **validation batches only** (not training to reduce overhead)
2. Save results every N epochs to: `{experiment_dir}/k_{fold}/gradients/jacobian_epoch_{XXXX}.npz`
3. Store Jacobians in their original shape as returned by `torch.autograd.functional.jacobian`

### File Structure

```
experiments/your_experiment/
├── k_0/
│   ├── gradients/
│   │   ├── jacobian_epoch_0000.npz
│   │   ├── jacobian_epoch_0005.npz
│   │   └── jacobian_epoch_0010.npz
│   └── ...
├── k_1/
│   └── gradients/
│       └── ...
```

Each `.npz` file contains:
- Keys: `batch_0`, `batch_1`, ..., `batch_N`
- Values: Jacobian matrices in shape `(B, Y_seq_len, out_dim, X_seq_len, features)`

## Loading Saved Gradients

```python
import numpy as np

# Load a specific epoch's gradients
data = np.load('gradients/jacobian_epoch_0005.npz')

# Access specific batch
batch_0_jacobian = data['batch_0']
print(f"Shape: {batch_0_jacobian.shape}")
# Shape: (B, Y_seq_len, out_dim, X_seq_len, features)

# List all batches
print(f"Available batches: {data.files}")
```

## Jacobian Shape Explanation

For a model with:
- Input: `(B, X_seq_len, features)` 
- Output: `(B, Y_seq_len, out_dim)`

The saved Jacobian has shape `(B, Y_seq_len, out_dim, X_seq_len, features)` where:
- `jacobian[b, t_out, s, t_in, d]` = ∂output[b, t_out, s] / ∂input[b, t_in, d]

This is the **diagonal Jacobian** - only same-batch gradients (no cross-batch terms). This is much more efficient than computing the full Jacobian.

## Post-Processing

The gradients are already in a convenient shape. Examples:

```python
# Access gradient for specific sample
sample_0_grad = batch_0_jacobian[0]
# Shape: (Y_seq_len, out_dim, X_seq_len, features)

# Rearrange to: (X_seq_len, features, Y_seq_len, out_dim) if needed
sample_0_reshaped = sample_0_grad.transpose(2, 3, 0, 1)

# Check which input features most influence a specific output
t_out = 0  # output time step
s = 0      # output dimension
input_importance = np.abs(sample_0_grad[t_out, s, :, :])
# Shape: (X_seq_len, features) - importance of each input feature
```

## Performance Considerations

- **Memory**: For batch_size=64, X_seq_len=3, features=2, Y_seq_len=3, out_dim=1:
  - ~4.5 KB per batch (very manageable)
  - ~225 KB per epoch (with ~50 validation batches)

- **Computation Time**: Jacobian computation adds overhead. Use `jacobian_every_n_epochs` to balance between:
  - Frequent updates (every epoch) → More data, slower training
  - Sparse updates (every 10+ epochs) → Less overhead, coarser temporal resolution

## Disabling

To disable gradient logging, set in your config:

```yaml
training:
  log_jacobian: false
```

## Implementation Details

- Uses `torch.autograd.functional.jacobian` for automatic differentiation
- Computes gradients with respect to raw input (before embedding layer)
- Only runs during validation phase to avoid training slowdown
- Automatically creates `gradients/` subdirectory in experiment folder
- Saves with compression using `np.savez_compressed`
