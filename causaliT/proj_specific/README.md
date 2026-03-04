# Project-Specific Modules

This package contains dataset-specific and project-specific functions that are not general enough to be part of the core framework, but are needed for particular experiments or datasets.

## In-Context Masks (dyconex_masks.py)

These functions generate attention masks computed **online** (per-batch) from data features, as opposed to static hard masks loaded from files. They are particularly useful for datasets where the sequence structure varies between samples.

### Available Functions

#### `build_causal_order_mask(query_order, key_order)`

Builds a causal mask based on the "order" feature from the data.

**Logic**: A query token at position `i` can only attend to key tokens at positions where `order[key] <= order[query]`.

```python
# Example
query_order = torch.tensor([[1, 2, 3]])  # (B, L_q)
key_order = torch.tensor([[1, 2, 3]])    # (B, L_k)

mask = build_causal_order_mask(query_order, key_order)
# Result: [[1, 0, 0],
#          [1, 1, 0],
#          [1, 1, 1]]
# Order 1 can see order 1
# Order 2 can see orders 1, 2
# Order 3 can see orders 1, 2, 3
```

#### `build_category_cross_mask(query_features, key_features, category_indices)`

Builds a category-based mask for cross-attention. Tokens can only attend to other tokens that share the same category.

**Category definition**: A tuple of feature values at specified indices. For dyconex, this is `{process, occurrence, step}`.

```python
# Example for dyconex dataset
# Category is defined by indices [1, 2, 3] = (process, occurrence, step)
category_indices = [1, 2, 3]

mask = build_category_cross_mask(X_tensor, S_tensor, category_indices)
# X tokens can only attend to S tokens with same (process, occurrence, step)
```

#### `build_dyconex_in_context_masks(S, X, Y, config)`

Master function that builds all masks for the dyconex dataset based on configuration.

**Returns**: Dictionary with keys `dec1_cross`, `dec1_self`, `dec2_cross`, `dec2_self`.

```python
config = {
    'dec1_cross': {
        'causal': True,
        'category': True,
        'order_idx_q': 6,
        'order_idx_k': 6,
        'category_idx': [1, 2, 3]
    },
    'dec1_self': {
        'causal': True,
        'order_idx': 6
    },
    # ...
}

masks = build_dyconex_in_context_masks(S, X, Y, config)
```

#### `merge_masks(static_masks, in_context_masks)`

Merges static hard masks (from files) with in-context masks (computed from data).

When both masks are present for the same attention layer, they are combined using element-wise multiplication (logical AND).

### Configuration in YAML

```yaml
training:
  use_in_context_masks: true
  
  in_context_mask_config:
    # Decoder 1 Cross-Attention: X queries S
    dec1_cross:
      causal: true              # Use causal order mask
      category: true            # Use category mask (only for dec1_cross)
      order_idx_q: 6            # Index of order feature in X
      order_idx_k: 6            # Index of order feature in S
      category_idx: [1, 2, 3]   # Indices of (process, occurrence, step)
    
    # Decoder 1 Self-Attention: X self-attention
    dec1_self:
      causal: true
      order_idx: 6
    
    # Decoder 2 Cross-Attention: Y queries X
    dec2_cross:
      causal: true
      order_idx_q: 1            # Y uses 'position' instead of 'order'
      order_idx_k: 6
    
    # Decoder 2 Self-Attention: Y self-attention
    dec2_self:
      causal: true
      order_idx: 1
```

### Use with Forecaster

The `StageCausalForecaster` automatically handles in-context masks:

1. Reads configuration from `config['training']['use_in_context_masks']`
2. Computes masks in the `forward()` method using batch data
3. Merges with any static hard masks
4. Passes final masks to the model's attention layers

You can also override mask usage at inference time:

```python
# Use default (from config) - masks enabled if trained with them
pred_x, pred_y, *_ = forecaster.forward(S, X, Y)

# Disable hard masks for ablation studies (useful during inference)
pred_x, pred_y, *_ = forecaster.forward(S, X, Y, disable_hard_masks=True)

# Disable in-context masks for ablation studies
pred_x, pred_y, *_ = forecaster.forward(S, X, Y, disable_in_context_masks=True)

# Disable both mask types for full ablation
pred_x, pred_y, *_ = forecaster.forward(S, X, Y, disable_hard_masks=True, disable_in_context_masks=True)
```

**Note on parameter naming**: The `disable_hard_masks` and `disable_in_context_masks` parameters
are used during **inference** to optionally disable masks that the model was trained with. 
The class attributes `self.use_hard_masks` and `self.use_in_context_masks` (set from config)
control whether masks are used during **training**.

### Mask Semantics

- **Mask value `1`**: Attention allowed
- **Mask value `0`**: Attention blocked

The masks are applied to attention scores before softmax (for softmax-based attention) by converting to additive masks where `0 -> -inf`.

### NaN Handling

- NaN values in `order` features are treated as infinity (blocked from attending)
- NaN values in category features are treated as unique values that don't match anything
