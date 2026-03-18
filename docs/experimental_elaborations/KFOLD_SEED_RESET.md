# K-Fold Cross-Validation Seed Reset

## Issue

When running SVFA (Structure-Value Factorized Attention) experiments with k-fold cross-validation, different folds would learn different DAGs even with the same seed setting. This was particularly noticeable when comparing:
- Cosine similarities of embeddings at the first epoch (different starting values across folds)
- Learned DAG structures (different attention patterns across folds)

## Root Cause

The issue was traced to how PyTorch's random number generator (RNG) state is consumed during training:

1. **`seed_everything(seed)` was called once** at the beginning of training
2. **`nn.Embedding` weights are initialized from N(0,1)** using PyTorch's RNG
3. **Training consumes many random numbers** (dropout, data shuffling, etc.)
4. **By fold N**, the RNG state has been consumed differently → different model initialization

```
Timeline:
├── seed_everything(42)           → RNG at position 0
├── Fold 0:
│   ├── create_model_instance()   → Uses RNG at position 0
│   ├── Training (dropout, shuffle, etc.) → Consumes M random numbers
│   └── End of fold               → RNG at position M
├── Fold 1:
│   ├── create_model_instance()   → Uses RNG at position M (NOT 0!)
│   └── Different initialization than Fold 0!
```

## Affected Components

| Component | Type | Affected |
|-----------|------|----------|
| `embedding_S` | `OrthogonalMaskEmbedding` | ✓ (contains `nn.Linear`) |
| `embedding_X` | `ModularEmbedding` | ✓ (contains `nn.Embedding`) |
| Attention projections | `nn.Linear` | ✓ |
| Feed-forward layers | `nn.Linear` | ✓ |
| DAG mask (phi) | Parameters | ✓ (random init with `init_std`) |

## Solution

Add `seed_everything(seed)` **before each fold's model creation** in `trainer.py`:

```python
for fold, (train_local_idx, val_local_idx) in enumerate(kfold.split(train_val_idx)):
    
    # RESET SEED BEFORE MODEL CREATION
    seed_everything(seed)
    
    model = create_model_instance(config, data_dir)
    # ... rest of fold training ...
```

## Why This Works

1. **KFold splits are controlled by NumPy's RNG** (`random_state=seed`), which is separate from PyTorch's RNG
2. **Resetting PyTorch's RNG** before model creation ensures identical initialization
3. **Data splits remain different** across folds (as intended for cross-validation)

## Result

| Property | Before Fix | After Fix |
|----------|------------|-----------|
| Model initialization | Different per fold | Identical per fold |
| Data splits | Different per fold | Different per fold |
| Initial cosine similarities | Different | Identical |
| DAG learning | Confounded by init | Fair comparison |

## Diagnostic Script

A diagnostic script is available at `scripts/diagnose_fold_initialization.py` that demonstrates:
1. `nn.Embedding` random initialization behavior
2. `OrthogonalMaskEmbedding` initialization
3. K-fold simulation comparing current vs. fixed behavior
4. Initial cosine similarity differences
5. RNG state consumption analysis

Run with:
```bash
python scripts/diagnose_fold_initialization.py
```

## Key Insight for SVFA

SVFA aims to make attention patterns sample-independent by separating structure embeddings (for Q, K) from value embeddings (for V). However:

- **Structure embeddings still contain learnable parameters** (e.g., `nn.Linear` in `OrthogonalMaskEmbedding.value_embedding`)
- **These are randomly initialized**, so different initialization → different attention scores at epoch 0
- **The seed reset ensures all folds start from the same point**, making cross-fold comparisons valid

## Date

Fix implemented: 2026-03-18
