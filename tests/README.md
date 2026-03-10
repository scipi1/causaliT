# CausaliT Tests

## 1. Training Tests

Quick test suite to verify all model configurations can train successfully.

### Quick Start

```bash
# List available models (default: experiments/)
python tests/test_training_models.py --list

# Test a single model (fast)
python tests/test_training_models.py --models-dir experiments/single/scm6 --model single_Lie_CC_scm6

# Run all training tests with pytest
pytest tests/test_training_models.py -v
```

### Specifying Models Directory

The default models directory is `experiments/`. You can specify a subdirectory:

```bash
# Test models in a specific subdirectory
python tests/test_training_models.py --models-dir experiments/single/scm6 --list

# Or with pytest
pytest tests/test_training_models.py --models-dir=experiments/single/scm6 -v

# Or via environment variable
CAUSALT_MODELS_DIR=experiments/single/scm6 pytest tests/test_training_models.py -v
```

### Recursive Search

Use the `-r` flag to search recursively through all subdirectories:

```bash
# List all models recursively from experiments root
python tests/test_training_models.py --list -r

# List all models recursively from a specific subdirectory
python tests/test_training_models.py --models-dir experiments/single --list -r
```

### Test Configuration

Tests run with reduced parameters for speed:
- `max_epochs = 1`
- `k_fold = 3`
- `batch_size = 32`

**Note:** Original config files are never modified.

---

## 2. Naming Consistency Tests

Validates that experiment folder names match the config file parameters.

### Naming Convention

```
forecaster_SelfAttentionClass_CrossAttentionClass_dataset_[PhiParametrization]_[embeddingsComposition]_[hard]
```

Example names:
- `single_Lie_CC_scm6` → basic experiment
- `single_PhiSM_PhiSM_scm6_antisym` → with antisymmetric phi
- `single_SM_SM_scm6_SVFA` → with SVFA embeddings
- `single_Toeplitz_CC_scm6_gated` → with gated phi

### Quick Start

```bash
# Check all experiments
python tests/test_naming_consistency.py

# Check a specific experiment
python tests/test_naming_consistency.py --experiment single_Lie_CC_scm6

# Run with pytest
pytest tests/test_naming_consistency.py -v
```

### What It Checks

| Component | Config Key | Options |
|-----------|-----------|---------|
| forecaster | `model.model_object` | single→SingleCausalLayer |
| SelfAttention | `model.kwargs.dec_self_attention_type` | Lie, PhiSM, SM, Toeplitz |
| CrossAttention | `model.kwargs.dec_cross_attention_type` | CC, PhiSM, SM |
| dataset | `data.dataset` | scm6, scm7, etc. |
| PhiParam (optional) | `model.kwargs.dag_parameterization_self` | antisym, gated, indep |
| embeddings (optional) | `model.kwargs.comps_embed_X` | SVFA (default: summation) |
| hard (optional) | `training.use_hard_masks` | hard (default: false) |

**Note:** This test does NOT modify config files - it only reports inconsistencies.
