# Staged Training for Causal Structure Learning

## The Problem: Flat HSIC Landscape

When the model has too much capacity (large d_model), the HSIC regularizer becomes ineffective: any parameter configuration achieves low HSIC because the model can fit the data perfectly while ignoring the causal structure.

**Key insight**: For HSIC to guide causal structure learning, the model must be capacity-constrained such that only the *true* causal mechanism achieves the HSIC global minimum.

## Method

### Group LASSO (L2,1 Regularization)

We constrain model capacity using **Group L1** (L2,1 norm) on embedding columns:

$$||W||_{2,1} = \sum_j ||W[:,j]||_2$$

- **L2 within columns**: Keeps values grouped together
- **L1 over column norms**: Induces column-level sparsity (LASSO effect)

Unlike element-wise L1, Group L1 zeroes out *entire* embedding dimensions, effectively reducing d_model and creating an information bottleneck.

**References**: Yuan & Lin (2006), Argyriou et al. (2008), Nie et al. (2010)

### Three-Stage Pipeline

| Stage | Goal | Process |
|-------|------|---------|
| **0. Calibration** | Balance gradients separately for cross & self HSIC | Binary search for λ_group where min(ratio_cross, ratio_self) ≈ 1 |
| **1. Causal Init** | Initialize toward causal structure | Train with calibrated high HSIC (λ_hsic × multiplier × boost) |
| **2. Main Training** | Fit data while preserving structure | Standard training with HSIC annealing |

### Separate HSIC Calibration

**IMPORTANT**: The calibration stage tracks **cross-attention** (S→X) and **self-attention** (X→X) HSIC gradients **separately**:

```
ratio_cross = ||∇Recon||_F / ||∇HSIC_cross||_F
ratio_self  = ||∇Recon||_F / ||∇HSIC_self||_F
```

**Why separate?** Different attention pathways have different gradient magnitudes. Aggregating them risks one signal drowning out the other. Separate tracking ensures both causal signals are properly balanced.

**Convergence criterion**: `min(ratio_cross, ratio_self) ∈ [1/threshold, threshold]`

**Output**: Two separate multipliers:
- `lambda_hsic_cross_multiplier`: Scale λ_hsic_cross by this to balance with reconstruction
- `lambda_hsic_self_multiplier`: Scale λ_hsic_self by this to balance with reconstruction

## Implementation

### Files

```
causaliT/training/
├── calibration.py           # Stage 0: gradient balance calibration
├── causal_initialization.py # Stage 1: HSIC-dominated pre-training
└── staged_trainer.py        # Orchestrates full pipeline
```

### Usage

```python
from causaliT.training import staged_trainer

df = staged_trainer(config, data_dir, save_dir, cluster=False)
```

### Config

```yaml
staged_training:
  use_calibration: true
  use_causal_init: true
  calibration_epochs: 10
  causal_init_epochs: 20
  causal_init_hsic_multiplier: 10.0

training:
  lambda_group_l1: 0.0      # Set by calibration, or manually
  log_group_l1: true
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_group_l1` | 0.0 | Group LASSO weight (calibrated or manual) |
| `calibration_epochs` | 10 | Epochs per calibration trial |
| `calibration_balance_threshold` | 2.0 | Gradient ratio tolerance |
| `causal_init_epochs` | 20 | HSIC-dominated pre-training epochs |
| `causal_init_hsic_multiplier` | 10.0 | λ_hsic multiplier for Stage 1 |

## When to Use

- Model has too much capacity (d_model >> needed)
- HSIC doesn't correlate with SHD across seeds
- DAG recovery is inconsistent despite good reconstruction

## Output

Each stage saves checkpoints and summaries:
```
save_dir/
├── calibration/
│   ├── calibration_summary.json
│   └── lambda_*/checkpoint.ckpt
├── causal_init/
│   ├── causal_init_summary.json
│   └── causal_init_checkpoint.ckpt
└── staged_training_summary.json
```
