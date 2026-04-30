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

### Four-Stage Pipeline

| Stage | Goal | Process |
|-------|------|---------|
| **0. Calibration** | Balance gradients separately for cross & self HSIC | Binary search for λ_group where min(ratio_cross, ratio_self) ≈ 1 |
| **1. Causal Init** | Initialize toward causal structure | Train with calibrated high HSIC (λ_hsic × multiplier × boost) |
| **2. Score Sparsity CV** | Select optimal λ_score for DAG sparsity | k-fold CV over λ_score candidates; select by min val HSIC or val recon |
| **3. Main Training** | Fit data while preserving structure | Standard training with HSIC annealing and selected λ_score |

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
├── score_sparsity_cv.py     # Stage 2: λ_score selection via k-fold CV
├── config_operations.py     # Pure config transforms between stages
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
  use_score_sparsity_cv: true
  
  # Stage 0
  calibration_epochs: 10
  
  # Stage 1
  causal_init_epochs: 20
  causal_init_hsic_multiplier: 10.0
  
  # Stage 2
  score_sparsity_lambda_candidates: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
  score_sparsity_cv_folds: 5
  score_sparsity_cv_epochs: 20
  score_sparsity_selection_rule: "min_hsic"  # or "min_recon" for baselines

training:
  lambda_group_l1: 0.0      # Set by calibration, or manually
  log_group_l1: true
```

Each stage is independently toggleable. All 8 combinations are valid:
- Score sparsity CV alone: `use_score_sparsity_cv: true` (others false)
- Full pipeline: all three stages enabled
- Skip CV: `use_score_sparsity_cv: false` (uses `training.lambda_*_score_sparse`)

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_group_l1` | 0.0 | Group LASSO weight (calibrated or manual) |
| `calibration_epochs` | 10 | Epochs per calibration trial |
| `calibration_balance_threshold` | 2.0 | Gradient ratio tolerance |
| `causal_init_epochs` | 20 | HSIC-dominated pre-training epochs |
| `causal_init_hsic_multiplier` | 10.0 | λ_hsic multiplier for Stage 1 |
| `score_sparsity_lambda_candidates` | [0.0, ..., 0.1] | Grid of λ_score values to try |
| `score_sparsity_cv_folds` | 5 | Number of CV folds |
| `score_sparsity_cv_epochs` | 20 | Epochs per CV fold |
| `score_sparsity_selection_rule` | "min_hsic" | "min_hsic" or "min_recon" |
| `lambda_score_suggested` | null | Auto-populated by Stage 2 |

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
├── score_sparsity_cv/
│   ├── score_sparsity_cv_summary.json   # Best λ_score + both selection criteria
│   ├── lambda_0.0000/                   # Per-lambda results
│   │   └── cv_result.json
│   ├── lambda_0.0010/
│   │   └── cv_result.json
│   └── ...
└── staged_training_summary.json
```

### Score Sparsity CV Summary

The `score_sparsity_cv_summary.json` always reports **both** selection criteria
for easy comparison, regardless of which rule was used:

```json
{
  "best_lambda_score": 0.01,
  "primary_criterion": "min_val_hsic",
  "selection_by_hsic": {"lambda_score": 0.01, "mean_val_hsic": 0.042, ...},
  "selection_by_recon": {"lambda_score": 0.001, "mean_val_recon": 0.015, ...},
  "per_lambda_results": [...]
}
```

## Alternatives: Sweep-Based λ_score Selection

For more detailed analysis (LASSO-path plots, variable importance paths),
use the `euler_sweep` approach described in `docs/SCORE_SPARSITY_GUIDE.md`.
The CV-based Stage 2 is lighter-weight and integrates directly into the
staged pipeline, while the sweep approach gives richer diagnostic outputs.
