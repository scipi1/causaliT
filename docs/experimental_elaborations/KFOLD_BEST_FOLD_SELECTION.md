# K-Fold Best Fold Selection Strategy

## Overview

When using k-fold cross-validation, we need to select a single model from k trained models. This document describes the selection strategy implemented in `KFoldResultsTracker`.

## Selection Strategy: Causal-First

The default selection prioritizes **causal quality** over prediction performance:

1. **Primary**: Minimum `val_hsic_reg` (HSIC regularization value)
2. **Fallback 1**: Minimum `val_hsic_cross` (S→X independence)
3. **Fallback 2**: Minimum `val_hsic` (general HSIC)
4. **Fallback 3**: Minimum `val_mae` (prediction-based)
5. **Last Resort**: First available fold

## Rationale

### Why HSIC instead of Prediction Metrics?

The goal of causal structure learning is to recover the correct DAG, not to minimize prediction error. Low HSIC indicates that:
- Residuals are independent of parent variables
- The model has captured the causal relationships correctly
- The learned attention pattern approximates the true causal structure

### Empirical Justification

Analysis of d_model sweeps (see `notebooks/nb_eval_sweep_dmodel.ipynb`) revealed:

1. **Within-seed clustering**: CV fold results are clustered together within a seed
2. **HSIC-SHD correlation**: Minimum validation HSIC rarely corresponds to worst SHD
3. **Seed dominates**: Model initialization (seed) has more impact than data split

This means:
- Cross-validation variance is small compared to seed variance
- Selecting by HSIC is a reliable proxy for causal quality
- We can report std across seeds as the "worst case" variance estimate

## Implementation

```python
class KFoldResultsTracker:
    # HSIC metrics to try, in order of preference
    HSIC_METRICS_PRIORITY = ["val_hsic_reg", "val_hsic_cross", "val_hsic"]
    
    def _select_best_fold(self, metric_names: list) -> dict:
        # Try HSIC-based selection first (causal inference default)
        hsic_metric = self._find_hsic_metric(metric_names)
        
        if hsic_metric is not None:
            # Select fold with minimum HSIC
            best_fold = min(valid_folds,
                           key=lambda x: self.fold_results[x]["metrics"][hsic_metric])
            return {
                "fold_number": best_fold,
                "selection_criterion": hsic_metric,
                ...
            }
        
        # Fallback to val_mae
        ...
```

## Output Format

The `kfold_summary.json` now includes the selection criterion:

```json
{
  "best_fold": {
    "fold_number": 2,
    "selection_criterion": "val_hsic_reg",
    "selection_value": 0.0012,
    "metrics": { ... },
    "checkpoint_path": "k_2/checkpoints/best_checkpoint.ckpt"
  }
}
```

## Enabling HSIC Logging

For the selection to work, HSIC must be logged during training. In config:

```yaml
training:
  log_hsic: true  # Required for HSIC-based fold selection
```

If `log_hsic: false` and no HSIC lambdas are set, the tracker will fall back to `val_mae`.

## Compatibility

This is the default behavior for all models in causaliT:
- `SingleCausalForecaster`
- `NoiseAwareSingleCausalForecaster`
- `StageCausalForecaster`
- `TransformerForecaster`

For models that don't log HSIC (e.g., pure prediction tasks), the selection automatically falls back to `val_mae`.

## Date

Implemented: 2026-03-26
