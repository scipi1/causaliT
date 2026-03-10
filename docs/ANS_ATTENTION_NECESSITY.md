# Attention Necessity Score (ANS) - Empirical Causal Capacity Criterion

## Motivation

In transformer-based causal discovery, attention is the primary mechanism for creating dependencies between tokens. For fully observable causal systems, the model should use attention to learn the causal structure and perform good predictions.

**The Problem:** If the model has too much capacity (particularly in embeddings), it could bypass attention entirely and learn a direct mapping using only embeddings + MLP. This would result in:
- Good prediction performance, but
- No meaningful causal structure in attention weights

**The Goal:** Define an empirical criterion to determine if attention is *necessary* for the model to fit the data.

## The ANS Framework

### Key Idea

We measure the **Attention Necessity Score (ANS)**:

```
ANS(λ) = Loss(ablated, λ) - Loss(full, λ)
```

Where:
- `λ` = embedding L1 regularization strength (`lambda_embed_l1`)
- `Loss(full, λ)` = validation loss with learned attention
- `Loss(ablated, λ)` = validation loss with uniform attention (bypass)

### Interpretation

| ANS Value | Meaning |
|-----------|---------|
| ANS > 0 | Attention improves performance → attention is useful |
| ANS ≈ 0 | Attention doesn't help → embeddings alone suffice |
| ANS < 0 | Shouldn't happen (attention shouldn't hurt) |

### The ANS Curve

By sweeping λ from 0 to large values, we can plot the ANS curve:

```
     ANS(λ)
       │
       │     ┌─────────────────  ANS high: attention crucial
       │    ╱
       │   ╱
       │  ╱
       │ ╱
       │╱
  ─────┼──────────────────────── λ
       │     ↑                 ↑
       │  λ_critical        λ_max
       │
       │  Zone 1: Both models fit well (embeddings too powerful)
       │  Zone 2: Only attention model fits (attention necessary)
       │  Zone 3: Neither model fits (too much regularization)
```

**λ_critical** = The smallest λ where ANS becomes significantly positive.

## Implementation

### New Parameters

#### Model (`model.kwargs.attention_bypass`)
```yaml
model:
  attention_bypass: false  # Top-level for sweep compatibility
  kwargs:
    # ... other kwargs
```

When `attention_bypass: true`, attention is replaced with uniform weights:
- Cross-attention: each X token attends equally to all S tokens
- Self-attention: each X token attends equally to all X tokens

#### Training (`training.lambda_embed_l1`)
```yaml
training:
  lambda_embed_l1: 0.0  # L1 regularization on X embeddings
  log_embed_l1: true    # Log the L1 loss
```

The L1 loss is normalized by parameter count for scale-independence:
```python
embed_l1 = sum(|p|) / num_parameters
total_loss += lambda_embed_l1 * embed_l1
```

### Running an ANS Evaluation

1. **Copy templates to your experiment folder:**
   ```bash
   mkdir -p experiments/<your_exp>/sweeper
   cp causaliT/config/config_ans_sweep_template.yaml experiments/<your_exp>/config.yaml
   cp causaliT/config/sweep_ans_template.yaml experiments/<your_exp>/sweeper/sweep.yaml
   ```

2. **Modify config.yaml for your dataset:**
   - Update `data.dataset`, `data.S_seq_len`, `data.X_seq_len`
   - Adjust model architecture if needed

3. **Run the sweep:**
   ```bash
   # Local (sequential)
   python -m causaliT.euler_sweep.euler_sweep.cli sweep \
       --exp_id <your_exp> \
       --sweep_mode combination

   # Cluster (parallel via SLURM)
   python -m causaliT.euler_sweep.euler_sweep.cli sweep \
       --exp_id <your_exp> \
       --sweep_mode combination \
       --parallel --cluster \
       --max_concurrent_jobs 12
   ```

4. **Analyze results:**
   After the sweep completes, compute ANS for each λ by comparing:
   - `val_loss_x` for `attention_bypass=false` (full model)
   - `val_loss_x` for `attention_bypass=true` (ablated model)

### Analysis Script (Example)

```python
import pandas as pd
import numpy as np
from pathlib import Path

def compute_ans_from_sweep(sweep_dir: str, metric: str = "val_loss_x"):
    """
    Compute ANS curve from sweep results.
    
    Args:
        sweep_dir: Path to sweep results (e.g., experiments/<exp>/sweeper/runs/combinations/)
        metric: Metric to use for ANS computation
        
    Returns:
        DataFrame with columns: lambda, ANS_mean, ANS_std, p_value
    """
    from scipy import stats
    
    results = {}
    sweep_path = Path(sweep_dir)
    
    # Group runs by lambda_embed_l1
    for run_dir in sweep_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Parse lambda and bypass from folder name
        # Expected format: combo_lambda_embed_l1_0.01_attention_bypass_false
        name = run_dir.name
        
        # Extract values
        lambda_val = None
        bypass = None
        
        for part in name.split("_"):
            if "lambda" in name and lambda_val is None:
                # Find lambda value
                pass  # Parse logic here
            # ... parsing logic
        
        # Read metrics.csv from each fold
        # Compute mean across folds
        # Store in results[lambda_val][bypass] = list of fold metrics
    
    # Compute ANS per fold (paired comparison)
    ans_summary = []
    for lambda_val in sorted(results.keys()):
        if 'full' in results[lambda_val] and 'ablated' in results[lambda_val]:
            full_losses = np.array(results[lambda_val]['full'])
            ablated_losses = np.array(results[lambda_val]['ablated'])
            
            # Paired ANS per fold
            ans_per_fold = ablated_losses - full_losses
            
            # One-tailed t-test: is ANS > 0?
            t_stat, p_value = stats.ttest_1samp(ans_per_fold, 0)
            
            ans_summary.append({
                'lambda': lambda_val,
                'ANS_mean': np.mean(ans_per_fold),
                'ANS_std': np.std(ans_per_fold),
                'p_value': p_value / 2,  # One-tailed
                'attention_necessary': np.mean(ans_per_fold) > 0 and p_value / 2 < 0.05
            })
    
    return pd.DataFrame(ans_summary)
```

## Theoretical Background

### Why L1 Regularization on Embeddings?

L1 regularization encourages sparsity in embedding weights. This effectively:
1. Reduces the "expressiveness" of embeddings
2. Forces information to be aggregated through attention
3. Acts as a soft capacity constraint

### Why Uniform Attention as Ablation?

We use uniform attention (not zero attention) because:
1. It preserves the information flow structure (embeddings → attention → FF → output)
2. It represents "attention that learned nothing useful"
3. It's closer to a fair comparison (same architecture, same capacity)

### Relation to Information Bottleneck

The ANS framework relates to the **Information Bottleneck (IB) principle** (Tishby et al.):
- The embedding layer acts as a bottleneck
- If the bottleneck is too wide (low λ), the model can memorize
- If the bottleneck is narrow (high λ), the model must learn structure

## Caveats and Limitations

1. **λ Selection**: The "right" λ depends on the dataset and task. There's no universal value.

2. **Training Variance**: Different random seeds can give different results. Use k-fold CV and report mean±std.

3. **Not a Guarantee**: High ANS doesn't prove the attention learned the *correct* causal structure, only that it learned *something useful*.

4. **Uniform ≠ No Attention**: Uniform attention still aggregates information, just without selectivity.

## References

- Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle.
- Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference.
- Zheng, X., et al. (2018). DAGs with NO TEARS: Continuous optimization for structure learning.
