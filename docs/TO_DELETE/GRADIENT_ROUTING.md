# Gradient Routing for SVFA Causal Structure Learning

## Problem

In SVFA models with HSIC regularization, the HSIC term oscillates during training without decreasing, while its magnitude dominates the reconstruction loss. This creates a gradient conflict: HSIC gradients flow through all parameters (including V projections, FF layers, MLP head), corrupting the reconstruction signal. Meanwhile, reconstruction gradients update structural parameters (Q, K, embeddings), preventing the attention structure from converging to the true DAG.

**Observed in experiments** (scm1/scm2/scm3 with Toeplitz+CC SVFA):
- HSIC decreases slightly during causal init (dominated by HSIC loss + annealing)
- During main training, HSIC oscillates around ~0.002-0.004 without converging
- val_hsic_reg (~0.019-0.023) >> val_loss_x (~0.0004-0.0026)
- R² remains high (0.92-0.98) but HSIC never reaches zero

## Solution: Dual-Optimizer Gradient Routing

SVFA naturally separates model parameters into two groups:

| Group | Parameters (θ_S) | Updated by |
|-------|-------------------|------------|
| **Structural** | Structure embeddings, Q/K projections, attention internals (log_gain, log_tau), structure-path norms | HSIC + score sparsity + group L1 |
| **Reconstruction** | Value embeddings, V projection, output projection, FF layers, MLP head, value-path norms, noise params | Reconstruction loss (MSE or NLL) |

Each group gets its own optimizer. Per training step:
1. Single forward pass computes all losses
2. `loss_recon` backward → updates only θ_R via `opt_recon`
3. `loss_structural` backward → updates only θ_S via `opt_struct`

This eliminates gradient interference: HSIC can freely reshape the attention structure without fighting reconstruction gradients, and the MLP/FF layers can fit the data without being pulled by HSIC noise.

## Usage

```yaml
training:
  use_gradient_routing: true  # Enable dual-optimizer mode
  lambda_hsic_cross: 10.0
  lambda_hsic_self: 10.0
```

Works with staged training — just add the flag to your config. Compatible with all existing features (HSIC annealing, score sparsity, group L1, hard masks).

## Parameter Classification

Classification is done by name pattern matching in `gradient_routing.py`:

**Structural patterns** (→ θ_S):
- `query_projection`, `key_projection`
- `structure_modules` (SVFA structure embeddings)
- `inner_attention.log_gain`, `inner_attention.log_tau`, `inner_attention.temperature`
- `norm1_struct`, `norm2_struct`

**Everything else** → θ_R (value_projection, out_projection, linear1/2, forecaster, value_modules, etc.)

To inspect the classification for your model:
```python
from causaliT.training.gradient_routing import classify_parameters
structural, reconstruction = classify_parameters(model, verbose=True)
```

## Theoretical Motivation

This approach is a form of **bilevel optimization** where:
- Outer level: structure parameters minimize HSIC (causal discovery)
- Inner level: reconstruction parameters minimize prediction loss (data fitting)

Related work:
- **DARTS** (Liu et al., ICLR 2019): Bilevel optimization separating architecture parameters from network weights
- **PCGrad** (Yu et al., NeurIPS 2020): Gradient surgery to resolve conflicting gradients in multi-task learning
- **GradNorm** (Chen et al., ICML 2018): Adaptive gradient balancing across tasks
- **CASTLE** (Kyono et al., NeurIPS 2020): Neural network-based causal structure learning with masked gradients
- **DAG-GNN** (Yu et al., ICML 2019): Separate structural and functional parameter optimization for DAG learning

The key insight from DARTS applies directly: architecture/structure parameters and model weights serve fundamentally different purposes and benefit from separate optimization.

## Logged Metrics

When gradient routing is active, additional metrics are logged:
- `train_loss_recon_routed`: Reconstruction loss only
- `train_loss_structural_routed`: Structural loss only (HSIC + sparsity + group L1)

## Files

```
causaliT/training/
├── gradient_routing.py                    # Parameter classification
├── forecasters/single_causal_forecaster.py  # Dual optimizer support
└── forecasters/noise_aware_forecaster.py    # Dual optimizer support
```
