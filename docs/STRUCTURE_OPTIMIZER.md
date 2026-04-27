# Structure Optimizer: Separate Optimizer for DAG Learning

**Date**: 2026-04-23  
**Status**: Implemented

## Problem

DAG learning via HSIC is steady but painfully slow. Training dynamics show:
- Attention scores plateau for many epochs before changing
- Movement still visible at end of training → not converged to wrong DAG
- Score trajectories suggest sharp loss landscape features and optimizer momentum exhaustion

## Solution

Extended the gradient routing system to support **independent optimizer configuration** for structural parameters (Q, K, embeddings, gain, tau) vs reconstruction parameters (V, FF, MLP head).

## New Config Fields

All fields are optional and backward-compatible (default to reconstruction optimizer settings when null).

```yaml
training:
  # Reconstruction optimizer (existing, unchanged)
  optimizer: "adamw"
  lr: 0.001
  weight_decay: 0.01
  
  # Structure optimizer (new, only used when use_gradient_routing: true)
  structural_optimizer: null      # null = same as optimizer. Options: adamw, adam, sgd, adagrad, rmsprop
  structural_lr: null             # null = same as lr
  structural_weight_decay: null   # null = same as weight_decay
  structural_optimizer_kwargs: {} # Extra kwargs (e.g., {momentum: 0.9, nesterov: true})
  
  # LR scheduler for structural optimizer only
  structural_scheduler: null      # null, cosine_warm_restarts, step, cosine
  structural_scheduler_kwargs: {} # e.g., {T_0: 100, T_mult: 1}
  
  # Gradient noise injection for structural params (Langevin exploration)
  structural_gradient_noise: 0.0      # std of Gaussian noise added to structural grads
  structural_gradient_noise_decay: 1.0  # multiplicative decay per epoch
```

## Implementation

### Files Modified
- `causaliT/training/forecasters/single_causal_forecaster.py` — `configure_optimizers()` and `training_step()`
- `causaliT/training/forecasters/noise_aware_forecaster.py` — same changes

### Files Created
- `causaliT/training/optimizer_factory.py` — Shared optimizer/scheduler factory with config helpers
- `experiments/tests/structure_opt/` — 13 experiment configs (8 optimizer sweep + 5 λ_group sweep)

## Experiment Configs

### Phase 1: Optimizer Sweep (`opt_*`)

| Config | Structure Optimizer | Hypothesis |
|--------|-------------------|------------|
| `opt_baseline_adamw` | AdamW(lr=1e-3) | Current baseline |
| `opt_sgd_momentum` | SGD(lr=1e-3, mom=0.9) | SGD explores more; less adaptive = less stuck |
| `opt_sgd_nesterov` | SGD(lr=1e-3, mom=0.9, nesterov) | Look-ahead helps escape saddle points |
| `opt_adagrad` | Adagrad(lr=1e-2) | Never forgets past gradients; persistent LR |
| `opt_adam_high_lr` | AdamW(lr=1e-2) | 10× higher struct LR → faster structure changes |
| `opt_adam_restart` | AdamW + CosineWarmRestart(T₀=100) | Periodic momentum reset to escape plateaus |
| `opt_adam_noise` | AdamW + grad_noise(0.01) | Langevin-style stochastic exploration |
| `opt_rmsprop` | RMSprop(lr=1e-3) | Different adaptive scheme (no momentum bias) |

### Phase 2: Lambda Group Sweep (`lgrp_*`)

Tests whether high L1 regularization prevents the reconstruction model from shortcutting causal paths.

| Config | λ_group_l1 |
|--------|-----------|
| `lgrp_0_0` | 0.0 |
| `lgrp_0_01` | 0.01 |
| `lgrp_0_1` | 0.1 |
| `lgrp_0_5` | 0.5 |
| `lgrp_1_0` | 1.0 |

## Running

```bash
# Phase 1: optimizer sweep (can run in parallel)
for d in experiments/tests/structure_opt/opt_*; do
    python -m causaliT.cli calitrain --exp_id "$d"
done

# Phase 2: lambda group sweep
for d in experiments/tests/structure_opt/lgrp_*; do
    python -m causaliT.cli calitrain --exp_id "$d"
done
```

## Evaluation

All configs use fast evaluation only:
- `eval_train_metrics` — training curves
- `eval_attention_scores` — DAG recovery (final checkpoint)
- `eval_attention_evolution` — attention score evolution over epochs

Key metrics to compare:
- `soft_hamming_self`, `soft_hamming_cross` — DAG accuracy
- `mec_distance` — structural distance
- `skeleton_recall` — edge recovery
- Epoch at which scores first move away from initialization plateau
