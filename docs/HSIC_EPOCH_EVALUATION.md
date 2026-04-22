# HSIC Batch Size Hypothesis & Bilevel Optimization Plan

## Date: 2026-04-21

## Problem Statement

Per-batch HSIC computation (batch_size=128) with adaptive bandwidth produces noisy,
oscillating values (~0.002–0.005) that are nearly constant across training. This creates
two downstream failures:

1. **HSIC doesn't guide structure learning** — the optimization minimizes loss primarily
   via sparsity (score LASSO), not by learning the correct causal structure.
2. **Score sparsity CV selects on noise** — the `min_hsic` selection rule picks lambda
   values based on HSIC differences smaller than the estimation variance (see SCM2:
   λ=0.1 selected over λ=0.005, causing 2x worse reconstruction and empty DAG).

## Experiments Analyzed

| Metric | SCM1 (64182170) | SCM2 (64182198) | SCM3 (64182223) |
|--------|-----------------|-----------------|-----------------|
| val_loss_x (MSE) | 0.0017 | 0.0009 | 0.0006 |
| val_R² | 0.946 | 0.965 | 0.977 |
| val_hsic_cross | 0.0046 | 0.0028 | 0.0029 |
| val_hsic_self | 0.0042 | 0.0023 | 0.0026 |
| val_hsic_reg | 0.013 | 0.011 | 0.012 |
| effective_dims | 29 | 48 | 45 |
| best_lambda_score | 0.005 | **0.1** | 0.005 |

**Key observations:**
- HSIC values are remarkably similar across SCMs (~0.002-0.005) despite different data
- SCM2 selected λ_score=0.1 (max) due to HSIC noise → skeleton_recall=0.001 (empty DAG)
- All experiments: gradient routing OFF, per-variable HSIC, adaptive bandwidth

## Root Cause Hypothesis: Batch Size → HSIC Variance

With n=128 samples per batch, the HSIC kernel estimator has high variance. The HSIC
estimation variance scales as O(1/n²):

| N samples | Relative variance (vs N=128) |
|-----------|------------------------------|
| 128 | 1.0× (current) |
| 512 | 0.06× (16× better) |
| 1024 | 0.016× (64× better) |
| 2048 | 0.004× (256× better) |

The D1 sanity test used the full dataset for HSIC, which worked perfectly (W recovery
error 0.014). The gap between "HSIC works on full data" and "HSIC oscillates per-batch"
is the core hypothesis to test.

### Why adaptive bandwidth doesn't cause the problem

The adaptive bandwidth (median heuristic) is NOT the issue. With adaptive bandwidth:
- If residuals shrink uniformly (better reconstruction), σ_y shrinks too, HSIC stays constant — this is correct behavior (reconstruction improvement alone shouldn't decrease HSIC)
- If the model selects correct parents, residuals become **structurally independent** of S_i — the dependence pattern disappears regardless of scale
- The HSIC should still decrease when true structure is learned

The problem is that with n=128, the kernel estimate is too noisy to detect the independence
pattern at all.

## Proof-of-Concept Experiment: batch_size=1024

**Before implementing any complex solutions, test the simplest hypothesis:**

Run the existing code (unchanged) with `batch_size: 1024` instead of 128.

### Expected outcomes:
- **If HSIC starts decreasing during training**: Batch size is the bottleneck → proceed to scalable solution (bilevel optimization)
- **If HSIC still oscillates**: Problem is elsewhere (gradient dominance, structural parameter responsiveness) → investigate other causes

### Config change (only):
```yaml
experiment:
  batch_size: 1024   # was 128
```

Everything else stays the same: per-variable HSIC, adaptive bandwidth, gradient routing OFF,
staged training pipeline.

### Memory consideration:
- HSIC kernel matrix: 1024×1024 × 4 bytes = 4 MB (fits easily)
- Forward pass: small model (48K params), no concern
- Should run on the same GPU without issues

## Future: DARTS-Style Bilevel Optimization

If the batch_size experiment confirms the hypothesis, the scalable solution is a
bilevel optimization approach inspired by DARTS (Liu et al., ICLR 2019):

### Architecture

| | DARTS | Our approach |
|---|---|---|
| **Upper level** (slow) | Architecture params α | Structural params θ_S (Q, K, embeddings) |
| **Lower level** (fast) | Network weights w | Reconstruction params θ_R (V, FF, MLP head) |
| **Upper objective** | L_val(w*(α), α) | HSIC(θ_S, θ_R*) on large accumulated batch |
| **Lower objective** | L_train(w, α) | MSE reconstruction loss |
| **Update pattern** | w per-batch, α per-epoch on val | θ_R per-batch (MSE), θ_S per-epoch (HSIC) |

### How it works:

1. **Within each epoch** (Phase 1 — per batch):
   - θ_S is effectively fixed
   - Update θ_R via MSE reconstruction loss
   - Buffer raw (S, X) data tensors (detached, on CPU)

2. **End of epoch** (Phase 2 — structural update):
   - Sample a large batch (e.g., 1024-2048) from the buffer
   - **Fresh forward pass** through the model (with gradients) → creates live computation graph
   - Compute HSIC on the large-batch residuals → live gradients flow through θ_S
   - Backward through θ_S only → step structural optimizer

### Key insight on backpropagation:
We buffer raw INPUT data (S, X), NOT residuals. At epoch end, we do a fresh forward pass:
```
HSIC(S, residuals) → residuals = X_target - pred_x → pred_x = f(S, X; θ_S, θ_R) → θ_S
```
The gradient chain flows from HSIC → through the live forward pass → to θ_S.
The buffered data are just inputs (like any batch), they don't need gradients themselves.

### Why this works better:
- Per-batch HSIC: noisy gradient on 128 samples, N updates per epoch
- Epoch-level HSIC: clean gradient on 1024+ samples, 1 update per epoch
- Each update is much more informative; structural learning is slower but correct

### Memory-safe:
- Buffer stores raw data on CPU (cheap)
- Only one large forward pass at epoch end (manageable)
- HSIC kernel: 1024×1024 = 4 MB (trivial)

## Implementation Status

- [x] Analysis of experiments (scm1/scm2/scm3 with per-variable HSIC, no gradient routing)
- [x] Identified batch size as primary hypothesis
- [x] Identified DARTS bilevel optimization as scalable solution
- [ ] **NEXT**: Run batch_size=1024 proof-of-concept on cluster
- [ ] If confirmed: Implement bilevel optimization in SingleCausalForecaster
- [ ] If confirmed: Implement EpochHSICEvaluator callback for monitoring
