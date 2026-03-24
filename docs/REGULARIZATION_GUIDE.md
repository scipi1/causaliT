# Regularization Guide for CausaliT

This document describes all regularization techniques and annealing schedules available in the CausaliT forecasters.

## Table of Contents
1. [Overview](#overview)
2. [Regularization Parameters by Category](#regularization-parameters-by-category)
3. [Architecture Support Matrix](#architecture-support-matrix)
4. [Attention Type Compatibility](#attention-type-compatibility)
5. [Annealing Schedules](#annealing-schedules)
6. [Configuration Examples](#configuration-examples)

---

## Overview

CausaliT supports various regularization techniques to encourage:
- **Sparsity**: Fewer edges in the learned DAG
- **Acyclicity**: No cycles in the DAG (NOTEARS constraint)
- **Decisiveness**: Edge probabilities pushed toward 0 or 1
- **Independence**: Residuals independent of source variables (HSIC)
- **Focused Attention**: Low entropy = concentrated attention weights

### Key Design Principles

1. **Unified Score Tensor**: Each attention type exposes `score_tensor_for_sparsity` property
2. **Architecture-Specific Support**: Not all regularizers work with all architectures
3. **Attention-Type Compatibility**: Some regularizers only work with specific attention types

---

## Regularization Parameters by Category

### A. Structure Learning (DAG)

| Parameter | Target | Description |
|-----------|--------|-------------|
| `kappa` | NOTEARS | Acyclicity constraint: penalizes cycles in self-attention DAG |
| `lambda_sparse` | σ(phi).mean() | Sparsity on self-attention edge probabilities |
| `lambda_sparse_cross` | σ(phi).mean() | Sparsity on cross-attention edge probabilities |
| `lambda_decisive` | Binary entropy | Pushes self-attention edges away from 0.5 |
| `lambda_decisive_cross` | Binary entropy | Pushes cross-attention edges away from 0.5 |

### B. Unified Score Sparsity (Attention Sharpening)

The new unified approach combines L1 and entropy regularization with automatic mode selection and fallback:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_self_score_sparse` | Weight for self-attention score sparsity | 0.0 |
| `lambda_cross_score_sparse` | Weight for cross-attention score sparsity | 0.0 |
| `self_sparsity_regularizer` | Mode: `"l1"` or `"entropy"` | `"l1"` |
| `cross_sparsity_regularizer` | Mode: `"l1"` or `"entropy"` | `"entropy"` |

**Mode Selection**:
- `"l1"`: Uses `score_tensor_for_sparsity` property (effective for GeLU-based attention)
- `"entropy"`: Uses attention entropy -Σ p log(p) (effective for **all** attention types)

**Automatic Fallback**: If `"l1"` mode is selected but `score_tensor_for_sparsity` is `None` (softmax-based attention), the forecaster automatically falls back to entropy with a warning.

**Note**: L1 on attention scores is **ineffective for softmax-based attention** because outputs always sum to 1, making the mean approximately constant (1/seq_len). The automatic fallback handles this gracefully.

### C. Temperature Control

| Parameter | Description |
|-----------|-------------|
| `lambda_tau` | Penalizes high Gumbel-Softmax temperature |
| `target_tau` | Target temperature for penalty (no penalty below this) |

### D. Independence / Causal

| Parameter | Description |
|-----------|-------------|
| `lambda_hsic` | HSIC independence test between S and residuals |
| `hsic_sigma` | Kernel bandwidth for HSIC computation |

### E. Priors

| Parameter | Description |
|-----------|-------------|
| `lambda_kl` | KL divergence from running average prior |
| `adaptive_z_scaling` | Use SNR-based scaling for KL prior (recommended: true) |
| `lambda_noise_prior` | Prior on noise parameters (noise-aware only) |

### F. Capacity Control

| Parameter | Description |
|-----------|-------------|
| `lambda_embed_l1` | L1 on embedding parameters (for ANS experiments) |

---

## Architecture Support Matrix

| Regularizer | SingleCausal | NoiseAware | StageCausal |
|-------------|:------------:|:----------:|:-----------:|
| `kappa` (NOTEARS) | ✅ | ✅ | ✅ |
| `lambda_sparse[_cross]` | ✅ | ✅ | ✅ |
| `lambda_entropy_*` | ✅ | ✅ | ✅ |
| `lambda_l1_*_scores` | ✅ | ✅ | ✅ |
| `lambda_hsic` | ✅ | ✅ | ❌ |
| `lambda_kl` | ✅ | ✅ | ❌ |
| `lambda_decisive[_cross]` | ✅ | ✅ | ❌ |
| `lambda_tau` | ✅ | ✅ | ❌ |
| `lambda_embed_l1` | ✅ | ❌ | ❌ |
| `lambda_noise_prior` | ❌ | ✅ | ❌ |
| All annealing schedules | ✅ | ✅ | ❌ |

---

## Attention Type Compatibility

### Sparsity Regularizers by Attention Type

| Attention Type | Activation | `lambda_l1_*_scores` | `lambda_sparse` | `lambda_entropy_*` |
|----------------|------------|:--------------------:|:---------------:|:------------------:|
| ScaledDotProduct | Softmax | ❌ Ineffective | ❌ No phi | ✅ **Use this** |
| PhiSoftMax | Softmax + phi | ❌ Ineffective | ✅ Works | ✅ Works |
| LieAttention | GeLU | ✅ Works | ✅ Works | ✅ Works |
| CausalCrossAttention | GeLU(Tanh) | ✅ Works | ✅ Works | ✅ Works |
| ToeplitzLieAttention | ReLU(Tanh) | ✅ Works | ✅ Works on gate | ✅ Works |

### Recommended Combinations

| Attention Type | Recommended Sparsity |
|----------------|---------------------|
| ScaledDotProduct | `lambda_entropy_*` (only effective option) |
| PhiSoftMax | `lambda_sparse` + `lambda_entropy_*` |
| LieAttention | `lambda_l1_*_scores` + `lambda_sparse` |
| ToeplitzLieAttention | `lambda_l1_*_scores` (applies to gate) |

### ToeplitzLieAttention Specifics

ToeplitzLieAttention decomposes attention into:
- **Gate (symmetric)**: Controls edge existence - P(edge exists)
- **Direction (antisymmetric)**: Controls flow direction - P(i→j | edge exists)

The `lambda_l1_*_scores` regularizer applies to the **gate probabilities**, encouraging sparse edge existence rather than penalizing direction. This is the recommended approach because:
1. We want to penalize "is there an edge?" not "which direction?"
2. The gate controls edge existence (both directions)
3. Sparse gates → sparse DAG

---

## Annealing Schedules

### 1. Gumbel-Softmax Temperature (`tau_gs`)

```yaml
training:
  use_tau_gs_annealing: true
  tau_gs_start: 2.0          # High = exploration (soft masks)
  tau_gs_end: 0.2            # Low = exploitation (hard masks)
  tau_gs_anneal_epochs: 80
```

**Schedule**: Exponential `τ(t) = τ_start × (τ_end/τ_start)^(t/T)`

**When to use**: When you want predictable exploration→exploitation behavior. Recommended for ToeplitzLieAttention.

### 2. Activation Temperature (`tau_gate`, `tau_dir`)

For ToeplitzLieAttention only:

```yaml
training:
  use_tau_act_annealing: true
  tau_gate_start: 1.0
  tau_gate_end: 0.2
  tau_dir_start: 0.5
  tau_dir_end: 0.1
  tau_act_anneal_epochs: 80
```

### 3. HSIC Annealing

```yaml
training:
  use_hsic_annealing: true
  hsic_lambda_start: 1.0     # Strong early
  hsic_lambda_end: 0.0       # Disabled late
  hsic_anneal_epochs: 50
```

**Schedule**: Linear annealing

### Conflicting Mechanisms

**Choose one, not both:**

| Mechanism A | Mechanism B | Conflict |
|-------------|-------------|----------|
| `lambda_tau` | `use_tau_gs_annealing` | Both control τ_gs |
| `lambda_hsic` (fixed) | `use_hsic_annealing` | Both control HSIC strength |

---

## Configuration Examples

### Minimal Baseline (No Regularization)

```yaml
training:
  # All regularization disabled - pure reconstruction loss
  kappa: 0.0
  lambda_sparse: 0.0
  lambda_sparse_cross: 0.0
  lambda_entropy_self: 0.0
  lambda_entropy_cross: 0.0
  lambda_l1_self_scores: 0.0
  lambda_l1_cross_scores: 0.0
  lambda_hsic: 0.0
  lambda_decisive: 0.0
  lambda_decisive_cross: 0.0
  lambda_tau: 0.0
  lambda_kl: 0.0
  lambda_embed_l1: 0.0
```

### DAG Learning with ToeplitzLieAttention

```yaml
experiment:
  dec_self_attention_type: "ToeplitzLieAttention"

training:
  # Temperature annealing (exploration → exploitation)
  use_tau_gs_annealing: true
  tau_gs_start: 2.0
  tau_gs_end: 0.2
  tau_gs_anneal_epochs: 80
  
  # Sparsity via L1 on gate probabilities
  lambda_l1_self_scores: 0.1
  
  # Decisiveness (push edges away from 0.5)
  lambda_decisive: 0.05
  
  # Acyclicity constraint
  kappa: 0.1
  
  # Logging
  log_entropy: true
  log_acyclicity: true
  log_decisiveness: true
  log_l1_scores: true
```

### ScaledDotProduct with Entropy Regularization

```yaml
experiment:
  dec_self_attention_type: "ScaledDotProduct"

training:
  # Use entropy for sparsity (L1 scores ineffective with softmax)
  lambda_entropy_self: 0.1
  lambda_entropy_cross: 0.1
  
  # No phi-based regularizers available
  lambda_sparse: 0.0
  lambda_decisive: 0.0
```

### ANS (Attention Necessity Score) Experiment

```yaml
model:
  attention_bypass: false  # Toggle in sweep

training:
  # Embedding L1 to limit capacity
  lambda_embed_l1: 0.01  # Sweep this: [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]
  log_embed_l1: true
  
  # Disable other regularizers for clean comparison
  kappa: 0.0
  lambda_sparse: 0.0
```

---

## Logging Configuration

Enable logging with these flags:

```yaml
training:
  log_entropy: true          # Attention entropy
  log_acyclicity: true       # NOTEARS value
  log_sparsity: true         # Sparsity regularizer value
  log_l1_scores: true        # L1 attention scores
  log_hsic: true             # HSIC value
  log_decisiveness: true     # Decisiveness metrics
  log_embed_l1: true         # Embedding L1 norm
  log_tau_annealing: true    # Annealed temperature values
  log_hsic_annealing: true   # Annealed HSIC coefficient
  log_noise_params: true     # σ_A, σ_R (noise-aware only)
```

---

## Debugging Tips

### Problem: DAG values stuck at ~0.5

**Cause**: Insufficient decisiveness pressure

**Solution**:
```yaml
training:
  lambda_decisive: 0.1
  use_tau_gs_annealing: true
  tau_gs_end: 0.1  # Lower temperature
```

### Problem: DAG too sparse (all edges pruned)

**Cause**: Over-regularization

**Solution**: Reduce sparsity penalties:
```yaml
training:
  lambda_sparse: 0.01  # Lower
  lambda_l1_self_scores: 0.01  # Lower
```

### Problem: HSIC too high throughout training

**Cause**: Model not learning causal structure

**Solution**: Use HSIC annealing to allow fitting first:
```yaml
training:
  use_hsic_annealing: true
  hsic_lambda_start: 0.0  # Disabled early
  hsic_lambda_end: 0.5    # Active late
```

---

## Migration Notes

### Deprecated Parameters

The following parameters are no longer used and should be removed from configs:

| Deprecated | Replacement |
|------------|-------------|
| `gamma` | Removed (was never a config parameter, internal to ToeplitzLieAttention) |
| `lambda_l1_toeplitz_gate` | Use `lambda_l1_self_scores` instead (unified via `score_tensor_for_sparsity` property) |

### Unified Score Regularization

All attention types now expose a `score_tensor_for_sparsity` property that returns the appropriate tensor for L1 sparsity regularization:

| Attention Type | `score_tensor_for_sparsity` Returns | Notes |
|----------------|-------------------------------------|-------|
| LieAttention | `sigmoid(phi)` | Edge existence probabilities |
| CausalCrossAttention | `sigmoid(phi)` | Edge existence probabilities |
| PhiSoftMax | `None` | Use entropy regularization instead |
| ScaledDotAttention | `None` | Use entropy regularization instead |
| ToeplitzAttention | `P_edge_for_reg` | Edge existence probabilities |
| ToeplitzLieAttention | `gate_probs_for_reg` | Gate (edge existence) probabilities |

The forecasters automatically use this property via `lambda_l1_self_scores` and `lambda_l1_cross_scores`. If the property returns `None`, no regularization is applied (appropriate for softmax-based attention where L1 is ineffective).

### Parameter Renaming

For consistency, consider using these patterns:
- `lambda_*_self` for self-attention
- `lambda_*_cross` for cross-attention
