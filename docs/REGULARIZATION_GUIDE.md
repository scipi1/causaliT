# Regularization & Annealing Guide for causaliT

This document describes all regularization techniques and annealing schedules available in the causaliT forecasters, particularly for the ToeplitzLieAttention mechanism.

## Table of Contents
1. [Overview](#overview)
2. [ToeplitzLieAttention Regularizers](#toeplitzlieattention-regularizers)
3. [Annealing Schedules](#annealing-schedules)
4. [Configuration Examples](#configuration-examples)

---

## Overview

The ToeplitzLieAttention mechanism learns DAG structure through two components:
- **Gate probabilities** (`σ(γ)`) - Controls edge existence (symmetric)
- **Direction probabilities** (`σ(φ)`) - Controls edge direction (antisymmetric)

Final edge probability: `P(i→j) = σ(γ_ij) × σ(φ_ij)`

### Problem: Constant DAG Values (0.84)

If DAG probabilities converge to constant values (e.g., 0.84) across all experiments:
1. **Temperature too high**: Gumbel-Softmax relaxation is too smooth
2. **Missing sparsity pressure**: No incentive for gates to close
3. **Gradient saturation**: Tanh activations saturating before learning

---

## ToeplitzLieAttention Regularizers

### Sparsity Regularizers Comparison

There are **multiple sparsity mechanisms** that work on different targets:

| Parameter | Target | Formula | Works With |
|-----------|--------|---------|------------|
| `lambda_l1_*_scores` | Attention weights | `mean(A_ij)` | GeLU-based only (see below) |
| `lambda_sparse` | DAG probabilities | `mean(σ(phi))` | Only phi-based attention |
| `lambda_entropy_*` | Distribution entropy | `-Σ A log(A)` | **All attention types** |
| `lambda_l1_toeplitz_gate` | Gate probabilities | `mean(σ(γ))` | Only ToeplitzLieAttention |

### L1 Scores Compatibility by Attention Type

| Attention Type | Activation | `lambda_l1_*_scores` Works? |
|----------------|-----------|----------------------------|
| ScaledDotProduct | Softmax | ❌ No (sums to 1) |
| PhiSoftMax | Softmax + phi mask | ❌ No (sums to 1) |
| LieAttention | GeLU(commutator) | ✅ Yes |
| **CausalCrossAttention** | GeLU(Tanh) | ✅ **Yes** |
| ToeplitzLieAttention | GeLU(Tanh) | ✅ Yes |

**Why `lambda_l1_*_scores` fails for softmax:**
Softmax outputs sum to 1, so `mean(A) ≈ 1/seq_len` regardless of sparsity. Use entropy instead.

**Why it works for GeLU-based attention:**
GeLU/Tanh outputs don't sum to 1, so penalizing mean directly encourages sparsity.

**Recommendations by attention type:**

| Attention Type | Best Sparsity Method |
|----------------|---------------------|
| ScaledDotProduct | `lambda_entropy_*` (only option that works) |
| LieAttention | `lambda_l1_self_scores` + `lambda_sparse` |
| ToeplitzLieAttention | `lambda_l1_toeplitz_gate` + `lambda_l1_self_scores` |
| CausalCrossAttention | `lambda_l1_cross_scores` + `lambda_sparse_cross` |

### 1. L1 Gate Sparsity (`lambda_l1_toeplitz_gate`)

Penalizes average gate probability to encourage sparse DAGs (ToeplitzLieAttention only):

```yaml
training:
  lambda_l1_toeplitz_gate: 0.1  # Penalize open gates
```

**Effect**: Pushes `σ(γ) → 0` for unnecessary edges.

### 2. Decisiveness Regularizer (`lambda_decisive`)

Pushes DAG probabilities away from 0.5 toward 0 or 1:

```yaml
training:
  lambda_decisive: 0.1
  lambda_decisive_cross: 0.1
```

**Note:** `lambda_decisive` acts on **phi (DAG logits)**, not on tau_gs. It is **complementary** to temperature annealing and can be used together with `use_tau_gs_annealing`.

### Parameter Interaction Summary

| Parameter | Target | Effect |
|-----------|--------|--------|
| `lambda_decisive` | phi (DAG logits) | Pushes σ(phi) toward 0 or 1 |
| `lambda_tau` | tau_gs (via loss) | Soft incentive to decrease temperature |
| `use_tau_gs_annealing` | tau_gs (direct) | Hard schedule for temperature |

**Complementary pairs** (use together):
- `lambda_decisive` + `use_tau_gs_annealing` ✓
- `lambda_decisive` + `lambda_tau` ✓

**Conflicting pairs** (use one, not both):
- `lambda_tau` + `use_tau_gs_annealing` ✗
- `lambda_hsic` + `use_hsic_annealing` ✗ (hsic_lambda_start/end override lambda_hsic)

### HSIC Parameter Interaction

| Parameter | Role | Affected by Annealing? |
|-----------|------|------------------------|
| `lambda_hsic` | Fixed coefficient | N/A (not used if annealing enabled) |
| `hsic_lambda_start/end` | Annealing schedule | Yes (overrides lambda_hsic) |
| `hsic_sigma` | Kernel bandwidth | **No** (stays constant) |

**Usage:**
```yaml
# Option A: Fixed HSIC coefficient
training:
  lambda_hsic: 0.1
  hsic_sigma: 1.0
  use_hsic_annealing: false

# Option B: Annealed HSIC (recommended for staged learning)
training:
  lambda_hsic: 0.0  # Ignored when annealing enabled
  hsic_sigma: 1.0   # Kernel bandwidth (not annealed)
  use_hsic_annealing: true
  hsic_lambda_start: 1.0  # Strong early
  hsic_lambda_end: 0.0    # Disabled late
```

### τ_gs Control: `lambda_tau` vs `use_tau_gs_annealing`

There are **two complementary but conflicting** mechanisms for controlling τ_gs:

| Parameter | Mechanism | τ_gs Learnable? |
|-----------|-----------|-----------------|
| `lambda_tau` + `target_tau` | Gradient-based loss penalty | Yes |
| `use_tau_gs_annealing` | Direct override at epoch start | No (reset each epoch) |

**Recommendation:** Use one or the other, not both.

```yaml
# Option A: Gradient-driven (soft incentive)
training:
  lambda_tau: 0.1
  target_tau: 0.1
  use_tau_gs_annealing: false

# Option B: Scheduled annealing (hard schedule, recommended for ToeplitzLieAttention)
training:
  lambda_tau: 0.0
  use_tau_gs_annealing: true
  tau_gs_start: 2.0
  tau_gs_end: 0.2
```

**When to use which:**
- **`lambda_tau`**: When you want the model to discover optimal temperature via gradient descent
- **`use_tau_gs_annealing`**: When you want predictable exploration→exploitation behavior (recommended for ToeplitzLieAttention)

### 3. HSIC Regularizer (`lambda_hsic`)

Encourages independence between interventions and residuals:

```yaml
training:
  lambda_hsic: 0.1
  hsic_sigma: 1.0
```

---

## Annealing Schedules

### 1. Gumbel-Softmax Temperature Annealing (`tau_gs`)

Anneals the Gumbel-Softmax temperature from high (exploration) to low (exploitation).

```yaml
training:
  use_tau_gs_annealing: true
  tau_gs_start: 2.0          # Start with smooth relaxation
  tau_gs_end: 0.2            # End with sharp decisions
  tau_gs_anneal_epochs: 80   # Anneal over 80 epochs
```

**Schedule**: Exponential `τ(t) = τ_start × (τ_end/τ_start)^(t/T)`

### 2. Toeplitz Activation Temperature Annealing (`tau_gate`, `tau_dir`)

Anneals the tanh activation temperatures in ToeplitzLieAttention.

```yaml
training:
  use_tau_act_annealing: true
  tau_gate_start: 1.0        # Gate activation temperature
  tau_gate_end: 0.2
  tau_dir_start: 0.5         # Direction activation temperature
  tau_dir_end: 0.1
  tau_act_anneal_epochs: 80
```

### 3. HSIC Annealing

Decreases HSIC regularization over training (allows fitting early, then enforces independence).

```yaml
training:
  use_hsic_annealing: true
  hsic_lambda_start: 1.0     # Strong early
  hsic_lambda_end: 0.0       # Disabled late
  hsic_anneal_epochs: 50     # Anneal over 50 epochs
```

**Schedule**: Linear `λ(t) = λ_start + (t/T) × (λ_end - λ_start)`

---

## Configuration Examples

### Recommended Settings for DAG Learning

```yaml
experiment:
  dec_self_attention_type: "ToeplitzLieAttention"
  
training:
  # Temperature annealing
  use_tau_gs_annealing: true
  tau_gs_start: 2.0
  tau_gs_end: 0.2
  tau_gs_anneal_epochs: 80
  
  # Gate sparsity
  lambda_l1_toeplitz_gate: 0.1
  
  # Decisiveness
  lambda_decisive: 0.05
  
  # HSIC (optional)
  use_hsic_annealing: true
  hsic_lambda_start: 0.5
  hsic_lambda_end: 0.0
  
  # Logging
  log_tau_annealing: true
  log_decisiveness: true
```

### Debugging Constant DAG Values

If DAG values stay constant at ~0.84:

1. **Check temperature**:
   ```yaml
   training:
     use_tau_gs_annealing: true
     tau_gs_start: 1.0  # Lower start
     tau_gs_end: 0.1    # Lower end
   ```

2. **Add sparsity pressure**:
   ```yaml
   training:
     lambda_l1_toeplitz_gate: 0.2
     lambda_decisive: 0.1
   ```

3. **Check initialization**:
   ```yaml
   model:
     kwargs:
       dag_parameterization_self: "gated"
   ```

---

## Logging

Enable detailed logging for debugging:

```yaml
training:
  log_entropy: true
  log_l1_scores: true
  log_decisiveness: true
  log_tau_annealing: true
  log_hsic_annealing: true
```

Logged metrics:
- `annealed_tau_gs`: Current Gumbel-Softmax temperature
- `annealed_tau_gate`: Current gate activation temperature
- `annealed_tau_dir`: Current direction activation temperature
- `annealed_lambda_hsic`: Current HSIC coefficient

---

## Mathematical Background

### Why 0.84 appears

With `σ(γ) ≈ 0.84` and `σ(φ) ≈ 1.0`:
- `γ ≈ 1.6` (log-odds)
- This is a local optimum where reconstruction loss balances with implicit regularization

### Solution via Annealing

1. **Early training** (high τ): Explore structure space
2. **Mid training**: Learn meaningful edges
3. **Late training** (low τ): Commit to discrete structure

The exponential schedule ensures most annealing happens early, allowing the model to stabilize.
