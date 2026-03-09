# Noise-Aware Causal Transformer - Implementation Summary

## Overview

This document summarizes the implementation of the noise-aware causal transformer architecture as specified in `noise_aware_transformer_summary.md` and `NOISE_LEARNING.md`.

The implementation creates a new forecaster (`NoiseAwareCausalForecaster`) that maintains backward compatibility with existing code while adding explicit noise modeling for uncertainty quantification.

---

## Architecture

### Conceptual Model

```
S → H_det → H → U → X
    ^       ^   ^   ^
    |       |   |   |
    cross   σ_A self σ_R
    attn    noise attn noise
```

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Cross-attention | H_det = CrossAtt(X_struct, S, S) — deterministic |
| 2 | Ambient noise | H = H_det + σ_A * ε — environmental variability |
| 3 | Self-attention | U = SelfAtt(X_struct, X_struct, H) — mixing noisy states |
| 4 | Output head | (μ, log_τ) = head(U) — includes reading noise σ_R |

### Key Properties

1. **Attention structure remains DETERMINISTIC** — noise does not affect Q, K
2. **Noise affects VALUES only** — V in self-attention uses noisy H
3. **Variance propagates through causal mixing**: `Var(X_i) = Σ_j α_ij² σ_A[j]² + σ_R[i]²`
4. **SVFA required** — Structure-Value Factorized Attention for clean separation

---

## Files Created

### Core Modules

```
causaliT/core/modules/noise_layers.py
```

| Class | Purpose |
|-------|---------|
| `AmbientNoiseLayer` | Injects per-node ambient noise σ_A[j] into hidden representations |
| `ReadingNoiseHead` | Probabilistic output head with per-node reading noise σ_R[i] |
| `GaussianNLLLoss` | Gaussian negative log-likelihood loss with stability features |
| `VariancePropagationTracker` | Tracks variance propagation for analysis |

### Architecture

```
causaliT/core/architectures/noise_aware/
├── __init__.py
├── model.py        # NoiseAwareSingleCausalLayer
└── decoder.py      # NoiseAwareReversedDecoderLayer
```

| Class | Purpose |
|-------|---------|
| `NoiseAwareSingleCausalLayer` | Main model with noise injection |
| `NoiseAwareReversedDecoder` | Stack of decoder layers |
| `NoiseAwareReversedDecoderLayer` | Single layer with noise between cross/self attention |
| `NoiseAwareReversedDecoderLayerV2` | Alternative: noise after normalization |

### Training

```
causaliT/training/forecasters/noise_aware_forecaster.py
```

| Class | Purpose |
|-------|---------|
| `NoiseAwareCausalForecaster` | PyTorch Lightning wrapper with Gaussian NLL training |

### Configuration

```
causaliT/config/config_noise_aware_example.yaml
```

Example configuration demonstrating all noise-aware parameters.

---

## Design Decisions (Marked for Paper)

### 1. Per-Node Noise Parameterization

**Choice**: σ_A[j] and σ_R[i] are node-specific learnable parameters

**Rationale**: 
- Different sensors have different precision
- Physically meaningful (some variables more noisy than others)
- Allows model to learn heteroscedastic noise

**Alternative considered**: Per-embedding-dimension noise σ_A[j, d]
- Available via `noise_per_dimension: true` in config
- More flexible but harder to interpret

### 2. Noise Injection Point

**Choice**: Ambient noise injected BEFORE W_v projection (in embedding space)

**Rationale**:
- Noise in embedding space is transformed consistently with values
- Aligns with conceptual model: H_det → H happens before attention aggregation

**Alternative implemented**: `NoiseAwareReversedDecoderLayerV2`
- Noise AFTER normalization (in projected space)
- Kept for experimentation

### 3. SVFA Requirement

**Choice**: Noise-aware model requires SVFA factorization

**Rationale**:
- Clean separation: Q, K from structure (deterministic) | V from value (noisy)
- Attention pattern doesn't change based on noise realizations
- Consistent with "attention structure remains deterministic" principle

---

## Training Details

### Loss Function

Gaussian Negative Log-Likelihood:

```
L = (x - μ)² / (2τ²) + log(τ)
```

The `log(τ)` term naturally penalizes unnecessarily large variance.

### Stability Measures (Best Practices)

| Measure | Implementation |
|---------|---------------|
| Variance clamping | `var = exp(log_var).clamp(min=eps)` with `eps=1e-6` |
| Log-variance bounds | `log_var.clamp(-10, 5)` |
| Positivity | `σ = exp(log_σ)` parameterization |
| Small initialization | `σ_A ≈ 0.01`, `σ_R ≈ 0.05` (near-deterministic start) |

### Known Pitfalls Addressed

| Pitfall | Solution |
|---------|----------|
| Sigma growth plateau | LR scheduler recommended (`use_scheduler: true`) |
| Collapse to mean | `log(τ)` penalty prevents this |
| Identifiability | Optional noise prior regularizer (`lambda_noise_prior`) |

---

## Usage

### Basic Training

```python
from causaliT.training.forecasters import NoiseAwareCausalForecaster

# Load config (see config_noise_aware_example.yaml)
config = load_config("config_noise_aware.yaml")

# Create forecaster
forecaster = NoiseAwareCausalForecaster(config, data_dir=data_dir)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(forecaster, train_loader, val_loader)
```

### Inference with Uncertainty

```python
# Get predictions with uncertainty
mu, std = forecaster.predict(S, X)

# Get confidence intervals
mu, lower, upper = forecaster.predict_with_intervals(S, X, confidence=0.95)

# Get full distribution parameters
mu, var = forecaster.get_predictive_distribution(S, X)

# Inspect learned noise parameters
noise_params = forecaster.get_noise_parameters()
print(f"Ambient noise: {noise_params['sigma_A'].mean():.4f}")
print(f"Reading noise: {noise_params['sigma_R'].mean():.4f}")
```

---

## Configuration Parameters

### Noise-Aware Specific

```yaml
model:
  kwargs:
    # Initial noise standard deviations
    init_sigma_A: 0.01      # Ambient noise (small = near-deterministic start)
    init_sigma_R: 0.05      # Reading noise
    
    # Noise parameterization
    noise_per_dimension: false  # Per-node (true for per-dimension)
    
    # Variance tracking
    track_variance: false   # Track variance propagation

training:
  # NLL settings
  nll_eps: 1.0e-6           # Numerical stability
  nll_full: false           # Include constant term
  
  # Logging
  log_noise_params: true    # Log σ_A, σ_R during training
  
  # Optional noise prior (for identifiability)
  lambda_noise_prior: 0.0   # KL prior on noise parameters
  prior_sigma_A: 0.01       # Prior mean for σ_A
  prior_sigma_R: 0.05       # Prior mean for σ_R
```

### All Regularizers Supported

All regularizers from `SingleCausalForecaster` are supported:
- Entropy regularization (`lambda_entropy_*`)
- Acyclicity (NOTEARS) (`kappa`)
- Sparsity (`lambda_sparse*`)
- HSIC (`lambda_hsic`)
- Decisiveness (`lambda_decisive*`)
- KL prior (`lambda_kl`)
- L1 on attention scores (`lambda_l1_*_scores`)

---

## Logged Metrics

During training, the following metrics are logged:

| Metric | Description |
|--------|-------------|
| `*_nll` | Gaussian NLL loss |
| `*_x_mae`, `*_x_rmse`, `*_x_r2` | Reconstruction metrics (using mean μ) |
| `*_pred_var_mean`, `*_pred_var_std` | Predicted variance statistics |
| `*_sigma_A_mean`, `*_sigma_A_std` | Ambient noise parameters |
| `*_sigma_R_mean`, `*_sigma_R_std` | Reading noise parameters |

---

## Testing Recommendations

### Unit Tests

1. **Noise injection**: Verify noise is injected correctly
   ```python
   # Training: H ≠ H_det (noise added)
   # Inference: H = H_det (no noise)
   ```

2. **Output distribution**: Verify (μ, log_var) shapes match targets

3. **Gradient flow**: Verify gradients flow through noise parameters

### Integration Tests

1. **Train on scm6**: Compare against `SingleCausalForecaster`
2. **Check NLL convergence**: Loss should decrease
3. **Check noise learning**: σ_A, σ_R should move from initial values

### Validation

1. **Uncertainty calibration**: Are confidence intervals well-calibrated?
2. **DAG recovery**: Does noise awareness improve DAG learning?
3. **Cross-validation consistency**: More stable DAG across folds?

---

## Future Extensions

### Implemented but Optional

- `NoiseAwareReversedDecoderLayerV2`: Noise after normalization
- `VariancePropagationTracker`: Tracks variance flow for analysis
- `noise_per_dimension`: Per-embedding-dimension noise

### Not Yet Implemented

- Heavy-tailed noise (Student-t distribution)
- Heteroscedastic propagated noise: `σ_A[j] = g(V_j)`
- Correlated node noise: `Σ = D + UU^T`

---

## References

- `docs/noise_aware_transformer_summary.md` — Conceptual model
- `docs/NOISE_LEARNING.md` — Theoretical background
- `docs/SVFA_ATTENTION.md` — Structure-Value factorization
- PyTorch `GaussianNLLLoss` — Stability best practices
