# Noise-Aware Causal Transformer

## Overview

The noise-aware transformer introduces explicit noise modeling for uncertainty
quantification in causal structure learning. It models two distinct sources of
uncertainty — **ambient (process) noise** and **reading (measurement) noise** —
while keeping the attention structure deterministic.

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
| 2 | Ambient noise | H = H_det + σ_A · ε — environmental variability |
| 3 | Self-attention | U = SelfAtt(X_struct, X_struct, H) — mixing noisy states |
| 4 | Output head | (μ, log τ) = head(U) — includes reading noise σ_R |

### Full Generative Model

$$X_i = \sum_j \alpha_{ij} (H_{\text{det},j} + \sigma_{A,j} \cdot \varepsilon_j) + \sigma_{R,i} \cdot \eta_i$$

with $\varepsilon_j, \eta_i \sim \mathcal{N}(0, 1)$.

### Key Design Principles

1. **Attention structure remains deterministic** — noise does not affect Q, K
2. **Noise affects values only** — V in self-attention uses noisy H
3. **Variance propagates through causal mixing**: $\text{Var}(X_i) = \sum_j \alpha_{ij}^2 \sigma_{A,j}^2 + \sigma_{R,i}^2$
4. **SVFA required** — Structure-Value Factorized Attention for clean separation

---

## Motivation

In structural causal models (SCMs), each variable is generated from its parents
plus exogenous noise:

$$X_i = f_i(\text{PA}(X_i)) + N_i$$

In many real systems, uncertainty propagates through causal mechanisms: downstream
variables inherit transformed upstream uncertainty plus local innovation noise.
Standard transformers do not explicitly represent this propagation. The noise-aware
architecture introduces noise-aware message passing while keeping the graph
structure deterministic.

---

## Architecture

### Noise-Aware Decoder Layer

```python
# 1. Cross-attention (deterministic)
H_det = CrossAttention(X_struct, S, S)

# 2. Ambient noise injection
eps_A ~ Normal(0, 1)
H = H_det + sigma_A * eps_A

# 3. Self-attention mixing
U = SelfAttention(X_struct, X_struct, H)  # Q,K from X_struct, V from H

# 4. Predictive distribution
mu = head_mu(U)
log_tau = head_sigma(U)
tau = exp(log_tau)
```

### Variance Propagation

Downstream nodes inherit upstream uncertainty through attention weights:

$$\text{Var}(X_i) = \sum_j \alpha_{ij}^2 \sigma_{A,j}^2 + \sigma_{R,i}^2$$

This directional variance propagation provides a statistical asymmetry between
forward and backward models, potentially useful for causal direction identification.

---

## Training

### Loss Function

Gaussian Negative Log-Likelihood:

$$L = \frac{(x - \mu)^2}{2\tau^2} + \log \tau$$

The $\log \tau$ term penalizes unnecessarily large variance.

### Stability Measures

| Measure | Implementation |
|---------|---------------|
| Variance clamping | `var = exp(log_var).clamp(min=1e-6)` |
| Log-variance bounds | `log_var.clamp(-10, 5)` |
| Positivity | `σ = exp(log_σ)` parameterization |
| Small initialization | `σ_A ≈ 0.01`, `σ_R ≈ 0.05` (near-deterministic start) |

### Regularizers Supported

All regularizers from `SingleCausalForecaster` are supported: entropy,
NOTears acyclicity, sparsity, HSIC, decisiveness, KL prior, L1 on scores.

---

## Implementation

### Files

```
causaliT/core/modules/noise_layers.py          # AmbientNoiseLayer, ReadingNoiseHead, GaussianNLLLoss
causaliT/core/architectures/noise_aware/
├── model.py                                   # NoiseAwareSingleCausalLayer
└── decoder.py                                 # NoiseAwareReversedDecoderLayer (+V2)
causaliT/training/forecasters/noise_aware_forecaster.py  # NoiseAwareCausalForecaster
causaliT/config/templates/config_noise_aware.yaml         # Example config
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `AmbientNoiseLayer` | Injects per-node ambient noise σ_A[j] |
| `ReadingNoiseHead` | Probabilistic output head with reading noise σ_R[i] |
| `GaussianNLLLoss` | Gaussian NLL with stability features |
| `NoiseAwareSingleCausalLayer` | Main model with noise injection |
| `NoiseAwareReversedDecoderLayer` | Decoder layer with noise between cross/self attention |
| `NoiseAwareCausalForecaster` | Lightning wrapper with Gaussian NLL training |

### Design Decisions

1. **Per-node noise**: σ_A[j] and σ_R[i] are node-specific learnable parameters.
   Different sensors have different precision. Per-dimension noise available via
   `noise_per_dimension: true`.

2. **Noise injection point**: Ambient noise injected in embedding space, before
   W_v projection. Aligns with conceptual model: H_det → H happens before
   attention aggregation. Alternative (`NoiseAwareReversedDecoderLayerV2`)
   injects after normalization.

3. **SVFA requirement**: Clean separation between Q,K (structure, deterministic)
   and V (value, noisy). Attention pattern doesn't change based on noise realizations.

---

## Configuration

```yaml
model:
  model_object: "NoiseAwareSingleCausalLayer"
  kwargs:
    init_sigma_A: 0.01
    init_sigma_R: 0.05
    noise_per_dimension: false
    track_variance: false

training:
  nll_eps: 1.0e-6
  nll_full: false
  log_noise_params: true
  lambda_noise_prior: 0.0       # Optional prior for identifiability
```

### Inference with Uncertainty

```python
mu, std = forecaster.predict(S, X)
mu, lower, upper = forecaster.predict_with_intervals(S, X, confidence=0.95)
noise_params = forecaster.get_noise_parameters()
```

### Logged Metrics

| Metric | Description |
|--------|-------------|
| `*_nll` | Gaussian NLL loss |
| `*_x_mae`, `*_x_rmse`, `*_x_r2` | Reconstruction metrics (using mean μ) |
| `*_pred_var_mean`, `*_pred_var_std` | Predicted variance statistics |
| `*_sigma_A_mean`, `*_sigma_A_std` | Ambient noise parameters |
| `*_sigma_R_mean`, `*_sigma_R_std` | Reading noise parameters |

---

## Limitations and Extensions

### Limitations

1. Hidden physical states H are not explicitly identifiable
2. Ambient noise and structural errors may trade off
3. Noise parameters require regularization

### Implemented but Optional

- `NoiseAwareReversedDecoderLayerV2`: Noise after normalization
- `VariancePropagationTracker`: Tracks variance flow for analysis
- `noise_per_dimension`: Per-embedding-dimension noise

### Not Yet Implemented

- Heavy-tailed noise (Student-t distribution)
- Heteroscedastic propagated noise: σ_A[j] = g(V_j)
- Correlated node noise: Σ = D + UU^T

---

## References

- `docs/SVFA.md` — Structure-Value Factorized Attention
- `docs/SVFA_DUAL_RESIDUAL.md` — Dual-residual SVFA variant
- `docs/STAGED_TRAINING.md` — Calibration and staged training pipeline