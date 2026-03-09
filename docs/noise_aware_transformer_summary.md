
# Noise-Aware Transformer with Ambient and Reading Noise

## Goal

Introduce a **probabilistic noise model** inside the transformer architecture that reflects realistic physical systems where:

1. **Underlying physical quantities are not perfectly deterministic functions of inputs**
2. **Environmental disturbances affect the true state**
3. **Sensors add additional measurement noise**

The architecture therefore models two distinct sources of uncertainty:

| Noise | Meaning |
|------|------|
| **Ambient noise** (`σ_A`) | variability in the true physical state |
| **Reading noise** (`σ_R`) | measurement/sensor noise |

The transformer remains structurally deterministic; noise only affects **value transmission and final observation**.

---

# Conceptual Model

We assume an underlying causal mechanism

```
S → H → X
```

where

| Variable | Meaning |
|--------|--------|
| `S` | upstream control variables (sources) |
| `H` | underlying physical quantities (not explicitly modeled) |
| `X` | observed sensor readings |

We do **not introduce explicit latent variables**. Instead, we reinterpret intermediate transformer representations as approximations of these quantities.

---

# Deterministic Hidden Representation

After cross-attention we obtain a deterministic representation

```
H_det,j = f_j(S)
```

which represents the **deterministic component of the physical quantity** associated with node `j`.

This corresponds to the transformer learning

```
P(S → H_det)
```

---

# Ambient (Process) Noise

Real systems exhibit environmental variability.

We model this as **ambient noise** added to the deterministic hidden state:

```
H_j = H_det,j + σ_A ε_j
```

where

```
ε_j ~ N(0,1)
```

Interpretation:

| Component | Meaning |
|----------|---------|
| `H_det` | deterministic physical relation |
| `σ_A` | environmental variability |
| `H` | actual physical state |

This noise represents:

- ambient temperature fluctuations
- environmental disturbances
- imperfect actuation
- system drift

---

# Structural Mixing (Self-Attention)

Nodes interact through deterministic attention weights

```
α_ij = softmax_j( (Q_i K_j^T) / sqrt(d_k) )
```

Self-attention aggregates the **noisy physical states**

```
U_i = Σ_j α_ij H_j
```

Important property:

**mixing occurs before measurement noise**.

Thus downstream nodes respond to the **true (noisy) physical state**, not to sensor noise.

---

# Reading (Measurement) Noise

Sensors introduce additional noise.

We model the observed variable as

```
X_i = U_i + σ_R η_i
```

where

```
η_i ~ N(0,1)
```

Interpretation:

| Noise | Meaning |
|------|------|
| `σ_A` | uncertainty in the physical system |
| `σ_R` | sensor measurement noise |

---

# Full Generative Model

Combining all components:

```
X_i = Σ_j α_ij ( H_det,j + σ_A ε_j ) + σ_R η_i
```

This represents:

1. deterministic physics
2. ambient disturbances
3. causal mixing
4. measurement noise

---

# Special Case: Root Nodes

If node `j` has no parents in the self-attention graph:

```
U_j = H_j
```

Thus

```
X_j = H_det,j + σ_A ε_j + σ_R η_j
```

Meaning root sensors simply observe the physical state with combined noise.

---

# Variance Propagation

Downstream nodes inherit upstream uncertainty:

```
Var(X_i) = Σ_j α_ij² σ_A² + σ_R²
```

Thus causal edges propagate variance through the system.

This property may provide **additional statistical signal for identifying causal direction**.

---

# Training Objective

The model predicts a Gaussian distribution

```
X_i ~ N( μ_i , τ_i² )
```

with

```
μ_i = g_μ(U_i)
log τ_i = g_σ(U_i)
```

Training minimizes the **negative log likelihood**

```
L_i = ((x_i - μ_i)²)/(2 τ_i²) + log τ_i
```

Total loss

```
L = Σ_i L_i
```

---

# Implementation Outline

### 1. Cross-Attention

```python
H_det = CrossAttention(X_embed, S_embed)
```

### 2. Ambient Noise

```python
eps_A = torch.randn_like(H_det)
H = H_det + sigma_A * eps_A
```

### 3. Self-Attention Mixing

```python
U = SelfAttention(H)
```

### 4. Predict Distribution

```python
mu = head_mu(U)
log_tau = head_sigma(U)
tau = torch.exp(log_tau)
```

### 5. NLL Loss

```python
loss = ((x_true - mu)**2)/(2*tau**2) + log_tau
```

---

# Noise Parameterization

Ensure positivity with

```python
sigma = torch.nn.functional.softplus(raw_sigma)
```

Suggested initialization

```
σ_A ≈ 0.01
σ_R ≈ 0.05
```

---

# Key Design Principles

1. **Attention structure remains deterministic**
2. **Noise affects values, not keys**
3. **Ambient noise models system variability**
4. **Reading noise models measurement uncertainty**
5. **Variance propagates through causal mixing**

---

# Limitations

1. Hidden physical states `H` are **not explicitly identifiable**
2. Ambient noise and structural errors may trade off
3. Noise parameters require **regularization**

---

# Future Research Direction

Because causal mixing propagates ambient noise, downstream nodes inherit upstream variance:

```
Var(X_i) = Σ_j α_ij² σ_A² + σ_R²
```

This directional variance propagation may provide a **statistical asymmetry between forward and backward models**, potentially useful for:

- causal discovery
- direction identification
- structural validation of learned graphs
