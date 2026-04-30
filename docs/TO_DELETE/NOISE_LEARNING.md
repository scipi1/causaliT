# Noise-Aware Transformer with NLL Training

## Motivation

In structural causal models (SCMs), each variable is generated from its parents plus **exogenous noise**:

$$
X_i = f_i(\mathrm{PA}(X_i)) + N_i
$$

where:

* $\mathrm{PA}(X_i)$ are the parents of $X_i$,
* $N_i$ is node-local exogenous noise,
* $N_i \perp N_j$ for $i\neq j$.

In many real systems, **uncertainty propagates through causal mechanisms**:

$$
X_i = f_i(X_j + N_j) + N_i
$$

---

N.B. Not always the case. Consider a temperature sensor $X_t$, a humidity sensor $X_h$ and a set temperature parameter $S_t$. We know that $S_t\to X_t$ and $X_t\to X_h$ but let's try to be more precise by introducing the hidden variable temperature $H_t$. The causal relations are then updated: $S_t\to H_t$, $H_t \to X_t$, $H_t \to X_h$, i.e. it's the true temperature to influence both sensors. The temperature reading is the true temperature + its noise. The humidity reading is the also some function of the true temperature + its noise. Nonetheless, we never observe $H_t$, only its noisy measurement $X_t$. Therefore, it is reasonable to learn a relation from $X_t\to X_h$, when we account the noisy nature of $X_t$. Shall we use two noise terms, one for mixing one for final output?

---

Thus downstream variables inherit **transformed upstream uncertainty** plus **local innovation noise**.

Standard transformers do not explicitly represent this uncertainty propagation. The proposed method introduces **noise-aware message passing** within the transformer while keeping the **graph structure deterministic**.

The training objective is a **negative log-likelihood (NLL)** of the observed data under the model’s predictive distribution.

---

# Model Formulation

## Inputs

We assume:

* Source variables $S$
* Covariates $X = (X_1,\dots,X_d)$

The architecture consists of:

1. **Cross-attention:** models $S \rightarrow X$
2. **Self-attention:** models $X \rightarrow X$

The attention structure is deterministic (e.g., **Lie attention mask** enforcing DAG constraints).

Noise only affects **value transmission**, not attention routing.

---

# Noisy Message Passing

Let the value representation of node $j$ be

$$
V_j \in \mathbb{R}^{d_v}
$$

## Propagated Noise

Each transmitted value is perturbed by **propagated noise**

$$
m_j = V_j + \sigma_j^{\text{prop}} \odot \epsilon_j^{\text{prop}}
$$

where

$$
\epsilon_j^{\text{prop}} \sim \mathcal{N}(0,I)
$$

and

$$
\sigma_j^{\text{prop}} \ge 0
$$

is a learned scale.

Interpretation:

* upstream nodes transmit **noisy messages**
* uncertainty propagates through the causal graph

---

## Attention Aggregation

Self-attention weights remain deterministic:

$$
\alpha_{ij} =
\mathrm{softmax}_j
\left(
\frac{Q_i K_j^\top}{\sqrt{d_k}}
\right)
$$

The aggregated representation is

$$
u_i =
\sum_j
\alpha_{ij} m_j
$$

This implements **causal mixing of noisy parent states**.

If the attention mask enforces a DAG:

* parents contribute
* non-parents have zero weight

---

## Local Innovation Noise

After aggregation, node $i$ adds **local innovation noise**:

$$
h_i = u_i + \sigma_i^{\text{loc}} \odot \epsilon_i^{\text{loc}}
$$

with

$$
\epsilon_i^{\text{loc}} \sim \mathcal{N}(0,I)
$$

This corresponds to SCM noise (N_i).

Thus uncertainty arises from:

1. propagated upstream noise
2. node-local noise

---

# Predictive Distribution

The model predicts a **Gaussian output distribution**

$$
X_i \sim \mathcal{N}(\mu_i,\tau_i^2)
$$

with parameters

$$
\mu_i = g_\mu(h_i)
$$

$$
\log \tau_i = g_\sigma(h_i)
$$

where $g_\mu$ and $g_\sigma$ are small MLP heads.

---

# Training Objective

Given an observed sample $x_i^*$, the loss is the **negative log-likelihood**

$$
\mathcal{L}_i
=

\frac{(x_i^*-\mu_i)^2}{2\tau_i^2}
+
\log \tau_i
$$

Total loss:

$$
\mathcal{L}
=

\sum_i \mathcal{L}_i
$$

This objective corresponds to maximizing the predictive density

$$
p_\theta(X|S)
$$

where noise variables are **latent and marginalized**.

---

# Algorithm

For each training batch:

1. **Compute embeddings**

   ```
   S_embed = embed(S)
   X_embed = embed(X)
   ```

2. **Cross-attention**

   ```
   X_cross = CrossAttention(X_embed, S_embed)
   ```

3. **Value projection**

   ```
   V = W_v(X_cross)
   ```

4. **Add propagated noise**

   ```
   eps_prop ~ Normal(0,1)
   M = V + sigma_prop * eps_prop
   ```

5. **Self-attention aggregation**

   ```
   alpha = softmax(QK^T / sqrt(d_k))
   U = alpha @ M
   ```

6. **Add local noise**

   ```
   eps_loc ~ Normal(0,1)
   H = U + sigma_loc * eps_loc
   ```

7. **Predict distribution parameters**

   ```
   mu = head_mu(H)
   log_tau = head_sigma(H)
   tau = exp(log_tau)
   ```

8. **Compute NLL loss**

   ```
   loss =
   ((x_true - mu)**2)/(2*tau**2)
   + log_tau
   ```

---

# Causal Interpretation

The architecture implements a stochastic structural mechanism:

$$
X_i =
f_i
\left(
\sum_j \alpha_{ij}
(X_j + N_j^{\text{prop}})
\right)
+
N_i^{\text{loc}}
$$

Properties:

* **noise propagates through causal edges**
* **local innovation noise remains node-specific**
* **attention structure defines causal mixing**

The correct causal orientation should produce **more coherent uncertainty propagation**, leading to a higher likelihood under the NLL objective.

---

# Practical Recommendations

### Noise parameterization

Use

```
sigma = softplus(raw_sigma)
```

to enforce positivity.

---

### Initialization

Initialize noise scales small:

```
sigma_prop ≈ 0.01
sigma_loc ≈ 0.01
```

so training starts near a deterministic regime.

---

### Stability

Clamp log variance:

```
log_tau ∈ [-10, 5]
```

to prevent exploding likelihood.

---

### Independence assumption

Start with

$$
N_i \perp N_j
$$

(no cross-node noise correlations).

Correlations should arise **through the learned causal structure**.

---

# Advantages

* consistent with **SCM noise propagation**
* preserves **deterministic attention structure**
* learns **predictive uncertainty**
* allows NLL-based statistical training
* may provide additional signal for **causal direction selection**

---

# Possible Extensions

### Heavy-tailed noise

Replace Gaussian with Student-t:

$$
X_i \sim t_\nu(\mu_i,\tau_i)
$$

to model drift or outliers.

---

### Heteroscedastic propagated noise

Make propagated noise depend on parent states:

$$
\sigma_j^{\text{prop}} = g_{\text{prop}}(V_j)
$$

---

### Correlated node noise

Use a low-rank covariance

$$
\Sigma = D + UU^\top
$$

to capture shared disturbances.

---

# Summary

The proposed method introduces **noise-aware message passing** inside a transformer:

1. deterministic attention defines causal mixing
2. noisy values propagate upstream uncertainty
3. nodes add local innovation noise
4. the model predicts a Gaussian distribution
5. training maximizes **negative log-likelihood**

This provides a principled framework for combining **transformer attention**, **structural noise propagation**, and **probabilistic training**.

---
