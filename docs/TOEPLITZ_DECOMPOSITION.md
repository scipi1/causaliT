# Toeplitz Decomposition for Causal Attention

## Overview

Any matrix can be decomposed into symmetric and antisymmetric parts. For attention scores $QK^T$, this decomposition provides a natural separation between **correlation** (symmetric) and **causality** (antisymmetric).

## Mathematical Foundation

### Decomposition

$$QK^T = \underbrace{\frac{QK^T + KQ^T}{2}}_{\text{Symmetric Part } S} + \underbrace{\frac{QK^T - KQ^T}{2}}_{\text{Antisymmetric Part } A}$$

Where:
- $S_{ij} = S_{ji}$ (symmetric: correlation/alignment)
- $A_{ij} = -A_{ji}$ (antisymmetric: direction/flow)
- $A_{ii} = 0$ (diagonal is zero)

### Interpretation

| Component | Mathematical Property | Causal Interpretation |
|-----------|----------------------|----------------------|
| Symmetric $S$ | $S_{ij} = S_{ji}$ | **Edge existence**: Do $i$ and $j$ share information? (undirected) |
| Antisymmetric $A$ | $A_{ij} = -A_{ji}$ | **Flow direction**: Does information flow $i \to j$ or $j \to i$? |

## Connection to LieAttention

The current `LieAttention` uses only the antisymmetric part (Lie commutator):

```python
# Current LieAttention
comm = scores - scores.transpose(-1, -2)  # A = QK^T - KQ^T
comm_amp = torch.tanh((gain / tau) * comm)
scores = F.gelu(comm_amp)
```

This captures **direction** but loses information about **edge existence**.

## Proposed: Gated Causal Attention

### Motivation

The antisymmetric parameterization has a fundamental limitation:

$$P(i \to j) + P(j \to i) = 1$$

This means **at least one edge must exist** between any pair $(i, j)$. We cannot represent "no edge in either direction."

### Solution: Symmetric Gate

Combine both parts:

$$P(i \to j) = \underbrace{\sigma(\gamma_{ij})}_{\text{Edge exists?}} \times \underbrace{\sigma(\phi_{ij})}_{\text{Direction } i \to j}$$

Where:
- $\gamma_{ij} = \gamma_{ji}$ is derived from the symmetric part $S$
- $\phi_{ij} = -\phi_{ji}$ is derived from the antisymmetric part $A$ (or learnable)

### Properties

| Desired Property | Achieved? |
|-----------------|-----------|
| $P(i \to j) = 0$ and $P(j \to i) = 0$ | ✅ When $\gamma_{ij} \to -\infty$ |
| $P(i \to j) + P(j \to i) \leq 1$ | ✅ Always |
| $P(i \to i) = 0$ | ✅ Force diagonal of $\gamma$ to $-\infty$ |
| Sparsity via regularization | ✅ L1 on $\gamma$ |

## Implementation Options

### Option A: Attention-Derived (Data-Driven)

```python
S = (QK.T + KQ.T) / 2  # Symmetric alignment
gamma = gain * tanh(S / tau)
P_edge = sigmoid(gamma)
```

**Pros**: Emerges from data, no extra parameters
**Cons**: Depends on embedding quality

### Option B: Learnable Symmetric Matrix

```python
G_upper = nn.Parameter(...)  # Upper triangular
gamma = G_upper + G_upper.T  # Symmetric
P_edge = sigmoid(gamma)
```

**Pros**: Full control, can learn arbitrary structure
**Cons**: More parameters

### Option C: Hybrid (Recommended)

```python
S = (QK.T + KQ.T) / 2  # Attention alignment
G_bias = nn.Parameter(...)  # Learnable offset (symmetric)
gamma = S + (G_bias + G_bias.T)
P_edge = sigmoid(gamma)
```

**Pros**: Attention provides inductive bias, learnable offset allows fine-tuning

## Connection to SVFA

When using **Structure-Value Factorized Attention (SVFA)**, the Q and K matrices depend only on variable embeddings (not values). This makes:

1. The symmetric part $S$ depend only on variable structure (global, not sample-specific)
2. Edge existence becomes naturally consistent across samples
3. Potentially eliminates the need for learnable $\phi$

See `docs/SVFA_ATTENTION.md` for details.

## References

- Lie bracket / commutator: [Wikipedia](https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields)
- DAG learning with neural networks: Zheng et al. "DAGs with NO TEARS" (2018)
