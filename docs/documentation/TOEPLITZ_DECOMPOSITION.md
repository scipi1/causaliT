# Toeplitz Decomposition for Causal Attention

## Overview

Any matrix can be decomposed into symmetric and antisymmetric parts. For attention scores $QK^T$, this decomposition provides a natural separation between **correlation** (symmetric) and **causality** (antisymmetric).

The `ToeplitzLieAttention` class in `causaliT/core/modules/attention.py` implements this decomposition for DAG learning in causal attention mechanisms.

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

## Current Implementation: `ToeplitzLieAttention`

### Core Formula

The implementation computes edge probabilities as:

$$P(i \to j) = \sigma(\gamma_{ij}) \times \sigma(\phi_{ij})$$

Where:
- $\gamma_{ij}$ = **gate logits** from symmetric part (edge existence)
- $\phi_{ij}$ = **direction logits** from antisymmetric part (flow direction)
- $\sigma$ = sigmoid function

### Gate and Direction Computation

```python
# Toeplitz decomposition
S_sym = (scores + scores.T) / 2      # Symmetric: edge existence
A_antisym = (scores - scores.T) / 2  # Antisymmetric: flow direction (Lie commutator)

# Compute logits with gain-temperature amplification
gamma_att = gain_gate * tanh(S_sym / tau_gate)   # Gate logits
phi_att = gain_dir * tanh(A_antisym / tau_dir)   # Direction logits

# Optional learnable biases (from DAGMaskGated)
gamma_att = gamma_att + gamma_bias  # if available
phi_att = phi_att + phi_bias        # if available

# DAG probabilities
gate_probs = sigmoid(gamma_att)
dir_probs = sigmoid(phi_att)
dag_probs = gate_probs * dir_probs
```

### Key Properties

| Desired Property | How Achieved |
|-----------------|--------------|
| $P(i \to j) = 0$ and $P(j \to i) = 0$ | ✅ When gate is closed: $\gamma_{ij} \to -\infty$ |
| $P(i \to j) + P(j \to i) \leq 1$ | ✅ Always, by construction |
| $P(i \to i) = 0$ | ✅ Diagonal explicitly masked to 0 |
| Sparsity via regularization | ✅ L1 on gate probabilities (`lambda_l1_toeplitz_gate`) |

## Configurable Parameters

### Initialization Parameters

These parameters control the **initial steepness** of the activation functions and affect how quickly probabilities saturate to 0 or 1:

| Parameter | Default | Original | Description |
|-----------|---------|----------|-------------|
| `init_gain_gate` | 2.0 | 5.0 | Initial gain for symmetric gate |
| `init_gain_dir` | 3.0 | 10.0 | Initial gain for direction |
| `init_tau_gate` | 0.5 | 0.5 | Initial temperature for gate |
| `init_tau_dir` | 0.3 | 0.2 | Initial temperature for direction |
| `max_gain` | 20.0 | 100.0 | Maximum allowed gain during training |

**Effective slope** = `gain / tau`

The higher the effective slope, the more the `tanh` function saturates, leading to more decisive (0/1) probabilities.

### Why These Defaults Matter

With the **original hardcoded values**:
- Gate: slope = 5.0 / 0.5 = 10.0
- Direction: slope = 10.0 / 0.2 = 50.0 → **Very steep!**

After temperature annealing (tau → 0.1):
- Direction: slope = 10.0 / 0.1 = 100.0 → **Extremely steep!**

This caused `sigmoid(±100) ≈ 0 or 1`, making all edges fully decisive.

With the **new conservative defaults**:
- Gate: slope = 2.0 / 0.5 = 4.0
- Direction: slope = 3.0 / 0.3 = 10.0

After temperature annealing (tau → 0.2):
- Direction: slope = 3.0 / 0.2 = 15.0 → **Moderate steepness**

This allows probabilities to remain in an intermediate range when the data doesn't provide strong evidence.

### Temperature Annealing Parameters

These are configured in the **training section** of the config and applied by the forecaster during training:

| Parameter | Recommended | Original | Description |
|-----------|-------------|----------|-------------|
| `tau_gate_end` | 0.3 | 0.2 | Final gate temperature |
| `tau_dir_end` | 0.2 | 0.1 | Final direction temperature |
| `tau_gs_end` | 0.4 | 0.2 | Final Gumbel-Softmax temperature |

## Configuration Example

### Model Configuration (in `model.kwargs`)

```yaml
# ToeplitzLieAttention parameters
# Lower gains and higher temperatures = more uncertain edge probabilities
toeplitz_init_gain_gate: 2.0   # Symmetric gate gain
toeplitz_init_gain_dir: 3.0    # Direction gain
toeplitz_init_tau_gate: 0.5    # Gate temperature
toeplitz_init_tau_dir: 0.3     # Direction temperature
toeplitz_max_gain: 20.0        # Max gain during training
```

### Training Configuration (in `training`)

```yaml
# Temperature annealing
use_tau_act_annealing: true
tau_gate_start: 1.0
tau_gate_end: 0.3      # Higher = more uncertainty in final DAG
tau_dir_start: 0.5
tau_dir_end: 0.2       # Higher = more uncertainty in final DAG
tau_act_anneal_epochs: 50

# Gumbel-Softmax annealing
use_tau_gs_annealing: true
tau_gs_start: 2.0
tau_gs_end: 0.4        # Higher = more stochastic sampling
tau_gs_anneal_epochs: 50

# Sparsity regularization on symmetric gate
lambda_l1_toeplitz_gate: 0.1
```

## Architecture Flow

```
Input: Q, K, V tensors

1. Compute raw scores
   scores = scale * Q @ K.T

2. Toeplitz decomposition
   S = (scores + scores.T) / 2   # Symmetric
   A = (scores - scores.T) / 2   # Antisymmetric

3. Gate logits (from symmetric part)
   gamma = gain_gate * tanh(S / tau_gate) + gamma_bias

4. Direction logits (from antisymmetric part)
   phi = gain_dir * tanh(A / tau_dir) + phi_bias

5. DAG probabilities
   gate_probs = sigmoid(gamma)
   dir_probs = sigmoid(phi)
   dag_probs = gate_probs * dir_probs

6. Gumbel-Softmax sampling (training only)
   M = sample(gamma, phi, tau_gs)

7. Attention with DAG mask
   att = relu(tanh(A * gain_dir / tau_dir)) * M

8. Output
   V_out = att @ V
```

## Connection to SVFA

When using **Structure-Value Factorized Attention (SVFA)**, the Q and K matrices depend only on variable embeddings (not values). This makes:

1. The symmetric part $S$ depend only on variable structure (global, not sample-specific)
2. Edge existence becomes naturally consistent across samples
3. The attention structure is decoupled from the values being mixed

See `docs/SVFA_ATTENTION.md` for details.

## Comparison with Other Attention Types

| Attention Type | DAG Learning | Edge Existence | Direction | Use Case |
|----------------|--------------|----------------|-----------|----------|
| `ScaledDotAttention` | No | N/A | N/A | Standard transformer attention |
| `LieAttention` | Yes | Implicit | Via commutator | Self-attention DAG |
| `CausalCrossAttention` | Yes | Via phi | Via phi | Cross-attention DAG |
| `ToeplitzLieAttention` | Yes | **Via symmetric gate** | **Via antisymmetric** | **Recommended for self-attention DAG** |
| `PhiSoftMax` | Yes | Via phi | Via phi | Softmax + learned DAG |

## Regularization Options

| Regularizer | Purpose | Applies To |
|-------------|---------|------------|
| `lambda_l1_toeplitz_gate` | Encourage sparse gate (fewer edges) | Gate probabilities |
| `lambda_entropy_self` | Encourage focused attention | Attention weights |
| `kappa` | NOTEARS acyclicity constraint | Learned DAG |
| `lambda_hsic` | Decorrelate residuals from sources | Residuals vs S |

## References

- Lie bracket / commutator: [Wikipedia](https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields)
- DAG learning with neural networks: Zheng et al. "DAGs with NO TEARS" (2018)
- Toeplitz matrices: [Wikipedia](https://en.wikipedia.org/wiki/Toeplitz_matrix)
