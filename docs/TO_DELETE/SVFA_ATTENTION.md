# Structure-Value Factorized Attention (SVFA)

## Overview

Structure-Value Factorized Attention (SVFA) is a factorization strategy for transformer attention that separates **structural alignment** (which variables interact) from **value aggregation** (what information flows).

## The Problem with Standard Attention

In standard attention, Q, K, and V are all projected from the same fused embedding:

```python
# Standard attention embedding
embedding = f(variable_id) + g(value)  # Fused embedding

# Projections
Q = W_q @ embedding  # Depends on both variable ID and value
K = W_k @ embedding  # Depends on both variable ID and value
V = W_v @ embedding  # Depends on both variable ID and value
```

### Issue for Causal Discovery

The attention pattern $\text{softmax}(QK^T)$ depends on **values**, which are:
- Sample-specific (different values per data point)
- Potentially correlated (even non-causally related variables may have correlated values)

This causes:
1. **Inconsistent DAG learning**: Attention patterns vary across samples
2. **Spurious correlations**: Value correlations may override structural relationships
3. **Fold sensitivity**: Different train/test splits learn different DAG structures

## SVFA Solution

SVFA decouples the projections:

```python
# SVFA embeddings
structure_embedding = f(variable_id)  # Only variable identity
value_embedding = g(value)            # Only realization

# Factorized projections
Q = W_q @ structure_embedding  # Structural alignment
K = W_k @ structure_embedding  # Structural alignment
V = W_v @ value_embedding      # Value aggregation
```

### Key Insight

$$\text{Attention}(Q, K, V) = \underbrace{\text{softmax}(Q_s K_s^T)}_{\text{Structure-dependent}} \cdot \underbrace{V_v}_{\text{Value-dependent}}$$

Where:
- $Q_s, K_s$ depend only on variable structure → **consistent attention patterns**
- $V_v$ depends on values → **information still flows**

## Properties of SVFA

| Property | Standard Attention | SVFA |
|----------|-------------------|------|
| Attention pattern depends on values? | Yes | **No** |
| Consistent across samples? | No | **Yes** |
| Consistent across folds? | Often no | **Yes** |
| Can still learn variable interactions? | Yes | Yes |
| Value information flows? | Yes | Yes |

## Implementation in CausaliT

### Current Implementation

In `AttentionLayer` (attention.py), SVFA is enabled by the caller passing different embeddings:

```python
# In encoder/decoder layers
if use_svfa:
    # Query/Key: structure embeddings only
    # Value: value embeddings
    out, attn, ent = attention_layer(
        query=structure_embedding,  # For Q projection
        key=structure_embedding,    # For K projection
        value=value_embedding,      # For V projection
        ...
    )
else:
    # Standard: same embedding for all
    out, attn, ent = attention_layer(
        query=fused_embedding,
        key=fused_embedding,
        value=fused_embedding,
        ...
    )
```

### Embedding Design for SVFA

For SVFA to work, the model needs separate embedding paths:

```python
# Structure embedding: Variable identity only
structure_emb = nn.Embedding(num_variables, d_model)

# Value embedding: Realization only
value_emb = nn.Linear(1, d_model)  # Project scalar value to d_model

# Usage
struct = structure_emb(variable_ids)    # (B, L, d_model)
val = value_emb(values)                  # (B, L, d_model)

# For SVFA attention
Q = W_q(struct)  # Structure only
K = W_k(struct)  # Structure only
V = W_v(val)     # Value only
```

## Connection to DAG Learning

### Why SVFA Matters for Causal Discovery

With SVFA:
1. **Attention patterns become global**: $Q_s K_s^T$ is the same for all samples
2. **DAG structure is consistent**: The learned DAG doesn't vary by sample
3. **Learnable phi may be redundant**: If attention patterns are already global, why add learnable parameters?

### Hypothesis H13

> **Learnable phi doesn't help if SVFA is used**

**Rationale**: 
- Without SVFA: Attention patterns are noisy (value-dependent), so learnable phi provides a stable DAG structure
- With SVFA: Attention patterns are already stable (structure-dependent), making phi redundant

**Test**:
- Compare `SVFA + phi` vs `SVFA + no_phi`
- If similar DAG recovery → phi is redundant with SVFA
- If phi helps → phi provides additional useful bias

## Connection to Toeplitz Decomposition

When using SVFA with the Toeplitz decomposition:

$$Q_s K_s^T = S + A$$

Both $S$ (symmetric) and $A$ (antisymmetric) become **sample-independent**, making:
- Edge existence (from $S$) globally consistent
- Flow direction (from $A$) globally consistent

See `docs/TOEPLITZ_DECOMPOSITION.md` for details.

## Configuration

In experiment configs:

```yaml
model:
  kwargs:
    use_svfa: true  # Enable SVFA factorization
    
    # SVFA-specific embedding settings
    ds_embed_S:
      # Structure embedding for source variables
      ...
    ds_embed_X:
      # Separate structure and value embeddings for intermediate variables
      ...
```

## Related Work

- Attention mechanisms: Vaswani et al. "Attention Is All You Need" (2017)
- Causal discovery: Peters et al. "Elements of Causal Inference" (2017)
- DAG learning with neural networks: Zheng et al. "DAGs with NO TEARS" (2018)
