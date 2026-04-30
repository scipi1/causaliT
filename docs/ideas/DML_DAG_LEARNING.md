# Double Machine Learning for DAG Learning

## Overview

This document proposes leveraging **Double Machine Learning (DML)** principles to make causal DAG learning more robust to nuisance parameter estimation in the causaliT framework, specifically for architectures where the DAG structure **emerges** from learned components (e.g., Toeplitz + SVFA) rather than being a simple parametric φ.

---

## Background: Double Machine Learning

DML (Chernozhukov et al., 2018) addresses the problem of estimating a **target parameter θ** when there are **nuisance parameters η** that must also be estimated:

1. **Cross-fitting**: Split data, estimate η on one fold, estimate θ on the other
2. **Orthogonalization**: Construct moment conditions such that ∂ψ/∂η = 0 at the true η
3. **Result**: √n-consistent estimates of θ even when η is estimated at slower rates

**Key insight**: By making the score function **Neyman-orthogonal**, small errors in nuisance estimation don't propagate to the target parameter.

---

## The Problem: Emergent DAG in Toeplitz + SVFA

### Where is the DAG?

With **Toeplitz Self-Attention + SVFA**, the DAG probabilities are:

$$P(i \to j) = \sigma(\gamma_{ij}) \cdot \sigma(\phi_{ij})$$

Where:
```python
# From ToeplitzLieAttention
scores = scale * Q_s @ K_s.T  # Q_s, K_s from structure embeddings
S_sym, A_antisym = (scores + scores.T)/2, (scores - scores.T)/2

gamma_att = gain_gate * tanh(S_sym / tau_gate)   # Gate from symmetric part
phi_att = gain_dir * tanh(A_antisym / tau_dir)   # Direction from antisymmetric part
```

With SVFA:
```python
Q_s = W_q @ E_structure
K_s = W_k @ E_structure
```

### The DAG is Emergent, Not Parametric

The DAG is a **functional** of learned components:

$$\text{DAG} = g(E_{\text{struct}}, W_q, W_k, \text{gains}, \text{taus})$$

This is fundamentally different from a simple learnable parameter φ.

---

## DML Reformulation for Emergent DAG

### Target vs. Nuisance Parameters

| Component | Role | Classification |
|-----------|------|----------------|
| Edge probability $P(i \to j)$ | What we want to learn | **Target** |
| Structure embeddings $E_{\text{struct}}$ | Learned for prediction | **Nuisance** |
| Projections $W_q, W_k$ | Learned for prediction | **Nuisance** |
| Value embeddings, noise params | Auxiliary | **Nuisance** |

### The Key Insight

Even though the DAG emerges from embeddings + projections, the **information used to determine edge $(i \to j)$** should be **orthogonal** to the **information used to predict variable $j$**.

**Problem**: In standard training, all components are **jointly optimized** for prediction loss. This creates coupling:
- $E_{\text{struct}}[i]$ affects all edges involving variable $i$
- $W_q, W_k$ affect all edges globally  
- A prediction error on variable $k$ can change embeddings for all variables

---

## Proposed Solution: Cross-Fitted DAG Learning

### ⚠️ Critical Question: What is the Supervisory Signal?

Before presenting the algorithm, we must address a fundamental question:

> **When we exclude variable $j$ from the loss, what supervisory signal trains the embeddings to learn edges INTO $j$?**

This is the key tension in the proposal. Let's analyze it carefully.

### The Signal Problem

In Toeplitz + SVFA, the edge $P(k \to j)$ depends on:
- $E_{\text{struct}}[k]$ — embedding of potential parent $k$
- $E_{\text{struct}}[j]$ — embedding of target $j$
- $W_q, W_k$ — projection matrices

If we **exclude $j$ from the prediction loss**:
- $E_{\text{struct}}[j]$ gets **no direct gradient** for predicting $j$
- However, $E_{\text{struct}}[j]$ **does get gradient** if $j$ is a parent of other variables!

| Scenario | Gradient Signal for $E_{\text{struct}}[j]$ |
|----------|------------------------------------------|
| $j$ has children ($j \to m$) | ✅ Learns to be a good "key" for $m$'s query |
| $j$ has no children (leaf) | ❌ **No signal at all!** |

### The Core Insight

The DML intuition is:
1. **Embeddings for potential parents $k$** are trained to predict other variables (including $j$'s siblings/ancestors)
2. **These embeddings encode "how variable $k$ relates to the causal structure"**
3. **When we then evaluate $P(k \to j)$**, the embedding $E_{\text{struct}}[k]$ reflects $k$'s general causal role, not specifically tuned to predict $j$

But this only works if:
- The causal graph is **sufficiently connected**
- Each variable gets gradient signal from **at least some** prediction tasks
- The **projection matrices** $W_q, W_k$ are trained on the full loss (not excluded)

### Refined Core Principle

The cross-fitting should be understood as:

> **Learn the embedding of potential parent $k$ using prediction tasks that don't directly involve $k \to j$, then evaluate whether this embedding supports edge $k \to j$.**

This is closer to the residualization idea in DML: we're asking "does $k$'s embedding, learned from other tasks, have predictive power for $j$?"

### What About $E_{\text{struct}}[j]$? (The Target's Embedding)

Two options:

**Option A: Freeze $E_{\text{struct}}[j]$ completely**
- Don't update $j$'s embedding during the "nuisance" phase
- Evaluate edges using the initial/random embedding for $j$
- **Problem**: May miss structure if $j$'s role requires learning

**Option B: Train $E_{\text{struct}}[j]$ as a parent only**
- $E_{\text{struct}}[j]$ gets gradients only from predictions of $j$'s children
- This is what happens naturally when excluding $j$ from loss
- **Problem**: If $j$ is a leaf node, no gradient at all

**Option C: Separate training for $E_{\text{struct}}[j]$**
- First train $E_{\text{struct}}[j]$ using $j$'s descendants' predictions
- Then freeze and evaluate parent edges
- **Problem**: Requires knowing descendants a priori (circular!)

### Revised Algorithm: Partial Exclusion

Given these considerations, the algorithm should be:

```python
def cross_fitted_dag_learning(model, data):
    """
    Learn DAG structure with variable-level cross-fitting.
    
    Key insight: For edge k→j, we want E_struct[k] trained without
    optimizing specifically for j's prediction. However, W_q, W_k
    and other structural params ARE trained on the full loss.
    
    The orthogonalization is PARTIAL: we decouple the embedding
    learning from the specific edge being evaluated.
    """
    n_vars = model.n_variables
    edge_scores = torch.zeros(n_vars, n_vars)
    
    # === PHASE 1: Full training to get W_q, W_k, gains, taus ===
    # These are "global" parameters that should see all prediction tasks
    model_full = train_full_model(data)
    
    # Freeze projection matrices (nuisance parameters)
    W_q_frozen = model_full.W_q.detach()
    W_k_frozen = model_full.W_k.detach()
    
    for j in range(n_vars):  # For each target variable
        
        # === PHASE 2a: Re-train ONLY embeddings, excluding j from loss ===
        # This learns E_struct[k] for k ≠ j without j's supervision
        model_excl_j = clone_model(model_full)
        model_excl_j.W_q.requires_grad = False  # Freeze projections
        model_excl_j.W_k.requires_grad = False
        
        optimizer = optim.Adam([model_excl_j.structure_embeddings.parameters()])
        for epoch in range(nuisance_epochs):
            loss = prediction_loss(model_excl_j, data, target_vars=exclude(j))
            loss.backward()
            optimizer.step()
        
        # === PHASE 2b: Evaluate edges INTO j using cross-fitted embeddings ===
        E_struct_excl_j = model_excl_j.structure_embeddings.weight.detach()
        
        with torch.no_grad():
            # Use frozen projections + cross-fitted embeddings
            Q_j = W_q_frozen @ E_struct_excl_j[j]
            K_all = W_k_frozen @ E_struct_excl_j
            
            # Toeplitz decomposition
            scores = Q_j.unsqueeze(0) @ K_all.T
            S = (scores + scores.T) / 2
            A = (scores - scores.T) / 2
            
            gamma_j = model_full.gain_gate * torch.tanh(S[:, j] / model_full.tau_gate)
            phi_j = model_full.gain_dir * torch.tanh(A[:, j] / model_full.tau_dir)
            
            edge_scores[:, j] = torch.sigmoid(gamma_j) * torch.sigmoid(phi_j)
    
    return edge_scores
```

### ⚠️ Fundamental Limitation: The Disconnected Parent Problem

**Concrete Example** (raised during review):

Consider the following graph:
```
S1 ──────────────────→ X2    (S1 is parent of X2 only)
S2, S3 ───→ X1               (other sources are parents of X1)
           (no edge X1 ↔ X2)
```

**When we exclude X2 from the loss to learn edges INTO X2**:

| Variable | Gradient Signal When X2 Excluded |
|----------|----------------------------------|
| E_struct[S1] | ❌ **NONE** — S1 doesn't contribute to X1 |
| E_struct[S2] | ✅ From predicting X1 |
| E_struct[S3] | ✅ From predicting X1 |
| E_struct[X1] | ✅ From X1's prediction (self) |
| E_struct[X2] | ❌ Excluded from loss, no children |

**The problem**: How can we learn S1 → X2 if:
1. X2 is excluded from loss (can't use X2's prediction to learn S1's role)
2. S1 doesn't contribute to X1 (no gradient for S1's embedding from X1)
3. X2 has no children (no indirect signal through descendants)

**Answer: We can't!** This is a fundamental limitation.

### Why Cross-Fitting Fails Here

The DML analogy breaks down because:

| Standard DML | Our Problem |
|--------------|-------------|
| Outcome Y provides supervision | Supervision IS predicting each variable |
| Nuisances are confounders | Nuisances are the same embeddings that determine DAG |
| Cross-fitting removes confounding | Cross-fitting removes ALL signal for disconnected edges |

**The hope was**: Information about edge k → j comes indirectly through other prediction tasks.

**Reality**: If k's only role is as a parent of j, and we exclude j, there's NO signal for k at all.

### When Cross-Fitting Works vs. Fails

| Graph Structure | Cross-Fitting | Why |
|-----------------|---------------|-----|
| Fully connected | ✅ Works | Every variable gets signal from multiple tasks |
| Dense DAG | ⚠️ Partial | Most edges can be learned, some may be missed |
| Sparse DAG | ❌ Fails | Many edges have disconnected parents |
| Hub-spoke (S → all X) | ⚠️ Depends | If each S affects multiple X, may work |
| Chain (S1→X1→X2) | ❌ Fails | Excluding X2 loses signal for X1→X2 edge |

### Conclusion: Cross-Fitting is NOT the Right Approach

The variable-exclusion cross-fitting approach fundamentally breaks down because:
1. The supervisory signal for edge k → j **must** come from j's prediction
2. Excluding j removes exactly the signal we need
3. Indirect signal only exists in highly connected graphs

### Comparison with Standard DML

| Aspect | Standard DML | Our Proposal |
|--------|--------------|--------------|
| Target | Treatment effect θ | Edge probability $P(k \to j)$ |
| Nuisance | E[Y\|X], E[T\|X] | Embeddings trained without $j$ |
| Residualization | Y - E[Y\|X] | Implicit via embedding exclusion |
| Supervisory signal | Outcome Y | **Other variables' predictions** |
| Orthogonality | Exact (Neyman) | **Approximate** |

---

## Orthogonality Condition

In DML, we want the score function $\psi(\theta, \eta)$ to satisfy:

$$\frac{\partial \psi}{\partial \eta}\bigg|_{\eta_0} = 0$$

For emergent DAG learning:
- **θ** = edge probability $P(i \to j)$  
- **η** = embeddings/projections learned for predicting **other variables**
- **Condition**: The score for edge $(i \to j)$ should not depend (at first order) on how well we predict variables other than $j$

### SVFA Provides Partial Orthogonalization

With SVFA, attention patterns (which determine DAG) are **already orthogonal** to value embeddings:
- $Q_s, K_s$ depend only on `E_structure`
- Value predictions depend on `E_value` and `W_v`

The remaining coupling is **within the structure side**: embeddings trained jointly for all variables.

---

## Implementation Variants

### Variant A: Full Variable Cross-Fitting

Train separate embedding sets for each target variable (most rigorous, most expensive):

```python
for target_var in range(n_vars):
    # Train model excluding target_var loss
    model_cv = train_excluding_variable(data, target_var)
    
    # Evaluate DAG edges to target_var
    dag[:, target_var] = evaluate_edges(model_cv, target_var)
```

**Pros**: Cleanest orthogonalization  
**Cons**: $O(n_{\text{vars}})$ training runs

### Variant B: Leave-One-Out Embedding Update

Train once, but compute DAG with leave-one-out embedding correction:

```python
# Train full model
model = train_full(data)

for target_var in range(n_vars):
    # Approximate: what would embeddings be without target_var loss?
    E_corrected = embedding_influence_correction(model, target_var)
    
    # Evaluate edges using corrected embeddings
    dag[:, target_var] = evaluate_edges_with_embeddings(model, E_corrected, target_var)
```

**Pros**: Single training run  
**Cons**: Requires influence function approximation

### Variant C: Alternating Optimization

Alternate between nuisance training and DAG evaluation phases:

```python
for phase in range(n_phases):
    # Phase 1: Fix DAG structure, train nuisance parameters
    for epoch in nuisance_epochs:
        loss = prediction_loss(model, data)
        loss.backward()
        # Update W_q, W_k, W_v, noise params
        # Do NOT update DAG-specific params (or update with smaller LR)
    
    # Phase 2: Fix nuisances, evaluate/update DAG
    with torch.no_grad():
        dag_scores = compute_dag_from_embeddings(model)
    
    # Optionally: apply DAG-specific regularization
```

**Pros**: Practical for training loop integration  
**Cons**: Approximate orthogonalization

---

## Connection to Existing Architecture

### Current Training Flow

```
Forward: E_struct → (W_q, W_k) → QK^T → Toeplitz → DAG → Attention → Prediction
                                          ↑
                                    Emerges here

Backward: Prediction Loss → All parameters updated jointly
```

### Proposed DML-Enhanced Flow

```
Phase 1 (Nuisance): 
    E_struct → ... → Prediction Loss (excluding var j)
    ↓
    Update E_struct, W_q, W_k
    
Phase 2 (DAG Evaluation):
    Frozen E_struct → QK^T → Toeplitz → DAG scores for var j
    ↓
    No gradient (pure evaluation)
```

---

## Theoretical Justification

### Why Cross-Fitting on Variables (Not Samples)?

For DAG learning, the relevant structure is across **variables**, not samples:
- Each variable has a fixed position in the DAG
- Sample-level variation is noise around the true structure
- Cross-fitting should ensure DAG(i→j) is evaluated without optimizing for j

### When Does This Help?

DML-style cross-fitting helps when:
1. **Confounding exists**: Prediction quality for j influences estimated DAG structure
2. **Nuisance estimation is imperfect**: Embeddings don't perfectly capture true structure
3. **Limited data**: Joint optimization leads to overfitting

### When Might It Not Help?

1. **Perfect identification**: If true DAG is identifiable from attention patterns alone
2. **Unlimited data**: Joint optimization converges to correct solution
3. **Strong regularization**: Other constraints (HSIC, sparsity) already enforce correct structure

---

## Experimental Protocol

### Hypotheses to Test

**H1**: Cross-fitted DAG estimates have lower variance across folds than jointly-trained estimates.

**H2**: Cross-fitted DAG estimates are closer to ground truth on SCM datasets.

**H3**: Cross-fitted estimates are more stable under different random seeds.

### Proposed Experiments

1. **SCM Datasets** (known ground truth DAG):
   - Compare DAG recovery: joint training vs. DML cross-fitting
   - Metrics: SHD, F1, AUROC for edge detection

2. **Cross-Validation Consistency**:
   - Train on k folds, compare learned DAG across folds
   - Hypothesis: DML reduces fold-to-fold variance

3. **Ablation**:
   - Full cross-fitting vs. partial (only embeddings) vs. none
   - Measure computational cost vs. DAG quality tradeoff

---

## Configuration (Proposed)

```yaml
training:
  # DML-style cross-fitting for DAG learning
  use_dml_dag_learning: true
  dml_mode: "variable_crossfit"  # "variable_crossfit", "alternating", "influence"
  
  # For variable_crossfit mode
  dml_nuisance_epochs: 10       # Epochs per variable for nuisance training
  dml_freeze_dag_params: true   # Freeze DAG-specific params during nuisance
  
  # For alternating mode
  dml_nuisance_epochs_per_phase: 5
  dml_dag_eval_every: 10        # Evaluate DAG every N epochs
  
  # Regularization during DML phases
  dml_dag_phase_lr_multiplier: 0.1  # Lower LR for DAG params during nuisance phase
```

---

## Summary

| Aspect | Standard Training | DML-Enhanced |
|--------|------------------|--------------|
| Embedding training | Joint for all vars | Exclude target var |
| DAG evaluation | During training | After nuisance convergence |
| Orthogonality | Partial (SVFA) | Full (cross-fitting) |
| Computational cost | 1x | ~n_vars × |
| Expected benefit | Baseline | Lower variance, better DAG recovery |

### Key Takeaway

For emergent DAG learning (Toeplitz + SVFA), the relevant DML cross-fitting should be across **variables**, not samples. Each variable's incoming edges should be evaluated using embeddings trained without that variable's prediction loss.

---

## References

- Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters."
- Zheng, X., et al. (2018). "DAGs with NO TEARS: Continuous optimization for structure learning."
- Peters, J., et al. (2017). "Elements of Causal Inference."

---

## Implementation Status

- [ ] Define DML training loop variant
- [ ] Implement variable cross-fitting
- [ ] Add configuration options
- [ ] Test on SCM datasets
- [ ] Evaluate computational tradeoffs
- [ ] Compare with baseline Toeplitz + SVFA


