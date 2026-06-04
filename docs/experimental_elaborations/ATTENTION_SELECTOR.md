# AttentionSelector: Testing Attention as a Learnable Causal Variable Selector

**Status**: Experimental (foundational test)  
**Architecture**: `AttentionSelectorLayer`  
**Forecaster**: `AttentionSelectorForecaster`  
**Config template**: `causaliT/config/templates/config_attention_selector.yaml`  
**Experiment**: `experiments/0_TESTS/ARCH_attention_selector_DS_scm1/`

---

## 1. Motivation

All architectures in causaliT implicitly assume that cross-attention can learn
causal structure: the attention weights should converge to a sparse matrix that
reflects the DAG adjacency. This assumption has been used to design training
objectives (HSIC, score sparsity, NOTEARS), embedding strategies (orthogonal S
embeddings, SVFA), and gradient routing procedures — but it was never directly
verified in a minimal setting.

This experiment tests the core hypothesis in the most favorable possible
conditions, stripping away all architectural complexity.

**Research question:**
> Can a single cross-attention block — with X tokens as queries and [S, X]
> actual values as keys — act as a learnable variable selector that recovers
> causal parent sets from observational data, when trained with MSE + HSIC +
> score sparsity?

A positive result here is a *necessary condition* for any downstream
architecture to work. A negative result identifies fundamental limitations in
the attention + HSIC training recipe.

---

## 2. Architecture

### 2.1 Forward Pass

```
Inputs:
  S  ∈ ℝ^{B × L_S × F}   (exogenous source variables, actual values)
  X  ∈ ℝ^{B × L_X × F}   (endogenous target variables, actual values)

Step 1 — Embed queries (blanked X):
  x_blanked[:, :, val_idx] = 0
  Q = embedding_X(x_blanked)        (B, L_X, d_model)   ← identity only

Step 2 — Embed keys/values (actual S and X):
  K_S = embedding_S(S)              (B, L_S, d_model)   ← identity + value
  K_X = embedding_X(X)              (B, L_X, d_model)   ← identity + value
  K   = cat([K_S, K_X], dim=1)      (B, L_S+L_X, d_model)

Step 3 — Combined cross-attention with mask:
  mask  = (L_X, L_S + L_X)
         [ ones(L_X, L_S)  |  eye_complement(L_X) ]
         S-block: all 1s   |  X-block: off-diagonal 1s (diagonal = 0)

  A, att_weights = CausalCrossAttention(Q, K, K, mask=mask)
                                     (B, L_X, L_S+L_X)

Step 4 — Residual + Norm + FFN + Norm:
  x = norm1(Q + dropout(A))
  x = x + dropout(FFN(x))
  x = norm2(x)  [if use_final_norm]

Step 5 — MLP head:
  pred_X = MLP(x)                   (B, L_X, 1)
```

### 2.2 Attention Matrix Structure

The combined attention matrix `att_weights` of shape `(B, L_X, L_S + L_X)` encodes:

```
att_weights[:, :, :L_S]       →  A_SX  ∈ ℝ^{B × L_X × L_S}
                                  Learned S→X adjacency.
                                  A_SX[b, i, j] = attention of X_i to S_j.
                                  Ground truth: S_j ∈ pa(X_i) iff edge S_j → X_i.

att_weights[:, :, L_S:]       →  A_XX  ∈ ℝ^{B × L_X × L_X}
                                  Learned X→X adjacency.
                                  A_XX[b, i, j] = attention of X_i to X_j.
                                  Diagonal = 0 by mask construction.
                                  Ground truth: X_j ∈ pa(X_i) iff edge X_j → X_i.
```

### 2.3 Key Design Choices

**Single block, no self-attention.** The existing `SingleCausalLayer` has a
cross-attention (S→X) and a self-attention (X→X) operating sequentially on
embeddings. Here, a single cross-attention handles both simultaneously, with
direct access to actual variable values. This is structurally simpler and
more favorable for the attention mechanism.

**Same embedding module for queries and keys.** The `embedding_X` module is
called twice: once with `x_blanked` (zero value column, only variable-ID
embedding active) and once with `x_actual` (both value and variable-ID active).
The Q and K linear projections inside `AttentionLayer` handle the role
differentiation. This minimizes the number of parameters.

**`CausalCrossAttention` recommended.** It uses `ReLU(Tanh(score / τ))` as the
activation — non-normalized, non-negative, can be zero. Unlike softmax, it
can produce sparse attention maps where inactive edges have weight exactly zero.
`ScaledDotProduct` (softmax) is always dense and should be used only as a
dense baseline.

**Diagonal masking of X-X sub-block.** Without this, the trivially optimal
solution for every X_i is to attend to itself with weight 1 (perfect
reconstruction, R²=1, zero learning). The mask forces the model to use
parent information.

---

## 3. Theoretical Motivation

### 3.1 Attention as Causal Regression

For a linear additive noise model (ANM):

```
X_i = Σ_{j ∈ pa(X_i)} α_ij · V_j  +  ε_i,    ε_i ⊥ pa(X_i)
```

where `V_j` is the value of parent j (either S_j or X_j). The attention output
for X_i is:

```
output_i = Σ_j  A_ij · value_embed(V_j)
```

If `value_embed` is a linear layer (as configured) and the output head is
linear, the full computation is:

```
pred_X_i = W_out · Σ_j  A_ij · W_val · V_j
```

This is a linear function of the parent values, parameterized by the attention
weights `A_ij`. The model can in principle learn `A_ij ∝ α_ij` and recover
the causal coefficients.

**Claim**: For a linear SCM and linear embedding, MSE-optimal attention weights
correspond exactly to the causal parent coefficients (up to a rescaling by the
output and value projection matrices). The structural signal is directly
encoded in the attention map.

### 3.2 Why HSIC Is Necessary

MSE alone does not guarantee sparse attention aligned with causal parents.
The Bayes-optimal predictor for any X_i is the full conditional expectation,
which is a *dense* function of all correlated variables (parents, children,
Markov boundary members). Multiple attention patterns achieve the same MSE.

HSIC (Hilbert-Schmidt Independence Criterion) provides the additional
constraint: the model's residuals `ε̂_i = X_i - pred_X_i` must be independent
of every potential parent variable. By the ANM assumption, the *true* residuals
(noise terms) are independent of the true parents. HSIC drives the attention
to find the parent set where this independence holds.

**Formal property**: In a correctly specified linear ANM with
non-Gaussian noise, HSIC(ε̂_i, V_j) = 0 iff V_j is not in the Markov boundary
of X_i given the current residuals. Combined with sparsity, this singles out
the true parent set.

### 3.3 The "Attend to Descendants" Problem and How HSIC Resolves It

A key failure mode is attending to *descendants* of X_i rather than its
parents. In observational data, X_k (a child of X_i) carries information
about X_i's noise realization and can reduce MSE. This would lead to
*anticausal* attention.

HSIC penalizes this: if X_i is reconstructed using X_k (its child), the
residual ε̂_i still contains information about X_k (because X_k = f(X_i, ...),
so `Cov(X_k, ε̂_i) ≠ 0`). Therefore `HSIC(X_k, ε̂_i) > 0`, and gradient
descent will penalize the weight A_ik.

For the true parents, perfect reconstruction gives `ε̂_i ≈ ε_i`, and by the
ANM assumption `HSIC(V_j, ε_i) = 0` for all j. HSIC is minimized at the
correct parent set.

### 3.4 Markov Equivalence

For linear Gaussian noise, multiple DAGs generate the same observational
distribution (Markov Equivalence Class, MEC). HSIC alone cannot distinguish
DAGs within the same MEC from observational data.

For non-Gaussian noise (as in `ds_scm1`, where S variables are uniform), the
Darmois-Skitovitch theorem guarantees that the true causal direction is
*identifiable*: the residuals in the correct causal direction have higher
independence from the inputs than in the reverse direction. HSIC exploits this
asymmetry.

**Expected outcomes**:
- Gaussian noise: converge to MEC representative (possibly wrong orientation
  on some edges), SHD > 0.
- Non-Gaussian / mixed noise (ds_scm1): full DAG recovery possible.

### 3.5 Comparison with Existing Architecture

| Property | `SingleCausalLayer` | `AttentionSelectorLayer` |
|----------|---------------------|--------------------------|
| X→X information | Variable-ID embeddings only | Actual X values |
| S→X information | Actual S values | Actual S values |
| Structural pathway | Separate cross + self blocks | Single combined block |
| HSIC target | Cross: S vs res; Self: X vs res | Combined: [S,X] vs res |
| Difficulty | Higher (no raw X values) | Lower (favorable test) |

A positive result for `AttentionSelectorLayer` is necessary but not sufficient
for `SingleCausalLayer`. If the selector fails, it means HSIC + attention
cannot recover structure even in the best case.

---

## 4. Training Objectives

The total loss is:

```
L_total = λ_recon · MSE(X, pred_X)
        + λ_score_sparse · mean(|A|)          ← L1 on attention weights
        + λ_hsic · HSIC([S, X], residuals)    ← independence: [S∪X] ⊥ ε̂
        + λ_group_l1 · L2,1(embeddings)       ← embedding column sparsity
```

**HSIC term detail**: `combined_source = cat([S_values, X_values], dim=1)`
of shape `(B, L_S + L_X)`. Then `hsic_cross_per_pair(combined_source,
residuals)` sums `HSIC(combined_source_j, residual_i)` over all `(i, j)` pairs.
This is equivalent to summing separate S→X and X→X HSIC terms with equal
weighting — no separate `lambda_hsic_cross` / `lambda_hsic_self` needed.

### 4.1 Gradient Routing (Optional)

When `use_gradient_routing=True`:
- **Structural optimizer**: Q, K projections (control attention pattern).
  Updated by: `λ_hsic · HSIC + λ_score_sparse · |A| + λ_group_l1 · L2,1`
- **Reconstruction optimizer**: V, output, FFN, forecaster.
  Updated by: `λ_recon · MSE`

This prevents the reconstruction gradient from interfering with structure
learning.

---

## 5. Experimental Protocol

### 5.1 Suggested Stages (using ds_scm1)

| Stage | λ_hsic | λ_score_sparse | grad_routing | Expected outcome |
|-------|--------|----------------|--------------|------------------|
| 1 — Baseline | 0 | 0 | false | High R², dense attention, SHD large |
| 2 — Sparsity | 0 | 0.01 | false | Sparse attention, R² slightly lower, no directional preference |
| 3 — HSIC+sparsity | 0.1 | 0.01 | false | Sparse + causally aligned attention, SHD small |
| 4 — Full | 0.1 | 0.01 | true | Cleanest structural convergence |

### 5.2 Key Metrics

After training, evaluate structure recovery:

```python
# Get split attention
att_sx, att_xx = forecaster.get_split_attention(S_batch, X_batch)

# Average over batch dimension and threshold
A_sx_mean = att_sx.mean(dim=0)  # (L_X, L_S)
A_xx_mean = att_xx.mean(dim=0)  # (L_X, L_X)

# Threshold (e.g., 0.1) to get binary adjacency
pred_sx = (A_sx_mean > threshold).float()
pred_xx = (A_xx_mean > threshold).float()

# Compare to ground truth masks (from hard_mask_files)
SHD_sx = (pred_sx != true_sx_mask).sum()
SHD_xx = (pred_xx != true_xx_mask).sum()
```

### 5.3 Success Criteria

- **R² > 0.95**: Reconstruction is working (prerequisite).
- **SHD_sx + SHD_xx ≤ 2**: Near-perfect structure recovery (strong positive result).
- **SHD_sx + SHD_xx ≤ 5**: Partial structure recovery (MEC or near-MEC).
- **SHD decreasing over epochs**: Structural signal is present and converging.

### 5.4 Dataset Structure (ds_scm1)

```
S→X true edges:
  S2 → X1, S3 → X2, S3 → X3, S4 → X4, S5 → X4

X→X true edges:
  X2 → X4, X1 → X5, X2 → X5

Critical test nodes:
  X4: parents are S4, S5, X2  — only node with both S and X parents
  X5: parents are X1, X2      — no S parent (forces use of X-X block)
  X3: parent is S3 only       — single S parent, simple case
```

X5 is the most important test: it has no S parent, so the model must use the
X-X sub-block to reconstruct it. If A_XX[X5, X1] and A_XX[X5, X2] are the
dominant weights and all others near zero, structure recovery is working.

---

## 6. Relationship to Existing Work

This experiment is related to:
- **NOTEARS** (Zheng et al., 2018): Treats DAG learning as a continuous
  optimization problem. AttentionSelector similarly parameterizes the adjacency
  as continuous weights and regularizes toward sparsity.
- **DAGS with NO TEARS via NFs** (Lachapelle et al.): Uses neural networks
  to parameterize causal mechanisms. AttentionSelector uses attention weights
  as explicit edge parameters.
- **In-context causal discovery** (Muller et al., 2023): Transformers trained
  in-context can implicitly perform causal discovery. AttentionSelector tests
  whether a single-layer attention trained *in-distribution* (not in-context)
  achieves this.
- **HSIC-based causal discovery** (Janzing et al., Peters et al.): HSIC(ε̂_i, X_j)
  tests the ANM independence criterion. AttentionSelector uses this as a
  differentiable training objective rather than a post-hoc test statistic.

---

## 7. Limitations and Next Steps

### Current limitations
- **Single layer**: No depth to compose indirect causal effects. X5 = g·X1 + h·X2
  is learned in one shot; multi-hop effects (e.g., S2 → X1 → X5) require the
  model to correctly assign weight to X1 rather than S2 for X5.
- **Observational data only**: Interventional data would make structure recovery
  straightforward. This experiment tests the harder observational case.
- **No acyclicity constraint**: The X-X attention can in principle form cycles.
  NOTEARS can be applied to the X-X sub-matrix if needed.

### Follow-up experiments
1. **Ablation: non-blanked queries.** What if X queries also have actual values?
   The diagonal mask still prevents self-loops, but the query now carries the
   target value. Expected: higher R² but worse structure (model attends to
   variables correlated with its own value, not its parents).
2. **Noise sensitivity.** Vary the noise variance. Higher noise → easier structure
   recovery (descendants less informative, parents more informative).
3. **Non-linear SCM.** Replace linear mechanisms with nonlinear ones. HSIC should
   still work (it is nonparametric), but the attention output (linear) may be
   insufficient for reconstruction — tests the linear embedding assumption.
4. **Link to full architecture.** If AttentionSelector works, does
   `SingleCausalLayer` (which uses embeddings instead of raw values) also
   converge to the correct structure? The gap would quantify the cost of
   the embedding approximation.
