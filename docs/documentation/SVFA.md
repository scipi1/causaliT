# Structure-Value Factorized Attention (SVFA)

SVFA decouples *what* attends from *what flows*: queries and keys are
projected from a **structure** embedding (variable identity only) while
values are projected from a **value** embedding (realization only).
Coupling enters only through the dot product `Q K^T`, which therefore
depends on the variable indices but **not** on their realizations.

This document covers:

1. The factorization itself (`SingleCausalLayer` with
   `factorization="svfa"`).
2. The dual-optimizer **gradient routing** that exploits the
   structure/value split.
3. The **dual-residual variant** `SingleCausalLayerRes` that adds a
   second residual on the structural stream.

Related docs: `TOEPLITZ_DECOMPOSITION.md`, `STAGED_TRAINING.md`,
`REGULARIZATION.md`.

---

## 1. Factorization

### 1.1 Standard vs SVFA attention

Standard attention fuses identity and value into a single embedding,
then projects Q, K, V from it:

```
embedding = f(variable_id) + g(value)
Q, K, V   = W_q(embedding), W_k(embedding), W_v(embedding)
```

The attention pattern `softmax(Q K^T)` therefore depends on the
realization, which makes it sample-specific and fold-sensitive.

SVFA splits the embeddings:

```
struct = f(variable_id)           # identity only
val    = g(value)                 # realization only

Q = W_q(struct)
K = W_k(struct)                   # structural alignment
V = W_v(val)                      # value flow
```

Because Q and K depend only on variable identity, the attention pattern
is **the same for every sample** in a given (S, X) configuration.

| Property                           | Standard | SVFA |
|------------------------------------|----------|------|
| Attention pattern depends on values? | yes      | **no** |
| Pattern consistent across samples?   | no       | **yes** |
| Information flow remains?            | yes      | yes |

### 1.2 Implementation

`SingleCausalLayer` accepts `factorization="svfa"`. Together with
`comps_embed_S: "svfa"` and `comps_embed_X: "svfa"` this builds the
two-stream embedding (`X_struct`, `X_val`) and threads it through the
decoder. The default is `factorization="summation"` (legacy fused
behavior).

```yaml
model:
  model_object: "SingleCausalLayer"
  kwargs:
    factorization: "svfa"
    comps_embed_S: "svfa"
    comps_embed_X: "svfa"
```

In SVFA mode the value stream is the only one observed by the
reconstruction head; the structure stream is read-only inside the
decoder and produces Q and K. See `SVFA_DUAL_RESIDUAL` (Section 3
below) for the variant that lets the structure stream evolve too.

---

## 2. Gradient routing

### 2.1 Motivation

Even with SVFA, a single optimizer mixes two qualitatively different
gradient signals:

- **Reconstruction** (MSE / NLL) updates the value pathway, the FFN,
  and the prediction head.
- **HSIC + sparsity / group-L1** updates the structural pathway
  (Q/K projections, structure embeddings, attention internals).

In practice the HSIC term often oscillates while reconstruction
dominates the value pathway, and reconstruction gradients leak into Q,
K and embeddings, fighting the structural objective. Standard SVFA
*identifies* the two parameter groups but does not enforce separate
optimization.

### 2.2 Mechanism

When `training.use_gradient_routing: true` the forecaster builds two
optimizers and routes losses to disjoint parameter sets:

| Group              | Parameters (θ)                                                                                     | Updated by |
|--------------------|----------------------------------------------------------------------------------------------------|------------|
| **Structural** θ_S | structure embeddings, `query_projection`, `key_projection`, attention internals (`log_gain`, `log_tau`, `temperature`), structural pre-norms, dual-residual `value_projection_struct` / `out_projection_struct` (Section 3) | HSIC + score sparsity + group L1 |
| **Reconstruction** θ_R | value embeddings, `value_projection`, `out_projection`, FFN, prediction head, value pre-norms, noise parameters                                | reconstruction loss (MSE / NLL) |

Per training step:

1. One forward pass computes all losses.
2. `loss_recon.backward()` → `opt_recon.step()` updates θ_R only.
3. `loss_structural.backward()` → `opt_struct.step()` updates θ_S only.

### 2.3 Parameter classification

Classification is by name pattern in
`causaliT/training/gradient_routing.py`. Structural patterns include
`query_projection`, `key_projection`, `structure_modules`,
`inner_attention.log_gain`, `inner_attention.log_tau`,
`inner_attention.temperature`, `norm1_struct`, `norm2_struct`,
`value_projection_struct`, `out_projection_struct`. Everything else is
classified as reconstruction.

To inspect the split for your model:

```python
from causaliT.training.gradient_routing import classify_parameters
structural, reconstruction = classify_parameters(model, verbose=True)
```

### 2.4 Usage

```yaml
training:
  use_gradient_routing: true
  lambda_hsic_cross: 1.0
  lambda_hsic_self: 1.0
```

Compatible with HSIC annealing, score sparsity, group L1 and the staged
training pipeline. When active, two extra metrics are logged:

- `train_loss_recon_routed` — reconstruction loss only.
- `train_loss_structural_routed` — HSIC + sparsity + group L1 only.

### 2.5 Related work

The split is a form of bilevel optimization. See DARTS
(Liu et al., ICLR 2019), PCGrad (Yu et al., NeurIPS 2020), GradNorm
(Chen et al., ICML 2018), CASTLE (Kyono et al., NeurIPS 2020),
DAG-GNN (Yu et al., ICML 2019).

---

## 3. Dual-residual variant: `SingleCausalLayerRes`

### 3.1 What changes

Standard SVFA puts every residual on the value stream:

```
X_val    ← X_val + Attn(...)
X_struct ← X_struct                      # passed through unchanged
```

When `dec_layers > 1` this is restrictive: every layer recomputes
attention from the *same* `X_struct`, so the DAG hypothesis cannot
evolve with depth. The dual-residual variant adds a second residual on
the structural stream:

```
X_val    ← X_val    + Attn_val(Q, K, V_val)
X_struct ← X_struct + Attn_struct(Q, K, V_struct)
```

Q and K are still built from `X_struct` (one shared attention pattern),
but the pattern is read out twice — once against `V_val = LN(X_val)`
(reconstruction-driven) and once against `V_struct = LN(X_struct)`
(structural-loss-driven). The FFN sub-block remains value-only.

The two streams are coupled through Q/K but their value pathways are
separated, which makes them a clean target for gradient routing: the
new `value_projection_struct` / `out_projection_struct` parameters
receive the structural loss only.

### 3.2 Forward pass (per layer)

```
# Cross-attention
Q = LN_struct(X_struct);  K = ext_struct (or ext)
V_val    = ext_val (or ext);  V_struct = ext_struct
out_val, out_struct = AttentionLayer(Q, K, V_val, key_value=V_struct)
X_val    ← X_val    + dropout(out_val)
X_struct ← X_struct + dropout(out_struct)

# Self-attention
Q = K = LN_struct(X_struct)
V_val    = LN_val(X_val);     V_struct = LN_struct(X_struct)
out_val, out_struct = AttentionLayer(Q, K, V_val, key_value=V_struct)
X_val    ← X_val    + dropout(out_val)
X_struct ← X_struct + dropout(out_struct)

# Feedforward (value stream only)
X_val ← X_val + FFN(LN(X_val))
```

Implementation:
`causaliT/core/architectures/single_causal_res/decoder.py`
(`DualResidualDecoderLayer`, `DualResidualDecoder`).

### 3.3 `AttentionLayer.dual_value`

Defined in `causaliT/core/modules/attention.py`. When
`dual_value=True` the layer instantiates two extra modules that
mirror the recon-side V-pair:

| Module                       | Shape                                            | Role |
|------------------------------|--------------------------------------------------|------|
| `value_projection_struct`    | `Linear(d_model_keys, d_model_keys * n_heads)`   | builds `V_struct` from `key_value` |
| `out_projection_struct`      | `Linear(d_model_keys * n_heads, d_model_keys)` (only when `n_heads > 1`) | mixes structural multi-head output |

The forward pass returns a 4-tuple `(out_val, out_struct, attn, ent)`
instead of the usual 3-tuple. Call-sites that expect the 3-tuple form
must either keep `dual_value=False` (default, unchanged behavior) or
explicitly handle the 4-tuple. The optional `key_value` argument
selects the input that `value_projection_struct` sees; in the
dual-residual decoder it is set to `ext_struct` (cross) or
`X_struct_norm` (self).

**Multi-head semantics.** The attention pattern `A` is computed once
from Q, K (both built from `X_struct`) and shared between value
pathways (`shared_dag_across_heads=True` keeps `A` of shape
`(B, L, S)`). Each pathway then produces its own per-head V and
collapses them with its own output projection. Mirroring the recon
pathway's head structure prevents an asymmetric capacity bottleneck
between reconstruction and HSIC. Cost: ≈ `2 · H · d²` extra parameters
per attention block, all matching the `value_projection_struct` /
`out_projection_struct` patterns and routed to the structural
optimizer by `gradient_routing.py`.

### 3.4 Configuration

`SingleCausalLayerRes` requires `factorization="svfa"` and rejects
`attention_bypass=True`. A turn-key template is at
`causaliT/config/templates/config_single_causal_res.yaml`:

```yaml
model:
  model_object: "SingleCausalLayerRes"
  attention_bypass: false
  kwargs:
    factorization: "svfa"
    comps_embed_S: "svfa"
    comps_embed_X: "svfa"
    # all other kwargs identical to the SVFA template
```

### 3.5 Tests

`tests/test_svfa_dual_residual.py` covers:

- `AttentionLayer(dual_value=True)` returns the 4-tuple, exposes the new
  projections, and routes `key_value` to the structural pathway only.
- `DualResidualDecoderLayer` rejects single-value attentions, rejects
  non-tuple inputs, updates **both** streams, and propagates gradients
  to `value_projection_struct` from a structure-only loss.
- `DualResidualDecoder` preserves the SVFA tuple across stacked layers.
- `STRUCTURAL_PATTERNS` correctly classifies the new params and does
  *not* mis-classify the standard `value_projection`.

`tests/test_svfa.py` continues to validate that **standard** SVFA
preserves the read-only `X_struct` invariant. Together the two suites
encode the contract: standard SVFA = single residual on `X_val`;
`SingleCausalLayerRes` = residuals on both.

---

## 4. Limitations and non-goals

- ANS bypass (`attention_bypass=True`) is implemented for
  `SingleCausalLayer` only; the dual-residual variant rejects it.
- The FFN is not mirrored on the structural stream — its only
  non-linearity is the inner attention activation. This is intentional:
  it keeps `X_struct` linearly comparable across layers for DAG-recovery
  diagnostics.
- For sources that use a fused embedding (e.g. an orthogonal frozen S
  embedding), `ext_struct = ext_val = ext`; the structural and value
  cross paths only diverge through `value_projection` vs
  `value_projection_struct`.
