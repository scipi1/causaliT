# SVFA Dual-Residual Variant (`SingleCausalLayerRes`)

## TL;DR

Standard SVFA (`SingleCausalLayer` with `factorization="svfa"`) puts every
residual connection on the **value** stream:

```
X_val    ← X_val + Attn(...)        # updated each layer
X_struct ← X_struct                  # passed through unchanged
```

The dual-residual variant adds a second residual on the **structure**
stream so that the DAG hypothesis itself is refined layer by layer:

```
X_val    ← X_val    + Attn_val(Q, K, V_val)
X_struct ← X_struct + Attn_struct(Q, K, V_struct)
```

The same `Q`, `K` (both from `X_struct`) drive a single attention
pattern; the pattern is then read out **twice**, once against
`V_val = LN(X_val)` (reconstruction-driven) and once against
`V_struct = LN(X_struct)` (structural-loss-driven). The FFN sub-block
remains value-only.

This document describes the implementation, the gradient-routing
contract, and how to use the variant.

---

## 1. Why a second residual?

In standard SVFA the structural embedding `X_struct` is read-only inside
the decoder: it provides Q and K for attention and is then thrown away.
When `dec_layers > 1`, every layer recomputes attention from the
**same** `X_struct`; the DAG hypothesis cannot evolve with depth.

The dual-residual variant gives `X_struct` its own update path so that
later layers can refine the structural representation produced by
earlier layers, without sacrificing the SVFA property that **only the
value stream is observed by the reconstruction head**.

The two streams are therefore *coupled through Q/K* but *separated in
their value pathways*. This is what enables clean gradient routing: a
dedicated set of structural-only parameters
(`value_projection_struct`, `out_projection_struct`) can be assigned to
the HSIC / structural loss, while the standard
`value_projection` / `out_projection` continue to receive only the
reconstruction signal.

---

## 2. Architecture

### 2.1 Forward pass (per decoder layer)

Notation: `X = (X_struct, X_val)` are the two SVFA streams. `ext` is
either a single tensor (when the source uses summation embeddings) or
a tuple `(ext_struct, ext_val)` (when the source also uses SVFA).

```
# 1. Cross-attention (X attends to ext)
Q = LN_struct(X_struct)
K = ext_struct                        # (or ext if ext is a single tensor)
V_val    = ext_val                    # (or ext)
V_struct = ext_struct                 # routed via key_value=ext_struct

out_val, out_struct = AttentionLayer(
    query=Q, key=K, value=V_val, key_value=V_struct,   # dual_value=True
)
X_val    ← X_val    + dropout(out_val)
X_struct ← X_struct + dropout(out_struct)

# 2. Self-attention (X attends to itself)
Q = K = LN_struct(X_struct)
V_val    = LN_val(X_val)
V_struct = LN_struct(X_struct)        # routed via key_value

out_val, out_struct = AttentionLayer(query=Q, key=K, value=V_val, key_value=V_struct)
X_val    ← X_val    + dropout(out_val)
X_struct ← X_struct + dropout(out_struct)

# 3. Feedforward (value stream only — same as standard SVFA)
X_val ← X_val + FFN(LN(X_val))
```

Implementation: `causaliT/core/architectures/single_causal_res/decoder.py`
(`DualResidualDecoderLayer`, `DualResidualDecoder`).

### 2.2 `AttentionLayer.dual_value`

Implemented in `causaliT/core/modules/attention.py`. When
`dual_value=True` the layer instantiates two extra modules:

| Module                       | Shape                                 | Purpose                                    |
|------------------------------|----------------------------------------|--------------------------------------------|
| `value_projection_struct`    | `Linear(d_model_keys, d_model_keys * n_heads)` | Builds `V_struct` from `key_value`.        |
| `out_projection_struct`      | `Linear(d_model_keys * n_heads, d_model_keys)` (only when `n_heads > 1`) | Mixes the multi-head structural output.    |

The forward pass runs the **same** Q,K-derived attention scores against
both `V` and `V_struct`, returning a 4-tuple
`(out_val, out_struct, attn, ent)` instead of the usual 3-tuple
`(out, attn, ent)`. All call-sites that expect the 3-tuple form must
either (a) leave `dual_value=False` (the default, so behavior is
unchanged), or (b) explicitly handle the 4-tuple.

The optional `key_value` argument selects the input that
`value_projection_struct` sees. It defaults to `key`; in the dual-
residual decoder it is set explicitly to `ext_struct` (cross) or
`X_struct_norm` (self) so the structural value path stays decoupled
from the K input even when K and `key_value` happen to be derived from
the same tensor.

### 2.2.1 Multi-head semantics

The structural value path **mirrors the reconstruction value path
exactly** under multi-head. There is no asymmetry: every multi-head
construct on the recon side has a shape-for-shape twin on the
structural side.

For one attention block at `d_model = d` and `H` heads:

| Recon stream (always present)                                                | Structural stream (only when `dual_value=True`)                                                |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `value_projection: Linear(d_model_values, d_model_values * H)`                | `value_projection_struct: Linear(d_model_keys, d_model_keys * H)`                              |
| `out_projection : Linear(d_model_values * H, d_model_values)`                 | `out_projection_struct : Linear(d_model_keys  * H, d_model_keys)` *(only when `H > 1`)*        |

Notes:

* The recon path uses `d_model_values`; the struct path uses
  `d_model_keys` because `V_struct` is built from the key-side input
  (`key_value`), which lives in the structural space. In our standard
  SVFA configs `d_model_values == d_model_keys == d`, so the two paths
  end up the same width; the code stays correct for asymmetric configs.
* `out_projection_struct` is created **only when `H > 1`**, mirroring
  the existing `out_projection is None` shortcut for single-head on the
  recon side. The unit test
  `tests/test_svfa_dual_residual.py::test_single_head_no_out_projection_struct`
  locks this in.

The **attention pattern `A` is computed once** from Q,K (both built from
`X_struct`) and shared between the two value pathways. SVFA's
`shared_dag_across_heads=True` keeps `A` of shape `(B, L, S)` (one DAG
broadcast over heads); the multi-head dimension lives **entirely in the
value pathways**, and with dual-value it lives there *twice* — once for
recon, once for struct:

```
Q, K  (built from X_struct → single attention pattern A, shape (B, L, S))
  │
  ├──► V_recon  = value_projection(value)             reshape (B, S, H, d_head)
  │                A · V_recon  ──► out_projection          ──► out_val
  │
  └──► V_struct = value_projection_struct(key_value)  reshape (B, S, H, d_head)
                   A · V_struct ──► out_projection_struct   ──► out_struct
```

**Why mirror the head structure?** If the structural V were single-head
while the recon V were `H`-head, structural updates would be
capacity-bottlenecked relative to reconstruction, and gradient routing
would push HSIC into a strictly weaker subspace than the one the recon
loss sees. Mirroring the head structure removes that asymmetry: each
head has its own `(recon, struct)` pair of value vectors, and the `H`
value channels are then collapsed independently by `out_projection` and
`out_projection_struct`.

**Parameter cost.** Dual-value adds approximately `2 · H · d²`
parameters per attention block (≈ the same cost as the existing recon
V/out pair). All extra parameters live under names containing
`value_projection_struct` / `out_projection_struct`, which is exactly
what `STRUCTURAL_PATTERNS` in `gradient_routing.py` matches —
ensuring the new capacity is routed to the structural (HSIC) loss only.

### 2.3 `SingleCausalLayerRes`

`causaliT/core/architectures/single_causal_res/model.py`. Subclasses
`SingleCausalLayer` and:

1. Forces `factorization="svfa"`. Any other value raises `ValueError`.
2. Rejects `attention_bypass=True` (the bypass formulation assumes a
   single residual stream — out of scope for this variant).
3. Overrides `_attn(...)` to inject `dual_value=True` into every
   `AttentionLayer` constructor call, so cross- and self-attentions
   are built dual-value.
4. After the parent `__init__` finishes, swaps the parent's
   `ReversedDecoder` for a `DualResidualDecoder`, **reusing every
   parameter** the parent already created (attention modules, norms,
   FFN linears, dropouts). Parameter count and initial state are
   therefore identical to a hypothetical `ReversedDecoder` built with
   `dual_value=True`.

This architecture is registered in
`causaliT/core/architectures/__init__.py` under the model object name
`"SingleCausalLayerRes"`.

---

## 3. Training pipeline

| Component                    | File                                                                  |
|------------------------------|-----------------------------------------------------------------------|
| Forecaster (Lightning)       | `causaliT/training/forecasters/single_causal_res_forecaster.py`       |
| Trainer dispatch             | `causaliT/training/trainer.py` (`make_forecaster`, `make_datamodule`) |
| Predictor (eval)             | `causaliT/evaluation/predictors/single_causal_res_predictor.py`       |
| Eval-fn dispatch             | `causaliT/evaluation/predict.py`, `eval_lib.py`, `eval_interventions.py` |
| Gradient routing patterns    | `causaliT/training/gradient_routing.py` (`STRUCTURAL_PATTERNS`)       |

The forecaster and predictor are thin shims around the
`SingleCausal*` equivalents — the only behavioral differences are
internal to the model itself.

### Gradient routing

`STRUCTURAL_PATTERNS` now contains:

```
"value_projection_struct"   # the new V projection on the structural stream
"out_projection_struct"     # the matching multi-head output projection
```

These names are matched by **substring**, so all `*.value_projection_struct.*`
and `*.out_projection_struct.*` parameters land in the structural
optimizer group. Note that `"value_projection"` (without `_struct`) is
*not* a structural pattern — it is the standard recon V projection and
must continue to receive reconstruction gradients only. The unit tests
in `tests/test_svfa_dual_residual.py` lock in this distinction.

When `training.use_gradient_routing: true`, the HSIC loss therefore
flows into:

* the embedding `nn_embedding` for variable IDs (existing behavior),
* the standard `query_projection` / `key_projection` (existing behavior),
* the new `value_projection_struct` / `out_projection_struct`
  (dual-residual specific).

The reconstruction loss continues to flow into `value_projection`,
`out_projection`, the FFN linears, and the value-stream pre-norms.

---

## 4. Configuration

A turn-key template is provided at:

```
causaliT/config/templates/config_single_causal_res.yaml
```

Key fields:

```yaml
model:
  model_object: "SingleCausalLayerRes"
  attention_bypass: false           # required (true is rejected)
  kwargs:
    factorization: "svfa"           # required
    comps_embed_S: "svfa"
    comps_embed_X: "svfa"
    # All other kwargs are identical to SingleCausalLayer (SVFA template).
```

A smoke-test experiment folder is at:

```
experiments/tests/test_single_res_scm1/
```

Run it with:

```bash
python -m causaliT.cli train experiments/tests/test_single_res_scm1/
```

---

## 5. Tests

`tests/test_svfa_dual_residual.py` covers:

* `AttentionLayer(dual_value=True)` returns the 4-tuple, exposes the
  new structural projections, and routes `key_value` to the structural
  path only.
* `DualResidualDecoderLayer` rejects single-value attentions, rejects
  non-tuple inputs, updates **both** streams, and propagates gradients
  to `value_projection_struct` from a structure-only loss.
* `DualResidualDecoder` (stack) preserves the SVFA tuple across layers.
* `STRUCTURAL_PATTERNS` correctly classifies the new params and does
  *not* mis-classify the standard `value_projection`.

Run with:

```bash
pytest tests/test_svfa_dual_residual.py -v
```

The pre-existing `tests/test_svfa.py` continues to pass and validates
that **standard** SVFA (`SingleCausalLayer`, `factorization="svfa"`)
still has the read-only `X_struct` invariant. The two suites together
encode the contract: standard SVFA = single residual on `X_val`,
`SingleCausalLayerRes` = residuals on both.

---

## 6. Limitations / non-goals

* **ANS bypass** (`attention_bypass=True`) is not implemented for this
  variant. Use `SingleCausalLayer` + `config_single_causal_svfa.yaml`
  for ANS experiments.
* The FFN is **not** mirrored on the structural stream. The structural
  stream's only non-linearity is therefore the inner softmax (or
  whatever the chosen attention type uses). This is intentional — the
  goal is to refine `X_struct` while keeping it linearly comparable
  across layers for DAG-recovery diagnostics.
* The `external_context` may still be a single tensor (e.g. coming
  from a frozen orthogonal S embedding). In that case `ext_struct =
  ext_val = ext`, and the cross-attention's value/struct paths only
  diverge through `value_projection` vs `value_projection_struct`.
