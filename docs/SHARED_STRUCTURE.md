# Shared Structure Across Decoder Layers

## Motivation

In a multi-layer decoder, each layer independently learns its own Q/K projections and inner attention mechanism. Since these components determine the **causal structure** (which variables attend to which), nothing prevents different layers from learning *different* causal graphs.

The key insight is that **causal structure is an easy map** compared to the reconstruction task. Attention weights converge quickly to a stable DAG representation, even in shallow models. When we add depth (multiple decoder layers), we want the extra capacity to improve **reconstruction quality** (via deeper value transformations and feedforward layers), not to learn redundant or conflicting causal structures.

## The Idea: Share Structural Parameters, Free Reconstruction Parameters

When `share_structure_across_layers: true`, all decoder layers share the same:

1. **Query projection** (`W_Q`): How each variable formulates its "question"
2. **Key projection** (`W_K`): How each variable presents its "identity"  
3. **Inner attention mechanism**: The attention function itself (e.g., LieAttention parameters, Toeplitz gains/temperatures)

Each layer retains its own **independent**:

1. **Value projection** (`W_V`): What information each variable contributes
2. **Output projection** (`W_out`): How attended information is combined
3. **Feedforward layers** (`FF1`, `FF2`): Non-linear transformations
4. **Layer normalization**: Per-layer statistics

This means every layer computes the **same attention pattern** (same DAG) but transforms the values differently, giving the model depth for reconstruction without structural ambiguity.

## Critical Evaluation

### Arguments For

1. **Structural consistency**: Guarantees a single coherent causal graph across all layers. Without sharing, layer 1 might learn S₁→X₂ while layer 2 learns S₁→X₃, creating an incoherent causal interpretation.

2. **Parameter efficiency**: For `L` layers, structural parameters are `O(1)` instead of `O(L)`. This is especially relevant for LieAttention and ToeplitzAttention which have per-edge learnable parameters.

3. **Stronger regularization signal**: Sparsity and HSIC regularization applied to attention scores affect a single set of structural parameters, concentrating the gradient signal rather than diluting it across layers.

4. **Cleaner evaluation**: Attention score extraction for DAG recovery gives one unambiguous answer instead of `L` potentially different attention matrices.

5. **Theoretical alignment**: In the SCM framework, the causal graph is a fixed object. Multiple layers should refine the *functional mapping* (how causes produce effects), not the *structural mapping* (which variables are causes).

### Arguments Against / Risks

1. **Reduced expressiveness**: If the true data-generating process benefits from different attention patterns at different abstraction levels (e.g., direct effects in layer 1, indirect effects in layer 2), sharing prevents this.

2. **Gradient coupling**: All layers route structural gradients to the same parameters. If layer 1's reconstruction loss wants to strengthen edge A→B but layer 3's loss wants to weaken it, the shared parameters receive conflicting signals. However, this is arguably a *feature* — it forces consensus.

3. **Not needed for `dec_layers=1`**: The feature has no effect for the common single-layer case. It only matters when experimenting with deeper decoders.

4. **Interaction with SVFA**: In SVFA mode, structure and value are already factorized. Sharing Q/K across layers is a natural extension of this factorization — the structure pathway becomes truly layer-agnostic.

## Implementation

### Architecture

```
Layer 0 (owns parameters):
  cross_att.W_Q  ←── shared ──→  Layer 1 cross_att.W_Q (same object)
  cross_att.W_K  ←── shared ──→  Layer 1 cross_att.W_K (same object)
  cross_att.inner ←── shared ──→  Layer 1 cross_att.inner (same object)
  cross_att.W_V  (independent)    Layer 1 cross_att.W_V (independent)
  cross_att.W_out (independent)   Layer 1 cross_att.W_out (independent)

  self_att.W_Q   ←── shared ──→  Layer 1 self_att.W_Q (same object)
  self_att.W_K   ←── shared ──→  Layer 1 self_att.W_K (same object)
  self_att.inner ←── shared ──→  Layer 1 self_att.inner (same object)
  self_att.W_V   (independent)    Layer 1 self_att.W_V (independent)
  self_att.W_out (independent)    Layer 1 self_att.W_out (independent)

  FF, LayerNorm  (independent)    Layer 1 FF, LayerNorm (independent)
```

### How Sharing Works

The implementation uses Python object identity for weight sharing:

1. Layer 0's `AttentionLayer` creates Q/K projections and inner attention normally
2. `get_shared_qk_inner()` extracts references to these modules
3. Layers 1..N-1 receive these references via `shared_qk_inner` parameter
4. When `shared_qk_inner` is provided, `AttentionLayer.__init__` uses the shared modules instead of creating new ones

Because shared modules are the same Python objects, PyTorch's autograd naturally accumulates gradients from all layers into the shared parameters.

### Config

```yaml
model:
  kwargs:
    dec_layers: 3
    share_structure_across_layers: true
```

When `share_structure_across_layers: false` (default), behavior is identical to the existing codebase — each layer has fully independent parameters. This ensures backward compatibility.

## Experimental Recommendations

1. **Baseline comparison**: Always compare `share_structure=true` vs `false` with the same `dec_layers` to isolate the effect of sharing.

2. **Start with `dec_layers: 2`**: The simplest multi-layer case. If sharing helps here, try deeper.

3. **Monitor per-layer attention**: Even with sharing, log attention weights per layer (they will be identical). Compare against non-shared to see if layers converge to similar structures anyway.

4. **Combine with HSIC**: Shared structure + HSIC regularization should give cleaner causal discovery than either alone.

## Files Modified

- `causaliT/core/modules/attention.py` — `AttentionLayer`: added `shared_qk_inner` parameter and `get_shared_qk_inner()` method
- `causaliT/core/architectures/single_causal/model.py` — `SingleCausalLayer`: added `share_structure_across_layers` parameter
- `causaliT/core/architectures/noise_aware/model.py` — `NoiseAwareSingleCausalLayer`: added `share_structure_across_layers` parameter
- `causaliT/config/templates/config_single_causal_svfa.yaml` — added config key
- `causaliT/config/templates/config_noise_aware.yaml` — added config key
