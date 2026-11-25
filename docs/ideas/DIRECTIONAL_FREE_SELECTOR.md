# Complementary structural selector

Select the architecture using the existing stage-policy key:

```yaml
adaptive_training:
  structure:
    gate_opt: directional_free
    order_sampling: hard_st
```

The normal separate-structure-forward training path and the adaptive
trainer use this same configuration. No user-facing edge_st mode is needed.

One parameter per unordered pair produces opposite logits and complementary
sigmoid probabilities. Initialization is exactly 0.5 in both directions,
independent of the legacy existence initialization logit. There is no
existence factor, and the selector contributes no L0 penalty. Parameters
retain the struct_gate_logits name for structural gradient routing.

For structural training with hard_st, form p = M * G_S, where M = P T P^T
and P is the relaxed Sinkhorn permutation. One uniform per unordered pair
selects forward, reverse, or no edge. Pair mass is normalized only if it
exceeds one. The surrogate is B + (p - p.detach()). The draw is cached in
the step noise and reused across forwards. Longer cycles are possible.
G_R remains a separate reconstruction mixing factor, not part of p.

Soft mode applies the product continuously. Hard evaluation uses the
noiseless order multiplied by continuous G_S, so its support is acyclic.
The separate-forward helper selects this hard evaluation policy for hard_st.
The structural posterior includes G_S, but loss pair existence remains one
and the loss order mask remains M. Neither changes the order-loss formula.

Shared selectors and reconstruction-to-structure handoff are rejected for
this architecture. Legacy gated and none behavior is unchanged. Joint edge
sampling is currently implemented for directional_free only.

Existing ORDER_ONLY checkpoints have no directional parameters. Enabling
this architecture therefore requires explicit warm-start initialization of
the new logits and optimizer state; an unchanged strict resume is not valid.
No checkpoint migration or experiment continuation is performed here.