# TODO: Stage-policy and attention cleanup

Status: proposed internal refactor, not yet implemented as a policy object.

## Goal

Make every forward follow one complete, validated policy instead of several
independently mutable attributes. Preserve the experimental options and their
numerical behavior; do not add another configuration framework.

Keep the existing public configuration organization:

```yaml
adaptive_training:
  recon:
    gate_opt: softmax
    gate_sampling: hard
    order_sampling: none
  structure:
    gate_opt: gated
    gate_sampling: soft
    order_sampling: soft
```

Gate sampling applies to gated selectors; it does not make softmax stochastic.

## 1. Separate selector design from forward policy

| Category | Settings | Lifetime |
| --- | --- | --- |
| Selector design | G_R gated/softmax/none, G_S present/absent, shared or separate selectors, handoff configuration | Fixed at construction |
| Forward policy | Reconstruction/structure stage, gate sampling, order sampling, orientation, applicable oriented-sampling settings | Changes between phases or passes |

`gate_opt` determines model construction and parameter ownership. Sampling
settings determine how the constructed model is used. Phase transitions must
not implicitly change which parameters exist or who owns them.

## 2. Introduce an immutable, validated forward policy

Use a small immutable object containing the complete effective forward settings:

- Stage: reconstruction or structure.
- Gate sampling: hard or soft.
- Order sampling: none, hard, soft, or the supported stage-specific variants.
- Orientation: causal or anticausal.
- Oriented-sampling settings when applicable.

Validate the complete object before installing it. A module should expose one
current policy rather than independently mutable stage, order, gate-mode, and
orientation attributes. A forward must never observe a partial update.

Resolve omitted fields before applying a policy; do not inherit an incompatible
order setting from the previous stage.

## 3. Resolve phase policies outside attention

The configuration resolver should produce complete policies for warmup,
reconstruction, structure, and final reconstruction:

```text
reconstruction defaults + phase overrides = effective phase policy
```

Make final-reconstruction inheritance explicit, including nested settings. The
attention module should not know where a setting came from or how phase blocks
inherit from each other. The trainer applies one resolved policy per transition.

## 4. Use exception-safe temporary overrides

Provide a scoped override that saves the current policy, applies a temporary
policy, and restores the original even if execution raises an exception.

Use it for:

- Reconstruction-quality probes.
- Order initialization.
- Paired causal and anticausal forwards.
- Deterministic evaluation overrides.

Keep policy separate from stochastic state. Changing policy must not resample
noise or advance sampling clocks implicitly. Paired passes must reuse the same
step's draws. Test nested overrides and restoration after exceptions.

## 5. Organize the forward around distinct quantities

Extract focused helpers for:

1. Order availability under the chosen sampling policy.
2. Structural existence from G_S, or ones when G_S is absent.
3. Reconstruction weights from G_R: gated, softmax, or none.
4. Applied value weights, including downstream gating and dropout.

Keep two explicit paths:

```text
Graph path:
    structural existence + order
        -> edge posterior
        -> sampled/exclusion graph
        -> graph diagnostics

Value path:
    reconstruction scores + availability
        -> reconstruction weights
        -> downstream structural gate
        -> dropout
        -> value aggregation
```

The edge posterior is `E = pi_S * M`, or `E = M` without G_S. Reconstruction
scores and pi_R must never define edge existence.

For separate gated selectors, structure-stage value weights compose G_R, G_S,
and order. For softmax, order and BKD availability enter before normalization;
G_S multiplies afterwards without renormalization. Fully unavailable rows are
zero. Shared gates must be applied once, not squared.

Retain explicit matrix-layout conventions at helper boundaries: order matrices
are source-to-target; applied attention rows are child-to-parent.

Preserve FiLM's existing context and modulation behavior. This cleanup is not
a redesign of the regressor or its subset conditioning.

## 6. Treat gradient ownership as an invariant

Do not introduce user-facing flags for arbitrary detachment or ownership.
Derive the rules from stage and selector design:

- Reconstruction trains G_R and the regressor.
- Structure trains order and G_S; reconstruction weights are nuisance weights
  under the defined routing rules.
- Structural `lambda_l0` penalizes only G_S or a structurally active shared gate.
- Shared/handoff behavior must have explicit ownership and transition rules.
- A shared tensor existing is not sufficient to give it a structural role.

Distinguish direct gradient routing from any explicitly supported unrolled
optimization behavior. Preserve the latter deliberately rather than accidentally
changing it while moving detach operations into helpers.

## Implementation checklist

- [ ] Introduce the immutable policy and centralized validation.
- [ ] Resolve complete phase policies before applying them.
- [ ] Replace multi-attribute phase updates with atomic policy installation.
- [ ] Replace manual save/restore code with scoped overrides.
- [ ] Extract order, structural-existence, reconstruction-weight, and composition helpers.
- [ ] Preserve step-level noise reuse and BKD sampling behavior.
- [ ] Add policy transition, exception-restoration, and gradient-ownership tests.
- [ ] Verify numerical equivalence for supported existing configurations.
- [ ] Verify edge diagnostics remain independent of G_R.
- [ ] Verify structural L0 never enters reconstruction loss.
- [ ] Test shared/handoff gates for single application and correct ownership.
- [ ] Retain normalization tests for extreme scores, tiny/zero priors, and empty rows.
- [ ] Build both ORDER_FIRST_2 experiments and run focused regression tests.
- [ ] Remove obsolete setters, mutable attributes, and duplicated validation.

## Scope limits

Start with one small policy object, one resolver, and focused helpers. Avoid a
general-purpose policy framework or a selector class hierarchy until concrete
requirements justify them. Backward compatibility is not a design requirement.

Success means a future reconstruction activation, such as entmax, can be added
to reconstruction-weight computation without changing edge semantics, L0
routing, phase transitions, or FiLM. Any numerical change should be a separate,
explicitly tested research change rather than hidden inside this refactor.