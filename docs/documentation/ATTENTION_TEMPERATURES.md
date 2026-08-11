# Attention Temperatures

The gated attention modules carry THREE temperatures, all fixed (non-learnable,
never annealed) constants.  This document is the single reference for what each
one does, where it is wired, and how its default value was CALCULATED.

| # | Temperature | Config key (`experiment.`) | Architecture kwarg | Module attribute | Gate it scales |
|---|-------------|----------------------------|--------------------|------------------|----------------|
| 1 | Cross-attention Hard-Concrete | `init_tau_cross` | `init_tau_cross` | `GatedCrossAttention.beta` / `HardConcreteCrossAttention.beta` | S->X existence gate z |
| 2 | Self-attention Hard-Concrete | `init_tau_self` | `init_tau_self` | `GatedSelfAttention.beta` / `CommutatorSelfAttention.beta` | X->X existence gate z_edge (also the single block in homogeneous mode) |
| 3 | Self-attention asymmetric term | `dir_tau_self` | `dir_tau_self` (`dir_tau` = legacy fallback) | `*.dir_beta` | antisymmetric direction gate d |

Fallback chain (backwards compatibility): an unset split key falls back to the
legacy shared `init_tau` (resp. `dir_tau`); when those are also unset the
calculated defaults below apply.  `init_tau` alone remains the activation
temperature of the NON-gated attentions (`CausalCrossAttention` /
`SigmoidCrossAttention` / `ToeplitzAttention`, default 3.0) - it is a different
quantity and is NOT a Hard-Concrete temperature.

The constants live in `causaliT/utils/query_norm.py`
(`DEFAULT_GATE_TAU`, `DEFAULT_GATE_GAMMA`, `DEFAULT_GATE_ZETA`,
`DEFAULT_DIR_TAU`) and are the Python-level defaults of all four gated modules,
so every entry point (config, architecture, bare module) agrees.

> A fourth temperature, `gain_tau` (reconstruction-gain sigmoid scale, default
> 1.0), exists on the gated modules.  The reconstruction gain is being
> dismantled, so it is deliberately NOT part of this harmonization.

## 1. Gate math (Louizos et al., ICLR 2018)

Existence gate (Hard-Concrete on a structural logit l = <q, k> * scale, with an
optional additive offset T = `init_edge_offset` on the cross gate only):

    s  = sigmoid((log u - log(1-u) + l - T) / tau)         u ~ U(0, 1)
    z  = clamp(s * (zeta - gamma) + gamma, 0, 1)

with the two derived threshold logits

    kappa   = tau * ln(-gamma / zeta)          (z leaves 0 above l - T = kappa)
    kappa_1 = tau * ln((1 - gamma) / (zeta-1)) (z pins at 1 above l - T = kappa_1)

and the closed-form edge posterior used by the L0 penalty and by evaluation:

    P(z > 0) = sigmoid(l - T - kappa)

Direction gate (plain Binary-Concrete, NO stretch, on the antisymmetric score
A_anti; coupled noise eps_ji = -eps_ij so d_ij + d_ji = 1 per sample):

    d_ij = sigmoid((eps_ij + A_anti_ij) / tau_dir)

## 2. The calculated defaults

### Existence gates: tau_cross = tau_self = 0.5, gamma = -1.1, zeta = 1.1

The stretch is chosen SYMMETRIC about the open/closed thresholds (zeta = 1.1 =
-gamma), which makes the opening threshold temperature-independent:

    kappa = tau * ln(1.1 / 1.1) = 0        for ANY tau.

Consequence: P(z>0) = sigmoid(l - T) - the gate posterior at zero logit is
exactly 1/2 (maximally undecided), with no temperature-induced bias.

The temperature then sets ONLY the sharpness, i.e. the saturation threshold:

    kappa_1 = 0.5 * ln((1 - (-1.1)) / (1.1 - 1)) = 0.5 * ln(21) = 1.5223

and the posterior at saturation

    p_sat = sigmoid(kappa_1 - kappa) = sigmoid(1.5223) = 0.8209

This is precisely the `query_centroid_max_p: 0.8209` operating point used by
the query-fanin calibration: a centroid-initialised query whose per-parent
logit reaches x(p*) = logit(p*) + T + kappa saturates the deterministic gate
(z_init = 1, see `query_norm.init_gate_at_centroid`).  tau = 0.5 is the value
used by every modern gated config (it replaced the paper default 2/3, whose
wider noise band delayed commitment without any measurable benefit).

### Direction gate: dir_tau_self = 2/3

The direction gate has no stretch, so it is undecided at A_anti = 0 by
symmetry.  Its temperature is the Louizos Binary-Concrete default 2/3.  The
check that matters is the ORDERING against the existence gate: direction
should commit slightly BEFORE existence saturates, so the directed posterior
P(z>0) * d is dominated by existence at the operating point.  The 0.9-point of
the direction gate is

    A_anti such that sigmoid(A_anti / tau_dir) = 0.9
        => A_anti = tau_dir * ln(9) = (2/3) * 2.1972 = 1.4648

which sits just below the existence saturation logit kappa_1 = 1.5223, as
required.  Equivalently, at l = kappa_1 the direction posterior is already

    sigmoid(1.5223 / (2/3)) = sigmoid(2.2835) = 0.9074.

### init_edge_offset (not a temperature; recorded here for completeness)

The cross existence gate additionally accepts T = `init_edge_offset`, an
additive logit shift that balances the S->X cross prior against a directed X->X
edge at initialisation: without it P_cross(init) = sigmoid(-kappa) = 0.5 while
a directed self edge starts at p_exist * d = 0.5 * 0.5 = 0.25.  T = ln 3 =
1.0986 lands P_cross(init) = sigmoid(-ln 3) = 0.25 - no 2x head start.

## 3. Interaction with the query-fanin derivation

`query_fanin_scale: auto` derives F = n_keys * (x(p*) / M)^2 at data-load time.
F is SHARED by the cross and self blocks, so the derivation needs ONE
existence-gate temperature: `query_norm.gate_tau_from_experiment` uses
`init_tau_cross` in split mode (the block that carries `init_edge_offset`) and
`init_tau_self` in homogeneous mode (the single block IS the self gate),
falling back to legacy `init_tau`.  Setting DIFFERENT cross/self temperatures
is legal but the gates then saturate at different logits, which the single-F
capacity calculus cannot represent - a warning is emitted in that case.

## 4. Why the temperatures are NOT annealed

Earlier iterations tried to anneal the temperature to make the model
artificially more decisive over training.  That machinery is REMOVED:

* `use_tau_act_annealing` / `tau_gate_*` / `tau_dir_*` wrote the learnable
  `log_tau_gate` / `log_tau_dir` parameters of the long-removed
  ToeplitzLieAttention - dead code.
* `use_tau_annealing` / `tau_anneal_*` / `freeze_tau_during_anneal` rewrote the
  constant float `tau` of the legacy non-gated attentions and never touched the
  Hard-Concrete `beta` / `dir_beta`.

Decisiveness is a property of the OPERATING POINT, not of a schedule: the
symmetric stretch fixes kappa = 0, tau = 0.5 fixes the saturation logit
kappa_1 = 1.5223, and the L0 penalty (`lambda_l0`) supplies the pressure
towards 0/1 gates.  If sharper commitment is needed, lower the temperature at
INIT (e.g. tau = 0.25 quarters the noise band) instead of annealing - the
change is then visible in the logged hparams and reproducible.
