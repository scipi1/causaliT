# Prior-Softmax Reconstruction Gain (GainSoftmax)

Status: implemented (module + wiring + trainer schedule + tests).  Arm:
`experiments/7_PUBLISH/ATE_FIXCAP/svfa_gain/`.

## Motivation

The gated attentions apply the structure gate directly as the attention weight,
`A = z` (Hard-Concrete, binary).  Every selected parent therefore contributes
with weight ~1 and the value stream carries all magnitude.  This is provably
rigid: with `out_i = sum_j z_ij * f_j(v_j) + (sum_j z_ij) * g(e_i)`, the
per-source functional `f_j` is shared across the children of `j` and the
per-child term scales with the parent *count*, so **no per-edge coefficient is
expressible** (a parent feeding two children with different weights, or a child
mixing two parents, cannot be represented).  The cheater arm (softmax over the
true support) holds exactly this advantage: continuous, data-dependent,
per-edge weights.

The GainSoftmax stage adds that freedom *within the gate's own support*,
without repeating the removed `A = z * g` gain stream (unnormalised, co-trained
with the structure -> it annealed whole rows to zero).

## Mechanism

```
A_ij = (1 - lambda) * z_ij  +  lambda * n_i * z_ij * exp(s_ij) / D_i
D_i  = sum_k z_ik * exp(s_ik)          (row normaliser)
n_i  = sum_k z_ik                       (row mass, DETACHED)
s_ij = a_ij + <q^v_i, k^v_j> / sqrt(d_g)
```

At `lambda = 1` the applied weight is `n_i * softmax(log z + s)`: the gate
enters the softmax as a **multiplicative prior** (equivalently a `log z`
additive log-prior) and the gain score `s` only **redistributes** the row's
existing mass `n_i` across the gate's support.  This is Bayes' rule: the gate
is the prior probability of the edge, `s` a data log-likelihood-ratio, and the
applied attention the posterior.

* `a_ij` — a zero-initialised static per-edge logit table `(L, S)` living in
  the `GainSoftmax` module (one per gated block).
* `<q^v_i, k^v_j>` — the optional data-dependent term (`gain_data`), computed
  by the `AttentionLayer` from the caller-supplied gain tensors: the
  reconstruction-routed value-identity tables (`val_id_embed_*` /
  `val_q_id_embed_*`, the "separate"/"learned_sum" schemes) and the value
  stream.  **Never the structural embeddings** — the gain carries no
  structural signal.  `gain_softmax_k_proj` is zero-initialised so the data
  term is exactly 0 at init.

## Properties (by construction)

* `lambda = 0` — `A = z`: the gate-only baseline runs bit-identically (the
  whole alternating schedule).
* `s = 0` (zero-init) — `softmax(log z) = z / n_i`, so `A = z` at ANY lambda:
  the gain is an exact identity at initialisation and departs from uniform
  only when the reconstruction loss pulls it there (smooth turn-on).
* `z_ij = 0` — `A_ij = 0` EXACTLY (both terms): forbidden and gated-off edges
  keep their by-construction zeros, so the ATE "zero" categories and the DAG
  extraction (which thresholds the gate posterior, returned unchanged in the
  second slot) are unaffected.
* Row mass preserved — `sum_j A_ij = sum_j z_ij` at every lambda: the gain can
  neither amplify beyond the gate's row mass nor collapse a row to zero.  The
  old gain-stream failure mode is structurally impossible.
* Differentiable in `z` everywhere, INCLUDING `z = 0` (the product form
  `z * exp(s) / D` never evaluates `log z`): `dA_j/dz_j = exp(s_j)/D > 0` at
  `z_j = 0`, so a falsely closed edge receives finite re-opening pressure from
  the reconstruction loss — the structure keeps training in its new role as
  the softmax prior.  The gradient reaches the structural logits through the
  same Hard-Concrete relaxation the L0 penalty already uses.

## Guards

* All-off row (`D_i = 0`): the gain term is zeroed for that row
  (`torch.where`), reproducing the gate-only zero row.
* `n_i` is detached: the gain cannot inflate the row mass.

## Training schedule (adaptive trainer)

The gain turns on DURING the alternating schedule — there is NO separate final
reconstruction-only phase.  The interpolation weight `lambda` ramps
`0 -> gain_lambda_final` driven by the GLOBAL epoch (phase-agnostic), so the
gate's role morphs from the multiplicative weight to the softmax support while
the trainer keeps alternating reconstruct <-> structure:

```yaml
adaptive_training:
  gain_lambda_start: 2500        # global epoch where the turn-on begins
                                 #   (null -> 0.5 * total_epoch_budget)
  gain_lambda_ramp: 500          # epochs to ramp lambda 0 -> final (0 = jump)
  gain_lambda_final: 1.0         # ramp target (1.0 = full prior-softmax)
```

The alternating dynamics need no extra machinery:

* **reconstruct phases**: the structure is frozen, the gain (reconstruction
  group) trains within the current support;
* **structure phases**: the gain is frozen, the gate trains under HSIC + L0 +
  NOTEARS.  To ALSO let the reconstruction-through-prior gradient reach the
  gate, set `training.lambda_struct_recon > 0` (the pre-existing
  reconstruction-into-structure mixing knob, `0.0` by default) — an optional,
  separate ablation axis.

* `gain_lambda` is a PERSISTENT buffer, so a trained checkpoint carries its
  final lambda and the evaluation suite reloads the model with the gain ACTIVE.

## Post-hoc probe on existing runs (anneal_reconstruction.py)

The idea can be tested on an already-trained SVFA run WITHOUT retraining the
structure — the gain modules are created by a config override and start at the
exact identity (zero-init), so only the gain phase is new:

```bash
python scripts/anneal_reconstruction.py --run_dir <path/to/svfa_run> \
    --override model.kwargs.use_gain_softmax=true \
    --override model.kwargs.gain_data=true \
    --mode frozen --gain_lambda 1.0 --gain_ramp 50
```

`--mode frozen` keeps the structure frozen (gain trains in the reconstruction
group); `--mode joint` lets the structure keep training as the prior.  The
annealed pseudo-experiment carries the gain-active checkpoint, so the standard
ATE / DAG evaluations run on it unchanged.

## Gradient routing

All gain parameters (`gain_static_logits`, `gain_softmax_q_proj`,
`gain_softmax_k_proj`) are named WITHOUT the structural routing patterns
(`query_projection` / `key_projection` / `query_embed` / `log_gain` / ...), so
the name-based gradient router classifies them as RECONSTRUCTION parameters.

## Tests

`tests/test_atsel_gain_softmax.py`: the invariants above (identity at
lambda=0 / at zero-init, exact zeros, mass preservation, all-off row,
re-opening pressure), the inner-attention / layer / model wiring, the
gradient-routing classification, and the PhaseController's phase-agnostic,
global-epoch-driven lambda ramp.
