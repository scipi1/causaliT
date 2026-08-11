"""
Size-aware Optuna search space for the grouped DAG sweep.

Why this module exists
----------------------
A scaling benchmark is only fair if every DAG size gets a model of PROPORTIONAL
capacity.  A fixed search space cannot do that: a width that is generous for a
6-node graph is structurally illegal for a 400-node one, because the orthogonal
fixed structural embedding needs at least one dimension per node
(``d_model >= n_keys``, see ``orthogonal_embedding``).  So the space itself must
be a function of the sampled DAG.

Four decisions are implemented here, each isolated in a pure function so it can
be unit-tested without touching Optuna or a GPU:

1. ADAPTIVE WIDTH.  ``d_model`` is drawn from a bounded list of aligned widths in
   ``[n_keys, size_mult * n_keys]``.  The lower end is the hard feasibility
   floor; the upper end is the "twice as much room as strictly needed" ceiling.
   The list length is capped (``max_choices``) so the number of candidates does
   NOT grow with the DAG - a 400-node graph has as many choices as a 6-node one.

2. RECONSTRUCTION PROTOCOL.  Trials are scored on reconstruction ALONE: gradient
   routing off, every structural lambda zeroed, one fold, early stopping, best
   checkpoint.  Rationale: capacity is a property of the value/function stream,
   and the structural objective (HSIC) itself rewards low residuals, so a model
   that cannot reconstruct cannot do structure either.  Tuning under the full
   objective would confound capacity with the structure/reconstruction
   trade-off, which is exactly what the benchmark wants to measure.

3. PARSIMONIOUS SELECTION.  Reconstruction error is near-monotone in capacity,
   so plain ``argmin`` would always return the largest model in the range and the
   search would do no work.  Instead we take the KNEE: the SMALLEST model whose
   metric is within ``tol`` (relative) of the best observed.  Model size is read
   from the real ``trainable_params`` count that the trainer already reports.

4. SIZE-DERIVED FIELDS.  Some config fields are not hyper-parameters at all, they
   are functions of the node count and must be recomputed per group:
   * ``batch_size`` - from an activation budget, so large DAGs do not OOM;
   * ``query_fanin_scale`` - ``F = n_keys * x_sat^2`` (opt-in for now).

All samplers emit DOTTED config paths as the Optuna parameter names.  This is
load-bearing: ``best_trial.yaml`` stores ``trial.params`` verbatim and the sweep
applies those keys with ``OmegaConf.update``, so a short name like ``d_model_set``
would create a dead top-level key and every tuned value would be silently
discarded.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from omegaconf import OmegaConf

from causaliT.utils.query_norm import (
    DEFAULT_GATE_GAMMA,
    DEFAULT_GATE_ZETA,
    gate_tau_from_experiment,
    is_auto_fanin,
    kappa_1,
)



logger = logging.getLogger(__name__)

# Default alignment / range of the adaptive width list.
DEFAULT_ALIGN = 8
DEFAULT_SIZE_MULT = 2.0
DEFAULT_MAX_CHOICES = 8

# Activation-budget constant (see batch_budget.calibrate_activation_budget).
# Placeholder tuned for a 24 GB card; calibrate per device before large runs.
DEFAULT_ACTIVATION_BUDGET = 4.9e8
DEFAULT_MIN_BATCH = 32
DEFAULT_MAX_BATCH = 2048


# =============================================================================
# Search-only training protocol
# =============================================================================

#: Reconstruction-only protocol for the capacity search.
#:
#: Every structural term is zeroed and gradient routing is disabled, so the trial
#: measures ONLY how well the model can reproduce the data given the (untrained)
#: structure.  ``k_fold=1`` keeps a trial to a single fit; early stopping plus
#: best-checkpoint scoring means a trial ends at its own plateau instead of
#: burning the full epoch budget.
RECONSTRUCTION_PROTOCOL: Dict[str, Any] = {
    "training.k_fold": 1,
    "training.use_gradient_routing": False,
    "training.lambda_recon": 1.0,
    "training.lambda_struct_recon": 0.0,
    # Structural objectives (unified + legacy cross/self variants).
    "training.lambda_hsic": 0.0,
    "training.lambda_hsic_cross": 0.0,
    "training.lambda_hsic_self": 0.0,
    "training.lambda_l0": 0.0,
    "training.kappa": 0.0,
    "training.lambda_group_l1": 0.0,
    "training.lambda_query_norm": 0.0,
    "training.lambda_score_sparse": 0.0,
    "training.lambda_self_score_sparse": 0.0,
    "training.lambda_cross_score_sparse": 0.0,
    "training.lambda_noise_prior": 0.0,
    "training.lambda_sparse": 0.0,
    "training.lambda_sparse_cross": 0.0,
    # Diagnostics that cost time and need structural grads.
    "training.log_l0_hsic_interference": False,
    # Stop a trial when it stops improving, and score its BEST epoch.
    "training.early_stopping.enabled": True,
    "training.early_stopping.monitor": "val_x_mae",
    "training.early_stopping.patience": 10,
    "training.early_stopping.min_delta": 1.0e-5,
    "training.early_stopping.mode": "min",
    # No post-training evaluation suite: a trial is scored on val_x_mae alone, so
    # SHD/MEC (eval_attention_scores) and the Monte-Carlo ATE (eval_interventions,
    # which rebuilds the SCM and intervenes per node) are pure cost - and they are
    # meaningless here anyway, since every structural lambda above is zero and the
    # DAG therefore never trains.
    #
    # MUST be an EMPTY LIST, never None/null: both trainer._run_post_training_evaluations
    # and run_evaluations_from_config treat `functions is None` as "run ALL default
    # evaluations", so null would do the OPPOSITE of disabling them.
    "evaluation.functions": [],
}


SEARCH_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "reconstruction": RECONSTRUCTION_PROTOCOL,
    "none": {},
}


def apply_protocol(config: Any, name: str,
                   extra_overrides: Optional[Dict[str, Any]] = None) -> Any:
    """
    Apply a named search protocol (plus free-form overrides) to a config.

    Mutates and returns ``config``.  Unknown protocol names raise: silently
    training under the full objective would invalidate the whole search.
    """
    if name not in SEARCH_PROTOCOLS:
        raise ValueError(
            f"Unknown search protocol '{name}'. Available: {sorted(SEARCH_PROTOCOLS)}."
        )
    OmegaConf.set_struct(config, False)
    for dotted, value in SEARCH_PROTOCOLS[name].items():
        OmegaConf.update(config, dotted, value, merge=True)

    # The adaptive trainer keeps its OWN copies of several settings, so the flat
    # `training.*` overrides above are not enough to neutralise a config written
    # for it.  Mirror them here (only when the block exists, so a `standard`
    # config is not given a spurious one).
    if name == "reconstruction" and "adaptive_training" in config:
        # Private L0 weight: without this a config written for `adaptive` could
        # smuggle structure into a trial.
        OmegaConf.update(config, "adaptive_training.structure.lambda_l0", 0.0,
                         merge=True)
        # Second evaluation path: `eval_dag` runs DAG diagnostics at EVERY phase
        # switch, inside the fit, so `evaluation.functions: []` alone does not
        # stop it.  Meaningless under this protocol (the DAG never trains).
        OmegaConf.update(config, "adaptive_training.eval_dag", False, merge=True)
        # Post-training suite: skip the step entirely rather than dispatch an
        # empty list (avoids importing the evaluation package per trial).
        OmegaConf.update(config, "adaptive_training.run_final_evaluations", False,
                         merge=True)


    for dotted, value in (extra_overrides or {}).items():
        OmegaConf.update(config, str(dotted), value, merge=True)
    return config


# =============================================================================
# Node count
# =============================================================================

def n_keys_from_metadata(datasets_dir: str, dataset_name: str,
                         fallback: Optional[int] = None) -> int:
    """
    Number of attention keys (= nodes) of a sampled dataset.

    Read from ``dataset_metadata.json`` rather than from the ``n_nodes`` group
    axis, so the value stays correct when the groups are formed along another
    axis (``degree``, ``linearity``, ...) or when the generator adjusts the node
    count.  ``fallback`` (typically the axis value) is used only if the metadata
    is unavailable.
    """
    from causaliT.evaluation.eval_funs.helpers.eval_utils import load_dataset_metadata

    metadata = load_dataset_metadata(datasets_dir, dataset_name) or {}
    info = metadata.get("variable_info", {}) if isinstance(metadata, dict) else {}
    sources = info.get("source_labels") or []
    inputs = info.get("input_labels") or []
    n_keys = len(sources) + len(inputs)
    if n_keys > 0:
        return int(n_keys)
    if fallback is None:
        raise ValueError(
            f"Cannot determine the node count of dataset '{dataset_name}' in "
            f"{datasets_dir} and no fallback was given."
        )
    logger.warning("No variable_info for %s; falling back to n_keys=%s",
                   dataset_name, fallback)
    return int(fallback)


# =============================================================================
# Adaptive width list
# =============================================================================

def _ceil_to(value: float, align: int) -> int:
    return int(math.ceil(float(value) / align) * align)


def model_width_choices(n_keys: int, align: int = DEFAULT_ALIGN,
                        size_mult: float = DEFAULT_SIZE_MULT,
                        max_choices: int = DEFAULT_MAX_CHOICES) -> List[int]:
    """
    Admissible ``d_model`` values for a DAG with ``n_keys`` nodes.

    The list spans ``[ceil(n_keys), ceil(size_mult * n_keys)]`` in multiples of
    ``align``, always contains both endpoints, and never exceeds ``max_choices``
    entries - so the search cost is independent of the DAG size.

    Examples (align=8, size_mult=2, max_choices=8)::

        n_keys=6   -> [8, 16]
        n_keys=10  -> [16, 24]
        n_keys=64  -> [64, 72, 80, 88, 96, 104, 120, 128]
        n_keys=400 -> [400, 456, 512, 568, 624, 680, 744, 800]

    The lower bound is a hard feasibility constraint: the orthogonal fixed
    structural embedding needs one dimension per node.
    """
    if n_keys < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys}")
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")
    if size_mult < 1.0:
        raise ValueError(f"size_mult must be >= 1.0, got {size_mult}")
    if max_choices < 2:
        raise ValueError(f"max_choices must be >= 2, got {max_choices}")

    low = _ceil_to(n_keys, align)
    high = _ceil_to(size_mult * n_keys, align)
    if high <= low:
        return [low]

    n_steps = (high - low) // align + 1          # every aligned width in range
    if n_steps <= max_choices:
        return [low + i * align for i in range(n_steps)]

    # Evenly spaced subset, endpoints pinned, snapped back onto the grid.
    picks = []
    for i in range(max_choices):
        frac = i / (max_choices - 1)
        picks.append(low + int(round(frac * (high - low) / align)) * align)
    return sorted(set(picks))


# =============================================================================
# Parameter sampler (dotted Optuna names)
# =============================================================================

def build_sample_params_fn(search_space: Dict[str, Any],
                           n_keys: int) -> Callable[[Any], Dict[str, Any]]:
    """
    Build an Optuna ``sample_params`` callable from a declarative spec.

    ``search_space`` maps a DOTTED config path to a distribution::

        experiment.d_model_set: {type: adaptive_width, align: 8, size_mult: 2.0}
        experiment.n_heads:     {type: categorical, choices: [1, 2, 4]}
        training.lr:            {type: float, low: 1.0e-4, high: 5.0e-3, log: true}

    The dotted path is used as the Optuna parameter name as well, so
    ``trial.params`` (and therefore ``best_trial.yaml``) is directly applicable
    with ``OmegaConf.update``.  Adding a hyper-parameter for another benchmark
    model is a YAML edit, not a code change.
    """
    if not search_space:
        raise ValueError("search_space is empty: nothing to optimise.")

    spec: Dict[str, Dict[str, Any]] = {}
    for dotted, raw in search_space.items():
        entry = dict(raw) if isinstance(raw, dict) else {}
        if "type" not in entry:
            raise ValueError(f"search_space['{dotted}'] must declare a 'type'.")
        spec[str(dotted)] = entry

    def sample(trial: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for dotted, entry in spec.items():
            kind = str(entry["type"])
            if kind == "adaptive_width":
                choices = model_width_choices(
                    n_keys,
                    align=int(entry.get("align", DEFAULT_ALIGN)),
                    size_mult=float(entry.get("size_mult", DEFAULT_SIZE_MULT)),
                    max_choices=int(entry.get("max_choices", DEFAULT_MAX_CHOICES)),
                )
                params[dotted] = trial.suggest_categorical(dotted, choices)
            elif kind == "categorical":
                params[dotted] = trial.suggest_categorical(
                    dotted, list(entry["choices"])
                )
            elif kind == "int":
                params[dotted] = trial.suggest_int(
                    dotted, int(entry["low"]), int(entry["high"]),
                    step=int(entry.get("step", 1)),
                )
            elif kind == "float":
                params[dotted] = trial.suggest_float(
                    dotted, float(entry["low"]), float(entry["high"]),
                    log=bool(entry.get("log", False)),
                )
            else:
                raise ValueError(
                    f"Unknown search_space type '{kind}' for '{dotted}'. "
                    "Use adaptive_width | categorical | int | float."
                )
        return params

    return sample


# =============================================================================
# Size-derived (non-tunable) fields
# =============================================================================

def _prev_pow2(value: float) -> int:
    if value < 1:
        return 1
    return 2 ** int(math.floor(math.log2(value)))


def activation_batch_size(n_keys: int, d_model: int, n_heads: int,
                          budget: float = DEFAULT_ACTIVATION_BUDGET,
                          min_batch: int = DEFAULT_MIN_BATCH,
                          max_batch: int = DEFAULT_MAX_BATCH) -> int:
    """
    Largest power-of-two batch size that fits an activation budget.

    Peak activation memory of one attention block scales as the value stream
    ``B * N * d * H`` plus the attention maps ``B * H * N * N``, i.e.

        cost ~ B * N * H * (N + d)

    so ``B = budget / (N * H * (N + d))``, snapped down to a power of two and
    clamped.  ``budget`` is a single device-specific constant, measured by
    ``batch_budget.calibrate_activation_budget``.

    Deriving the batch size instead of fixing it is what allows one config to
    cover 6-node and 400-node DAGs without OOM.  The SAME derived value must be
    used in the search and in the evaluation runs, otherwise the tuned learning
    rate is calibrated for a batch that never occurs.
    """
    denom = float(n_keys) * float(n_heads) * (float(n_keys) + float(d_model))
    raw = float(budget) / max(denom, 1.0)
    return int(min(max(_prev_pow2(raw), int(min_batch)), int(max_batch)))


def saturating_query_fanin(config: Any, n_keys: int) -> float:
    """
    ``query_fanin_scale`` that puts the centroid initialisation at gate maximum.

    With a unit-norm query on an orthonormal key frame the centroid logit is
    ``sqrt(F / n)``, and the Hard-Concrete gate saturates (z = 1) at

        x_sat = init_tau * ln((1 - init_gamma) / (init_zeta - 1)) + init_edge_offset

    so ``F = n_keys * x_sat^2``.  F therefore SCALES WITH THE NODE COUNT: a value
    derived for 10 nodes under-scores a 400-node row by a factor of 40.
    See docs/experimental_elaborations/QUERY_FANIN_SCALE_BUDGET.md.

    This PINS F to gate saturation (z = 1).  Leaving ``query_fanin_scale: auto``
    in the config is the general alternative: it targets an explicit centroid
    posterior ``query_centroid_max_p`` instead (query_norm.py).

    ``x_sat = kappa_1 + T`` is eq (2e) of
    docs/experimental_elaborations/QUERY_NORM_CAPACITY_AND_FANIN_PRIOR.md; the
    constant is imported from ``query_norm`` so it has ONE definition.
    """
    exp = config.get("experiment", {}) if config is not None else {}
    homogeneous = bool(exp.get("homogeneous_nodes", False))
    # The split keys (init_tau_cross / init_tau_self) win over the legacy
    # shared init_tau via gate_tau_from_experiment; in homogeneous mode the
    # single block IS the self gate and carries no edge offset.
    tau = gate_tau_from_experiment(exp, homogeneous)
    gamma = float(exp.get("init_gamma", None) or DEFAULT_GATE_GAMMA)
    zeta = float(exp.get("init_zeta", None) or DEFAULT_GATE_ZETA)
    offset = 0.0 if homogeneous else float(exp.get("init_edge_offset", 0.0) or 0.0)
    x_sat = kappa_1(tau, gamma, zeta) + offset
    return float(n_keys) * x_sat ** 2




#: Recipes usable in ``dagsweep.yaml``'s ``size_derived`` block.
SIZE_DERIVED_RULES = ("activation_budget", "fanin_saturating")


def derive_size_fields(config: Any, n_keys: int,
                       size_derived: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Write size-derived config fields and return ``{dotted: value}`` for logging.

    Mutates ``config``.  Only the fields listed in ``size_derived`` are touched,
    so enabling a recipe is an explicit, reviewable decision.
    """
    written: Dict[str, Any] = {}
    if not size_derived:
        return written

    OmegaConf.set_struct(config, False)
    exp = config.get("experiment", {})

    for dotted, raw in size_derived.items():
        entry = dict(raw) if isinstance(raw, dict) else {"rule": str(raw)}
        rule = str(entry.get("rule", ""))

        if rule == "activation_budget":
            from causaliT.euler_sweep.euler_sweep.batch_budget import resolve_budget

            # Reference width = the CEILING of the adaptive range, not the
            # sampled width.  The batch size is then a pure function of the DAG
            # size, so it is identical for every trial of a group AND for the
            # evaluation runs - a tuned learning rate stays valid, and the
            # largest candidate model still fits.
            d_ref = float(entry.get("d_ref_mult", DEFAULT_SIZE_MULT)) * n_keys
            value = activation_batch_size(
                n_keys=n_keys,
                d_model=int(math.ceil(d_ref)),
                n_heads=int(entry.get("n_heads") or exp.get("n_heads") or 1),
                budget=resolve_budget(entry.get("C", None)),
                min_batch=int(entry.get("min", DEFAULT_MIN_BATCH)),
                max_batch=int(entry.get("max", DEFAULT_MAX_BATCH)),
            )
        elif rule == "fanin_saturating":
            value = saturating_query_fanin(config, n_keys)
        else:
            raise ValueError(
                f"Unknown size_derived rule '{rule}' for '{dotted}'. "
                f"Available: {SIZE_DERIVED_RULES}."
            )

        OmegaConf.update(config, str(dotted), value, merge=True)
        written[str(dotted)] = value

    if written:
        logger.info("  size-derived (n_keys=%d): %s", n_keys, written)
    return written


# =============================================================================
# Dimension validation
# =============================================================================

def validate_dimensions(config: Any, n_keys: int, repair: bool = True,
                        align: int = DEFAULT_ALIGN,
                        fanin_tolerance: float = 0.2) -> Dict[str, Any]:
    """
    Make a config dimensionally consistent with a DAG of ``n_keys`` nodes.

    Every rule below has a unique, well-defined right answer, so with
    ``repair=True`` (default) the value is CORRECTED and logged loudly instead of
    aborting: a scaling sweep spends hours per run, and killing one because a
    derived number was stale wastes the run for no information.  Set
    ``repair=False`` to get the exception instead (used by the tests and useful
    for a strict re-run of published numbers).

    Rules, in dependency order:

    1. ``d_model >= n_keys`` - the orthogonal fixed structural frame needs one
       dimension per node.  Repair: raise the width to the next multiple of
       ``align`` at or above ``n_keys``.
    2. ``d_model % n_heads == 0`` - the value stream splits the width across
       heads.  Repair: raise the width to the next multiple of ``n_heads``
       (raising, never lowering, so capacity is never silently reduced).
    3. ``d_qk * n_heads_struct == d_model`` when W_q / W_K are removed - without
       the projections the embedding IS the query/key.  Repair: set ``d_qk``
       explicitly to ``d_model // n_heads_struct``.
    4. ``query_fanin_scale ~= n_keys * x_sat^2`` - F scales with the node count,
       and a stale value silently changes how many parents a row can afford.
       Repair: write the saturating value.

    Returns ``{dotted: new_value}`` for everything it changed (empty dict when
    the config was already consistent), so the caller can log/record it.
    """
    repairs: Dict[str, Any] = {}
    if config is None:
        return repairs
    exp = config.get("experiment", {})
    d_model = exp.get("d_model_set", None)
    if d_model is None:
        return repairs
    d_model = int(d_model)

    def _fail_or_fix(dotted: str, value: Any, message: str) -> None:
        if not repair:
            raise ValueError(message)
        logger.warning("%s -> setting %s = %s", message, dotted, value)
        OmegaConf.set_struct(config, False)
        OmegaConf.update(config, dotted, value, merge=True)
        repairs[dotted] = value

    # 1. Structural embedding floor.
    if d_model < n_keys:
        fixed = _ceil_to(n_keys, align)
        _fail_or_fix(
            "experiment.d_model_set", fixed,
            f"d_model_set={d_model} < n_keys={n_keys}: the orthogonal fixed "
            "structural embedding needs one dimension per node",
        )
        d_model = fixed

    # 2. Head divisibility (value stream).
    n_heads = int(exp.get("n_heads", 1) or 1)
    if n_heads > 0 and d_model % n_heads != 0:
        fixed = int(math.ceil(d_model / n_heads) * n_heads)
        _fail_or_fix(
            "experiment.d_model_set", fixed,
            f"d_model_set={d_model} is not divisible by n_heads={n_heads}: the "
            "value stream splits the width across heads",
        )
        d_model = fixed

    # 3. Q/K width when the projections are removed.
    if exp.get("remove_query_projection", False) or exp.get("remove_key_projection", False):
        shared = bool(exp.get("shared_dag_across_heads", True))
        n_heads_struct = 1 if shared else n_heads
        expected = d_model // max(n_heads_struct, 1)
        d_qk = exp.get("d_qk", None)
        if d_qk is None:
            mult = exp.get("d_qk_mult", None)
            d_qk = max(1, int(round(float(mult) * d_model))) if mult is not None else None
        if d_qk is not None and int(d_qk) * n_heads_struct != d_model:
            _fail_or_fix(
                "experiment.d_qk", expected,
                f"remove_query/key_projection requires d_qk * n_heads_struct == "
                f"d_model, got {d_qk} * {n_heads_struct} != {d_model}",
            )

    # 4. Fan-in scale (F = n * x_sat^2).
    # ``auto`` (or null) is NOT a violation: F is then derived from n_keys at
    # data-load time by causaliT.utils.query_norm.resolve_query_fanin_scale,
    # which is exactly what this check enforces for pinned values.
    #
    # ``query_norm: true`` OWNS the scale: resolve_query_norm derives
    # F = x(p*) sqrt(N) from the target posterior (which is NOT the saturation
    # point x_sat unless p* happens to equal sigmoid(kappa_1)), and a declared
    # ``fanin_prior`` is priced on the penalty target mu, not on F.  "Repairing"
    # F here would silently overwrite a deliberate capacity with the saturating
    # value and delete the feature, so the rule is skipped entirely.
    fanin = exp.get("query_fanin_scale", None)
    if bool(exp.get("query_norm", False)):
        logger.info(
            "  [validate_dimensions] query_norm=true: query_fanin_scale is owned "
            "by resolve_query_norm (F = x(p*)*sqrt(n_keys)); skipping the "
            "saturating-F check."
        )
    elif fanin is not None and not is_auto_fanin(fanin):

        recommended = saturating_query_fanin(config, n_keys)

        if recommended > 0 and abs(float(fanin) - recommended) / recommended > fanin_tolerance:
            _fail_or_fix(
                "experiment.query_fanin_scale", recommended,
                f"query_fanin_scale={float(fanin):.4g} does not match n_keys="
                f"{n_keys} (F = n * x_sat^2 = {recommended:.4g}); F scales with "
                "the node count",
            )

    return repairs


# =============================================================================
# Best-trial selection
# =============================================================================

def _capacity_of(trial: Any, capacity_params: Sequence[str]) -> float:
    """
    Model size of a trial: the real parameter count when available.

    ``trainable_params`` is already reported by the trainer and aggregated into
    ``trainable_params_mean``, so the knee is measured on the actual model size.
    The product of the capacity hyper-parameters is only a fallback (e.g. for a
    stubbed study in the tests).
    """
    for key in ("trainable_params_mean", "trainable_params"):
        value = trial.user_attrs.get(key)
        if value:
            return float(value)
    product = 1.0
    for name in capacity_params:
        product *= float(trial.params.get(name, 1))
    return product


def select_best(study: Any, selection: Optional[Dict[str, Any]] = None,
                direction: str = "minimize") -> Dict[str, Any]:
    """
    Pick the trial that best DIMENSIONS the model, not merely the argmin.

    ``mode="parsimonious"`` (default) returns the SMALLEST model whose metric is
    within ``tol`` (relative) of the best observed value; ``mode="argmin"``
    returns Optuna's own best trial.

    Why parsimony: reconstruction error decreases monotonically with capacity, so
    argmin always lands on the largest width in the range and the adaptive range
    buys nothing.  The knee is the honest answer to "how much capacity does this
    DAG size actually need", and it is the quantity a scaling benchmark reports.

    Only COMPLETE trials are considered (a failed trial has no metric).
    """
    import optuna

    cfg = dict(selection or {})
    mode = str(cfg.get("mode", "parsimonious"))
    tol = float(cfg.get("tol", 0.02))
    capacity_params = [str(p) for p in cfg.get("capacity_params", [])]

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not complete:
        # Every trial crashed.  Reporting only "no completed trial" hides the
        # actual cause (an OOM, a bad dimension, a dataloader failure), so echo
        # the recorded reason of the last failures - that is the information the
        # user needs, and the study object is the only place that still has it.
        failed = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.FAIL]
        details = []
        for t in failed[-3:]:
            reason = (t.user_attrs.get("failure")
                      or t.system_attrs.get("fail_reason")
                      or "reason not recorded")
            details.append(f"  trial {t.number} {dict(t.params)}: {reason}")
        raise ValueError(
            "No completed trial: cannot select hyper-parameters.\n"
            f"{len(failed)}/{len(study.trials)} trial(s) FAILED"
            + (":\n" + "\n".join(details) if details else ".")
            + "\nFix the underlying training failure above (the search itself is "
              "fine); nothing about the search space can rescue a run that "
              "cannot train."
        )

    best = study.best_trial
    result: Dict[str, Any] = {
        "mode": mode,
        "tol": tol,
        "raw_best": {"trial": best.number, "value": float(best.value),
                     "params": dict(best.params)},
    }

    if mode == "argmin":
        chosen = best
    elif mode == "parsimonious":
        # Relative tolerance band around the best value.  For a minimised metric
        # (MAE) the band is [best, best * (1 + tol)]; for a maximised one it is
        # [best * (1 - tol), best].
        if str(direction).lower().startswith("max"):
            threshold = float(best.value) * (1.0 - tol)
            within = [t for t in complete if float(t.value) >= threshold]
        else:
            threshold = float(best.value) * (1.0 + tol)
            within = [t for t in complete if float(t.value) <= threshold]
        # Smallest model inside the band; ties broken by the better metric.
        chosen = min(
            within,
            key=lambda t: (_capacity_of(t, capacity_params), float(t.value)),
        )
        result["threshold"] = threshold
        result["n_within_tol"] = len(within)
    else:
        raise ValueError(
            f"Unknown selection mode '{mode}'. Use 'parsimonious' or 'argmin'."
        )

    result["trial_number"] = chosen.number
    result["optimization_value"] = float(chosen.value)
    result["params"] = dict(chosen.params)
    result["capacity"] = _capacity_of(chosen, capacity_params)
    result["metrics"] = {
        k: (float(v) if isinstance(v, (int, float)) else v)
        for k, v in chosen.user_attrs.items() if k != "config_path"
    }
    result["config_path"] = chosen.user_attrs.get("config_path")
    result["curve"] = [
        {"trial": t.number, "value": float(t.value),
         "capacity": _capacity_of(t, capacity_params), "params": dict(t.params)}
        for t in sorted(complete, key=lambda t: _capacity_of(t, capacity_params))
    ]
    return result


__all__ = [
    "DEFAULT_ACTIVATION_BUDGET",
    "RECONSTRUCTION_PROTOCOL",
    "SEARCH_PROTOCOLS",
    "SIZE_DERIVED_RULES",
    "activation_batch_size",
    "apply_protocol",
    "build_sample_params_fn",
    "derive_size_fields",
    "model_width_choices",
    "n_keys_from_metadata",
    "saturating_query_fanin",
    "select_best",
    "validate_dimensions",
]
