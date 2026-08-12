"""
Tests for the size-aware Optuna search space of the grouped DAG sweep.

Covers the four decisions that make the search a fair MODEL-DIMENSIONING
procedure (causaliT/euler_sweep/euler_sweep/search_space.py):

1. the adaptive width range is feasible and size-independent in cost,
2. the sampler emits DOTTED config paths (so best_trial.yaml is applicable),
3. the reconstruction protocol really zeroes every structural term,
4. selection returns the KNEE (smallest model within tolerance), and the
   dimension rules repair a stale config instead of wasting a run.
"""

import math

import pytest
from omegaconf import OmegaConf

from causaliT.euler_sweep.euler_sweep.batch_budget import estimate_budget, resolve_budget
from causaliT.euler_sweep.euler_sweep.search_space import (
    RECONSTRUCTION_PROTOCOL,
    activation_batch_size,
    apply_protocol,
    build_sample_params_fn,
    derive_size_fields,
    model_width_choices,
    saturating_query_fanin,
    select_best,
    validate_dimensions,
)


# =============================================================================
# 1. Adaptive width range
# =============================================================================

@pytest.mark.parametrize("n_keys", [1, 6, 10, 37, 64, 200, 400, 800])
def test_width_range_is_feasible_and_bounded(n_keys):
    """Every candidate must be >= n_keys (embedding floor) and <= 2x, aligned."""
    choices = model_width_choices(n_keys, align=8, size_mult=2.0, max_choices=8)

    assert choices, "width list must never be empty"
    assert len(choices) <= 8, "search cost must not grow with the DAG size"
    assert choices == sorted(set(choices)), "choices must be unique and sorted"
    for width in choices:
        assert width >= n_keys, "d_model < n_keys breaks the orthogonal frame"
        assert width % 8 == 0, "widths must stay aligned"
    # Endpoints are pinned so both extremes of the range stay reachable.
    assert choices[0] == 8 * math.ceil(n_keys / 8)
    assert choices[-1] == 8 * math.ceil(2.0 * n_keys / 8)


def test_width_range_examples():
    assert model_width_choices(6) == [8, 16]
    assert model_width_choices(10) == [16, 24]
    # A dense range is returned in full when it fits under max_choices.
    assert model_width_choices(64, max_choices=8)[0] == 64
    assert model_width_choices(64, max_choices=8)[-1] == 128


def test_width_alignment_divisible_by_head_counts():
    """align=8 keeps every width divisible by the searched head counts."""
    for n_keys in (6, 50, 400):
        for width in model_width_choices(n_keys, align=8):
            for n_heads in (1, 2, 4, 8):
                assert width % n_heads == 0


def test_width_range_rejects_nonsense():
    with pytest.raises(ValueError):
        model_width_choices(0)
    with pytest.raises(ValueError):
        model_width_choices(10, size_mult=0.5)
    with pytest.raises(ValueError):
        model_width_choices(10, max_choices=1)


# =============================================================================
# 2. Sampler: dotted parameter names
# =============================================================================

class _RecordingTrial:
    """Minimal Optuna-trial stand-in that records the names it is asked for."""

    def __init__(self):
        self.names = []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        return choices[0]

    def suggest_int(self, name, low, high, step=1):
        self.names.append(name)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.names.append(name)
        return low


def test_sampler_uses_dotted_config_paths():
    """
    Optuna param names must BE the config paths.

    best_trial.yaml stores trial.params verbatim and the sweep applies them with
    OmegaConf.update, so a short name would create a dead top-level key and the
    tuned value would be silently discarded.
    """
    space = {
        "experiment.d_model_set": {"type": "adaptive_width", "align": 8},
        "experiment.n_heads": {"type": "categorical", "choices": [1, 2, 4]},
        "training.lr": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
    }
    trial = _RecordingTrial()
    params = build_sample_params_fn(space, n_keys=10)(trial)

    assert set(trial.names) == set(space), "suggest names must be the dotted paths"
    assert set(params) == set(space)
    assert params["experiment.d_model_set"] == 16      # = ceil(10/8)*8
    assert params["experiment.n_heads"] == 1
    assert params["training.lr"] == pytest.approx(1e-4)

    # And they must actually land in the right place in a config.
    config = OmegaConf.create({"experiment": {"d_model_set": 32, "n_heads": 8},
                               "training": {"lr": 0.1}})
    for dotted, value in params.items():
        OmegaConf.update(config, dotted, value, merge=True)
    assert config.experiment.d_model_set == 16
    assert config.experiment.n_heads == 1
    assert config.training.lr == pytest.approx(1e-4)


def test_sampler_rejects_unknown_type_and_empty_space():
    with pytest.raises(ValueError):
        build_sample_params_fn({"a.b": {"type": "wat"}}, n_keys=8)(_RecordingTrial())
    with pytest.raises(ValueError):
        build_sample_params_fn({}, n_keys=8)
    with pytest.raises(ValueError):
        build_sample_params_fn({"a.b": {"low": 1}}, n_keys=8)  # no 'type'


# =============================================================================
# 3. Reconstruction protocol
# =============================================================================

def _full_config():
    return OmegaConf.create({
        "experiment": {"d_model_set": 32, "n_heads": 4, "init_tau": 0.5,
                       "init_gamma": -1.1, "init_zeta": 1.1,
                       "init_edge_offset": 1.0986122886681098,
                       "query_fanin_scale": 68.69, "batch_size": 2048},
        "training": {"k_fold": 3, "use_gradient_routing": True,
                     "lambda_hsic": 1.0, "lambda_l0": 1e-6, "kappa": 1e-3,
                     "lambda_query_norm": 1e-3, "lr": 5e-4,
                     "early_stopping": {"enabled": False}},
        "adaptive_training": {"structure": {"lambda_l0": 1e-6},
                              "eval_dag": True, "run_final_evaluations": True},
        "evaluation": {"functions": ["eval_attention_scores", "eval_interventions"]},
    })


def test_reconstruction_protocol_zeroes_every_structural_term():
    config = apply_protocol(_full_config(), "reconstruction")

    assert config.training.use_gradient_routing is False
    assert config.training.k_fold == 1
    assert config.training.lambda_recon == 1.0
    assert config.training.early_stopping.enabled is True
    # Every lambda the protocol knows about must be off ...
    for dotted, expected in RECONSTRUCTION_PROTOCOL.items():
        if dotted.startswith("training.lambda") or dotted == "training.kappa":
            assert OmegaConf.select(config, dotted) == expected
    # ... including the adaptive trainer's private copy.
    assert config.adaptive_training.structure.lambda_l0 == 0.0


def test_protocol_extra_overrides_win_and_unknown_name_raises():
    config = apply_protocol(_full_config(), "reconstruction",
                            {"experiment.max_epochs": 20, "training.k_fold": 2})
    assert config.experiment.max_epochs == 20
    assert config.training.k_fold == 2

    with pytest.raises(ValueError):
        apply_protocol(_full_config(), "full_objective")


def test_protocol_none_is_a_noop():
    config = apply_protocol(_full_config(), "none")
    assert config.training.lambda_hsic == 1.0
    assert config.training.use_gradient_routing is True
    # Phase 2 asymmetry: the evaluation suite must survive untouched, because
    # SHD/ATE ARE the benchmark result for the evaluation seeds.
    assert list(config.evaluation.functions) == ["eval_attention_scores",
                                                 "eval_interventions"]
    assert config.adaptive_training.run_final_evaluations is True
    assert config.adaptive_training.eval_dag is True


def test_reconstruction_protocol_disables_every_evaluation_path():
    """
    A trial is scored on val_x_mae alone, so SHD (eval_attention_scores) and the
    Monte-Carlo ATE (eval_interventions) are pure cost - and meaningless, since
    the protocol zeroes every structural term so the DAG never trains.

    All THREE paths must be closed: the shared suite plus the adaptive trainer's
    two private ones.
    """
    config = apply_protocol(_full_config(), "reconstruction")

    # The empty list is load-bearing.  `functions is None` is the "run ALL
    # default evaluations" sentinel in BOTH trainer._run_post_training_evaluations
    # and run_evaluations_from_config, so null here would do the OPPOSITE.
    assert config.evaluation.functions is not None, \
        "None means 'run all evaluations' - the disable value is an empty list"
    assert list(config.evaluation.functions) == []

    # In-fit DAG diagnostics at every phase switch: not covered by the list above.
    assert config.adaptive_training.eval_dag is False
    # Post-training suite: skipped outright, no evaluation import per trial.
    assert config.adaptive_training.run_final_evaluations is False


def test_protocol_does_not_invent_an_adaptive_block():
    """
    The adaptive-only keys are written ONLY where the block already exists, so a
    `standard`-trainer config is not given a spurious `adaptive_training`
    section that a later reader could mistake for an enabled schedule.
    """
    config = OmegaConf.create({
        "experiment": {"d_model_set": 32, "n_heads": 4},
        "training": {"k_fold": 3, "lambda_hsic": 1.0},
    })
    config = apply_protocol(config, "reconstruction")

    assert "adaptive_training" not in config
    # The shared evaluation switch still applies to a standard config.
    assert list(config.evaluation.functions) == []


# =============================================================================
# 4a. Size-derived fields
# =============================================================================

def test_batch_size_decreases_with_dag_size_and_is_pow2():
    budget = 4.9e8
    batches = [activation_batch_size(n, 2 * n, 4, budget=budget, min_batch=1,
                                    max_batch=4096)
               for n in (10, 50, 100, 200, 400, 800)]
    assert batches == sorted(batches, reverse=True), "bigger DAG -> smaller batch"
    for batch in batches:
        assert batch & (batch - 1) == 0, "batch size must be a power of two"


def test_batch_size_respects_clamps():
    assert activation_batch_size(4, 8, 1, budget=1e12, max_batch=256) == 256
    assert activation_batch_size(4000, 8000, 8, budget=1.0, min_batch=16) == 16


def test_size_derived_batch_is_independent_of_the_sampled_width():
    """
    The batch rule must depend on the DAG only.

    Otherwise the tuned learning rate would belong to a batch size that the
    evaluation run never uses.
    """
    values = []
    for d_model in (8, 16):
        config = _full_config()
        config.experiment.d_model_set = d_model
        derive_size_fields(config, 6, {
            "experiment.batch_size": {"rule": "activation_budget", "C": 4.9e8,
                                      "d_ref_mult": 2.0, "n_heads": 4,
                                      "max": 256},
        })
        values.append(config.experiment.batch_size)
    assert values[0] == values[1]


def test_fanin_rule_matches_the_closed_form():
    config = _full_config()
    derive_size_fields(config, 10, {
        "experiment.query_fanin_scale": {"rule": "fanin_saturating"},
    })
    # T-free: x_sat = kappa_1 = 0.5 * ln(2.1/0.1) = 1.5223 -> F = 10 * x_sat^2 = 23.17
    # (init_edge_offset no longer enters F; it is the cross-gate init-balance
    # device resolved separately by resolve_init_edge_offset.)
    assert config.experiment.query_fanin_scale == pytest.approx(23.17, abs=0.01)

    # And it scales linearly with the node count.
    assert saturating_query_fanin(config, 400) == pytest.approx(
        40 * saturating_query_fanin(config, 10)
    )


def test_size_derived_rejects_unknown_rule():
    with pytest.raises(ValueError):
        derive_size_fields(_full_config(), 10, {"experiment.batch_size": "magic"})


def test_activation_budget_helpers():
    assert estimate_budget(24 * 1024 ** 3, dtype_bytes=4, multiplicity=12,
                           safety=0.35) > 0
    assert resolve_budget(1234.0) == 1234.0        # declared value wins
    assert resolve_budget("auto") > 0              # cache or default


# =============================================================================
# 4b. Dimension repair
# =============================================================================

def test_repair_lifts_width_to_the_embedding_floor():
    config = _full_config()          # d_model_set = 32
    repairs = validate_dimensions(config, n_keys=50)
    assert repairs["experiment.d_model_set"] == 56      # ceil(50/8)*8
    assert config.experiment.d_model_set == 56


def test_repair_fixes_head_divisibility_upwards():
    config = _full_config()
    config.experiment.d_model_set = 30
    config.experiment.n_heads = 4
    repairs = validate_dimensions(config, n_keys=10)
    assert repairs["experiment.d_model_set"] == 32
    assert config.experiment.d_model_set % config.experiment.n_heads == 0


def test_repair_sets_d_qk_when_projections_are_removed():
    config = _full_config()
    config.experiment.remove_query_projection = True
    config.experiment.remove_key_projection = True
    config.experiment.shared_dag_across_heads = True
    config.experiment.d_qk = 8                     # inconsistent with d_model=32
    repairs = validate_dimensions(config, n_keys=10)
    assert repairs["experiment.d_qk"] == 32
    assert config.experiment.d_qk == 32


def test_repair_updates_a_stale_fanin_scale():
    config = _full_config()                        # 68.69, a stale pinned value
    validate_dimensions(config, n_keys=6)
    # n * kappa_1^2 = 6 * 1.5223^2 = 13.90
    assert config.experiment.query_fanin_scale == pytest.approx(13.90, abs=0.05)


def test_consistent_config_is_left_untouched():
    config = _full_config()
    config.experiment.d_model_set = 16
    config.experiment.n_heads = 4
    config.experiment.query_fanin_scale = saturating_query_fanin(config, 10)
    assert validate_dimensions(config, n_keys=10) == {}


def test_repair_false_raises_instead():
    config = _full_config()
    with pytest.raises(ValueError, match="n_keys"):
        validate_dimensions(config, n_keys=50, repair=False)


# =============================================================================
# 4c. Best-trial selection
# =============================================================================

def _study_with(points, direction="minimize"):
    """
    Build an in-memory study from ``[(value, params, user_attrs), ...]``.
    """
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    for value, params, attrs in points:
        distributions = {
            name: optuna.distributions.CategoricalDistribution([val])
            for name, val in params.items()
        }
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions, value=value,
            user_attrs=attrs,
        ))
    return study


def test_parsimonious_selection_takes_the_knee():
    """
    Reconstruction error saturates with capacity: the knee, not the argmin.

    Trial 1 (16 wide) is only 1% worse than the best (32 wide) but half the size,
    so with tol=0.02 it must win - otherwise the adaptive range would always
    collapse onto its upper end.
    """
    study = _study_with([
        (0.50, {"experiment.d_model_set": 8}, {"trainable_params_mean": 1_000}),
        (0.101, {"experiment.d_model_set": 16}, {"trainable_params_mean": 2_000}),
        (0.100, {"experiment.d_model_set": 32}, {"trainable_params_mean": 8_000}),
    ])
    chosen = select_best(study, {"mode": "parsimonious", "tol": 0.02})

    assert chosen["params"]["experiment.d_model_set"] == 16
    assert chosen["capacity"] == 2_000
    assert chosen["raw_best"]["params"]["experiment.d_model_set"] == 32
    assert chosen["n_within_tol"] == 2
    # The whole capacity/metric curve is recorded for the scaling plot.
    assert [p["capacity"] for p in chosen["curve"]] == [1_000, 2_000, 8_000]


def test_tight_tolerance_falls_back_to_the_best_metric():
    study = _study_with([
        (0.101, {"experiment.d_model_set": 16}, {"trainable_params_mean": 2_000}),
        (0.100, {"experiment.d_model_set": 32}, {"trainable_params_mean": 8_000}),
    ])
    chosen = select_best(study, {"mode": "parsimonious", "tol": 0.0})
    assert chosen["params"]["experiment.d_model_set"] == 32


def test_argmin_mode_and_capacity_fallback():
    study = _study_with([
        (0.101, {"experiment.d_model_set": 16, "experiment.n_heads": 1}, {}),
        (0.100, {"experiment.d_model_set": 32, "experiment.n_heads": 4}, {}),
    ])
    assert select_best(study, {"mode": "argmin"})["params"][
        "experiment.d_model_set"] == 32

    # No parameter count reported -> fall back to the product of the capacity
    # params (16*1 = 16 < 32*4 = 128), so the smaller model still wins.
    chosen = select_best(study, {
        "mode": "parsimonious", "tol": 0.02,
        "capacity_params": ["experiment.d_model_set", "experiment.n_heads"],
    })
    assert chosen["params"]["experiment.d_model_set"] == 16


def test_pruned_and_failed_trials_are_ignored():
    optuna = pytest.importorskip("optuna")
    study = _study_with([
        (0.20, {"experiment.d_model_set": 16}, {"trainable_params_mean": 2_000}),
    ])
    study.add_trial(optuna.trial.create_trial(
        params={"experiment.d_model_set": 8},
        distributions={"experiment.d_model_set":
                       optuna.distributions.CategoricalDistribution([8])},
        state=optuna.trial.TrialState.PRUNED,
    ))
    chosen = select_best(study, {"mode": "parsimonious", "tol": 0.5})
    assert chosen["params"]["experiment.d_model_set"] == 16
    assert len(chosen["curve"]) == 1


def test_maximised_metric_uses_the_other_side_of_the_band():
    study = _study_with([
        (0.90, {"experiment.d_model_set": 16}, {"trainable_params_mean": 2_000}),
        (0.91, {"experiment.d_model_set": 32}, {"trainable_params_mean": 8_000}),
    ], direction="maximize")
    chosen = select_best(study, {"mode": "parsimonious", "tol": 0.02},
                         direction="maximize")
    assert chosen["params"]["experiment.d_model_set"] == 16


def test_unknown_selection_mode_raises():
    study = _study_with([(0.1, {"experiment.d_model_set": 8}, {})])
    with pytest.raises(ValueError):
        select_best(study, {"mode": "lucky_dip"})
