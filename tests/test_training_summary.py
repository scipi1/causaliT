"""
Tests for the unified training summary.

Covers the properties the runtime metric depends on:
  - a fit is the same object for models (folds) and benchmarks (seeds),
  - runtime is aggregated per method, over the repetition axis,
  - missing quantities stay missing (never silently zero),
  - censoring flags (max_epochs / stopped_early) survive the round trip,
  - legacy kfold_summary.json files remain readable,
  - the sweep walker still recognises runs in EITHER format.
"""

import json

import pytest

from causaliT.training.training_summary import (
    LEGACY_SUMMARY_FILE,
    TRAINING_SUMMARY_FILE,
    TrainingSummaryWriter,
    get_statistic,
    load_training_summary,
)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_model_and_benchmark_produce_the_same_file(tmp_path):
    """A fold and a seed are both 'fits' - that is what makes them comparable."""
    model_dir = tmp_path / "model_run"
    bench_dir = tmp_path / "bench_run"

    m = TrainingSummaryWriter(str(model_dir), kind="model", method="atsel")
    m.add_fit("k_0", seconds=120.0, metrics={"val_x_mae": 0.10})
    m.finalize()

    b = TrainingSummaryWriter(str(bench_dir), kind="benchmark", method="notears_linear")
    b.add_fit("seed_0", seconds=3.0)
    b.finalize()

    for run_dir in (model_dir, bench_dir):
        summary = load_training_summary(str(run_dir))
        assert (run_dir / TRAINING_SUMMARY_FILE).exists()
        assert summary["schema_version"] == 1
        # Same keys, regardless of which path produced the run.
        assert {"run", "environment", "fits", "statistics"} <= set(summary)
        assert summary["n_fits"] == 1


def test_seconds_are_averaged_over_the_repetition_axis(tmp_path):
    """Runtime is reported as mean/std over fits, exactly like the DAG metrics."""
    w = TrainingSummaryWriter(str(tmp_path), kind="benchmark", method="dagma_linear")
    for i, secs in enumerate([10.0, 20.0, 30.0]):
        w.add_fit(f"seed_{i}", seconds=secs)
    w.finalize()

    summary = load_training_summary(str(tmp_path))
    assert get_statistic(summary, "seconds", "mean") == pytest.approx(20.0)
    assert get_statistic(summary, "seconds", "n") == 3


def test_statistics_are_grouped_per_method(tmp_path):
    """One benchmark run may fit several methods; they must not be averaged together."""
    w = TrainingSummaryWriter(str(tmp_path), kind="benchmark", method="mixed")
    w.add_fit("seed_0", method="pc", seconds=1.0)
    w.add_fit("seed_0", method="notears_linear", seconds=100.0)
    w.finalize()

    stats = load_training_summary(str(tmp_path))["statistics"]
    assert stats["pc"]["seconds"]["mean"] == pytest.approx(1.0)
    assert stats["notears_linear"]["seconds"]["mean"] == pytest.approx(100.0)


def test_missing_fields_are_absent_not_zero(tmp_path):
    """A benchmark has no epochs; reporting 0 would be a lie, absence is honest."""
    w = TrainingSummaryWriter(str(tmp_path), kind="benchmark", method="pc")
    w.add_fit("seed_0", seconds=2.5)
    w.finalize()

    fit = load_training_summary(str(tmp_path))["fits"][0]
    assert "epochs_run" not in fit
    assert "trainable_params" not in fit
    assert get_statistic(load_training_summary(str(tmp_path)), "epochs_run") is None


def test_censoring_flags_round_trip(tmp_path):
    """
    Without max_epochs/stopped_early, a run that hit its budget is
    indistinguishable from one that converged - the runtime would then report
    the budget rather than the method.
    """
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", seconds=100.0, epochs_run=200, max_epochs=200, stopped_early=False)
    w.finalize()

    fit = load_training_summary(str(tmp_path))["fits"][0]
    assert fit["epochs_run"] == 200
    assert fit["max_epochs"] == 200
    assert fit["stopped_early"] is False
    # Budget exhausted: the number is censored and must be readable as such.
    assert fit["epochs_run"] == fit["max_epochs"]


def test_avg_time_per_epoch_is_derived(tmp_path):
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    fit = w.add_fit("k_0", seconds=100.0, epochs_run=50)
    assert fit["avg_time_per_epoch"] == pytest.approx(2.0)


def test_unknown_runtime_field_is_rejected(tmp_path):
    """A typo must fail loudly, not create a column that is quietly always empty."""
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    with pytest.raises(TypeError):
        w.add_fit("k_0", second=1.0)  # typo for "seconds"


def test_tensor_metrics_are_serialised_as_numbers(tmp_path):
    """The reason fix_kfold_summary existed: tensors leaking in as strings."""
    torch = pytest.importorskip("torch")

    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", seconds=1.0, metrics={"val_x_mae": torch.tensor(0.0005)})
    w.finalize()

    raw = (tmp_path / TRAINING_SUMMARY_FILE).read_text()
    assert "tensor(" not in raw
    value = json.loads(raw)["fits"][0]["metrics"]["val_x_mae"]
    assert isinstance(value, float)


def test_runtime_is_not_duplicated_into_metrics(tmp_path):
    """Exactly one place to read each quantity from."""
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", metrics={"seconds": 42.0, "val_x_mae": 0.1})
    w.finalize()

    fit = load_training_summary(str(tmp_path))["fits"][0]
    assert fit["seconds"] == pytest.approx(42.0)
    assert "seconds" not in fit["metrics"]


def test_best_fit_prefers_hsic_over_mae(tmp_path):
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", metrics={"val_hsic_reg": 0.9, "val_x_mae": 0.01})
    w.add_fit("k_1", metrics={"val_hsic_reg": 0.1, "val_x_mae": 0.99})
    w.finalize()

    best = load_training_summary(str(tmp_path))["best_fit"]
    assert best["selection_criterion"] == "val_hsic_reg"
    assert best["id"] == "k_1"


def test_partial_run_is_still_readable(tmp_path):
    """The file is rewritten per fit, so an interrupted run is not lost."""
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", seconds=5.0)
    # No finalize() - simulate a crash mid-run.
    summary = load_training_summary(str(tmp_path))
    assert summary["n_fits"] == 1


# ---------------------------------------------------------------------------
# Reading legacy files
# ---------------------------------------------------------------------------

def test_legacy_kfold_summary_is_readable(tmp_path):
    """Historical experiments must not become unreadable."""
    legacy = {
        "total_folds": 2,
        "completed_folds": 2,
        "fold_results": {
            "0": {
                "metrics": {"val_x_mae": 0.2, "total_training_time": 100.0},
                "best_checkpoint_path": "k_0/checkpoints/best.ckpt",
            },
            "1": {
                "metrics": {"val_x_mae": 0.3, "total_training_time": 200.0},
                "best_checkpoint_path": None,
            },
        },
        "statistics": {"val_x_mae": {"mean": 0.25, "std": 0.05}},
        "best_fold": {"fold_number": 0, "selection_criterion": "val_x_mae",
                      "selection_value": 0.2, "metrics": {},
                      "checkpoint_path": "k_0/checkpoints/best.ckpt"},
    }
    (tmp_path / LEGACY_SUMMARY_FILE).write_text(json.dumps(legacy))

    summary = load_training_summary(str(tmp_path))
    assert summary["legacy"] is True
    assert summary["n_fits"] == 2
    # The old format buried the timing among the metrics; it is promoted.
    assert summary["fits"][0]["seconds"] == pytest.approx(100.0)
    assert summary["best_fit"]["id"] == "k_0"


def test_legacy_run_does_not_invent_missing_fields(tmp_path):
    """An old run should look less informative, not falsely complete."""
    legacy = {"fold_results": {"0": {"metrics": {"val_x_mae": 0.2}}}}
    (tmp_path / LEGACY_SUMMARY_FILE).write_text(json.dumps(legacy))

    summary = load_training_summary(str(tmp_path))
    assert summary["environment"] == {}
    assert "seconds" not in summary["fits"][0]


def test_new_file_wins_over_legacy(tmp_path):
    (tmp_path / LEGACY_SUMMARY_FILE).write_text(json.dumps({"fold_results": {}}))
    w = TrainingSummaryWriter(str(tmp_path), kind="model", method="atsel")
    w.add_fit("k_0", seconds=1.0)
    w.finalize()

    assert load_training_summary(str(tmp_path))["schema_version"] == 1


def test_missing_summary_returns_none(tmp_path):
    """'Not a run' must be distinguishable from 'a run with no metrics'."""
    assert load_training_summary(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Sweep traversal
# ---------------------------------------------------------------------------

def test_is_trained_run_accepts_both_formats(tmp_path):
    """
    The walker recurses while this is False, so missing a marker yields an
    empty DataFrame and no error - the failure would be silent.
    """
    from causaliT.evaluation.eval_sweeps import is_trained_run

    empty = tmp_path / "not_a_run"
    empty.mkdir()
    assert is_trained_run(str(empty)) is False

    new_run = tmp_path / "new_run"
    new_run.mkdir()
    (new_run / TRAINING_SUMMARY_FILE).write_text("{}")
    assert is_trained_run(str(new_run)) is True

    old_run = tmp_path / "old_run"
    old_run.mkdir()
    (old_run / LEGACY_SUMMARY_FILE).write_text("{}")
    assert is_trained_run(str(old_run)) is True
