"""Tests for the adaptive trainer's evaluation-suite compatibility artefacts.

Run with:  pytest tests/test_adaptive_final_eval.py -v

Background
----------
The default evaluation suite (``causaliT/evaluation/eval_funs/eval_funs_wraps.py``)
assumes the standard experiment layout produced by ``trainer()``:

    <experiment>/
        config*.yaml            <- located by eval_attention / eval_interventions
        k_0/checkpoints/...     <- written by train_single_fold
        kfold_summary.json      <- fixed/enriched by the suite

``adaptive_trainer`` previously left the root-level ``kfold_summary.json`` out
(it never instantiated ``KFoldResultsTracker``) and never triggered evaluation at
all.  It now:

    * writes ``kfold_summary.json`` for its single fold (``_write_kfold_summary``),
    * writes a RESOLVED ``config.yaml`` snapshot only when no ``config*.yaml`` is
      already present (``_save_config_snapshot`` — never a second, ambiguous
      candidate for the suite's "first glob hit" lookup),
    * runs the standard dispatcher at the end, gated by
      ``adaptive_training.run_final_evaluations`` (default True).

These tests exercise the helpers directly (no model / pl.Trainer needed) plus a
wiring guard on the orchestrator.
"""

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training import adaptive_trainer as at_mod
from causaliT.training.adaptive_trainer import (
    _save_config_snapshot,
    _write_kfold_summary,
    adaptive_trainer,
)


@pytest.fixture
def tmp_dir():
    """Workspace-local temp dir (the system temp root is not writable here)."""
    base = project_root / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# kfold_summary.json
# ---------------------------------------------------------------------------

def test_write_kfold_summary_creates_expected_file(tmp_dir):
    """The single adaptive fold lands in kfold_summary.json like trainer() does."""
    fold_metrics = {
        "val_x_mae": 0.25,
        "val_hsic": 0.01,
        "test_x_mae": 0.30,
        "_best_checkpoint_path": str(tmp_dir / "k_0" / "checkpoints" / "best.ckpt"),
    }

    _write_kfold_summary(str(tmp_dir), fold_metrics)

    summary_path = tmp_dir / "kfold_summary.json"
    assert summary_path.exists(), "kfold_summary.json was not created"

    summary = json.loads(summary_path.read_text())
    fold_results = summary["fold_results"]
    assert set(fold_results.keys()) == {"0"}, "expected exactly one fold entry"

    metrics = fold_results["0"]["metrics"]
    assert metrics["val_x_mae"] == pytest.approx(0.25)
    assert metrics["val_hsic"] == pytest.approx(0.01)
    # The private key is promoted to its own field, not kept among the metrics.
    assert "_best_checkpoint_path" not in metrics
    assert fold_results["0"]["best_checkpoint_path"].endswith("best.ckpt")


def test_write_kfold_summary_does_not_mutate_caller_metrics(tmp_dir):
    """The caller's dict still feeds adaptive_training_summary.json unchanged."""
    fold_metrics = {"val_x_mae": 0.5, "_best_checkpoint_path": "some/ckpt.ckpt"}

    _write_kfold_summary(str(tmp_dir), fold_metrics)

    assert "_best_checkpoint_path" in fold_metrics, "caller dict was mutated"


# ---------------------------------------------------------------------------
# config snapshot
# ---------------------------------------------------------------------------

def test_save_config_snapshot_writes_resolved_config_when_absent(tmp_dir):
    """With no config in the folder, a resolved config.yaml is written."""
    config = OmegaConf.create({
        "experiment": {"d_model": 16, "d_ff": "${experiment.d_model}"},
        "training": {"k_fold": 1, "max_epochs": 800},
    })

    written = _save_config_snapshot(config, str(tmp_dir))

    assert written is not None
    snapshot = OmegaConf.load(tmp_dir / "config.yaml")
    # ``resolve=True`` means interpolations are materialised, so an offline
    # evaluation never has to resolve them against a missing context.
    assert snapshot["experiment"]["d_ff"] == 16
    assert snapshot["training"]["k_fold"] == 1


def test_save_config_snapshot_is_noop_when_config_present(tmp_dir):
    """An existing config*.yaml must not gain a second, ambiguous sibling."""
    existing = tmp_dir / "config_my_experiment.yaml"
    existing.write_text("training:\n  k_fold: 5\n")

    written = _save_config_snapshot({"training": {"k_fold": 1}}, str(tmp_dir))

    assert written is None
    assert not (tmp_dir / "config.yaml").exists()
    # The user's file is untouched.
    assert "k_fold: 5" in existing.read_text()


# ---------------------------------------------------------------------------
# orchestrator wiring
# ---------------------------------------------------------------------------

def test_adaptive_trainer_wires_eval_artefacts_and_dispatcher():
    """Guard the orchestrator wiring (a full fit is too heavy for a unit test)."""
    src = inspect.getsource(adaptive_trainer)

    assert "_write_kfold_summary(save_dir, fold_metrics)" in src
    assert "_save_config_snapshot(config, save_dir)" in src
    # Evaluation runs through the SAME dispatcher as trainer(), gated by the
    # opt-out flag which defaults to True.
    assert 'ad_cfg.get("run_final_evaluations", True)' in src
    assert "_run_post_training_evaluations(config, str(save_dir), data_dir)" in src


def test_run_post_training_evaluations_is_importable():
    """The dispatcher reused from trainer() must exist with that exact name."""
    from causaliT.training.trainer import _run_post_training_evaluations

    assert callable(_run_post_training_evaluations)
    # KFoldResultsTracker must be importable at adaptive_trainer import time
    # (no circular-import regression).
    assert at_mod.KFoldResultsTracker is not None
