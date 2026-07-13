"""Tests for the reconstruct-phase minimum-epoch floor in ``PhaseController``.

Run with:  pytest tests/test_adaptive_min_epochs.py -v

Background
----------
``PhaseController`` (causaliT.training.adaptive_trainer) alternates between a
``reconstruct`` and a ``structure`` phase.  Historically the reconstruct phase
could exit as soon as the monitored validation metric *plateaued*
(``plateau_counter >= plateau_patience``), which — at the very first warmup —
could switch to structure learning after only a handful of epochs, before the
predictor was fully reconstructed.

Two floors now suppress the plateau-based early exit until the phase has run a
minimum number of epochs:

    * ``warmup_min_epochs`` — applies ONLY to the initial warmup reconstruct
      phase (``phase_index == 0`` and ``start_phase == "reconstruct"``).
    * ``min_epochs``        — applies to every later reconstruct phase.
      ``warmup_min_epochs`` falls back to ``min_epochs`` when unset.

The ``max_epochs`` safety cap always takes precedence over the floor.

These tests drive the controller's ``on_validation_epoch_end`` state machine
directly with lightweight fakes so no model / pl.Trainer is required.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.adaptive_trainer import PhaseController


@pytest.fixture
def tmp_path():
    """Workspace-local temp dir.

    The default pytest ``tmp_path`` fixture points at the system temp root,
    which is not readable in this environment (WinError 5).  Create the dir
    under the project instead and clean it up afterwards.
    """
    base = project_root / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)



# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------

class _FakeTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False

    def save_checkpoint(self, path):  # pragma: no cover - never hit (stubbed)
        pass


class _FakeModule:
    def __init__(self):
        self.training = True

    def log(self, *args, **kwargs):
        pass

    def train(self):
        self.training = True


def _make_controller(tmp_path, monitor="val_x_mae", **recon_overrides):
    """Build a PhaseController with stubbed phase-switching side effects."""
    recon = {
        "max_epochs": 100,
        "min_epochs": 0,
        "warmup_min_epochs": 0,
        "plateau_patience": 2,
        "plateau_min_delta": 1e-4,
    }
    recon.update(recon_overrides)
    config = {
        "adaptive_training": {
            "monitor": monitor,
            "start_phase": "reconstruct",
            "max_cycles": 100,
            "eval_dag": False,
            "reconstruct": recon,
            "structure": {"max_epochs": 200, "drop_pct": 0.2, "drop_patience": 5},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    controller = PhaseController(
        config=config,
        data_dir=str(tmp_path),
        save_dir=str(tmp_path),
        cluster=True,
    )

    # Record transitions instead of touching disk / DAG diagnostics.
    events = []

    def _fake_record(trainer, pl_module, reason, from_phase, to_phase, monitor_val):
        events.append({
            "reason": reason,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "epoch": trainer.current_epoch,
            "phase_epochs": trainer.current_epoch - controller._phase_start_epoch + 1,
        })

    # Mimic the real _apply_phase's state reset without needing a model.
    def _fake_apply(trainer, pl_module, phase):
        controller.current_phase = phase
        controller._phase_start_epoch = trainer.current_epoch
        controller._phase_best = float("inf")
        controller._plateau_counter = 0
        controller._drop_counter = 0

    controller._record_transition = _fake_record
    controller._apply_phase = _fake_apply
    return controller, events


def _run_plateau(controller, events, n_epochs, monitor="val_x_mae",
                 value=1.0, start_epoch=0):
    """Feed a constant (plateauing) monitor value across ``n_epochs`` epochs.

    Returns the epoch of the first recon->structure transition, or None.
    """
    trainer = _FakeTrainer()
    module = _FakeModule()
    for i in range(n_epochs):
        trainer.current_epoch = start_epoch + i
        trainer.callback_metrics = {monitor: value}
        controller.on_validation_epoch_end(trainer, module)
        recon_events = [e for e in events if e["from_phase"] == "reconstruct"]
        if recon_events:
            return recon_events[0]["epoch"]
    return None


# ---------------------------------------------------------------------------
# 1. Default (no floor): plateau exits as soon as patience is hit
# ---------------------------------------------------------------------------

def test_no_floor_exits_on_plateau(tmp_path):
    controller, events = _make_controller(
        tmp_path, min_epochs=0, warmup_min_epochs=0, plateau_patience=2
    )
    # Constant value -> improvement only on epoch 0, then counter increments.
    # counter reaches patience(2) at the 3rd validation epoch (phase_epochs=3).
    exit_epoch = _run_plateau(controller, events, n_epochs=10)
    assert exit_epoch is not None
    assert events[0]["reason"] == "recon_plateau"
    assert events[0]["phase_epochs"] == 3


# ---------------------------------------------------------------------------
# 2. Warmup floor: the INITIAL reconstruct phase is held to warmup_min_epochs
# ---------------------------------------------------------------------------

def test_warmup_floor_suppresses_early_plateau(tmp_path):
    controller, events = _make_controller(
        tmp_path, min_epochs=1, warmup_min_epochs=5, plateau_patience=2
    )
    exit_epoch = _run_plateau(controller, events, n_epochs=20)
    assert exit_epoch is not None
    assert events[0]["reason"] == "recon_plateau"
    # Plateau would have fired at phase_epochs=3, but the warmup floor of 5
    # holds the phase until phase_epochs >= 5.
    assert events[0]["phase_epochs"] == 5


# ---------------------------------------------------------------------------
# 3. Later-phase floor uses min_epochs (not warmup_min_epochs)
# ---------------------------------------------------------------------------

def test_later_phase_uses_min_epochs(tmp_path):
    controller, events = _make_controller(
        tmp_path, min_epochs=4, warmup_min_epochs=10, plateau_patience=2
    )
    # Simulate a *non-initial* reconstruct phase.
    controller._phase_index = 2
    controller.current_phase = "reconstruct"
    controller._phase_start_epoch = 0

    exit_epoch = _run_plateau(controller, events, n_epochs=20)
    assert exit_epoch is not None
    # min_epochs=4 governs (NOT warmup_min_epochs=10) since this is not warmup.
    assert events[0]["phase_epochs"] == 4


# ---------------------------------------------------------------------------
# 4. max_epochs safety cap always wins over the floor
# ---------------------------------------------------------------------------

def test_budget_cap_precedes_floor(tmp_path):
    # A huge floor would forbid a plateau exit forever, but the max_epochs cap
    # must still force a switch.
    controller, events = _make_controller(
        tmp_path, min_epochs=1000, warmup_min_epochs=1000,
        max_epochs=3, plateau_patience=2,
    )
    exit_epoch = _run_plateau(controller, events, n_epochs=20)
    assert exit_epoch is not None
    assert events[0]["reason"] == "recon_budget"
    assert events[0]["phase_epochs"] == 3


# ---------------------------------------------------------------------------
# 5. Config parsing: warmup_min_epochs falls back to min_epochs when unset
# ---------------------------------------------------------------------------

def test_warmup_falls_back_to_min_epochs(tmp_path):
    controller, _ = _make_controller(tmp_path)
    # Rebuild with only min_epochs specified (drop warmup key entirely).
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "reconstruct",
            "reconstruct": {"min_epochs": 7, "plateau_patience": 2},
            "structure": {},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    ctrl = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True,
    )
    assert ctrl.recon_min_epochs == 7
    assert ctrl.recon_warmup_min_epochs == 7


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
