"""Tests for the structure-phase HSIC-plateau early switch in ``PhaseController``.

Run with:  pytest tests/test_adaptive_hsic_patience.py -v

Background
----------
``PhaseController`` (causaliT.training.adaptive_trainer) alternates between a
``reconstruct`` and a ``structure`` phase.  Historically the structure phase
could only end on (a) ``val_x_mae`` rising ``drop_pct`` over the phase-best for
``drop_patience`` epochs (the frozen predictor going stale) or (b) the
``max_epochs`` safety cap.  Neither watches whether the structural objective
(HSIC) is still improving, so a structure phase could burn its whole budget
while HSIC had already plateaued — wasting compute and delaying the next
reconstruction update.

A new early-exit watches the structural signal (``hsic_monitor``, default
``val_hsic``, lower is better).  When it fails to improve (relative to the
per-phase best by more than ``hsic_min_delta``) for ``hsic_patience``
consecutive validation epochs, the controller switches back to reconstruct with
reason ``struct_hsic_plateau``.  ``hsic_patience == 0`` disables the feature
(backward-compatible).  A ``min_epochs`` floor suppresses the early exit for the
first N epochs, and the ``max_epochs`` cap still takes precedence.

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
    """Workspace-local temp dir (system temp root is not readable here)."""
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


def _make_controller(tmp_path, **struct_overrides):
    """Build a PhaseController (started in structure phase) with stubbed effects."""
    struct = {
        "max_epochs": 200,
        "drop_pct": 0.20,
        "drop_patience": 5,
        "hsic_monitor": "val_hsic",
        "hsic_patience": 2,
        "hsic_min_delta": 1e-4,
        "min_epochs": 0,
    }
    struct.update(struct_overrides)
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "structure",
            "max_cycles": 100,
            "eval_dag": False,
            "reconstruct": {"max_epochs": 100, "plateau_patience": 5},
            "structure": struct,
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    controller = PhaseController(
        config=config,
        data_dir=str(tmp_path),
        save_dir=str(tmp_path),
        cluster=True,
    )

    events = []

    def _fake_record(trainer, pl_module, reason, from_phase, to_phase, monitor_val):
        events.append({
            "reason": reason,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "epoch": trainer.current_epoch,
            "phase_epochs": trainer.current_epoch - controller._phase_start_epoch + 1,
        })

    def _fake_apply(trainer, pl_module, phase):
        controller.current_phase = phase
        controller._phase_start_epoch = trainer.current_epoch
        controller._phase_best = float("inf")
        controller._plateau_counter = 0
        controller._drop_counter = 0
        controller._hsic_best = float("inf")
        controller._hsic_plateau_counter = 0

    controller._record_transition = _fake_record
    controller._apply_phase = _fake_apply

    # Start the controller in the structure phase.
    controller.current_phase = "structure"
    controller._phase_start_epoch = 0
    return controller, events


def _run(controller, events, n_epochs, hsic_values, x_mae=1.0, start_epoch=0):
    """Feed (val_x_mae constant, val_hsic from ``hsic_values``) over epochs.

    ``hsic_values`` may be a scalar (constant) or a per-epoch sequence.
    Returns the first structure->reconstruct transition event, or None.
    """
    trainer = _FakeTrainer()
    module = _FakeModule()
    for i in range(n_epochs):
        trainer.current_epoch = start_epoch + i
        hsic = hsic_values[i] if isinstance(hsic_values, (list, tuple)) else hsic_values
        trainer.callback_metrics = {"val_x_mae": x_mae, "val_hsic": hsic}
        controller.on_validation_epoch_end(trainer, module)
        struct_events = [e for e in events if e["from_phase"] == "structure"]
        if struct_events:
            return struct_events[0]
    return None


# ---------------------------------------------------------------------------
# 1. HSIC plateau fires after `hsic_patience` non-improving epochs
# ---------------------------------------------------------------------------

def test_hsic_plateau_triggers_switch(tmp_path):
    controller, events = _make_controller(tmp_path, hsic_patience=2)
    # Constant HSIC -> improvement only on epoch 0, counter reaches patience(2)
    # at the 3rd validation epoch (phase_epochs=3).
    event = _run(controller, events, n_epochs=10, hsic_values=0.5)
    assert event is not None
    assert event["reason"] == "struct_hsic_plateau"
    assert event["to_phase"] == "reconstruct"
    assert event["phase_epochs"] == 3


# ---------------------------------------------------------------------------
# 2. hsic_patience == 0 disables the feature (backward-compatible)
# ---------------------------------------------------------------------------

def test_hsic_patience_zero_disables(tmp_path):
    controller, events = _make_controller(
        tmp_path, hsic_patience=0, max_epochs=1000
    )
    event = _run(controller, events, n_epochs=20, hsic_values=0.5)
    # No drop (x_mae constant), no HSIC exit, budget not hit -> no transition.
    assert event is None


# ---------------------------------------------------------------------------
# 3. min_epochs floor suppresses an early HSIC-plateau exit
# ---------------------------------------------------------------------------

def test_min_epochs_floor_suppresses_early_exit(tmp_path):
    controller, events = _make_controller(
        tmp_path, hsic_patience=2, min_epochs=6
    )
    # Plateau counter would fire at phase_epochs=3, but the floor holds until 6.
    event = _run(controller, events, n_epochs=20, hsic_values=0.5)
    assert event is not None
    assert event["reason"] == "struct_hsic_plateau"
    assert event["phase_epochs"] == 6


# ---------------------------------------------------------------------------
# 4. A still-improving HSIC does not trigger a switch
# ---------------------------------------------------------------------------

def test_improving_hsic_does_not_switch(tmp_path):
    controller, events = _make_controller(
        tmp_path, hsic_patience=2, max_epochs=1000
    )
    # Monotonically decreasing HSIC -> counter resets every epoch.
    decreasing = [0.5 - 0.01 * i for i in range(20)]
    event = _run(controller, events, n_epochs=20, hsic_values=decreasing)
    assert event is None
    assert controller._hsic_plateau_counter == 0


# ---------------------------------------------------------------------------
# 5. max_epochs cap still wins over an unreachable min_epochs floor
# ---------------------------------------------------------------------------

def test_budget_cap_precedes_floor(tmp_path):
    controller, events = _make_controller(
        tmp_path, hsic_patience=2, min_epochs=1000, max_epochs=4
    )
    event = _run(controller, events, n_epochs=20, hsic_values=0.5)
    assert event is not None
    assert event["reason"] == "struct_budget"
    assert event["phase_epochs"] == 4


# ---------------------------------------------------------------------------
# 6. min_epochs floor ALSO suppresses the struct_drop early-exit
# ---------------------------------------------------------------------------

def _run_drop(controller, events, n_epochs, x_mae_values, start_epoch=0):
    """Feed a rising val_x_mae (constant val_hsic) to exercise the drop trigger."""
    trainer = _FakeTrainer()
    module = _FakeModule()
    for i in range(n_epochs):
        trainer.current_epoch = start_epoch + i
        x_mae = (x_mae_values[i] if isinstance(x_mae_values, (list, tuple))
                 else x_mae_values)
        trainer.callback_metrics = {"val_x_mae": x_mae, "val_hsic": 0.5}
        controller.on_validation_epoch_end(trainer, module)
        struct_events = [e for e in events if e["from_phase"] == "structure"]
        if struct_events:
            return struct_events[0]
    return None


def test_min_epochs_floor_suppresses_struct_drop(tmp_path):
    controller, events = _make_controller(
        tmp_path, hsic_patience=0, drop_pct=0.20, drop_patience=2, min_epochs=8
    )
    # Epoch 0 sets phase_best=1.0; from epoch 1 val_x_mae jumps to 2.0 (>1.2
    # threshold), so the drop counter reaches drop_patience(2) at phase_epochs=3.
    # Without the floor the switch fires at epoch 3; the floor holds it to 8.
    x_mae = [1.0] + [2.0] * 19
    event = _run_drop(controller, events, n_epochs=20, x_mae_values=x_mae)
    assert event is not None
    assert event["reason"] == "struct_drop"
    assert event["phase_epochs"] == 8


# ---------------------------------------------------------------------------
# 7. Config parsing defaults
# ---------------------------------------------------------------------------


def test_config_defaults(tmp_path):
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "reconstruct",
            "reconstruct": {},
            "structure": {},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    ctrl = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True,
    )
    assert ctrl.struct_hsic_monitor == "val_hsic"
    assert ctrl.struct_hsic_patience == 0          # disabled by default
    assert ctrl.struct_hsic_min_delta == pytest.approx(1e-4)
    assert ctrl.struct_min_epochs == 0


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
