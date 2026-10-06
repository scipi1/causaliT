"""Tests for the optional final reconstruction-only phase in ``PhaseController``.

Run with:  pytest tests/test_adaptive_final_reconstruct.py -v

Background
----------
``PhaseController`` (causaliT.training.adaptive_trainer) alternates between a
``reconstruct`` and a ``structure`` phase for the first ``total_epoch_budget``
epochs.  The optional ``final_reconstruct`` phase is APPENDED on top of that
budget (``Trainer.max_epochs = total_epoch_budget + final_reconstruct.max_epochs``)
and refines the predictor before the model is used for ATE estimation:

    * structural parameters stay frozen (same freeze as a reconstruct phase),
    * the cross-fit data split is disabled - the FULL training set is used,
    * the phase exits on a validation plateau (same rate-of-improvement logic
      as the reconstruct phase, with its own patience / min_delta / min_epochs)
      or its own ``max_epochs`` cap, then the run stops.

Entry happens either at the alternating-budget boundary (reason
``alternating_budget``) or when ``max_cycles`` is reached first (the stop is
rerouted into the final phase).

These tests drive the controller's ``on_validation_epoch_end`` state machine
directly with lightweight fakes so no model / pl.Trainer is required.
"""

import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.adaptive_trainer import (
    _PHASE_CODE,
    PhaseController,
    adaptive_trainer,
)


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
    def __init__(self, max_epochs=100):
        self.current_epoch = 0
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False
        self.max_epochs = max_epochs

    def save_checkpoint(self, path):  # pragma: no cover - never hit (stubbed)
        pass


class _FakeModule:
    def __init__(self):
        self.training = True

    def log(self, *args, **kwargs):
        pass

    def modules(self):
        return iter([])

    def train(self):
        self.training = True


class _ParamModule(_FakeModule):
    """Fake module exposing the gradient-routing parameter groups."""

    def __init__(self):
        super().__init__()
        self._structural_params = [torch.nn.Parameter(torch.randn(3))]
        self._reconstruction_params = [torch.nn.Parameter(torch.randn(3))]


class _FakeDM:
    """Minimal datamodule stub owning the phase -> subset mapping."""

    def __init__(self, splits):
        self._splits = splits
        self.active_phase = None

    def set_active_phase(self, phase):
        subset = self._splits.get(phase)
        if subset is None:
            return None
        self.active_phase = phase
        return int(len(subset))


def _make_controller(tmp_path, final=None, monitor="val_x_mae",
                     recon_overrides=None, struct_overrides=None,
                     max_cycles=100, stub_side_effects=True):
    """Build a PhaseController with (optionally) stubbed transition side effects."""
    recon = {
        "max_epochs": 100,
        "min_epochs": 0,
        "warmup_min_epochs": 0,
        "plateau_patience": 2,
        "plateau_min_delta": 1e-4,
    }
    recon.update(recon_overrides or {})
    struct = {"max_epochs": 200, "drop_pct": 0.2, "drop_patience": 5}
    struct.update(struct_overrides or {})
    ad = {
        "monitor": monitor,
        "start_phase": "reconstruct",
        "max_cycles": max_cycles,
        "eval_dag": False,
        "reconstruct": recon,
        "structure": struct,
    }
    if final is not None:
        ad["final_reconstruct"] = final
    config = {
        "adaptive_training": ad,
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    controller = PhaseController(
        config=config,
        data_dir=str(tmp_path),
        save_dir=str(tmp_path),
        cluster=True,
    )

    events = []
    if stub_side_effects:
        # Record transitions instead of touching disk / DAG diagnostics.
        def _fake_record(trainer, pl_module, reason, from_phase, to_phase,
                         monitor_val):
            events.append({
                "reason": reason,
                "from_phase": from_phase,
                "to_phase": to_phase,
                "epoch": trainer.current_epoch,
                "phase_epochs": (trainer.current_epoch
                                 - controller._phase_start_epoch + 1),
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


def _drive(controller, trainer, module, epochs, monitor="val_x_mae",
           values=None):
    """Feed monitor values across ``epochs`` (an iterable of epoch indices).

    ``values`` is either a callable(epoch) -> float or a constant float.
    """
    for e in epochs:
        trainer.current_epoch = e
        value = values(e) if callable(values) else values
        trainer.callback_metrics = {monitor: value}
        controller.on_validation_epoch_end(trainer, module)
        if trainer.should_stop:
            break


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

def test_disabled_by_default(tmp_path):
    """No final_reconstruct block -> feature off, no behaviour change."""
    controller, _ = _make_controller(tmp_path)
    assert controller.final_enabled is False


def test_plateau_triggers_fall_back_to_reconstruct(tmp_path):
    """Unset plateau knobs inherit the reconstruct block's values."""
    controller, _ = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10},
        recon_overrides={"plateau_patience": 7, "plateau_min_delta": 1e-3},
    )
    assert controller.final_enabled is True
    assert controller.final_max_epochs == 10
    assert controller.final_plateau_patience == 7
    assert controller.final_plateau_min_delta == pytest.approx(1e-3)


def test_explicit_plateau_triggers_override(tmp_path):
    controller, _ = _make_controller(
        tmp_path,
        final={"enabled": True, "max_epochs": 10, "plateau_patience": 3,
               "plateau_min_delta": 5e-3, "min_epochs": 2},
        recon_overrides={"plateau_patience": 7},
    )
    assert controller.final_plateau_patience == 3
    assert controller.final_plateau_min_delta == pytest.approx(5e-3)
    assert controller.final_min_epochs == 2


def test_zero_max_epochs_disables(tmp_path):
    controller, _ = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 0},
    )
    assert controller.final_enabled is False


def test_phase_code_contains_final_reconstruct():
    assert _PHASE_CODE["final_reconstruct"] == 2


# ---------------------------------------------------------------------------
# 2. Entry trigger: alternating-budget boundary
# ---------------------------------------------------------------------------

def test_boundary_trigger_enters_final_phase(tmp_path):
    """The final phase starts once the alternating budget is exhausted.

    total_budget = 50, final = 10 -> Trainer max_epochs = 60; the alternating
    schedule owns epochs 0..49 and the switch fires at the validation boundary
    of epoch 49 (0-based), handing epochs 50..59 to the final phase.
    """
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10},
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    # Strictly improving monitor -> the reconstruct plateau never fires first.
    _drive(controller, trainer, module, range(0, 49),
           values=lambda e: 1.0 - 0.01 * e)
    assert events == []
    assert controller.current_phase == "reconstruct"

    # Epoch 49 = last epoch owned by the alternating schedule -> switch.
    _drive(controller, trainer, module, range(49, 50), values=0.5)
    assert len(events) == 1
    assert events[0]["reason"] == "alternating_budget"
    assert events[0]["from_phase"] == "reconstruct"
    assert events[0]["to_phase"] == "final_reconstruct"
    assert events[0]["epoch"] == 49
    assert controller.current_phase == "final_reconstruct"
    assert not trainer.should_stop


def test_boundary_trigger_from_structure_phase(tmp_path):
    """The boundary cuts whichever phase is active - including structure."""
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10},
    )
    controller.current_phase = "structure"
    controller._phase_start_epoch = 40
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()

    _drive(controller, trainer, module, range(40, 50), values=1.0)
    assert events[-1]["from_phase"] == "structure"
    assert events[-1]["to_phase"] == "final_reconstruct"
    assert events[-1]["reason"] == "alternating_budget"
    assert controller.current_phase == "final_reconstruct"


def test_no_boundary_trigger_when_disabled(tmp_path):
    """Without the final phase, crossing the (non-existent) boundary is a no-op."""
    controller, events = _make_controller(tmp_path)  # final disabled
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    # Improving values: no plateau; nothing else should happen either.
    _drive(controller, trainer, module, range(0, 60),
           values=lambda e: 1.0 - 0.01 * e)
    assert events == []
    assert controller.current_phase == "reconstruct"
    assert not trainer.should_stop


# ---------------------------------------------------------------------------
# 3. Final phase exit: plateau / min-epoch floor / budget cap
# ---------------------------------------------------------------------------

def _enter_final_phase(controller, trainer, start_epoch):
    """Put the controller into the final phase at ``start_epoch``."""
    controller.current_phase = "final_reconstruct"
    controller._phase_start_epoch = start_epoch
    controller._phase_best = float("inf")
    controller._plateau_counter = 0


def test_final_plateau_stops_run(tmp_path):
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10,
                         "plateau_patience": 2},
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    _enter_final_phase(controller, trainer, start_epoch=50)

    # Constant value: improvement on epoch 50 only, then the counter climbs to
    # patience(2) at the 3rd validation epoch (phase_epochs=3).
    _drive(controller, trainer, module, range(50, 60), values=1.0)
    assert trainer.should_stop
    assert events[-1]["reason"] == "final_recon_plateau"
    assert events[-1]["from_phase"] == "final_reconstruct"
    assert events[-1]["to_phase"] == "stop"
    assert events[-1]["phase_epochs"] == 3


def test_final_min_epochs_floor_suppresses_plateau(tmp_path):
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10,
                         "plateau_patience": 1, "min_epochs": 3},
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    _enter_final_phase(controller, trainer, start_epoch=50)

    # patience=1 would fire at phase_epochs=2, but the floor of 3 holds it.
    _drive(controller, trainer, module, range(50, 60), values=1.0)
    assert trainer.should_stop
    assert events[-1]["reason"] == "final_recon_plateau"
    assert events[-1]["phase_epochs"] == 3


def test_final_budget_cap_precedes_floor(tmp_path):
    """A huge floor would forbid the plateau exit, but the cap must still win."""
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 4,
                         "plateau_patience": 1, "min_epochs": 1000},
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    _enter_final_phase(controller, trainer, start_epoch=50)

    _drive(controller, trainer, module, range(50, 60), values=1.0)
    assert trainer.should_stop
    assert events[-1]["reason"] == "final_recon_budget"
    assert events[-1]["phase_epochs"] == 4


def test_final_phase_improving_metric_does_not_plateau(tmp_path):
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 5,
                         "plateau_patience": 2},
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()
    _enter_final_phase(controller, trainer, start_epoch=50)

    # Strictly improving: no plateau; the cap ends the phase at 5 epochs.
    _drive(controller, trainer, module, range(50, 60),
           values=lambda e: 1.0 - 0.01 * (e - 50))
    assert trainer.should_stop
    assert events[-1]["reason"] == "final_recon_budget"
    assert events[-1]["phase_epochs"] == 5


# ---------------------------------------------------------------------------
# 4. max_cycles reroutes into the final phase instead of stopping
# ---------------------------------------------------------------------------

def test_max_cycles_enters_final_phase_when_enabled(tmp_path):
    controller, events = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10,
                         "plateau_patience": 2},
        struct_overrides={"max_epochs": 3},
        max_cycles=1,
    )
    controller.current_phase = "structure"
    controller._phase_start_epoch = 0
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()

    # Structure budget exit at phase_epochs=3 completes cycle 1 = max_cycles.
    _drive(controller, trainer, module, range(0, 3), values=1.0)
    assert not trainer.should_stop
    assert events[-1]["reason"] == "struct_budget_final"
    assert events[-1]["to_phase"] == "final_reconstruct"
    assert controller.current_phase == "final_reconstruct"

    # The final phase then runs and stops the run on its plateau.
    _drive(controller, trainer, module, range(3, 20), values=1.0)
    assert trainer.should_stop
    assert events[-1]["reason"] == "final_recon_plateau"
    assert events[-1]["to_phase"] == "stop"


def test_max_cycles_still_stops_when_disabled(tmp_path):
    """Regression guard: without the final phase, max_cycles stops as before."""
    controller, events = _make_controller(
        tmp_path, struct_overrides={"max_epochs": 3}, max_cycles=1,
    )
    controller.current_phase = "structure"
    controller._phase_start_epoch = 0
    trainer = _FakeTrainer(max_epochs=60)
    module = _FakeModule()

    _drive(controller, trainer, module, range(0, 3), values=1.0)
    assert trainer.should_stop
    assert events[-1]["reason"] == "struct_budget_final"
    assert events[-1]["to_phase"] == "stop"


# ---------------------------------------------------------------------------
# 5. Phase application: freezing + full-data swap
# ---------------------------------------------------------------------------

def test_apply_phase_freezes_structure_unfreezes_reconstruction(tmp_path):
    """The final phase applies the same true freeze as a reconstruct phase."""
    controller, _ = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10},
        stub_side_effects=False,
    )
    trainer = _FakeTrainer(max_epochs=60)
    module = _ParamModule()
    # Start from the opposite state (structure training).
    for p in module._structural_params:
        p.requires_grad_(True)
    for p in module._reconstruction_params:
        p.requires_grad_(False)

    controller._apply_phase(trainer, module, "final_reconstruct")

    assert all(not p.requires_grad for p in module._structural_params)
    assert all(p.requires_grad for p in module._reconstruction_params)
    assert controller.current_phase == "final_reconstruct"


def test_swap_train_subset_restores_full_training_set(tmp_path):
    """Cross-fit OFF for the final phase: the full fold indices are restored."""
    splits = {
        "reconstruct": np.arange(0, 40),
        "structure": np.arange(40, 80),
        "final_reconstruct": np.arange(0, 80),  # full fold train indices
    }
    dm = _FakeDM(splits)
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "reconstruct",
            "eval_dag": False,
            "reconstruct": {},
            "structure": {},
            "final_reconstruct": {"enabled": True, "max_epochs": 10},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    controller = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True, dm=dm, stage_splits=splits,
    )
    assert controller.cross_fitting

    n = controller._swap_train_subset("final_reconstruct")
    assert n == 80
    assert dm.active_phase == "final_reconstruct"


def test_swap_train_subset_noop_without_cross_fitting(tmp_path):
    """No cross-fitting -> the full set is already in use, nothing to swap."""
    controller, _ = _make_controller(
        tmp_path, final={"enabled": True, "max_epochs": 10},
        stub_side_effects=False,
    )
    assert controller.cross_fitting is False
    assert controller._swap_train_subset("final_reconstruct") is None


# ---------------------------------------------------------------------------
# 6. Orchestrator wiring
# ---------------------------------------------------------------------------

def test_orchestrator_appends_budget_and_registers_full_split():
    """Guard the orchestrator wiring (a full fit is too heavy for a unit test)."""
    src = inspect.getsource(adaptive_trainer)

    # The final epochs are appended on top of the alternating budget.
    assert 'ad_cfg.get("final_reconstruct", {})' in src
    assert "total_budget + final_max_epochs" in src
    # The full fold train indices are registered under the final phase's key.
    assert 'stage_splits["final_reconstruct"]' in src
    # The summary reports the final phase.
    assert '"final_reconstruct"' in src


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
