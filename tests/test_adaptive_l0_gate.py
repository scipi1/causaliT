"""Tests for the L0-on-HSIC-progress gate in ``PhaseController``.

Run with:  pytest tests/test_adaptive_l0_gate.py -v

Background
----------
``PhaseController`` (causaliT.training.adaptive_trainer) alternates between a
``reconstruct`` and a ``structure`` phase; the structure phase applies
``adaptive_training.structure.lambda_l0`` on entry (via ``_apply_lambdas``).
The L0 pruner is only safe while HSIC still RANKS the edges: once the
structural signal is exhausted (bi-fit complete), L0 becomes the sole force on
the gates and deflates them uniformly - the "ill phase" of the n=20 per-node
arm (flat posterior, collapsing validation fit).

The gate (``l0_gate_on_hsic: true``) tracks the RUN-best ``hsic_monitor`` value
across structure phases.  After ``l0_gate_patience`` consecutive structure
phases without a relative improvement of ``l0_gate_min_delta``, ``lambda_l0``
is applied as 0 on the next structure-phase entry; a later improvement re-arms
it.  ``l0_gate_on_hsic: false`` (default) is a no-op.

These tests drive the controller's ``on_validation_epoch_end`` state machine
with lightweight fakes; the REAL ``_apply_phase`` is used so the lambda
application itself is exercised (the fake module exposes the gradient-routing
parameter groups and a ``lambda_l0`` attribute).
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

class _FakeParam:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, flag):
        self.requires_grad = flag
        return self


class _FakeModule:
    """Stands in for the forecaster: param groups + a lambda_l0 attribute."""

    def __init__(self, lambda_l0=1e-5):
        self.training = True
        self.lambda_l0 = lambda_l0
        self._structural_params = [_FakeParam()]
        self._reconstruction_params = [_FakeParam()]

    def log(self, *args, **kwargs):
        pass

    def modules(self):
        return iter([])

    def train(self):
        self.training = True


class _FakeTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False
        self.max_epochs = None

    def save_checkpoint(self, path):  # pragma: no cover - never hit (stubbed)
        pass


def _make_controller(tmp_path, **struct_overrides):
    """Build a PhaseController with the gate enabled (unless overridden)."""
    struct = {
        "max_epochs": 200,
        "drop_pct": 0.20,
        "drop_patience": 100,      # keep the drop trigger out of the way
        "hsic_monitor": "val_hsic",
        "hsic_patience": 1,        # structure phases exit quickly on a plateau
        "hsic_min_delta": 1e-4,
        "min_epochs": 0,
        "lambda_l0": 1e-5,
        "l0_gate_on_hsic": True,
        "l0_gate_patience": 1,
        "l0_gate_min_delta": 1e-4,
    }
    struct.update(struct_overrides)
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "structure",
            "max_cycles": None,
            "eval_dag": False,
            "reconstruct": {"max_epochs": 100, "plateau_patience": 1,
                            "min_epochs": 0},
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
    trainer = _FakeTrainer()
    module = _FakeModule(lambda_l0=1e-5)
    # Enter the starting (structure) phase through the REAL _apply_phase so the
    # gated lambda lands on the module.
    controller._apply_phase(trainer, module, "structure")
    return controller, trainer, module


def _run_epochs(controller, trainer, module, n_epochs, hsic, x_mae=1.0):
    """Feed ``n_epochs`` validation boundaries with the given (constant) HSIC."""
    for _ in range(n_epochs):
        trainer.current_epoch += 1
        trainer.callback_metrics = {"val_x_mae": x_mae, "val_hsic": hsic}
        controller.on_validation_epoch_end(trainer, module)


def _run_until_phase(controller, trainer, module, target, hsic, x_mae=1.0, cap=50):
    """Feed validation boundaries until the controller enters ``target`` phase."""
    for _ in range(cap):
        if controller.current_phase == target:
            return True
        trainer.current_epoch += 1
        trainer.callback_metrics = {"val_x_mae": x_mae, "val_hsic": hsic}
        controller.on_validation_epoch_end(trainer, module)
    return controller.current_phase == target


def _run_cycle(controller, trainer, module, hsic):
    """Run one structure phase at ``hsic`` plus the following reconstruct phase.

    Returns the lambda_l0 applied on the module when the NEXT structure phase
    was entered (None if the run never came back to structure).
    """
    assert controller.current_phase == "structure"
    # Structure phase exits (HSIC plateau), then the reconstruct phase exits
    # (recon plateau) and the next structure phase is entered via the REAL
    # _apply_phase -> _gated_struct_cfg, which sets module.lambda_l0.
    assert _run_until_phase(controller, trainer, module, "reconstruct", hsic)
    assert _run_until_phase(controller, trainer, module, "structure", hsic)
    return module.lambda_l0


# ---------------------------------------------------------------------------
# 1. Gate disabled by default (backward-compatible)
# ---------------------------------------------------------------------------

def test_gate_disabled_by_default(tmp_path):
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "structure",
            "reconstruct": {},
            "structure": {"lambda_l0": 1e-5},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    ctrl = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True,
    )
    assert ctrl.l0_gate_on_hsic is False
    assert ctrl.l0_gate_patience == 1
    assert ctrl.l0_gate_min_delta == pytest.approx(1e-4)
    # And the raw struct_cfg passes through untouched.
    module = _FakeModule()
    assert ctrl._gated_struct_cfg(module)["lambda_l0"] == 1e-5


# ---------------------------------------------------------------------------
# 2. First structure phase is always armed
# ---------------------------------------------------------------------------

def test_first_structure_phase_is_armed(tmp_path):
    controller, trainer, module = _make_controller(tmp_path)
    assert controller._l0_active is True
    assert module.lambda_l0 == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# 3. A stalled structure phase closes the gate for the NEXT one
# ---------------------------------------------------------------------------

def test_stalled_phase_disarms_l0(tmp_path):
    controller, trainer, module = _make_controller(tmp_path)

    # Cycle 1: HSIC = 0.5 sets the run best -> L0 stays armed.
    assert _run_cycle(controller, trainer, module, hsic=0.5) == pytest.approx(1e-5)
    assert controller._hsic_run_best == pytest.approx(0.5)

    # Cycle 2: HSIC = 0.6 does NOT beat the run best -> stall #1; with
    # l0_gate_patience=1 the gate closes for the NEXT structure phase.
    assert _run_cycle(controller, trainer, module, hsic=0.6) == 0.0
    assert controller._l0_active is False
    assert controller._hsic_stall_cycles == 1


# ---------------------------------------------------------------------------
# 4. An improvement re-arms the gate
# ---------------------------------------------------------------------------

def test_improvement_rearms_l0(tmp_path):
    controller, trainer, module = _make_controller(tmp_path)

    assert _run_cycle(controller, trainer, module, hsic=0.5) == pytest.approx(1e-5)
    assert _run_cycle(controller, trainer, module, hsic=0.6) == 0.0   # stalled
    # Cycle 3 runs with L0 disarmed; HSIC improves to 0.4 -> re-armed, so the
    # NEXT structure phase carries the configured lambda again.
    assert _run_cycle(controller, trainer, module, hsic=0.4) == pytest.approx(1e-5)
    assert controller._l0_active is True
    assert controller._hsic_run_best == pytest.approx(0.4)
    assert controller._hsic_stall_cycles == 0


# ---------------------------------------------------------------------------
# 5. l0_gate_patience > 1 tolerates consecutive stalls
# ---------------------------------------------------------------------------

def test_gate_patience_tolerates_consecutive_stalls(tmp_path):
    controller, trainer, module = _make_controller(tmp_path, l0_gate_patience=2)

    assert _run_cycle(controller, trainer, module, hsic=0.5) == pytest.approx(1e-5)
    # One stall: gate still open (patience=2).
    assert _run_cycle(controller, trainer, module, hsic=0.6) == pytest.approx(1e-5)
    assert controller._l0_active is True
    # Second consecutive stall: gate closes.
    assert _run_cycle(controller, trainer, module, hsic=0.7) == 0.0
    assert controller._l0_active is False


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
