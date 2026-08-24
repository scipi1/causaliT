"""Tests for the cross-fit ``swap_splits`` option in ``PhaseController``.

Run with:  pytest tests/test_adaptive_swap_splits.py -v

Background
----------
With cross-fitting enabled (``adaptive_training.data_split_ratio`` in (0, 1))
the fold's training indices are partitioned into two disjoint subsets:
``reconstruct`` (I_1) and ``structure`` (I_2).  By default they stay FIXED for
the whole run (recon always on I_1, structure always on I_2).

``adaptive_training.swap_splits: true`` exchanges the two subsets after EVERY
completed recon+structure cycle (i.e. at each structure -> reconstruct
transition)::

    recon_1(I_1) -> struct_1(I_2) -> [swap] -> recon_2(I_2) -> struct_2(I_1)
    -> [swap] -> recon_3(I_1) -> struct_3(I_2) -> ...

The swap is keyed to the parity of ``_cycle_count`` (completed structure
phases), which only changes at the end of a structure phase.  Within every
recon->struct pairing ``_cycle_count`` is constant, so the two phases always
train on DISJOINT subsets — the central guarantee under test:

    every structure phase trains on a DIFFERENT split than the reconstruction
    phase that immediately preceded it

(DML/DARTS honesty: residual-HSIC stays out-of-sample w.r.t. the reconstruction
fit, while each sample serves both roles across the run).

These tests drive the controller's ``on_validation_epoch_end`` state machine
directly with lightweight fakes (no model / pl.Trainer required), using the
REAL ``_apply_phase`` / ``_record_transition`` so the genuine split-request
path (``_swap_train_subset`` -> ``dm.set_active_phase``) is exercised.
"""

import inspect
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.adaptive_trainer import PhaseController, adaptive_trainer


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

    def train(self):
        self.training = True


class _ParamModule(_FakeModule):
    """Fake module exposing the gradient-routing parameter groups."""

    def __init__(self):
        super().__init__()
        self._structural_params = [torch.nn.Parameter(torch.randn(3))]
        self._reconstruction_params = [torch.nn.Parameter(torch.randn(3))]


class _FakeDM:
    """Datamodule stub owning the phase -> subset mapping.

    Records the full history of requested split keys so the test can assert
    exactly which subset each phase was pointed at.
    """

    def __init__(self, splits):
        self._splits = splits
        self.active_phase = None
        self.history = []

    def set_active_phase(self, phase):
        subset = self._splits.get(phase)
        if subset is None:
            return None
        self.active_phase = phase
        self.history.append(phase)
        return int(len(subset))


def _make_controller(tmp_path, swap_splits, with_crossfit=True,
                     recon_overrides=None, struct_overrides=None,
                     max_cycles=None, start_phase="reconstruct"):
    """Build a PhaseController with the REAL apply/record path (no stubbing).

    Per-phase ``max_epochs=2`` with a strictly improving monitor makes every
    phase exit on its budget cap, giving deterministic cycling.
    """
    recon = {
        "max_epochs": 2,
        "min_epochs": 0,
        "warmup_min_epochs": 0,
        "plateau_patience": 5,      # high: budget (not plateau) ends the phase
        "plateau_min_delta": 1e-4,
    }
    recon.update(recon_overrides or {})
    struct = {
        "max_epochs": 2,
        "drop_pct": 0.2,
        "drop_patience": 5,          # high: budget (not drop) ends the phase
        "hsic_patience": 0,          # HSIC-plateau exit disabled
    }
    struct.update(struct_overrides or {})
    ad = {
        "monitor": "val_x_mae",
        "start_phase": start_phase,
        "max_cycles": max_cycles,
        "eval_dag": False,
        "swap_splits": swap_splits,
        "reconstruct": recon,
        "structure": struct,
    }
    config = {
        "adaptive_training": ad,
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    dm = None
    splits = None
    if with_crossfit:
        splits = {
            "reconstruct": np.arange(0, 40),    # I_1
            "structure": np.arange(40, 80),     # I_2
        }
        dm = _FakeDM(splits)
    controller = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True, dm=dm, stage_splits=splits,
        val_local_idx=np.arange(80, 90), test_idx=np.arange(90, 100),
    )
    return controller, dm


def _drive(controller, trainer, module, epochs, monitor="val_x_mae",
           values=None):
    """Feed monitor values across ``epochs`` (an iterable of epoch indices)."""
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
    """No swap_splits key -> feature off, no behaviour change."""
    config = {
        "adaptive_training": {
            "monitor": "val_x_mae",
            "start_phase": "reconstruct",
            "eval_dag": False,
            "reconstruct": {},
            "structure": {},
        },
        "model": {"model_object": "AttentionSelectorLayer"},
    }
    splits = {"reconstruct": np.arange(0, 40), "structure": np.arange(40, 80)}
    controller = PhaseController(
        config=config, data_dir=str(tmp_path), save_dir=str(tmp_path),
        cluster=True, dm=_FakeDM(splits), stage_splits=splits,
    )
    assert controller.cross_fitting is True
    assert controller.swap_splits is False


def test_enabled_when_set(tmp_path):
    controller, _ = _make_controller(tmp_path, swap_splits=True)
    assert controller.cross_fitting is True
    assert controller.swap_splits is True


def test_forced_off_without_cross_fitting(tmp_path, caplog):
    """swap_splits is meaningless without a split -> forced off with a warning."""
    with caplog.at_level(logging.WARNING,
                         logger="causaliT.training.adaptive_trainer"):
        controller, _ = _make_controller(
            tmp_path, swap_splits=True, with_crossfit=False,
        )
    assert controller.cross_fitting is False
    assert controller.swap_splits is False
    assert any("swap_splits" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. _resolve_split_key parity
# ---------------------------------------------------------------------------

def test_resolve_split_key_parity(tmp_path):
    """Odd cycle counts swap the key; even counts and final_reconstruct don't."""
    controller, _ = _make_controller(tmp_path, swap_splits=True)

    controller._cycle_count = 0     # no completed cycle yet -> identity
    assert controller._resolve_split_key("reconstruct") == "reconstruct"
    assert controller._resolve_split_key("structure") == "structure"

    controller._cycle_count = 1     # one completed cycle -> swapped
    assert controller._resolve_split_key("reconstruct") == "structure"
    assert controller._resolve_split_key("structure") == "reconstruct"

    controller._cycle_count = 2     # two completed cycles -> identity again
    assert controller._resolve_split_key("reconstruct") == "reconstruct"
    assert controller._resolve_split_key("structure") == "structure"

    # The final reconstruction phase (full training set) is never swapped.
    controller._cycle_count = 1
    assert controller._resolve_split_key("final_reconstruct") == "final_reconstruct"


def test_resolve_split_key_identity_when_disabled(tmp_path):
    controller, _ = _make_controller(tmp_path, swap_splits=False)
    controller._cycle_count = 1
    assert controller._resolve_split_key("reconstruct") == "reconstruct"
    assert controller._resolve_split_key("structure") == "structure"


# ---------------------------------------------------------------------------
# 3. End-to-end swap sequence (the spec example)
# ---------------------------------------------------------------------------

def test_swap_sequence_matches_spec(tmp_path):
    """recon_1(I_1), struct_1(I_2), recon_2(I_2), struct_2(I_1), ...

    The keys requested from the datamodule and the per-transition ``train_split``
    records must both follow the swap pattern, and the central invariant must
    hold: every structure phase trains on a DIFFERENT split than the
    reconstruction phase that immediately preceded it.
    """
    controller, dm = _make_controller(tmp_path, swap_splits=True)
    assert controller.cross_fitting and controller.swap_splits
    trainer = _FakeTrainer(max_epochs=100)
    module = _ParamModule()

    controller.on_train_start(trainer, module)
    # Strictly improving monitor: recon never plateaus and struct never drops,
    # so every phase ends on its 2-epoch budget cap -> deterministic cycling.
    _drive(controller, trainer, module, range(0, 9),
           values=lambda e: 1.0 - 0.001 * e)

    # Keys requested from the datamodule at each phase application.  The first
    # entry is the initial warmup (on_train_start); each later entry is one
    # phase switch.  Note the swap after every completed cycle.
    assert dm.history == [
        "reconstruct",  # recon_1 (cycle 0) -> I_1
        "structure",    # struct_1 (cycle 0) -> I_2
        "structure",    # recon_2 (cycle 1, swapped) -> I_2
        "reconstruct",  # struct_2 (cycle 1, swapped) -> I_1
        "reconstruct",  # recon_3 (cycle 2) -> I_1
        "structure",    # struct_3 (cycle 2) -> I_2
        "structure",    # recon_4 (cycle 3, swapped) -> I_2
        "reconstruct",  # struct_4 (cycle 3, swapped) -> I_1
        "reconstruct",  # recon_5 (cycle 4) -> I_1
    ]

    # The split each ENDING phase actually trained on.
    from_phases = [t["from_phase"] for t in controller.transitions]
    train_splits = [t["train_split"] for t in controller.transitions]
    assert from_phases == [
        "reconstruct", "structure", "reconstruct", "structure",
        "reconstruct", "structure", "reconstruct", "structure",
    ]
    assert train_splits == [
        "reconstruct",  # recon_1 -> I_1
        "structure",    # struct_1 -> I_2
        "structure",    # recon_2 -> I_2
        "reconstruct",  # struct_2 -> I_1
        "reconstruct",  # recon_3 -> I_1
        "structure",    # struct_3 -> I_2
        "structure",    # recon_4 -> I_2
        "reconstruct",  # struct_4 -> I_1
    ]

    # INVARIANT: structure never reuses the split of the previous reconstruction.
    for i, t in enumerate(controller.transitions):
        if t["from_phase"] != "structure":
            continue
        prev = controller.transitions[i - 1]
        assert prev["from_phase"] == "reconstruct"
        assert t["train_split"] != prev["train_split"], (
            f"structure phase {i} reused the previous reconstruction split "
            f"({t['train_split']})"
        )


def test_no_swap_when_disabled(tmp_path):
    """Backward compatibility: flag off -> every phase trains on its own split."""
    controller, dm = _make_controller(tmp_path, swap_splits=False)
    trainer = _FakeTrainer(max_epochs=100)
    module = _ParamModule()

    controller.on_train_start(trainer, module)
    _drive(controller, trainer, module, range(0, 9),
           values=lambda e: 1.0 - 0.001 * e)

    # Keys always equal the phase names (no exchange).
    assert dm.history == [
        "reconstruct", "structure", "reconstruct", "structure", "reconstruct",
        "structure", "reconstruct", "structure", "reconstruct",
    ]
    for t in controller.transitions:
        assert t["train_split"] == t["from_phase"]


# ---------------------------------------------------------------------------
# 4. Orchestrator wiring
# ---------------------------------------------------------------------------

def test_orchestrator_reports_swap_splits():
    """Guard the orchestrator wiring (a full fit is too heavy for a unit test)."""
    src = inspect.getsource(adaptive_trainer)
    # The summary JSON reports the (validated) swap flag.
    assert '"swap_splits": controller.swap_splits' in src


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
