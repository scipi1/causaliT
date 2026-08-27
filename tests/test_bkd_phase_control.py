"""
Tests for per-phase batch-key dropout (BKD) curriculum.

Covers:
1. BatchConsistentKeyDropout.set_phase_active: inactive phases pass tensors
   through unchanged and clear the HSIC gating mask, but the annealing
   clock keeps advancing (global run-level schedule).
2. BatchConsistentKeyDropout.set_schedule: run-time schedule override with
   validation; anneal re-anchors on progress already made (no reset).
3. PhaseController._apply_bkd_cfg: BKD active during reconstruct phases,
   inactive during structure phases; unmanaged when no phase block sets
   batch_key_dropout (backward compatible).
"""

import torch
import torch.nn as nn
import pytest
import tempfile
from pathlib import Path

from causaliT.core.modules.extra_layers import BatchConsistentKeyDropout
from causaliT.training.adaptive_trainer import PhaseController


@pytest.fixture
def work_dir():
    """Scratch dir inside the repo (Windows temp dirs are not writable here)."""
    root = Path("_pytest_tmp")
    root.mkdir(exist_ok=True)
    d = tempfile.mkdtemp(dir=root)
    yield Path(d)


# ---------------------------------------------------------------------------
# BatchConsistentKeyDropout unit tests
# ---------------------------------------------------------------------------

def _x(B=2, L=3, S=5):
    return torch.ones(B, L, S)


def test_phase_inactive_passes_through_and_clears_mask():
    bkd = BatchConsistentKeyDropout(p_init=1.0)  # drop everything when active
    bkd.train()
    out_active = bkd(_x())
    assert out_active.sum() == 0.0
    assert bkd._last_key_mask is not None

    bkd.set_phase_active(False)
    x = _x()
    out = bkd(x)
    assert torch.equal(out, x)
    assert bkd._last_key_mask is None
    assert bkd.get_hsic_active_mask() is None  # forecaster includes all vars


def test_phase_inactive_still_advances_annealing_clock():
    bkd = BatchConsistentKeyDropout(p_init=1.0, p_final=0.0, annealing_batches=4)
    bkd.train()
    bkd.set_phase_active(False)
    for _ in range(4):  # 4 inactive training forwards
        bkd(_x())
    # Global clock fully elapsed: p must be at the anneal target already.
    assert bkd._current_p() == pytest.approx(0.0)
    bkd.set_phase_active(True)
    out = bkd(_x())
    assert torch.equal(out, _x())  # p == 0: active but nothing dropped


def test_set_schedule_overrides_p():
    bkd = BatchConsistentKeyDropout(p_init=0.0)
    bkd.train()
    assert torch.equal(bkd(_x()), _x())
    bkd.set_schedule(p_init=1.0)
    assert bkd(_x()).sum() == 0.0
    with pytest.raises(ValueError):
        bkd.set_schedule(p_init=1.5)


def test_set_schedule_does_not_reset_progress():
    bkd = BatchConsistentKeyDropout(p_init=1.0, p_final=0.0, annealing_batches=10)
    bkd.train()
    for _ in range(5):
        bkd(_x())
    assert bkd._current_p() == pytest.approx(0.5)
    # Re-anchor mid-run: same budget, different endpoints — progress kept.
    bkd.set_schedule(p_init=0.6, p_final=0.0, annealing_batches=10)
    assert bkd._current_p() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# PhaseController integration
# ---------------------------------------------------------------------------

class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = nn.Linear(2, 2)
        self.att.bkd = BatchConsistentKeyDropout(p_init=0.1)
        self.logged = {}

    def log(self, name, value, on_step=False, on_epoch=True):
        self.logged[name] = value


def _make_controller(work_dir, recon_cfg=None, struct_cfg=None):
    config = {
        "model": {"model_object": "SingleCausalLayer"},
        "adaptive_training": {
            "reconstruct": recon_cfg or {},
            "structure": struct_cfg or {},
        },
    }
    return PhaseController(
        config=config,
        data_dir=str(work_dir),
        save_dir=str(work_dir / "out"),
        cluster=True,
    )


def test_controller_activates_bkd_in_reconstruct_only(work_dir):
    ctrl = _make_controller(
        work_dir,
        recon_cfg={"batch_key_dropout": 0.6, "batch_key_dropout_final": 0.0},
    )
    mod = _DummyModule()

    ctrl._apply_bkd_cfg(mod, "reconstruct")
    assert mod.att.bkd._phase_active is True
    assert mod.att.bkd.p_init == pytest.approx(0.6)
    assert mod.att.bkd.p_final == pytest.approx(0.0)

    ctrl._apply_bkd_cfg(mod, "structure")
    assert mod.att.bkd._phase_active is False
    # Schedule values retained (anneal continues on the global clock).
    assert mod.att.bkd.p_init == pytest.approx(0.6)


def test_controller_unmanaged_without_phase_keys(work_dir):
    ctrl = _make_controller(work_dir)  # no batch_key_dropout anywhere
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    assert mod.att.bkd._phase_active is True  # untouched (construction default)
    assert mod.att.bkd.p_init == pytest.approx(0.1)


def test_controller_structure_opt_in(work_dir):
    ctrl = _make_controller(
        work_dir,
        recon_cfg={"batch_key_dropout": 0.6},
        struct_cfg={"batch_key_dropout": 0.3},
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "structure")
    assert mod.att.bkd._phase_active is True
    assert mod.att.bkd.p_init == pytest.approx(0.3)


def test_final_reconstruct_inherits_reconstruct_bkd(work_dir):
    ctrl = _make_controller(
        work_dir, recon_cfg={"batch_key_dropout": 0.6}
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "final_reconstruct")
    assert mod.att.bkd._phase_active is True
    assert mod.att.bkd.p_init == pytest.approx(0.6)
