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


# ---------------------------------------------------------------------------
# Count-based (min_keys) phase curriculum
# ---------------------------------------------------------------------------

def test_min_keys_only_phase_activates_bkd_at_p1(work_dir):
    """A phase block with ONLY batch_key_dropout_min_keys activates BKD at
    p=1 and applies the min-keys floor (max_keys regime)."""
    ctrl = _make_controller(
        work_dir, recon_cfg={"batch_key_dropout_min_keys": 1}
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    bkd = mod.att.bkd
    assert bkd._phase_active is True
    assert bkd.p_init == pytest.approx(1.0)
    assert bkd.min_keys == 1
    assert bkd.deterministic is False  # unchanged (construction default)
    # Effective behaviour: exactly one key kept per forward.
    bkd.train()
    bkd(_x(S=5))
    assert int(bkd._last_key_mask.sum().item()) == 1


def test_min_keys_with_deterministic_phase_override(work_dir):
    """Per-phase deterministic switch: exact-count sampling, min_keys keys."""
    ctrl = _make_controller(
        work_dir,
        recon_cfg={
            "batch_key_dropout_min_keys": 2,
            "batch_key_dropout_deterministic": True,
        },
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    bkd = mod.att.bkd
    assert bkd.deterministic is True
    assert bkd.min_keys == 2
    bkd.train()
    for _ in range(10):
        bkd(_x(S=6))
        assert int(bkd._last_key_mask.sum().item()) == 2


def test_min_keys_combined_with_p_schedule(work_dir):
    """min_keys acts as a floor on top of a p-schedule phase."""
    ctrl = _make_controller(
        work_dir,
        recon_cfg={"batch_key_dropout": 0.9, "batch_key_dropout_min_keys": 1},
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    bkd = mod.att.bkd
    assert bkd.p_init == pytest.approx(0.9)
    assert bkd.min_keys == 1
    bkd.train()
    for _ in range(20):
        bkd(_x(S=4))
        assert int(bkd._last_key_mask.sum().item()) >= 1


def test_sampling_settings_persist_across_phases_without_override(work_dir):
    """Phases that omit the sampling keys leave the previous settings
    untouched (schedule activation still toggles)."""
    ctrl = _make_controller(
        work_dir,
        recon_cfg={
            "batch_key_dropout": 0.5,
            "batch_key_dropout_min_keys": 1,
            "batch_key_dropout_deterministic": True,
        },
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    assert mod.att.bkd.min_keys == 1
    assert mod.att.bkd.deterministic is True
    ctrl._apply_bkd_cfg(mod, "structure")
    assert mod.att.bkd._phase_active is False
    assert mod.att.bkd.min_keys == 1  # retained
    assert mod.att.bkd.deterministic is True


def test_set_sampling_validation():
    bkd = BatchConsistentKeyDropout(p_init=0.5)
    with pytest.raises(ValueError):
        bkd.set_sampling(min_keys=-2)
    bkd.set_sampling(min_keys=2, deterministic=True)
    assert bkd.min_keys == 2 and bkd.deterministic is True
    bkd.set_sampling()  # no-op
    assert bkd.min_keys == 2 and bkd.deterministic is True



# ---------------------------------------------------------------------------
# bkd_min_keys_ladder (count-based per-cycle curriculum)
# ---------------------------------------------------------------------------

def _make_ladder_controller(work_dir, ladder, phase_cfg=None):
    config = {
        "model": {"model_object": "SingleCausalLayer"},
        "adaptive_training": {
            "reconstruct": phase_cfg or {},
            "structure": {},
            "bkd_min_keys_ladder": ladder,
        },
    }
    return PhaseController(
        config=config,
        data_dir=str(work_dir),
        save_dir=str(work_dir / "out"),
        cluster=True,
    )


def test_min_keys_ladder_applies_rung_per_cycle(work_dir):
    """Rung k is applied identically to both phases of cycle k."""
    ctrl = _make_ladder_controller(work_dir, [3, 1])
    mod = _DummyModule()

    ctrl._apply_bkd_cfg(mod, "reconstruct")   # cycle 0
    assert mod.att.bkd.min_keys == 3
    assert mod.att.bkd._phase_active is True
    ctrl._apply_bkd_cfg(mod, "structure")     # same cycle: same rung
    assert mod.att.bkd.min_keys == 3

    ctrl._cycle_count = 1
    ctrl._apply_bkd_cfg(mod, "reconstruct")   # cycle 1
    assert mod.att.bkd.min_keys == 1

    ctrl._cycle_count = 5                     # clamp to last rung
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    assert mod.att.bkd.min_keys == 1


def test_min_keys_ladder_activates_bkd_at_p1_without_p_key(work_dir):
    """The ladder alone (no p keys) activates BKD at p=1 with the rung."""
    ctrl = _make_ladder_controller(work_dir, [2])
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    bkd = mod.att.bkd
    assert bkd._phase_active is True
    assert bkd.p_init == pytest.approx(1.0)
    assert bkd.min_keys == 2


def test_min_keys_ladder_with_deterministic_keeps_exact_count(work_dir):
    ctrl = _make_ladder_controller(
        work_dir, [2], phase_cfg={"batch_key_dropout_deterministic": True}
    )
    mod = _DummyModule()
    ctrl._apply_bkd_cfg(mod, "reconstruct")
    bkd = mod.att.bkd
    assert bkd.deterministic is True
    bkd.train()
    for _ in range(10):
        bkd(_x(S=6))
        assert int(bkd._last_key_mask.sum().item()) == 2


def test_min_keys_ladder_mutually_exclusive_with_p_ladder(work_dir):
    config = {
        "model": {"model_object": "SingleCausalLayer"},
        "adaptive_training": {
            "reconstruct": {},
            "structure": {},
            "bkd_ladder": [0.5],
            "bkd_min_keys_ladder": [2],
        },
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        PhaseController(
            config=config,
            data_dir=str(work_dir),
            save_dir=str(work_dir / "out"),
            cluster=True,
        )


def test_min_keys_ladder_validation(work_dir):
    with pytest.raises(ValueError, match="bkd_min_keys_ladder"):
        _make_ladder_controller(work_dir, [])
    with pytest.raises(ValueError, match="bkd_min_keys_ladder"):
        _make_ladder_controller(work_dir, [2, -1])

