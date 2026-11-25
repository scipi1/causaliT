"""Tests for the declarative schedule and the ``joint`` phase of PhaseController.

Run with:  pytest tests/test_adaptive_joint_schedule.py -v
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.adaptive_trainer import PhaseController, resolve_schedule


@pytest.fixture
def tmp_path():
    """Workspace-local temp dir (system temp is not accessible here)."""
    base = project_root / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _FakeTrainer:
    def __init__(self, max_epochs=1000):
        self.current_epoch = 0
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False
        self.max_epochs = max_epochs

    def save_checkpoint(self, path):
        pass


class _Module:
    """Fake forecaster with gradient-routing groups and loss weights."""

    def __init__(self, structural_grad="hsic"):
        self.training = True
        self._structural_params = [torch.nn.Parameter(torch.randn(2))]
        self._reconstruction_params = [torch.nn.Parameter(torch.randn(2))]
        self.lambda_hsic = 1.0
        self.lambda_l0 = 0.5
        self.kappa = 0.0
        self.structural_grad = structural_grad
        self.joint_loader = "unset"

    def log(self, *a, **k):
        pass

    def modules(self):
        return iter([])

    def train(self):
        self.training = True

    def set_joint_structure_loader(self, loader):
        if loader is not None and self.structural_grad == "hsic_unrolled":
            raise ValueError("hsic_unrolled")
        self.joint_loader = loader


class _DM:
    """Datamodule fake with the stage-split API."""

    def __init__(self, splits):
        self._splits = splits
        self.active_phase = None

    def set_active_phase(self, key):
        self.active_phase = key
        return len(self._splits[key])

    def stage_split_dataloader(self, key):
        return ("loader", key)


def _make(tmp_path, ad, dm=None, splits=None):
    base = {"monitor": "val_x_mae", "total_epoch_budget": 1000,
            "reconstruct": {"max_epochs": 2}, "structure": {"max_epochs": 2},
            "eval_dag": False}
    base.update(ad)
    return PhaseController(config={"adaptive_training": base, "model": {}},
                           data_dir=str(tmp_path), save_dir=str(tmp_path),
                           cluster=True, dm=dm, stage_splits=splits)


def _drive(c, trainer, module, epochs, hsic=None):
    for e in epochs:
        trainer.current_epoch = e
        trainer.callback_metrics = {"val_x_mae": 1.0 - 1e-3 * e}
        if hsic is not None:
            trainer.callback_metrics["val_hsic"] = hsic(e)
        c.on_validation_epoch_end(trainer, module)
        if trainer.should_stop:
            break


# --- resolve_schedule --------------------------------------------------------

def test_legacy_translation():
    assert resolve_schedule({}) == ["alternate"]
    assert resolve_schedule({"start_phase": "warmup"}) == ["warmup", "alternate"]
    assert resolve_schedule({"final_reconstruct": {"enabled": True}}) == [
        "alternate", "final_reconstruct"]


@pytest.mark.parametrize("bad", [["joint", "warmup"], ["alternate", "alternate"],
                                 ["foo"], []])
def test_invalid_schedule_raises(bad):
    with pytest.raises(ValueError):
        resolve_schedule({"schedule": bad})


# --- ordering and joint semantics ---------------------------------------------

def test_warmup_joint_alternate_final_order(tmp_path):
    c = _make(tmp_path, {
        "schedule": ["warmup", "joint", "alternate", "final_reconstruct"],
        "warmup": {"max_epochs": 2}, "joint": {"max_epochs": 3},
        "final_reconstruct": {"max_epochs": 2}, "max_cycles": 1,
    })
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    assert c.current_phase == "warmup"
    _drive(c, trainer, m, range(0, 20))
    assert [t["from_phase"] for t in c.transitions] == [
        "warmup", "joint", "structure", "final_reconstruct"]
    assert c.transitions[-1]["to_phase"] == "stop"
    assert c.transitions[1]["reason"] == "joint_budget"
    assert trainer.should_stop


def test_joint_trains_both_groups_and_l0_defaults_zero(tmp_path):
    c = _make(tmp_path, {"schedule": ["joint", "alternate"],
                         "joint": {"lambda_hsic": 3.0}})
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    assert c.current_phase == "joint"
    assert all(p.requires_grad for p in m._structural_params)
    assert all(p.requires_grad for p in m._reconstruction_params)
    assert m.lambda_l0 == 0.0 and m.lambda_hsic == 3.0
    assert m.joint_loader == "unset"   # no cross-fitting -> no side loader


def test_lambdas_restored_per_phase(tmp_path):
    """Structure lambdas must not leak into the following reconstruct phase."""
    c = _make(tmp_path, {"start_phase": "structure",
                         "structure": {"max_epochs": 1, "lambda_l0": 0.9}})
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    assert m.lambda_l0 == 0.9
    _drive(c, trainer, m, [0])
    assert c.current_phase == "reconstruct"
    assert m.lambda_l0 == 0.5   # training-level value


def test_joint_hsic_plateau_exit(tmp_path):
    c = _make(tmp_path, {"schedule": ["joint", "alternate"],
                         "joint": {"max_epochs": 100, "hsic_patience": 2}})
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    _drive(c, trainer, m, range(0, 10), hsic=lambda e: 0.5)
    assert c.transitions[0]["reason"] == "joint_hsic_plateau"
    assert c.transitions[0]["phase_epochs"] == 3
    assert c.transitions[0]["to_phase"] == "structure"


def test_joint_cross_fit_uses_disjoint_split(tmp_path):
    splits = {"reconstruct": np.arange(5), "structure": np.arange(5, 10)}
    dm = _DM(splits)
    c = _make(tmp_path, {"schedule": ["joint", "alternate"],
                         "joint": {"max_epochs": 2}}, dm=dm, splits=splits)
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    assert dm.active_phase == "reconstruct"
    assert m.joint_loader == ("loader", "structure")
    _drive(c, trainer, m, range(0, 2))
    assert c.current_phase == "structure"
    assert m.joint_loader is None      # cleared after joint


def test_joint_rejects_hsic_unrolled(tmp_path):
    c = _make(tmp_path, {"schedule": ["joint", "alternate"]})
    with pytest.raises(ValueError):
        c.on_train_start(_FakeTrainer(), _Module(structural_grad="hsic_unrolled"))


# --- run end and resume ---------------------------------------------------------

def test_train_end_records_last_phase(tmp_path):
    c = _make(tmp_path, {"start_phase": "reconstruct",
                         "reconstruct": {"max_epochs": 100}})
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    _drive(c, trainer, m, range(0, 5))
    c.on_train_end(trainer, m)
    assert c.transitions[-1]["reason"] == "global_budget"
    assert c.transitions[-1]["from_phase"] == "reconstruct"
    c.on_train_end(trainer, m)          # idempotent
    assert len(c.transitions) == 1


def test_resume_skips_entry_actions(tmp_path):
    c = _make(tmp_path, {"start_phase": "structure"})
    trainer, m = _FakeTrainer(), _Module()
    c.on_train_start(trainer, m)
    state = c.state_dict()
    calls = []
    c2 = _make(tmp_path, {"start_phase": "structure"})
    c2._apply_phase_entry_actions = lambda *a, **k: calls.append(1)
    c2.load_state_dict(state)
    c2.on_train_start(trainer, m)
    assert calls == []
    assert c2._struct_phase_count == state["_struct_phase_count"] == 1

