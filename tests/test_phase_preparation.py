"""
Tests for the adaptive phase-preparation contract and same-split drift probe.

Covers:
1. HSIC cross-fit folds track the active train_ds (stale-fold regression).
2. Structure phase entry measures a relaxed-gate reconstruction baseline on
   the structure-optimization split BEFORE any structural update.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from causaliT.training.adaptive_trainer import PhaseController
from causaliT.training.stage_causal_dataloader import StageCausalDataModule


def test_xfit_folds_track_current_train_ds():
    dm = StageCausalDataModule.__new__(StageCausalDataModule)
    dm.batch_size = 4
    dm.num_workers = 0
    dm.persistent_workers = False
    dm._xfit_ds_a = None
    dm._xfit_ds_b = None
    dm._xfit_cfg = None

    ds1 = TensorDataset(torch.randn(10, 2, 1), torch.randn(10, 2, 1))
    ds2 = TensorDataset(torch.randn(6, 2, 1), torch.randn(6, 2, 1))
    dm.train_ds = ds1

    n_a, n_b = dm.set_hsic_cross_fit(ratio=0.5, seed=0)
    assert (n_a, n_b) == (5, 5)
    assert dm._xfit_ds_a.dataset is ds1

    # A phase switch rebuilds train_ds; the fold subsets must follow.
    dm.train_ds = ds2
    n_a, n_b = dm._recompute_hsic_cross_fit()
    assert (n_a, n_b) == (3, 3)
    assert dm._xfit_ds_a.dataset is ds2
    assert dm._xfit_ds_b.dataset is ds2


class _FakeParam:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, flag):
        self.requires_grad = flag
        return self


class _FakeModule:
    def __init__(self, probe_values=()):
        self.training = True
        self._structural_params = [_FakeParam()]
        self._reconstruction_params = [_FakeParam()]
        self._probe_values = list(probe_values)
        self.probe_calls = 0
        self.logged = {}

    def modules(self):
        return iter([])

    def log(self, name, value, on_step=False, on_epoch=True):
        self.logged[name] = value

    def train(self):
        self.training = True

    def reconstruction_mse(self, dataloader, gate_mode="relaxed"):
        assert gate_mode == "relaxed"
        self.probe_calls += 1
        return float(self._probe_values.pop(0))


class _FakeDM:
    def __init__(self, splits):
        self._splits = {k: np.asarray(v) for k, v in splits.items()}
        self.active_phase = None
        self.batch_size = 4
        self.num_workers = 0
        self.persistent_workers = False
        self.train_ds = TensorDataset(torch.randn(8, 2, 1), torch.randn(8, 2, 1))
        self._xfit_ds_a = None
        self._xfit_ds_b = None

    def set_active_phase(self, phase):
        subset = self._splits.get(phase)
        if subset is None:
            return None
        self.active_phase = phase
        return int(len(subset))

    def hsic_cross_fit_eval_dataloader(self, fold="b"):
        return DataLoader(self.train_ds, batch_size=4, shuffle=False)


class _FakeTrainer:
    def __init__(self, epoch=0):
        self.current_epoch = epoch
        self.callback_metrics = {}
        self.sanity_checking = False
        self.max_epochs = 1000
        self.should_stop = False
        self.optimizers = []

    def save_checkpoint(self, path):
        pass


def _make_controller(tmp_path, **struct_overrides):
    struct = {"max_epochs": 50, "min_epochs": 1, "drop_pct": 0.05, "drop_patience": 1}
    struct.update(struct_overrides)
    config = {
        "model": {"model_object": "AttentionSelectorLayer"},
        "adaptive_training": {
            "monitor": "val_loss_x",
            "start_phase": "reconstruct",
            "eval_dag": False,
            "reconstruct": {},
            "structure": struct,
        },
    }
    splits = {"reconstruct": np.arange(8), "structure": np.arange(8, 16)}
    dm = _FakeDM(splits)
    ctl = PhaseController(
        config=config,
        data_dir=str(tmp_path),
        save_dir=str(tmp_path / "out"),
        cluster=True,
        dm=dm,
        stage_splits=splits,
        val_local_idx=np.arange(16, 20),
        test_idx=np.arange(20, 24),
    )
    return ctl, dm


def test_structure_entry_measures_relaxed_same_split_baseline(tmp_path):
    ctl, dm = _make_controller(tmp_path)
    module = _FakeModule(probe_values=[0.123])
    trainer = _FakeTrainer(epoch=10)

    ctl._apply_phase(trainer, module, "structure")

    assert module.probe_calls == 1
    assert ctl._phase_entry_monitor == pytest.approx(0.123)
    assert module.logged["adaptive_phase_entry_monitor"] == pytest.approx(0.123)
    assert ctl._active_split_key == "structure"
    assert all(p.requires_grad for p in module._structural_params)
    assert all(not p.requires_grad for p in module._reconstruction_params)


def test_contract_raises_on_split_mismatch(tmp_path):
    ctl, dm = _make_controller(tmp_path)
    module = _FakeModule(probe_values=[0.1])
    trainer = _FakeTrainer()

    # The datamodule accepts the request but stays on the old split.
    dm.set_active_phase = lambda phase: int(len(dm._splits[phase]))

    with pytest.raises(RuntimeError, match="split mismatch"):
        ctl._apply_phase(trainer, module, "structure")


def test_first_epoch_degradation_triggers_drift_exit(tmp_path):
    ctl, dm = _make_controller(tmp_path)
    module = _FakeModule(probe_values=[1.0, 2.0])
    trainer = _FakeTrainer(epoch=0)

    events = []
    ctl._record_transition = lambda tr, pl, reason, from_phase, to_phase, monitor_val: events.append(
        {"reason": reason, "from_phase": from_phase, "to_phase": to_phase,
         "monitor": monitor_val}
    )

    ctl._apply_phase(trainer, module, "structure")  # baseline = 1.0
    trainer.callback_metrics = {"val_loss_x": 0.5}  # ignored: probe wins
    ctl.on_validation_epoch_end(trainer, module)    # probe = 2.0 > 1.05

    assert len(events) == 1
    assert events[0]["reason"] == "struct_recon_drift"
    assert events[0]["to_phase"] == "reconstruct"
    assert events[0]["monitor"] == pytest.approx(2.0)
    assert module.logged["adaptive/struct_recon_mse_relaxed"] == pytest.approx(2.0)


def _capture_transitions(ctl):
    events = []
    ctl._record_transition = lambda tr, pl, reason, from_phase, to_phase, monitor_val: events.append(
        {"reason": reason, "from_phase": from_phase, "to_phase": to_phase,
         "monitor": monitor_val}
    )
    return events


def _feed_structure_epochs(ctl, module, trainer, hsic_values, start_epoch=0):
    for i, h in enumerate(hsic_values):
        trainer.current_epoch = start_epoch + i
        trainer.callback_metrics = {"val_loss_x": 0.5, "train_hsic": h}
        ctl.on_validation_epoch_end(trainer, module)
        if ctl.current_phase != "structure":
            return True
    return False


def test_hsic_plateau_fires_after_own_floor(tmp_path):
    ctl, dm = _make_controller(
        tmp_path,
        hsic_monitor="train_hsic", hsic_patience=2, hsic_min_epochs=3,
        drift_min_epochs=10, drop_pct=0.50, drop_patience=1,
    )
    module = _FakeModule(probe_values=[1.0] * 8)
    trainer = _FakeTrainer(epoch=0)
    events = _capture_transitions(ctl)

    ctl._apply_phase(trainer, module, "structure")  # baseline consumes 1.0
    switched = _feed_structure_epochs(
        ctl, module, trainer, [1.0, 1.0, 1.0]
    )

    assert switched
    assert events[0]["reason"] == "struct_hsic_plateau"
    assert events[0]["to_phase"] == "reconstruct"


def test_drift_fires_before_hsic_floor(tmp_path):
    ctl, dm = _make_controller(
        tmp_path,
        hsic_monitor="train_hsic", hsic_patience=2, hsic_min_epochs=10,
        drift_min_epochs=1, drop_pct=0.05, drop_patience=1,
    )
    module = _FakeModule(probe_values=[1.0, 2.0])
    trainer = _FakeTrainer(epoch=0)
    events = _capture_transitions(ctl)

    ctl._apply_phase(trainer, module, "structure")  # baseline = 1.0
    switched = _feed_structure_epochs(ctl, module, trainer, [1.0])

    assert switched
    assert events[0]["reason"] == "struct_recon_drift"


def test_trigger_floors_fall_back_to_shared_min_epochs(tmp_path):
    ctl, _ = _make_controller(tmp_path, min_epochs=4)
    assert ctl.struct_drift_min_epochs == 4
    assert ctl.struct_hsic_min_epochs == 4
