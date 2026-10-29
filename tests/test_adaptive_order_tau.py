"""Tests for the order-temperature anneal and the late L0 turn-on.

Run with:  pytest tests/test_adaptive_order_tau.py -v
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


class _FakeOrderAttention:
    def __init__(self, tau=1.0, noise=1.0):
        self.perm_tau = tau
        self.noise_scale = noise

    def set_order_temperature(self, tau=None, noise_scale=None):
        if tau is not None:
            self.perm_tau = float(tau)
        if noise_scale is not None:
            self.noise_scale = float(noise_scale)


class _FakeModel:
    def __init__(self, oa):
        self._oa = oa

    def order_attention(self):
        return self._oa

    def set_order_temperature(self, tau=None, noise_scale=None):
        if self._oa is None:
            return False
        self._oa.set_order_temperature(tau=tau, noise_scale=noise_scale)
        return True


class _FakeModule:
    def __init__(self, oa=None, lambda_l0=0.0):
        self.training = True
        self.lambda_l0 = lambda_l0
        self.model = _FakeModel(oa)
        self.logged = {}
        self._structural_params = [_FakeParam()]
        self._reconstruction_params = [_FakeParam()]

    def log(self, name, value, **kwargs):
        self.logged[name] = value

    def modules(self):
        return iter([])

    def train(self):
        self.training = True


class _FakeTrainer:
    def __init__(self, epoch=0):
        self.current_epoch = epoch
        self.sanity_checking = False
        self.callback_metrics = {}
        self.optimizers = []
        self.should_stop = False
        self.max_epochs = None


# ---------------------------------------------------------------------------
# 1. Off by default: tau/noise untouched when the keys are unset
# ---------------------------------------------------------------------------

def test_off_by_default(tmp_path):
    ctrl = _make_controller(tmp_path)
    oa = _FakeOrderAttention(tau=1.0, noise=1.0)
    module = _FakeModule(oa)
    ctrl._update_order_temperature(_FakeTrainer(epoch=50), module)
    assert oa.perm_tau == 1.0
    assert oa.noise_scale == 1.0
    assert "order_tau" not in module.logged


# ---------------------------------------------------------------------------
# 2. Schedule: geometric tau, linear noise, exact start/mid/end values
# ---------------------------------------------------------------------------

def test_tau_and_noise_schedule(tmp_path):
    ctrl = _make_controller(
        tmp_path,
        order_tau_final=0.1, order_noise_final=0.0,
        order_tau_anneal_start=10, order_tau_anneal_end=20,
    )
    oa = _FakeOrderAttention(tau=1.0, noise=1.0)
    module = _FakeModule(oa)
    for epoch, exp_tau, exp_noise in [
        (0, 1.0, 1.0), (10, 1.0, 1.0),
        (15, 1.0 * 0.1 ** 0.5, 0.5), (20, 0.1, 0.0), (30, 0.1, 0.0),
    ]:
        ctrl._update_order_temperature(_FakeTrainer(epoch), module)
        assert oa.perm_tau == pytest.approx(exp_tau)
        assert oa.noise_scale == pytest.approx(exp_noise)
    assert module.logged["order_tau"] == pytest.approx(0.1)
    assert module.logged["order_noise_scale"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Defaults: start = warmup end, end = 0.8 * total_epoch_budget
# ---------------------------------------------------------------------------

def test_default_window_from_warmup_and_budget(tmp_path):
    ctrl = _make_controller(
        tmp_path,
        total_epoch_budget=5000,
        order_tau_final=0.1,
        warmup={"enabled": True, "max_epochs": 800},
    )
    assert ctrl.order_tau_anneal_start == 800
    assert ctrl.order_tau_anneal_end == 4000


# ---------------------------------------------------------------------------
# 4. No order-factorized block: no-op (no error, no logging)
# ---------------------------------------------------------------------------

def test_no_order_block_is_noop(tmp_path):
    ctrl = _make_controller(tmp_path, order_tau_final=0.1)
    module = _FakeModule(oa=None)
    ctrl._update_order_temperature(_FakeTrainer(epoch=50), module)
    assert "order_tau" not in module.logged


# ---------------------------------------------------------------------------
# 5. Late L0 turn-on: held at 0.0 before lambda_l0_start_epoch
# ---------------------------------------------------------------------------

def test_late_l0_turn_on(tmp_path):
    ctrl = _make_controller(
        tmp_path,
        structure={"lambda_l0": 1e-4, "lambda_l0_start_epoch": 10},
    )
    module = _FakeModule(lambda_l0=0.0)
    ctrl._apply_phase(_FakeTrainer(epoch=5), module, "structure")
    assert module.lambda_l0 == 0.0
    assert module.logged["lambda_l0_eff"] == 0.0
    ctrl._apply_phase(_FakeTrainer(epoch=10), module, "structure")
    assert module.lambda_l0 == pytest.approx(1e-4)
    assert module.logged["lambda_l0_eff"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# 6. lambda_l0_start_epoch default: active from the first structure phase
# ---------------------------------------------------------------------------

def test_l0_start_epoch_default_unchanged(tmp_path):
    ctrl = _make_controller(tmp_path, structure={"lambda_l0": 1e-4})
    assert ctrl.struct_l0_start_epoch == 0
    module = _FakeModule(lambda_l0=0.0)
    ctrl._apply_phase(_FakeTrainer(epoch=0), module, "structure")
    assert module.lambda_l0 == pytest.approx(1e-4)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

    def save_checkpoint(self, path):
        pass


def _make_controller(tmp_path, **ad_overrides):
    ad = {
        "monitor": "val_x_mae",
        "start_phase": "structure",
        "total_epoch_budget": 100,
        "reconstruct": {},
        "structure": {},
    }
    ad.update(ad_overrides)
    config = {"adaptive_training": ad,
              "model": {"model_object": "AttentionSelectorLayer"}}
    return PhaseController(config=config, data_dir=str(tmp_path),
                           save_dir=str(tmp_path), cluster=True)
