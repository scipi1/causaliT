"""Fast tests for the optional post-warmup initializer."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from causaliT.training.order_initialization import initialize_order_from_hsic, scores_from_hsic


def test_scores_and_guards():
    h = torch.tensor([1., 3., 2.])
    r = scores_from_hsic(h, 0.05, 1e-6)
    assert r.mean().abs() < 1e-7
    assert torch.allclose(r.std(unbiased=False), torch.tensor(0.05))
    assert torch.equal(h.argsort(), r.argsort())
    assert scores_from_hsic(torch.ones(3), 1., 1e-6) is None
    assert scores_from_hsic(torch.tensor([float('nan'), 1.]), 1., 1e-6) is None
    with pytest.raises(ValueError):
        scores_from_hsic(h, -1., 1e-6)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.order_scores = nn.Parameter(torch.zeros(3))
        self.order_path = 'hard'
        self.model = SimpleNamespace(order_attention=lambda: self)
        self.homogeneous_nodes = True
        self.val_idx = 0
        self.hsic_sigma = 1.
        self.hsic_adaptive_bandwidth = True
        self.hsic_mode = 'normalized'
        self.nhsic_epsilon = 0.01
        self.hsic_kernel_source = 'rbf'

    def set_order_path(self, mode):
        self.order_path = mode

    def forward(self, data_source, data_intermediate):
        assert not self.training and self.order_path == 'relaxed'
        target = torch.cat([data_source[:, :, 0], data_intermediate[:, :, 0]], 1)
        return torch.zeros_like(target).unsqueeze(-1), None, None


def test_probe_once_and_restore(tmp_path):
    torch.manual_seed(12)
    values = torch.randn(32, 3, 1)
    values[:, 1] = values[:, 0] + 0.1 * values[:, 1]
    loader = DataLoader(TensorDataset(values[:, :1], values[:, 1:]), batch_size=16)
    f = Probe()
    state = torch.random.get_rng_state().clone()
    path = tmp_path / 'report.json'
    report = initialize_order_from_hsic(f, loader, dict(batch_size=16, n_batches=2), path)
    assert report['status'] == 'applied'
    assert report['batch_sizes'] == [16, 16]
    assert f.training and f.order_path == 'hard'
    assert torch.equal(state, torch.random.get_rng_state())
    assert json.loads(path.read_text()) == report
    before = f.order_scores.clone()
    assert initialize_order_from_hsic(f, None, {}, path) is report
    assert torch.equal(before, f.order_scores)


def test_checkpoint_report_hooks():
    from causaliT.training.forecasters.attention_selector_forecaster import AttentionSelectorForecaster
    report = {'completed': True, 'status': 'applied'}
    # Exercise loading with a minimal checkpoint; legacy files leave the flag unset.
    obj = SimpleNamespace()
    AttentionSelectorForecaster.on_load_checkpoint(obj, {'order_initialization': report})
    assert obj._order_initialization_report == report
    AttentionSelectorForecaster.on_load_checkpoint(obj, {})
    assert obj._order_initialization_report is None


def test_probe_failure_restores_modes(tmp_path, monkeypatch):
    import causaliT.training.order_initialization as init
    f = Probe()
    data = torch.randn(8, 3, 1)
    loader = DataLoader(TensorDataset(data[:, :1], data[:, 1:]), batch_size=8)
    def fail(*args, **kwargs):
        raise RuntimeError('probe failure')
    monkeypatch.setattr(init, 'hsic_cross_per_pair', fail)
    with pytest.raises(RuntimeError, match='probe failure'):
        initialize_order_from_hsic(f, loader, {}, tmp_path / 'report.json')
    assert f.training and f.order_path == 'hard'
    assert not hasattr(f, '_order_initialization_report')
    assert torch.equal(f.order_scores, torch.zeros(3))