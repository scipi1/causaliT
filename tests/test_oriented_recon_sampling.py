"""Unit tests for oriented recon sampling
(docs/ideas/ORIENTED_RECON_SAMPLING.md)."""

import types

import pytest
import torch

from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention
from causaliT.training.adaptive_trainer import PhaseController
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)

N, E, B = 4, 4, 3


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    v = torch.randn(B, N, 2, generator=g)
    return q, k, v


def _mod():
    torch.manual_seed(0)
    m = OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random")
    m.train()
    m.set_order_path("hard")
    m.set_gate_mode("hard")
    return m


def _run(m, seed=1, orientation="causal"):
    m.set_orientation(orientation)
    m.resample(torch.Generator().manual_seed(seed))
    torch.manual_seed(123)
    out, _, _ = m(*_qk())
    return out.detach().clone(), m.last_applied_A[0].clone()


def test_none_reproduces_default():
    m = _mod()
    out0, A0 = _run(m)
    m.set_recon_noise_scale(None)
    out1, A1 = _run(m)
    assert torch.equal(out0, out1) and torch.equal(A0, A1)
    # Same scale as noise_scale -> identical as well.
    m.set_recon_noise_scale(m.noise_scale)
    out2, _ = _run(m)
    assert torch.equal(out0, out2)


def test_anticausal_row_is_transposed_order():
    m = _mod()
    m.set_recon_noise_scale(1.0)
    m.set_gate_mode("soft")
    _, A_c = _run(m, orientation="causal")
    _, A_a = _run(m, orientation="anticausal")
    # Recon order of the step (same noise draw).
    theta = m.theta()
    M_rec = m.order_mask(
        __import__("causaliT.core.modules.order_factorized_attention",
                   fromlist=["hard_assignment"]).hard_assignment(
            theta + 1.0 * m._get_noise()["gumbel"]))
    # last_applied_A is [target, source]; M_rec is [source, target].
    # Causal: weight only on predecessors; anticausal: only on successors.
    assert torch.all(A_c[M_rec.transpose(0, 1) == 0] == 0)
    assert torch.all(A_a[M_rec == 0] == 0)
    assert torch.allclose(A_a, A_c.transpose(0, 1))
    m.set_orientation("causal")


def test_structural_graph_unaffected_by_recon_scale():
    m = _mod()
    _run(m)
    post0 = m._last_M.clone()
    sparse0 = m.score_tensor_for_sparsity.detach().clone()
    m.set_recon_noise_scale(5.0)
    _run(m)
    assert torch.equal(post0, m._last_M)
    assert torch.equal(sparse0, m.score_tensor_for_sparsity.detach())
    with pytest.raises(ValueError):
        m.set_recon_noise_scale(-1.0)


def test_orientation_frequency():
    fake = types.SimpleNamespace(config={"training": {"seed": 1}})
    AttentionSelectorForecaster.set_recon_order_sampling(fake, True, 0.3)
    draws = [AttentionSelectorForecaster._draw_recon_orientation(fake)
             for _ in range(4000)]
    frac = sum(d == "anticausal" for d in draws) / len(draws)
    assert abs(frac - 0.3) < 0.03


@pytest.fixture
def tmp_path():
    # Local temp dir (the default pytest tmp root may be access-denied).
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _controller(tmp_path, **ad_overrides):
    ad = {"monitor": "val_x_mae", "start_phase": "structure",
          "total_epoch_budget": 100, "reconstruct": {}, "structure": {}}
    ad.update(ad_overrides)
    config = {"adaptive_training": ad,
              "model": {"model_object": "AttentionSelectorLayer"}}
    return PhaseController(config=config, data_dir=str(tmp_path),
                           save_dir=str(tmp_path), cluster=True)


def test_config_conflict_raises(tmp_path):
    with pytest.raises(ValueError):
        _controller(tmp_path, use_order_in_recon=True,
                    recon_order_sampling={"enabled": True})


def test_config_block_override(tmp_path):
    ctrl = _controller(tmp_path, recon_order_sampling={"enabled": True},
                       reconstruct={"recon_order_sampling": {"p_anticausal": 0.2}})
    ros = ctrl._recon_order_sampling_cfg("reconstruct")
    assert ros["enabled"] and ros["p_anticausal"] == 0.2 and ros["noise_scale"] == 1.0
