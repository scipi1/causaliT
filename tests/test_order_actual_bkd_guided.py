"""M_actual and order-guided BKD of OrderFactorizedSelfAttention."""

import math

import pytest
import torch

from causaliT.core.modules.extra_layers import bkd_guide_fraction, sample_bkd_keep_matrix
from causaliT.core.modules.order_factorized_attention import OrderFactorizedSelfAttention

N, E, B = 5, 4, 3
GUIDE = {"enabled": True, "p_selected": 0.05, "p_unselected": 0.85,
         "start_epoch": 10, "anneal_epochs": 20}


def _qk(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    k = torch.randn(1, N, E, generator=g).expand(B, N, E).clone()
    v = torch.randn(B, N, 2, generator=g)
    return q, k, v


def _mod(**kw):
    torch.manual_seed(0)
    m = OrderFactorizedSelfAttention(num_nodes=N, order_r_init="random", **kw)
    m.train()
    return m


def _run(m, seed=1, stage="recon", osamp="none"):
    m.set_stage_policy(stage, gate_sampling="hard", order_sampling=osamp)
    m.resample(torch.Generator().manual_seed(seed))
    torch.manual_seed(123)
    out, _, _ = m(*_qk())
    return out.detach().clone()


def test_m_actual_without_gs_is_m():
    m = _mod()
    _run(m, stage="structure", osamp="soft")
    torch.testing.assert_close(m.last_M_actual, m._last_M.detach())


def test_m_actual_with_gs_is_pi_times_m():
    m = _mod(struct_gate_opt="gated")
    _run(m, stage="structure", osamp="hard_st")
    torch.testing.assert_close(m.last_M_actual, (m._last_pi * m._last_M).detach())
    hard = m.last_M_actual_hard
    assert torch.all((hard == 0) | (hard == 1))
    assert torch.all(hard <= (m._last_M > 0).float() + 1e-6)


def test_guide_disabled_or_zero_matches_uniform_bkd():
    base = _mod(batch_key_dropout=0.5)
    out0 = _run(base)
    keep0 = base.last_bkd_keep.clone()
    guided = _mod(batch_key_dropout=0.5, bkd_order_guided=GUIDE)
    guided.set_bkd_guide_epoch(0.0)
    out1 = _run(guided)
    assert guided.last_bkd_keep.dim() == 1
    torch.testing.assert_close(out0, out1)
    torch.testing.assert_close(keep0, guided.last_bkd_keep)


def test_cosine_ramp():
    assert bkd_guide_fraction(5, 10, 20) == 0.0
    assert bkd_guide_fraction(20, 10, 20) == pytest.approx(0.5)
    assert bkd_guide_fraction(30, 10, 20) == 1.0
    assert bkd_guide_fraction(50, 10, 20) == 1.0
    m = _mod(batch_key_dropout=0.5, bkd_order_guided=GUIDE)
    assert m.set_bkd_guide_epoch(15) == pytest.approx(0.5 * (1 - math.cos(math.pi * 0.25)))
    assert "bkd_guide_frac" in m.state_dict()


def test_guided_drop_endpoints():
    m = _mod(batch_key_dropout=0.5, bkd_order_guided=GUIDE)
    M = torch.tensor([[0.0, 1.0], [0.0, 0.0]])          # 0 -> 1 selected
    m.set_bkd_guide_epoch(0)
    assert torch.allclose(m.bkd_guided_drop(0.5, M), torch.full((2, 2), 0.5))
    m.set_bkd_guide_epoch(100)
    d = m.bkd_guided_drop(0.5, M)                         # child <- parent
    assert d[1, 0] == pytest.approx(0.05) and d[0, 1] == pytest.approx(0.85)


def test_keep_matrix_rates_and_min_keys():
    drop = torch.full((N, N), 0.85)
    drop[:, 0] = 0.05
    g = torch.Generator().manual_seed(0)
    keeps = torch.stack([sample_bkd_keep_matrix(drop, 0, g) for _ in range(4000)]).float()
    rate = 1.0 - keeps.mean(0)
    assert abs(rate[1:, 0].mean() - 0.05) < 0.02
    assert abs(rate[2:, 1].mean() - 0.85) < 0.02
    assert torch.all(keeps[:, torch.arange(N), torch.arange(N)] == 1)
    k = sample_bkd_keep_matrix(torch.ones(N, N) * 0.999, 2, g)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.all((k & off).sum(1) >= 2)


def test_guided_bkd_applied_per_pair_and_in_eval():
    m = _mod(batch_key_dropout=0.5, batch_key_dropout_eval=True, bkd_order_guided=GUIDE)
    m.set_bkd_guide_epoch(100)
    _run(m)
    keep = m.last_bkd_keep
    assert keep.shape == (N, N)
    A = m.last_applied_A[0]
    assert torch.all(A[keep == 0] == 0)
    m.eval()
    _run(m)
    assert m.last_bkd_keep is not None and m.last_bkd_keep.shape == (N, N)


def test_cosine_log_tau_anneal():
    from causaliT.core.modules.extra_layers import cosine_log_anneal
    assert cosine_log_anneal(0, 2.0, 0.2, 50, 100) == pytest.approx(2.0)
    assert cosine_log_anneal(100, 2.0, 0.2, 50, 100) == pytest.approx(math.sqrt(0.4))
    assert cosine_log_anneal(500, 2.0, 0.2, 50, 100) == pytest.approx(0.2)


def test_order_tau_schedule_parsing():
    import types
    from causaliT.training.forecasters.attention_selector_forecaster import (
        AttentionSelectorForecaster as F)
    sched = {"tau_start": 2.0, "tau_end": 0.2, "start_epoch": 50, "anneal_epochs": 300}
    for learnable, ok in ((False, True), (True, False)):
        oa = _mod(order_tau_learnable=learnable, order_tau=1.0,
                  order_tau_min=0.1, order_tau_max=2.0)
        fake = types.SimpleNamespace(config={"training": {"order_tau_schedule": sched}},
                                     model=types.SimpleNamespace(order_attention=lambda: oa))
        if ok:
            assert F._parse_order_tau_schedule(fake)["tau_end"] == 0.2
        else:
            with pytest.raises(ValueError):
                F._parse_order_tau_schedule(fake)


def test_schedule_clock_ignores_global_step():
    """Regression: with two optimizers global_step advances twice per
    batch; the schedule clock must use epoch + batch_idx / batches."""
    import types
    from causaliT.training.forecasters.attention_selector_forecaster import (
        AttentionSelectorForecaster as F)
    fake = types.SimpleNamespace(current_epoch=10, global_step=999,
                                 _trainer=types.SimpleNamespace(num_training_batches=4))
    assert F._fractional_epoch(fake, 0) == pytest.approx(10.0)
    assert F._fractional_epoch(fake, 2) == pytest.approx(10.5)
    fake._trainer.num_training_batches = float("inf")
    assert F._fractional_epoch(fake, 2) == pytest.approx(10.0)


def test_guided_bkd_hard_source_reaches_targets():
    """source=hard: drop is exactly p_sel / p_unsel on the step's sampled
    M_actual, even with a huge tau (flat soft M)."""
    captured = {}
    for src in ("hard", "soft"):
        m = _mod(batch_key_dropout=0.5, bkd_order_guided={**GUIDE, "source": src},
                 order_tau=50.0)
        m.set_bkd_guide_epoch(1e6)
        orig = m.bkd_guided_drop
        m.bkd_guided_drop = lambda p, M, o=orig: captured.__setitem__(src, o(p, M)) or captured[src]
        _run(m)
        hard = m.last_M_actual_hard.T
        off = ~torch.eye(N, dtype=torch.bool)
        d = captured[src][off]
        if src == "hard":
            torch.testing.assert_close(captured[src], torch.where(hard > 0.5, 0.05, 0.85)
                                       .masked_fill(~off, captured[src][0, 0]))
        else:
            assert d.min() > 0.3 and d.max() < 0.6     # flat soft M -> ~uniform


def test_invalid_guide_config():
    with pytest.raises(ValueError):
        _mod(batch_key_dropout=0.5, bkd_order_guided={**GUIDE, "source": "x"})
    with pytest.raises(ValueError):
        _mod(batch_key_dropout=0.5, bkd_order_guided={**GUIDE, "p_selected": 0.9})
    with pytest.raises(ValueError):
        _mod(bkd_order_guided=GUIDE)
