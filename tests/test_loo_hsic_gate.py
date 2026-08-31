"""Tests for the LOO-HSIC null calibration and Bayes multiplier (Stage 1).

Covers ``hsic_null_calibration`` and ``bayes_multiplier`` from
``causaliT/utils/hsic_utils.py`` — see
``docs/ideas/CONDITIONAL_HSIC_COUNTERPROPOSAL.md`` Section 3.
"""

import math

import torch

from causaliT.utils.hsic_utils import bayes_multiplier, hsic_null_calibration

B = 256
PERMS = 100
TRIALS = 60


def _calibrate(x, y, gen):
    return hsic_null_calibration(
        x, y, adaptive_bandwidth=True, n_permutations=PERMS, generator=gen
    )


def test_null_pvalues_approximately_uniform():
    """Independent x, y: empirical p-values should be ~Uniform(0, 1)."""
    gen = torch.Generator().manual_seed(0)
    ps = []
    for _ in range(TRIALS):
        x = torch.randn(B, generator=gen)
        y = torch.randn(B, generator=gen)
        ps.append(float(_calibrate(x, y, gen)["p_empirical"]))
    ps = torch.tensor(ps)
    assert 0.35 < ps.mean().item() < 0.65, f"mean p = {ps.mean():.3f}"
    frac_small = (ps < 0.05).float().mean().item()
    assert frac_small < 0.20, f"frac(p<0.05) = {frac_small:.3f}"


def test_power_under_dependence():
    """Nonlinear dependence y = x^2: p-values should be small."""
    gen = torch.Generator().manual_seed(1)
    ps = []
    for _ in range(TRIALS):
        x = torch.randn(B, generator=gen)
        y = x * x + 0.1 * torch.randn(B, generator=gen)
        ps.append(float(_calibrate(x, y, gen)["p_empirical"]))
    ps = torch.tensor(ps)
    assert ps.median().item() <= 0.05, f"median p = {ps.median():.4f}"


def test_gamma_fit_tracks_empirical_pvalue():
    """The moment-matched gamma p should correlate with the permutation p."""
    gen = torch.Generator().manual_seed(2)
    diffs = []
    for t in range(TRIALS):
        x = torch.randn(B, generator=gen)
        if t % 2:
            y = x * x + 0.1 * torch.randn(B, generator=gen)
        else:
            y = torch.randn(B, generator=gen)
        r = _calibrate(x, y, gen)
        diffs.append(abs(float(r["p_gamma"]) - float(r["p_empirical"])))
    assert sorted(diffs)[len(diffs) // 2] < 0.15, f"median |dp| = {sorted(diffs)[len(diffs)//2]:.3f}"


def test_calibration_is_detached_and_finite():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(64, generator=gen, requires_grad=True)
    y = torch.randn(64, generator=gen, requires_grad=True)
    r = _calibrate(x, y, gen)
    for k in ("stat", "p_empirical", "p_gamma", "log_q0", "alpha", "beta"):
        v = r[k]
        assert isinstance(v, torch.Tensor) and not v.requires_grad, k
        assert torch.isfinite(v).all(), k


def test_bayes_multiplier_equal_likelihoods_returns_prior():
    """H^{-i} == H^{+} (redundant edge) must leave the prior untouched."""
    p_m = torch.tensor([0.2, 0.5, 0.9])
    g = bayes_multiplier(torch.zeros(3), torch.zeros(3), p_m)
    assert torch.allclose(g, p_m, atol=1e-5)


def test_bayes_multiplier_load_bearing_edge_locks():
    """log q0(H^{-i}) << log q0(H^{+}): the edge is load-bearing -> gamma -> 1."""
    g = bayes_multiplier(torch.tensor(0.0), torch.tensor(-40.0), torch.tensor(0.3))
    assert g.item() > 0.999


def test_bayes_multiplier_harmful_edge_suppressed():
    """log q0(H^{-i}) >> log q0(H^{+}): removal is better -> gamma -> 0."""
    g = bayes_multiplier(torch.tensor(-40.0), torch.tensor(0.0), torch.tensor(0.7))
    assert g.item() < 1e-6


def test_bayes_multiplier_monotone_in_prior():
    lqp, lqm = torch.tensor(-1.0), torch.tensor(-3.0)
    p = torch.linspace(0.05, 0.95, 19)
    g = bayes_multiplier(lqp.expand(19), lqm.expand(19), p)
    assert bool((g[1:] >= g[:-1]).all()), "gamma must be monotone in P_m"


def test_bayes_multiplier_is_detached():
    g = bayes_multiplier(
        torch.tensor(0.0, requires_grad=True),
        torch.tensor(-1.0, requires_grad=True),
        torch.tensor(0.5, requires_grad=True),
    )
    assert not g.requires_grad
    assert 0.0 <= g.item() <= 1.0

# ---------------------------------------------------------------------------
# Stage 2/3: forecaster-side LOO gamma gate
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)


class _FakeLinearModel(torch.nn.Module):
    """Minimal stand-in for AttentionSelectorLayer.

    Predicts each target as a fixed linear combination of the sources:
    ``pred = (W * mask) @ x`` — cutting column i removes exactly that term,
    which is what the LOO measurement needs to detect.
    """

    def __init__(self, W):
        super().__init__()
        self.W = W  # (n_targets, n_sources)

    def forward_with_actual(
        self, source_tensor, x_blanked, x_actual,
        oracle=False, oracle_combined_mask=None, s_blanked=None,
    ):
        A = self.W if oracle_combined_mask is None else self.W * oracle_combined_mask
        pred = (A @ x_actual.squeeze(-1).T).T.unsqueeze(-1)
        return pred, None, None, None


class _LooStub(SimpleNamespace):
    """Attribute stub that also binds the forecaster's LOO helper methods."""

    _loo_measure_forward = AttentionSelectorForecaster._loo_measure_forward
    _loo_gamma_loop = AttentionSelectorForecaster._loo_gamma_loop


def _loo_stub(W, **overrides):
    base = dict(
        model=_FakeLinearModel(W),
        homogeneous_nodes=True,
        val_idx=0,
        loo_gamma_topk=None,
        loo_gamma_permutations=30,
    )
    base.update(overrides)
    return _LooStub(**base)


def _chain_data(B=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(B, generator=g)
    x2 = 0.8 * x1 + 0.6 * torch.randn(B, generator=g)
    x3 = torch.randn(B, generator=g)
    X = torch.stack([x1, x2, x3], dim=1).unsqueeze(-1)  # (B, 3, 1)
    return X, X.clone()


def test_compute_loo_gamma_detects_load_bearing_edge():
    # Truth: X1 -> X2 ; X3 isolated.  Model has learned the true edge.
    W = torch.zeros(3, 3)
    W[1, 0] = 0.8
    stub = _loo_stub(W)
    S, X = _chain_data()
    gamma = AttentionSelectorForecaster._compute_loo_gamma(stub, S, X, X)
    assert gamma.shape == (3, 3)
    assert not gamma.requires_grad
    assert bool(((gamma >= 0) & (gamma <= 1)).all())
    # Load-bearing edge X1->X2 (row target 1, col source 0) must dominate.
    assert gamma[1, 0] > 0.8
    # Irrelevant source X3 -> X2 must stay near the prior (p_m = 0.5 default).
    assert gamma[1, 2] < 0.8
    assert gamma[1, 0] > gamma[1, 2]


def test_compute_loo_gamma_topk_leaves_unmeasured_at_one():
    W = torch.zeros(3, 3)
    W[1, 0] = 0.8
    stub = _loo_stub(W, loo_gamma_topk=1)
    S, X = _chain_data()
    gamma = AttentionSelectorForecaster._compute_loo_gamma(stub, S, X, X)
    # Only one column per row measured; all others remain at the neutral 1.
    measured = (gamma != 1.0)
    assert int(measured.sum()) <= 3  # at most one entry per row


def test_compute_loo_gamma_restores_training_mode():
    W = torch.zeros(3, 3)
    stub = _loo_stub(W)
    stub.model.train()
    S, X = _chain_data(B=64)
    AttentionSelectorForecaster._compute_loo_gamma(stub, S, X, X)
    assert stub.model.training


def _gamma_stub(**overrides):
    base = dict(
        loo_gamma_refresh=1,
        loo_gamma_ema=0.9,
        _loo_gamma_cache=None,
        _loo_gamma_step=0,
        _last_loo_gamma_mean=1.0,
        _last_loo_gamma_min=1.0,
        calls=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_compute(value):
    def _impl(self, S, X, x_target):
        self.calls += 1
        return torch.full((2, 2), value)
    return _impl


def test_maybe_update_recomputes_on_refresh_cadence():
    stub = _gamma_stub(loo_gamma_refresh=2)
    stub._compute_loo_gamma = _fake_compute(0.5).__get__(stub)
    fn = AttentionSelectorForecaster._maybe_update_loo_gamma
    g1 = fn(stub, None, None, None, "train")  # step 1: no recompute -> None
    assert g1 is None and stub.calls == 0
    g2 = fn(stub, None, None, None, "train")  # step 2: refresh
    assert g2 is not None and stub.calls == 1
    assert torch.allclose(g2, torch.full((2, 2), 0.5))


def test_maybe_update_ema_smooths_and_val_reuses_cache():
    stub = _gamma_stub(loo_gamma_ema=0.5)
    vals = iter([1.0, 0.0])
    def _impl(self, S, X, x_target):
        self.calls += 1
        return torch.full((2, 2), next(vals))
    stub._compute_loo_gamma = _impl.__get__(stub)
    fn = AttentionSelectorForecaster._maybe_update_loo_gamma
    g1 = fn(stub, None, None, None, "train")
    assert torch.allclose(g1, torch.ones(2, 2))
    g2 = fn(stub, None, None, None, "train")
    assert torch.allclose(g2, torch.full((2, 2), 0.5))  # EMA of 1.0 and 0.0
    g3 = fn(stub, None, None, None, "val")   # no recompute on val
    assert torch.allclose(g3, torch.full((2, 2), 0.5)) and stub.calls == 2


def test_maybe_update_failure_keeps_previous_cache():
    stub = _gamma_stub(loo_gamma_ema=0.0)
    def _ok(self, S, X, x_target):
        return torch.full((2, 2), 0.7)
    def _boom(self, S, X, x_target):
        raise RuntimeError("masked forward exploded")
    fn = AttentionSelectorForecaster._maybe_update_loo_gamma
    stub._compute_loo_gamma = _ok.__get__(stub)
    g1 = fn(stub, None, None, None, "train")
    assert torch.allclose(g1, torch.full((2, 2), 0.7))
    stub._compute_loo_gamma = _boom.__get__(stub)
    g2 = fn(stub, None, None, None, "train")  # failure: previous cache kept
    assert torch.allclose(g2, torch.full((2, 2), 0.7))
