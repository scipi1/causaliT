"""Tests for ``AttentionSelectorForecaster.causal_predict`` (interventional roll-out).

Run with:  pytest tests/test_atsel_causal_predict.py -v

The roll-out is the model's GENERATIVE forward pass: instead of teacher-forcing
the X keys/values with observed data (training mode), the model's own
predictions are fed back until a fixed point.  Clamping a node every round is
the graph mutilation of a do-intervention.  See
docs/documentation/ATE_INTERVENTIONAL_ROLLOUT.md.

To make the dynamics exact and checkable we MONKEYPATCH ``forward`` on a real
forecaster instance with known structural equations, so the iteration,
clamping, convergence and noise machinery are tested independently of learning.

Guarantees under test
---------------------
1. A linear chain X1->X2->X3: the roll-out recovers the TOTAL effect while a
   single (teacher-forced) pass returns 0.
2. Clamped (mutilated) slots never move across rounds.
3. An acyclic graph converges EXACTLY (rollout_delta == 0) within L_X rounds.
4. A cyclic graph does NOT converge -> the rollout_delta diagnostic fires.
5. A DAG with no X->X edges reproduces the one-shot result exactly (back-compat).
6. Variant B (residual_pool) recovers the Jensen term E[(m+e)^2] = m^2 + Var(e)
   that variant A (deterministic) drops.
7. Common random numbers: identically-seeded generators give identical draws.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)

# Reuse the proven config builder from the homogeneous test-suite.
from tests.test_atsel_homogeneous import _make_forecaster_config

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Stub forecaster: real instance, monkeypatched forward with known equations
# ---------------------------------------------------------------------------


def _x_init(batch: int, x_len: int) -> torch.Tensor:
    """Zero-valued X state with 1-indexed variable IDs in the index column."""
    x = torch.zeros(batch, x_len, 2)
    for i in range(x_len):
        x[:, i, VAR_COL] = float(i + 1)
    return x


def _s_init(batch: int, s_len: int, value: float = 0.0) -> torch.Tensor:
    s = torch.zeros(batch, s_len, 2)
    s[:, :, VALUE_COL] = float(value)
    for i in range(s_len):
        s[:, i, VAR_COL] = float(i + 1)
    return s


def _make_stub(forward_fn, s_len: int = 3, x_len: int = 4):
    """Real forecaster (split mode) whose forward is replaced by ``forward_fn``.

    ``forward_fn(data_source, data_intermediate)`` must return a
    ``(pred_x, None, {})`` tuple with ``pred_x`` of shape (B, L_X, 1).
    """
    fc = AttentionSelectorForecaster(
        _make_forecaster_config(homogeneous_nodes=False)
    )
    fc.eval()

    def forward(data_source, data_intermediate):
        return forward_fn(data_source, data_intermediate)

    fc.forward = forward
    # Keep the stub's declared sequence lengths in sync with the tensors used.
    fc.S_seq_len = s_len
    fc.X_seq_len = x_len
    return fc


# ---------------------------------------------------------------------------
# 1. Linear chain: roll-out recovers the total effect, one-shot returns 0
# ---------------------------------------------------------------------------


class TestLinearChain:
    """X1 = 2*S1, X2 = 3*X1, X3 = 4*X2  (positions X1=0, X2=1, X3=2)."""

    @staticmethod
    def _eqns(data_source, data_intermediate):
        s = data_source[:, 0, VALUE_COL]          # S1
        x1 = data_intermediate[:, 0, VALUE_COL]
        x2 = data_intermediate[:, 1, VALUE_COL]
        pred = torch.stack(
            [2.0 * s, 3.0 * x1, 4.0 * x2, torch.zeros_like(s)], dim=-1
        )
        return pred.unsqueeze(-1), None, {}

    def test_rollout_recovers_total_effect(self):
        fc = _make_stub(self._eqns)
        B, s_val = 8, 0.5
        S = _s_init(B, 3, value=s_val)            # do(S1 = 0.5)
        x0 = _x_init(B, 4)
        x_final, _, delta = fc.causal_predict(S, x0)
        # Total effect: X3 = 4*X2 = 4*(3*X1) = 4*3*(2*S1) = 24*S1 = 12.0
        expected = torch.tensor([2 * s_val, 3 * 2 * s_val, 4 * 3 * 2 * s_val, 0.0])
        got = x_final[0, :3, VALUE_COL]
        assert torch.allclose(got, expected[:3], atol=1e-5)
        assert delta == pytest.approx(0.0, abs=1e-6)

    def test_one_shot_returns_zero(self):
        fc = _make_stub(self._eqns)
        B = 8
        S = _s_init(B, 3, value=0.5)
        x0 = _x_init(B, 4)
        # A single forward pass = one iteration: X3 still sees stale X2 = 0.
        x1, _, _ = fc.causal_predict(S, x0, n_iter=1)
        assert x1[0, 2, VALUE_COL].item() == pytest.approx(0.0, abs=1e-6)

    def test_converges_within_x_len_rounds(self):
        fc = _make_stub(self._eqns)
        S = _s_init(4, 3, value=1.0)
        x0 = _x_init(4, 4)
        _, rounds, delta = fc.causal_predict(S, x0)
        assert rounds <= 4                       # L_X rounds suffice for any DAG
        assert delta == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Clamping = graph mutilation
# ---------------------------------------------------------------------------


class TestClamp:
    @staticmethod
    def _eqns(data_source, data_intermediate):
        s = data_source[:, 0, VALUE_COL]
        x1 = data_intermediate[:, 0, VALUE_COL]
        pred = torch.stack(
            [2.0 * s, 3.0 * x1, torch.zeros_like(s), torch.zeros_like(s)], dim=-1
        )
        return pred.unsqueeze(-1), None, {}

    def test_clamped_slot_never_moves(self):
        fc = _make_stub(self._eqns)
        S = _s_init(4, 3, value=0.5)
        x0 = _x_init(4, 4)
        x_final, _, _ = fc.causal_predict(S, x0, clamp={0: 7.0})
        # do(X1 = 7): X1 pinned, downstream sees the intervened value.
        assert x_final[:, 0, VALUE_COL].eq(7.0).all()
        assert x_final[0, 1, VALUE_COL].item() == pytest.approx(3.0 * 7.0, abs=1e-5)

    def test_no_clamp_is_observational_run(self):
        fc = _make_stub(self._eqns)
        S = _s_init(4, 3, value=0.5)
        x0 = _x_init(4, 4)
        x_final, _, _ = fc.causal_predict(S, x0, clamp={})
        assert x_final[0, 0, VALUE_COL].item() == pytest.approx(2.0 * 0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 3/4. Convergence diagnostics: acyclic converges, cyclic does not
# ---------------------------------------------------------------------------


class TestConvergence:
    @staticmethod
    def _cyclic_eqns(data_source, data_intermediate):
        # X1 = X2 + 1, X2 = X1  -> 2-cycle, never converges.
        x1 = data_intermediate[:, 0, VALUE_COL]
        x2 = data_intermediate[:, 1, VALUE_COL]
        z = torch.zeros_like(x1)
        pred = torch.stack([x2 + 1.0, x1, z, z], dim=-1)
        return pred.unsqueeze(-1), None, {}

    def test_cyclic_graph_flags_nonzero_delta(self):
        fc = _make_stub(self._cyclic_eqns)
        S = _s_init(4, 3, value=0.0)
        x0 = _x_init(4, 4)
        _, _, delta = fc.causal_predict(S, x0)
        assert delta > 0.0                       # diagnostic fires


# ---------------------------------------------------------------------------
# 5. Back-compat: no X->X edges -> roll-out == one-shot
# ---------------------------------------------------------------------------


class TestBackCompat:
    @staticmethod
    def _eqns(data_source, data_intermediate):
        s = data_source[:, :, VALUE_COL]          # (B, L_S)
        o = torch.zeros_like(s[:, 0])
        # X_i depend only on S: X1=S1, X2=S2, X3=S3, X4=0.
        pred = torch.stack([s[:, 0], s[:, 1], s[:, 2], o], dim=-1)
        return pred.unsqueeze(-1), None, {}

    def test_matches_one_shot(self):
        fc = _make_stub(self._eqns)
        S = _s_init(4, 3, value=2.0)
        x0 = _x_init(4, 4)
        x_roll, _, delta = fc.causal_predict(S, x0)
        x_one, _, _ = fc.causal_predict(S, x0, n_iter=1)
        assert torch.allclose(
            x_roll[:, :, VALUE_COL], x_one[:, :, VALUE_COL], atol=1e-6
        )
        assert delta == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6/7. Variant B: residual bootstrap recovers the Jensen term; CRN
# ---------------------------------------------------------------------------


class TestResidualBootstrap:
    @staticmethod
    def _eqns(data_source, data_intermediate):
        # X1 = S1 (noise added by the pool), X2 = X1^2.
        s = data_source[:, 0, VALUE_COL]
        x1 = data_intermediate[:, 0, VALUE_COL]
        z = torch.zeros_like(s)
        pred = torch.stack([s, x1 * x1, z, z], dim=-1)
        return pred.unsqueeze(-1), None, {}

    def _pool(self, n: int, x_len: int) -> torch.Tensor:
        # Balanced +/-1 noise on X1, zero elsewhere -> Var(e1) = 1.
        pool = torch.zeros(n, x_len)
        pool[0::2, 0] = 1.0
        pool[1::2, 0] = -1.0
        return pool

    def test_variant_a_drops_jensen_term(self):
        fc = _make_stub(self._eqns)
        B, s_val = 2000, 0.3
        S = _s_init(B, 3, value=s_val)
        x0 = _x_init(B, 4)
        x_a, _, _ = fc.causal_predict(S, x0)     # no pool -> variant A
        # A propagates the mean: X2 = (S1)^2 exactly, no +Var(e).
        assert x_a[:, 1, VALUE_COL].mean().item() == pytest.approx(
            s_val * s_val, abs=1e-4
        )

    def test_variant_b_recovers_jensen_term(self):
        fc = _make_stub(self._eqns)
        B, s_val = 2000, 0.3
        S = _s_init(B, 3, value=s_val)
        x0 = _x_init(B, 4)
        pool = self._pool(64, 4)
        gen = torch.Generator().manual_seed(0)
        x_b, _, _ = fc.causal_predict(S, x0, residual_pool=pool, generator=gen)
        # B: X2 = (S1 + e)^2, mean = S1^2 + Var(e) = s^2 + 1.
        assert x_b[:, 1, VALUE_COL].mean().item() == pytest.approx(
            s_val * s_val + 1.0, abs=0.1
        )

    def test_common_random_numbers_reproducible(self):
        fc = _make_stub(self._eqns)
        S = _s_init(16, 3, value=0.3)
        x0 = _x_init(16, 4)
        pool = self._pool(64, 4)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        xa, _, _ = fc.causal_predict(S, x0, residual_pool=pool, generator=g1)
        xb, _, _ = fc.causal_predict(S, x0, residual_pool=pool, generator=g2)
        assert torch.equal(xa[:, :, VALUE_COL], xb[:, :, VALUE_COL])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
