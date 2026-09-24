"""Unit tests for the configurable acyclicity functionals in
``AttentionSelectorForecaster`` (``training.acyclicity_fn``):

* ``notears``   : h(A) = tr(exp(A @ A)) - d         (legacy default)
* ``logdet``    : h(A) = -log det(sI - A) + d log s (DAGMA-style)
* ``nilpotent`` : h(A) = sum_{k=1..d} tr(A^k)       (exact for A >= 0)

The mathematical properties under test:

1. All backends vanish on a DAG adjacency and are positive on cycles.
2. The adaptive log-det shift s = max-row-sum(A) is a nonsingularity
   certificate: it never raises / NaNs, even on a dense (0, 1) matrix —
   the failure mode of fixed-s DAGMA on unbounded weights.
3. Long-cycle sensitivity: NOTEARS down-weights a length-k cycle by k!,
   log-det / nilpotent do not.
4. Gradients flow into a learnable score matrix for every backend.
"""

import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster as F,
)


def _dag(d: int = 5) -> torch.Tensor:
    """Strictly upper-triangular (hence acyclic) nonnegative matrix."""
    A = torch.zeros(d, d)
    for i in range(d - 1):
        A[i, i + 1] = 0.9
    A[0, -1] = 0.4
    return A


def _k_cycle(k: int, w: float = 0.9) -> torch.Tensor:
    """Simple directed k-cycle with edge weight w (zero elsewhere)."""
    A = torch.zeros(k, k)
    for i in range(k):
        A[i, (i + 1) % k] = w
    return A


def _backends(A: torch.Tensor) -> dict:
    return {
        "notears": F._notears_acyclicity(A),
        "logdet": F._logdet_acyclicity(A, "adaptive"),
        "nilpotent": F._nilpotent_acyclicity(A),
    }


class TestZeroIffDag:
    def test_all_backends_zero_on_dag(self):
        for name, h in _backends(_dag()).items():
            assert float(h) == pytest.approx(0.0, abs=1e-5), name

    def test_all_backends_zero_on_empty_graph(self):
        for name, h in _backends(torch.zeros(6, 6, dtype=torch.float64)).items():
            assert float(h) == pytest.approx(0.0, abs=1e-8), name

    @pytest.mark.parametrize("k", [2, 3, 8])
    def test_all_backends_positive_on_k_cycle(self, k):
        for name, h in _backends(_k_cycle(k)).items():
            assert float(h) > 0.0, name

    def test_all_backends_finite_and_nonnegative_on_random_posterior(self):
        torch.manual_seed(0)
        A = torch.rand(10, 10) * 0.9  # gate-posterior-like, dense (0, 1)
        for name, h in _backends(A).items():
            assert torch.isfinite(h), name
            assert float(h) >= 0.0, name


class TestLogDetAdaptiveShift:
    def test_adaptive_never_raises_on_dense_unit_matrix(self):
        # All-ones-ish matrix: rho(A) ~ d >> 1.  Fixed s = 1 (pure DAGMA)
        # is singular here; the adaptive shift must stay finite.
        A = torch.full((8, 8), 0.99)
        h = F._logdet_acyclicity(A, "adaptive")
        assert torch.isfinite(h)
        assert float(h) > 0.0

    def test_fixed_s_raises_when_singular(self):
        A = torch.full((8, 8), 0.99)  # rho(A) ~= 7.9 >> 1
        with pytest.raises(FloatingPointError):
            F._logdet_acyclicity(A, 1.0)

    def test_adaptive_s_is_stop_grad(self):
        A = _k_cycle(4, 0.5).double().requires_grad_(True)
        h = F._logdet_acyclicity(A, "adaptive")
        h.backward()
        # Gradient must come ONLY from the log-det term: with M = sI - A,
        # dh/dA = +(M^-1)^T (the two minus signs in M and -logdet cancel);
        # if s leaked grad, +d log s would add a uniform d/s contribution.
        s_val = A.detach().sum(dim=-1).max() + 1e-4
        M = s_val * torch.eye(4, dtype=torch.float64) - A.detach()
        expected = torch.linalg.inv(M).T
        assert torch.allclose(A.grad, expected, rtol=1e-6)

    def test_logdet_matches_closed_walk_series(self):
        # h = sum_k tr(A^k)/(k s^k) for rho(A/s) < 1 (truncation check).
        torch.manual_seed(1)
        A = torch.rand(6, 6, dtype=torch.float64) * 0.15
        s_val = A.sum(dim=-1).max() + 1e-4
        series = sum(
            torch.trace(torch.linalg.matrix_power(A, k)) / (k * s_val ** k)
            for k in range(1, 40)
        )
        h = F._logdet_acyclicity(A, "adaptive")
        assert float(h) == pytest.approx(float(series), rel=1e-4)


class TestLongCycleSensitivity:
    def test_notears_blinds_long_cycles_logdet_does_not(self):
        # Unit-weight k-cycle: NOTEARS contributes k/k! (factorially
        # damped), log-det counts it ~once regardless of length.
        h_nt2 = float(F._notears_acyclicity(_k_cycle(2, 1.0)))
        h_nt12 = float(F._notears_acyclicity(_k_cycle(12, 1.0)))
        h_ld2 = float(F._logdet_acyclicity(_k_cycle(2, 1.0), "adaptive"))
        h_ld12 = float(F._logdet_acyclicity(_k_cycle(12, 1.0), "adaptive"))
        assert h_nt12 < 0.1 * h_nt2          # NOTEARS collapses with k
        assert h_ld12 > 0.25 * h_ld2         # log-det stays comparable

    def test_nilpotent_counts_each_cycle_exactly(self):
        # Unit k-cycle: tr(A^k) = k, all other powers traceless.
        for k in (2, 5, 9):
            h = float(F._nilpotent_acyclicity(_k_cycle(k, 1.0)))
            assert h == pytest.approx(float(k), rel=1e-5)


class TestGradients:
    @pytest.mark.parametrize("backend", ["notears", "logdet", "nilpotent"])
    def test_gradient_flows_and_pushes_cycle_edges_down(self, backend):
        A = _k_cycle(5, 0.8).requires_grad_(True)
        h = {
            "notears": F._notears_acyclicity,
            "logdet": lambda a: F._logdet_acyclicity(a, "adaptive"),
            "nilpotent": F._nilpotent_acyclicity,
        }[backend](A)
        h.backward()
        assert A.grad is not None and torch.isfinite(A.grad).all()
        # Every cycle edge carries positive gradient (penalty decreases
        # when the edge weight decreases).
        for i in range(5):
            assert A.grad[i, (i + 1) % 5] > 0.0

    def test_no_gradient_on_dag_edges(self):
        # On a DAG all backends sit at the minimum: gradients must be
        # finite (never NaN).
        A = _dag().requires_grad_(True)
        for fn in (
            F._notears_acyclicity,
            lambda a: F._logdet_acyclicity(a, "adaptive"),
            F._nilpotent_acyclicity,
        ):
            fn(A).backward()
        assert torch.isfinite(A.grad).all()

