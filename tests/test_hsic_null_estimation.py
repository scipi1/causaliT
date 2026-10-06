"""
Tests for the structural-objective null-distribution estimation
(``adaptive_training.structure.null_estimation``).

Covers:
1. The repurposed permutation machinery (``_hsic_permutation_replicates``)
   returns a genuine NULL distribution: with residuals independent of the
   sources, the replicates are finite and >= 0 (HSIC is non-negative) and the
   observed (unpermuted) statistic sits INSIDE the null (large empirical
   p-value); under dependence it lies far ABOVE the null mean.
2. Permutations come from a dedicated generator: reproducible for a fixed
   seed, and the global (training) RNG is untouched.
3. ``estimate_structural_null``: end-to-end through eval-mode forward passes
   on a synthetic loader; respects ``num_samples``; stores/logs
   ``hsic_null_stats``; computes no gradients; restores the training mode.
4. ``PhaseController`` parses the nested ``null_estimation`` config dict with
   backward-compatible defaults.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from causaliT.training.adaptive_trainer import PhaseController
from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic_cross_per_pair
from tests.test_atsel_bkd import _make_forecaster_config


def _make_fc() -> AttentionSelectorForecaster:
    return AttentionSelectorForecaster(_make_forecaster_config())


def _hsic_kw(fc) -> dict:
    """The HSIC hyper-parameters exactly as the forecaster passes them."""
    return dict(
        sigma=fc.hsic_sigma,
        adaptive_bandwidth=fc.hsic_adaptive_bandwidth,
        mode=fc.hsic_mode,
        nhsic_epsilon=fc.nhsic_epsilon,
        source_kernel=fc.hsic_kernel_source,
        bandwidth_multipliers=fc.hsic_bandwidth_multipliers,
    )


def _replicates(fc, src, res, n_perms=100, seed=123):
    gen = torch.Generator().manual_seed(seed)
    return fc._hsic_permutation_replicates(
        combined_source=src,
        residuals=res,
        attention_weights=None,
        bkd_keep_mask=None,
        hsic_pair_mask=None,
        n_perms=n_perms,
        generator=gen,
    )


def _observed(fc, src, res) -> float:
    return float(hsic_cross_per_pair(
        src, res, pair_mask=None, return_matrix=False, **_hsic_kw(fc),
    ))


class TestPermutationReplicates:
    """The repurposed function must return the null distribution."""

    def test_independent_data_observed_stat_inside_null(self):
        fc = _make_fc()
        torch.manual_seed(0)
        B, T, Ssrc = 128, 2, 4
        src = torch.randn(B, Ssrc)
        res = torch.randn(B, T)  # independent of the sources (H0 true)

        arr = np.asarray(_replicates(fc, src, res), dtype=np.float64)
        assert arr.size == 100
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)  # HSIC is strictly positive

        observed = _observed(fc, src, res)
        p_emp = (1.0 + (arr >= observed).sum()) / (arr.size + 1.0)
        assert p_emp > 0.05, "observed statistic rejected by its own null"

    def test_dependent_data_observed_stat_above_null(self):
        fc = _make_fc()
        torch.manual_seed(1)
        B, Ssrc = 128, 4
        src = torch.randn(B, Ssrc)
        # Residuals strongly dependent on the sources (H0 false).
        res = torch.stack([src[:, 0] ** 2, src[:, 1] * src[:, 2]], dim=1)
        res = res + 0.01 * torch.randn(B, 2)

        arr = np.asarray(_replicates(fc, src, res, seed=321), dtype=np.float64)
        observed = _observed(fc, src, res)
        assert observed > arr.mean() + 3.0 * arr.std(ddof=1), (
            "dependent statistic not separated from the null "
            f"(observed={observed:.3e}, null mean={arr.mean():.3e}, "
            f"std={arr.std(ddof=1):.3e})"
        )

    def test_generator_reproducible_and_global_rng_untouched(self):
        fc = _make_fc()
        torch.manual_seed(7)
        src = torch.randn(64, 3)
        res = torch.randn(64, 2)

        state0 = torch.random.get_rng_state()
        r1 = _replicates(fc, src, res, n_perms=20, seed=5)
        state1 = torch.random.get_rng_state()
        r2 = _replicates(fc, src, res, n_perms=20, seed=5)

        assert r1 == r2, "same generator seed must give identical replicates"
        assert torch.equal(state0, state1), "global (training) RNG was consumed"


def _make_loader(n=96, S_len=3, X_len=3, batch_size=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(n, S_len, 2)
    S[:, :, 0] = torch.randint(1, 8, (n, S_len), generator=g).float()
    S[:, :, 1] = torch.randn(n, S_len, generator=g)
    X = torch.zeros(n, X_len, 2)
    X[:, :, 0] = torch.randint(1, 8, (n, X_len), generator=g).float()
    X[:, :, 1] = torch.randn(n, X_len, generator=g)
    return DataLoader(TensorDataset(S, X), batch_size=batch_size, shuffle=False)


class TestEstimateStructuralNull:
    def test_end_to_end_stats(self):
        fc = _make_fc()
        fc.train()
        stats = fc.estimate_structural_null(
            _make_loader(), num_samples=40, seed=99,
        )
        assert stats is not None
        assert stats["n"] == 40  # num_samples respected
        assert stats["mean"] >= 0.0
        assert stats["std"] >= 0.0
        assert stats["distribution"] == "gaussian"
        assert fc.hsic_null_stats == stats  # stored on the module
        assert fc.training, "training mode must be restored"
        assert all(p.grad is None for p in fc.parameters()), (
            "null estimation must not compute gradients"
        )

    def test_seed_reproducible(self):
        torch.manual_seed(0)
        fc1 = _make_fc()
        torch.manual_seed(0)
        fc2 = _make_fc()
        s1 = fc1.estimate_structural_null(_make_loader(), num_samples=40, seed=99)
        s2 = fc2.estimate_structural_null(_make_loader(), num_samples=40, seed=99)
        assert s1["mean"] == pytest.approx(s2["mean"])
        assert s1["std"] == pytest.approx(s2["std"])

    def test_unsupported_distribution_falls_back(self):
        fc = _make_fc()
        stats = fc.estimate_structural_null(
            _make_loader(), num_samples=10, distribution="gamma",
        )
        assert stats is not None
        assert stats["distribution"] == "gaussian"


class TestConfigParsing:
    def _controller(self, structure_cfg):
        cfg = {"adaptive_training": {"structure": structure_cfg}}
        return PhaseController(
            config=cfg, data_dir=".", save_dir=".", cluster=True,
        )

    def test_defaults_disabled(self):
        ctrl = self._controller({})
        assert ctrl.null_est_enabled is False
        assert ctrl.null_est_num_samples == 100
        assert ctrl.null_est_distribution == "gaussian"

    def test_nested_dict_parsed(self):
        ctrl = self._controller({
            "null_estimation": {
                "run_bool": True,
                "num_samples": 100,
                "distribution": "gaussian",
            },
        })
        assert ctrl.null_est_enabled is True
        assert ctrl.null_est_num_samples == 100
        assert ctrl.null_est_distribution == "gaussian"

