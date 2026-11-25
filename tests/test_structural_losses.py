"""Unit tests of causaliT.training.structural_losses (config + pure losses)."""

import pytest
import torch

from causaliT.training.structural_losses import (
    AntiAdjacencyConfig,
    KernelOpts,
    anti_adjacency_loss,
    build_acyclicity_penalty,
    dhsic_residual_loss,
    resolve_hsic_objective,
)


def _data(n=64, N=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N, generator=g)
    x[:, 1] = x[:, 1] + torch.tanh(x[:, 0])
    return x, x + 0.3 * torch.randn(n, N, generator=g)


class TestConfig:
    def test_acyclicity_builder_defaults_and_validation(self):
        pen = build_acyclicity_penalty({})
        assert pen.name == "notears" and pen.s == "adaptive"
        with pytest.raises(ValueError, match="acyclicity_fn"):
            build_acyclicity_penalty({"acyclicity_fn": "bad"})
        with pytest.raises(ValueError, match="acyclicity_s"):
            build_acyclicity_penalty({"acyclicity_fn": "logdet", "acyclicity_s": -1})

    def test_acyclicity_penalty_zero_on_dag(self):
        A = torch.triu(torch.rand(4, 4), diagonal=1)
        for name in ("notears", "logdet", "nilpotent"):
            assert float(build_acyclicity_penalty({"acyclicity_fn": name})(A)) == pytest.approx(0.0, abs=1e-5)

    def test_hsic_objective(self):
        assert resolve_hsic_objective({}) == "pairwise"
        with pytest.raises(ValueError, match="hsic_objective"):
            resolve_hsic_objective({"hsic_objective": "nope"})

    def test_anti_adjacency_from_config(self):
        cfg = AntiAdjacencyConfig.from_config({"anti_adjacency_lambda": 0.5,
                                               "anti_adjacency_ratio_cap": 3})
        assert cfg.on and cfg.lam == 0.5 and cfg.ratio_cap == 3.0
        assert not AntiAdjacencyConfig.from_config({}).on

    @pytest.mark.parametrize("tr, match", [
        ({"anti_adjacency_lambda": -1}, "anti_adjacency_lambda"),
        ({"anti_adjacency_mode": "x"}, "anti_adjacency_mode"),
        ({"anti_adjacency_test": "joint"}, "anti_adjacency_test=joint"),
        ({"anti_adjacency_null": "per_pair", "hsic_mode": "normalized"}, "per_pair"),
        ({"anti_adjacency_ratio_cap": 0}, "ratio_cap"),
    ])
    def test_anti_adjacency_rejects_invalid(self, tr, match):
        with pytest.raises(ValueError, match=match):
            AntiAdjacencyConfig.from_config(tr)


class TestLosses:
    def test_dhsic_nonnegative_and_differentiable(self):
        _, r = _data()
        r = r.clone().requires_grad_(True)
        v = dhsic_residual_loss(r)
        v.backward()
        assert float(v) >= 0.0 and r.grad is not None

    def test_anti_adjacency_terms_and_stats(self):
        x, r = _data()
        N = x.shape[1]
        M = torch.full((N, N), 0.5, requires_grad=True)
        kern = KernelOpts(adaptive=True)
        both, st, paths = anti_adjacency_loss(
            x, r, M, torch.ones(N, N), AntiAdjacencyConfig(lam=1.0), kern)
        causal, st_c, _ = anti_adjacency_loss(
            x, r, M, torch.ones(N, N), AntiAdjacencyConfig(lam=1.0, terms="causal"), kern)
        assert paths == {}
        assert st_c["anticausal_log_term"] == 0.0
        assert float(both) == pytest.approx(st["causal_log_term"] + st["anticausal_log_term"])
        assert float(causal) == pytest.approx(st["causal_log_term"])
        both.backward()
        assert M.grad.abs().sum() > 0

    def test_surgery_paths_returned(self):
        x, r = _data()
        N = x.shape[1]
        M = torch.full((N, N), 0.5, requires_grad=True)
        _, _, paths = anti_adjacency_loss(
            x, r, M, torch.ones(N, N), AntiAdjacencyConfig(lam=1.0), KernelOpts(adaptive=True),
            surgery=True, anticausal_surgery=True)
        assert set(paths) == {"c_path", "c_weight", "c_residual", "a_weight", "a_residual"}


class TestLinearTransform:
    """anti_adjacency_transform=linear: mean of HSIC / nu, no log."""

    def _hard_order(self, N):
        # Fixed order 0 < 1 < ... < N-1: M[i, j] = 1 if i before j.
        return torch.triu(torch.ones(N, N), diagonal=1)

    def test_validation(self):
        assert AntiAdjacencyConfig.from_config({}).transform == "log_ratio"
        with pytest.raises(ValueError, match="anti_adjacency_transform"):
            AntiAdjacencyConfig.from_config({"anti_adjacency_transform": "bad"})

    def test_pairwise_equals_mean_of_ratios(self):
        from causaliT.utils.hsic_utils import (
            hsic_cross_per_pair, hsic_permutation_null_matrix)
        x, r = _data()
        N = x.shape[1]
        cfg = AntiAdjacencyConfig.from_config({
            "anti_adjacency_lambda": 1.0, "anti_adjacency_terms": "causal",
            "anti_adjacency_nd": "hard", "anti_adjacency_null": "per_pair",
            "anti_adjacency_transform": "linear"})
        kernel = KernelOpts(adaptive=True)
        M = self._hard_order(N)
        loss, _, _ = anti_adjacency_loss(x, r, M, torch.ones(N, N), cfg, kernel)
        H = hsic_cross_per_pair(x, r, **kernel.pair_kwargs())[1]   # H[t, s]
        nu = hsic_permutation_null_matrix(x, r, **kernel.null_kwargs()).clamp_min(1e-12)
        ref = torch.stack([H[j, i] / nu[j, i] for i in range(N)
                           for j in range(i + 1, N)]).mean()
        assert float(loss) == pytest.approx(float(ref), rel=1e-5)

    def test_joint_equals_mean_of_ratios(self):
        from causaliT.training.structural_losses import joint_nd_loss
        from causaliT.utils.hsic_utils import hsic_joint_masked
        x, r = _data()
        N = x.shape[1]
        cfg = AntiAdjacencyConfig.from_config({
            "anti_adjacency_lambda": 1.0, "anti_adjacency_terms": "causal",
            "anti_adjacency_nd": "hard", "anti_adjacency_test": "joint",
            "anti_adjacency_transform": "linear"})
        M = self._hard_order(N)
        loss, _, _ = joint_nd_loss(x, r, M, cfg, 1.0, True)
        T, nu, count = hsic_joint_masked(x, r, M.T, sigma=1.0, adaptive_bandwidth=True)
        valid = count > 0.5
        ref = (T / nu.clamp_min(1e-12))[valid].mean()
        assert int(valid.sum()) == N - 1
        assert float(loss) == pytest.approx(float(ref), rel=1e-5)
