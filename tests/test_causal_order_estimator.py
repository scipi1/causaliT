"""Tests for the raw residual-HSIC causal-order estimator."""

import pytest
import torch

from causaliT.utils.causal_order import (
    build_offdiagonal_pair_mask,
    estimate_hsic_causal_order,
)


class TestOffdiagonalMask:
    def test_homogeneous_square_diagonal(self):
        mask = build_offdiagonal_pair_mask(3, 3, diagonal_offset=0)
        expected = torch.ones(3, 3)
        expected.fill_diagonal_(0.0)
        torch.testing.assert_close(mask, expected)

    def test_split_diagonal_uses_s_offset(self):
        # Three X targets over [S1, S2, X1, X2, X3] candidate sources.
        mask = build_offdiagonal_pair_mask(3, 5, diagonal_offset=2)
        assert torch.all(mask[:, :2] == 1.0)
        assert mask[0, 2].item() == 0.0
        assert mask[1, 3].item() == 0.0
        assert mask[2, 4].item() == 0.0
        assert mask.sum().item() == 12.0

    def test_pure_cross_layout_can_disable_diagonal(self):
        mask = build_offdiagonal_pair_mask(3, 4, diagonal_offset=None)
        assert torch.all(mask == 1.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_offdiagonal_pair_mask(0, 3)


def _chain_data(n=1024):
    """Ideal ANM residuals for a chain X0 -> X1 -> X2."""
    gen = torch.Generator().manual_seed(7)
    e0 = torch.randn(n, generator=gen)
    e1 = torch.randn(n, generator=gen)
    e2 = torch.randn(n, generator=gen)
    x0 = e0
    x1 = torch.tanh(1.5 * x0) + 0.10 * e1
    x2 = torch.tanh(1.5 * x1) + 0.10 * e2
    x = torch.stack([x0, x1, x2], dim=1)
    residuals = torch.stack([e0, e1, e2], dim=1)
    return x, residuals


class TestCausalOrderEstimator:
    def test_pair_matrix_excludes_diagonal_and_is_detached(self):
        x, residuals = _chain_data(n=256)
        x.requires_grad_(True)
        residuals.requires_grad_(True)
        out = estimate_hsic_causal_order(
            x,
            residuals,
            method="row_mean",
            adaptive_bandwidth=True,
            return_pair_matrix=True,
        )
        assert not out.scores.requires_grad
        assert not out.pair_matrix.requires_grad
        assert torch.isnan(out.pair_matrix.diagonal()).all()
        assert torch.isfinite(out.pair_matrix[~torch.eye(3, dtype=torch.bool)]).all()

    def test_split_layout_excludes_own_x_column(self):
        gen = torch.Generator().manual_seed(0)
        s = torch.randn(128, 2, generator=gen)
        x = torch.randn(128, 3, generator=gen)
        source = torch.cat([s, x], dim=1)
        residuals = torch.randn(128, 3, generator=gen)
        out = estimate_hsic_causal_order(
            source,
            residuals,
            diagonal_offset=2,
            return_pair_matrix=True,
        )
        assert out.pair_matrix.shape == (3, 5)
        assert torch.isnan(out.pair_matrix[0, 2])
        assert torch.isnan(out.pair_matrix[1, 3])
        assert torch.isnan(out.pair_matrix[2, 4])
        assert torch.isfinite(out.pair_matrix[:, :2]).all()

    def test_chain_root_scores_above_sink(self):
        x, residuals = _chain_data()
        out = estimate_hsic_causal_order(
            x,
            residuals,
            method="row_mean",
            adaptive_bandwidth=True,
            mode="biased",
        )
        assert out.scores[0] > out.scores[2]
        assert out.order[0].item() == 0
        assert out.order[-1].item() == 2
        assert out.ranks[0].item() == 0
        assert out.ranks[2].item() == 2

    def test_normalized_mode_runs(self):
        x, residuals = _chain_data(n=256)
        out = estimate_hsic_causal_order(
            x,
            residuals,
            adaptive_bandwidth=True,
            mode="normalized",
        )
        assert out.scores.shape == (3,)
        assert torch.isfinite(out.scores).all()

    def test_iterative_sink_reverses_sink_sequence(self):
        x, residuals = _chain_data()
        out = estimate_hsic_causal_order(
            x,
            residuals,
            method="iterative_sink",
            adaptive_bandwidth=True,
        )
        assert out.sink_sequence.shape == (3,)
        assert torch.equal(out.order, torch.flip(out.sink_sequence, dims=(0,)))
        # On the ideal chain, the fixed-residual heuristic should find the
        # sink first and the root last.
        assert out.sink_sequence[0].item() == 2
        assert out.order[0].item() == 0

    def test_iterative_sink_requires_square_layout(self):
        source = torch.randn(64, 4)
        residuals = torch.randn(64, 3)
        with pytest.raises(ValueError, match="square homogeneous"):
            estimate_hsic_causal_order(
                source,
                residuals,
                method="iterative_sink",
                diagonal_offset=1,
            )

    def test_permutation_null_is_reproducible_and_bounded(self):
        x, residuals = _chain_data(n=192)
        kwargs = dict(
            adaptive_bandwidth=True,
            null_permutations=3,
            null_seed=123,
        )
        first = estimate_hsic_causal_order(x, residuals, **kwargs)
        second = estimate_hsic_causal_order(x, residuals, **kwargs)
        torch.testing.assert_close(first.scores, second.scores)
        torch.testing.assert_close(first.null_mean, second.null_mean)
        torch.testing.assert_close(first.zscores, second.zscores)
        torch.testing.assert_close(first.p_values, second.p_values)
        assert torch.all(first.p_values > 0.0)
        assert torch.all(first.p_values <= 1.0)
        assert torch.isfinite(first.zscores).all()

    def test_invalid_method_and_null_count_raise(self):
        x, residuals = _chain_data(n=64)
        with pytest.raises(ValueError, match="method"):
            estimate_hsic_causal_order(x, residuals, method="unknown")
        with pytest.raises(ValueError, match="null_permutations"):
            estimate_hsic_causal_order(x, residuals, null_permutations=-1)
