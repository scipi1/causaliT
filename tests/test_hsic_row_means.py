"""Tests for per-row HSIC diagnostics (bilevel commit groundwork, Phase 0).

Covers ``hsic_row_means`` and the ``return_matrix`` flag of
``hsic_cross_per_pair`` in causaliT/utils/hsic_utils.py.
"""

import math

import torch

from causaliT.utils.hsic_utils import (
    hsic_cross_per_pair,
    hsic_pair_matrix,
    hsic_row_means,
)


torch.manual_seed(0)


def _data(batch=64, n_src=4, n_tgt=3):
    s = torch.randn(batch, n_src)
    res = torch.randn(batch, n_tgt)
    return s, res


class TestHsicRowMeans:
    def test_plain_rows_match_nanmean(self):
        mat = torch.rand(3, 4)
        rows = hsic_row_means(mat)
        assert torch.allclose(rows, mat.mean(dim=1))

    def test_nan_entries_are_dropped(self):
        mat = torch.rand(3, 4)
        mat[0, 2] = float("nan")
        rows = hsic_row_means(mat)
        expected0 = (mat[0, 0] + mat[0, 1] + mat[0, 3]) / 3.0
        assert torch.isclose(rows[0], expected0)
        assert torch.allclose(rows[1:], mat[1:].mean(dim=1))

    def test_fully_excluded_row_is_nan(self):
        mat = torch.rand(3, 4)
        mat[1, :] = float("nan")
        rows = hsic_row_means(mat)
        assert torch.isnan(rows[1])
        assert not torch.isnan(rows[0])

    def test_weighted_rows(self):
        mat = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1.0
        w = torch.zeros(3, 4)
        w[:, 0] = 1.0
        w[:, 3] = 2.0
        rows = hsic_row_means(mat, pair_mask=w)
        expected = (mat[:, 0] * 1.0 + mat[:, 3] * 2.0) / 3.0
        assert torch.allclose(rows, expected)

    def test_zero_weight_row_is_nan(self):
        mat = torch.rand(2, 3)
        w = torch.ones(2, 3)
        w[0] = 0.0
        rows = hsic_row_means(mat, pair_mask=w)
        assert torch.isnan(rows[0])
        assert torch.isclose(rows[1], mat[1].mean())

    def test_shape_mismatch_raises(self):
        mat = torch.rand(2, 3)
        try:
            hsic_row_means(mat, pair_mask=torch.ones(2, 2))
        except ValueError:
            return
        raise AssertionError("expected ValueError on pair_mask shape mismatch")


class TestReturnMatrix:
    def test_scalar_matches_default_call(self):
        s, res = _data()
        scalar = hsic_cross_per_pair(s, res)
        scalar2, mat = hsic_cross_per_pair(s, res, return_matrix=True)
        assert torch.isclose(scalar, scalar2)
        assert mat.shape == (3, 4)

    def test_matrix_matches_pair_matrix(self):
        s, res = _data()
        _, mat = hsic_cross_per_pair(s, res, return_matrix=True)
        ref = hsic_pair_matrix(source_values=s, residuals=res)
        assert torch.allclose(mat, ref, equal_nan=True)

    def test_scalar_is_plain_mean_of_matrix(self):
        s, res = _data()
        scalar, mat = hsic_cross_per_pair(s, res, return_matrix=True)
        assert torch.isclose(scalar, mat.mean())

    def test_weighted_scalar_and_rows_agree(self):
        """Global weighted mean == weight-normalised mean of the row means."""
        s, res = _data()
        w = torch.rand(3, 4)
        w[0, 0] = 0.0  # excluded pair
        scalar, mat = hsic_cross_per_pair(s, res, pair_mask=w, return_matrix=True)
        rows = hsic_row_means(mat, pair_mask=w)
        # Row 0 has a skipped pair -> NaN at [0, 0] in mat.
        assert torch.isnan(mat[0, 0])
        assert not torch.isnan(rows).any()
        # Rebuild the global weighted mean from the rows.
        w_rows = torch.zeros(3)
        valid = ~torch.isnan(mat)
        for j in range(3):
            w_rows[j] = w[j][valid[j]].sum()
        global_from_rows = (rows * w_rows).sum() / w_rows.sum()
        assert torch.isclose(scalar, global_from_rows)

    def test_all_masked_returns_zero_and_matrix(self):
        s, res = _data()
        w = torch.zeros(3, 4)
        scalar, mat = hsic_cross_per_pair(s, res, pair_mask=w, return_matrix=True)
        assert float(scalar) == 0.0
        assert torch.isnan(mat).all()
        assert torch.isnan(hsic_row_means(mat, pair_mask=w)).all()
