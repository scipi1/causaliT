"""Pin the removal of the ``/ sum(att)`` normalisation in hsic_attention_weighted.

The original implementation returned ``sum(att*H)/sum(att)``.  That is a WEIGHTED
MEAN, so:
  * its minimum over the attention simplex is ``min_ij H_ij``, attained at a
    one-hot attention -> "attend to a single already-independent pair" was a
    global optimum reachable by gradient descent, killing the fit for free;
  * a target row with NO parents has ``sum(att) ~ 0`` -> division by ~0, or the
    ``weight_sum > 1e-8`` guard returning a hard 0.0 with NO gradient.

It is now the unnormalised weighted sum (row-mean over targets).
"""
import torch

from causaliT.utils.hsic_utils import hsic_attention_weighted


def _fixed_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(256, 3, generator=g)
    res = torch.randn(256, 3, generator=g)
    return src, res


def _call(att, **kw):
    src, res = _fixed_inputs()
    return hsic_attention_weighted(
        source_values=src, residuals=res, attention_weights=att,
        adaptive_bandwidth=True, **kw,
    )


class TestNotAWeightedMean:
    def test_halving_all_attention_halves_the_objective(self):
        """A weighted MEAN is scale-invariant; the sum must NOT be."""
        att = torch.rand(3, 3, generator=torch.Generator().manual_seed(1)) + 0.1
        full = _call(att)
        half = _call(att * 0.5)
        assert torch.isclose(half, full * 0.5, rtol=1e-5), (
            "objective must scale with attention (normalisation removed)"
        )

    def test_shrinking_attention_lowers_it(self):
        """The documented consequence: callers MUST add a reconstruction term."""
        att = torch.rand(3, 3, generator=torch.Generator().manual_seed(2)) + 0.1
        assert _call(att * 0.01) < _call(att)

    def test_one_hot_is_not_automatically_optimal(self):
        """Under the old ratio, a one-hot on the min pair hit the global min."""
        _, res = _fixed_inputs()
        src, _ = _fixed_inputs()
        _, mat = hsic_attention_weighted(
            source_values=src, residuals=res,
            attention_weights=torch.ones(3, 3),
            adaptive_bandwidth=True, return_matrix=True,
        )
        i, j = divmod(int(torch.argmin(mat).item()), mat.shape[1])

        one_hot = torch.zeros(3, 3)
        one_hot[i, j] = 1.0
        # Same total mass spread over the same row: the ratio would have made
        # these EQUAL to min/that-pair; the sum ranks them by actual dependence.
        spread = torch.zeros(3, 3)
        spread[i, :] = 1.0 / 3.0
        assert not torch.isclose(_call(one_hot), _call(spread), rtol=1e-4)


class TestParentlessRows:
    def test_zero_attention_row_is_finite_and_contributes_nothing(self):
        """The dangerous case: a row with no parents divided by ~0 before."""
        att = torch.zeros(3, 3)
        att[0, :] = 0.5          # row 0 has parents
        # rows 1 and 2 are parentless
        out = _call(att)
        assert torch.isfinite(out), "parentless rows must not produce inf/NaN"
        assert out > 0.0

    def test_all_zero_attention_is_exactly_zero_and_differentiable(self):
        """Old code returned a detached 0.0 constant here (no gradient)."""
        att = torch.zeros(3, 3, requires_grad=True)
        out = _call(att)
        assert torch.isclose(out, torch.zeros(()))
        out.backward()
        assert att.grad is not None, "must stay on the autograd graph"
        assert torch.isfinite(att.grad).all()
        assert (att.grad != 0).any(), (
            "gradient must push back from the collapsed state"
        )

    def test_row_contribution_is_independent_of_other_rows_mass(self):
        """Under the ratio, one row's mass rescaled EVERY other row's term."""
        a = torch.zeros(3, 3)
        a[0, :] = 1.0
        b = a.clone()
        b[1, :] = 5.0            # add mass to a different row
        # Row 0's own contribution must be unchanged by row 1's mass.
        assert _call(b) > _call(a)
        assert torch.isclose(_call(b) - _call(a), _call(b - a), rtol=1e-5)
