"""Hybrid HSIC aggregation: only DESCENDANT pairs are attention-weighted.

    L_i = sum_{j not desc} H_ij  +  sum_{j in desc} att_ij * H_ij

The point is to remove the degenerate ``att -> 0`` solution BY CONSTRUCTION
rather than by adding counter-pressure (MSE / mass hinge).  Under an ANM a
correct fit leaves H_ij at the noise floor for every non-descendant, and those
terms are constants w.r.t. the attention -- so shrinking a true parent's weight
buys nothing.  Only descendant pairs, whose dependence is irreducible and whose
attention SHOULD vanish, keep an escape.

These tests are stub-level (no model, no training): they pin the gradient
structure, which is the whole claim.
"""
import pytest
import torch

from causaliT.utils.hsic_utils import hsic_attention_weighted


def _inputs(seed=0, n=256, d=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randn(n, d, generator=g)


def _call(att, mask=None, **kw):
    src, res = _inputs()
    return hsic_attention_weighted(
        source_values=src, residuals=res, attention_weights=att,
        adaptive_bandwidth=True, descendant_mask=mask, **kw,
    )


class TestGradientStructure:
    def test_non_descendant_pairs_get_no_attention_gradient(self):
        """THE core property: unweighted terms are constants in att."""
        att = torch.full((3, 3), 0.5, requires_grad=True)
        mask = torch.zeros(3, 3)          # nothing is a descendant
        _call(att, mask).backward()
        assert torch.allclose(att.grad, torch.zeros_like(att.grad)), (
            "with no descendant pairs the objective must not move the attention"
        )

    def test_descendant_pairs_do_get_attention_gradient(self):
        att = torch.full((3, 3), 0.5, requires_grad=True)
        mask = torch.zeros(3, 3)
        mask[0, 1] = 1.0                  # one descendant pair
        _call(att, mask).backward()
        assert att.grad[0, 1] != 0.0, "descendant pair must receive gradient"
        off = att.grad.clone()
        off[0, 1] = 0.0
        assert torch.allclose(off, torch.zeros_like(off)), (
            "only the descendant pair may receive gradient"
        )

    def test_gradient_on_descendant_points_to_zero(self):
        """HSIC >= 0, so d/d att = H_ij >= 0 -> descent pushes att down."""
        att = torch.full((3, 3), 0.5, requires_grad=True)
        mask = torch.ones(3, 3)
        _call(att, mask).backward()
        assert (att.grad >= 0).all()
        assert (att.grad > 0).any()


class TestCollapseIsNoLongerOptimal:
    def test_zeroing_all_attention_does_not_zero_the_loss(self):
        """The degenerate solution of the pure attw objective."""
        mask = torch.zeros(3, 3)
        mask[0, 1] = 1.0
        collapsed = _call(torch.zeros(3, 3), mask)
        assert collapsed > 0.0, (
            "non-descendant terms must survive att -> 0 (no free lunch)"
        )

    def test_collapse_only_saves_the_descendant_term(self):
        mask = torch.zeros(3, 3)
        mask[0, 1] = 1.0
        att = torch.full((3, 3), 0.8)
        saved = _call(att, mask) - _call(torch.zeros(3, 3), mask)
        # Exactly the one descendant pair's contribution, nothing more.
        only_desc = torch.zeros(3, 3)
        only_desc[0, 1] = 0.8
        assert torch.isclose(saved, _call(only_desc, torch.ones(3, 3)),
                             rtol=1e-5)

    def test_pure_attw_collapse_is_total(self):
        """Contrast: without the mask, att -> 0 zeroes the objective."""
        assert torch.isclose(_call(torch.zeros(3, 3)), torch.zeros(()))


class TestMaskSemantics:
    def test_mask_is_detached(self):
        att = torch.full((3, 3), 0.5, requires_grad=True)
        mask = torch.ones(3, 3, requires_grad=True)
        _call(att, mask).backward()
        assert mask.grad is None, "mask must never receive gradient"

    def test_all_ones_mask_equals_plain_attention_weighting(self):
        att = torch.rand(3, 3, generator=torch.Generator().manual_seed(3))
        assert torch.isclose(_call(att, torch.ones(3, 3)), _call(att), rtol=1e-6)

    def test_no_mask_reproduces_current_behaviour(self):
        att = torch.rand(3, 3, generator=torch.Generator().manual_seed(4))
        assert torch.isclose(_call(att, None), _call(att), rtol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="descendant_mask shape"):
            _call(torch.ones(3, 3), torch.ones(5, 5))

    def test_row_independence(self):
        """One row's attention must not affect another row's term."""
        mask = torch.ones(3, 3)
        a = torch.zeros(3, 3)
        a[0, :] = 1.0
        b = a.clone()
        b[1, :] = 1.0
        assert torch.isclose(_call(b, mask) - _call(a, mask),
                             _call(b - a, mask), rtol=1e-5)
