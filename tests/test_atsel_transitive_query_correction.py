"""Geometric transitive correction (grandparent suppression).

Covers
------
1. ``causaliT/utils/query_geometry.py`` — the trigger algebra, the init-safety
   property that replaces a warmup schedule, and the exactness of the component
   removal on an orthonormal frame.
2. The attention modules — the correction is applied where the probe applied it
   (on the unit query, before the norm budget) and is a strict no-op when off.
3. ``AttentionSelectorLayer`` — the two-pass orchestration in BOTH topologies
   (homogeneous single square block, split cross + self) and the fail-loud guard
   on a non-orthonormal key frame.

Reference numbers come from ``scripts/_probe_transitive_correction.py`` on
``baseline_equal_ds_centroid_init_9494866`` (see
``experiments/6_INVESTIGATIONS/HOMOGENEOUS/TRANSITIVE_CORRECTION.md``).
"""

import pytest
import torch

from causaliT.utils.query_geometry import (
    assert_orthonormal_frame,
    correct_query,
    frame_offdiag,
    key_frame,
    mediation_mass,
    transitive_weights,
)


# =============================================================================
# Fixtures: a 3-node chain  j -> k -> i  with a leaked indirect edge j -> i
# =============================================================================

@pytest.fixture
def chain_pi():
    """``pi[i, j] = P(j -> i)`` for the chain ``0 -> 1 -> 2`` (+ leak 0 -> 2).

    Values taken from the real X5 row: the two true edges sit at 0.80 / 0.65 and
    the indirect one leaked to 0.44.
    """
    pi = torch.zeros(3, 3)
    pi[1, 0] = 0.80        # 0 -> 1   true
    pi[2, 1] = 0.65        # 1 -> 2   true
    pi[2, 0] = 0.44        # 0 -> 2   INDIRECT (to be suppressed)
    return pi


# =============================================================================
# 1. Trigger algebra
# =============================================================================

def test_mediation_mass_min_keeps_the_weakest_link(chain_pi):
    """Goedel t-norm: the mediated pair keeps the confidence of its weakest edge."""
    m = mediation_mass(chain_pi, tnorm="min")
    assert m[2, 0] == pytest.approx(min(0.65, 0.80), abs=1e-6)   # 0.65
    # No mediator exists for the two true edges of a chain.
    assert m[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert m[2, 1] == pytest.approx(0.0, abs=1e-6)


def test_mediation_mass_prod_deflates(chain_pi):
    """The literal product is smaller than either edge -> a much weaker trigger."""
    m_prod = mediation_mass(chain_pi, tnorm="prod")
    m_min = mediation_mass(chain_pi, tnorm="min")
    assert m_prod[2, 0] == pytest.approx(0.65 * 0.80, abs=1e-6)   # 0.52
    assert m_prod[2, 0] < m_min[2, 0]
    # ... and the margin over the direct edge is what the correction acts on:
    assert (m_min[2, 0] - chain_pi[2, 0]) > 2 * (m_prod[2, 0] - chain_pi[2, 0])


def test_mediation_mass_has_zero_diagonal_and_rejects_non_square(chain_pi):
    m = mediation_mass(chain_pi)
    assert torch.allclose(m.diagonal(), torch.zeros(3))
    with pytest.raises(ValueError, match="square"):
        mediation_mass(torch.rand(3, 4))


def test_mediation_mass_batched_matches_per_sample(chain_pi):
    batch = torch.stack([chain_pi, chain_pi.roll(1, dims=0)])
    m_batch = mediation_mass(batch)
    for b in range(2):
        assert torch.allclose(m_batch[b], mediation_mass(batch[b]), atol=1e-6)


def test_transitive_weights_target_only_the_indirect_edge(chain_pi):
    w = transitive_weights(chain_pi, alpha=1.0, tnorm="min", margin=True,
                           symmetric=False)
    assert w[2, 0] == pytest.approx(0.65 - 0.44, abs=1e-6)   # the margin
    # The true edges are untouched: no mediator, hence no weight.
    assert w[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert w[2, 1] == pytest.approx(0.0, abs=1e-6)
    assert w.max() <= 1.0


def test_transitive_weights_alpha_scales_and_detaches(chain_pi):
    pi = chain_pi.clone().requires_grad_(True)
    w_half = transitive_weights(pi, alpha=0.5)
    w_full = transitive_weights(pi, alpha=1.0)
    assert torch.allclose(w_half, 0.5 * w_full, atol=1e-6)
    # A differentiable weight would let the model delete a real edge elsewhere
    # just to raise a logit -> the weights MUST be detached.
    assert not w_half.requires_grad


def test_transitive_weights_margin_is_silent_at_the_all_on_init():
    """Init safety (replaces a warmup schedule).

    At the centroid initialisation every edge sits at the same value, so no
    mediated path can beat a direct edge and the correction must stay silent.
    Without the margin gate the same state yields a LARGE weight.
    """
    pi = torch.full((6, 6), 0.418)
    pi.fill_diagonal_(0.0)
    w_margin = transitive_weights(pi, alpha=1.0, tnorm="min", margin=True)
    w_plain = transitive_weights(pi, alpha=1.0, tnorm="min", margin=False)
    assert float(w_margin.max()) < 0.01          # probe measured 0.0074
    assert float(w_plain.max()) > 0.4            # probe measured 0.4304


def test_transitive_weights_symmetric_span_limits_symmetrisation():
    """Split mode: only the square self block may be symmetrised."""
    pi = torch.zeros(4, 4)
    pi[3, 2], pi[2, 1], pi[3, 1] = 0.8, 0.7, 0.4      # mediated inside [1:4]
    full = transitive_weights(pi, alpha=1.0, symmetric=True)
    span = transitive_weights(pi, alpha=1.0, symmetric=True, symmetric_span=(2, 4))
    none = transitive_weights(pi, alpha=1.0, symmetric=False)
    # Inside the span the transpose is mirrored; outside it is left alone.
    assert span[2, 3] == pytest.approx(float(none[3, 2]), abs=1e-6)
    assert span[1, 3] == pytest.approx(float(none[1, 3]), abs=1e-6)
    assert full[1, 3] == pytest.approx(float(none[3, 1]), abs=1e-6)


# =============================================================================
# 2. Geometry: the removal is exact on an orthonormal frame
# =============================================================================

def _orthonormal_keys(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, n, generator=g))
    return q.T.contiguous()          # (n, d), mutually orthonormal rows


def test_correct_query_moves_only_the_targeted_coordinate():
    keys = _orthonormal_keys(5, 16)
    query = torch.randn(5, 16)
    w = torch.zeros(5, 5)
    w[4, 0] = 1.0                                     # kill edge 0 -> 4 only
    u0 = torch.nn.functional.normalize(query, dim=-1)
    c0 = u0 @ key_frame(keys).T
    c1 = correct_query(query, keys, w, delta=0.5) @ key_frame(keys).T
    # The targeted coordinate lands on -delta ...
    assert float(c1[4, 0]) == pytest.approx(-0.5, abs=1e-5)
    # ... and every other coordinate of that row is untouched (exactness).
    for j in range(1, 5):
        assert float(c1[4, j]) == pytest.approx(float(c0[4, j]), abs=1e-6)
    # Other rows are untouched entirely.
    assert torch.allclose(c1[:4], c0[:4], atol=1e-6)


def test_correct_query_delta_zero_is_the_plain_projection():
    keys = _orthonormal_keys(4, 12)
    query = torch.randn(4, 12)
    w = torch.zeros(4, 4)
    w[3, 1] = 1.0
    c = correct_query(query, keys, w, delta=0.0) @ key_frame(keys).T
    # Projection reaches 0 -> exactly the P(z>0) = 0.5 THRESHOLD, which is why
    # the default instrument pushes past zero instead.
    assert float(c[3, 1]) == pytest.approx(0.0, abs=1e-6)


def test_correct_query_zero_weights_is_a_noop_up_to_normalisation():
    keys = _orthonormal_keys(4, 12)
    query = torch.randn(4, 12)
    out = correct_query(query, keys, torch.zeros(4, 4), delta=0.5)
    assert torch.allclose(out, torch.nn.functional.normalize(query, dim=-1),
                          atol=1e-6)


def test_correct_query_frees_budget_for_the_true_parents():
    """The renormalisation reallocation: suppressing one edge RAISES the others."""
    keys = _orthonormal_keys(4, 8)
    kh = key_frame(keys)
    query = (0.6 * kh[0] + 0.6 * kh[1] + 0.5 * kh[2]).unsqueeze(0)
    w = torch.zeros(1, 4)
    w[0, 0] = 1.0
    c0 = torch.nn.functional.normalize(query, dim=-1) @ kh.T
    corrected = correct_query(query, keys, w, delta=0.5)
    c1 = torch.nn.functional.normalize(corrected, dim=-1) @ kh.T
    assert float(c1[0, 0]) < 0.0                       # suppressed
    assert float(c1[0, 1]) > float(c0[0, 1])           # true parents go UP
    assert float(c1[0, 2]) > float(c0[0, 2])


# =============================================================================
# 3. The frame guard
# =============================================================================

def test_frame_guard_accepts_an_orthonormal_frame():
    keys = _orthonormal_keys(6, 20)
    assert float(frame_offdiag(key_frame(keys))) < 1e-5
    assert assert_orthonormal_frame(keys) < 1e-5


def test_frame_guard_raises_on_a_correlated_frame():
    keys = _orthonormal_keys(4, 10)
    keys[1] = keys[0] + 0.05 * keys[1]        # break mutual orthogonality
    with pytest.raises(ValueError, match="ORTHONORMAL key frame"):
        assert_orthonormal_frame(keys)


# =============================================================================
# 4. delta lives in COSINE space
# =============================================================================

def test_delta_is_a_cosine_and_stays_reachable():
    """Guard against rescaling ``delta`` by a LOGIT offset.

    The push target is a query coordinate, so it must stay inside [-1, 1].  The
    default Hard-Concrete stretch would suggest a +1.6 logit shift; interpreting
    that as a cosine drives the query PAST the antipode and flips the sign of the
    other coordinates, which is exactly the bug this test pins.
    """
    keys = _orthonormal_keys(3, 8)
    kh = key_frame(keys)
    query = (0.7 * kh[0] + 0.7 * kh[1]).unsqueeze(0)
    w = torch.zeros(1, 3)
    w[0, 0] = 1.0

    sane = correct_query(query, keys, w, delta=0.5)
    c_sane = torch.nn.functional.normalize(sane, dim=-1) @ kh.T
    assert -1.0 <= float(c_sane[0, 0]) <= 1.0
    assert float(c_sane[0, 1]) > 0.0          # the surviving parent stays positive

    absurd = correct_query(query, keys, w, delta=2.1)   # a logit masquerading as a cosine
    c_absurd = torch.nn.functional.normalize(absurd, dim=-1) @ kh.T
    # Overshoot: the removed axis now DOMINATES the query (|c| > the true parent),
    # i.e. the query points away from every key instead of at the true parents.
    assert abs(float(c_absurd[0, 0])) > abs(float(c_absurd[0, 1]))
