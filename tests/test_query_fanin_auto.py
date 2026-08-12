"""Automatic ``query_fanin_scale`` calibration (causaliT.utils.query_norm).

F is the only temperature left in the capped scoring path and it SCALES WITH THE
NODE COUNT, so it must be derived from the data instead of hard-coded.  These
tests pin the closed form, its inverse (the realised centroid posterior), the
T-free semantics (``init_edge_offset`` never enters F; it is resolved on its own
by ``resolve_init_edge_offset``) and the config plumbing.

The PURE closed form ``query_fanin_scale_from_centroid_p`` keeps its
``init_edge_offset`` parameter for legacy/ablation use; the RESOLVER passes 0.
"""

import math

import pytest
from omegaconf import OmegaConf

from causaliT.utils.query_norm import (
    DEFAULT_CENTROID_MAX_P,
    is_auto_fanin,
    query_fanin_scale_from_centroid_p,
    resolve_query_fanin_scale,
)

# Documented reference arm: n=10, T=ln 3, beta=0.5, symmetric stretch (c=0).
REF = dict(n_keys=10, init_tau=0.5, init_gamma=-1.1, init_zeta=1.1,
           init_edge_offset=math.log(3.0))


def centroid_posterior(fanin, n_keys, init_tau, init_gamma, init_zeta,
                       init_edge_offset=0.0, m=1.0):
    """P(z>0) actually produced at the centroid by a given F (independent impl)."""
    x = m * math.sqrt(fanin / n_keys)
    stretch = init_tau * math.log(-init_gamma / init_zeta)
    return 1.0 / (1.0 + math.exp(-(x - init_edge_offset - stretch)))


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------

def test_matches_documented_posterior_arm():
    # x = logit(0.9) + ln 3 = 2.1972 + 1.0986 = 3.2958  ->  F = 10 * x^2
    fanin = query_fanin_scale_from_centroid_p(max_p=0.9, **REF)
    assert fanin == pytest.approx(108.62, rel=1e-3)


@pytest.mark.parametrize("max_p", [0.5, 0.6, 0.75, 0.9])
def test_round_trip_realises_the_requested_posterior(max_p):
    fanin = query_fanin_scale_from_centroid_p(max_p=max_p, **REF)
    assert centroid_posterior(fanin, **REF) == pytest.approx(max_p, rel=1e-9)


def test_asymmetric_stretch_is_absorbed():
    # gamma=-0.1, zeta=1.1 -> c = beta*ln(1/11) < 0, so the SAME max_p needs a
    # smaller score than the symmetric arm: the stretch already opens the gate.
    kw = dict(n_keys=10, init_tau=2.0 / 3.0, init_gamma=-0.1, init_zeta=1.1)
    fanin = query_fanin_scale_from_centroid_p(max_p=0.9, **kw)
    assert centroid_posterior(fanin, **kw) == pytest.approx(0.9, rel=1e-9)
    assert fanin < query_fanin_scale_from_centroid_p(
        max_p=0.9, n_keys=10, init_tau=2.0 / 3.0, init_gamma=-1.1, init_zeta=1.1)


def test_scales_linearly_with_the_node_count():
    small = query_fanin_scale_from_centroid_p(**{**REF, "n_keys": 10}, max_p=0.9)
    large = query_fanin_scale_from_centroid_p(**{**REF, "n_keys": 400}, max_p=0.9)
    assert large == pytest.approx(40.0 * small, rel=1e-12)


def test_query_norm_init_scale_divides_quadratically():
    base = query_fanin_scale_from_centroid_p(max_p=0.9, **REF)
    scaled = query_fanin_scale_from_centroid_p(max_p=0.9, query_norm_init_scale=2.0,
                                               **REF)
    assert scaled == pytest.approx(base / 4.0, rel=1e-12)


@pytest.mark.parametrize("bad_p", [0.0, 1.0, 1.5, -0.1])
def test_rejects_non_probability_targets(bad_p):
    # It is a POSTERIOR, not a gate value: 1.0 is unreachable by the sigmoid.
    with pytest.raises(ValueError, match="query_centroid_max_p"):
        query_fanin_scale_from_centroid_p(max_p=bad_p, **REF)


def test_rejects_targets_unreachable_with_a_positive_score():
    # A strongly asymmetric stretch already opens the gate past 0.5 at score 0,
    # so max_p=0.5 would need a NEGATIVE centroid score -> impossible.
    with pytest.raises(ValueError, match="non-positive"):
        query_fanin_scale_from_centroid_p(
            max_p=0.5, n_keys=10, init_tau=2.0 / 3.0, init_gamma=-0.1,
            init_zeta=1.1)



# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def _config(**experiment):
    base = {"init_tau": 0.5, "init_gamma": -1.1, "init_zeta": 1.1,
            "init_edge_offset": math.log(3.0), "query_fanin_scale": "auto"}
    base.update(experiment)
    return OmegaConf.create({"experiment": base})


@pytest.mark.parametrize("sentinel", ["auto", "AUTO", None])
def test_auto_sentinels_are_resolved(sentinel):
    assert is_auto_fanin(sentinel)
    config = _config(query_fanin_scale=sentinel, query_centroid_max_p=0.9)
    info = resolve_query_fanin_scale(config, n_keys=10)
    assert info is not None
    # T-free: F = n * logit(0.9)^2 = 10 * (ln 9)^2 = 48.28; the configured
    # init_edge_offset (ln 3 in _config) is NOT read into F.
    assert config.experiment.query_fanin_scale == pytest.approx(48.28, rel=1e-3)


def test_explicit_float_is_never_overwritten():
    # Legacy configs must reproduce exactly.
    config = _config(query_fanin_scale=12.07)
    assert resolve_query_fanin_scale(config, n_keys=10) is None
    assert config.experiment.query_fanin_scale == 12.07


def test_default_max_p_is_used_when_unset():
    config = _config()
    info = resolve_query_fanin_scale(config, n_keys=10)
    assert info["query_centroid_max_p"] == DEFAULT_CENTROID_MAX_P


def test_the_fanin_scale_never_reads_the_offset():
    # init_edge_offset is a cross-gate init-balance device resolved separately
    # (resolve_init_edge_offset); F is identical with it pinned, absent, or in
    # homogeneous mode (where no cross gate exists at all).
    split = resolve_query_fanin_scale(_config(query_centroid_max_p=0.9), n_keys=10)
    homo = resolve_query_fanin_scale(
        _config(query_centroid_max_p=0.9, homogeneous_nodes=True), n_keys=10)
    free = resolve_query_fanin_scale(
        _config(query_centroid_max_p=0.9, init_edge_offset=0.0), n_keys=10)
    assert "init_edge_offset" not in split
    assert "init_edge_offset" not in homo
    assert split["query_fanin_scale"] == homo["query_fanin_scale"]
    assert homo["query_fanin_scale"] == free["query_fanin_scale"]
    assert homo["query_fanin_scale"] == pytest.approx(
        10.0 * math.log(9.0) ** 2, rel=1e-9)


def test_missing_experiment_section_is_a_no_op():
    assert resolve_query_fanin_scale(OmegaConf.create({"data": {}}), n_keys=10) is None
    assert resolve_query_fanin_scale(OmegaConf.create({"experiment": {}}),
                                     n_keys=10) is None
