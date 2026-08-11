"""Fan-in capacity and the in-degree prior (causaliT.utils.query_norm).

Every number pinned here is a line of
docs/experimental_elaborations/QUERY_NORM_CAPACITY_AND_FANIN_PRIOR.md, so the
document and the code cannot drift apart silently:

* the gate constants kappa / kappa_1 and the threshold logit x(p*)  - eq (2e), (3);
* the capacity algebra F = x*sqrt(K) and its inverse                - eq (3b);
* the penalty target mu = sqrt(K*/N)                                - eq (3c);
* the init-gate table of Section 3.1, which is the REASON the prior moves mu
  instead of shrinking F                                            - eq (8);
* the ER-k in-degree quantile table of Section 4                    - eq (14);
* the squeeze schedule and its STRUCTURE-epoch clock                - eq (3d)-(3e).

The load-bearing invariant, asserted several times below: with
``fanin_prior: null`` nothing changes at all.
"""

import math

import pytest
import torch
import torch.nn as nn

from causaliT.utils.query_norm import (
    FaninPriorSchedule,
    capacity_from_f,
    capacity_schedule,
    collect_query_norm_penalty,
    er_indegree_quantile,
    f_from_capacity,
    format_query_norm_log,
    init_gate_at_centroid,
    kappa,
    kappa_1,
    mu_from_capacity,
    overspend_penalty,
    p_at_saturation,
    query_norm_capacity,
    resolve_query_fanin_scale,
    resolve_query_norm,
    set_query_norm_target,
    x_of_p,
)

# Split-mode reference arm of the document: T = ln 3, symmetric stretch.
T = math.log(3.0)
GATE = dict(init_tau=0.5, init_gamma=-1.1, init_zeta=1.1)
P_SAT = 0.8209                      # sigmoid(kappa_1), the doc's p*
X_REF = 2.6211                      # x(p*) in split mode


# ---------------------------------------------------------------------------
# Gate constants, eq (2e)
# ---------------------------------------------------------------------------

def test_kappa_vanishes_at_the_symmetric_stretch():
    # -gamma == zeta, so ln(-gamma/zeta) = 0 and the posterior (2f) simplifies.
    assert kappa(**GATE) == pytest.approx(0.0, abs=1e-12)
    assert kappa_1(**GATE) == pytest.approx(0.5 * math.log(21.0), rel=1e-12)


def test_saturation_posterior_is_the_documented_p_star():
    assert p_at_saturation(**GATE) == pytest.approx(P_SAT, abs=5e-5)


def test_gate_constants_reject_an_invalid_parameterisation():
    for bad in (dict(init_tau=0.0), dict(init_gamma=0.1), dict(init_zeta=0.9)):
        with pytest.raises(ValueError):
            kappa(**{**GATE, **bad})


def test_x_of_p_is_the_documented_threshold_logit():
    assert x_of_p(P_SAT, init_edge_offset=T, **GATE) == pytest.approx(X_REF, abs=1e-3)
    # Homogeneous mode drops T, so the same p* needs a smaller score.
    assert x_of_p(P_SAT, **GATE) == pytest.approx(1.5225, abs=1e-3)


def test_x_of_p_rejects_a_non_probability_and_a_dead_threshold():
    with pytest.raises(ValueError):
        x_of_p(1.0, **GATE)
    with pytest.raises(ValueError):
        # p* = 0.5 -> logit 0, and with T = 0 the threshold collapses to x = 0:
        # no score can "hold" a parent, so the capacity algebra is undefined.
        x_of_p(0.5, **GATE)


# ---------------------------------------------------------------------------
# Capacity algebra, eq (3b) / (3c)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 7, 10, 400])
def test_capacity_and_scale_are_inverse(k):
    x = x_of_p(P_SAT, init_edge_offset=T, **GATE)
    assert capacity_from_f(f_from_capacity(k, x), x) == pytest.approx(k, rel=1e-12)


def test_scale_grows_like_sqrt_capacity():
    x = x_of_p(P_SAT, init_edge_offset=T, **GATE)
    # F^2 (= query_fanin_scale) is LINEAR in the capacity, so 4x the nodes is
    # 4x the config value but only 2x the score multiplier.
    assert f_from_capacity(400, x) / f_from_capacity(100, x) == pytest.approx(2.0)


def test_mu_is_the_square_root_of_the_capacity_fraction():
    assert mu_from_capacity(10, 400) == pytest.approx(math.sqrt(10 / 400))
    assert mu_from_capacity(400, 400) == pytest.approx(1.0)   # K* = N: no prior


def test_mu_rejects_a_non_positive_prior():
    with pytest.raises(ValueError):
        mu_from_capacity(0, 100)


# ---------------------------------------------------------------------------
# The init gate, Section 3.1 - why F must NOT be shrunk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, l, pi, z_open", [
    (10,  2.621, 0.821, True),
    (50,  1.172, 0.518, True),
    (100, 0.829, 0.433, False),
    (400, 0.414, 0.335, False),
])
def test_init_gate_table_of_section_3_1(n, l, pi, z_open):
    # A scale calibrated for K_init = 10 on an N-key centroid: the init signal
    # decays like sqrt(K_init/N) and the deterministic gate DIES past N ~ 57.
    l_init, pi_init, z_init = init_gate_at_centroid(
        10, n, X_REF, init_edge_offset=T, **GATE)
    assert l_init == pytest.approx(l, abs=1e-3)
    assert pi_init == pytest.approx(pi, abs=1e-3)
    assert (z_init > 0.0) is z_open


def test_keeping_the_capacity_at_n_saturates_the_gate_at_any_size():
    # The design choice: K_init = N always, so epoch 0 is size-independent.
    for n in (10, 100, 400):
        x = x_of_p(P_SAT, init_edge_offset=T, **GATE)
        _, pi_init, z_init = init_gate_at_centroid(
            n, n, x, init_edge_offset=T, **GATE)
        assert pi_init == pytest.approx(P_SAT, abs=1e-3)
        assert z_init == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ER-k in-degree law, eq (14) / Section 4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, degree, q", [
    (10, 4, 8), (50, 4, 9), (100, 4, 10),
    (400, 1, 3), (400, 2, 5), (400, 4, 10),
])
def test_indegree_quantile_table_of_section_4(n, degree, q):
    assert er_indegree_quantile(n, degree, 0.95) == q


def test_indegree_quantile_exceeds_the_mean_degree():
    # The whole point: `degree` is a MEAN, and the last topological node already
    # expects 2*degree parents (eq 13c), so a prior set to `degree` starves it.
    for n in (50, 100, 400):
        assert er_indegree_quantile(n, 4, 0.95) > 2 * 4 - 1


def test_indegree_quantile_is_monotone_in_alpha():
    values = [er_indegree_quantile(100, 2, a) for a in (0.5, 0.8, 0.95, 0.99)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# The over-spend penalty, eq (3c) / (12a)
# ---------------------------------------------------------------------------

def _log_scale(*multipliers):
    return torch.log(torch.tensor(list(multipliers), dtype=torch.float32))


@pytest.mark.parametrize("form", ["absolute", "mean", "capacity"])
def test_every_form_is_free_at_and_below_the_target(form):
    below = _log_scale(0.2, 0.5, 0.5)
    assert float(overspend_penalty(below, 0.5, form)) == pytest.approx(0.0, abs=1e-6)


def test_capacity_form_prices_the_relative_over_spend_identically_across_n():
    # "this row holds 50% more parents than the prior" at two problem sizes.
    k_star = 10
    for n in (50, 400):
        mu = mu_from_capacity(k_star, n)
        m = math.sqrt(1.5 * k_star / n)                   # holds 1.5 K* parents
        value = float(overspend_penalty(_log_scale(m), mu, "capacity"))
        assert value == pytest.approx(0.25, rel=1e-4)     # (1.5 - 1)^2


def test_absolute_form_does_not_transfer_across_n():
    # The reason the capacity form exists (Section 3.3, reason 2): the SAME
    # relative over-spend is charged ~8x less at n=400 than at n=50.
    k_star = 10
    prices = []
    for n in (50, 400):
        mu = mu_from_capacity(k_star, n)
        m = math.sqrt(1.5 * k_star / n)
        prices.append(float(overspend_penalty(_log_scale(m), mu, "absolute")))
    assert prices[0] > 5 * prices[1]


def test_unknown_penalty_form_raises():
    with pytest.raises(ValueError):
        overspend_penalty(_log_scale(2.0), 1.0, "quadratic")


# ---------------------------------------------------------------------------
# The schedule, eq (3d)-(3e)
# ---------------------------------------------------------------------------

def test_schedule_starts_at_mu_one_and_ends_at_the_prior():
    k_t, mu = capacity_schedule(0, 30, 400, 10)
    assert (k_t, mu) == (400.0, 1.0)                      # epoch 0 is untouched
    k_t, mu = capacity_schedule(30, 30, 400, 10)
    assert k_t == pytest.approx(10.0)
    assert mu == pytest.approx(math.sqrt(10 / 400))


def test_schedule_is_linear_in_edges_and_clamped():
    assert capacity_schedule(15, 30, 400, 10)[0] == pytest.approx(205.0)
    assert capacity_schedule(999, 30, 400, 10)[0] == pytest.approx(10.0)


def test_zero_anneal_is_the_immediate_squeeze():
    assert capacity_schedule(0, 0, 400, 10)[1] == pytest.approx(math.sqrt(10 / 400))


def test_schedule_is_inert_without_a_prior():
    assert capacity_schedule(5, 30, 400, None) == (400.0, 1.0)
    assert capacity_schedule(5, 30, 400, 400) == (400.0, 1.0)   # K* = N


# ---------------------------------------------------------------------------
# Writing the target onto the modules (shared_query de-duplication)
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    """Minimal stand-in for a gated attention block owning a multiplier."""

    def __init__(self, n_rows=4, param=None):
        super().__init__()
        self.query_norm_learnable = True
        self.query_norm_log_scale = (
            param if param is not None else nn.Parameter(torch.zeros(n_rows)))
        self.query_norm_target = 1.0


class _Model(nn.Module):
    def __init__(self, shared: bool):
        super().__init__()
        self.cross = _Block()
        self.self_attn = _Block(
            param=self.cross.query_norm_log_scale if shared else None)


def test_target_is_written_once_per_shared_parameter():
    assert set_query_norm_target(_Model(shared=True), 0.5, "capacity") == 1
    assert set_query_norm_target(_Model(shared=False), 0.5, "capacity") == 2


def test_written_target_and_form_reach_the_penalty():
    model = _Model(shared=False)
    with torch.no_grad():                       # every M_i = 2 x the target
        for b in (model.cross, model.self_attn):
            b.query_norm_log_scale.fill_(math.log(1.0))
    set_query_norm_target(model, 0.5, "capacity")
    # (M/mu)^2 - 1 = 3 per row, mean over rows, summed over the two blocks.
    penalty = collect_query_norm_penalty(model)
    assert penalty is not None and penalty.requires_grad      # it must train M
    assert float(penalty.detach()) == pytest.approx(18.0, rel=1e-5)


def test_set_target_rejects_an_unknown_form():

    with pytest.raises(ValueError):
        set_query_norm_target(_Model(shared=False), 0.5, "nonsense")


def test_realised_capacity_is_m_squared_times_n():
    model = _Model(shared=True)
    with torch.no_grad():
        model.cross.query_norm_log_scale.fill_(math.log(0.5))
    assert query_norm_capacity(model, 400) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# FaninPriorSchedule: the STRUCTURE-epoch clock
# ---------------------------------------------------------------------------

def _cfg(n_keys=400, prior=10, anneal=30, form="capacity"):
    return {
        "experiment": {"fanin_prior": prior, "query_norm_penalty_form": form},
        "training": {"fanin_anneal_epochs": anneal},
    }


def test_schedule_object_is_inert_without_a_prior():
    model = _Model(shared=False)
    sched = FaninPriorSchedule(_cfg(prior=None), n_keys=400)
    assert not sched.enabled
    sched.on_epoch_start(model)
    assert model.cross.query_norm_target == 1.0        # untouched
    assert sched.struct_epoch == 0


def test_first_epoch_writes_mu_one_exactly():
    model = _Model(shared=False)
    sched = FaninPriorSchedule(_cfg(), n_keys=400)
    sched.on_epoch_start(model)
    assert model.cross.query_norm_target == pytest.approx(1.0)
    assert model.cross.query_norm_penalty_form == "capacity"


def test_target_walks_to_mu_end_over_the_annealing_window():
    model = _Model(shared=False)
    sched = FaninPriorSchedule(_cfg(anneal=4), n_keys=400)
    for _ in range(5):
        sched.on_epoch_start(model)
    assert model.cross.query_norm_target == pytest.approx(math.sqrt(10 / 400))


def test_the_clock_freezes_outside_a_structure_phase():
    model = _Model(shared=False)
    sched = FaninPriorSchedule(_cfg(anneal=4), n_keys=400)
    sched.in_structure_phase = False
    for _ in range(10):                       # a long reconstruct phase
        sched.on_epoch_start(model)
    assert sched.struct_epoch == 0
    assert model.cross.query_norm_target == pytest.approx(1.0)
    sched.in_structure_phase = True
    for _ in range(5):
        sched.on_epoch_start(model)
    assert model.cross.query_norm_target == pytest.approx(math.sqrt(10 / 400))


def test_metrics_report_target_and_both_capacities():
    model = _Model(shared=False)
    sched = FaninPriorSchedule(_cfg(), n_keys=400)
    sched.on_epoch_start(model)
    metrics = sched.metrics(model)
    assert metrics["query_norm/target_mu"] == pytest.approx(1.0)
    assert metrics["query_norm/cap_target_edges"] == pytest.approx(400.0)
    assert metrics["query_norm/cap_actual_edges"] == pytest.approx(400.0)
    assert "query_norm/penalty_capacity" in metrics


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _exp_cfg(**overrides):
    exp = {
        "query_norm": True,
        "free_query_embedding": True,
        "query_fanin_scale": "auto",
        "query_centroid_max_p": P_SAT,
        "init_tau": 0.5, "init_gamma": -1.1, "init_zeta": 1.1,
        "init_edge_offset": T,
        "fanin_prior": None,
    }
    exp.update(overrides)
    return {"experiment": exp, "training": {"lambda_query_norm": 1.0e-3}}


def _resolve(cfg, n_keys=400):
    """``resolve_query_norm`` that must have fired (it returns None when off)."""
    info = resolve_query_norm(cfg, n_keys=n_keys)
    assert info is not None
    return info



def test_disabled_switch_is_a_no_op():
    cfg = _exp_cfg(query_norm=False)
    assert resolve_query_norm(cfg, n_keys=400) is None
    assert cfg["experiment"]["query_fanin_scale"] == "auto"


def test_master_switch_derives_the_whole_stack():
    cfg = _exp_cfg()
    info = _resolve(cfg)

    exp = cfg["experiment"]
    assert exp["normalize_query"] is True
    assert exp["query_centroid_init"] is True
    assert exp["query_norm_learnable"] is True
    assert exp["query_norm_init_scale"] == 1.0
    assert exp["query_norm_target"] == pytest.approx(1.0)     # no prior yet
    assert exp["query_fanin_scale"] == pytest.approx(400 * X_REF ** 2, rel=1e-3)
    assert info["z_init"] == pytest.approx(1.0)               # gate saturated


def test_an_explicit_fanin_scale_still_wins():
    cfg = _exp_cfg(query_fanin_scale=42.0)
    info = resolve_query_norm(cfg, n_keys=400)
    assert cfg["experiment"]["query_fanin_scale"] == 42.0
    assert info["query_fanin_scale_explicit"] is True


def test_legacy_auto_path_is_untouched_by_the_new_key():
    # resolve_query_fanin_scale (the OLD entry point) must keep working on a
    # config that never heard of query_norm.
    cfg = {"experiment": {"query_fanin_scale": "auto", "query_centroid_max_p": 0.9,
                          "init_tau": 0.5, "init_gamma": -1.1, "init_zeta": 1.1,
                          "init_edge_offset": T}}
    out = resolve_query_fanin_scale(cfg, n_keys=10)
    assert out["query_fanin_scale"] == pytest.approx(108.62, rel=1e-3)


def test_prior_sets_mu_and_selects_the_capacity_penalty():
    cfg = _exp_cfg(fanin_prior=10)
    info = resolve_query_norm(cfg, n_keys=400)
    assert info["mu_end"] == pytest.approx(math.sqrt(10 / 400))
    assert cfg["experiment"]["query_norm_target"] == pytest.approx(info["mu_end"])
    assert cfg["experiment"]["query_norm_penalty_form"] == "capacity"
    # ARM AXIS: an explicit form overrides the capacity default, so the three
    # reductions of Section 3.3 can be compared within one experiment.
    for form in ("absolute", "mean", "capacity"):
        cfg2 = _exp_cfg(fanin_prior=10, query_norm_penalty_form=form)
        info2 = resolve_query_norm(cfg2, n_keys=400)
        assert info2["query_norm_penalty_form"] == form
        assert cfg2["experiment"]["query_norm_penalty_form"] == form
    with pytest.raises(ValueError, match="penalty form"):
        resolve_query_norm(
            _exp_cfg(fanin_prior=10, query_norm_penalty_form="nonsense"),
            n_keys=400,
        )


    # F is NOT shrunk by the prior: the init gate must survive (Section 3.1).
    assert cfg["experiment"]["query_fanin_scale"] == pytest.approx(
        400 * X_REF ** 2, rel=1e-3)
    assert info["z_init"] == pytest.approx(1.0)


def test_auto_anneal_is_one_edge_per_structure_epoch():
    cfg = _exp_cfg(fanin_prior=10)
    cfg["training"]["fanin_anneal_epochs"] = "auto"
    info = resolve_query_norm(cfg, n_keys=400)
    assert info["fanin_anneal_epochs"] == 390
    assert cfg["training"]["fanin_anneal_epochs"] == 390


def test_prior_without_the_master_switch_raises():
    cfg = _exp_cfg(query_norm=False, fanin_prior=10)
    with pytest.raises(ValueError, match="query_norm=true"):
        resolve_query_norm(cfg, n_keys=400)


def test_prior_without_a_weight_raises_instead_of_being_inert():
    cfg = _exp_cfg(fanin_prior=10)
    cfg["training"]["lambda_query_norm"] = 0.0
    with pytest.raises(ValueError, match="lambda_query_norm"):
        resolve_query_norm(cfg, n_keys=400)


@pytest.mark.parametrize("k_star", [0, 401])
def test_out_of_range_prior_raises(k_star):
    with pytest.raises(ValueError, match="fanin_prior"):
        resolve_query_norm(_exp_cfg(fanin_prior=k_star), n_keys=400)


def test_a_closed_init_gate_raises_with_the_remedies():
    # p* <= 0.5 puts the centroid logit at or below the opening threshold.
    with pytest.raises(ValueError, match="query_centroid_max_p"):
        resolve_query_norm(_exp_cfg(query_centroid_max_p=0.5), n_keys=400)


def test_homogeneous_mode_drops_the_offset():
    cfg = _exp_cfg(homogeneous_nodes=True)
    info = _resolve(cfg)
    assert info["init_edge_offset"] == 0.0
    assert info["x"] == pytest.approx(1.5225, abs=1e-3)

    assert info["F"] < X_REF * math.sqrt(400)


def test_log_line_mentions_the_prior_state():
    assert "fanin_prior=off" in format_query_norm_log(
        _resolve(_exp_cfg(), n_keys=50))
    assert "mu_end" in format_query_norm_log(
        _resolve(_exp_cfg(fanin_prior=9), n_keys=50))


