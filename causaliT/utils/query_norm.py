"""Learnable per-node query-norm multiplier with a structural over-spend penalty.

Motivation
==========
When ``normalize_query=True`` the structural attention modules
(``GatedCrossAttention`` / ``GatedSelfAttention`` / ``CommutatorSelfAttention``)
L2-normalise the structural query onto the UNIT sphere and score with a fixed
``sqrt(query_fanin_scale)`` temperature.  With an orthonormal key frame this
HARD-caps the total directional budget ``sum_j cos^2(q_hat, k_j) <= 1``.

The SELF_ATTENTION spurious-``S3->X4`` investigation showed a node sometimes
needs to *overspend* that budget (host several parents at once).  The previous
fix relaxed the cap on a fixed EPOCH schedule, but once the ``W_q`` / ``W_K``
projections were removed the budget saturates much LATER than any preset window,
so the scheduled relief was already gone when it was needed.

This module replaces the schedule with an ADAPTIVE, learnable relief:

    q_eff = (q / ||q||) * M_i          # unit direction, per-node scaled norm
    score = <q_eff, k> * sqrt(fanin)   # == M_i * (unit-norm score)

where ``M_i = exp(log_scale_i)`` is a **per-node** learnable multiplier
(``log_scale`` initialised at ``log(init_scale)`` so ``M_i = init_scale``, 1.0 by
default).  The structural loss adds a penalty
``lambda_query_norm * sum_i relu(M_i - target)^2`` that only charges for
OVER-spending above the target (typically 1.0).  A node therefore raises its
own budget *only* when the structural signal (HSIC / L0) pays for it, and does
so *whenever* saturation actually bites — no epoch window to tune.

``log_scale`` is a STRUCTURAL parameter (matched by ``gradient_routing`` via the
``query_norm_log_scale`` name) so it is updated on the structural stream only.

Automatic ``query_fanin_scale``
-------------------------------
F is not a free hyper-parameter: it is the only temperature left in the capped
path, and it SCALES WITH THE NODE COUNT, so a hard-coded value silently breaks
on a new dataset.  ``resolve_query_fanin_scale`` derives it from one intent -
"a CENTROID-initialised query should give each candidate parent an edge
posterior of ``query_centroid_max_p``" - see the derivation on
``query_fanin_scale_from_centroid_p`` and
docs/experimental_elaborations/QUERY_FANIN_SCALE_BUDGET.md.

Capacity and the fan-in prior
-----------------------------
``F`` is not just a temperature: it is a fan-in CAPACITY measured in edges.
Because the keys are orthonormal the directional budget of a row is conserved,
``sum_j cos^2(u_i, k_j) = 1``, so a row can hold at most ``(M_i F / x(p*))^2``
parents at posterior ``p*`` (Lemma 1 of the design doc).  Two consequences:

* ``F = x(p*) sqrt(N)`` makes the centroid initialisation give EVERY candidate
  parent the posterior ``p*``, whatever the problem size N;
* a PRIOR on the in-degree, ``K*``, is then declared not by shrinking F (which
  kills the initialisation) but by lowering the per-row penalty target to
  ``mu = sqrt(K*/N)`` and annealing it there over structure epochs.

Two config keys drive all of this - ``experiment.query_norm`` (a master switch
that derives the whole normalised-query stack) and ``experiment.fanin_prior``
(``K*`` in EDGES).  See ``resolve_query_norm`` and
docs/experimental_elaborations/QUERY_NORM_CAPACITY_AND_FANIN_PRIOR.md.
"""

from typing import Any, Dict, List, Optional, Tuple

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Default centroid edge posterior used when the config does not set one.
DEFAULT_CENTROID_MAX_P = 0.9

#: Calculated operating point of the Hard-Concrete existence gates
#: (GatedCrossAttention / GatedSelfAttention / CommutatorSelfAttention /
#: HardConcreteCrossAttention): tau = 0.5 with the SYMMETRIC stretch
#: [gamma, zeta] = [-1.1, 1.1].  The two derived thresholds (eq 2e) are then
#:   kappa   = tau * ln(-gamma/zeta)        = 0           (gate opens at logit 0)
#:   kappa_1 = tau * ln((1-gamma)/(zeta-1))  = 0.5*ln 21 ~ 1.5223 (saturation)
#:   p_sat   = sigmoid(kappa_1 - kappa)      ~ 0.8209
#: See docs/documentation/ATTENTION_TEMPERATURES.md.
DEFAULT_GATE_TAU = 0.5
DEFAULT_GATE_GAMMA = -1.1
DEFAULT_GATE_ZETA = 1.1

#: Direction-gate (antisymmetric term) Binary-Concrete temperature.  The plain
#: Binary-Concrete has no stretch, so the gate is undecided at logit 0; the
#: Louizos et al. default 2/3 puts the 0.9-posterior point at
#: ln(9) * 2/3 ~ 1.465, just below the existence saturation logit
#: kappa_1 ~ 1.5223, so direction commits slightly ahead of existence.
DEFAULT_DIR_TAU = 2.0 / 3.0


#: Reductions available for the over-spend penalty; see ``overspend_penalty``.
PENALTY_FORMS = ("absolute", "mean", "capacity")

#: Penalty form selected automatically when a fan-in prior is declared.
FANIN_PRIOR_PENALTY_FORM = "capacity"



def coerce_fanin_scale(value: Any) -> float:
    """Cast a ``query_fanin_scale`` to float, rejecting an UNRESOLVED sentinel.

    ``auto`` must be turned into a number by ``resolve_query_fanin_scale`` at
    data-load time (it needs ``n_keys``).  If it reaches a module the config
    never went through that hook, so fail with the fix instead of the cryptic
    ``could not convert string to float: 'auto'``.
    """
    if is_auto_fanin(value):
        raise ValueError(
            f"query_fanin_scale={value!r} was never resolved to a number. "
            "It is derived from the node count by "
            "causaliT.utils.query_norm.resolve_query_fanin_scale, which runs in "
            "populate_seq_lengths_from_dataset; build the model from a config "
            "passed through that hook, or set an explicit float."
        )
    return float(value)




def make_query_norm_log_scale(num_nodes: int, init_scale: float = 1.0) -> nn.Parameter:
    """Per-node learnable log-multiplier ``log(M_i)`` initialised at ``log(init_scale)``.

    Parameters
    ----------
    num_nodes : int
        Number of query rows (children) that own an independent multiplier.
    init_scale : float
        Initial multiplier value ``M_i`` (must be > 0).  ``1.0`` reproduces the
        plain unit-norm cap at initialisation.

    Returns
    -------
    nn.Parameter of shape ``(num_nodes,)``.
    """
    if num_nodes is None or int(num_nodes) <= 0:
        raise ValueError(
            f"query_norm_num_nodes must be a positive int, got {num_nodes!r}."
        )
    if float(init_scale) <= 0.0:
        raise ValueError(f"query_norm_init_scale must be > 0, got {init_scale}.")
    return nn.Parameter(
        torch.full((int(num_nodes),), math.log(float(init_scale)), dtype=torch.float32)
    )


def apply_query_norm(
    query: torch.Tensor,
    log_scale: torch.Tensor,
    query_fanin_scale: float,
) -> Tuple[torch.Tensor, float]:
    """Unit-normalise ``query`` and scale each row by its learnable ``M_i``.

    Parameters
    ----------
    query : torch.Tensor
        Structural query ``(B, L, E)``, normalised along the last dim.
    log_scale : torch.Tensor
        Per-node log-multiplier ``(L,)``; ``M_i = exp(log_scale_i)``.
    query_fanin_scale : float
        Fixed score temperature; the returned ``scale_s`` is ``sqrt(fanin)``.

    Returns
    -------
    (q_s, scale_s)
        ``q_s = (query/||query||) * M`` (broadcast per node) and
        ``scale_s = sqrt(query_fanin_scale)``.
    """
    q_hat = F.normalize(query, p=2.0, dim=-1, eps=1e-8)
    m = torch.exp(log_scale)                    # (L,)
    q_s = q_hat * m.view(1, -1, 1)              # broadcast over batch & feature
    scale_s = math.sqrt(query_fanin_scale)
    return q_s, scale_s


def overspend_penalty(log_scale: torch.Tensor, target: float = 1.0,
                      form: str = "absolute") -> torch.Tensor:
    """Per-node over-spend penalty on ``M = exp(log_scale)`` above ``target``.

    Only budget grown ABOVE ``target`` is charged; contracting below the target
    is free (and unincentivised, since the structural signal generally prefers a
    larger budget).  Three reductions, all flat at ``M = target``:

    ``absolute`` (default, and the historical behaviour)
        ``sum_i relu(M_i - target)^2``.  Written in SCALE units and summed, so
        it does NOT transfer across problem sizes: the total grows like N while
        the price of one extra parent falls like 1/N.
    ``mean``
        ``mean_i relu(M_i - target)^2``.  Fixes the growth of the total only.
    ``capacity``
        ``mean_i relu((M_i/target)^2 - 1)^2``.  Written in CAPACITY units: by
        Lemma 1 a row holding ``f`` parents needs ``M_i = sqrt(f/N)``, so
        ``(M_i/target)^2 = f/K*`` and the N cancels identically.  "This row
        holds 50% more parents than the prior" then costs the same at every N,
        which is the only form whose ``lambda_query_norm`` transfers.

    See docs/experimental_elaborations/QUERY_NORM_CAPACITY_AND_FANIN_PRIOR.md,
    Section 3.3 and eq (12a).
    """
    m = torch.exp(log_scale)
    t = float(target)
    if form == "absolute":
        return F.relu(m - t).pow(2).sum()
    if form == "mean":
        return F.relu(m - t).pow(2).mean()
    if form == "capacity":
        if t <= 0.0:
            raise ValueError(
                f"the capacity penalty form needs a positive target, got {t!r}."
            )
        return F.relu((m / t).pow(2) - 1.0).pow(2).mean()
    raise ValueError(
        f"unknown query-norm penalty form {form!r}; available: {PENALTY_FORMS}."
    )


def _learnable_query_norm_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Modules owning a LEARNABLE ``query_norm_log_scale``, deduped by param id.

    A single ``log_scale`` shared by the cross & self blocks (``shared_query``)
    appears once, so it is charged / written exactly once.
    """
    out: List[torch.nn.Module] = []
    seen = set()
    for m in model.modules():
        p = getattr(m, "query_norm_log_scale", None)
        if p is None or not getattr(m, "query_norm_learnable", False):
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append(m)
    return out


def collect_query_norm_penalty(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """Sum the over-spend penalty over every module owning a learnable multiplier.

    Each module contributes with its own ``query_norm_target`` and
    ``query_norm_penalty_form`` (default ``absolute``).  The reduction ACROSS
    modules stays a sum, so the cross and self blocks keep separate budgets;
    the ``mean`` of the non-absolute forms is taken WITHIN a module, over its
    rows.  Returns ``None`` when no learnable multiplier exists.
    """
    total: Optional[torch.Tensor] = None
    for m in _learnable_query_norm_modules(model):
        pen = overspend_penalty(
            getattr(m, "query_norm_log_scale"),
            getattr(m, "query_norm_target", 1.0),
            getattr(m, "query_norm_penalty_form", "absolute"),
        )
        total = pen if total is None else total + pen
    return total



def set_query_norm_target(model: torch.nn.Module, target: float,
                          form: Optional[str] = None) -> int:
    """Write ``query_norm_target`` (and optionally the penalty form) on every module.

    Idempotent and deduplicated by parameter id, so it can be re-run on every
    epoch start and on every training-phase switch.  Returns how many distinct
    multipliers were written.
    """
    if form is not None and form not in PENALTY_FORMS:
        raise ValueError(
            f"unknown query-norm penalty form {form!r}; available: {PENALTY_FORMS}."
        )
    modules = _learnable_query_norm_modules(model)
    for m in modules:
        object.__setattr__(m, "query_norm_target", float(target))
        if form is not None:
            object.__setattr__(m, "query_norm_penalty_form", str(form))
    return len(modules)



def query_norm_stats(model: torch.nn.Module):
    """Return ``(mean_M, max_M)`` across all learnable multipliers (deduped).

    Returns ``(None, None)`` when no learnable multiplier exists.  Values are
    detached (diagnostics only).
    """
    allm = _all_multipliers(model)
    if allm is None:
        return None, None
    return allm.mean(), allm.max()


def _all_multipliers(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """Flat detached tensor of every ``M_i`` in the model (deduped by param id)."""
    scales = []
    seen = set()
    for m in model.modules():
        p = getattr(m, "query_norm_log_scale", None)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        scales.append(torch.exp(p.detach()).reshape(-1))
    return torch.cat(scales) if scales else None


def query_norm_capacity(model: torch.nn.Module, n_keys: int) -> Optional[float]:
    """Realised capacity in EDGES, ``mean_i M_i^2 * N`` (Lemma 1).

    The number of parents the average row could hold at posterior ``p*`` given
    its current budget; compare against ``K(t)`` from ``capacity_schedule``.
    Returns ``None`` when the model owns no multiplier.
    """
    allm = _all_multipliers(model)
    if allm is None:
        return None
    return float(allm.pow(2).mean().item() * float(n_keys))


def query_norm_penalty_by_form(model: torch.nn.Module) -> Dict[str, float]:
    """All three penalty reductions at the CURRENT target, for diagnostics.

    Appendix B question 1 of the design doc asks for the choice of reduction to
    be made on data rather than taste, so every form is reported regardless of
    which one is charged.  Detached floats; ``{}`` when nothing is learnable.
    """
    out: Dict[str, float] = {}
    modules = _learnable_query_norm_modules(model)
    if not modules:
        return out
    with torch.no_grad():
        for form in PENALTY_FORMS:
            total = 0.0
            for m in modules:
                target = getattr(m, "query_norm_target", 1.0)
                if form == "capacity" and float(target) <= 0.0:
                    continue
                total += float(
                    overspend_penalty(
                        getattr(m, "query_norm_log_scale"), target, form
                    ).item()
                )

            out[form] = total
    return out



# =============================================================================
# Hard-Concrete gate constants (eq 2e of the design doc)
# =============================================================================

def _validate_gate(init_tau: float, init_gamma: float, init_zeta: float) -> None:
    """Reject a Hard-Concrete parameterisation whose thresholds are undefined."""
    if float(init_tau) <= 0.0 or float(init_gamma) >= 0.0 or float(init_zeta) <= 1.0:
        raise ValueError(
            "Hard-Concrete needs init_tau > 0, init_gamma < 0, init_zeta > 1; got "
            f"{init_tau!r}, {init_gamma!r}, {init_zeta!r}."
        )


def kappa(init_tau: float = DEFAULT_GATE_TAU, init_gamma: float = DEFAULT_GATE_GAMMA,
          init_zeta: float = DEFAULT_GATE_ZETA) -> float:

    """Logit offset at which the gate OPENS: ``tau * ln(-gamma/zeta)``, eq (2e).

    The stretched Binary-Concrete leaves zero, ``z_ij > 0``, exactly when
    ``l_ij - T > kappa`` (deterministic gate).  Zero at the defaults, because
    ``-gamma == zeta``.
    """
    _validate_gate(init_tau, init_gamma, init_zeta)
    return float(init_tau) * math.log(-float(init_gamma) / float(init_zeta))


def kappa_1(init_tau: float = DEFAULT_GATE_TAU, init_gamma: float = DEFAULT_GATE_GAMMA,
            init_zeta: float = DEFAULT_GATE_ZETA) -> float:

    """Logit offset at which the gate SATURATES: ``tau * ln((1-gamma)/(zeta-1))``.

    Eq (2e).  Above ``l_ij - T >= kappa_1`` the clamp is flat (``z_ij = 1``), so
    anything reaching the logit through the gate VALUE stops giving gradient.
    """
    _validate_gate(init_tau, init_gamma, init_zeta)
    return float(init_tau) * math.log(
        (1.0 - float(init_gamma)) / (float(init_zeta) - 1.0))


def x_of_p(max_p: float, init_tau: float = DEFAULT_GATE_TAU,
           init_gamma: float = DEFAULT_GATE_GAMMA,
           init_zeta: float = DEFAULT_GATE_ZETA,
           init_edge_offset: float = 0.0) -> float:

    """Threshold LOGIT for an edge posterior ``max_p``: ``logit(p) + T + kappa``.

    Eq (3), the inverse of the posterior ``pi = sigmoid(l - T - kappa)`` of eq
    (2f).  This is the single quantity the whole capacity calculus needs: a row
    "holds" a parent when its logit reaches ``x(p*)``.
    """
    if not 0.0 < float(max_p) < 1.0:
        raise ValueError(
            f"query_centroid_max_p must be in (0, 1), got {max_p!r}: it is an "
            "edge POSTERIOR, so 1.0 is unreachable (use ~0.5-0.9)."
        )
    stretch = kappa(init_tau, init_gamma, init_zeta)
    x = math.log(max_p / (1.0 - max_p)) + float(init_edge_offset) + stretch
    if x <= 0.0:
        raise ValueError(
            f"query_centroid_max_p={max_p} needs a non-positive score (x={x:.4g}) "
            f"given init_edge_offset={init_edge_offset} and stretch={stretch:.4g}; "
            "raise query_centroid_max_p."
        )
    return x


def p_at_saturation(init_tau: float = DEFAULT_GATE_TAU,
                    init_gamma: float = DEFAULT_GATE_GAMMA,
                    init_zeta: float = DEFAULT_GATE_ZETA) -> float:

    """Smallest posterior whose threshold logit already saturates the gate.

    ``sigmoid(kappa_1 - kappa)``; 0.8209 at the defaults.  A ``p*`` below this
    leaves the deterministic gate OPEN but not saturated at initialisation, and
    a ``p* <= 0.5`` closes it altogether (eq 2g).
    """
    d = kappa_1(init_tau, init_gamma, init_zeta) - kappa(init_tau, init_gamma, init_zeta)
    return 1.0 / (1.0 + math.exp(-d))


# =============================================================================
# Capacity algebra (Lemma 1 and eq 3b / 3c / 3d / 8a)
# =============================================================================

def f_from_capacity(capacity: float, x: float) -> float:
    """Score scale ``F = x * sqrt(K)`` that affords ``K`` parents at ``x``, eq (3b).

    ``F`` is the MULTIPLIER of the cosine, not the config value: the config key
    ``query_fanin_scale`` holds ``F^2``.
    """
    if float(capacity) <= 0.0:
        raise ValueError(f"capacity must be > 0, got {capacity!r}.")
    return float(x) * math.sqrt(float(capacity))


def capacity_from_f(f: float, x: float) -> float:
    """Capacity in EDGES afforded by a score scale ``F``: ``(F/x)^2``, inverse of (3b)."""
    if float(x) <= 0.0:
        raise ValueError(f"x must be > 0, got {x!r}.")
    return (float(f) / float(x)) ** 2


def mu_from_capacity(k_star: float, n_keys: int) -> float:
    """Penalty target ``mu = sqrt(K*/N)`` for a declared in-degree prior, eq (3c).

    A row can reach ``x(p*)`` on ``K*`` keys only if ``M_i >= sqrt(K*/N)``
    (Lemma 1 with ``F = x(p*) sqrt(N)``), so this is the smallest budget the
    prior must leave free.  ``K* = N`` gives ``mu = 1``: no prior at all.
    """
    if int(n_keys) < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys!r}.")
    if float(k_star) <= 0.0:
        raise ValueError(f"fanin_prior must be > 0 edges, got {k_star!r}.")
    return math.sqrt(float(k_star) / float(n_keys))


def init_gate_at_centroid(
    k_init: int,
    n_keys: int,
    x: float,
    init_tau: float = DEFAULT_GATE_TAU,
    init_gamma: float = DEFAULT_GATE_GAMMA,
    init_zeta: float = DEFAULT_GATE_ZETA,
    init_edge_offset: float = 0.0,
) -> Tuple[float, float, float]:

    """``(l_init, pi_init, z_init)`` of a centroid-initialised row, eq (8a) + (2a-2c).

    The centroid of ALL ``N`` keys gives every cosine ``1/sqrt(N)``, so a scale
    calibrated for a capacity ``K_init`` produces
    ``l_init = x * sqrt(K_init/N)``: the init signal DECAYS like
    ``sqrt(K_init/N)``, which is why the capacity must not be shrunk below N
    (Section 3.1).  ``z_init`` is the DETERMINISTIC gate (no logistic noise).
    """
    _validate_gate(init_tau, init_gamma, init_zeta)
    if int(n_keys) < 1 or int(k_init) < 1:
        raise ValueError(f"need n_keys >= 1 and k_init >= 1, got {n_keys!r}, {k_init!r}.")
    l_init = float(x) * math.sqrt(float(k_init) / float(n_keys))
    shifted = l_init - float(init_edge_offset)
    pi_init = 1.0 / (1.0 + math.exp(-(shifted - kappa(init_tau, init_gamma, init_zeta))))
    s = 1.0 / (1.0 + math.exp(-shifted / float(init_tau)))
    z_init = min(max(s * (float(init_zeta) - float(init_gamma)) + float(init_gamma),
                     0.0), 1.0)
    return l_init, pi_init, z_init


def capacity_schedule(struct_epoch: int, anneal_epochs: int, n_keys: int,
                      k_star: Optional[float]) -> Tuple[float, float]:
    """``(K(t), mu(t))`` of the fan-in squeeze, eq (3d)-(3e).

    Linear in EDGES, ``K(t) = N + (K* - N) rho(t)`` with ``rho`` ramping 0 -> 1
    over ``anneal_epochs`` STRUCTURE epochs, then ``mu(t) = sqrt(K(t)/N)``.
    Linear in K rather than in mu so the schedule is interpretable in edges.

    ``rho(0) = 0`` gives ``mu = 1`` exactly, i.e. epoch 0 is the un-modified
    state - no initialisation is ever destroyed.  Gradualness is not cosmetic:
    an instantaneous target DEFLATES a row instead of pruning it, because
    re-concentrating the directional budget on fewer keys takes gradient steps.
    ``anneal_epochs = 0`` selects that instantaneous squeeze on purpose (the
    ablation).  ``k_star=None`` or ``K* >= N`` disables the prior.
    """
    n = int(n_keys)
    if k_star is None:
        return float(n), 1.0
    k = float(k_star)
    if k >= n:
        return float(n), 1.0
    a = int(anneal_epochs)
    rho = 1.0 if a <= 0 else min(max(float(struct_epoch) / float(a), 0.0), 1.0)
    k_t = float(n) + (k - float(n)) * rho
    return k_t, math.sqrt(k_t / float(n))


# =============================================================================
# In-degree law of the ER-k DAG sampler (eq 13-14)
# =============================================================================

def er_indegree_quantile(n_nodes: int, degree: float, alpha: float = 0.95) -> int:
    """Pooled in-degree quantile ``Q(alpha)`` of an ER-k DAG, eq (14).

    ``scm_ds/random_scm.py::_sample_dag`` places ``m = round(degree * n)`` edges
    uniformly over the ``C(n,2)`` forward slots of a random topological order,
    so ``degree`` fixes the MEAN in-degree, never the maximum: the node at
    topological position ``i`` has ``D_i ~ Bin(i, p)`` with
    ``p = 2*degree/(n-1)``, and the LAST node already expects ``2*degree``
    parents (eq 13c).  Setting a fan-in prior to ``degree`` would therefore
    starve the downstream nodes; this quantile - "the prior covers a fraction
    ``alpha`` of the nodes" - is the recommended estimator instead.

    Pure Binomial (the exact law is hypergeometric, since the draw is without
    replacement; the two agree to within one edge at the sizes we run).
    """
    n = int(n_nodes)
    if n <= 1:
        return 0
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    p = min(max(2.0 * float(degree) / (n - 1), 0.0), 1.0)

    pmf = [0.0] * n                       # pooled P[D = d], d = 0 .. n-1
    for i in range(n):
        if p >= 1.0:
            pmf[i] += 1.0 / n             # every admissible slot is filled
            continue
        term = (1.0 - p) ** i             # P[Bin(i, p) = 0]
        for d in range(i + 1):
            pmf[d] += term / n
            if d < i:                     # recurrence to P[Bin(i, p) = d+1]
                term *= (p / (1.0 - p)) * (i - d) / (d + 1)

    cdf = 0.0
    for d in range(n):
        cdf += pmf[d]
        if cdf >= float(alpha):
            return d
    return n - 1


# =============================================================================
# Automatic query_fanin_scale
# =============================================================================

def query_fanin_scale_from_centroid_p(
    n_keys: int,
    max_p: float = DEFAULT_CENTROID_MAX_P,
    init_tau: float = DEFAULT_GATE_TAU,
    init_gamma: float = DEFAULT_GATE_GAMMA,
    init_zeta: float = DEFAULT_GATE_ZETA,
    init_edge_offset: float = 0.0,
    query_norm_init_scale: float = 1.0,
) -> float:

    """Fan-in scale F whose CENTROID init gives every parent posterior ``max_p``.

    At the centroid every cosine is ``1/sqrt(n)``, so the score is
    ``x = M * sqrt(F/n)`` (M = ``query_norm_init_scale``).  The Hard-Concrete
    edge posterior is ``P(z>0) = sigmoid(x - T - c)`` with ``T =
    init_edge_offset`` and the stretch term ``c = beta * ln(-gamma/zeta)``
    (beta = ``init_tau``).  Inverting for ``P = max_p``::

        x = logit(max_p) + T + c        ->      F = n * (x / M)^2

    ``max_p`` is a PROBABILITY (the sigmoid never reaches 1); useful values are
    in [0.5, 0.9].  F scales with ``n``, which is why it must be derived per
    dataset instead of hard-coded.

    Thin wrapper over ``x_of_p`` (eq 3) and ``f_from_capacity`` (eq 3b), kept
    for its config-facing signature and for byte-identical legacy behaviour.
    """
    if int(n_keys) < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys!r}.")
    if float(query_norm_init_scale) <= 0.0:
        raise ValueError(
            f"query_norm_init_scale must be > 0, got {query_norm_init_scale!r}."
        )
    x = x_of_p(max_p, init_tau, init_gamma, init_zeta, init_edge_offset)
    return float(n_keys) * (x / float(query_norm_init_scale)) ** 2



def is_auto_fanin(value: Any) -> bool:
    """True when ``query_fanin_scale`` asks to be derived (``auto`` / ``null``)."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in ("auto", "derive", "null")


def gate_tau_from_experiment(exp: Any, homogeneous: bool) -> float:
    """Existence-gate temperature of the block the F-derivation targets.

    Harmonized split keys win: ``init_tau_cross`` (split mode: the S->X cross
    gate, where ``init_edge_offset`` lives) or ``init_tau_self`` (homogeneous
    mode: the single square block IS the self gate).  The legacy shared
    ``init_tau`` is the fallback so pre-split configs reproduce exactly;
    ``DEFAULT_GATE_TAU`` is the calculated default.  When the split keys set
    DIFFERENT cross/self temperatures the shared scale F is derived for the
    block named above and a warning is emitted: the gates then saturate at
    different logits, which the single-F capacity calculus cannot represent.
    """
    if not hasattr(exp, "get"):
        return DEFAULT_GATE_TAU

    def _get(key):
        value = exp.get(key, None)
        return None if value is None else float(value)

    primary = _get("init_tau_self" if homogeneous else "init_tau_cross")
    secondary = _get("init_tau_cross" if homogeneous else "init_tau_self")
    legacy = _get("init_tau")
    tau = primary if primary is not None else (
        legacy if legacy is not None else DEFAULT_GATE_TAU)
    if (
        not homogeneous
        and primary is not None
        and secondary is not None
        and primary != secondary
    ):
        logger.warning(
            "[query-norm] init_tau_cross=%.4g differs from init_tau_self=%.4g: "
            "query_fanin_scale is derived for the CROSS gate (the one carrying "
            "init_edge_offset); the self gate saturates at a different logit.",
            primary, secondary,
        )
    return float(tau)


def resolve_query_fanin_scale(config: Any, n_keys: int) -> Optional[Dict[str, Any]]:

    """Fill ``experiment.query_fanin_scale`` IN PLACE when it is ``auto``.

    ``init_edge_offset`` lives ONLY on the S->X ``GatedCrossAttention`` gate, so
    it is dropped in ``homogeneous_nodes`` mode (one square block, no cross
    block).  An explicit numeric ``query_fanin_scale`` is always honoured (old
    configs reproduce exactly) and returns ``None``.
    """
    exp = config.get("experiment", None) if hasattr(config, "get") else None
    if exp is None or "query_fanin_scale" not in exp:
        return None
    if not is_auto_fanin(exp.get("query_fanin_scale", None)):
        return None

    def _get(key, default):
        value = exp.get(key, default)
        return default if value is None else value

    homogeneous = bool(_get("homogeneous_nodes", False))
    offset = 0.0 if homogeneous else float(_get("init_edge_offset", 0.0))
    max_p = float(_get("query_centroid_max_p", DEFAULT_CENTROID_MAX_P))
    fanin = query_fanin_scale_from_centroid_p(
        n_keys=n_keys,
        max_p=max_p,
        init_tau=gate_tau_from_experiment(exp, homogeneous),
        init_gamma=float(_get("init_gamma", DEFAULT_GATE_GAMMA)),
        init_zeta=float(_get("init_zeta", DEFAULT_GATE_ZETA)),
        init_edge_offset=offset,
        query_norm_init_scale=float(_get("query_norm_init_scale", 1.0)),
    )

    exp["query_fanin_scale"] = fanin
    return {"query_fanin_scale": fanin, "n_keys": int(n_keys),
            "query_centroid_max_p": max_p, "init_edge_offset": offset}


# =============================================================================
# The two front-door keys: experiment.query_norm and experiment.fanin_prior
# =============================================================================

def _unstruct(config: Any) -> None:
    """Allow new keys on an OmegaConf container; no-op for a plain dict."""
    try:
        from omegaconf import OmegaConf  # local import: dicts must work too

        if OmegaConf.is_config(config):
            OmegaConf.set_struct(config, False)
    except Exception:                                       # pragma: no cover
        pass


def resolve_query_norm(config: Any, n_keys: int) -> Optional[Dict[str, Any]]:
    """Expand ``query_norm`` / ``fanin_prior`` into the individual keys, IN PLACE.

    Two config keys carry the whole normalised-query + capacity design; every
    other key below is DERIVED from them and from the node count ``N``.

    ``experiment.query_norm: true`` - a master switch.  It OVERWRITES
    ``normalize_query``, ``query_centroid_init`` (following
    ``free_query_embedding``), ``query_norm_learnable``,
    ``query_norm_init_scale`` and ``query_fanin_scale``; only an EXPLICIT
    numeric ``query_fanin_scale`` survives, so a legacy run still reproduces.
    The scale is ``F^2 = N * x(p*)^2`` (eq 3b at ``K_init = N``), i.e. the
    centroid initialisation gives every candidate parent the posterior ``p*``
    whatever ``N`` is.

    ``experiment.fanin_prior: K*`` - the prior in-degree, in EDGES.  It sets
    ``query_norm_target = mu = sqrt(K*/N)`` (eq 3c), selects the ``capacity``
    penalty form (eq 12a, the only one whose ``lambda_query_norm`` transfers
    across ``N``) and resolves ``training.fanin_anneal_epochs``.  The squeeze
    itself is applied by the forecaster over STRUCTURE epochs; here we only
    record the endpoint.

    Guards (Section 3.1 of the design doc, specialised to ``K_init = N``):
    ``p* <= 0.5`` closes the deterministic gate at initialisation and RAISES;
    ``p* < sigmoid(kappa_1 - kappa)`` leaves it open but unsaturated and warns.

    Returns a dict of everything resolved (for the startup log), or ``None``
    when neither key is active.
    """
    exp = config.get("experiment", None) if hasattr(config, "get") else None
    if exp is None:
        return None

    def _get(key, default):
        value = exp.get(key, default)
        return default if value is None else value

    enabled = bool(_get("query_norm", False))
    prior_raw = exp.get("fanin_prior", None)
    if not enabled and prior_raw is None:
        return None
    if not enabled and prior_raw is not None:
        raise ValueError(
            f"experiment.fanin_prior={prior_raw!r} needs experiment.query_norm=true: "
            "the prior is a target on the learnable query-norm multiplier M_i, "
            "which only exists on the normalised-query path."
        )

    n = int(n_keys)
    if n < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys!r}.")
    _unstruct(config)

    # --- gate constants and the threshold logit x(p*) -----------------------
    homogeneous = bool(_get("homogeneous_nodes", False))
    tau = gate_tau_from_experiment(exp, homogeneous)
    gamma = float(_get("init_gamma", DEFAULT_GATE_GAMMA))
    zeta = float(_get("init_zeta", DEFAULT_GATE_ZETA))

    offset = 0.0 if homogeneous else float(_get("init_edge_offset", 0.0))
    p_star = float(_get("query_centroid_max_p", DEFAULT_CENTROID_MAX_P))
    x = x_of_p(p_star, tau, gamma, zeta, offset)

    if p_star <= 0.5:
        raise ValueError(
            f"query_centroid_max_p={p_star} closes the deterministic gate at "
            f"initialisation: with the centroid init the logit is x(p*)={x:.4g} "
            f"and the gate opens only above T+kappa="
            f"{offset + kappa(tau, gamma, zeta):.4g} (eq 2g), which needs p* > 0.5. "
            "Raise query_centroid_max_p (>= "
            f"{p_at_saturation(tau, gamma, zeta):.4f} also saturates it) or drop "
            "init_edge_offset."
        )
    p_sat = p_at_saturation(tau, gamma, zeta)
    if p_star < p_sat:
        logger.warning(
            "[query-norm] query_centroid_max_p=%.4f is below the gate saturation "
            "posterior %.4f: the deterministic gate starts OPEN but not saturated "
            "(0 < z_init < 1).  Raise it to %.4f to start at z_init=1.",
            p_star, p_sat, p_sat,
        )


    # --- F from the capacity K_init = N (eq 3b) -----------------------------
    # K_init is ALWAYS N: the centroid is taken over all N keys, so shrinking
    # the capacity here would decay the init signal like sqrt(K_init/N) and can
    # kill the gate outright (Section 3.1).  The prior is priced on mu instead.
    f_scale = f_from_capacity(n, x)
    fanin_explicit = not is_auto_fanin(exp.get("query_fanin_scale", None))
    if not fanin_explicit:
        exp["query_fanin_scale"] = f_scale ** 2

    free_query = bool(_get("free_query_embedding", False))
    exp["normalize_query"] = True
    exp["query_centroid_init"] = free_query
    exp["query_norm_learnable"] = True
    exp["query_norm_init_scale"] = 1.0          # so mu(0) = 1 is exact
    if not free_query:
        logger.warning(
            "[query-norm] free_query_embedding=false: the query is NOT initialised "
            "at the key centroid, but query_fanin_scale is calibrated for that "
            "centroid (F = x(p*)*sqrt(N)).  The init posterior below is therefore "
            "indicative only."
        )

    l_init, pi_init, z_init = init_gate_at_centroid(
        n, n, x, tau, gamma, zeta, offset)

    # --- the prior (eq 3c) --------------------------------------------------
    k_star: Optional[int] = None
    mu_end = 1.0
    anneal = 0
    penalty_form = "absolute"
    if prior_raw is not None:
        k_star = int(prior_raw)
        if not 1 <= k_star <= n:
            raise ValueError(
                f"experiment.fanin_prior={k_star} is out of range: it is an "
                f"in-degree in EDGES and must satisfy 1 <= K* <= n_keys={n}."
            )
        mu_end = mu_from_capacity(k_star, n)
        # ``capacity`` is the default because it is the only form whose
        # lambda_query_norm transfers across N (eq 12a); an explicit key wins so
        # the three forms can be compared as arms of one experiment.
        form_raw = exp.get("query_norm_penalty_form", None)
        penalty_form = (FANIN_PRIOR_PENALTY_FORM if form_raw is None
                        else str(form_raw).strip().lower())
        if penalty_form not in PENALTY_FORMS:
            raise ValueError(
                f"experiment.query_norm_penalty_form={form_raw!r} is not a known "
                f"penalty form; available: {PENALTY_FORMS}."
            )


        train = config.get("training", None) if hasattr(config, "get") else None
        lam = 0.0 if train is None else float(train.get("lambda_query_norm", 0.0) or 0.0)
        if lam <= 0.0:
            raise ValueError(
                f"experiment.fanin_prior={k_star} is set but "
                "training.lambda_query_norm=0: the prior would be silently inert. "
                "Set a positive lambda_query_norm (it is NOT derivable and does "
                "not transfer across n; see Section 3.3 of the design doc)."
            )
        anneal_raw = train.get("fanin_anneal_epochs", "auto") if train is not None else "auto"
        if anneal_raw is None or (isinstance(anneal_raw, str)
                                  and anneal_raw.strip().lower() == "auto"):
            anneal = max(n - k_star, 0)      # one edge per structure epoch
        else:
            anneal = int(anneal_raw)
            if anneal < 0:
                raise ValueError(
                    f"training.fanin_anneal_epochs={anneal_raw!r} must be >= 0 "
                    "(0 = immediate squeeze, the ablation)."
                )
        if train is not None:
            train["fanin_anneal_epochs"] = anneal

    exp["query_norm_target"] = mu_end
    exp["query_norm_penalty_form"] = penalty_form
    exp["fanin_prior"] = k_star

    return {
        "n_keys": n,
        "p_star": p_star,
        "p_saturation": p_sat,
        "x": x,
        "kappa": kappa(tau, gamma, zeta),
        "kappa_1": kappa_1(tau, gamma, zeta),
        "init_edge_offset": offset,
        "F": f_scale,
        "query_fanin_scale": float(exp["query_fanin_scale"]),
        "query_fanin_scale_explicit": fanin_explicit,
        "query_centroid_init": free_query,
        "l_init": l_init,
        "pi_init": pi_init,
        "z_init": z_init,
        "fanin_prior": k_star,
        "mu_end": mu_end,
        "fanin_anneal_epochs": anneal,
        "query_norm_penalty_form": penalty_form,
    }


class FaninPriorSchedule:
    """The fan-in squeeze clock, shared by both selector forecasters.

    Owns three things a forecaster would otherwise duplicate: the STRUCTURE-
    epoch counter, the write of ``mu(t)`` onto every module, and the
    diagnostics.

    Why a structure clock and not ``current_epoch``?  Under adaptive training
    everything runs in a single ``fit()``, so the global epoch is consumed by
    the (long) reconstruct phases and a raw window expires before the structure
    has moved at all - the same lesson that produced
    ``_descendant_warmup_anchor``.  ``query_norm_log_scale`` is routed as a
    STRUCTURAL parameter, so structural time is the correct clock: the counter
    only advances while ``in_structure_phase`` is True (the adaptive trainer
    flips it; the plain trainers leave it True forever, which reproduces the
    global-epoch behaviour exactly).
    """

    def __init__(self, config: Any, n_keys: int):
        exp = config.get("experiment", {}) if hasattr(config, "get") else {}
        train = config.get("training", {}) if hasattr(config, "get") else {}
        prior = exp.get("fanin_prior", None) if hasattr(exp, "get") else None
        self.n_keys = int(n_keys)
        self.k_star: Optional[int] = None if prior is None else int(prior)
        anneal = train.get("fanin_anneal_epochs", 0) if hasattr(train, "get") else 0
        if anneal is None or isinstance(anneal, str):
            # Unresolved 'auto': resolve_query_norm did not run (e.g. a model
            # built straight from a dict in a test).  One edge per epoch.
            anneal = max(self.n_keys - (self.k_star or self.n_keys), 0)
        self.anneal_epochs = int(anneal)
        self.penalty_form = str(
            exp.get("query_norm_penalty_form", "absolute")
            if hasattr(exp, "get") else "absolute"
        )
        self.struct_epoch = 0
        self.in_structure_phase = True
        self.k_t = float(self.n_keys)
        self.mu_t = 1.0

    @property
    def enabled(self) -> bool:
        return self.k_star is not None and self.k_star < self.n_keys

    def on_epoch_start(self, model: torch.nn.Module) -> None:
        """Write ``mu(t)`` on every module, then advance the structural clock.

        Idempotent, so it is safe to re-run on a phase switch.  The write
        happens BEFORE the increment, so epoch 0 sees ``mu(0) = 1`` exactly and
        the initialisation is never destroyed.
        """
        if not self.enabled:
            return
        self.k_t, self.mu_t = capacity_schedule(
            self.struct_epoch, self.anneal_epochs, self.n_keys, self.k_star)
        set_query_norm_target(model, self.mu_t, self.penalty_form)
        if self.in_structure_phase:
            self.struct_epoch += 1

    def metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Diagnostics: the target, the scheduled capacity and the realised one.

        ``cap_actual_edges`` is the fan-in the average row could hold at ``p*``
        with its CURRENT budget (Lemma 1); it should track ``cap_target_edges``
        down if the squeeze prunes, and the gap is the buy-out of eq (11).
        """
        out = {
            "query_norm/target_mu": self.mu_t,
            "query_norm/cap_target_edges": self.k_t,
            "query_norm/struct_epoch": float(self.struct_epoch),
        }
        actual = query_norm_capacity(model, self.n_keys)
        if actual is not None:
            out["query_norm/cap_actual_edges"] = actual
        for form, value in query_norm_penalty_by_form(model).items():
            out[f"query_norm/penalty_{form}"] = value
        return out


def format_query_norm_log(info: Dict[str, Any]) -> str:

    """Human-readable startup block for the dict returned by ``resolve_query_norm``."""
    gate = ("saturated" if info["z_init"] >= 1.0
            else ("open" if info["z_init"] > 0.0 else "CLOSED"))
    lines = [
        "[query-norm] n_keys={n} | p*={p:.4f} (saturates at {ps:.4f}) x={x:.4f}".format(
            n=info["n_keys"], p=info["p_star"], ps=info["p_saturation"], x=info["x"]),
        "             F={f:.4f} (query_fanin_scale={fs:.4f}){pin} | centroid_init={ci}".format(
            f=info["F"], fs=info["query_fanin_scale"],
            pin=" [pinned by config]" if info["query_fanin_scale_explicit"] else "",
            ci=info["query_centroid_init"]),
        "             pi_init={pi:.4f} z_init={z:.4f} ({gate})".format(
            pi=info["pi_init"], z=info["z_init"], gate=gate),
    ]
    if info["fanin_prior"] is None:
        lines.append("             fanin_prior=off -> mu=1 (no in-degree prior)")
    else:
        lines.append(
            "             fanin_prior={k} edges -> mu_end={mu:.4f} | anneal={a} "
            "struct epochs | penalty={form}".format(
                k=info["fanin_prior"], mu=info["mu_end"],
                a=info["fanin_anneal_epochs"], form=info["query_norm_penalty_form"]))
    return "\n".join(lines)



