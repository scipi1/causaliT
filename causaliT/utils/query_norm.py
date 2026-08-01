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
"""

from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Default centroid edge posterior used when the config does not set one.
DEFAULT_CENTROID_MAX_P = 0.9


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


def overspend_penalty(log_scale: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """Per-node over-spend penalty ``sum_i relu(M_i - target)^2`` (``M = exp(log_scale)``).

    Only budget grown ABOVE ``target`` is charged; contracting below the target
    is free (and unincentivised, since the structural signal generally prefers a
    larger budget).
    """
    m = torch.exp(log_scale)
    return F.relu(m - float(target)).pow(2).sum()


def collect_query_norm_penalty(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """Sum the over-spend penalty over every module owning a learnable multiplier.

    Tied multipliers (a single ``log_scale`` shared by the cross & self blocks
    under ``shared_query=True``) are DEDUPLICATED by parameter id so they are
    charged exactly once.  Returns ``None`` when no learnable multiplier exists.
    """
    total: Optional[torch.Tensor] = None
    seen = set()
    for m in model.modules():
        p = getattr(m, "query_norm_log_scale", None)
        if p is None or not getattr(m, "query_norm_learnable", False):
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        pen = overspend_penalty(p, getattr(m, "query_norm_target", 1.0))
        total = pen if total is None else total + pen
    return total



def query_norm_stats(model: torch.nn.Module):
    """Return ``(mean_M, max_M)`` across all learnable multipliers (deduped).

    Returns ``(None, None)`` when no learnable multiplier exists.  Values are
    detached (diagnostics only).
    """
    scales = []
    seen = set()
    for m in model.modules():
        p = getattr(m, "query_norm_log_scale", None)
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        scales.append(torch.exp(p.detach()).reshape(-1))
    if not scales:
        return None, None
    allm = torch.cat(scales)
    return allm.mean(), allm.max()


# =============================================================================
# Automatic query_fanin_scale
# =============================================================================

def query_fanin_scale_from_centroid_p(
    n_keys: int,
    max_p: float = DEFAULT_CENTROID_MAX_P,
    init_tau: float = 0.5,
    init_gamma: float = -1.1,
    init_zeta: float = 1.1,
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
    """
    if not 0.0 < float(max_p) < 1.0:
        raise ValueError(
            f"query_centroid_max_p must be in (0, 1), got {max_p!r}: it is an "
            "edge POSTERIOR, so 1.0 is unreachable (use ~0.5-0.9)."
        )
    if int(n_keys) < 1:
        raise ValueError(f"n_keys must be >= 1, got {n_keys!r}.")
    if float(init_tau) <= 0.0 or float(init_gamma) >= 0.0 or float(init_zeta) <= 1.0:
        raise ValueError(
            "Hard-Concrete needs init_tau > 0, init_gamma < 0, init_zeta > 1; got "
            f"{init_tau!r}, {init_gamma!r}, {init_zeta!r}."
        )
    if float(query_norm_init_scale) <= 0.0:
        raise ValueError(
            f"query_norm_init_scale must be > 0, got {query_norm_init_scale!r}."
        )

    stretch = float(init_tau) * math.log(-float(init_gamma) / float(init_zeta))
    x = math.log(max_p / (1.0 - max_p)) + float(init_edge_offset) + stretch
    if x <= 0.0:
        raise ValueError(
            f"query_centroid_max_p={max_p} needs a non-positive score (x={x:.4g}) "
            f"given init_edge_offset={init_edge_offset} and stretch={stretch:.4g}; "
            "raise query_centroid_max_p."
        )
    return float(n_keys) * (x / float(query_norm_init_scale)) ** 2


def is_auto_fanin(value: Any) -> bool:
    """True when ``query_fanin_scale`` asks to be derived (``auto`` / ``null``)."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in ("auto", "derive", "null")


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

    offset = 0.0 if bool(_get("homogeneous_nodes", False)) else float(
        _get("init_edge_offset", 0.0))
    max_p = float(_get("query_centroid_max_p", DEFAULT_CENTROID_MAX_P))
    fanin = query_fanin_scale_from_centroid_p(
        n_keys=n_keys,
        max_p=max_p,
        init_tau=float(_get("init_tau", 0.5)),
        init_gamma=float(_get("init_gamma", -1.1)),
        init_zeta=float(_get("init_zeta", 1.1)),
        init_edge_offset=offset,
        query_norm_init_scale=float(_get("query_norm_init_scale", 1.0)),
    )
    exp["query_fanin_scale"] = fanin
    return {"query_fanin_scale": fanin, "n_keys": int(n_keys),
            "query_centroid_max_p": max_p, "init_edge_offset": offset}


