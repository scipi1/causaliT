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
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
