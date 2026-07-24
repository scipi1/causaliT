"""Elastic contraction of the structural query normalization.

Motivation
==========
When ``normalize_query=True`` the structural attention modules
(``GatedCrossAttention`` / ``GatedSelfAttention`` / ``CommutatorSelfAttention``)
L2-normalise the query onto the UNIT sphere and score with a fixed
``sqrt(query_fanin_scale)`` temperature.  With an orthonormal key frame this
caps the total alignment energy ``sum_j cos^2(q_hat, k_j) <= 1`` — a HARD
directional budget of 1 from the very first optimiser step.

The SELF_ATTENTION spurious-``S3->X4`` investigation showed this cap is reached
almost immediately: every candidate edge lights up together, saturates the
budget, and is then shoved back into a zero-sum competition, damping genuinely
good multi-parent pushes before they can pay off (the "hump-before-valley"
barrier).

Elastic contraction relaxes that cap EARLY and gently contracts it back to the
target as training proceeds.  Concretely we keep the direction-only selection
philosophy but make the effective query norm a scheduled multiplier ``M(e)``::

    q_eff = (q / ||q||) * M(e)          # unit direction, scheduled norm
    score = <q_eff, k> * sqrt(fanin)    # == M(e) * (current score)

so the directional budget is ``M(e)^2``.  ``M`` starts at ``start_scale`` (>= 1,
loosening the cap so exploration is free — no premature pushback) and linearly
contracts to ``end_scale`` (typically 1.0, the original hard cap) between
``start_epoch`` and ``end_epoch``.  With ``start_scale == end_scale == 1.0``
(or ``enabled=False``) this is EXACTLY the original unit-normalisation.

Epoch plumbing
==============
The schedule is expressed in EPOCHS, but a submodule does not know the current
epoch during ``forward``.  Each module therefore carries an ``_elastic_epoch``
buffer that the trainer pushes once per epoch via :func:`set_elastic_query_epoch`
from its ``on_train_epoch_start`` hook.  If the epoch is never pushed the buffer
stays at its default so the schedule evaluates to ``end_scale`` (fully
contracted == original behaviour), so untracked trainers degrade gracefully.
"""

from typing import Tuple

import math
import torch
import torch.nn.functional as F


class ElasticQueryNormConfig:
    """Container for the elastic query-norm schedule hyper-parameters.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False the helper reproduces plain unit
        normalisation (multiplier fixed at 1.0).
    start_epoch, end_epoch : int
        Linear-schedule endpoints (inclusive).  ``M = start_scale`` for
        ``epoch <= start_epoch`` and ``M = end_scale`` for ``epoch >= end_epoch``.
    start_scale, end_scale : float
        Query-norm multiplier at/before ``start_epoch`` and at/after
        ``end_epoch``.  ``start_scale`` typically > 1 (relaxed budget),
        ``end_scale`` typically 1.0 (original hard cap).
    """

    __slots__ = ("enabled", "start_epoch", "end_epoch", "start_scale", "end_scale")

    def __init__(
        self,
        enabled: bool = False,
        start_epoch: int = 0,
        end_epoch: int = 0,
        start_scale: float = 1.0,
        end_scale: float = 1.0,
    ):
        self.enabled = bool(enabled)
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)

    def multiplier(self, epoch: float) -> float:
        """Linear query-norm multiplier ``M(epoch)`` (returns 1.0 when disabled)."""
        if not self.enabled:
            return 1.0
        e = float(epoch)
        s0, s1 = self.start_epoch, self.end_epoch
        if s1 <= s0:
            # Degenerate (zero/negative-width) window: instant contraction at
            # start_epoch -> start_scale strictly before it, end_scale at/after.
            return self.start_scale if e < s0 else self.end_scale
        if e <= s0:
            return self.start_scale
        if e >= s1:
            return self.end_scale
        frac = (e - s0) / (s1 - s0)
        return self.start_scale + frac * (self.end_scale - self.start_scale)


    def __repr__(self) -> str:
        return (
            f"ElasticQueryNormConfig(enabled={self.enabled}, "
            f"start_epoch={self.start_epoch}, end_epoch={self.end_epoch}, "
            f"start_scale={self.start_scale}, end_scale={self.end_scale})"
        )


def elastic_normalize_query(
    query: torch.Tensor,
    query_fanin_scale: float,
    cfg: ElasticQueryNormConfig,
    epoch: int,
) -> Tuple[torch.Tensor, float]:
    """Unit-normalise ``query`` and apply the elastic norm multiplier.

    Parameters
    ----------
    query : torch.Tensor
        Structural query, normalised along the last dim.
    query_fanin_scale : float
        Fixed score temperature; the returned ``scale_s`` is ``sqrt(fanin)``.
    cfg : ElasticQueryNormConfig
        Schedule hyper-parameters.
    epoch : int
        Current training epoch (pushed by the trainer; see module docstring).

    Returns
    -------
    (q_s, scale_s)
        ``q_s = (query/||query||) * M(epoch)`` and ``scale_s = sqrt(fanin)``.
        With ``cfg.enabled=False`` (or both scales 1.0) ``M == 1`` and this is
        identical to plain ``F.normalize`` + ``sqrt(fanin)``.
    """
    q_hat = F.normalize(query, p=2.0, dim=-1, eps=1e-8)
    m = cfg.multiplier(epoch)
    if m != 1.0:
        q_hat = q_hat * m
    scale_s = math.sqrt(query_fanin_scale)
    return q_hat, scale_s


def set_elastic_query_epoch(model: torch.nn.Module, epoch: int) -> int:
    """Push ``epoch`` into every submodule carrying an ``_elastic_epoch`` buffer.

    Call from the trainer's ``on_train_epoch_start`` so the epoch-based schedule
    inside the structural attention modules advances.  Returns the number of
    modules updated (0 when the elastic feature is unused).
    """
    n = 0
    for m in model.modules():
        buf = getattr(m, "_elastic_epoch", None)
        if buf is not None and torch.is_tensor(buf):
            buf.fill_(int(epoch))
            n += 1
    return n
