"""
Prior-softmax reconstruction gain for the gated attentions.

Design
======
The gated attentions (``GatedCrossAttention`` / ``GatedSelfAttention``) apply
the structure gate directly as the attention weight (``A = z``): every selected
parent contributes with weight ~1, so no per-edge coefficient is expressible.
This module adds the **gain** stage discussed in the ATE_FIXCAP design notes::

    A_ij = (1 - lambda) * z_ij  +  lambda * n_i * z_ij * exp(s_ij) / D_i
    D_i  = sum_k z_ik * exp(s_ik)          (row normaliser)
    n_i  = sum_k z_ik                       (row mass, DETACHED)
    s_ij = a_ij + <q^v_i, k^v_j> / sqrt(d_g)

i.e. at ``lambda = 1`` the applied weight is ``n_i * softmax(log z + s)``: the
gate enters the softmax as a **multiplicative prior** (equivalently a
``log z`` additive log-prior), and the gain score ``s`` only REDISTRIBUTES the
row's existing mass ``n_i`` across the gate's own support.

Properties (all by construction)
--------------------------------
* ``lambda = 0``        -> ``A = z``: the current method runs bit-identically.
* ``s = 0`` (zero-init) -> ``softmax(log z) = z / n_i``, so ``A = z`` at ANY
  lambda: the gain is an exact identity at initialisation and departs from
  uniform only when the reconstruction loss pulls it there.
* ``z_ij = 0``          -> ``A_ij = 0`` EXACTLY (both terms): forbidden and
  gated-off edges keep their by-construction zeros (the ATE "zero" categories
  and the DAG extraction semantics are unchanged).
* Row sum is ``n_i`` at every lambda: the gain can neither amplify beyond the
  gate's row mass nor collapse a row to zero - the failure mode of the former
  unnormalised gain stream (``A = z * g``) is structurally impossible.
* Differentiable in ``z`` everywhere, INCLUDING ``z = 0`` (the product form
  avoids evaluating ``log z``): ``dA_j/dz_j = exp(s_j)/D > 0`` at ``z_j = 0``,
  so a falsely closed edge receives finite re-opening pressure from the
  reconstruction loss and the structure keeps training in its new role as the
  softmax prior.  The gradient reaches the structural logits through the same
  Hard-Concrete relaxation the L0 penalty already uses.

Guards
------
* A row that sampled all-off gates has ``D_i = 0``: the gain term is zeroed
  for that row (``torch.where``), reproducing the current method's zero row.
* ``n_i`` is detached so the gain cannot inflate the row mass; the gate alone
  owns it.

The gain score ``s`` is supplied by the caller (``AttentionLayer``) as a
precomputed ``(B, L, S)`` tensor; the static per-edge logit table ``a_ij``
(zero-initialised, ``(L, S)``) lives here.  All gain parameters are named
WITHOUT the structural routing patterns (``query_projection`` /
``key_projection`` / ``query_embed`` / ``log_gain`` / ...) so the name-based
gradient router classifies them as RECONSTRUCTION parameters.
"""

from typing import Optional

import torch
import torch.nn as nn


class GainSoftmax(nn.Module):
    """Prior-softmax gain: redistribute the gate's row mass within its support.

    Args:
        num_queries: Number of query rows (children), L.
        num_keys:    Number of key columns (candidate parents), S.
    """

    def __init__(self, num_queries: int, num_keys: int):
        super().__init__()
        # Static per-edge logit a_ij, zero-initialised so the gain is an exact
        # identity at init (s = 0 -> softmax(log z) = z / n_i -> A = z).
        # Named without any structural routing pattern -> RECONSTRUCTION group.
        self.gain_static_logits = nn.Parameter(
            torch.zeros(int(num_queries), int(num_keys))
        )
        # Schedule buffer: the trainer ramps it 0 -> 1 in the gain phase.
        # PERSISTENT so a trained checkpoint carries its final lambda and the
        # evaluation suite reloads the model with the gain ACTIVE (a
        # non-persistent buffer would silently evaluate the gate-only model).
        self.register_buffer(
            "gain_lambda", torch.zeros((), dtype=torch.float32), persistent=True
        )
        # Diagnostics hook (batch-mean applied gain multiplier), populated in
        # forward() when the gain is active.
        self.last_gain: Optional[torch.Tensor] = None

    def set_gain_lambda(self, value: float) -> None:
        """Set the gain interpolation weight lambda in [0, 1]."""
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"gain_lambda must be in [0, 1], got {v}.")
        self.gain_lambda.fill_(v)

    @property
    def active(self) -> bool:
        """Whether the gain contributes to the applied weights right now."""
        return float(self.gain_lambda.item()) > 0.0

    def forward(
        self,
        gate: torch.Tensor,
        gain_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the prior-softmax gain to a gate matrix.

        Args:
            gate:        The applied structure gate ``z`` AFTER diagonal zeroing
                         and hard masking, shape ``(B, L, S)``.  Used as the
                         multiplicative softmax prior.
            gain_scores: Data-dependent score term ``<q^v, k^v> / sqrt(d_g)``,
                         shape ``(B, L, S)``, or None (static-logit-only gain).

        Returns:
            The attention weight ``(B, L, S)``: ``(1-lambda) * gate + lambda *
            n * gate * exp(s) / D``.  At ``lambda == 0`` the gate is returned
            unchanged (no gain computation, no diagnostics write).
        """
        lam = float(self.gain_lambda.item())
        if lam <= 0.0:
            return gate

        s = self.gain_static_logits  # (L, S), broadcasts over the batch
        if gain_scores is not None:
            s = s + gain_scores

        # Weighted softmax with the gate as multiplicative prior (the numerically
        # stable form of softmax(log z + s); never evaluates log z).
        numer = gate * torch.exp(s)
        D = numer.sum(dim=-1, keepdim=True)                       # (B, L, 1)
        n = gate.sum(dim=-1, keepdim=True).detach()               # row mass
        # All-off row (D == 0): the gain term is zeroed -> the row stays a zero
        # row, exactly the gate-only behaviour for that row.
        safe_D = D.clamp_min(1e-20)
        m = torch.where(D > 1e-20, numer / safe_D, torch.zeros_like(numer))
        self.last_gain = (n * m).mean(dim=0).detach()             # (L, S)
        return (1.0 - lam) * gate + lam * n * m
