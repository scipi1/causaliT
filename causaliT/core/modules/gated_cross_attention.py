"""
GatedCrossAttention: disentangled structure-gate x reconstruction-gain attention.

Motivation
==========
In the original ``CausalCrossAttention`` / ``HardConcreteCrossAttention`` the
single edge score ``A_ij`` has to do two incompatible jobs at once:

* **Selection** — decide whether variable ``j`` is a parent of ``i`` (binary,
  structure; driven by HSIC + L0 / sparsity).
* **Gain** — decide *how strongly* ``j`` contributes to reconstructing ``i``
  (continuous, functional; driven by the MSE reconstruction loss).

Because the value stream ``V_j`` is *source-shared* (the same ``v_j`` is offered
to every child ``i``), the only place a per-child functional gain can live is
the score ``A_ij`` itself.  Forcing one scalar to carry both meanings conflates
structure with magnitude: the score cannot be pushed to a clean, thresholdable
{0,1} adjacency without destroying reconstruction, and vice-versa (the
first-phase R^2 collapse observed in the ARCH study).

Design
======
This module factorises the edge score into a **product of two independent,
target-conditioned scores**::

    A_ij = z_ij  *  g_ij

* ``z_ij`` — **structure gate**.  A Hard-Concrete L0 gate (Louizos et al.,
  ICLR 2018) computed from a *structural* query/key pair
  ``z_ij = HardConcrete(<q^s_i, k^s_j> / sqrt(E_s))``.  Stochastic-binary in
  {0,1}; its expected-active-edge count ``sum P(z_ij > 0)`` is the L0 penalty.
  The structural query/key projections + embeddings are classified as
  *structural* parameters, so ``z`` is driven by HSIC + L0 only.

* ``g_ij`` — **reconstruction gain**.  A bounded (sigmoid) score computed from a
  *separate* gain query/key pair ``g_ij = sigmoid(<q^g_i, k^g_j> / (tau*sqrt(E_g)))``
  in ``(0, 1)``.  It can only *dampen* the gate (>1 amplification is left to
  ``W_V`` / FFN / head).  The gain query/key projections + embeddings are
  classified as *reconstruction* parameters, so ``g`` is driven by the MSE
  reconstruction loss only.

Why this is rigorous
====================
The reconstruction loss sees only the product ``A = z*g``, so *a priori* its
gradient flows into both factors.  The disentanglement is enforced by the
existing name-based dual-optimizer gradient routing
(``causaliT/training/gradient_routing.py``): during the structural backward the
reconstruction-group gradients are overwritten by the saved MSE gradients, so

* gate params  (structural)  receive HSIC + L0 gradients only, and
* gain params  (reconstruction) receive MSE gradients only.

For this reason ``GatedCrossAttention`` MUST be trained with
``use_gradient_routing=True``.  With routing off the product re-conflates.

The structure regularisers (score-sparsity, L0, NOTEARS) read
``score_tensor_for_sparsity`` / the returned ``l0_penalty``, both of which are
functions of the **gate** only — never the gain — so they act purely on
structure.  Evaluation should threshold ``last_p_edge_on`` (the gate posterior
edge probability) to obtain the recovered adjacency.

Contract
========
Mirrors the other inner-attention modules:
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False, gain_query=None, gain_key=None)``
returns ``(out, attn, aux)`` with
``aux = {"entropy": Tensor|None, "l0_penalty": Tensor}``.

``query`` / ``key`` are the *structural* (gate) projections; ``gain_query`` /
``gain_key`` are the *gain* projections.  All are expected pre-projected with a
single structural head (3-D: ``(B, L, E)`` / ``(B, S, E)``) — i.e.
``shared_dag_across_heads=True``.  ``value`` may be 3-D ``(B, S, d)`` or 4-D
``(B, S, H, d)`` (multiple value heads sharing the single gate/gain score).
"""

from typing import Optional

import math
import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """Structure-gate x reconstruction-gain cross attention (see module docstring)."""

    def __init__(
        self,
        attention_dropout: float = 0.0,
        register_entropy: bool = False,
        layer_name: Optional[str] = None,
        # Hard-Concrete gate hyper-parameters (Louizos et al., ICLR 2018).
        init_tau: float = 2.0 / 3.0,   # beta: gate temperature
        gamma: float = -0.1,           # stretch lower bound (< 0)
        zeta: float = 1.1,             # stretch upper bound (> 1)
        # Reconstruction-gain temperature (scales the sigmoid logit).
        gain_tau: float = 1.0,
        # Batch-consistent key dropout (columns zeroed identically across batch).
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        # Constant-score capacity protocol (Optuna): when not None the STRUCTURE
        # gate z is frozen at this constant on every edge (0 = residual floor,
        # 1 = uniform mixing) while the reconstruction gain g stays learnable.
        optuna_protocol: Optional[float] = None,
    ):
        super().__init__()


        if not (gamma < 0.0 < zeta and zeta > 1.0):
            raise ValueError(
                f"HardConcrete stretch bounds require gamma < 0 < 1 < zeta, "
                f"got gamma={gamma}, zeta={zeta}."
            )

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name

        # Gate params are non-learnable constants (matching HardConcreteCrossAttention).
        self.beta = float(init_tau)
        self.gamma = float(gamma)
        self.zeta = float(zeta)

        self.gain_tau = float(gain_tau)

        # Constant-score capacity protocol (gate-only override); see forward().
        self.optuna_protocol: Optional[float] = (
            float(optuna_protocol) if optuna_protocol is not None else None
        )


        # Batch-consistent key dropout probability (linear anneal, optional).
        self._bkd_p0 = batch_key_dropout
        self._bkd_p1 = (
            batch_key_dropout_p_final
            if batch_key_dropout_p_final is not None
            else batch_key_dropout
        )
        self._bkd_anneal = batch_key_dropout_annealing_batches
        self.register_buffer("_bkd_step", torch.zeros((), dtype=torch.long), persistent=False)

        # Diagnostics / regularisation hooks (populated in forward).
        #   score_tensor_for_sparsity — the GATE posterior edge prob (B-mean),
        #     read by the L1 score-sparsity and NOTEARS terms.
        #   last_p_edge_on           — the GATE posterior edge prob (B-mean),
        #     thresholded at eval to obtain the recovered adjacency.
        #   last_gain                — the reconstruction gain (B-mean), logged
        #     for diagnostics only (never regularised).
        self.score_tensor_for_sparsity: Optional[torch.Tensor] = None
        self.last_p_edge_on: Optional[torch.Tensor] = None
        self.last_gain: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Batch-consistent key dropout probability (with optional annealing)
    # ------------------------------------------------------------------
    def _current_bkd_p(self) -> Optional[float]:
        if self._bkd_p0 is None:
            return None
        if self._bkd_anneal is None or self._bkd_anneal <= 0:
            return float(self._bkd_p0)
        frac = min(1.0, float(self._bkd_step.item()) / float(self._bkd_anneal))
        return float(self._bkd_p0) + frac * (float(self._bkd_p1) - float(self._bkd_p0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: Optional[torch.Tensor] = None,
        mask_miss_q: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        hard_mask: Optional[torch.Tensor] = None,
        oracle: bool = False,
        gain_query: Optional[torch.Tensor] = None,
        gain_key: Optional[torch.Tensor] = None,
    ):
        if causal_mask:

            raise NotImplementedError(
                "GatedCrossAttention does not support causal masking."
            )
        if query.dim() != 3 or key.dim() != 3:
            raise ValueError(
                "GatedCrossAttention expects a single structural head "
                "(3-D query/key: (B, L, E) / (B, S, E)); use "
                "shared_dag_across_heads=True."
            )
        if gain_query is None or gain_key is None:
            raise ValueError(
                "GatedCrossAttention requires gain_query and gain_key "
                "(the reconstruction-gain projections)."
            )

        B, L, E_s = query.shape
        _, S, _ = key.shape

        # ---- Structure gate: Hard-Concrete L0 gate -----------------------
        # log_alpha = <q^s, k^s> / sqrt(E_s)     (B, L, S)
        scale_s = 1.0 / math.sqrt(E_s)
        log_alpha = torch.einsum("ble,bse->bls", query, key) * scale_s

        if oracle:
            # ---- Oracle gate: the ground-truth DAG IS the structure gate ----
            # z_ij = hard_mask_ij (true topology, held constant); only the
            # learned reconstruction gain g modulates edge magnitude.  Used to
            # measure the reconstruction ceiling given the correct structure.
            if hard_mask is None:
                raise ValueError(
                    "GatedCrossAttention oracle mode requires hard_mask "
                    "(the ground-truth adjacency used as the structure gate)."
                )
            hm_gate = hard_mask
            if hm_gate.dim() == 2:
                hm_gate = hm_gate.unsqueeze(0)               # (1, L, S)
            z = hm_gate.to(log_alpha.dtype).expand(B, L, S)
            p_edge_on = z                                    # diagnostics = oracle
        elif self.optuna_protocol is not None:
            # ---- Constant-score capacity protocol (gate-only override) ------
            # Freeze the STRUCTURE gate at a constant on every edge
            # (0 = residual floor, 1 = uniform mixing) so the value/gain
            # reconstruction capacity is measurable independently of an
            # untrained structure.  The gain g below stays fully learnable, and
            # a heavy batch_key_dropout forces reconstruction from random parent
            # subsets.  Applies in BOTH train and eval (deterministic).
            z = torch.full_like(log_alpha, float(self.optuna_protocol))
            p_edge_on = torch.full_like(log_alpha, float(self.optuna_protocol))
        else:
            if self.training:
                # Stochastic Hard-Concrete sample.
                u = torch.rand_like(log_alpha).clamp_(1e-6, 1.0 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / self.beta)
            else:
                # Deterministic (mean) gate at eval.
                s = torch.sigmoid(log_alpha / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            z = s_bar.clamp(0.0, 1.0)                     # (B, L, S) in [0, 1]

            # Posterior probability that the gate is open: P(z > 0).
            #   P(z>0) = sigmoid(log_alpha - beta * log(-gamma / zeta))
            p_edge_on = torch.sigmoid(
                log_alpha - self.beta * math.log(-self.gamma / self.zeta)
            )


        # ---- Reconstruction gain: bounded sigmoid score ------------------
        E_g = gain_query.shape[-1]
        scale_g = 1.0 / (self.gain_tau * math.sqrt(E_g))
        gain_logit = torch.einsum("ble,bse->bls", gain_query, gain_key) * scale_g
        g = torch.sigmoid(gain_logit)                   # (B, L, S) in (0, 1)

        # ---- Combined edge weight ----------------------------------------
        A = z * g                                        # (B, L, S)

        # ---- Structural hard mask (allowed-edge topology) ----------------
        # Applied to BOTH the aggregation weight and the L0 penalty so that
        # forbidden edges (e.g. X self-loops) never contribute or count.
        if hard_mask is not None:
            hm = hard_mask
            if hm.dim() == 2:
                hm = hm.unsqueeze(0)                      # (1, L, S)
            hm = hm.to(A.dtype)
            A = A * hm
            p_edge_masked = p_edge_on * hm
        else:
            p_edge_masked = p_edge_on

        # ---- L0 penalty: expected number of active (allowed) edges -------
        l0_penalty = p_edge_masked.sum(dim=(-2, -1)).mean()

        # ---- Batch-consistent key dropout --------------------------------
        bkd_p = self._current_bkd_p()
        if self.training and bkd_p is not None and bkd_p > 0.0:
            keep = (torch.rand(S, device=A.device) >= bkd_p).to(A.dtype)  # (S,)
            A = A * keep.view(1, 1, S)
            self._bkd_step += 1

        # ---- Attention-weight dropout ------------------------------------
        A = self.dropout(A)

        # ---- Value aggregation -------------------------------------------
        # value: (B, S, d) single head  OR  (B, S, H, d) multiple value heads
        # sharing the single gate/gain score.
        if value.dim() == 4:
            out = torch.einsum("bls,bshd->blhd", A, value)   # (B, L, H, d)
        elif value.dim() == 3:
            out = torch.einsum("bls,bsd->bld", A, value)     # (B, L, d)
        else:
            raise ValueError(
                f"GatedCrossAttention value must be 3-D or 4-D, got {value.dim()}-D."
            )

        # ---- Diagnostics / regularisation signals (gate-only) ------------
        # Batch-mean (2-D) tensors so the forecaster's score-sparsity / NOTEARS
        # paths (which expect 2-D score tensors) operate on the GATE.
        self.score_tensor_for_sparsity = p_edge_on.mean(dim=0)   # (L, S) gate
        self.last_p_edge_on = p_edge_on.mean(dim=0)              # (L, S) gate
        self.last_gain = g.mean(dim=0).detach()                 # (L, S) diag only

        # ---- Entropy (over the combined weights, for logging) ------------
        entropy = None
        if self.register_entropy:
            w = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)  # (B, L)

        aux = {"entropy": entropy, "l0_penalty": l0_penalty}
        return out, A, aux

    def __repr__(self):
        return (
            f"GatedCrossAttention(beta={self.beta}, gamma={self.gamma}, "
            f"zeta={self.zeta}, gain_tau={self.gain_tau})"
        )
