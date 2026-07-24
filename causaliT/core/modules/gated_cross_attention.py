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

Mirrors the other inner-attention modules:
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False, gain_query=None, gain_key=None)``
returns ``(out, attn, aux)`` with
``aux = {"entropy": Tensor|None, "l0_penalty": Tensor}``.
Contract
Mirrors the other inner-attention modules:
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False, gain_query=None, gain_key=None)``
returns ``(out, attn, aux)`` with
``aux = {"entropy": Tensor|None, "l0_penalty": Tensor}``.

GCA-SPECIFIC use of the second return slot
------------------------------------------
For every *other* attention module the second element ``attn`` is the weight
matrix actually applied to the values.  For GCA that applied weight is the
conflated product ``A = z*g`` (structure gate x reconstruction gain), which is
**not** a clean structural signal.  Because ``out`` (first element) is already
computed from ``A`` internally and **no downstream calculation consumes the
returned ``attn`` for reconstruction** — the training loss reads ``out``,
the residuals, the ``score_tensor_for_sparsity`` attribute and ``aux``; only
the evaluation/DAG-extraction path reads the returned ``attn`` — GCA reuses this
slot to emit the **structure gate posterior** ``p_edge_on * hard_mask`` (the
masked ``P(z>0)`` edge-existence probability, shape ``(B, L, S)``, values in
``(0, 1)``, forbidden/diagonal edges = 0).  Evaluation can therefore threshold
the returned matrix at 0.5 to recover the adjacency with no special-casing.
This is intentional and scoped to GCA only; it is not a convention imposed on
the other modules.

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
import torch.nn.functional as F

from causaliT.utils.elastic_query import (
    ElasticQueryNormConfig,
    elastic_normalize_query,
)



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
        # Additive logit offset on the STRUCTURE gate (init-balancing).  The
        # effective alignment logit becomes ``log_alpha - init_edge_offset`` in
        # BOTH the Hard-Concrete sample and the P(z>0) posterior, so at init
        # (``log_alpha ~= 0``) the edge-existence posterior is
        # ``sigmoid(-init_edge_offset)`` instead of 0.5.  Set it to ``ln 3``
        # (~1.0986) to bring a directed S->X cross edge to the same 0.25 init
        # probability a directed X->X self edge gets from its undecided
        # direction gate (P = p_exist * 0.5), removing the 2x head start the
        # cross edges otherwise enjoy (see COND_INDEPENDENCE investigation
        # investigate_S2_X5_spurious_first.ipynb).  Default 0.0 = no offset.
        init_edge_offset: float = 0.0,
        # Reconstruction-gain temperature (scales the sigmoid logit).
        gain_tau: float = 1.0,

        # When False, BYPASS the learnable reconstruction gain entirely: the
        # structure gate ``z`` becomes the final attention weight (A = z) instead
        # of A = z * g.  ``gain_query`` / ``gain_key`` are then optional and the
        # gain projections/embeddings can be omitted upstream.  Default True
        # preserves the original disentangled structure-gate x gain behaviour.
        use_gain: bool = True,
        # Centroid-collapse fix: L2-normalise the STRUCTURAL query before the
        # score so its *direction* (not its norm) decides selection, and replace
        # the 1/sqrt(E) score scale with a fixed sqrt(query_fanin_scale).  With
        # unit query and (orthonormal) unit keys the raw score of a single
        # aligned parent is sqrt(query_fanin_scale); a centroid over m parents
        # gives sqrt(query_fanin_scale/m) per edge — so an all-on "block" row is
        # sqrt(m) cheaper per edge and pays a growing L0 cost, while true parents
        # (up to in-degree query_fanin_scale) stay clearly ON.  Only affects the
        # structure gate, never the reconstruction gain.
        normalize_query: bool = False,
        query_fanin_scale: float = 1.0,
        # Elastic contraction of the query normalization (see
        # ``causaliT/utils/elastic_query.py``).  Only meaningful when
        # ``normalize_query=True``: relaxes the unit-norm cap to
        # ``query_norm_elastic_start_scale`` early (looser directional budget so
        # exploration is free) and linearly contracts it to
        # ``query_norm_elastic_end_scale`` (typically 1.0) between the two
        # epochs.  ``enabled=False`` (or both scales 1.0) == original behaviour.
        query_norm_elastic: bool = False,
        query_norm_elastic_start_epoch: int = 0,
        query_norm_elastic_end_epoch: int = 0,
        query_norm_elastic_start_scale: float = 1.0,
        query_norm_elastic_end_scale: float = 1.0,
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

        # Additive logit offset on the structure gate (init-balancing); see the
        # ``init_edge_offset`` docstring above.  Non-learnable constant.
        self.edge_offset = float(init_edge_offset)

        self.gain_tau = float(gain_tau)


        # When False the reconstruction gain is bypassed and A = z (the gate).
        self.use_gain = bool(use_gain)

        # Centroid-collapse fix (structure gate only): unit-normalise the query
        # and use a fixed sqrt(query_fanin_scale) score scale (see __init__ doc).
        self.normalize_query = bool(normalize_query)
        self.query_fanin_scale = float(query_fanin_scale)

        # Elastic query-norm schedule (only active with normalize_query=True).
        self.elastic_query_cfg = ElasticQueryNormConfig(
            enabled=query_norm_elastic,
            start_epoch=query_norm_elastic_start_epoch,
            end_epoch=query_norm_elastic_end_epoch,
            start_scale=query_norm_elastic_start_scale,
            end_scale=query_norm_elastic_end_scale,
        )
        # Current training epoch, pushed by the trainer's on_train_epoch_start
        # via causaliT.utils.elastic_query.set_elastic_query_epoch.  Defaults to
        # end_epoch so an un-pushed module evaluates M=end_scale (== original
        # unit normalisation when end_scale=1.0).
        self.register_buffer(
            "_elastic_epoch",
            torch.tensor(int(query_norm_elastic_end_epoch), dtype=torch.long),
            persistent=False,
        )

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
        # Value-structure QUERY injection: per-QUERY value term already projected
        # to the value output width (shape (B, L, d) or (B, L, H, d)).  Added as
        # ``(sum_j A_ij) * value_query`` — the exact, memory-cheap decomposition
        # of concatenating the query identity into a (linear, bias-free) W_V^q.
        value_query: Optional[torch.Tensor] = None,
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
        if self.use_gain and (gain_query is None or gain_key is None):
            raise ValueError(
                "GatedCrossAttention requires gain_query and gain_key "
                "(the reconstruction-gain projections) when use_gain=True."
            )

        B, L, E_s = query.shape
        _, S, _ = key.shape

        # ---- Structure gate: Hard-Concrete L0 gate -----------------------
        # Default:            log_alpha = <q^s, k^s> / sqrt(E_s)
        # normalize_query:    q̂ = q/||q||  and  log_alpha = <q̂, k^s> * sqrt(fanin)
        #   -> the query DIRECTION (not its unbounded norm) decides selection,
        #      killing the "point the query at the key centroid to light every
        #      key at once" block shortcut.  A single aligned unit-key parent
        #      then scores sqrt(fanin); a centroid over m parents scores
        #      sqrt(fanin/m) per edge (blocks get sqrt(m)-cheaper per edge and
        #      pay a growing L0 cost while true parents stay ON).
        q_s = query
        if self.normalize_query:
            # Elastic contraction: q̂ * M(epoch); M==1 -> plain unit-norm.
            q_s, scale_s = elastic_normalize_query(
                q_s, self.query_fanin_scale,
                self.elastic_query_cfg, int(self._elastic_epoch.item()),
            )
        else:
            scale_s = 1.0 / math.sqrt(E_s)

        log_alpha = torch.einsum("ble,bse->bls", q_s, key) * scale_s

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
            # Additive init-balancing offset: shift the alignment logit so that
            # at init (log_alpha ~= 0) the edge posterior is sigmoid(-offset)
            # rather than 0.5.  Applied consistently to the sample AND the
            # posterior so the gate's expected behaviour and its P(z>0) agree.
            la = log_alpha - self.edge_offset
            if self.training:
                # Stochastic Hard-Concrete sample.
                u = torch.rand_like(la).clamp_(1e-6, 1.0 - 1e-6)
                s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / self.beta)
            else:
                # Deterministic (mean) gate at eval.
                s = torch.sigmoid(la / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            z = s_bar.clamp(0.0, 1.0)                     # (B, L, S) in [0, 1]

            # Posterior probability that the gate is open: P(z > 0).
            #   P(z>0) = sigmoid(la - beta * log(-gamma / zeta))
            p_edge_on = torch.sigmoid(
                la - self.beta * math.log(-self.gamma / self.zeta)
            )



        # ---- Reconstruction gain: bounded sigmoid score ------------------
        if self.use_gain:
            E_g = gain_query.shape[-1]
            scale_g = 1.0 / (self.gain_tau * math.sqrt(E_g))
            gain_logit = torch.einsum("ble,bse->bls", gain_query, gain_key) * scale_g
            g = torch.sigmoid(gain_logit)               # (B, L, S) in (0, 1)
            # ---- Combined edge weight ------------------------------------
            A = z * g                                    # (B, L, S)
        else:
            # ---- Gain bypassed: the structure gate IS the final weight ---
            g = None
            A = z                                        # (B, L, S)

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

        # ---- Value-structure QUERY injection (additive query term) --------
        # V_ij = W_V([v_j;e_j]) + W_V^q(e_i^q).  Since W_V^q(e_i^q) is
        # independent of the key j, it factors out of the aggregation as
        # ``(sum_j A_ij) * value_query_i`` — the exact, memory-cheap equivalent
        # of concatenating the query identity into a linear, bias-free W_V.
        # ``A`` here is the TRUE applied weight (gate * gain, masked, dropped).
        if value_query is not None:
            row_sum = A.sum(dim=-1)                            # (B, L)
            if out.dim() == 4:
                out = out + row_sum[:, :, None, None] * value_query   # (B, L, H, d)
            else:
                out = out + row_sum[:, :, None] * value_query         # (B, L, d)


        # ---- Diagnostics / regularisation signals (gate-only) ------------
        # Batch-mean (2-D) tensors so the forecaster's score-sparsity / NOTEARS
        # paths (which expect 2-D score tensors) operate on the GATE.
        self.score_tensor_for_sparsity = p_edge_on.mean(dim=0)   # (L, S) gate
        self.last_p_edge_on = p_edge_on.mean(dim=0)              # (L, S) gate
        # last_gain is None when the gain is bypassed (use_gain=False).
        self.last_gain = None if g is None else g.mean(dim=0).detach()  # (L, S) diag

        # ---- Entropy (over the combined weights, for logging) ------------
        entropy = None
        if self.register_entropy:
            w = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)  # (B, L)

        aux = {"entropy": entropy, "l0_penalty": l0_penalty}
        # GCA reuses the second slot to return the STRUCTURE GATE posterior
        # (masked P(z>0), shape (B, L, S)) instead of the applied weight A=z*g.
        # `out` above is already computed from A, and nothing downstream consumes
        # this slot for reconstruction — only eval/DAG-extraction reads it, which
        # wants the clean structural score (thresholdable at 0.5). Scoped to GCA.
        return out, p_edge_masked, aux


    def __repr__(self):
        return (
            f"GatedCrossAttention(beta={self.beta}, gamma={self.gamma}, "
            f"zeta={self.zeta}, gain_tau={self.gain_tau})"
        )
