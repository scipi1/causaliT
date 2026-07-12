"""
GatedSelfAttention: direction-aware differentiable variable selector.

Motivation
==========
``GatedCrossAttention`` (GCA) disentangles each edge score into a structure
gate ``z`` (a Hard-Concrete L0 selector) times a reconstruction gain ``g``.  It
assumes a KNOWN set of source nodes S (the cross-attention is X -> [S, X]).  In
a standard causal-discovery setting we do NOT know which variables are sources:
the model must infer the WHOLE directed acyclic graph over all variables.

This module performs **self-attention over a single set of N variables** (every
variable may be a parent of every other) and factorises each edge into three
disentangled, target-conditioned components.  The novelty is that it fuses the
Hard-Concrete L0 selector of GCA with the **directional Toeplitz
parametrisation**: a differentiable, direction-aware variable selector.

Design
======
One structural score, decomposed Toeplitz-style into orthogonal parts::

    raw    = <q^s_i, k^s_j> / sqrt(E_s)          # (B, N, N), asymmetric
    S_sym  = (raw + raw^T) / 2                    # symmetric      -> edge EXISTENCE
    A_anti = (raw - raw^T) / 2                    # antisymmetric  -> edge DIRECTION

**Existence gate** ``z_edge`` — a SYMMETRIC Hard-Concrete L0 gate on ``S_sym``.
The stochastic training draw uses ONE uniform per unordered pair (mirrored to
the lower triangle) so ``z_edge_ij == z_edge_ji`` for every sample.  Its
expected-active-edge count ``sum_{i<j} P(z_edge>0)`` is the L0 penalty (each
undirected edge counted once).  This is the clean, thresholdable SELECTION
signal — driven by HSIC + L0 only.

**Direction gate** ``d`` — an ANTISYMMETRIC coupled Binary-Concrete gate on
``A_anti`` (Option B: stochastic, so orientation is *explored* during training).
The training draw shares ONE logistic noise per unordered pair, ANTI-mirrored to
the lower triangle (``eps_ji = -eps_ij``).  Because both the logit and the noise
are antisymmetric::

    d_ij = sigmoid((eps_ij + A_anti_ij) / beta_dir)
    d_ji = sigmoid((-eps_ij - A_anti_ij) / beta_dir) = 1 - d_ij       (per sample)

i.e. sampling ``i->j`` with probability ``p`` forces ``j->i`` to be ``1-p`` —
exactly the Toeplitz ``d_ij + d_ji = 1`` two-cycle-suppression property, now
realised inside the stochastic gate.

**Reconstruction gain** ``g`` — bounded sigmoid score from a SEPARATE gain
query/key pair (identical to GCA); driven by the MSE reconstruction loss only.

Combined edge weight::

    A_ij = z_edge_ij * d_ij * g_ij     (diagonal zeroed, hard_mask applied)

Key invariant: ``A_ij + A_ji`` is proportional to ``z_edge`` (the pair's total
edge mass is the sparse existence gate, merely split by direction).  All three
outcomes are reachable: no edge (``z_edge ~ 0``), ``i->j`` (``d_ij ~ 1``),
``j->i`` (``d_ij ~ 0``).

Why a SINGLE antisymmetric gate is NOT enough
---------------------------------------------
If one folded direction into the Hard-Concrete logit alone (antisymmetric
logit), the two open-probabilities ``sigmoid(A - c)`` and ``sigmoid(-A - c)``
are locked together: one can be pushed down only by pushing the other up, so
BOTH cannot approach zero — "no edge" becomes unrepresentable and sparsity is
lost.  The symmetric existence factor is therefore mandatory.

Gradient routing
=================
``query`` / ``key`` are the *structural* (gate) projections; both ``S_sym`` and
``A_anti`` derive from them, so the L0 penalty and the direction gate are driven
by the structural loss.  ``gain_query`` / ``gain_key`` are the *reconstruction*
projections (named ``gain_*`` so the name-based router classifies them as
reconstruction).  Must be trained with ``use_gradient_routing=True`` so the
product ``z*d*g`` does not re-conflate structure and gain.

Contract (mirrors the other inner-attention modules)
=====================================================
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False, gain_query=None, gain_key=None)``
returns ``(out, attn, aux)`` with ``aux = {"entropy": ..., "l0_penalty": ...}``.

Like GCA, the second return slot ``attn`` is NOT the applied weight ``A``; it is
the **directed structure posterior** ``P(z_edge>0) * d`` (masked), values in
``(0, 1)``, so evaluation can threshold it at 0.5 to recover the adjacency.
``query`` / ``key`` MUST be a single structural head (3-D: ``(B, N, E)``) with
``L == S`` (square self-attention).  ``value`` may be 3-D ``(B, N, d)`` or 4-D
``(B, N, H, d)``.
"""

from typing import Optional

import math
import torch
import torch.nn as nn


class GatedSelfAttention(nn.Module):
    """Direction-aware selector: symmetric L0 existence x antisymmetric direction x gain."""

    def __init__(
        self,
        attention_dropout: float = 0.0,
        register_entropy: bool = False,
        layer_name: Optional[str] = None,
        # Hard-Concrete existence-gate hyper-parameters (Louizos et al., ICLR 2018).
        init_tau: float = 2.0 / 3.0,   # beta: existence-gate temperature
        gamma: float = -0.1,           # stretch lower bound (< 0)
        zeta: float = 1.1,             # stretch upper bound (> 1)
        # Direction-gate Binary-Concrete temperature (Option B, coupled stochastic).
        dir_tau: float = 2.0 / 3.0,    # beta_dir
        # Reconstruction-gain temperature (scales the sigmoid logit).
        gain_tau: float = 1.0,
        # Batch-consistent key dropout (columns zeroed identically across batch).
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        # Constant-score capacity protocol (Optuna): when not None the STRUCTURE
        # gate (existence) is frozen at this constant on every edge while the
        # reconstruction gain g stays learnable.
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
        self.dir_beta = float(dir_tau)
        self.gain_tau = float(gain_tau)

        # Pre-computed L0 offset:  P(z>0) = sigmoid(log_alpha - beta*log(-gamma/zeta)).
        self._l0_offset: float = float(self.beta * math.log(-self.gamma / self.zeta))

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
        #   score_tensor_for_sparsity — DIRECTED posterior P(z_edge>0)*d (B-mean),
        #     read by the L1 score-sparsity and NOTEARS terms.
        #   last_p_edge_on           — same DIRECTED posterior (B-mean), thresholded
        #     at eval to obtain the recovered adjacency.
        #   last_p_edge_undirected   — SYMMETRIC skeleton posterior P(z_edge>0) (B-mean).
        #   last_direction           — direction gate d (B-mean), diagnostics only.
        #   last_gain                — reconstruction gain g (B-mean), diagnostics only.
        self.score_tensor_for_sparsity: Optional[torch.Tensor] = None
        self.last_p_edge_on: Optional[torch.Tensor] = None
        self.last_p_edge_undirected: Optional[torch.Tensor] = None
        self.last_direction: Optional[torch.Tensor] = None
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
    # Noise helpers (upper-triangle draws mirrored to enforce pair-consistency)
    # ------------------------------------------------------------------
    @staticmethod
    def _logit(u: torch.Tensor) -> torch.Tensor:
        """Logistic (inverse-sigmoid) noise from a uniform sample."""
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(u) - torch.log1p(-u)

    @staticmethod
    def _symmetric_noise(shape, device, dtype) -> torch.Tensor:
        """A symmetric logistic-noise matrix: eps_ij == eps_ji, diagonal irrelevant.

        One draw per unordered pair (upper triangle), mirrored to the lower
        triangle, so the existence gate is identical for both directions.
        """
        B, N, _ = shape
        u = torch.rand(B, N, N, device=device, dtype=dtype)
        eps = GatedSelfAttention._logit(u)
        triu = torch.triu(eps, diagonal=1)          # strictly-upper entries
        return triu + triu.transpose(-1, -2)        # symmetric, zero diagonal

    @staticmethod
    def _antisymmetric_noise(shape, device, dtype) -> torch.Tensor:
        """An antisymmetric logistic-noise matrix: eps_ji == -eps_ij, zero diagonal.

        One draw per unordered pair (upper triangle), anti-mirrored to the lower
        triangle, so the direction gate satisfies d_ij + d_ji == 1 per sample.
        """
        B, N, _ = shape
        u = torch.rand(B, N, N, device=device, dtype=dtype)
        eps = GatedSelfAttention._logit(u)
        triu = torch.triu(eps, diagonal=1)          # strictly-upper entries
        return triu - triu.transpose(-1, -2)        # antisymmetric, zero diagonal

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
                "GatedSelfAttention does not support causal masking."
            )
        if query.dim() != 3 or key.dim() != 3:
            raise ValueError(
                "GatedSelfAttention expects a single structural head "
                "(3-D query/key: (B, N, E)); use shared_dag_across_heads=True."
            )
        B, L, E_s = query.shape
        _, S, _ = key.shape
        if L != S:
            raise ValueError(
                "GatedSelfAttention requires SQUARE self-attention scores "
                f"(L == S) for the Toeplitz symmetric/antisymmetric split, "
                f"got L={L}, S={S}."
            )
        if gain_query is None or gain_key is None:
            raise ValueError(
                "GatedSelfAttention requires gain_query and gain_key "
                "(the reconstruction-gain projections)."
            )

        N = L

        # ---- Structural score, Toeplitz-decomposed ----------------------
        scale_s = 1.0 / math.sqrt(E_s)
        raw = torch.einsum("bne,bme->bnm", query, key) * scale_s   # (B, N, N)
        raw = torch.nan_to_num(raw, nan=0.0)
        S_sym = 0.5 * (raw + raw.transpose(-1, -2))                # symmetric
        A_anti = 0.5 * (raw - raw.transpose(-1, -2))               # antisymmetric

        if oracle:
            # ---- Oracle: the ground-truth DAG IS the structure gate ------
            # z_ij = hard_mask_ij (true topology).  Direction/existence are
            # taken directly from the oracle adjacency; only the learned gain g
            # modulates edge magnitude.
            if hard_mask is None:
                raise ValueError(
                    "GatedSelfAttention oracle mode requires hard_mask "
                    "(the ground-truth adjacency used as the structure gate)."
                )
            hm_gate = hard_mask
            if hm_gate.dim() == 2:
                hm_gate = hm_gate.unsqueeze(0)                      # (1, N, N)
            structure = hm_gate.to(raw.dtype).expand(B, N, N)
            p_edge_undirected = structure                          # diagnostics
            direction = structure                                  # diagnostics
            p_directed = structure                                 # directed posterior
        elif self.optuna_protocol is not None:
            # ---- Constant-score capacity protocol (gate-only override) ----
            c = float(self.optuna_protocol)
            structure = torch.full_like(S_sym, c)
            p_edge_undirected = torch.full_like(S_sym, c)
            direction = torch.full_like(S_sym, 0.5)
            p_directed = torch.full_like(S_sym, c) * direction
        else:
            # ---- Existence gate: SYMMETRIC Hard-Concrete -----------------
            if self.training:
                eps_e = self._symmetric_noise(
                    (B, N, N), device=S_sym.device, dtype=S_sym.dtype
                )
                s_e = torch.sigmoid((eps_e + S_sym) / self.beta)
            else:
                s_e = torch.sigmoid(S_sym / self.beta)
            s_bar = s_e * (self.zeta - self.gamma) + self.gamma
            z_edge = s_bar.clamp(0.0, 1.0)                         # (B, N, N), symmetric

            # ---- Direction gate: ANTISYMMETRIC coupled Binary-Concrete ----
            if self.training:
                eps_d = self._antisymmetric_noise(
                    (B, N, N), device=A_anti.device, dtype=A_anti.dtype
                )
                direction = torch.sigmoid((eps_d + A_anti) / self.dir_beta)
            else:
                direction = torch.sigmoid(A_anti / self.dir_beta)

            structure = z_edge * direction                        # directed structure gate

            # Posterior that the (undirected) edge exists: P(z_edge > 0).
            p_edge_undirected = torch.sigmoid(S_sym - self._l0_offset)
            p_directed = p_edge_undirected * direction

        # ---- Reconstruction gain: bounded sigmoid score -----------------
        E_g = gain_query.shape[-1]
        scale_g = 1.0 / (self.gain_tau * math.sqrt(E_g))
        gain_logit = torch.einsum("bne,bme->bnm", gain_query, gain_key) * scale_g
        g = torch.sigmoid(gain_logit)                             # (B, N, N) in (0, 1)

        # ---- Combined edge weight ---------------------------------------
        A = structure * g                                         # (B, N, N)

        # ---- Zero the diagonal (no self-loops) --------------------------
        diag = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(diag, 0.0)
        p_directed = p_directed.masked_fill(diag, 0.0)

        # ---- Structural hard mask (allowed-edge topology) ---------------
        if hard_mask is not None:
            hm = hard_mask
            if hm.dim() == 2:
                hm = hm.unsqueeze(0)                              # (1, N, N)
            hm = hm.to(A.dtype)
            A = A * hm
            p_directed = p_directed * hm
            p_edge_masked = p_edge_undirected * hm
        else:
            p_edge_masked = p_edge_undirected.masked_fill(diag, 0.0)

        # ---- L0 penalty: expected active (allowed) UNDIRECTED edges ------
        # Count each unordered pair once (strictly-upper triangle) since the
        # existence posterior is symmetric.
        triu_mask = torch.triu(
            torch.ones(N, N, device=A.device, dtype=A.dtype), diagonal=1
        ).unsqueeze(0)
        l0_penalty = (p_edge_masked * triu_mask).sum(dim=(-2, -1)).mean()

        # ---- Batch-consistent key dropout -------------------------------
        bkd_p = self._current_bkd_p()
        if self.training and bkd_p is not None and bkd_p > 0.0:
            keep = (torch.rand(N, device=A.device) >= bkd_p).to(A.dtype)  # (N,)
            A = A * keep.view(1, 1, N)
            self._bkd_step += 1

        # ---- Attention-weight dropout -----------------------------------
        A = self.dropout(A)

        # ---- Value aggregation ------------------------------------------
        if value.dim() == 4:
            out = torch.einsum("bnm,bmhd->bnhd", A, value)        # (B, N, H, d)
        elif value.dim() == 3:
            out = torch.einsum("bnm,bmd->bnd", A, value)          # (B, N, d)
        else:
            raise ValueError(
                f"GatedSelfAttention value must be 3-D or 4-D, got {value.dim()}-D."
            )

        # ---- Diagnostics / regularisation signals -----------------------
        self.score_tensor_for_sparsity = p_directed.mean(dim=0)   # (N, N) directed
        self.last_p_edge_on = p_directed.mean(dim=0)              # (N, N) directed
        self.last_p_edge_undirected = p_edge_undirected.mean(dim=0)  # (N, N) skeleton
        self.last_direction = direction.mean(dim=0).detach()      # (N, N) diag only
        self.last_gain = g.mean(dim=0).detach()                   # (N, N) diag only

        # ---- Entropy (over the combined weights, for logging) -----------
        entropy = None
        if self.register_entropy:
            w = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)  # (B, N)

        aux = {"entropy": entropy, "l0_penalty": l0_penalty}
        # Second slot: the DIRECTED structure posterior P(z_edge>0)*d (masked),
        # thresholdable at 0.5 to recover the adjacency (GCA convention).
        return out, p_directed, aux

    def __repr__(self):
        return (
            f"GatedSelfAttention(beta={self.beta}, gamma={self.gamma}, "
            f"zeta={self.zeta}, dir_beta={self.dir_beta}, gain_tau={self.gain_tau})"
        )
