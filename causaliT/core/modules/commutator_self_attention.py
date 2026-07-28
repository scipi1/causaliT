"""
CommutatorSelfAttention: unified-gating, direction-aware variable selector.

Motivation
==========
``GatedSelfAttention`` (GSA) fuses a **symmetric** Hard-Concrete existence gate
(on the symmetric Toeplitz part ``S_sym``) with an **antisymmetric** direction
gate (on ``A_anti``).  This module keeps the direction machinery but **unifies
the existence gate with ``GatedCrossAttention`` (GCA)**: the edge-existence gate
is a plain Hard-Concrete gate applied *directly* to the raw alignment score
``raw = Q Kᵀ`` — exactly the same mechanism GCA uses for S→X edges.  The X→X
existence decision is therefore made with the *identical* gate as the S→X one
(one gating code path, one L0 semantics — directed edges), while the direction
is resolved by the commutator/antisymmetric score.

Design
======
One structural score ``raw`` feeds two disentangled factors::

    raw    = <q_i, k_j> * scale                    # (B, N, N), asymmetric alignment
    A_anti = (raw - raw^T) / 2 = ½(Q Kᵀ - K Qᵀ)    # antisymmetric  -> DIRECTION

**Existence gate** ``z_edge`` — a Hard-Concrete L0 gate on ``raw`` itself
(GCA-style, asymmetric).  One independent uniform per *directed* edge, so
``z_edge_ij`` and ``z_edge_ji`` are independent.  Its expected-active-edge count
``sum_{i != j} P(z_edge>0)`` is the L0 penalty (each directed edge counted once,
matching GCA).  Driven by HSIC + L0.

**Direction gate** ``d`` — the ANTISYMMETRIC coupled Binary-Concrete gate on
``A_anti = ½(Q Kᵀ - K Qᵀ)`` (the antisymmetric part of the *projected* score
matrix — the operative so(N) "commutator").  Because both the logit and the
shared logistic noise are antisymmetric, ``d_ij + d_ji = 1`` per sample: the
Toeplitz two-cycle-suppression property, realised inside the stochastic gate.
For parallel/sibling embeddings (``e_i ∥ e_j``) ``A_anti_ij → 0`` so ``d → 0.5``
(undecided) — edge *removal* then rests entirely on the existence gate.

**Reconstruction gain** ``g`` — bounded sigmoid score from a SEPARATE gain
query/key pair (identical to GCA / GSA); driven by the MSE reconstruction loss.

Combined edge weight::

    A_ij = z_edge_ij * d_ij * g_ij     (diagonal zeroed, hard_mask applied)

Why keeping direction as a SEPARATE factor is still required
-----------------------------------------------------------
If direction were folded into the existence logit alone, ``sigmoid(A - c)`` and
``sigmoid(-A - c)`` would be locked together and "no edge" would be
unrepresentable.  Here existence (``z_edge``) and direction (``d``) are separate
multiplicative factors, so ``z_edge -> 0`` still means "no edge" and sparsity is
preserved.

Contract
========
Mirrors ``GatedSelfAttention`` exactly:
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False, gain_query=None, gain_key=None,
          value_query=None)`` returns ``(out, p_directed, aux)`` with
``aux = {"entropy": ..., "l0_penalty": ...}``.  The second slot ``p_directed``
is the DIRECTED structure posterior ``P(z_edge>0) * d`` (masked), thresholdable
at 0.5 to recover the adjacency (GCA convention).
"""

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.utils.query_norm import (
    apply_query_norm,
    make_query_norm_log_scale,
    overspend_penalty,
)




class CommutatorSelfAttention(nn.Module):
    """Unified-gating selector: asymmetric GCA L0 existence x antisymmetric direction x gain."""

    def __init__(
        self,
        attention_dropout: float = 0.0,
        register_entropy: bool = False,
        layer_name: Optional[str] = None,
        # Hard-Concrete existence-gate hyper-parameters (Louizos et al., ICLR 2018).
        init_tau: float = 2.0 / 3.0,   # beta: existence-gate temperature
        gamma: float = -0.1,           # stretch lower bound (< 0)
        zeta: float = 1.1,             # stretch upper bound (> 1)
        # Direction-gate Binary-Concrete temperature (coupled stochastic).
        dir_tau: float = 2.0 / 3.0,    # beta_dir
        # Reconstruction-gain temperature (scales the sigmoid logit).
        gain_tau: float = 1.0,
        # When False, bypass the learnable reconstruction gain: A = structure.
        use_gain: bool = True,
        # Centroid-collapse fix: L2-normalise the query so its DIRECTION (not its
        # norm) drives selection, with a fixed sqrt(query_fanin_scale) score
        # scale replacing 1/sqrt(E). Feeds BOTH the existence gate (via raw) and
        # the direction gate (via A_anti).
        normalize_query: bool = False,
        query_fanin_scale: float = 1.0,
        # Learnable per-node query-norm multiplier (see
        # ``causaliT/utils/query_norm.py``); only active with
        # ``normalize_query=True``.  Each child owns ``M_i = exp(log_scale_i)``
        # (init ``query_norm_init_scale``) scaling its unit query so it can
        # ADAPTIVELY overspend the directional budget when the structural signal
        # pays for it; the structural loss charges ``relu(M_i - target)^2``.
        # ``query_norm_num_nodes`` is the number of query rows (children).
        query_norm_learnable: bool = False,
        query_norm_init_scale: float = 1.0,
        query_norm_target: float = 1.0,
        query_norm_num_nodes: Optional[int] = None,

        # Direction-gate parametrisation.  "qk" (default) derives the

        # antisymmetric direction score from the SAME raw alignment used by the
        # existence gate: A_anti = ½(QKᵀ − KQᵀ).  That is a valid so(N)
        # commutator ONLY when Q and K share the same embedding.  "skew_query"
        # instead builds a genuine so(d) commutator on the QUERY alone:
        #     A_anti_ij = q_iᵀ Ω q_j ,   Ω = W_a W_bᵀ − W_b W_aᵀ
        # (Ω antisymmetric ⇒ d_ij + d_ji = 1, and q_i = q_j ⇒ A_anti = 0 →
        # "undecided"), so the direction stays a valid Lie generator even when
        # the query and key come from DIFFERENT embeddings (e.g. a free query
        # aligning to a fixed orthonormal key frame).  See Option B in the
        # shared-query design.
        direction_mode: str = "qk",
        # Width of the (projected) query fed to the skew generator; REQUIRED
        # when direction_mode="skew_query".  ``direction_rank`` sets the rank of
        # Ω (defaults to direction_dim → full rank).
        direction_dim: Optional[int] = None,
        direction_rank: Optional[int] = None,
        # Batch-consistent key dropout (columns zeroed identically across batch).
        batch_key_dropout: Optional[float] = None,

        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        # Constant-score capacity protocol (Optuna): freeze the existence gate at
        # this constant on every edge while the gain g stays learnable.
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

        # Gate params are non-learnable constants (matching GatedCrossAttention).
        self.beta = float(init_tau)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.dir_beta = float(dir_tau)
        self.gain_tau = float(gain_tau)

        self.use_gain = bool(use_gain)

        self.normalize_query = bool(normalize_query)
        self.query_fanin_scale = float(query_fanin_scale)

        # Learnable per-node query-norm multiplier (only active with
        # normalize_query=True).  ``M_i = exp(log_scale_i)`` init at
        # ``query_norm_init_scale``; classified STRUCTURAL via the
        # ``query_norm_log_scale`` name (gradient_routing).
        self.query_norm_learnable = bool(query_norm_learnable) and self.normalize_query
        self.query_norm_target = float(query_norm_target)
        if self.query_norm_learnable:
            self.query_norm_log_scale = make_query_norm_log_scale(
                int(query_norm_num_nodes), query_norm_init_scale
            )
        else:
            self.query_norm_log_scale = None



        # ---- Direction-gate parametrisation -----------------------------
        if direction_mode not in ("qk", "skew_query"):
            raise ValueError(
                f"direction_mode='{direction_mode}' is invalid. "
                f"Must be one of: 'qk', 'skew_query'."
            )
        self.direction_mode = str(direction_mode)
        if self.direction_mode == "skew_query":
            if direction_dim is None:
                raise ValueError(
                    "direction_mode='skew_query' requires direction_dim "
                    "(the projected-query width) to build the so(d) generator."
                )
            r = int(direction_rank) if direction_rank is not None else int(direction_dim)
            # A_anti_ij = q_iᵀ Ω q_j with Ω = W_a W_bᵀ − W_b W_aᵀ, realised by
            # antisymmetrising the bilinear form ⟨W_a q_i, W_b q_j⟩ (see forward).
            self.direction_proj_a = nn.Linear(int(direction_dim), r, bias=False)
            self.direction_proj_b = nn.Linear(int(direction_dim), r, bias=False)
            # Small init so A_anti ≈ 0 at start → direction ≈ 0.5 (undecided),
            # letting orientation be EXPLORED rather than committed at init.
            nn.init.normal_(self.direction_proj_a.weight, std=0.02)
            nn.init.normal_(self.direction_proj_b.weight, std=0.02)
        else:
            self.direction_proj_a = None
            self.direction_proj_b = None

        # Pre-computed L0 offset:  P(z>0) = sigmoid(log_alpha - beta*log(-gamma/zeta)).

        self._l0_offset: float = float(self.beta * math.log(-self.gamma / self.zeta))

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
    # Noise helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _logit(u: torch.Tensor) -> torch.Tensor:
        """Logistic (inverse-sigmoid) noise from a uniform sample."""
        u = u.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(u) - torch.log1p(-u)

    @staticmethod
    def _antisymmetric_noise(shape, device, dtype) -> torch.Tensor:
        """An antisymmetric logistic-noise matrix: eps_ji == -eps_ij, zero diagonal.

        One draw per unordered pair (upper triangle), anti-mirrored to the lower
        triangle, so the direction gate satisfies d_ij + d_ji == 1 per sample.
        """
        B, N, _ = shape
        u = torch.rand(B, N, N, device=device, dtype=dtype)
        eps = CommutatorSelfAttention._logit(u)
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
        value_query: Optional[torch.Tensor] = None,
    ):
        if causal_mask:
            raise NotImplementedError(
                "CommutatorSelfAttention does not support causal masking."
            )

        if query.dim() != 3 or key.dim() != 3:
            raise ValueError(
                "CommutatorSelfAttention expects a single structural head "
                "(3-D query/key: (B, N, E)); use shared_dag_across_heads=True."
            )
        B, L, E_s = query.shape
        _, S, _ = key.shape
        if L != S:
            raise ValueError(
                "CommutatorSelfAttention requires SQUARE self-attention scores "
                f"(L == S) for the antisymmetric direction split, got L={L}, S={S}."
            )
        if self.use_gain and (gain_query is None or gain_key is None):
            raise ValueError(
                "CommutatorSelfAttention requires gain_query and gain_key "
                "(the reconstruction-gain projections) when use_gain=True."
            )

        N = L

        # ---- Structural alignment score ---------------------------------
        # Centroid-collapse fix: unit-normalise the query so its DIRECTION (not
        # its norm) drives selection, with a fixed sqrt(query_fanin_scale) score
        # scale replacing 1/sqrt(E).
        q_s = query
        if self.normalize_query:
            if self.query_norm_learnable:
                # q̂ * M_i (per-node learnable budget); scale = sqrt(fanin).
                q_s, scale_s = apply_query_norm(
                    q_s, self.query_norm_log_scale, self.query_fanin_scale
                )
            else:
                # Plain unit-norm cap (M == 1).
                q_s = F.normalize(q_s, p=2.0, dim=-1, eps=1e-8)
                scale_s = math.sqrt(self.query_fanin_scale)

        else:
            scale_s = 1.0 / math.sqrt(E_s)

        raw = torch.einsum("bne,bme->bnm", q_s, key) * scale_s   # (B, N, N)
        raw = torch.nan_to_num(raw, nan=0.0)

        # ---- Direction score (antisymmetric) ----------------------------
        if self.direction_mode == "skew_query":
            # Genuine so(d) commutator on the QUERY alone: A_anti_ij = q_iᵀ Ω q_j
            # with Ω = W_a W_bᵀ − W_b W_aᵀ.  Antisymmetrising the bilinear form
            # ⟨W_a q_i, W_b q_j⟩ realises exactly q_iᵀ skew(W_a W_bᵀ) q_j, so the
            # direction stays a valid Lie generator even when the query and key
            # come from DIFFERENT embeddings (free query ↔ fixed orthonormal key).
            a = self.direction_proj_a(q_s)                         # (B, N, r)
            b = self.direction_proj_b(q_s)                         # (B, N, r)
            form = torch.einsum("bnr,bmr->bnm", a, b)              # (B, N, N)
            form = torch.nan_to_num(form, nan=0.0)
            A_anti = 0.5 * (form - form.transpose(-1, -2))         # antisymmetric
        else:
            # "qk": antisymmetric part of the raw alignment (the so(N)
            # commutator).  Valid only when Q and K share the same embedding.
            A_anti = 0.5 * (raw - raw.transpose(-1, -2))           # antisymmetric


        if oracle:
            # ---- Oracle: the ground-truth DAG IS the structure gate ------
            if hard_mask is None:
                raise ValueError(
                    "CommutatorSelfAttention oracle mode requires hard_mask "
                    "(the ground-truth adjacency used as the structure gate)."
                )
            hm_gate = hard_mask
            if hm_gate.dim() == 2:
                hm_gate = hm_gate.unsqueeze(0)                      # (1, N, N)
            structure = hm_gate.to(raw.dtype).expand(B, N, N)
            p_exist = structure                                    # diagnostics
            direction = structure                                  # diagnostics
            p_directed = structure                                 # directed posterior
        elif self.optuna_protocol is not None:
            # ---- Constant-score capacity protocol (gate-only override) ----
            c = float(self.optuna_protocol)
            structure = torch.full_like(raw, c)
            p_exist = torch.full_like(raw, c)
            direction = torch.full_like(raw, 0.5)
            p_directed = torch.full_like(raw, c) * direction
        else:
            # ---- Existence gate: ASYMMETRIC Hard-Concrete on raw (GCA-style)
            if self.training:
                u = torch.zeros_like(raw).uniform_().clamp_(1e-8, 1.0 - 1e-8)
                s_e = torch.sigmoid(
                    (torch.log(u) - torch.log1p(-u) + raw) / self.beta
                )
            else:
                s_e = torch.sigmoid(raw / self.beta)
            s_bar = s_e * (self.zeta - self.gamma) + self.gamma
            z_edge = s_bar.clamp(0.0, 1.0)                         # (B, N, N), asymmetric

            # ---- Direction gate: ANTISYMMETRIC coupled Binary-Concrete ----
            if self.training:
                eps_d = self._antisymmetric_noise(
                    (B, N, N), device=A_anti.device, dtype=A_anti.dtype
                )
                direction = torch.sigmoid((eps_d + A_anti) / self.dir_beta)
            else:
                direction = torch.sigmoid(A_anti / self.dir_beta)

            structure = z_edge * direction                        # directed structure gate

            # Posterior that the (directed) edge exists: P(z_edge > 0).
            p_exist = torch.sigmoid(raw - self._l0_offset)
            p_directed = p_exist * direction

        # ---- Reconstruction gain: bounded sigmoid score -----------------
        if self.use_gain:
            E_g = gain_query.shape[-1]
            scale_g = 1.0 / (self.gain_tau * math.sqrt(E_g))
            gain_logit = torch.einsum("bne,bme->bnm", gain_query, gain_key) * scale_g
            g = torch.sigmoid(gain_logit)                         # (B, N, N) in (0, 1)
            A = structure * g                                     # (B, N, N)
        else:
            g = None
            A = structure                                         # (B, N, N)

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
            p_exist_masked = p_exist * hm
        else:
            p_exist_masked = p_exist.masked_fill(diag, 0.0)

        # ---- L0 penalty: expected active (allowed) DIRECTED edges --------
        # Count each DIRECTED edge once (all off-diagonal), matching GCA — the
        # existence posterior is asymmetric here.
        off_diag = (~torch.eye(N, device=A.device, dtype=torch.bool)).unsqueeze(0).to(A.dtype)
        l0_penalty = (p_exist_masked * off_diag).sum(dim=(-2, -1)).mean()

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
                f"CommutatorSelfAttention value must be 3-D or 4-D, got {value.dim()}-D."
            )

        # ---- Value-structure QUERY injection (additive query term) --------
        if value_query is not None:
            row_sum = A.sum(dim=-1)                                # (B, N)
            if out.dim() == 4:
                out = out + row_sum[:, :, None, None] * value_query   # (B, N, H, d)
            else:
                out = out + row_sum[:, :, None] * value_query         # (B, N, d)

        # ---- Diagnostics / regularisation signals -----------------------
        self.score_tensor_for_sparsity = p_directed.mean(dim=0)   # (N, N) directed
        self.last_p_edge_on = p_directed.mean(dim=0)              # (N, N) directed
        self.last_p_edge_undirected = p_exist.mean(dim=0)         # (N, N) existence
        self.last_direction = direction.mean(dim=0).detach()      # (N, N) diag only
        self.last_gain = None if g is None else g.mean(dim=0).detach()  # (N, N) diag

        # ---- Entropy (over the combined weights, for logging) -----------
        entropy = None
        if self.register_entropy:
            w = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)  # (B, N)

        aux = {"entropy": entropy, "l0_penalty": l0_penalty}
        return out, p_directed, aux

    def __repr__(self):
        return (
            f"CommutatorSelfAttention(beta={self.beta}, gamma={self.gamma}, "
            f"zeta={self.zeta}, dir_beta={self.dir_beta}, gain_tau={self.gain_tau}, "
            f"direction_mode={self.direction_mode})"
        )


