"""
GatedSelfAttention: direction-aware differentiable variable selector.

Self-attention over a single set of N variables (every variable may be a parent
of every other), fusing the Hard-Concrete L0 selector of ``GatedCrossAttention``
with a directional Toeplitz parametrisation.  One structural score is decomposed
Toeplitz-style into orthogonal parts::

    raw    = <q^s_i, k^s_j> * scale             # (B, N, N), asymmetric
    S_sym  = (raw + raw^T) / 2                   # symmetric      -> edge EXISTENCE
    A_anti = (raw - raw^T) / 2                   # antisymmetric  -> edge DIRECTION

**Existence gate** ``z_edge`` - a SYMMETRIC Hard-Concrete L0 gate on ``S_sym``.
The stochastic training draw uses ONE uniform per unordered pair (mirrored to
the lower triangle) so ``z_edge_ij == z_edge_ji`` for every sample.  Its
expected-active-edge count ``sum_{i<j} P(z_edge>0)`` is the L0 penalty (each
undirected edge counted once).  This is the clean, thresholdable SELECTION
signal, driven by HSIC + L0 only.

**Direction gate** ``d`` - an ANTISYMMETRIC coupled Binary-Concrete gate on
``A_anti`` (stochastic, so orientation is explored during training).  One
logistic noise per unordered pair, ANTI-mirrored (``eps_ji = -eps_ij``), so
``d_ij + d_ji == 1`` per sample: the Toeplitz two-cycle-suppression property.

Combined edge weight (diagonal zeroed, hard_mask applied)::

    A_ij = z_edge_ij * d_ij

The reconstruction magnitude is carried entirely by the value stream; the former
multiplicative reconstruction-gain factor (``A = z_edge * d * g``) has been
REMOVED (it conflated structure with magnitude and was unsafe).

Key invariant: ``A_ij + A_ji`` is proportional to ``z_edge`` (the pair's total
edge mass is the sparse existence gate, merely split by direction).  All three
outcomes are reachable: no edge (``z_edge ~ 0``), ``i->j`` (``d_ij ~ 1``),
``j->i`` (``d_ij ~ 0``).

Why a SINGLE antisymmetric gate is NOT enough: if direction were folded into the
Hard-Concrete logit alone, the two open-probabilities ``sigmoid(A - c)`` and
``sigmoid(-A - c)`` would be locked together, so BOTH cannot approach zero -
"no edge" becomes unrepresentable and sparsity is lost.  The symmetric existence
factor is therefore mandatory.

Gradient routing: ``query`` / ``key`` are the *structural* (gate) projections;
both ``S_sym`` and ``A_anti`` derive from them, so the L0 penalty and the
direction gate are driven by the structural loss.  Must be trained with
``use_gradient_routing=True`` so the structure gate ``z_edge * d`` is driven by
the structural stream only.

Contract (mirrors the other inner-attention modules):
``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
          hard_mask=None, oracle=False)``
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
from causaliT.core.modules.extra_layers import sample_bkd_keep_mask
import torch.nn.functional as F

from causaliT.core.modules.gain_softmax import GainSoftmax
from causaliT.core.modules.topk_gate import TopKGate
from causaliT.utils.query_geometry import correct_query
from causaliT.utils.query_norm import (
    DEFAULT_DIR_TAU,
    DEFAULT_GATE_GAMMA,
    DEFAULT_GATE_TAU,
    DEFAULT_GATE_ZETA,
    apply_query_norm,
    coerce_fanin_scale,
    make_query_norm_log_scale,
    overspend_penalty,
)


class GatedSelfAttention(nn.Module):
    """Direction-aware selector: symmetric L0 existence x antisymmetric direction."""

    def __init__(
        self,
        attention_dropout: float = 0.0,
        register_entropy: bool = False,
        layer_name: Optional[str] = None,
        # Hard-Concrete existence-gate hyper-parameters (Louizos et al., ICLR 2018).
        # Defaults are the CALCULATED operating point documented in
        # docs/documentation/ATTENTION_TEMPERATURES.md: tau = 0.5 with the
        # symmetric stretch [-1.1, 1.1] gives kappa = 0 (the gate opens exactly
        # at logit 0) and kappa_1 = 0.5*ln(21) ~ 1.5223 (saturation).
        init_tau: float = DEFAULT_GATE_TAU,      # beta: existence-gate temperature
        gamma: float = DEFAULT_GATE_GAMMA,       # stretch lower bound (< 0)
        zeta: float = DEFAULT_GATE_ZETA,         # stretch upper bound (> 1)
        # Direction-gate Binary-Concrete temperature (coupled stochastic).
        dir_tau: float = DEFAULT_DIR_TAU,        # beta_dir
        # Additive bias on the direction-gate logit:
        # ``d = sigmoid(A_anti / dir_beta + dir_bias)``.  0.0 (default) is the
        # legacy coupled gate (``d_ij + d_ji == 1`` per sample).  A positive
        # bias opens BOTH directions (``E[d] = sigmoid(dir_bias)`` at the
        # symmetric centroid init) so the self path enters reconstruction near
        # full magnitude during BKD warmups; the bias must be annealed to 0
        # for structure phases (the adaptive trainer phase controller does
        # this via :meth:`set_dir_bias`).  While nonzero the Toeplitz
        # two-cycle-suppression coupling is relaxed.
        dir_bias: float = 0.0,

        # Centroid-collapse fix (structure score only): L2-normalise the query
        # so its DIRECTION, not its norm, drives selection, and use a fixed
        # sqrt(query_fanin_scale) score scale instead of 1/sqrt(E).  See
        # GatedCrossAttention for the full rationale; here it feeds the symmetric
        # existence gate (via S_sym) and the antisymmetric direction gate.
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

        # Batch-consistent key dropout (columns zeroed identically across batch).
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        batch_key_dropout_min_keys: int = 0,
        batch_key_dropout_deterministic: bool = False,
        # Eval-mode BKD (rung-aware validation): when True, BKD is ALSO applied
        # in eval mode using a dedicated seeded generator (seed =
        # ``batch_key_dropout_eval_seed`` + eval-forward counter), so validation
        # metrics measure the SAME key-budget regime as the current training
        # rung instead of the full-gate regime.  Deterministic across epochs
        # for a fixed val-set ordering; does not touch the training RNG.
        batch_key_dropout_eval: bool = False,
        batch_key_dropout_eval_seed: int = 12345,

        # Constant-score capacity protocol (Optuna): when not None the STRUCTURE
        # gate (existence) is frozen at this constant on every edge.
        optuna_protocol: Optional[float] = None,
        # Prior-softmax reconstruction gain (see causaliT/core/modules/
        # gain_softmax.py and GatedCrossAttention).  The DIRECTED gate
        # ``z_edge * d`` (diagonal zeroed, hard-masked) is the softmax prior:
        # the gain redistributes each row's mass across the directed support
        # only, so direction and existence stay gate-owned and gated-off edges
        # stay EXACTLY zero.  Inert while the buffer ``gain_lambda`` == 0.
        use_gain_softmax: bool = False,
        gain_num_queries: Optional[int] = None,
        gain_num_keys: Optional[int] = None,
        # Source-side top-k budget (see causaliT/core/modules/topk_gate.py).
        # When not None, the applied gate A is blanked per query row to its
        # top-k entries (rule selected inside the module) immediately after
        # BKD and before the attention dropout / value aggregation.  The
        # RETURNED posterior (second slot) stays the RAW directed posterior,
        # so HSIC weighting / eval DAG extraction are unaffected.  Default
        # None keeps the forward bit-identical to the dense behaviour.
        # Construct with exclude_diagonal=True for this square block.
        topk_gate: Optional[TopKGate] = None,
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

        # Source-side top-k budget (None = disabled, dense behaviour).
        self.topk_gate = topk_gate

        # Gate params are non-learnable constants (matching HardConcreteCrossAttention).
        self.beta = float(init_tau)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.dir_beta = float(dir_tau)
        # Direction-gate logit bias (see __init__); runtime-adjustable via
        # set_dir_bias (adaptive-trainer phase controller).
        self.dir_bias = float(dir_bias)

        # Centroid-collapse fix (structure score only); see __init__ doc.
        self.normalize_query = bool(normalize_query)
        self.query_fanin_scale = coerce_fanin_scale(query_fanin_scale)

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
        self._bkd_min_keys = int(batch_key_dropout_min_keys)
        self._bkd_deterministic = bool(batch_key_dropout_deterministic)
        self.register_buffer("_bkd_step", torch.zeros((), dtype=torch.long), persistent=False)
        # Eval-mode BKD state (see ctor docstring above): separate counter and
        # seed so validation masks are reproducible and independent of the
        # training RNG / annealing clock.
        self._bkd_eval: bool = bool(batch_key_dropout_eval)
        self._bkd_eval_seed: int = int(batch_key_dropout_eval_seed)
        self.register_buffer("_bkd_eval_step", torch.zeros((), dtype=torch.long), persistent=False)

        # Phase switch (adaptive trainer): when False, BKD is not applied but
        # the annealing clock keeps advancing (global run-level schedule).
        self._bkd_phase_active: bool = True
        # BKD keep mask of the last forward: ``(N,)`` float (1 = key kept)
        # when BKD was applied, else None (see GatedCrossAttention).
        self.last_bkd_keep: Optional[torch.Tensor] = None

        # BKD schedule shape: "linear" (legacy default) or "cosine" — a
        # periodic warmup curriculum oscillating in [p_base - amp, p_base + amp]
        # with a linearly decaying envelope, landing exactly on p1 at t = T.
        self._bkd_schedule: str = "linear"
        self._bkd_p_base: Optional[float] = None
        self._bkd_amp: Optional[float] = None
        self._bkd_cycles: Optional[float] = None

        # Open-gate override (dedicated warmup phase): when active the learned
        # structure gate is replaced by the BKD-coupled constant
        #   c(t) = c_end + (1 - c_end) * (p(t) - p_end) / (p_max - p_end)
        # i.e. fully open (~1) at high key dropout, equal to c_end — the value
        # the (frozen) learned gates hold at the warmup -> structure switch —
        # at p_end.  The constant carries no structural gradient.  c_end=None
        # -> auto-measured from the learned gate posterior on the first forward.
        self._open_gate_active: bool = False
        self._open_gate_c_end: Optional[float] = None
        self.last_open_gate_c: Optional[float] = None


        # Prior-softmax reconstruction gain (inert at lambda=0, the default).
        self.gain_softmax: Optional[GainSoftmax] = None
        if use_gain_softmax:
            if gain_num_queries is None or gain_num_keys is None:
                raise ValueError(
                    "use_gain_softmax=True requires gain_num_queries and "
                    "gain_num_keys (the static per-edge logit table shape)."
                )
            self.gain_softmax = GainSoftmax(gain_num_queries, gain_num_keys)

        # Diagnostics / regularisation hooks (populated in forward).
        #   score_tensor_for_sparsity - DIRECTED posterior P(z_edge>0)*d (B-mean),
        #     read by the L1 score-sparsity and NOTEARS terms.
        #   last_p_edge_on            - same DIRECTED posterior (B-mean), thresholded
        #     at eval to obtain the recovered adjacency.
        #   last_p_edge_undirected    - SYMMETRIC skeleton posterior P(z_edge>0) (B-mean).
        #   last_direction            - direction gate d (B-mean), diagnostics only.
        self.score_tensor_for_sparsity: Optional[torch.Tensor] = None
        self.last_p_edge_on: Optional[torch.Tensor] = None
        self.last_p_edge_undirected: Optional[torch.Tensor] = None
        self.last_direction: Optional[torch.Tensor] = None
        # TRUE applied weight of the last forward (B, N, N), DETACHED: the
        # directed gate after diagonal zeroing, hard mask, gain, BKD, top-k
        # blanking and dropout — exactly the matrix that multiplied the
        # values.  Read by the model's per-node adjacency-context injection
        # so the regressor context reflects the key subset actually used
        # (BKD/top-k consistent).
        self.last_applied_A: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Batch-consistent key dropout probability (with optional annealing)
    # ------------------------------------------------------------------
    def _current_bkd_p(self) -> Optional[float]:
        if self._bkd_p0 is None:
            return None
        if self._bkd_anneal is None or self._bkd_anneal <= 0:
            return float(self._bkd_p0)
        frac = min(1.0, float(self._bkd_step.item()) / float(self._bkd_anneal))
        if self._bkd_schedule == "cosine":
            # Periodic warmup curriculum: oscillates in [p_base - amp,
            # p_base + amp] under a linearly decaying envelope, landing
            # exactly on p1 (p_end) at frac = 1.
            p_end = float(self._bkd_p1)
            p_base = (float(self._bkd_p_base) if self._bkd_p_base is not None
                      else float(self._bkd_p0))
            amp = float(self._bkd_amp) if self._bkd_amp is not None else 0.0
            cycles = float(self._bkd_cycles) if self._bkd_cycles else 1.0
            osc = 0.5 * (1.0 + math.cos(2.0 * math.pi * cycles * frac))
            return p_end + (1.0 - frac) * ((p_base - p_end) + amp * osc)
        return float(self._bkd_p0) + frac * (float(self._bkd_p1) - float(self._bkd_p0))

    def set_bkd_schedule(
        self,
        p0: Optional[float],
        p1: Optional[float] = None,
        annealing_batches: Optional[int] = None,
        schedule: str = "linear",
        p_base: Optional[float] = None,
        amp: Optional[float] = None,
        cycles: Optional[float] = None,
    ) -> None:
        """Override the BKD schedule at run time (adaptive-trainer phase
        controller).  The step counter is NOT reset: the anneal stays a
        global, run-level clock.

        ``schedule="cosine"`` selects the periodic warmup curriculum (see
        ``_current_bkd_p``); ``p_base``/``amp``/``cycles`` are its
        oscillation center / amplitude / period count and ``p1`` is the
        landing value (p_end)."""
        self._bkd_p0 = p0
        self._bkd_p1 = p1 if p1 is not None else p0
        self._bkd_anneal = annealing_batches
        self._bkd_schedule = str(schedule)
        self._bkd_p_base = p_base
        self._bkd_amp = amp
        self._bkd_cycles = cycles

    def set_bkd_phase_active(self, active: bool) -> None:
        """Enable/disable BKD application for the current training phase."""
        self._bkd_phase_active = bool(active)

    def set_bkd_sampling(
        self,
        min_keys: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> None:
        """Override the BKD sampling mode / min-keys floor at run time
        (adaptive-trainer phase controller).  ``None`` leaves the
        corresponding setting unchanged."""
        if min_keys is not None:
            self._bkd_min_keys = int(min_keys)
        if deterministic is not None:
            self._bkd_deterministic = bool(deterministic)
    def set_bkd_eval(
        self, enabled: bool, seed: Optional[int] = None
    ) -> None:
        """Toggle eval-mode BKD (rung-aware validation).  When enabled, eval
        forward passes apply BKD with a dedicated seeded generator so
        validation metrics reflect the current key-budget regime.  ``seed``
        (optional) overrides the eval generator base seed."""
        self._bkd_eval = bool(enabled)
        if seed is not None:
            self._bkd_eval_seed = int(seed)

    def _sample_bkd_keep_eval(
        self, num_keys: int, p: float, device: torch.device
    ) -> torch.Tensor:
        """Seeded, reproducible BKD keep mask for eval mode (float, 1=keep).

        Uses a private ``torch.Generator`` seeded by
        ``_bkd_eval_seed + _bkd_eval_step`` so successive val batches cycle
        through different key subsets deterministically, without consuming
        the training RNG stream."""
        gen = torch.Generator(device=device)
        gen.manual_seed(self._bkd_eval_seed + int(self._bkd_eval_step.item()))
        keep = sample_bkd_keep_mask(
            num_keys,
            p,
            self._bkd_min_keys,
            self._bkd_deterministic,
            device,
            generator=gen,
        ).to(torch.float32)
        self._bkd_eval_step += 1
        return keep


    def set_open_gate_mode(
        self, active: bool, c_end: Optional[float] = None
    ) -> None:
        """Toggle the BKD-coupled open-gate override (warmup phase).

        When active, forward() replaces the learned structure gate with the
        constant ``c(t) = c_end + (1-c_end)*(p(t)-p_end)/(p_max-p_end)``
        read live from the BKD schedule.  ``c_end=None`` auto-measures the
        learned gate posterior (mean off-diagonal) on the first forward —
        the gates are frozen all warmup, so this is exactly the value they
        hold at the warmup -> structure switch (continuity by construction).
        Deactivating restores the learned gates.
        """
        self._open_gate_active = bool(active)
        if c_end is not None:
            self._open_gate_c_end = float(c_end)
        if not active:
            self.last_open_gate_c = None

    def set_dir_bias(self, value: float) -> None:
        """Set the direction-gate logit bias (adaptive-trainer phase
        controller).  0.0 restores the legacy coupled gate."""
        self.dir_bias = float(value)

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
    # Structural score (shared by forward and the cheap posterior probe)
    # ------------------------------------------------------------------
    def _structural_raw(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        transitive_W: Optional[torch.Tensor] = None,
        transitive_delta: float = 0.0,
    ) -> torch.Tensor:
        """``raw[b, n, m]`` - the asymmetric structural score before the Toeplitz split.

        The optional TRANSITIVE CORRECTION is applied here, on the unit query and
        BEFORE the per-node norm budget ``M_i``, so the coordinate taken away from
        a mediated (grandparent) edge is handed back to the surviving parents by
        the subsequent re-normalisation.  ``transitive_W`` is detached, so the
        correction biases the geometry without giving the loss a shortcut.
        """
        E_s = query.shape[-1]
        q_s = query
        if transitive_W is not None:
            # u <- u - (W * (c + delta)) khat   (exact on an orthonormal frame)
            q_s = correct_query(q_s, key, transitive_W, delta=transitive_delta)
        if self.normalize_query:
            if self.query_norm_learnable:
                # q_hat * M_i (per-node learnable budget); scale = sqrt(fanin).
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
        return torch.nan_to_num(raw, nan=0.0)

    @torch.no_grad()
    def structure_posterior(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        hard_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DETERMINISTIC directed posterior ``P(z_edge>0) * d``, batch-averaged.

        A cheap probe of the structure only: no Hard-Concrete / direction noise,
        no value aggregation and no ``last_*`` diagnostics write.  Used
        as pass 1 of the transitive correction, where the trigger must be
        identical in train and eval mode (a noisy trigger would make the
        correction depend on the sampled gate).

        Returns:
            ``(N, N)`` posterior with ``[i, j] = P(j -> i)``, zero diagonal.
        """
        raw = self._structural_raw(query, key)
        S_sym = 0.5 * (raw + raw.transpose(-1, -2))
        A_anti = 0.5 * (raw - raw.transpose(-1, -2))
        pi = torch.sigmoid(S_sym - self._l0_offset) * torch.sigmoid(
            A_anti / self.dir_beta + self.dir_bias
        )
        n = pi.shape[-1]
        pi = pi.masked_fill(
            torch.eye(n, device=pi.device, dtype=torch.bool).unsqueeze(0), 0.0
        )
        if hard_mask is not None:
            hm = hard_mask if hard_mask.dim() == 3 else hard_mask.unsqueeze(0)
            pi = pi * hm.to(pi.dtype)
        return pi.mean(dim=0)

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
        # Value-structure QUERY injection: per-QUERY value term already projected
        # to the value output width (shape (B, N, d) or (B, N, H, d)).  Added as
        # ``(sum_j A_ij) * value_query`` - the exact, memory-cheap decomposition
        # of concatenating the query identity into a (linear, bias-free) W_V^q.
        value_query: Optional[torch.Tensor] = None,
        # Geometric TRANSITIVE correction (grandparent suppression).  ``(N, N)``
        # or ``(B, N, N)`` DETACHED weights computed by the LAYER from the square
        # posterior (see causaliT/utils/query_geometry.py); None = disabled.
        transitive_W: Optional[torch.Tensor] = None,
        transitive_delta: float = 0.0,
        # Prior-softmax gain: precomputed data-dependent score ``<q^v, k^v> /
        # sqrt(d_g)``, shape (B, N, N); None = static-logit-only gain.  Only
        # consumed when the GainSoftmax module exists and lambda > 0.
        gain_scores: Optional[torch.Tensor] = None,
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

        N = L

        # ---- Structural score, Toeplitz-decomposed ----------------------
        # Centroid-collapse fix (unit-normalised query + sqrt(fanin) scale) and
        # the optional transitive correction both live in ``_structural_raw``,
        # shared with the deterministic ``structure_posterior`` probe.
        raw = self._structural_raw(
            query, key, transitive_W=transitive_W, transitive_delta=transitive_delta
        )
        # Row-wise gradient routing: the transposed copy is DETACHED so that
        # query q_j receives no gradient through other rows' gates.  Without
        # this, A_anti[i, j] = (raw_ij - raw_ji)/2 lets the loss raise p_ij by
        # pushing q_j AWAY from k_i (the 'lower p_ji' shortcut), which produces
        # HSIC gradients against the true-parent centroid (toy-study evidence:
        # scripts/_toy_antisym_coupling.py).  Forward values are unchanged.
        rawT = raw.transpose(-1, -2).detach()
        S_sym = 0.5 * (raw + rawT)                              # symmetric
        A_anti = 0.5 * (raw - rawT)                             # antisymmetric

        if oracle:
            # ---- Oracle: the ground-truth DAG IS the structure gate ------
            # z_ij = hard_mask_ij (true topology).  Direction/existence are
            # taken directly from the oracle adjacency.
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
        elif self._open_gate_active:
            # ---- BKD-coupled open gates (dedicated warmup phase) --------
            p_now = self._current_bkd_p()
            p_end = float(self._bkd_p1) if self._bkd_p1 is not None else 0.0
            _p_base = (
                self._bkd_p_base if self._bkd_p_base is not None else self._bkd_p0
            )
            p_max = (
                float(_p_base) + float(self._bkd_amp or 0.0)
                if _p_base is not None
                else None
            )
            if self._open_gate_c_end is None:
                # Auto-measure c_end: the learned-gate eval posterior, mean
                # over off-diagonal entries, at the CURRENT (init) structural
                # state.  The gates are frozen for the whole warmup, so this
                # is exactly the value they will hold at the switch.
                with torch.no_grad():
                    pi0 = torch.sigmoid(S_sym - self._l0_offset) * torch.sigmoid(
                        A_anti / self.dir_beta + self.dir_bias
                    )
                    off = ~torch.eye(N, device=pi0.device, dtype=torch.bool)
                    self._open_gate_c_end = float(pi0[:, off].mean())
            if p_now is None or p_max is None or p_max <= p_end:
                c = self._open_gate_c_end
            else:
                c = self._open_gate_c_end + (1.0 - self._open_gate_c_end) * (
                    p_now - p_end
                ) / (p_max - p_end)
            c = float(min(max(c, 0.0), 1.0))
            self.last_open_gate_c = c
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
                direction = torch.sigmoid(
                    (eps_d + A_anti) / self.dir_beta + self.dir_bias
                )
            else:
                direction = torch.sigmoid(A_anti / self.dir_beta + self.dir_bias)

            structure = z_edge * direction                        # directed structure gate

            # Posterior that the (undirected) edge exists: P(z_edge > 0).
            p_edge_undirected = torch.sigmoid(S_sym - self._l0_offset)
            p_directed = p_edge_undirected * direction

        # ---- Final attention weight: the directed structure gate ---------
        A = structure                                             # (B, N, N)

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

        # ---- Prior-softmax reconstruction gain (lambda-ramped) -----------
        # Redistributes each row's DIRECTED gate mass within the directed
        # support: A = (1-lambda)*A + lambda*n*A*e^s/D.  Gated-off, forbidden
        # and diagonal edges stay EXACTLY zero; the row mass is preserved.
        # Inert (returns A unchanged) while lambda == 0.
        if self.gain_softmax is not None:
            A = self.gain_softmax(A, gain_scores)

        # ---- L0 penalty: expected active (allowed) UNDIRECTED edges ------
        # Count each unordered pair once (strictly-upper triangle) since the
        # existence posterior is symmetric.
        triu_mask = torch.triu(
            torch.ones(N, N, device=A.device, dtype=A.dtype), diagonal=1
        ).unsqueeze(0)
        l0_penalty = (p_edge_masked * triu_mask).sum(dim=(-2, -1)).mean()

        # ---- Batch-consistent key dropout -------------------------------
        bkd_p = self._current_bkd_p()
        self.last_bkd_keep = None
        if self.training and bkd_p is not None:
            # The annealing clock advances in every training phase (global
            # schedule); ``_bkd_phase_active`` only gates the application.
            self._bkd_step += 1
            if self._bkd_phase_active and bkd_p > 0.0:
                keep = sample_bkd_keep_mask(
                    N, bkd_p, self._bkd_min_keys, self._bkd_deterministic, A.device
                ).to(A.dtype)  # (N,)
                A = A * keep.view(1, 1, N)
                self.last_bkd_keep = keep
        elif (
            self._bkd_eval
            and bkd_p is not None
            and self._bkd_phase_active
            and bkd_p > 0.0
        ):
            # Rung-aware validation: apply the SAME key budget in eval mode
            # (seeded generator, no training-RNG consumption) so val metrics
            # measure the regime the decoder is actually being trained on.
            keep = self._sample_bkd_keep_eval(N, bkd_p, A.device).to(A.dtype)
            A = A * keep.view(1, 1, N)
            self.last_bkd_keep = keep


        # ---- Source-side top-k budget -----------------------------------
        # Blank all but the top-k gate entries per query row (after BKD, so
        # only visible keys compete for the budget).  The returned posterior
        # below is the RAW directed posterior - only the applied weight A is
        # blanked.
        if self.topk_gate is not None:
            A = self.topk_gate(A, hard_mask=hard_mask)

        # ---- Attention-weight dropout -----------------------------------
        A = self.dropout(A)
        # Expose the TRUE applied weight (detached) for the per-node
        # adjacency-context injection.
        self.last_applied_A = A.detach()

        # ---- Value aggregation ------------------------------------------
        if value.dim() == 4:
            out = torch.einsum("bnm,bmhd->bnhd", A, value)        # (B, N, H, d)
        elif value.dim() == 3:
            out = torch.einsum("bnm,bmd->bnd", A, value)          # (B, N, d)
        else:
            raise ValueError(
                f"GatedSelfAttention value must be 3-D or 4-D, got {value.dim()}-D."
            )

        # ---- Value-structure QUERY injection (additive query term) --------
        # V_ij = W_V([v_j;e_j]) + W_V^q(e_i^q).  W_V^q(e_i^q) is independent of
        # the key j, so it factors out of the aggregation as
        # ``(sum_j A_ij) * value_query_i`` - the exact, memory-cheap equivalent
        # of concatenating the query identity into a linear, bias-free W_V.
        # ``A`` here is the TRUE applied weight (the structure gate, diagonal
        # zeroed, masked, dropped).
        if value_query is not None:
            row_sum = A.sum(dim=-1)                                # (B, N)
            if out.dim() == 4:
                out = out + row_sum[:, :, None, None] * value_query   # (B, N, H, d)
            else:
                out = out + row_sum[:, :, None] * value_query         # (B, N, d)

        # ---- Diagnostics / regularisation signals -----------------------
        self.score_tensor_for_sparsity = p_directed.mean(dim=0)   # (N, N) directed
        self.last_p_edge_on = p_directed.mean(dim=0)              # (N, N) directed
        self.last_p_edge_undirected = p_edge_undirected.mean(dim=0)  # (N, N) skeleton
        self.last_direction = direction.mean(dim=0).detach()      # (N, N) diag only

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
            f"zeta={self.zeta}, dir_beta={self.dir_beta})"
        )
