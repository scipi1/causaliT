"""
OrderFactorizedSelfAttention: symmetric variable SELECTION x global causal ORDER.
Design document: ``docs/ideas/ORDER_FACTORIZED_VISION.md``.

Two symmetric selectors, each owned by ONE stage and toggled independently.
They never multiply: each forward applies exactly one of them.

* RECONSTRUCTION selector ``G_R`` (``toggle_on_recon_att_selector``):
  Hard-Concrete existence from the query/key descriptors, routed to the
  reconstruction parameter group.  Off: ``G_R = 1`` off-diagonal (queries
  frozen).
* STRUCTURAL selector ``G_S`` (``toggle_on_struct_att_selector``): free
  symmetric Hard-Concrete logits (``struct_gate_logits``, structural group),
  initialised open (``struct_att_selector_init_logit``).

Applied gate per forward:

* reconstruction forwards (``order_path`` ``"hard"`` / ``"none"``): ``G_R``;
* structure forwards (``order_path="relaxed"``): ``G_S`` if the structural
  selector is on, otherwise ``G_R`` (a plain weight: the structural optimizer
  does not own its parameters), times the relaxed order.

The LEARNED graph (posterior, exclusion graph, ``_last_pi``) uses ``pi_S``
when the structural selector is on, ``pi_R`` otherwise.  The structural
objective (``training.anti_adjacency_lambda``) lives in the forecaster; this
module exposes the hooks it needs:

* ``gate_mode`` (``set_gate_mode``, from ``adaptive_training.recon_gates``):
  ``G_R`` of reconstruction forwards.  ``"hard"``: exact 0/1 draw
  ``B ~ Bernoulli(pi)`` with the straight-through gradient
  ``B + pi - pi.detach()``.  ``"soft"``: Hard-Concrete sample.
* ``order_path`` (``set_order_path``): ``"relaxed"`` (structure phases,
  Sinkhorn mask with gradient), ``"hard"`` (the step's sampled hard order,
  detached; ``use_order_in_recon: true``) or ``"none"`` (no order mask:
  nodewise regression on the symmetric selection).
* Batch-key dropout (``BKDMixin``): constant per-key keep mask applied
  last; dropped keys get zero weight and zero gradient.
* ``orientation`` (``set_orientation``): ``"causal"`` applies ``Z = G * M``
  (each node attends to its selected EARLIER nodes); ``"anticausal"`` applies
  ``Z = G * M^T`` (selected LATER nodes), used by the anti-causal term.
  Noise, existence and the stored posterior / exclusion graph are always the
  CAUSAL ones, so the two passes of one step are paired.

Self-attention over N homogeneous nodes.  The directed structure factorizes
into a global causal ORDER and a symmetric EXISTENCE gate::

    raw    = <q_i, k_j> / sqrt(E)               # (N, N), global descriptors
    S_sym  = (raw + raw^T) / 2                   # existence score
    G      = HardConcrete(S_sym)                 # one draw per unordered pair
    P      = Sinkhorn((Theta(r) + s*E) / tau)    # relaxed permutation
    M      = P T P^T                             # M_ij ~ "i before j"
    Z      = G * M                               # Z_ij = weight of edge i -> j

Conventions
===========
Mathematical matrices (``M``, ``Z``) are SOURCE -> TARGET (``Z[i, j]`` is the
edge ``i -> j``).  The applied weight, the returned posterior and
``score_tensor_for_sparsity`` use the codebase layout ROW = CHILD,
COL = PARENT, i.e. the transpose.

Applied weight
==============
``A = (G * M_path * keep)^T`` with ``G`` the existence gate (``gate_mode``;
always soft on the relaxed order path), ``M_path`` the order mask
(``order_path``: relaxed ``M``, hard ``M_hard`` from a maximum-weight
assignment on the perturbed scores, or all-ones off-diagonal) and ``keep``
the BKD key mask.  With ``order_straight_through`` the relaxed path uses
``M_hard`` forward and the gradient of ``M``.

Noise is GLOBAL: drawn once per gradient step by :meth:`resample` (called by
the forecaster at the start of each training step) and reused by every
forward in that step.  Eval mode is deterministic: no noise, noiseless order
(sort of the scores), existence ``pi > thr`` for hard gates and the
deterministic relaxed gate otherwise.

Descendant exclusion
====================
* Detached (default): ``score_tensor_for_sparsity`` is the step's hard DAG
  ``(pi > thr) * M_hard`` (same Gumbel noise as the relaxed ``P``) in
  child <- parent layout, consumed by ``build_hsic_pair_mask``.  With
  ``order_exclusion_existence="sample"`` the training-time existence is the
  step's ``Bernoulli(pi)`` draw instead (with hard gates and the hard order
  path it is exactly the applied DAG).  The exclusion graph always uses the
  order, also when ``order_path="none"``.
* Attached (ablation): :meth:`soft_descendant_scores` returns the
  differentiable soft closure of the relaxed ``pi * M``.
"""

from typing import Dict, Optional

import math

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from causaliT.core.modules.extra_layers import BKDMixin
from causaliT.utils.descendant_mask import soft_transitive_closure
from causaliT.utils.query_norm import (
    DEFAULT_GATE_GAMMA,
    DEFAULT_GATE_TAU,
    DEFAULT_GATE_ZETA,
)


__all__ = ["OrderFactorizedSelfAttention", "log_sinkhorn", "hard_assignment"]


_ORDER_PATHS = ("relaxed", "hard", "none")
_GATE_MODES = ("hard", "soft")
_ORIENTATIONS = ("causal", "anticausal")
_R_INITS = ("flat", "random")
_TNORMS = ("min", "prod")
_EXCL_EXISTENCE = ("threshold", "sample")
_STRUCT_INITS = ("constant", "recon")


def log_sinkhorn(log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
    """Log-domain Sinkhorn normalization; returns an approx. doubly stochastic P."""
    for _ in range(int(n_iters)):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()


def hard_assignment(scores: torch.Tensor) -> torch.Tensor:
    """Maximum-weight assignment of a node-by-position matrix (no gradient).

    Returns a permutation matrix, ``P[i, k] = 1`` iff node i is at position k.
    """
    s = scores.detach().cpu().double().numpy()
    rows, cols = linear_sum_assignment(s, maximize=True)
    P = torch.zeros_like(scores)
    P[torch.as_tensor(rows, device=scores.device),
      torch.as_tensor(cols, device=scores.device)] = 1.0
    return P


class OrderFactorizedSelfAttention(BKDMixin, nn.Module):
    """Dual-orientation order-factorized self-attention (see module docstring).

    Contract (mirrors ``GatedSelfAttention``):
    ``forward(query, key, value, mask_miss_k, mask_miss_q, pos, causal_mask,
    hard_mask=None, oracle=False, value_query=None)`` returns
    ``(out, posterior, aux)`` with ``aux = {"entropy", "l0_penalty"}``.

    Args:
        num_nodes: N, number of nodes (square self-attention).
        attention_dropout: Dropout on the applied weights.
        register_entropy: Return the row entropy of the applied weights.
        layer_name: Optional name (diagnostics).
        init_tau: Hard-Concrete existence temperature (beta).
        gamma, zeta: Hard-Concrete stretch bounds (gamma < 0 < 1 < zeta).
        order_noise_scale: Gumbel noise scale on the order scores.
        order_tau: Sinkhorn relaxation temperature.
        order_sinkhorn_iters: Sinkhorn sweeps.
        order_r_init: ``"flat"`` (zeros) or ``"random"`` (N(0, 0.01^2)).
        order_existence_threshold: ``pi`` threshold, in (0, 1), for the hard
            existence evidence of the exclusion graph and eval hard graphs.
        order_detach_exclusion: Detached hard exclusion (True) or attached
            soft closure (False, ablation).  Read by the forecaster.
        order_exclusion_tnorm: T-norm of the soft closure (``"min"``/``"prod"``).
        order_exclusion_existence: Existence of the detached exclusion graph
            in training: ``"threshold"`` (``pi > thr``) or ``"sample"``
            (``Bernoulli(pi)``, the step's draw).  Eval always thresholds.
        toggle_on_recon_att_selector: Learned reconstruction selector ``G_R``
            (True) or all-ones (False; with the structural selector also off
            the graph is the full DAG of the order, order-only ablation).
        toggle_on_struct_att_selector: Learned structural selector ``G_S``,
            applied in structure forwards only (relaxed order path).
        struct_att_selector_init_logit: Initial ``G_S`` logit.  2.0 gives
            ``pi_S ~ 0.88`` (mostly open) while ~18% of the Hard-Concrete
            draws stay unclamped (with gradient) at beta = 0.5; higher
            values open the gate further but starve it of gradient.
            Used only with ``struct_att_selector_init="constant"``.
        struct_att_selector_init: ``"constant"`` (all logits =
            ``struct_att_selector_init_logit``) or ``"recon"``: ONE copy of the
            G_R existence logits ``S_sym - edge_offset`` (so ``pi_S = pi_R``)
            at the first forward after :meth:`request_struct_selector_sync`
            (called by the adaptive trainer on entering the first structure
            phase).  The persistent buffer ``struct_selector_synced`` prevents
            a second copy, also after a checkpoint reload.
        order_straight_through: Relaxed training forward uses the step's hard
            order ``M_hard`` with the gradient of the Sinkhorn mask ``M``
            (``M_hard + M - M.detach()``).  Eval is unaffected.
        batch_key_dropout*: BKD settings (see ``BKDMixin``); usually driven
            per phase by the adaptive trainer.
    """

    def __init__(
        self,
        num_nodes: int,
        attention_dropout: float = 0.0,
        register_entropy: bool = False,
        layer_name: Optional[str] = None,
        init_tau: float = DEFAULT_GATE_TAU,
        gamma: float = DEFAULT_GATE_GAMMA,
        zeta: float = DEFAULT_GATE_ZETA,
        order_noise_scale: float = 1.0,
        order_tau: float = 1.0,
        order_sinkhorn_iters: int = 20,
        order_r_init: str = "flat",
        order_existence_threshold: float = 0.5,
        order_detach_exclusion: bool = True,
        order_exclusion_tnorm: str = "min",
        order_exclusion_existence: str = "threshold",
        toggle_on_recon_att_selector: bool = True,
        toggle_on_struct_att_selector: bool = False,
        struct_att_selector_init_logit: float = 2.0,
        struct_att_selector_init: str = "constant",
        order_straight_through: bool = False,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        batch_key_dropout_min_keys: int = 0,
        batch_key_dropout_deterministic: bool = False,
        batch_key_dropout_eval: bool = False,
        batch_key_dropout_eval_seed: int = 12345,
    ):
        super().__init__()
        N = int(num_nodes)
        if N < 2:
            raise ValueError(f"num_nodes must be >= 2, got {num_nodes}.")
        if not (gamma < 0.0 < zeta and zeta > 1.0):
            raise ValueError(
                f"HardConcrete stretch bounds require gamma < 0 < 1 < zeta, "
                f"got gamma={gamma}, zeta={zeta}."
            )
        if order_noise_scale < 0.0:
            raise ValueError(f"order_noise_scale must be >= 0, got {order_noise_scale}.")
        if order_tau <= 0.0:
            raise ValueError(f"order_tau must be > 0, got {order_tau}.")
        if int(order_sinkhorn_iters) < 1:
            raise ValueError(f"order_sinkhorn_iters must be >= 1, got {order_sinkhorn_iters}.")
        if order_r_init not in _R_INITS:
            raise ValueError(f"order_r_init must be one of {_R_INITS}, got {order_r_init!r}.")
        if not (0.0 < float(order_existence_threshold) < 1.0):
            raise ValueError(
                f"order_existence_threshold must be in (0, 1), got {order_existence_threshold!r}."
            )
        if order_exclusion_existence not in _EXCL_EXISTENCE:
            raise ValueError(
                f"order_exclusion_existence must be one of {_EXCL_EXISTENCE}, "
                f"got {order_exclusion_existence!r}."
            )
        if order_exclusion_tnorm not in _TNORMS:
            raise ValueError(
                f"order_exclusion_tnorm must be one of {_TNORMS}, got {order_exclusion_tnorm!r}."
            )
        if struct_att_selector_init not in _STRUCT_INITS:
            raise ValueError(
                f"struct_att_selector_init must be one of {_STRUCT_INITS}, "
                f"got {struct_att_selector_init!r}.")
        if struct_att_selector_init == "recon" and not (
                toggle_on_struct_att_selector and toggle_on_recon_att_selector):
            raise ValueError(
                "struct_att_selector_init='recon' requires both "
                "toggle_on_struct_att_selector and toggle_on_recon_att_selector.")

        self.num_nodes = N
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = bool(register_entropy)
        self.layer_name = layer_name

        # Hard-Concrete constants (same semantics as GatedSelfAttention).
        self.beta = float(init_tau)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self._l0_offset = float(self.beta * math.log(-self.gamma / self.zeta))
        # Deterministic existence offset (plain attribute, as in GSA).
        self.edge_offset = 0.0

        self.noise_scale = float(order_noise_scale)
        self.perm_tau = float(order_tau)
        self.sinkhorn_iters = int(order_sinkhorn_iters)
        self.existence_threshold = float(order_existence_threshold)
        self.detach_exclusion = bool(order_detach_exclusion)
        self.exclusion_tnorm = str(order_exclusion_tnorm)
        self.exclusion_existence = str(order_exclusion_existence)
        self.recon_selector_on = bool(toggle_on_recon_att_selector)
        self.struct_selector_on = bool(toggle_on_struct_att_selector)
        self.straight_through = bool(order_straight_through)
        self.gate_mode = "hard"
        self._init_bkd(batch_key_dropout, batch_key_dropout_p_final,
                       batch_key_dropout_annealing_batches,
                       batch_key_dropout_min_keys, batch_key_dropout_deterministic,
                       batch_key_dropout_eval, batch_key_dropout_eval_seed)
        self.orientation = "causal"

        # Learnable node scores (larger = earlier).  The name ``order_scores``
        # is matched by gradient_routing.STRUCTURAL_PATTERNS.
        init = torch.zeros(N) if order_r_init == "flat" else 0.01 * torch.randn(N)
        self.order_scores = nn.Parameter(init)

        # Structural selector G_S: free symmetric logits (only the strict
        # upper triangle is read).  The name ``struct_gate_logits`` is matched
        # by gradient_routing.STRUCTURAL_PATTERNS.
        self.struct_gate_logits = (
            nn.Parameter(torch.full((N, N), float(struct_att_selector_init_logit)))
            if self.struct_selector_on else None)
        # One-off copy from G_R ("recon" init).  The buffer is registered only
        # in that mode, so "constant" state dicts are unchanged.
        self.struct_init = str(struct_att_selector_init)
        self._struct_sync_pending = False
        if self.struct_init == "recon":
            self.register_buffer("struct_selector_synced", torch.zeros((), dtype=torch.uint8))

        # Fixed decreasing, zero-centred position levels and position mask.
        k = torch.arange(1, N + 1, dtype=torch.float32)
        self.register_buffer("mu", (N + 1) / 2.0 - k, persistent=False)
        self.register_buffer("T", torch.triu(torch.ones(N, N), diagonal=1), persistent=False)

        self.order_path = "relaxed"
        self._noise: Optional[Dict[str, torch.Tensor]] = None

        # Per-step state / diagnostics (populated by forward).
        self.score_tensor_for_sparsity: Optional[torch.Tensor] = None
        self.last_applied_A: Optional[torch.Tensor] = None
        self.last_sinkhorn_err: float = 0.0
        self.last_score_gap: float = 0.0
        self._last_pi: Optional[torch.Tensor] = None
        self._last_M: Optional[torch.Tensor] = None


    # ------------------------------------------------------------------
    # Phase / noise control
    # ------------------------------------------------------------------
    def set_order_path(self, mode: str) -> None:
        """Order mask of the applied weight: ``"relaxed"``, ``"hard"`` or
        ``"none"``."""
        if mode not in _ORDER_PATHS:
            raise ValueError(f"order_path must be one of {_ORDER_PATHS}, got {mode!r}.")
        self.order_path = mode

    def set_gate_mode(self, mode: str) -> None:
        """Existence gate of non-relaxed forwards: ``"hard"`` (straight-
        through Bernoulli) or ``"soft"`` (Hard-Concrete)."""
        if mode not in _GATE_MODES:
            raise ValueError(f"gate_mode must be one of {_GATE_MODES}, got {mode!r}.")
        self.gate_mode = mode

    def set_orientation(self, mode: str) -> None:
        """Select the applied orientation: ``"causal"`` (G * M) or
        ``"anticausal"`` (G * M^T).  Reset to ``"causal"`` by the caller."""
        if mode not in _ORIENTATIONS:
            raise ValueError(f"orientation must be one of {_ORIENTATIONS}, got {mode!r}.")
        self.orientation = mode

    def request_struct_selector_sync(self) -> bool:
        """Arm the one-off ``G_S <- G_R`` copy (``"recon"`` init only); it is
        applied at the next forward.  Returns True if a copy is pending."""
        if self.struct_init == "recon" and not bool(self.struct_selector_synced):
            self._struct_sync_pending = True
        return self._struct_sync_pending

    @property
    def struct_selector_is_synced(self) -> bool:
        return self.struct_init == "recon" and bool(self.struct_selector_synced)

    def set_edge_offset(self, value: float) -> None:
        """Set the deterministic existence offset (adaptive-trainer drift)."""
        self.edge_offset = float(value)

    def resample(self, generator: Optional[torch.Generator] = None) -> None:
        """Draw the global structural noise for ONE gradient step.

        Gumbel order noise, Hard-Concrete uniforms and Bernoulli uniforms,
        each ``(N, N)``; reused by every forward until the next call.
        """
        N = self.num_nodes
        dev, dt = self.order_scores.device, self.order_scores.dtype

        def _u():
            u = torch.rand(N, N, generator=generator, dtype=dt)
            return u.clamp(1e-6, 1.0 - 1e-6).to(dev)

        self._noise = {
            "gumbel": -torch.log(-torch.log(_u())),
            "hc": _u(),
            "bern": _u(),
        }
        # Drawn last and only when needed, so runs without the structural
        # selector keep the exact same random stream.
        if self.struct_selector_on:
            self._noise["hc_s"] = _u()

    def _get_noise(self) -> Dict[str, torch.Tensor]:
        if self._noise is None or self._noise["gumbel"].device != self.order_scores.device:
            self.resample()
        return self._noise

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    def theta(self) -> torch.Tensor:
        """Node-by-position scores ``Theta_ik = -(r_i - mu_k)^2`` (centred r)."""
        r = self.order_scores - self.order_scores.mean()
        return -(r.unsqueeze(1) - self.mu.to(r.dtype).unsqueeze(0)) ** 2

    def order_mask(self, P: torch.Tensor) -> torch.Tensor:
        """``M = P T P^T`` with zero diagonal (source -> target)."""
        M = P @ self.T.to(P.dtype) @ P.transpose(0, 1)
        return M * (1.0 - torch.eye(self.num_nodes, device=P.device, dtype=P.dtype))

    def deterministic_order_mask(self) -> torch.Tensor:
        """Hard noiseless order mask (sort of the scores), source -> target."""
        return self.order_mask(hard_assignment(self.theta()))

    @staticmethod
    def _mirror_upper(x: torch.Tensor) -> torch.Tensor:
        """Copy the strict upper triangle onto the lower one (zero diagonal)."""
        up = torch.triu(x, diagonal=1)
        return up + up.transpose(0, 1)

    def _existence_score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Symmetric ``(N, N)`` existence score from global descriptors."""
        raw = torch.einsum("bne,bme->bnm", query, key) / math.sqrt(query.shape[-1])
        raw = raw.mean(dim=0)          # identical across the batch (global)
        return 0.5 * (raw + raw.transpose(0, 1))

    def inclusion_prob(self, S_sym: torch.Tensor) -> torch.Tensor:
        """Hard-Concrete inclusion probability ``P(z > 0)``."""
        return torch.sigmoid(S_sym - self._l0_offset - self.edge_offset)

    def _struct_logits(self) -> torch.Tensor:
        """Symmetric ``G_S`` logits (strict upper triangle mirrored)."""
        return self._mirror_upper(self.struct_gate_logits)

    def _recon_pi(self, query: torch.Tensor, key: torch.Tensor, off: torch.Tensor):
        """``(S_sym, pi_R)``; ``(None, off)`` with the selector off."""
        if not self.recon_selector_on:
            return None, off
        S_sym = self._existence_score(query, key)
        return S_sym, self.inclusion_prob(S_sym) * off

    def structural_pi(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Existence probability of the LEARNED graph: ``pi_S`` if the
        structural selector is on, else ``pi_R`` (zero diagonal)."""
        off = 1.0 - torch.eye(self.num_nodes, device=query.device, dtype=query.dtype)
        if self.struct_selector_on:
            # No edge_offset: the drift offset belongs to G_R only.
            return torch.sigmoid(self._struct_logits() - self._l0_offset) * off
        return self._recon_pi(query, key, off)[1]


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
        value_query: Optional[torch.Tensor] = None,
    ):
        if causal_mask or oracle:
            raise NotImplementedError(
                "OrderFactorizedSelfAttention supports neither causal masking nor oracle mode."
            )
        if query.dim() != 3 or key.dim() != 3:
            raise ValueError(
                "OrderFactorizedSelfAttention expects a single structural head "
                "(3-D query/key: (B, N, E)); use shared_dag_across_heads=True."
            )
        B, N, _ = query.shape
        if key.shape[1] != N or N != self.num_nodes:
            raise ValueError(
                f"OrderFactorizedSelfAttention requires square scores with "
                f"N={self.num_nodes}, got L={N}, S={key.shape[1]}."
            )

        off = 1.0 - torch.eye(N, device=query.device, dtype=query.dtype)
        # pi_R: reconstruction selector (all-ones when toggled off).  pi_S:
        # structural selector.  ``pi`` = existence of the LEARNED graph.
        S_sym, pi_R = self._recon_pi(query, key, off)
        if self._struct_sync_pending:
            # One-off G_S <- G_R: identical existence (pi_S = pi_R) and, with
            # G_S's offset fixed at 0, the identical relaxed gate.
            with torch.no_grad():
                self.struct_gate_logits.copy_(S_sym.detach() - self.edge_offset)
                self.struct_selector_synced.fill_(1)
            self._struct_sync_pending = False
        if self.struct_selector_on:
            L_S = self._struct_logits()
            pi = torch.sigmoid(L_S - self._l0_offset) * off
        else:
            pi = pi_R
        theta = self.theta()
        noise = self._get_noise() if self.training else None
        perturbed = theta + self.noise_scale * noise["gumbel"] if self.training else theta

        # Relaxed order (also feeds the posterior / soft closure) and the
        # hard order of THIS step (same noise).
        P = log_sinkhorn(perturbed / self.perm_tau, self.sinkhorn_iters)
        M = self.order_mask(P)                              # source -> target
        M_hard = self.order_mask(hard_assignment(perturbed))

        # Straight-through order (training only): forward value = the step's
        # hard order, gradient = that of the relaxed Sinkhorn mask.
        M_fwd = (M_hard + M - M.detach()) if (self.straight_through and self.training) else M
        # Applied orientation: the anti-causal pass flips every selected edge
        # (each node then attends to its selected LATER nodes).
        if self.orientation == "anticausal":
            M_fwd, M_app_hard = M_fwd.transpose(0, 1), M_hard.transpose(0, 1)
        else:
            M_app_hard = M_hard

        def _relaxed_gate(logits, noise_key, offset):
            if self.training:
                u = noise[noise_key]
                eps = self._mirror_upper(torch.log(u) - torch.log1p(-u))
                s_e = torch.sigmoid((eps + logits - offset) / self.beta)
            else:
                s_e = torch.sigmoid((logits - offset) / self.beta)
            return (s_e * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0) * off

        # Step's Bernoulli draws (train only, same uniforms): of pi_R for the
        # hard reconstruction gate, of the learned pi for the exclusion graph.
        if self.training:
            B_draw = self._mirror_upper((noise["bern"] < pi_R.detach()).to(pi.dtype))
            B_learn = (self._mirror_upper((noise["bern"] < pi.detach()).to(pi.dtype))
                       if self.struct_selector_on else B_draw)
        else:
            B_draw = B_learn = None

        # 1) Applied gate: ONE selector per forward, never both.
        #    structure (relaxed order path): G_S if on, else G_R (soft);
        #    reconstruction (hard / none): G_R per gate_mode.
        relaxed = self.order_path == "relaxed"
        if relaxed and self.struct_selector_on:
            G = _relaxed_gate(L_S, "hc_s", 0.0)
        elif not self.recon_selector_on:
            G = off
        elif not relaxed and self.gate_mode == "hard":
            if self.training:
                G = B_draw + pi_R - pi_R.detach()           # straight-through
            else:
                G = (pi_R.detach() > self.existence_threshold).to(pi.dtype)
        else:
            G = _relaxed_gate(S_sym, "hc", self.edge_offset)
        # 2) Order mask: relaxed (with gradient), hard (detached) or none.
        if relaxed:
            Z = G * M_fwd
        elif self.order_path == "hard":
            Z = G * M_app_hard
        else:
            Z = G * off

        A = Z.transpose(0, 1)                               # child <- parent
        # Returned posterior (child <- parent).  Train: relaxed pi * M.  Eval:
        # pi * M_hard with the NOISELESS order, so threshold-at-0.5 DAG
        # extraction reads the learned DAG (pi * M stays below 0.5 while the
        # relaxed order is flat, which would report an empty graph).
        posterior = (pi * (M if self.training else M_hard)).transpose(0, 1)
        # Exclusion graph: existence from the threshold (deterministic) or
        # from the step's Bernoulli draw (stochastic mask, train only; it is
        # the applied DAG in hard mode).  Eval always thresholds.
        if self.training and self.exclusion_existence == "sample":
            B_ex = B_learn
        else:
            B_ex = (pi.detach() > self.existence_threshold).to(pi.dtype)
        excl = (B_ex * M_hard).transpose(0, 1)
        pi_l0 = pi_R
        if hard_mask is not None:
            hm = (hard_mask[0] if hard_mask.dim() == 3 else hard_mask).to(A.dtype)
            A, posterior, excl, pi_l0 = A * hm, posterior * hm, excl * hm, pi_R * hm

        # L0 of the RECONSTRUCTION selector: expected open UNDIRECTED pairs,
        # not masked by the order (consumed by lambda_l0_selector).
        l0_penalty = torch.triu(pi_l0, diagonal=1).sum()

        self.score_tensor_for_sparsity = excl.detach()      # hard DAG
        self._last_pi = pi
        self._last_M = M
        with torch.no_grad():
            self.last_sinkhorn_err = float(torch.maximum(
                (P.sum(0) - 1).abs().max(), (P.sum(1) - 1).abs().max()))
            r_sorted = torch.sort(self.order_scores.detach()).values
            self.last_score_gap = float((r_sorted[1:] - r_sorted[:-1]).min())

        # 3) Batch-key dropout: constant per-key mask (no gradient to dropped
        # keys).  Diagnostics (posterior / exclusion graph) stay BKD-free.
        keep = self._bkd_keep(N, A.device)
        if keep is not None:
            A = A * keep.to(A.dtype).view(1, N)
        A_b = self.dropout(A.unsqueeze(0).expand(B, N, N))
        self.last_applied_A = A_b.detach()
        if value.dim() == 4:
            out = torch.einsum("bnm,bmhd->bnhd", A_b, value)
        elif value.dim() == 3:
            out = torch.einsum("bnm,bmd->bnd", A_b, value)
        else:
            raise ValueError(f"value must be 3-D or 4-D, got {value.dim()}-D.")
        if value_query is not None:
            row_sum = A_b.sum(dim=-1)
            if out.dim() == 4:
                out = out + row_sum[:, :, None, None] * value_query
            else:
                out = out + row_sum[:, :, None] * value_query

        entropy = None
        if self.register_entropy:
            w = A_b / (A_b.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=-1)

        aux = {"entropy": entropy, "l0_penalty": l0_penalty}
        return out, posterior.unsqueeze(0).expand(B, N, N), aux


    # ------------------------------------------------------------------
    # Attached exclusion and graph sampling
    # ------------------------------------------------------------------
    def soft_descendant_scores(self) -> torch.Tensor:
        """Differentiable soft descendant scores of the last forward.

        Soft closure of the relaxed ``pi * M`` (child <- parent input);
        ``desc[i, j]`` scores node j as a descendant of node i, matching the
        HSIC pair layout.  NOT detached.
        """
        if self._last_pi is None or self._last_M is None:
            raise RuntimeError("soft_descendant_scores() called before forward().")
        post = (self._last_pi * self._last_M).transpose(0, 1)
        return soft_transitive_closure(post, tnorm=self.exclusion_tnorm, hops=None)

    @torch.no_grad()
    def sample_graphs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo hard DAG samples from the learned distribution.

        Args:
            query, key: Projected structural query/key ``(B, N, E)``.
            n_samples: Number of graphs.
            seed: Optional seed of a dedicated generator.

        Returns:
            ``graphs`` ``(n, N, N)`` source -> target binary DAGs,
            ``edge_freq`` ``(N, N)``, and ``order_freq`` ``(N, N)``
            (fraction of samples with i before j).
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        N = self.num_nodes
        off = 1.0 - torch.eye(N, device=query.device, dtype=query.dtype)
        pi = self.structural_pi(query, key)
        theta = self.theta()
        graphs, orders = [], []
        for _ in range(int(n_samples)):
            u = torch.rand(N, N, generator=gen).clamp(1e-6, 1 - 1e-6).to(theta)
            M_h = self.order_mask(hard_assignment(
                theta - self.noise_scale * torch.log(-torch.log(u))))
            ub = torch.rand(N, N, generator=gen).to(theta)
            graphs.append(self._mirror_upper((ub < pi).to(pi.dtype)) * M_h)
            orders.append(M_h)
        graphs = torch.stack(graphs)
        return {
            "graphs": graphs,
            "edge_freq": graphs.mean(dim=0),
            "order_freq": torch.stack(orders).mean(dim=0),
        }

    def __repr__(self):
        return (
            f"OrderFactorizedSelfAttention(N={self.num_nodes}, beta={self.beta}, "
            f"noise_scale={self.noise_scale}, tau={self.perm_tau}, "
            f"order_path={self.order_path}, detach_exclusion={self.detach_exclusion}, "
            f"exclusion_existence={self.exclusion_existence}, "
            f"recon_selector={self.recon_selector_on}, "
            f"struct_selector={self.struct_selector_on}, "
            f"struct_init={self.struct_init}, "
            f"straight_through={self.straight_through}, "
            f"gate_mode={self.gate_mode}, "
            f"orientation={self.orientation})"
        )

