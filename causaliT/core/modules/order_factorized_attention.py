"""
OrderFactorizedSelfAttention: symmetric variable SELECTION x global causal ORDER.
Design document: ``docs/ideas/ORDER_FACTORIZED_VISION.md``.

Two symmetric selectors, each owned by ONE stage and chosen by
``adaptive_training.{recon,structure}.gate_opt``.

* RECONSTRUCTION selector ``G_R`` (``recon_gate_opt``):
  - ``"gated"``: Hard-Concrete existence from the query/key descriptors,
    routed to the reconstruction parameter group.
  - ``"softmax"``: competitive weights ``W = softmax_row(S_sym)`` over the
    AVAILABLE parents (see "Applied weight").
  - ``"none"``: ``G_R = 1`` off-diagonal (queries frozen).
* STRUCTURAL selector ``G_S`` (``struct_gate_opt``: ``"gated"`` | ``"none"``):
  free symmetric Hard-Concrete logits (``struct_gate_logits``, structural
  group), initialised open (``struct_att_selector_init_logit``).

SHARED selector (``shared_att_selector``, gated only): ONE existence matrix
(``struct_gate_logits``) applied in EVERY forward; the two gate_opts then
select which stages TRAIN it (recon loss / structural loss) instead of
which selector exists.  A stage whose selector is off applies the shared G
detached.

HANDOFF mode (``shared_att_selector`` + ``struct_att_selector_init="recon"``,
both selectors on): before the first structure phase only ``G_R`` exists in
the forwards.  On entering the first structure phase ``G_S <- G_R`` is
copied once and ``G_R`` is retired: from then on every forward applies
``G_S``; structure forwards train it, reconstruction forwards apply it
DETACHED.  State persists via ``struct_selector_synced``.

Stage policy (``set_stage_policy(stage, gate_sampling, order_sampling)``,
driven per phase by the adaptive trainer):

* ``stage="recon"``: applies ``G_R``; the order never gets gradient.
* ``stage="structure"``: applies ``G_S`` if on, otherwise ``G_R`` (a plain
  weight: the structural optimizer does not own its parameters).
* ``gate_sampling``: ``"hard"`` = exact 0/1 draw ``B ~ Bernoulli(pi)`` with
  the straight-through gradient ``B + pi - pi.detach()`` (in structure
  forwards: of the LEARNED ``pi``); ``"soft"`` = Hard-Concrete sample.
* ``order_sampling``: ``"soft"`` (relaxed Sinkhorn mask ``M``), ``"hard"``
  (the step's sampled hard order, detached), ``"hard_st"`` (structure:
  hard forward, gradient of ``M``), ``"oriented"`` (recon: hard order at a
  dedicated Gumbel scale with per-step orientation,
  docs/ideas/ORIENTED_RECON_SAMPLING.md) or ``"none"`` (no order mask).

The EDGE (posterior, exclusion graph, ``_last_pi``, L0) is existence x
order with the existence coming ONLY from ``G_S`` (or the shared G):
``E = pi_S * M``; without ``G_S``, ``E = M`` (full DAG of the order).
``G_R`` (gate or softmax) is a RECONSTRUCTION mixing weight and never
enters the edge; it is never L0-penalized either: the L0 penalty
(``lambda_l0``) applies to ``G_S`` / the shared G only.

* Batch-key dropout (``BKDMixin``): constant per-key keep mask; dropped keys
  get zero weight and zero gradient.
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
``M_path`` is the stage's order mask (``order_sampling``), ``keep`` the BKD
key mask.

* gated / none: ``A = (G * M_path * keep)^T``.  Masking REMOVES mass.
* softmax: ``A = G_S^T * W`` with (child <- parent)
  ``W_ij = softmax_j(S_ij + log M_ij)`` over the available parents.  Hard
  masks (diagonal, hard order, BKD, hard_mask) enter as -inf, the soft
  order as the log-prior ``log M`` (zero gradient at exact zero priors).
  Masking RENORMALIZES the survivors and preserves their
  ratios, so the regressor input keeps its scale under any order.
  ``hard_st``: ``W_hard + W_soft - W_soft.detach()``.  ``G_S`` (structure
  forwards, if on) multiplies afterwards without renormalization.  Rows
  with no available parent are zero.  ``S_sym`` is detached in structure
  forwards (owned by reconstruction).

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
  order, also when ``order_sampling="none"``.
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


_STAGES = ("recon", "structure")
_GATE_MODES = ("hard", "soft")
_RECON_GATE_OPTS = ("gated", "softmax", "none")
_STRUCT_GATE_OPTS = ("gated", "none")
# Order sampling per stage.  "oriented" is the hard order with a dedicated
# Gumbel scale and a per-step orientation (driven by trainer / forecaster).
_ORDER_SAMPLING = {
    "recon": ("none", "hard", "soft", "oriented"),
    "structure": ("soft", "hard", "hard_st", "none"),
}
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
        recon_gate_opt: Reconstruction selector ``G_R``
            (``adaptive_training.recon.gate_opt``): ``"gated"`` (Hard-Concrete
            gate), ``"softmax"`` (row softmax of ``S_sym`` over the available
            parents) or ``"none"`` (all-ones; queries frozen).
        struct_gate_opt: Structural selector ``G_S``
            (``adaptive_training.structure.gate_opt``): ``"gated"`` or
            ``"none"``.  Applied in structure forwards only.
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
        shared_att_selector: ONE selector (``struct_gate_logits``) applied in
            every forward; the two toggles select which stages train it (see
            module docstring).  Incompatible with
            ``struct_att_selector_init="recon"``.
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
        recon_gate_opt: str = "gated",
        struct_gate_opt: str = "none",
        struct_att_selector_init_logit: float = 2.0,
        struct_att_selector_init: str = "constant",
        shared_att_selector: bool = False,
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
        if recon_gate_opt not in _RECON_GATE_OPTS:
            raise ValueError(
                f"recon gate_opt must be one of {_RECON_GATE_OPTS}, got {recon_gate_opt!r}.")
        if struct_gate_opt not in _STRUCT_GATE_OPTS:
            raise ValueError(
                f"structure gate_opt must be one of {_STRUCT_GATE_OPTS}, got {struct_gate_opt!r}.")
        # Both 'none' is the order-only ablation (graph = full DAG of the order).
        recon_on = recon_gate_opt != "none"
        struct_on = struct_gate_opt != "none"
        if shared_att_selector and not (recon_on or struct_on):
            raise ValueError(
                "shared_att_selector with both gate_opt 'none': the shared G "
                "would never be trained.")
        if struct_att_selector_init == "recon" and not (struct_on and recon_on):
            raise ValueError(
                "struct_att_selector_init='recon' requires recon.gate_opt != "
                "'none' and structure.gate_opt == 'gated'.")
        if shared_att_selector and struct_gate_opt == "none":
            raise ValueError("shared_att_selector requires structure.gate_opt='gated'.")
        if shared_att_selector and recon_gate_opt == "softmax":
            raise ValueError(
                "shared_att_selector is incompatible with recon.gate_opt="
                "'softmax' (the shared G is a multiplicative gate).")
        if struct_att_selector_init == "recon" and recon_gate_opt == "softmax":
            raise ValueError(
                "struct_att_selector_init='recon' is incompatible with "
                "recon.gate_opt='softmax': softmax scores are not existence "
                "logits and cannot seed G_S.")

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
        self.recon_gate_opt = str(recon_gate_opt)
        self.struct_gate_opt = str(struct_gate_opt)
        self.recon_selector_on = recon_on
        self.struct_selector_on = struct_on
        # softmax: G_R is a row softmax of S_sym over the AVAILABLE parents
        # (order / BKD masks renormalize); G_S still multiplies afterwards.
        self.recon_softmax = self.recon_gate_opt == "softmax"
        # Shared mode: ONE existence matrix (struct_gate_logits) applied in
        # every forward; the toggles then select WHICH stages train it
        # (recon loss / structural loss) instead of which selector exists.
        self.shared_selector = bool(shared_att_selector)
        # Handoff mode (shared + "recon" init): G_R alone until the one-off
        # copy G_S <- G_R, then G_S alone in every forward (G_R retired).
        # After the handoff only the structural loss trains G_S; recon
        # forwards apply it detached.
        self.handoff = self.shared_selector and str(struct_att_selector_init) == "recon"
        self.gate_mode = "hard"
        self.struct_gate_mode = "soft"     # gate of relaxed (structure) forwards
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
        # by gradient_routing.STRUCTURAL_PATTERNS.  In shared mode this is THE
        # selector, so it exists whenever shared_selector is on.
        self.struct_gate_logits = (
            nn.Parameter(torch.full((N, N), float(struct_att_selector_init_logit)))
            if (self.struct_selector_on or self.shared_selector) else None)
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

        # Stage policy (set_stage_policy): which selector / order mask the
        # forward applies.  Default = structure stage, soft order.
        self.stage = "structure"
        self.order_sampling = "soft"
        # Gumbel scale of the APPLIED hard order in recon forwards
        # (None = use noise_scale; see set_recon_noise_scale).
        self.recon_noise_scale: Optional[float] = None
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
    def set_recon_noise_scale(self, value: Optional[float]) -> None:
        """Gumbel scale of the applied order with ``order_sampling=
        "oriented"`` (training only).  None = structural ``noise_scale``."""
        if value is not None and value < 0.0:
            raise ValueError(f"recon_noise_scale must be >= 0, got {value}.")
        self.recon_noise_scale = None if value is None else float(value)

    def set_stage_policy(
        self,
        stage: str,
        gate_sampling: Optional[str] = None,
        order_sampling: Optional[str] = None,
    ) -> None:
        """Select the forward of a training stage.

        Args:
            stage: ``"recon"`` or ``"structure"``.  Decides which selector
                is applied / trained (see module docstring).
            gate_sampling: ``"hard"`` (straight-through Bernoulli) or
                ``"soft"`` (Hard-Concrete) gate of THIS stage; None keeps
                the stage's current value.  Ignored by ``gate_opt="softmax"``
                (deterministic weights).
            order_sampling: applied order mask of this stage; None keeps the
                current value.  recon: none | hard | soft | oriented;
                structure: soft | hard | hard_st | none.
        """
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}.")
        if gate_sampling is not None and gate_sampling not in _GATE_MODES:
            raise ValueError(
                f"gate_sampling must be one of {_GATE_MODES}, got {gate_sampling!r}.")
        if order_sampling is None:
            order_sampling = self.order_sampling if stage == self.stage else (
                "none" if stage == "recon" else "soft")
        if order_sampling not in _ORDER_SAMPLING[stage]:
            raise ValueError(
                f"{stage} order_sampling must be one of {_ORDER_SAMPLING[stage]}, "
                f"got {order_sampling!r}.")
        self.stage = stage
        if gate_sampling is not None:
            if stage == "recon":
                self.gate_mode = gate_sampling
            else:
                self.struct_gate_mode = gate_sampling
        if order_sampling is not None:
            self.order_sampling = order_sampling

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

    @property
    def shared_active(self) -> bool:
        """True when the single shared G drives every forward: always in
        plain shared mode, only after the copy in handoff mode."""
        if self.handoff:
            return bool(self.struct_selector_synced)
        return self.shared_selector

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
        if self.struct_selector_on or self.shared_selector:
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

    @staticmethod
    def _masked_softmax(
        scores: torch.Tensor,
        avail: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Row softmax over the available entries, with an optional prior.

        ``W_ij = avail_ij * prior_ij * exp(s_ij) / sum_k avail_ik * prior_ik * exp(s_ik)``,
        i.e. ``softmax_j(s_ij + log prior_ij)`` with ``avail == 0`` as -inf.
        Log-domain normalization avoids underflow from unavailable maxima.
        Exact zero priors have zero gradient by convention; positive priors
        retain the log-prior gradient. Rows with no mass are zero.

        Args:
            scores: ``(N, N)`` logits (child <- parent).
            avail: ``(N, N)`` binary availability (0 = hard-masked).
            prior: optional ``(N, N)`` soft mask in [0, 1] (e.g. relaxed order).
        """
        is_av = avail > 0
        logits = scores
        if prior is not None:
            is_av = is_av & (prior > 0)
            safe_prior = torch.where(is_av, prior, torch.ones_like(prior))
            logits = logits + safe_prior.log()
        has_mass = is_av.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~is_av, float("-inf"))
        logits = torch.where(has_mass, logits, torch.zeros_like(logits))
        return torch.softmax(logits, dim=-1).masked_fill(~is_av, 0.0)

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
        """Existence of the EDGE: ``pi_S`` (or the shared G) when the
        structural selector is on, else all-ones (edge = full DAG of the
        order).  ``pi_R`` never enters the edge: ``G_R`` is a reconstruction
        mixing weight, not an edge existence."""
        off = 1.0 - torch.eye(self.num_nodes, device=query.device, dtype=query.dtype)
        if self.shared_active:
            # No edge_offset: the shared G is the learned existence in every
            # phase and its gate is centred at the l0 offset only.
            return torch.sigmoid(self._struct_logits() - self._l0_offset) * off
        if self.struct_selector_on and not (self.handoff and not self.shared_active):
            return torch.sigmoid(self._struct_logits() - self._l0_offset) * off
        return off


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
        # pi_R: G_R inclusion probability of the MIXING gate only (never the
        # edge).  pi_S / shared G: existence of the EDGE (posterior, exclusion
        # graph, L0); without G_S the edge is the full DAG of the order
        # (pi = off), so nothing from reconstruction enters the structure.
        is_struct = self.stage == "structure"
        if self._struct_sync_pending:
            # One-off G_S <- G_R: identical existence (pi_S = pi_R) and, with
            # G_S's offset fixed at 0, the identical relaxed gate.  Done
            # before the selector choice so handoff mode switches to G_S in
            # this very forward.
            with torch.no_grad():
                S_copy = self._existence_score(query, key)
                self.struct_gate_logits.copy_(S_copy - self.edge_offset)
                self.struct_selector_synced.fill_(1)
            self._struct_sync_pending = False
        shared = self.shared_active
        # Handoff mode before the copy: G_S does not exist yet in the
        # forwards; G_R is the only selector (also on the relaxed path).
        struct_on = self.struct_selector_on and not (self.handoff and not shared)
        if shared:
            S_sym, pi_R = None, off
            L_S = self._struct_logits()
            if self.handoff:
                # Post-handoff: only the structural loss trains G_S; recon
                # forwards apply it detached.
                train_stage = is_struct
            else:
                train_stage = self.struct_selector_on if is_struct else self.recon_selector_on
            L_use = L_S if train_stage else L_S.detach()
            pi = torch.sigmoid(L_use - self._l0_offset) * off
        else:
            # Includes handoff mode before the copy: G_R from query/key,
            # trained by the reconstruction loss.
            S_sym, pi_R = self._recon_pi(query, key, off)
        if not shared:
            if struct_on:
                L_S = self._struct_logits()
                pi = torch.sigmoid(L_S - self._l0_offset) * off
            else:
                pi = off
        theta = self.theta()
        noise = self._get_noise() if self.training else None
        perturbed = theta + self.noise_scale * noise["gumbel"] if self.training else theta

        # Relaxed order (also feeds the posterior / soft closure) and the
        # hard order of THIS step (same noise).
        P = log_sinkhorn(perturbed / self.perm_tau, self.sinkhorn_iters)
        M = self.order_mask(P)                              # source -> target
        M_hard = self.order_mask(hard_assignment(perturbed))

        # Applied order (source -> target) of this stage, as a pair
        # (M_soft, M_hard_app): M_soft carries the gradient (None = no
        # gradient path), M_hard_app the forward value of hard sampling.
        #   soft     -> (M, None)            relaxed Sinkhorn mask
        #   hard     -> (None, M_hard)       sampled hard order, detached
        #   hard_st  -> (M, M_hard)          hard forward, relaxed gradient
        #   oriented -> (None, M_rec)        hard order at the recon Gumbel
        #               scale (docs/ideas/ORIENTED_RECON_SAMPLING.md)
        #   none     -> (None, None)         no order mask
        # Recon forwards never train the order: their M is detached.
        # The structural M_hard (posterior / exclusion) is unchanged.
        osamp = self.order_sampling
        if osamp == "oriented" and self.training and self.recon_noise_scale is not None:
            M_rec = self.order_mask(hard_assignment(
                theta + self.recon_noise_scale * noise["gumbel"]))
        else:
            M_rec = M_hard
        M_rel = M if is_struct else M.detach()
        if osamp == "soft":
            M_soft, M_hard_app = M_rel, None
        elif osamp == "hard_st":
            # Eval keeps the relaxed mask (as the former straight-through flag).
            M_soft, M_hard_app = (M_rel, M_hard) if self.training else (M_rel, None)
        elif osamp in ("hard", "oriented"):
            M_soft, M_hard_app = None, M_rec
        else:
            M_soft, M_hard_app = None, None
        # Applied orientation: the anti-causal pass flips every selected edge
        # (each node then attends to its selected LATER nodes).
        if self.orientation == "anticausal":
            M_soft = None if M_soft is None else M_soft.transpose(0, 1)
            M_hard_app = None if M_hard_app is None else M_hard_app.transpose(0, 1)
        # Multiplicative applied order (gated mode): ST combination for hard_st.
        if M_soft is not None and M_hard_app is not None:
            M_mult = M_hard_app + M_soft - M_soft.detach()
        elif M_soft is not None:
            M_mult = M_soft
        elif M_hard_app is not None:
            M_mult = M_hard_app
        else:
            M_mult = off

        def _relaxed_gate(logits, noise_key, offset):
            if self.training:
                u = noise[noise_key]
                eps = self._mirror_upper(torch.log(u) - torch.log1p(-u))
                s_e = torch.sigmoid((eps + logits - offset) / self.beta)
            else:
                s_e = torch.sigmoid((logits - offset) / self.beta)
            return (s_e * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0) * off

        # Step's Bernoulli draws (train only, same uniforms): of pi_R for the
        # hard reconstruction MIXING gate, of the EDGE existence pi for the
        # exclusion graph (all-ones without G_S).
        if self.training:
            base_draw = pi.detach() if shared else pi_R.detach()
            B_draw = self._mirror_upper((noise["bern"] < base_draw).to(pi.dtype))
            B_learn = self._mirror_upper((noise["bern"] < pi.detach()).to(pi.dtype))
        else:
            B_draw = B_learn = None

        # 1) Structural or shared gate, followed by separate G_R composition.
        #    structure: G_S if on, else G_R (soft, or ST Bernoulli of the
        #    learned pi when struct_gate_mode == "hard");
        #    reconstruction: G_R per gate_mode.
        #    Shared mode: the shared G everywhere, per the stage's gate mode.
        #    recon gate_opt="softmax": G_R is NOT a multiplicative gate; it
        #    is the row softmax W built in step 2, so G = 1 except for G_S
        #    in structure forwards (multiplies W, no renormalization).
        if self.recon_softmax:
            if is_struct and struct_on:
                if self.struct_gate_mode == "hard":
                    G = (B_learn + pi - pi.detach()) if self.training else \
                        (pi.detach() > self.existence_threshold).to(pi.dtype)
                else:
                    G = _relaxed_gate(L_S, "hc_s", 0.0)
            else:
                G = off
        elif shared:
            if is_struct:
                if self.struct_gate_mode == "hard":
                    G = (B_learn + pi - pi.detach()) if self.training else \
                        (pi.detach() > self.existence_threshold).to(pi.dtype)
                else:
                    G = _relaxed_gate(L_use, "hc_s", 0.0)
            elif self.gate_mode == "hard":
                G = (B_draw + pi - pi.detach()) if self.training else \
                    (pi.detach() > self.existence_threshold).to(pi.dtype)
            else:
                G = _relaxed_gate(L_use, "hc", self.edge_offset)
        elif is_struct and self.struct_gate_mode == "hard" and (
                self.struct_selector_on or self.recon_selector_on):
            # Hard structure gates: ST Bernoulli of the MIXING gate: the edge
            # existence pi_S when G_S is on, else the G_R inclusion pi_R.
            # Eval thresholds it.
            pi_mix = pi if struct_on else pi_R
            B_mix = B_learn if struct_on else B_draw
            if self.training:
                G = B_mix + pi_mix - pi_mix.detach()
            else:
                G = (pi_mix.detach() > self.existence_threshold).to(pi_mix.dtype)
        elif is_struct and struct_on:
            G = _relaxed_gate(L_S, "hc_s", 0.0)
        elif not self.recon_selector_on:
            G = off
        elif not is_struct and self.gate_mode == "hard":
            if self.training:
                G = B_draw + pi_R - pi_R.detach()           # straight-through
            else:
                G = (pi_R.detach() > self.existence_threshold).to(pi.dtype)
        else:
            G = _relaxed_gate(S_sym, "hc", self.edge_offset)
        # Separate G_R and G_S compose multiplicatively in structure forwards.
        # Shared/handoff gates are applied once, never squared.
        if is_struct and struct_on and not shared and not self.recon_softmax:
            if not self.recon_selector_on:
                G_R = off
            elif self.gate_mode == "hard":
                G_R = B_draw if self.training else (
                    pi_R.detach() > self.existence_threshold).to(pi.dtype)
            else:
                G_R = _relaxed_gate(S_sym.detach(), "hc", self.edge_offset)
            G = G * G_R.detach()
        # Batch-key dropout: constant per-key mask (no gradient to dropped
        # keys).  Drawn once per forward; diagnostics stay BKD-free.
        keep = self._bkd_keep(N, query.device)
        hm = None
        if hard_mask is not None:
            hm = (hard_mask[0] if hard_mask.dim() == 3 else hard_mask).to(query.dtype)

        # 2) Applied weight (child <- parent layout).
        if self.recon_softmax:
            # Row softmax of the symmetric score over the AVAILABLE parents:
            # hard masks (diagonal, hard order, BKD, hard_mask) act as -inf,
            # the soft order as the log-prior log M (zero gradient at exact
            # zero priors). G_S multiplies afterwards (no renorm).
            avail = off
            if keep is not None:
                avail = avail * keep.to(avail.dtype).view(1, N)
            if hm is not None:
                avail = avail * hm
            Mt_soft = None if M_soft is None else M_soft.transpose(0, 1)
            Mt_hard = None if M_hard_app is None else M_hard_app.transpose(0, 1)
            # Scores are owned by reconstruction: detached in structure forwards.
            S_w = S_sym.detach() if is_struct else S_sym
            if Mt_soft is not None and Mt_hard is not None:
                W_soft = self._masked_softmax(S_w, avail, Mt_soft)
                W_hard = self._masked_softmax(S_w, avail * Mt_hard, None)
                W = W_hard + W_soft - W_soft.detach()
            elif Mt_hard is not None:
                W = self._masked_softmax(S_w, avail * Mt_hard, None)
            else:
                W = self._masked_softmax(S_w, avail, Mt_soft)
            A = G.transpose(0, 1) * W
        else:
            # Gated: multiplicative gate x order (masking removes mass).
            A = (G * M_mult).transpose(0, 1)
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
        # L0 of the EDGE: expected open undirected pairs of G_S (or the
        # shared G), not masked by the order.  Without G_S there is no
        # learned existence: zero L0.  G_R is never L0-penalized.
        pi_l0 = pi if (shared or struct_on) else torch.zeros_like(off)
        if hm is not None:
            A, posterior, excl, pi_l0 = A * hm, posterior * hm, excl * hm, pi_l0 * hm

        # L0: expected open undirected pairs (structural lambda_l0).
        l0_penalty = torch.triu(pi_l0, diagonal=1).sum()

        self.score_tensor_for_sparsity = excl.detach()      # hard DAG
        self._last_pi = pi
        self._last_M = M
        with torch.no_grad():
            self.last_sinkhorn_err = float(torch.maximum(
                (P.sum(0) - 1).abs().max(), (P.sum(1) - 1).abs().max()))
            r_sorted = torch.sort(self.order_scores.detach()).values
            self.last_score_gap = float((r_sorted[1:] - r_sorted[:-1]).min())

        # 3) Batch-key dropout on the applied weight (no-op for softmax,
        # already excluded there).
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

    def set_order_temperature(
        self,
        tau: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> None:
        """Anneal the Sinkhorn temperature / Gumbel noise of the order.

        Plain attribute writes (both are re-read at every forward), exposed
        as a setter so the adaptive trainer can anneal them over training.
        """
        if tau is not None:
            if tau <= 0.0:
                raise ValueError(f"tau must be > 0, got {tau}.")
            self.perm_tau = float(tau)
        if noise_scale is not None:
            if noise_scale < 0.0:
                raise ValueError(f"noise_scale must be >= 0, got {noise_scale}.")
            self.noise_scale = float(noise_scale)

    def __repr__(self):
        return (
            f"OrderFactorizedSelfAttention(N={self.num_nodes}, beta={self.beta}, "
            f"noise_scale={self.noise_scale}, tau={self.perm_tau}, "
            f"stage={self.stage}, order_sampling={self.order_sampling}, "
            f"recon_gate_opt={self.recon_gate_opt}, struct_gate_opt={self.struct_gate_opt}, "
            f"detach_exclusion={self.detach_exclusion}, "
            f"exclusion_existence={self.exclusion_existence}, "
            f"shared_selector={self.shared_selector}, "
            f"struct_init={self.struct_init}, "
            f"gate_mode={self.gate_mode}, "
            f"orientation={self.orientation})"
        )

