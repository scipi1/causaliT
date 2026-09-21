"""
TopKGate: source-side top-k budget for gate posteriors.

Motivation
==========
The gated attention modules assemble each node's regressor from ALL open
gates (a dense regressor over the selected set).  Because residual
independence is preserved under supersets of the true parents, the HSIC
objective is FLAT in the superfluous-gate direction and downstream pruning
does not occur without an explicit L0 penalty.  TopKGate imposes a
source-side cardinality budget instead: per query row (target node), only
the top-k gate entries are allowed to multiply the value matrix; every
other entry is blanked to EXACTLY zero before the value aggregation.  A
superfluous selected gate now costs a budget slot, turning sparsity into a
competitive selection problem.

Design
======
The class is a generic dispatcher.  The *selection rule* is chosen at
construction (``method=...``) from a registry; ``forward`` performs the
shared plumbing (validity mask, per-row budget broadcast, annealing clock,
diagnostics) and delegates the mask computation to the registered rule.

Gradient policy is a property of the RULE, not of the framework: the
generic class never detaches the returned mask.  Each registered rule
declares ``mask_grad``:

* ``mask_grad=False`` (e.g. ``noisy_hard_k``) - the mask is detached;
  excluded gates receive no gradient through the value path and are trained
  only through posterior-level losses (gate-weighted HSIC nodal terms,
  entropy, gate regularisers), which read the RAW posterior returned by the
  attention module, not the blanked weight.
* ``mask_grad=True`` (future differentiable rules: ramp between order
  statistics, Gumbel-top-k with straight-through, ...) - the mask itself
  carries gradient to (some of) the scores.

First rule: ``noisy_hard_k``
----------------------------
Hard top-k with stochastic budget slack, the hierarchical analogue of
batch-consistent key dropout (BKD): BKD randomises WHICH keys are visible;
this randomises HOW MANY visible keys are used.  During training the
effective per-row budget is ``k + eps`` with ``eps ~ Uniform{0, ..., c_t}``
(per-row by default); the slack ``c_t`` anneals linearly from
``slack_init`` to ``slack_final`` over ``annealing_batches`` training
forwards, converging to the exact hard top-k.  A gate sitting j slots below
the cutoff is explored with probability P(eps >= j), so exploration
concentrates near the decision boundary and no gradient estimator is
needed: each step's mask is a constant w.r.t. the scores, and gradients
flow honestly to whichever gates are included that step.  At eval the rule
is the deterministic exact top-k (eps = 0).

Contract of every registered rule
---------------------------------
``fn(module, scores, k_row, valid, slack) -> mask`` where

* ``scores`` : ``(..., L, S)`` gate values with forbidden entries already
  set to ``-inf`` (ranking scores; may or may not be detached - the rule
  decides),
* ``k_row``  : ``(L,)`` long tensor, the base per-row budget,
* ``valid``  : bool tensor broadcastable to ``(..., L, S)``, True = allowed
  edge (hard-mask permitted, non-self-loop),
* ``slack``  : current integer slack ``c_t`` (0 when annealing is off),
* returns a mask broadcastable to ``(..., L, S)``; exactly 0 wherever
  ``valid`` is False is NOT the rule's responsibility - the generic class
  multiplies the returned mask by the (detached) validity mask, so
  forbidden entries are exactly zero for every rule while any gradient the
  rule put on allowed entries survives.

Rows with fewer allowed candidates than the effective budget pass ALL
allowed entries (their allowed ranks are 0..count-1 < k_eff), so no
spurious blanking occurs.

Diagnostics (populated each forward, detached)
----------------------------------------------
* ``last_mask``      - the applied selection mask (batch-mean, ``(L, S)``).
* ``last_gates_in``  - the gate tensor received (batch-mean, ``(L, S)``).
* ``last_margin``    - per-row base-budget margin ``g_(k) - g_(k+1)`` over
  allowed entries (batch-mean, ``(L,)``); NaN where undefined (fewer than
  k+1 allowed candidates).  Watch this for boundary oscillation / lock-in.
"""

from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

class TopKGate(nn.Module):
    """Per-row top-k blanking of gate posteriors with pluggable selection rules.

    Args:
        k: Per-row budget.  An int applies the same budget to every query
            row; a 1-D sequence/tensor of length L gives each target node
            its own budget (e.g. the dataset's true parent counts).
        method: Registered selection-rule name (default ``"noisy_hard_k"``).
        slack_init: Initial max extra slots ``c_0`` for stochastic rules.
            ``None`` resolves to ``S - max(k)`` (no constraint at init) when
            annealing is configured, else 0 (pure hard top-k).
        slack_final: Final slack after annealing (default 0 = exact top-k).
        annealing_batches: Forwards over which the slack anneals linearly
            from ``slack_init`` to ``slack_final``.  ``None`` = static slack.
        per_row_slack: Sample the slack noise independently per query row
            (default) or once per forward for the whole batch.
        exclude_diagonal: Forbid self-loops (square self blocks, L == S).
            Forbidden entries never enter the ranking and are exactly zero.
    """

    _METHODS: Dict[str, Callable] = {}

    def __init__(
        self,
        k: Union[int, Sequence[int], torch.Tensor],
        method: str = "noisy_hard_k",
        slack_init: Optional[int] = None,
        slack_final: int = 0,
        annealing_batches: Optional[int] = None,
        per_row_slack: bool = True,
        exclude_diagonal: bool = False,
    ):
        super().__init__()

        if method not in self._METHODS:
            raise ValueError(
                f"Unknown TopKGate method {method!r}; "
                f"registered: {sorted(self._METHODS)}."
            )
        self.method = method
        self._method_fn = self._METHODS[method]
        self.mask_grad: bool = bool(getattr(self._method_fn, "mask_grad", False))

        # Per-row budget.  Sequences are stored as a non-persistent buffer so
        # checkpoints from TopKGate-enabled runs stay loadable without it.
        if isinstance(k, int):
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            self.k: Optional[int] = int(k)
            self.k_per_row = None
        else:
            k_t = torch.as_tensor(list(k), dtype=torch.long)
            if k_t.dim() != 1 or (k_t < 1).any():
                raise ValueError(
                    f"k must be a positive int or 1-D positive sequence, got {k!r}"
                )
            self.k = None
            self.register_buffer("k_per_row", k_t, persistent=False)

        self.slack_init = slack_init if slack_init is None else int(slack_init)
        self.slack_final = int(slack_final)
        self.annealing_batches = (
            int(annealing_batches)
            if annealing_batches is not None and int(annealing_batches) > 0
            else None
        )
        self.per_row_slack = bool(per_row_slack)
        self.exclude_diagonal = bool(exclude_diagonal)

        # Annealing clock (global run-level schedule, mirroring BKD): advances
        # on every TRAINING forward even when the phase switch is off.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._phase_active: bool = True

        # Diagnostics (detached, batch-mean).
        self.last_mask: Optional[torch.Tensor] = None
        self.last_gates_in: Optional[torch.Tensor] = None
        self.last_margin: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    @classmethod
    def register_method(cls, name: str, mask_grad: bool = False):
        """Decorator registering a selection rule.

        ``mask_grad`` documents whether the rule's mask carries gradient to
        the scores (False = detached hard mask, e.g. ``noisy_hard_k``).
        """
        def deco(fn: Callable) -> Callable:
            fn.mask_grad = bool(mask_grad)
            cls._METHODS[name] = fn
            return fn
        return deco

    # ------------------------------------------------------------------
    # Phase switch / annealing (mirrors the BKD pattern)
    # ------------------------------------------------------------------
    def set_phase_active(self, active: bool) -> None:
        """Enable/disable blanking for the current training phase.

        Inactive phases pass gates through unchanged and clear diagnostics,
        but the annealing clock keeps advancing (global run-level schedule).
        """
        self._phase_active = bool(active)

    def _current_slack(self, s_dim: int, k_max: int) -> int:
        """Current integer slack c_t (0 when the rule should be exact)."""
        if self.annealing_batches is None:
            return int(self.slack_init or 0)
        base = self.slack_init if self.slack_init is not None else max(0, s_dim - k_max)
        progress = min(1.0, float(self._step.item()) / float(self.annealing_batches))
        return int(round(base + progress * (self.slack_final - base)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _k_tensor(self, l_dim: int, device: torch.device) -> torch.Tensor:
        """Per-row base budget as an ``(L,)`` long tensor."""
        if self.k_per_row is not None:
            if self.k_per_row.numel() != l_dim:
                raise ValueError(
                    f"Per-row k has length {self.k_per_row.numel()}, "
                    f"expected {l_dim} (one budget per query row)."
                )
            return self.k_per_row.to(device)
        return torch.full((l_dim,), int(self.k), dtype=torch.long, device=device)


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        gates: torch.Tensor,
        hard_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blank all but the top-k gate entries of every query row.

        Args:
            gates: Gate posterior / applied-weight tensor ``(..., L, S)``.
            hard_mask: Optional allowed-edge mask, ``(L, S)`` or
                ``(..., L, S)``; entries == 0 are forbidden: excluded from
                the ranking and exactly zero in the output.

        Returns:
            ``gates * mask``, same shape; ``mask`` comes from the selected
            rule (detached for hard rules) times the detached validity mask.
        """
        l_dim, s_dim = gates.shape[-2], gates.shape[-1]

        if self.exclude_diagonal and l_dim != s_dim:
            raise ValueError(
                f"exclude_diagonal=True requires a square block, "
                f"got (L={l_dim}, S={s_dim})."
            )

        # ---- Validity mask (detached structural constraint) -------------
        valid = torch.ones(l_dim, s_dim, dtype=torch.bool, device=gates.device)
        if hard_mask is not None:
            hm = hard_mask if hard_mask.dim() == gates.dim() else hard_mask.unsqueeze(0)
            valid = valid.unsqueeze(0) & (hm != 0)
        if self.exclude_diagonal:
            diag = torch.eye(l_dim, dtype=torch.bool, device=gates.device)
            valid = valid & ~diag

        # ---- Phase switch: passthrough, but the clock advances ----------
        if self.training:
            self._step += 1
        if not self._phase_active:
            self.last_mask = None
            self.last_gates_in = None
            self.last_margin = None
            return gates

        # ---- Ranking scores: forbidden entries never compete ------------
        scores = gates.masked_fill(~valid, float("-inf"))

        k_row = self._k_tensor(l_dim, gates.device)
        slack = self._current_slack(s_dim, int(k_row.max()))

        # ---- Rule-specific selection mask --------------------------------
        mask = self._method_fn(self, scores, k_row, valid, slack)
        # Generic post-processing: forbidden entries are EXACTLY zero for
        # every rule; `valid` is detached so only the rule's own gradient
        # (if any) on allowed entries survives.
        mask = mask * valid.to(mask.dtype)

        # ---- Diagnostics (detached) --------------------------------------
        with torch.no_grad():
            g_in = gates.detach()
            m_det = mask.detach()
            self.last_gates_in = g_in.mean(dim=0) if g_in.dim() == 3 else g_in
            self.last_mask = m_det.mean(dim=0) if m_det.dim() == 3 else m_det

            # Base-budget margin g_(k) - g_(k+1) over allowed entries.
            svals = scores.detach().sort(dim=-1, descending=True).values
            kk = k_row.clamp(min=1)
            idx_hi = (kk - 1).expand(svals.shape[:-1]).unsqueeze(-1)
            g_hi = svals.gather(-1, idx_hi).squeeze(-1)
            if s_dim > 1:
                idx_lo = kk.expand(svals.shape[:-1]).unsqueeze(-1)
                g_lo = svals.gather(-1, idx_lo).squeeze(-1)
                margin = g_hi - g_lo
            else:
                margin = torch.full_like(g_hi, float("nan"))
            margin = torch.where(
                torch.isfinite(margin), margin, torch.full_like(margin, float("nan"))
            )
            self.last_margin = margin.nanmean(dim=0) if margin.dim() == 2 else margin

        return gates * mask

    def extra_repr(self) -> str:
        k_repr = self.k if self.k is not None else f"per-row[{self.k_per_row.numel()}]"
        return (
            f"k={k_repr}, method={self.method!r}, mask_grad={self.mask_grad}, "
            f"slack_init={self.slack_init}, slack_final={self.slack_final}, "
            f"annealing_batches={self.annealing_batches}, "
            f"per_row_slack={self.per_row_slack}, "
            f"exclude_diagonal={self.exclude_diagonal}"
        )


# ----------------------------------------------------------------------
# Selection rules
# ----------------------------------------------------------------------
@TopKGate.register_method("noisy_hard_k", mask_grad=False)
def _noisy_hard_k(
    module: TopKGate,
    scores: torch.Tensor,
    k_row: torch.Tensor,
    valid: torch.Tensor,
    slack: int,
) -> torch.Tensor:
    """Hard top-k with annealed uniform budget slack (detached mask).

    Training: per-row budget ``k + eps``, ``eps ~ Uniform{0, ..., c_t}``
    (sampled per row when ``module.per_row_slack``, else once per forward).
    Eval: exact top-k (eps = 0).  The mask is DETACHED: excluded gates get
    no gradient through the value path; selection quality is trained
    through the posterior-level losses only.

    The budget is effectively clamped to the allowed-set size by the rank
    comparison: allowed entries occupy ranks 0..count-1, so rows with fewer
    allowed candidates than k_eff pass everything allowed.
    """
    k_eff = k_row
    if module.training and slack > 0:
        if module.per_row_slack:
            eps = torch.randint(0, slack + 1, k_row.shape, device=scores.device)
        else:
            eps = torch.randint(0, slack + 1, (1,), device=scores.device).expand_as(k_row)
        k_eff = k_row + eps

    # Rank of every entry within its row (forbidden = -inf sort last).
    # Ties are broken arbitrarily but deterministically by argsort.
    ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    mask = (ranks < k_eff.unsqueeze(-1)) & valid
    return mask.to(scores.dtype).detach()

