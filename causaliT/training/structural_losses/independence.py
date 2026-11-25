"""
Residual-independence structural objectives (pure tensor functions).

Objectives (``training``):
    hsic_objective: pairwise        per-pair HSIC(x_j, r_i), aggregated by
                                    ``hsic_aggregation`` (kernels and
                                    aggregations: ``utils.hsic_utils``)
    hsic_objective: dhsic_residual  NOTIME: mutual independence of residuals
    anti_adjacency_lambda > 0       order log-ratio (:func:`anti_adjacency_loss`,
                                    or :func:`joint_nd_loss` with
                                    ``anti_adjacency_test: joint``)

Shapes: target / residuals ``(B, N)``; order mask ``M[i, j] = P(i before j)``
and selector weights ``pi`` are ``(N, N)``.

Coding Assistant:
    * Nothing here touches the model: the forecaster runs the forwards and
      passes tensors in, so every loss is unit-testable on plain tensors.
    * Loss functions return ``(loss, stats, paths)``: ``stats`` is detached
      logging data, ``paths`` the extra losses for order-path surgery
      (empty unless ``surgery=True``).
    * Log-ratio derivation: docs/documentation/ANTI_ADJACENCY_LOSS.md.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from causaliT.utils.hsic_utils import (
    dhsic_residual_independence,
    hsic_cross_per_pair,
    hsic_joint_masked,
    hsic_permutation_null_matrix,
)

HSIC_OBJECTIVES = ("pairwise", "dhsic_residual")


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class KernelOpts:
    """HSIC kernel settings of one step (bandwidths may be phase-frozen)."""

    sigma: Any = 1.0
    adaptive: bool = False
    mode: str = "biased"
    nhsic_epsilon: float = 0.01
    source_kernel: str = "rbf"
    bandwidth_multipliers: Any = None

    def pair_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments of ``hsic_cross_per_pair`` (matrix returned)."""
        return dict(sigma=self.sigma, adaptive_bandwidth=self.adaptive, mode=self.mode,
                    nhsic_epsilon=self.nhsic_epsilon, source_kernel=self.source_kernel,
                    bandwidth_multipliers=self.bandwidth_multipliers, return_matrix=True)

    def null_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments of ``hsic_permutation_null_matrix``."""
        return dict(sigma=self.sigma, adaptive_bandwidth=self.adaptive,
                    source_kernel=self.source_kernel,
                    bandwidth_multipliers=self.bandwidth_multipliers)


def resolve_hsic_objective(tr: dict) -> str:
    """Validated ``training.hsic_objective`` (pairwise | dhsic_residual)."""
    obj = str(tr.get("hsic_objective", "pairwise"))
    if obj not in HSIC_OBJECTIVES:
        raise ValueError("training.hsic_objective must be 'pairwise' or "
                         f"'dhsic_residual', got {obj!r}.")
    return obj


@dataclass(frozen=True)
class AntiAdjacencyConfig:
    """
    Order log-ratio settings (``training.anti_adjacency_*``).

    lam            weight a of the anti-causal term; 0 disables the objective
    eps            log offset (> 0)
    mode           causal_only (A from causal residuals) | legacy_anticausal
    terms          both | causal (A only logged)
    support        order (all pairs) | graph (closure of the hard edge draw)
    nd             soft (relaxed M) | hard (hard order, straight-through)
    test           pairwise | joint (RESIT: one test per node)
    null           global (scalar nu) | per_pair (closed-form null per pair)
    ratio_floor    clamp C/nu >= 1;  ratio_cap: clamp A/nu <= cap
    detach_a/_c    drop the residual path of A / C
    selector_grad  raw | none (G_S detached in the anti-causal forward)
    detach_order_mask  ablation: M carries no gradient
    transform      log_ratio (log(eps + ratio)) | linear (ratio = HSIC / nu,
                   no log; eps unused)
    """

    lam: float = 0.0
    eps: float = 0.1
    mode: str = "causal_only"
    terms: str = "both"
    support: str = "order"
    nd: str = "soft"
    test: str = "pairwise"
    null: str = "global"
    ratio_floor: bool = False
    ratio_cap: Optional[float] = None
    detach_a: bool = False
    detach_c: bool = False
    selector_grad: str = "raw"
    detach_order_mask: bool = False
    transform: str = "log_ratio"

    @property
    def on(self) -> bool:
        """True when the log-ratio replaces the pairwise HSIC."""
        return self.lam > 0.0

    @classmethod
    def from_config(cls, tr: dict) -> "AntiAdjacencyConfig":
        """Parse and validate the ``training`` section."""
        cap = tr.get("anti_adjacency_ratio_cap", None)
        cfg = cls(
            lam=float(tr.get("anti_adjacency_lambda", 0.0)),
            eps=float(tr.get("anti_adjacency_eps", 0.1)),
            mode=str(tr.get("anti_adjacency_mode", "causal_only")),
            terms=str(tr.get("anti_adjacency_terms", "both")),
            support=str(tr.get("anti_adjacency_support", "order")),
            nd=str(tr.get("anti_adjacency_nd", "soft")),
            test=str(tr.get("anti_adjacency_test", "pairwise")),
            null=str(tr.get("anti_adjacency_null", "global")),
            ratio_floor=bool(tr.get("anti_adjacency_ratio_floor", False)),
            ratio_cap=None if cap is None else float(cap),
            detach_a=bool(tr.get("anti_adjacency_detach_a_residuals", False)),
            detach_c=bool(tr.get("anti_adjacency_detach_c_residuals", False)),
            selector_grad=str(tr.get("anticausal_selector_grad", "raw")),
            detach_order_mask=bool(tr.get("detach_order_mask", False)),
            transform=str(tr.get("anti_adjacency_transform", "log_ratio")),
        )
        cfg.validate(str(tr.get("hsic_mode", "biased")))
        return cfg

    def validate(self, hsic_mode: str = "biased") -> None:
        """Raise ``ValueError`` on invalid or incompatible settings."""
        if self.lam < 0.0:
            raise ValueError(f"anti_adjacency_lambda must be >= 0, got {self.lam}.")
        choices = {
            "anti_adjacency_mode": (self.mode, ("causal_only", "legacy_anticausal")),
            "anti_adjacency_terms": (self.terms, ("both", "causal")),
            "anti_adjacency_support": (self.support, ("order", "graph")),
            "anti_adjacency_nd": (self.nd, ("soft", "hard")),
            "anti_adjacency_test": (self.test, ("pairwise", "joint")),
            "anti_adjacency_null": (self.null, ("global", "per_pair")),
            "anticausal_selector_grad": (self.selector_grad, ("raw", "none")),
            "anti_adjacency_transform": (self.transform, ("log_ratio", "linear")),
        }
        for key, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {value!r}.")
        if self.ratio_cap is not None and self.ratio_cap <= 0:
            raise ValueError("anti_adjacency_ratio_cap must be > 0 or null.")
        if self.test == "joint" and (self.nd != "hard" or self.terms != "causal"
                                     or self.mode != "causal_only" or hsic_mode != "biased"):
            raise ValueError("anti_adjacency_test=joint requires anti_adjacency_nd=hard, "
                             "terms=causal, mode=causal_only and hsic_mode=biased.")
        if self.null == "per_pair" and hsic_mode != "biased":
            raise ValueError("anti_adjacency_null=per_pair requires hsic_mode: biased.")



# =============================================================================
# dHSIC residual objective (NOTIME)
# =============================================================================

def dhsic_residual_loss(residuals: torch.Tensor, sigma: Any = None) -> torch.Tensor:
    """
    d-variable dHSIC of all residual columns (Berrevoets et al., 2025).

    ``sigma``: per-column tensor, scalar, or None (median heuristic per
    column).  No pair structure, so masks and row diagnostics do not apply.
    """
    return dhsic_residual_independence(residuals, sigma=sigma)


# =============================================================================
# Order log-ratio (anti-adjacency)
# =============================================================================

def _transform(ratio: torch.Tensor, eps: float, transform: str) -> torch.Tensor:
    """Per-term loss: ``log(eps + ratio)`` (log_ratio) or ``ratio`` (linear)."""
    if transform == "linear":
        return ratio
    return torch.log(eps + ratio)


def _pair_weights(pi: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
    """Off-diagonal pair weights ``pi``, zero for pairs touching a dropped node."""
    N = pi.shape[0]
    w = pi * (1.0 - torch.eye(N, device=pi.device, dtype=pi.dtype))
    if keep is not None:
        k = keep.detach().to(device=w.device, dtype=w.dtype)
        if k.dim() == 2:                          # per (child, parent) keep
            w = w * k * k.transpose(0, 1)
        else:
            k = k.view(-1)
            w = w * k.view(-1, 1) * k.view(1, -1)
    return w


def _anti_adjacency_stats(C, A, c_term, a_term, w_c, w_a, triu, c_loss, a_loss,
                          nu, nu_a, U, has_cycle, floor, cap) -> Dict[str, Any]:
    """Detached logging stats of :func:`anti_adjacency_loss`."""
    with torch.no_grad():
        wc_sum = w_c[triu].sum().clamp_min(1e-12)
        wa_sum = w_a[triu].sum().clamp_min(1e-12)
        rows = ((w_c * c_term).sum(dim=1) / w_c.sum(dim=1).clamp_min(1e-12)
                + (w_a * a_term).sum(dim=1) / w_a.sum(dim=1).clamp_min(1e-12))
        live_a = w_a[triu] > 0
        return {
            "C": float((w_c[triu] * C[triu]).sum() / wc_sum),
            "A": float((w_a[triu] * A[triu]).sum() / wa_sum),
            "causal_log_term": float(c_loss),
            "anticausal_log_term": float(a_loss),
            "nu": nu,
            "nu_a": nu if nu_a is None else float(nu_a.mean()),
            "c_pairs": float((w_c[triu] > 0).sum()),
            "a_pairs": float(live_a.sum()),
            "unconnected_pairs": 0.0 if U is None else float(U[triu].sum()),
            "cycle": float(has_cycle),
            "c_floor_frac": float((C[triu] <= 1.0).float().mean()) if floor else 0.0,
            "a_cap_frac": (float((A[triu][live_a] >= cap).float().mean())
                           if cap is not None and bool(live_a.any()) else 0.0),
            "rows": rows,
        }



def anti_adjacency_loss(
    target: torch.Tensor,
    r_c: torch.Tensor,
    M: torch.Tensor,
    pi: torch.Tensor,
    cfg: "AntiAdjacencyConfig",
    kernel: KernelOpts,
    r_a: Optional[torch.Tensor] = None,
    nu_global: float = 1.0,
    keep: Optional[torch.Tensor] = None,
    graph: Optional[Tuple[torch.Tensor, torch.Tensor, bool]] = None,
    surgery: bool = False,
    anticausal_surgery: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Causal / anti-causal log-ratio on the order (lower = better order).

    For each unordered pair e = (i, j), with H[t, s] = HSIC(x_s, r_t):
        C_e = M_ij H[j, i] + M_ji H[i, j]   later residual vs earlier node
        A_e = M_ij H[i, j] + M_ji H[j, i]   earlier residual vs later node
        L   = mean_e w log(eps + C_e/nu) - a mean_e w_a log(eps + A_e/nu)
    transform=linear drops the log: L = mean_e w C_e/nu - a mean_e w_a A_e/nu.

    Args:
        target, r_c: node values and causal residuals ``(B, N)``.
        M: relaxed order (gradient-carrying; hard/ST-mixed by the caller).
        pi: detached pair weights (selector probability; ones for nd=hard).
        r_a: anti-causal residuals (legacy_anticausal); None reuses ``r_c``.
        nu_global: permutation-null mean, used when ``cfg.null == global``.
        keep: BKD keep vector ``(N,)`` / matrix; a pair needs both kept.
        graph: ``(U, sel, has_cycle)`` from ``graph_supports`` (support=graph).
        surgery, anticausal_surgery: also return the path-split losses.

    Returns ``(loss, stats, paths)``; ``loss`` is exactly 0 without support.
    """
    a, eps = cfg.lam, cfg.eps
    kw = kernel.pair_kwargs()
    H_c = torch.nan_to_num(hsic_cross_per_pair(target, r_c, **kw)[1])
    H_c_C = H_c.detach() if cfg.detach_c else H_c
    if r_a is None:
        r_a = r_c
        H_a = H_c.detach() if cfg.detach_a else H_c
    else:
        if cfg.detach_a:
            r_a = r_a.detach()
        H_a = torch.nan_to_num(hsic_cross_per_pair(target, r_a, **kw)[1])

    if cfg.null == "per_pair":
        nkw = kernel.null_kwargs()
        nu_c = hsic_permutation_null_matrix(target, r_c, **nkw).clamp_min(1e-12)
        nu_a = (nu_c if r_a is r_c else
                hsic_permutation_null_matrix(target, r_a, **nkw).clamp_min(1e-12))
        nu = float(nu_c.mean())
    else:
        nu, nu_c, nu_a = nu_global, None, None

    U, sel, has_cycle = graph if graph is not None else (None, None, False)
    floor, cap = cfg.ratio_floor, cfg.ratio_cap

    def pair_ratios(Mw, Hc, Ha):
        """Symmetric (C, A) pair ratios; each direction divided by its null."""
        rc = Hc.transpose(0, 1) / (nu if nu_c is None else nu_c.transpose(0, 1))
        ra = Ha / (nu if nu_a is None else nu_a)
        Cd, Ad = Mw * rc, Mw * ra
        C_ = (Cd + Cd.transpose(0, 1)).clamp_min(0.0)
        A_ = (Ad + Ad.transpose(0, 1)).clamp_min(0.0)
        if U is not None:                         # unconnected: both C tests hold
            C_ = (1.0 - U) * C_ + U * (rc + rc.transpose(0, 1)).clamp_min(0.0)
        if floor:
            C_ = C_.clamp_min(1.0)
        if cap is not None:
            A_ = A_.clamp_max(cap)
        return C_, A_

    C, A = pair_ratios(M, H_c_C, H_a)
    tf = cfg.transform
    c_term = _transform(C, eps, tf)
    a_term = (torch.zeros_like(c_term) if cfg.terms == "causal"
              else -a * _transform(A, eps, tf))

    w_c = _pair_weights(pi, keep)
    w_a = w_c if sel is None else w_c * sel
    N = M.shape[0]
    off = 1.0 - torch.eye(N, device=M.device, dtype=M.dtype)
    triu = torch.triu(off, diagonal=1).bool()
    wc_sum = w_c[triu].sum().clamp_min(1e-12)
    wa_sum = w_a[triu].sum().clamp_min(1e-12)
    c_loss = (w_c[triu] * c_term[triu]).sum() / wc_sum
    a_loss = (w_a[triu] * a_term[triu]).sum() / wa_sum
    loss = c_loss + a_loss

    paths: Dict[str, torch.Tensor] = {}
    if surgery and torch.is_grad_enabled():
        def c_path(Mw, Hc):
            r = pair_ratios(Mw, Hc, H_a.detach())[0][triu]
            return (w_c[triu] * _transform(r, eps, tf)).sum() / wc_sum

        def a_path(Mw, Ha):
            r = pair_ratios(Mw, H_c.detach(), Ha)[1][triu]
            return (w_a[triu] * _transform(r, eps, tf)).sum() / wa_sum

        paths = {"c_path": loss, "c_weight": c_path(M, H_c.detach()),
                 "c_residual": c_path(M.detach(), H_c)}
        if anticausal_surgery:
            paths["a_weight"] = -a * a_path(M, H_a.detach())
            paths["a_residual"] = -a * a_path(M.detach(), H_a)

    stats = _anti_adjacency_stats(C, A, c_term, a_term, w_c, w_a, triu, c_loss,
                                  a_loss, nu, nu_a, U, has_cycle, floor, cap)
    return loss, stats, paths



def joint_nd_loss(
    target: torch.Tensor,
    r_c: torch.Tensor,
    M: torch.Tensor,
    cfg: "AntiAdjacencyConfig",
    sigma: Any,
    adaptive: bool,
    keep: Optional[torch.Tensor] = None,
    surgery: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Joint RESIT objective: one test per node against ALL its non-descendants.

        L = mean_i log(eps + T_i / nu_i),   over nodes with |ND(i)| >= 1
        (transform=linear: mean_i T_i / nu_i)

    T_i = joint HSIC of r_i and X_ND(i) (product-RBF set kernel), nu_i its
    exact permutation-null mean (detached).  ND(i) from ``M`` (use the hard
    order with straight-through gradient).  Returns ``(loss, stats, paths)``.
    """
    eps = cfg.eps
    N = M.shape[0]
    off = 1.0 - torch.eye(N, device=M.device, dtype=M.dtype)
    W = M.transpose(0, 1) * off                   # [t, s]: s earlier than t
    if keep is not None:
        k = keep.detach().to(W)
        W = W * (k if k.dim() == 2 else k.view(1, -1) * k.view(-1, 1))

    def node_loss(Wm, r):
        T, nu, count = hsic_joint_masked(target, r, Wm, sigma=sigma,
                                         adaptive_bandwidth=adaptive)
        ratio = T / nu.clamp_min(1e-12)
        if cfg.ratio_floor:
            ratio = ratio.clamp_min(1.0)
        valid = (count > 0.5).to(ratio.dtype)
        term = _transform(ratio.clamp_min(0.0), eps, cfg.transform)
        loss = ((valid * term).sum()
                / valid.sum().clamp_min(1.0))
        return loss, ratio, valid

    loss, ratio, valid = node_loss(W, r_c)
    paths: Dict[str, torch.Tensor] = {}
    if surgery and torch.is_grad_enabled():
        paths = {"c_path": loss,
                 "c_weight": node_loss(W, r_c.detach())[0],
                 "c_residual": node_loss(W.detach(), r_c)[0]}
    with torch.no_grad():
        rows = torch.where(valid > 0, _transform(ratio.clamp_min(0.0), eps, cfg.transform),
                           torch.zeros_like(ratio))
        stats = {
            "C": float((valid * ratio).sum() / valid.sum().clamp_min(1.0)), "A": 0.0,
            "causal_log_term": float(loss), "anticausal_log_term": 0.0,
            "nu": 1.0, "nu_a": 1.0,
            "c_pairs": float(valid.sum()), "a_pairs": 0.0,
            "unconnected_pairs": 0.0, "cycle": 0.0,
            "c_floor_frac": 0.0, "a_cap_frac": 0.0, "rows": rows,
        }
    return loss, stats, paths

