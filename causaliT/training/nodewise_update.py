"""
Node-wise (per-query) winner-take-all structural update.

Motivation
----------
With gradient routing, the structural stream updates the query embeddings of
ALL nodes at every step.  Empirically (see
experiments/6_INVESTIGATIONS/LARGER_DAGS/NT_safeguard_dropoutsel_11560942/
investigate_query_drift.ipynb) the per-node structural gradient is extremely
uneven (15x magnitude spread) and, for most nodes, dominated by batch and
Hard-Concrete gate noise: the mean direction carries the signal, the per-step
realisation is mostly angular diffusion that walks the unit-normalised queries
off their target.

This module implements a nodewise SNR gate: per structural step, only the
``topk`` query nodes whose gradient shows the strongest *evidence of being
non-zero* are updated; every other query row (and its optimizer state) is
reverted after the optimizer step.

Two selection rules (``selection`` constructor arg):

* ``snr`` (default): temporal EMA t-statistic below - consistent direction
  over ~20 steps.  Couples past updates with the current one: a consistent
  but uninformative (e.g. radial at a symmetric init) gradient passes.
* ``norm``: per-step greedy - the topk rows with the largest CURRENT
  gradient norm, no memory, no gate.  With an SGD structural optimizer this
  is the first-order greedy ranking of the expected loss decrease
  (s_i = lr * ||g_i||^2).

Statistic for ``snr`` (hard-coded, Option E - streaming EMA, optimizer-agnostic)
--------------------------------------------------------------------
Per node i with gradient vector g_i (d = embedding dim):

    m_i <- beta * m_i + (1 - beta) * g_i          (EMA of the mean)
    s_i <- beta * s_i + (1 - beta) * ||g_i||^2    (EMA of the second moment)

with Adam-style bias correction (bc = 1 - beta^t).  Since
E||g||^2 = ||mu||^2 + d*sigma^2 under a Gaussian noise model, the per-node
signal-to-noise ratio is

    SNR_i = ||m_hat_i||^2 / ((s_hat_i - ||m_hat_i||^2) / d + EPS)

i.e. a (squared) t-statistic for the hypothesis "the mean gradient of node i
is non-zero": "given the data, which query has the strongest evidence to
move?".  If even the best SNR is below the noise floor MIN_SNR = 1.0, NO node
is updated this step (the gate fires) - noise alone never moves a query.

Optimizer compatibility
-----------------------
Gradient masking alone is NOT enough: decoupled weight decay (AdamW) and
momentum act on parameters whose grad is zero (only ``grad is None`` is
skipped, and the grad tensor is a whole table, not per-row).  The selector
therefore snapshots the non-selected ROWS of every structural parameter and of
every same-shaped tensor in the optimizer state before ``opt.step()`` and
restores them after, which is generic across the optimizer suite
(AdamW/Adam/SGD-momentum/...).  Scalar state entries (e.g. Adam's ``step``)
are shared per tensor and cannot be reverted per row; the resulting bias
correction drift for rows that are selected only intermittently is negligible.

Selected node's ``query_norm_log_scale`` row (M_i, the per-node norm budget)
shares the update: M_i is that node's budget and moves with its query.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Hard-coded strategy constants (see module docstring).
BETA = 0.95          # EMA window ~ 1/(1-beta) = 20 structural steps
MIN_SNR = 1.0        # t-statistic noise floor: below it, no node updates
EPS = 1e-12


@dataclass
class NodewiseSnapshot:
    """Rows to restore after the optimizer step (params + optimizer state)."""

    param_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=list
    )  # (param, row_idx, saved_values)
    state_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=list
    )  # (state_tensor, row_idx, saved_values)



class NodewiseQuerySelector:
    """Winner-take-all per-node gate on the structural query gradient.

    Parameters
    ----------
    query_params :
        List of query embedding weight tensors, each ``(1 + n_rows, d)`` with
        the padding row at index 0 (e.g. ``query_embed_S.embedding.weight``).
        Nodes are indexed consecutively across the tables, in order.
    norm_param :
        Optional per-node norm-budget parameter ``query_norm_log_scale`` of
        shape ``(n_nodes,)`` (tied across blocks -> pass it once).  Row i is
        updated iff node i is selected.
    topk :
        Number of nodes updated per structural step.
    """

    def __init__(
        self,
        query_params: List[torch.Tensor],
        norm_param: Optional[torch.Tensor],
        topk: int = 1,
        selection: str = "snr",
    ):
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if selection not in ("snr", "norm"):
            raise ValueError(
                f"selection must be 'snr' or 'norm', got {selection!r}"
            )
        self.topk = int(topk)
        self.selection = selection

        # Node map: node i -> (param, row).  Row 0 of each table is padding.
        self.node_map: List[Tuple[torch.Tensor, int]] = []
        for p in query_params:
            for r in range(1, p.shape[0]):
                self.node_map.append((p, r))
        self.n_nodes = len(self.node_map)
        self.dim = query_params[0].shape[1]
        self.norm_param = norm_param
        if norm_param is not None and norm_param.shape[0] != self.n_nodes:
            raise ValueError(
                f"query_norm_log_scale has {norm_param.shape[0]} rows but the "
                f"query tables expose {self.n_nodes} nodes."
            )
        if self.topk > self.n_nodes:
            raise ValueError(
                f"topk={self.topk} exceeds the number of nodes ({self.n_nodes})."
            )

        dev = query_params[0].device
        self.ema_mean = torch.zeros(self.n_nodes, self.dim, device=dev)
        self.ema_sq = torch.zeros(self.n_nodes, device=dev)
        self.t = 0
        # Diagnostics (reset per epoch by the forecaster).
        self.sel_counts = torch.zeros(self.n_nodes, dtype=torch.long)
        self.n_gate_fired = 0
        self.n_steps = 0
        self.last_max_stat = 0.0   # max SNR ('snr') or max ||g_i|| ('norm')

    # ------------------------------------------------------------------
    def reset_stats(self):
        """Clear the SNR evidence (called at every phase switch)."""
        self.ema_mean.zero_()
        self.ema_sq.zero_()
        self.t = 0

    def reset_epoch_diagnostics(self):
        self.sel_counts.zero_()
        self.n_gate_fired = 0
        self.n_steps = 0
    # ------------------------------------------------------------------
    def _gather_grads(self) -> Optional[torch.Tensor]:
        """Per-node gradient matrix (n_nodes, d); None if nothing to do."""
        grads = torch.zeros(self.n_nodes, self.dim,
                            device=self.ema_mean.device)
        any_grad = False
        for i, (p, r) in enumerate(self.node_map):
            if p.grad is not None:
                grads[i] = p.grad[r].detach().to(grads.device)
                any_grad = True
        return grads if any_grad else None

    def current_snr(self) -> torch.Tensor:
        """Bias-corrected per-node SNR (n_nodes,)."""
        bc = 1.0 - BETA ** max(self.t, 1)
        m = self.ema_mean / bc
        s = self.ema_sq / bc
        var_per_coord = (s - m.pow(2).sum(dim=1)).clamp_min(0.0) / self.dim
        return m.pow(2).sum(dim=1) / (var_per_coord + EPS)

    # ------------------------------------------------------------------
    def select(self) -> Optional[List[int]]:
        """Update the EMA with the current grads and pick the winners.

        Returns None when the structural params received no gradient this
        step (e.g. frozen during a reconstruct phase): the caller skips the
        masking/snapshot and the diagnostics.  [] means the SNR gate fired.
        """
        grads = self._gather_grads()
        if grads is None:
            return None
        self.t += 1
        self.n_steps += 1

        if self.selection == "norm":
            # Per-step greedy: the topk rows with the largest CURRENT gradient
            # norm.  No temporal memory (no EMA, no SNR, no gate): each update
            # is confined to the current training step.  With an SGD
            # structural optimizer this is also the first-order greedy ranking
            # of the expected loss decrease, s_i = lr * ||g_i||^2.
            norms = grads.pow(2).sum(dim=1).sqrt()
            self.last_max_stat = float(norms.max())
            selected = norms.topk(min(self.topk, self.n_nodes)).indices.tolist()
            for i in selected:
                self.sel_counts[i] += 1
            return selected

        self.ema_mean.mul_(BETA).add_(grads, alpha=1.0 - BETA)
        self.ema_sq.mul_(BETA).add_(grads.pow(2).sum(dim=1), alpha=1.0 - BETA)

        snr = self.current_snr()
        best = int(snr.argmax())
        self.last_max_stat = float(snr[best])
        if float(snr[best]) < MIN_SNR:
            self.n_gate_fired += 1
            return []
        k = min(self.topk, int((snr >= MIN_SNR).sum()))
        selected = snr.topk(k).indices.tolist()
        for i in selected:
            self.sel_counts[i] += 1
        return selected

    # ------------------------------------------------------------------
    def mask_grads(self, selected: List[int]):
        """Zero the gradient rows of every non-selected node."""
        sel = set(selected)
        done = set()
        for i, (p, r) in enumerate(self.node_map):
            if p.grad is not None and id(p) not in done:
                p.grad[0].zero_()   # padding row is never a node
                done.add(id(p))
            if i not in sel and p.grad is not None:
                p.grad[r].zero_()
        if self.norm_param is not None and self.norm_param.grad is not None:
            keep = torch.zeros_like(self.norm_param.grad)
            for i in selected:
                keep[i] = self.norm_param.grad[i]
            self.norm_param.grad.copy_(keep)

    # ------------------------------------------------------------------
    def snapshot(self, optimizer: torch.optim.Optimizer,
                 selected: List[int]) -> NodewiseSnapshot:
        """Save non-selected rows of params + same-shaped optimizer state."""
        snap = NodewiseSnapshot()
        sel = set(selected)
        params = [p for p, _ in self.node_map] + (
            [self.norm_param] if self.norm_param is not None else []
        )
        for p in dict.fromkeys(params):  # dedupe tied tensors
            if p is self.norm_param:
                rows = torch.tensor(
                    [i for i in range(self.n_nodes) if i not in sel],
                    device=p.device,
                )
            else:
                # Row 0 is the padding row: not a node, but decoupled weight
                # decay would still shrink it -> always restore it.
                rows = torch.tensor(
                    [0] + [r for i, (q, r) in enumerate(self.node_map)
                           if q is p and i not in sel],
                    device=p.device,
                )
            if rows.numel() == 0:
                continue
            snap.param_rows.append((p, rows, p.data[rows].clone()))
            for st in optimizer.state.get(p, {}).values():
                if torch.is_tensor(st) and st.shape == p.shape:
                    snap.state_rows.append((st, rows, st[rows].clone()))
        return snap

    @staticmethod
    def restore(snap: NodewiseSnapshot):
        """Undo the optimizer step on the non-selected rows."""
        for p, rows, saved in snap.param_rows:
            p.data[rows] = saved
        for st, rows, saved in snap.state_rows:
            st[rows] = saved

