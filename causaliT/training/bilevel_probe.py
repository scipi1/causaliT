"""Bilevel probe: paired reconstruction refit for commit gating (Phase 1).

Implements the Tier-2 acceptance test of
``docs/ideas/BILEVEL_CENTROID_COMMIT.md``: starting from the CURRENT
reconstruction parameters ``theta_R``, run ``k_inner`` refit steps of the
reconstruction loss on validation batches, once with the incumbent query
rows and once with the candidate rows written, then compare the
node-responsible HSIC rows::

    accept(node i)  <=>  H_i(candidate) < H_i(incumbent) - accept_margin

Design invariants
-----------------
* The live forecaster is NEVER mutated (both arms are deepcopies).
* Both arms refit from the same ``theta_R``, with the same optimizer
  configuration (inherited from the reconstruction arm of the training
  config by default), the same seeded batch order and the same torch RNG
  seed, so any stochastic gates see identical masks in both arms.
* HSIC bandwidths are latched ONCE from the incumbent pre-refit state
  (median heuristic) and reused for both arms and the final evaluation,
  so bandwidth jitter cannot enter the paired comparison.
* The HSIC rows are measured in eval mode on a held-out validation batch
  (the last cached batch is reserved for evaluation, not refit).
"""

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.optimizer_factory import (
    get_recon_optimizer_config,
    make_optimizer,
)
from causaliT.utils.hsic_utils import (
    _median_bandwidth,
    hsic_cross_per_pair,
    hsic_row_means,
)

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Outcome of one paired refit probe."""
    rows_incumbent: torch.Tensor   # (N,) per-node HSIC after refit, no writes
    rows_candidate: torch.Tensor   # (N,) after refit WITH the candidate writes
    mse_incumbent: torch.Tensor    # (N,) per-node MSE on the eval batch
    mse_candidate: torch.Tensor
    deltas: Dict[int, float]       # node -> row_i(cand) - row_i(inc)
    accepted: Dict[int, bool]      # node -> delta < -accept_margin
    mse_deltas: Dict[int, float]   # diagnostic only (constant-shift tripwire)


# -----------------------------------------------------------------------------
# Model access helpers
# -----------------------------------------------------------------------------

def query_tables_of(forecaster) -> List[nn.Module]:
    """Free query tables in node order ([S, X] homogeneous, [X] split).

    Raises if free query embeddings are disabled.  Mirrors the table order
    used by ``CentroidCommitController`` so node indices align with the HSIC
    pair-matrix rows (targets).
    """
    m = forecaster.model
    tables = [t for t in (getattr(m, "query_embed_S", None),
                          getattr(m, "query_embed_X", None))
              if t is not None]
    if not tables:
        raise ValueError(
            "paired_refit_probe requires free query embeddings "
            "(no query_embed_S / query_embed_X table found on the model)."
        )
    return tables


def node_locations(forecaster) -> List[Tuple[int, int]]:
    """node index -> (table_idx, row); row 0 of each table is padding."""
    locs = []
    for ti, t in enumerate(query_tables_of(forecaster)):
        locs.extend((ti, i + 1) for i in range(t.num_variables))
    return locs


def _forward_pred(model, S, X):
    return model.forward(data_source=S, data_intermediate=X)[0]


def _target_residuals_source(model, S, X, pred):
    """(combined_source, residuals) exactly as _step's HSIC term builds them."""
    x_val = X[:, :, model.val_idx]
    if model.homogeneous_nodes:
        x_val = torch.cat([S[:, :, model.val_idx], x_val], dim=1)
    x_target = torch.nan_to_num(x_val)
    residuals = x_target.squeeze() - pred.squeeze()
    combined = (x_target.squeeze() if model.homogeneous_nodes else
                torch.cat([S[:, :, model.val_idx], x_target.squeeze()], dim=1))
    return combined, residuals


def recon_mse(model, S, X) -> torch.Tensor:
    pred = _forward_pred(model, S, X)
    x_val = X[:, :, model.val_idx]
    if model.homogeneous_nodes:
        x_val = torch.cat([S[:, :, model.val_idx], x_val], dim=1)
    x_target = torch.nan_to_num(x_val)
    return torch.nn.functional.mse_loss(pred.squeeze(), x_target.squeeze())


def mse_rows_eval(model, S, X) -> torch.Tensor:
    """Per-node MSE on the eval batch (diagnostic; mask-free)."""
    model.eval()
    with torch.no_grad():
        pred = _forward_pred(model, S, X)
        x_val = X[:, :, model.val_idx]
        if model.homogeneous_nodes:
            x_val = torch.cat([S[:, :, model.val_idx], x_val], dim=1)
        x_target = torch.nan_to_num(x_val)
        se = (pred.squeeze() - x_target.squeeze()).pow(2)   # (B, N)
        return se.mean(dim=0)


def hsic_rows_eval(model, S, X, sigma, adaptive_bandwidth,
                   pair_mask=None) -> torch.Tensor:
    """Per-row (per-node) HSIC in eval mode with explicit bandwidths.

    ``pair_mask``: optional detached (n_targets, n_sources) structural pair
    weights (descendant x LOO, no BKD) shared by both probe arms.
    """
    model.eval()
    with torch.no_grad():
        pred = _forward_pred(model, S, X)
        combined, residuals = _target_residuals_source(model, S, X, pred)
        _, mat = hsic_cross_per_pair(
            combined, residuals,
            sigma=sigma,
            adaptive_bandwidth=adaptive_bandwidth,
            mode=model.hsic_mode,
            nhsic_epsilon=model.nhsic_epsilon,
            source_kernel=model.hsic_kernel_source,
            bandwidth_multipliers=getattr(model, "hsic_bandwidth_multipliers", None),
            pair_mask=pair_mask, return_matrix=True)
    return hsic_row_means(mat, pair_mask=pair_mask)



# -----------------------------------------------------------------------------
# Probe internals
# -----------------------------------------------------------------------------

def _latch_bandwidths(model, S, X) -> Tuple[torch.Tensor, torch.Tensor]:
    """Median-heuristic bandwidths of the CURRENT state (eval forward)."""
    model.eval()
    with torch.no_grad():
        pred = _forward_pred(model, S, X)
        combined, residuals = _target_residuals_source(model, S, X, pred)
        sig_src = torch.stack([_median_bandwidth(combined[:, i])
                               for i in range(combined.shape[1])])
        sig_res = torch.stack([_median_bandwidth(residuals[:, j])
                               for j in range(residuals.shape[1])])
    return sig_src, sig_res


def _detach_graph_tensors(value, stashed, setitem):
    """Recursively detach graph-carrying / non-leaf tensors in ``value``.

    ``setitem(new)`` replaces the value in its container (attr, dict key,
    list index).  Stashed entries are ``(setitem, original)`` for restore.
    """
    if torch.is_tensor(value):
        if value.grad_fn is not None or not value.is_leaf:
            stashed.append((setitem, value))
            setitem(value.detach())
    elif isinstance(value, dict):
        for k, v in value.items():
            _detach_graph_tensors(v, stashed, lambda nv, k=k: value.__setitem__(k, nv))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _detach_graph_tensors(v, stashed, lambda nv, i=i: value.__setitem__(i, nv))
    elif isinstance(value, tuple):
        if any(torch.is_tensor(v) and (v.grad_fn is not None or not v.is_leaf)
               for v in value):
            stashed.append((setitem, value))
            setitem(tuple(v.detach() if torch.is_tensor(v) else v
                          for v in value))


def _prepare_copy(forecaster):
    """Deepcopy + deterministic probe regime (BKD off, eval mode).

    ``deepcopy`` refuses non-leaf tensors, and training leaves graph-carrying
    tensors all over the live module: attention caches
    (``score_tensor_for_sparsity``, ``last_p_edge_on``, ...), plain
    LightningModule attributes such as ``_last_loss_components`` (a DICT of
    tensors — the cluster crash of job 12281530), and non-leaf BUFFERS after
    device moves (the commit shadow after ``.cuda()`` — job 12294402).
    Recursively detach them (attrs, dicts, lists, tuples, buffers) before
    copying, then restore the live model's originals.
    """
    stashed = []
    owners = [forecaster] + list(forecaster.modules())
    for obj in owners:
        for name, val in list(obj.__dict__.items()):
            if name in ("_parameters", "_modules"):
                continue
            if name == "_buffers":
                # Buffers are NOT always leaves: nn.Module._apply (device /
                # dtype moves) replaces them with op results, preserving
                # requires_grad — e.g. the commit shadow after .cuda().
                for bname, bval in list(val.items()):
                    _detach_graph_tensors(
                        bval, stashed,
                        lambda nv, d=val, k=bname: d.__setitem__(k, nv))
                continue
            _detach_graph_tensors(
                val, stashed,
                lambda nv, o=obj, n=name: object.__setattr__(o, n, nv))
    try:
        work = copy.deepcopy(forecaster)
    finally:
        for setitem, val in stashed:
            setitem(val)
    # The copy inherits the live phase's requires_grad masks (e.g. theta_R is
    # frozen during structure phases); the probe's inner refit must train
    # theta_R regardless, so unfreeze everything ON THE COPY.  Only the
    # reconstruction group is optimized (see _refit).
    for p in work.parameters():
        p.requires_grad_(True)
    for m in work.model.modules():
        if hasattr(m, "set_bkd_phase_active"):
            m.set_bkd_phase_active(False)
    work.eval()
    return work


def _apply_writes(work, candidate_writes, locs) -> None:
    tables = query_tables_of(work)
    with torch.no_grad():
        for node, vec in candidate_writes.items():
            t, r = locs[node]
            w = tables[t].embedding.weight
            w[r].copy_(vec.to(device=w.device, dtype=w.dtype))


def _refit(work, recon_cfg, k_inner, refit_batches, seed) -> None:
    """k_inner reconstruction-only steps on the copy's theta_R."""
    if k_inner <= 0:
        return
    _, recon_params = classify_parameters(work.model, verbose=False)
    opt = make_optimizer(recon_params, **recon_cfg)
    torch.manual_seed(seed)   # paired stochastic gates across arms
    for t in range(k_inner):
        S, X = refit_batches[t % len(refit_batches)]
        work.train()
        opt.zero_grad()
        recon_mse(work, S, X).backward()
        opt.step()


def resolve_inner_config(forecaster, inner_optimizer=None, inner_lr=None,
                         inner_weight_decay=None) -> dict:
    """Inner optimizer config: inherit the reconstruction arm, override per-key."""
    cfg = get_recon_optimizer_config(forecaster.config["training"])
    if inner_optimizer is not None:
        cfg["optimizer_type"] = inner_optimizer
    if inner_lr is not None:
        cfg["lr"] = inner_lr
    if inner_weight_decay is not None:
        cfg["weight_decay"] = inner_weight_decay
    return cfg


def paired_refit_probe(
    forecaster,
    candidate_writes: Dict[int, torch.Tensor],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    k_inner: int,
    inner_optimizer: Optional[str] = None,
    inner_lr: Optional[float] = None,
    inner_weight_decay: Optional[float] = None,
    accept_margin: float = 0.0,
    seed: int = 0,
    pair_mask: Optional[torch.Tensor] = None,
) -> ProbeResult:
    """Paired bilevel acceptance test for candidate query writes.

    Args:
        forecaster:       live AttentionSelectorForecaster (never mutated).
        candidate_writes: node index -> new query row vector (e.g. a centroid
                          ``c(S')`` from the commit controller).
        val_batches:      cached validation batches ``(S, X)``; the LAST one
                          is reserved for the HSIC evaluation, the rest feed
                          the refit (with only one batch it serves both).
        k_inner:          reconstruction refit steps per arm (0 = no refit).
        inner_*:          per-key overrides of the reconstruction optimizer
                          config (None = inherit from the training config).
        accept_margin:    delta threshold; accept iff cand < inc - margin.
        seed:             torch seed for the (paired) refit of each arm.
        pair_mask:        optional detached structural pair weights
                          (descendant x LOO, NO BKD factor) stashed by the
                          forecaster's ``_step``; shared by both arms so the
                          acceptance test measures the training objective.

    Returns:
        ProbeResult with per-node rows for both arms and, restricted to the
        written nodes, the paired deltas and accept flags.
    """
    if not val_batches:
        raise ValueError("paired_refit_probe needs at least one val batch.")
    recon_cfg = resolve_inner_config(forecaster, inner_optimizer, inner_lr,
                                     inner_weight_decay)
    locs = node_locations(forecaster)
    eval_batch = val_batches[-1]
    refit_batches = val_batches[:-1] or val_batches

    rows, mses = {}, {}
    for arm in ("incumbent", "candidate"):
        work = _prepare_copy(forecaster)
        if arm == "candidate":
            _apply_writes(work, candidate_writes, locs)
        if arm == "incumbent":
            # Latch bandwidths once from the pre-refit incumbent state.
            bw = _latch_bandwidths(work, *eval_batch)
        _refit(work, recon_cfg, k_inner, refit_batches, seed)
        rows[arm] = hsic_rows_eval(work, *eval_batch, sigma=bw,
                                   adaptive_bandwidth=False,
                                   pair_mask=pair_mask)
        mses[arm] = mse_rows_eval(work, *eval_batch)
        del work

    deltas, accepted, mse_deltas = {}, {}, {}
    for node in candidate_writes:
        d = float(rows["candidate"][node] - rows["incumbent"][node])
        deltas[node] = d
        accepted[node] = bool(d < -accept_margin)
        mse_deltas[node] = float(mses["candidate"][node]
                                 - mses["incumbent"][node])
        logger.info(
            "bilevel probe: node %d delta=%+.5f dMSE=%+.5f -> %s",
            node, d, mse_deltas[node],
            "ACCEPT" if accepted[node] else "reject",
        )
    return ProbeResult(rows_incumbent=rows["incumbent"],
                       rows_candidate=rows["candidate"],
                       mse_incumbent=mses["incumbent"],
                       mse_candidate=mses["candidate"],
                       deltas=deltas, accepted=accepted,
                       mse_deltas=mse_deltas)
