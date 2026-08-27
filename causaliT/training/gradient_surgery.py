"""
Gradient surgery (PCGrad) reconciliation between HSIC and the structural
regularizers (L0, NOTEARS).

The L0 penalty and the NOTEARS acyclicity term back-propagate through the
**same structural pathway** as HSIC (Q/K projections, structural embeddings --
see :mod:`causaliT.training.interference_utils`).  When their gradients point
against the HSIC gradient they cancel the signal that pushes each query towards
its true parent.  This module implements the standard remedy from the
multi-task literature, **PCGrad** (Yu et al., NeurIPS 2020): for every
regularizer gradient ``g_reg`` that conflicts with the reference (HSIC)
gradient ``g_ref``,

    g_reg <- g_reg - (<g_reg, g_ref> / ||g_ref||^2) * g_ref   iff  <g_reg, g_ref> < 0

i.e. the regularizer keeps only the component that is **non-destructive for
HSIC**.  Aligned or gradient-free blocks are left untouched, so with no
conflict the update is bit-identical to the summed loss.

Surgery is applied **per module block** (the grouping from
:func:`causaliT.training.interference_utils.build_interference_blocks`), never
on the flattened full-model gradient: conflict is localised in the structural
pathway, and a global projection would let one conflicting block distort
unrelated blocks.  A block where either gradient has (near-)zero norm is
skipped (mirrors the NaN-cosine "no signal here" convention of the
interference diagnostic).

This is the *inner* (per-step, vector-level) safeguard.  It composes with the
existing *outer* (scalar) mechanisms, which are unchanged: the
``*_max_hsic_pct`` caps limit regularizer *magnitude*, the adaptive gates
disarm stale regularizers between phases, and PCGrad fixes the regularizer
*direction* within each step.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["pcgrad_reconcile"]


def _flat_or_zero(
    g: Optional[torch.Tensor], p: torch.nn.Parameter
) -> torch.Tensor:
    """Flatten one grad entry; ``None`` (unused param) becomes zeros."""
    if g is None:
        return torch.zeros(p.numel(), device=p.device, dtype=p.dtype)
    return g.reshape(-1)


def _cosine_from_flat(a: torch.Tensor, b: torch.Tensor, eps: float) -> float:
    na = a.norm()
    nb = b.norm()
    if na < eps or nb < eps:
        return float("nan")
    return float((a @ b) / (na * nb))



def pcgrad_reconcile(
    ref_grads: Sequence[Optional[torch.Tensor]],
    target_grads: Dict[str, Sequence[Optional[torch.Tensor]]],
    blocks: Dict[str, List[torch.nn.Parameter]],
    all_params: List[torch.nn.Parameter],
    eps: float = 1e-12,
) -> Tuple[Dict[str, List[Optional[torch.Tensor]]], Dict[str, float]]:
    """Project conflicting regularizer gradients per block (PCGrad).

    Args:
        ref_grads:    Gradients of the reference objective (HSIC term),
                      aligned with ``all_params`` (``None`` allowed).
        target_grads: ``{name: grads}`` for each regularizer (e.g. ``"l0"``,
                      ``"notears"``), each aligned with ``all_params``.
        blocks:       Mapping ``{block_name: [param, ...]}`` (e.g. from
                      :func:`build_interference_blocks`, possibly filtered).
                      Every parameter must appear in ``all_params``.
        all_params:   Flat parameter list defining the gradient order.
        eps:          Norm floor below which a block gradient counts as "no
                      signal" and is left untouched.

    Returns:
        ``(projected, metrics)`` where ``projected`` maps each target name to
        a list of per-param gradient tensors aligned with ``all_params``, and
        ``metrics`` contains, per target ``t``: ``"cos_pre_{t}"`` /
        ``"cos_post_{t}"`` (cosine with the reference over the concatenated
        valid blocks) and ``"frac_projected_{t}"`` (fraction of valid blocks
        in which the projection fired).  All three are NaN when no block had
        signal for that target.
    """
    param_index = {id(p): i for i, p in enumerate(all_params)}
    projected: Dict[str, List[Optional[torch.Tensor]]] = {}
    metrics: Dict[str, float] = {}

    for name, grads in target_grads.items():
        out: List[Optional[torch.Tensor]] = [
            g.clone() if g is not None else None for g in grads
        ]
        pre_ref: List[torch.Tensor] = []
        pre_tgt: List[torch.Tensor] = []
        post_ref: List[torch.Tensor] = []
        post_tgt: List[torch.Tensor] = []
        n_valid = 0
        n_projected = 0

        for _, plist in blocks.items():
            idxs = [param_index[id(p)] for p in plist]
            v_ref = torch.cat(
                [_flat_or_zero(ref_grads[i], p) for i, p in zip(idxs, plist)]
            )
            v_tgt = torch.cat(
                [_flat_or_zero(grads[i], p) for i, p in zip(idxs, plist)]
            )
            n_ref = v_ref.norm()
            n_tgt = v_tgt.norm()
            if n_ref < eps or n_tgt < eps:
                continue  # no signal in this block -> untouched

            n_valid += 1
            pre_ref.append(v_ref)
            pre_tgt.append(v_tgt)

            dot = float(v_ref @ v_tgt)
            if dot < 0.0:
                n_projected += 1
                v_tgt = v_tgt - (dot / (float(n_ref) ** 2)) * v_ref
                offset = 0
                for i, p in zip(idxs, plist):
                    numel = p.numel()
                    out[i] = v_tgt[offset: offset + numel].reshape(p.shape).clone()
                    offset += numel

            post_ref.append(v_ref)
            post_tgt.append(v_tgt)

        if n_valid > 0:
            metrics[f"cos_pre_{name}"] = _cosine_from_flat(
                torch.cat(pre_ref), torch.cat(pre_tgt), eps
            )
            metrics[f"cos_post_{name}"] = _cosine_from_flat(
                torch.cat(post_ref), torch.cat(post_tgt), eps
            )
            metrics[f"frac_projected_{name}"] = n_projected / n_valid
        else:
            metrics[f"cos_pre_{name}"] = float("nan")
            metrics[f"cos_post_{name}"] = float("nan")
            metrics[f"frac_projected_{name}"] = float("nan")

        projected[name] = out

    return projected, metrics
