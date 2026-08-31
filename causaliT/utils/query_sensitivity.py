"""Query-perturbation sensitivity of the training HSIC.

Motivation
----------
The train HSIC can go "flat" with respect to the structure (the *dilution*
regime: the per-node MLPs overfit, so the residuals no longer respond to which
parents the model attends to).  A usable structural signal requires the HSIC
to REACT to a change in the learned structure.  This module measures exactly
that, following the probe developed in
``experiments/6_INVESTIGATIONS/LARGER_DAGS/mlp_dropout_sweep_11489343``
(evaluate_mlp_dropout_sweep.ipynb):

    sensitivity = |HSIC(q + dq) - HSIC(q)| / (|dq| / |q|)

where ``q`` are the free query embeddings (the learned structure-to-be) and
``dq`` a small Gaussian perturbation scaled per row.  A HIGHER sensitivity
means the train HSIC carries a usable signal about the structure; ~0 is the
dilution signature.  The measurement runs in eval mode (dropout off): it
probes the smoothness of the learned function, not dropout noise.

The HSIC is computed exactly as in ``AttentionSelectorForecaster._step`` —
same target layout (split vs homogeneous), same kernel hyperparameters — so
the measured sensitivity is the one training actually optimizes.  The
descendant-exclusion pair mask is intentionally NOT applied: the probe is
meant to characterize the raw HSIC landscape at the post-warmup checkpoint.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from causaliT.utils.hsic_utils import hsic_cross_per_pair

logger = logging.getLogger(__name__)


def _free_query_weights(forecaster) -> List[torch.Tensor]:
    """Free query embedding tables (``query_embed_S/X``), padding row included.

    Returns an empty list when the model has no free query embedding
    (``free_query_embedding=False``) — the perturbation probe is undefined
    there.
    """
    layer = forecaster.model
    qs: List[torch.Tensor] = []
    for name in ("query_embed_S", "query_embed_X"):
        emb = getattr(layer, name, None)
        table = getattr(emb, "embedding", None)
        weight = getattr(table, "weight", None)
        if isinstance(weight, torch.Tensor):
            qs.append(weight)
    return qs


@torch.no_grad()
def scalar_hsic(forecaster, batches) -> float:
    """The training-time scalar HSIC (mean over pairs and batches), eval mode.

    ``batches`` is an iterable of ``(S, X)`` tensors (already on the model's
    device).  Mirrors ``AttentionSelectorForecaster._step``: the target is the
    X values in split mode and ``cat([S_values, X_values])`` in homogeneous
    mode; the candidate-parent source is the combined ``[S, X]`` values.
    """
    was_training = forecaster.training
    forecaster.eval()
    val_idx = forecaster.val_idx
    vals: List[float] = []
    for s_b, x_b in batches:
        pred = forecaster.forward(data_source=s_b, data_intermediate=x_b)[0]
        x_val = x_b[:, :, val_idx]
        if forecaster.homogeneous_nodes:
            target = torch.cat([s_b[:, :, val_idx], x_val], dim=1)  # (B, N)
        else:
            target = x_val                                          # (B, L_X)
        residuals = target - pred.squeeze(-1)
        # Candidate-parent source, mirroring _step: in homogeneous mode the
        # target already IS all N nodes; in split mode the S values are
        # prepended to form the combined [S, X] source.
        if forecaster.homogeneous_nodes:
            combined_source = target
        else:
            combined_source = torch.cat([s_b[:, :, val_idx], target], dim=1)
        vals.append(
            float(
                hsic_cross_per_pair(
                    combined_source,
                    residuals,
                    sigma=forecaster.hsic_sigma,
                    adaptive_bandwidth=forecaster.hsic_adaptive_bandwidth,
                    mode=forecaster.hsic_mode,
                    nhsic_epsilon=forecaster.nhsic_epsilon,
                    source_kernel=forecaster.hsic_kernel_source,
                    bandwidth_multipliers=getattr(
                        forecaster, "hsic_bandwidth_multipliers", None
                    ),
                ).item()
            )
        )
    if was_training:
        forecaster.train()
    return float(np.mean(vals))


@torch.no_grad()
def hsic_query_sensitivity(
    forecaster,
    batches,
    eps: float = 1.0,
    n_pert: int = 10,
    seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """|dHSIC| per unit RELATIVE query perturbation, over ``n_pert`` draws.

    Each realization adds Gaussian noise to the free query embeddings, scaled
    per row by ``eps * ||row|| / sqrt(d_model)`` (the padding row 0 is never
    touched), measures the HSIC change on ``batches``, then RESTORES the
    original weights — the model is left exactly as found.

    Args:
        forecaster: AttentionSelectorForecaster (eval mode is set inside).
        batches:    iterable of (S, X) tensors on the model's device.
        eps:        relative perturbation size (1.0 = row-norm scale).
        n_pert:     number of perturbation realizations.
        seed:       RNG seed for the perturbations (CPU generator).

    Returns:
        (base_hsic, sens, deltas): the unperturbed HSIC, the per-realization
        sensitivity array, and the per-realization signed HSIC change.

    Raises:
        ValueError: if the model has no free query embedding tables.
    """
    qs = _free_query_weights(forecaster)
    if not qs:
        raise ValueError(
            "hsic_query_sensitivity requires free query embeddings "
            "(query_embed_S/X); none found.  Set free_query_embedding=True."
        )

    q0 = [q.detach().clone() for q in qs]
    q_norm = float(np.sqrt(sum(float((r[1:] ** 2).sum()) for r in q0)))
    base = scalar_hsic(forecaster, batches)

    g = torch.Generator().manual_seed(seed)
    sens: List[float] = []
    deltas: List[float] = []
    for _ in range(n_pert):
        dqs = []
        for q, ref in zip(qs, q0):
            dq = torch.zeros_like(q)
            dq[1:] = (
                torch.randn(q[1:].shape, generator=g).to(q.device)
                * eps
                * ref[1:].norm(dim=1, keepdim=True)
                / np.sqrt(q.shape[1])
            )
            dqs.append(dq)
        for q, dq in zip(qs, dqs):
            q.add_(dq)
        pert = scalar_hsic(forecaster, batches)
        for q, ref in zip(qs, q0):
            q.copy_(ref)  # restore the original queries
        rel = float(np.sqrt(sum(float((dq ** 2).sum()) for dq in dqs))) / q_norm
        sens.append(abs(pert - base) / rel)
        deltas.append(pert - base)
    return base, np.asarray(sens), np.asarray(deltas)
