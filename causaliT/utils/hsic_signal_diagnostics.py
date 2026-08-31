"""Local diagnostics for node-wise HSIC and its query gradient.

The diagnostic answers two questions on a fixed checkpoint:
1. Does node-wise HSIC decrease at the true-parent centroid relative to wrong
   centroids?
2. Does the HSIC gradient point toward the true-parent centroid in eval mode
   and under the stochastic training path?

All query interventions restore the original weights before returning.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic_cross_per_pair, hsic_pair_matrix


@dataclass(frozen=True)
class DiagnosticBatch:
    source: torch.Tensor
    intermediate: torch.Tensor


def load_dataset_batches(
    dataset_dir: Path,
    batch_size: int,
    n_batches: int,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> List[DiagnosticBatch]:
    """Load a fixed, seeded subsample of an SCM ``ds.npz`` dataset."""
    data = np.load(dataset_dir / "ds.npz")
    n = len(data["x"])
    if batch_size <= 0 or n_batches <= 0:
        raise ValueError("batch_size and n_batches must be positive")
    if batch_size * n_batches > n:
        raise ValueError("requested batches exceed the dataset size")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n)[: batch_size * n_batches])
    out = []
    for b in range(n_batches):
        rows = idx[b * batch_size : (b + 1) * batch_size]
        s = torch.tensor(np.asarray(data["s"][rows]), dtype=torch.float32)
        x = torch.tensor(np.asarray(data["x"][rows]), dtype=torch.float32)
        if device is not None:
            s, x = s.to(device), x.to(device)
        out.append(DiagnosticBatch(source=s, intermediate=x))
    return out


def load_ground_truth(dataset_dir: Path) -> np.ndarray:
    """Return the square ``[child, parent]`` ground-truth adjacency."""
    cross = pd.read_csv(dataset_dir / "dec1_cross_att_mask.csv", index_col=0)
    self_gt = pd.read_csv(dataset_dir / "dec1_self_att_mask.csv", index_col=0)
    n_s, n_x = cross.shape[1], cross.shape[0]
    gt = np.zeros((n_s + n_x, n_s + n_x), dtype=bool)
    gt[n_s:, :n_s] = cross.values.astype(bool)
    gt[n_s:, n_s:] = self_gt.values.astype(bool)
    return gt


def free_query_weights(model: AttentionSelectorForecaster) -> List[torch.Tensor]:
    """Free query tables in global node order, padding row included."""
    weights = []
    for name in ("query_embed_S", "query_embed_X"):
        table = getattr(model.model, name, None)
        weight = getattr(getattr(table, "embedding", None), "weight", None)
        if isinstance(weight, torch.Tensor):
            weights.append(weight)
    if not weights:
        raise ValueError("the model has no free query embedding tables")
    return weights


def key_frame(model: AttentionSelectorForecaster) -> torch.Tensor:
    """Frozen structural key frame in global node order."""
    frames = []
    for name in ("orth_embed_S", "orth_embed_X"):
        frame = getattr(getattr(model.model, name, None), "frame", None)
        if frame is None:
            raise ValueError(f"missing frozen key frame {name}")
        frames.append(frame.detach())
    return torch.cat(frames, dim=0)


def node_query_views(model: AttentionSelectorForecaster) -> List[torch.Tensor]:
    """Return mutable query rows in global node order, excluding padding."""
    rows = []
    for weight in free_query_weights(model):
        rows.extend(weight[i] for i in range(1, weight.shape[0]))
    return rows


def normalized_centroid(k: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if len(indices) == 0:
        raise ValueError("a centroid requires at least one key")
    c = k[list(indices)].mean(dim=0)
    return c / c.norm().clamp_min(1e-12)



def _targets_and_sources(
    model: AttentionSelectorForecaster, batch: DiagnosticBatch
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = model.forward(
        data_source=batch.source, data_intermediate=batch.intermediate
    )[0]
    x_val = batch.intermediate[:, :, model.val_idx]
    if model.homogeneous_nodes:
        target = torch.cat([batch.source[:, :, model.val_idx], x_val], dim=1)
    else:
        target = x_val
    target = torch.nan_to_num(target)
    residuals = target.squeeze(-1) - pred.squeeze(-1)
    if model.homogeneous_nodes:
        source = target
    else:
        source = torch.cat([batch.source[:, :, model.val_idx], target], dim=1)
    return source, residuals


def node_hsic(
    model: AttentionSelectorForecaster,
    batches: Sequence[DiagnosticBatch],
    bandwidth_multipliers: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    """Mean node-row HSIC and scalar objective over fixed batches (eval mode)."""
    was_training = model.training
    model.eval()
    rows, totals = [], []
    with torch.no_grad():
        for batch in batches:
            source, residuals = _targets_and_sources(model, batch)
            mat = hsic_pair_matrix(
                source_values=source,
                residuals=residuals,
                sigma=model.hsic_sigma,
                adaptive_bandwidth=model.hsic_adaptive_bandwidth,
                mode=model.hsic_mode,
                nhsic_epsilon=model.nhsic_epsilon,
                source_kernel=model.hsic_kernel_source,
                bandwidth_multipliers=bandwidth_multipliers,
            )
            valid = ~torch.isnan(mat)
            row = mat[valid].reshape(mat.shape[0], -1).mean(dim=1)
            rows.append(row.cpu().numpy())
            totals.append(float(mat[valid].mean().item()))
    if was_training:
        model.train()
    return np.stack(rows).mean(axis=0), float(np.mean(totals))

def query_intervention_probe(
    model: AttentionSelectorForecaster,
    batches: Sequence[DiagnosticBatch],
    gt: np.ndarray,
    n_wrong: int = 3,
    seed: int = 0,
    bandwidth_multipliers: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Compare current, true-parent, wrong-parent, and random query positions."""
    k = key_frame(model)
    queries = node_query_views(model)
    n_nodes = len(queries)
    if gt.shape != (n_nodes, n_nodes):
        raise ValueError(f"GT shape {gt.shape} does not match {n_nodes} nodes")
    rng = np.random.default_rng(seed)
    base_rows, base_total = node_hsic(model, batches, bandwidth_multipliers)
    records = []

    def evaluate(node: int, value: torch.Tensor) -> Tuple[float, float]:
        q = queries[node]
        original = q.detach().clone()
        try:
            with torch.no_grad():
                q.copy_(value)
            rows, total = node_hsic(model, batches, bandwidth_multipliers)
            return float(rows[node]), total
        finally:
            with torch.no_grad():
                q.copy_(original)

    for node in range(n_nodes):
        parents = np.flatnonzero(gt[node])
        if len(parents) == 0:
            continue
        current = float(base_rows[node])
        true_value = normalized_centroid(k, parents)
        true_hsic, true_total = evaluate(node, true_value)

        nonparents = [j for j in range(n_nodes) if j != node and not gt[node, j]]
        wrong_deltas, wrong_values = [], []
        if len(nonparents) >= len(parents):
            for _ in range(n_wrong):
                subset = rng.choice(nonparents, size=len(parents), replace=False)
                value = normalized_centroid(k, subset)
                row_value, _ = evaluate(node, value)
                wrong_values.append(row_value)
                wrong_deltas.append(current - row_value)

        generator = torch.Generator().manual_seed(seed + node)
        random_value = torch.randn(k.shape[1], generator=generator)
        random_value = random_value / random_value.norm().clamp_min(1e-12)
        random_hsic, _ = evaluate(node, random_value.to(queries[node].dtype))
        records.append({
            "node": node,
            "n_parents": len(parents),
            "current_node_hsic": current,
            "true_centroid_hsic": true_hsic,
            "true_delta": current - true_hsic,
            "wrong_centroid_hsic_mean": float(np.mean(wrong_values)),
            "wrong_delta_mean": current - float(np.mean(wrong_values)),
            "wrong_delta_min": float(np.min(wrong_deltas)),
            "wrong_delta_max": float(np.max(wrong_deltas)),
            "random_query_hsic": random_hsic,
            "random_delta": current - random_hsic,
            "total_hsic_at_true": true_total,
            "base_total_hsic": base_total,
        })
    return pd.DataFrame(records)

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12)
    return float((a * b).sum().div(denom).item())


def _gradient_once(
    model: AttentionSelectorForecaster,
    batch: DiagnosticBatch,
    training_path: bool,
    bandwidth_multipliers: Optional[Sequence[float]],
) -> List[torch.Tensor]:
    weights = free_query_weights(model)
    if training_path:
        model.train()
        model._step((batch.source, batch.intermediate), stage="train")
        objective = model._last_hsic_reg
    else:
        model.eval()
        source, residuals = _targets_and_sources(model, batch)
        objective = hsic_cross_per_pair(
            source,
            residuals,
            sigma=model.hsic_sigma,
            adaptive_bandwidth=model.hsic_adaptive_bandwidth,
            mode=model.hsic_mode,
            nhsic_epsilon=model.nhsic_epsilon,
            source_kernel=model.hsic_kernel_source,
            bandwidth_multipliers=bandwidth_multipliers,
        )
    grads = torch.autograd.grad(objective, weights, allow_unused=False)
    return [g.detach().cpu() for g in grads]


def gradient_probe(
    model: AttentionSelectorForecaster,
    batches: Sequence[DiagnosticBatch],
    gt: np.ndarray,
    training_path: bool = False,
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-node query-gradient geometry over fixed batches.

    ``training_path=False`` is eval mode without BKD.  ``True`` calls the
    regular training ``_step``, including stochastic BKD and the matching
    dropped-source HSIC mask.
    """
    was_training = model.training
    old_multipliers = getattr(model, "hsic_bandwidth_multipliers", None)
    model.hsic_bandwidth_multipliers = bandwidth_multipliers
    k = key_frame(model).cpu()
    weights = free_query_weights(model)
    n_nodes = sum(w.shape[0] - 1 for w in weights)
    if gt.shape != (n_nodes, n_nodes):
        raise ValueError(f"GT shape {gt.shape} does not match {n_nodes} nodes")
    all_grads = []
    try:
        for b, batch in enumerate(batches):
            torch.manual_seed(seed + b)
            all_grads.append(_gradient_once(
                model, batch, training_path, bandwidth_multipliers
            ))
    finally:
        model.hsic_bandwidth_multipliers = old_multipliers
        if was_training:
            model.train()
        else:
            model.eval()

    # (B, N, d) node gradients in global node order.
    g = torch.stack([torch.cat([w[1:] for w in grads]) for grads in all_grads])
    records = []
    for node in range(n_nodes):
        parents = np.flatnonzero(gt[node])
        if len(parents) == 0:
            continue
        parent_c = normalized_centroid(k, parents)
        nonparents = [j for j in range(n_nodes) if j != node and not gt[node, j]]
        wrong_c = normalized_centroid(k, nonparents)
        node_g = g[:, node]
        update = -node_g
        mean_g = node_g.mean(dim=0)
        second = node_g.pow(2).sum(dim=1).mean()
        signal = mean_g.pow(2).sum()
        noise = (second - signal).clamp_min(0.0) / node_g.shape[1]
        snr = float(signal.item() / max(float(noise), 1e-12))
        pair_cos = []
        for a in range(len(node_g)):
            for b in range(a + 1, len(node_g)):
                pair_cos.append(_cosine(node_g[a], node_g[b]))
        records.append({
            "node": node,
            "n_parents": len(parents),
            "training_path": training_path,
            "grad_norm_mean": float(node_g.norm(dim=1).mean()),
            "grad_norm_std": float(node_g.norm(dim=1).std(unbiased=False)),
            "parent_alignment_mean": float(np.mean([_cosine(u, parent_c) for u in update])),
            "wrong_alignment_mean": float(np.mean([_cosine(u, wrong_c) for u in update])),
            "parent_minus_wrong_mean": float(np.mean([
                _cosine(u, parent_c) - _cosine(u, wrong_c) for u in update
            ])),
            "mean_gradient_parent_alignment": _cosine(-mean_g, parent_c),
            "mean_gradient_wrong_alignment": _cosine(-mean_g, wrong_c),
            "gradient_snr": snr,
            "direction_stability": float(np.mean(pair_cos)) if pair_cos else math.nan,
        })
    return pd.DataFrame(records)
