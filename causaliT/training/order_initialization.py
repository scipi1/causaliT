"""One-off residual-HSIC order initialization; existence gates are unchanged."""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from causaliT.utils.hsic_utils import hsic_cross_per_pair


def scores_from_hsic(values, strength, min_std):
    """Return centred population-standardized scores, or None for unsafe data."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("order initialization strength must be finite and nonnegative")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("order initialization min_std must be finite and positive")
    if not torch.isfinite(values).all():
        return None
    std = values.std(unbiased=False)
    if std <= min_std:
        return None
    return strength * (values - values.mean()) / std


@torch.no_grad()
def initialize_order_from_hsic(forecaster, loader, config, report_path):
    """Probe deterministic relaxed gates on fold B and initialize at most once.

    Row statistics are averaged equally across batches. No self or descendant
    tests enter the row means. Modes and RNG state are restored after probing.
    The returned report is also persisted in forecaster checkpoints.
    """
    previous = getattr(forecaster, "_order_initialization_report", None)
    if previous is not None:
        return previous
    strength = float(config.get("strength", 0.05))
    min_std = float(config.get("min_std", 1e-6))
    scores_from_hsic(torch.zeros(2), strength, min_std)
    batch_size = int(config.get("batch_size", 256))
    n_batches = int(config.get("n_batches", 4))
    if batch_size < 4 or n_batches < 1:
        raise ValueError("order initialization requires batch_size >= 4 and n_batches >= 1")
    oa = forecaster.model.order_attention()
    if oa is None or not forecaster.homogeneous_nodes:
        raise ValueError("HSIC order initialization requires homogeneous order attention")
    if loader is None:
        raise ValueError("HSIC order initialization requires a held-out fold B loader")
    # Explicit fold dataset, independent of the training loader's batch size.
    probe = DataLoader(loader.dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=loader.collate_fn,
                       generator=torch.Generator().manual_seed(0))
    modes = [(m, m.training) for m in forecaster.modules()]
    old_mode = oa.order_path
    device = oa.order_scores.device
    rows, mse, pred_var, target_var = [], [], [], []
    sizes = []
    try:
        forecaster.eval()
        oa.set_order_path("relaxed")
        with torch.random.fork_rng(devices=([device.index] if device.type == "cuda" else [])):
            for index, batch in enumerate(probe):
                if index >= n_batches:
                    break
                s, x = batch[0].to(device), batch[1].to(device)
                if len(s) < 4:
                    continue
                pred = forecaster.forward(data_source=s, data_intermediate=x)[0].squeeze(-1)
                target = torch.cat([s[:, :, forecaster.val_idx], x[:, :, forecaster.val_idx]], dim=1)
                residual = target - pred
                _, matrix = hsic_cross_per_pair(
                    target, residual, sigma=forecaster.hsic_sigma,
                    adaptive_bandwidth=forecaster.hsic_adaptive_bandwidth,
                    mode=forecaster.hsic_mode, nhsic_epsilon=forecaster.nhsic_epsilon,
                    source_kernel=forecaster.hsic_kernel_source,
                    bandwidth_multipliers=getattr(forecaster, "hsic_bandwidth_multipliers", None),
                    return_matrix=True)
                matrix = matrix.clone()
                matrix.fill_diagonal_(0)
                rows.append((matrix.sum(1) / (matrix.shape[1] - 1)).cpu())
                mse.append(residual.square().mean(0).cpu())
                pred_var.append(pred.var(0, unbiased=False).cpu())
                target_var.append(target.var(0, unbiased=False).cpu())
                sizes.append(len(s))
    finally:
        oa.set_order_path(old_mode)
        for module, training in modes:
            module.training = training
    if not rows:
        raise ValueError("No usable held-out batches for HSIC order initialization")
    h = torch.stack(rows).mean(0)
    scores = scores_from_hsic(h, strength, min_std)
    report = dict(completed=True, status="applied" if scores is not None else "skipped_unsafe_statistic",
                  strength=strength, min_std=min_std, batch_sizes=sizes,
                  hsic_rows=h.tolist(), hsic_rows_by_batch=torch.stack(rows).tolist(),
                  mse=torch.stack(mse).mean(0).tolist(),
                  prediction_variance=torch.stack(pred_var).mean(0).tolist(),
                  target_variance=torch.stack(target_var).mean(0).tolist(),
                  scores_before=oa.order_scores.detach().cpu().tolist(),
                  scores_after=(scores if scores is not None else oa.order_scores.detach().cpu()).tolist())
    report["descending_node_indices"] = torch.argsort(
        torch.tensor(report["scores_after"]), descending=True, stable=True).tolist()
    # Strict JSON represents invalid diagnostics as null, never NaN/Infinity.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    report = clean(report)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if scores is not None:
        oa.order_scores.copy_(scores.to(oa.order_scores))
    forecaster._order_initialization_report = report
    return report