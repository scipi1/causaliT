"""Pre-flight MLP-dropout selection by query-perturbation sensitivity.

Motivation
----------
The n=20 investigation (experiments/6_INVESTIGATIONS/LARGER_DAGS/README.md)
found that the train-HSIC sensitivity to query perturbations is bow-shaped in
the per-node MLP dropout: a maximum at the sweet spot between overfit (all
residuals similar and small) and underfit (all similar and large).  The
optimal dropout can therefore be SELECTED by maximizing that sensitivity after
the initial reconstruction phase, instead of hand-picking it.

This module implements the selection as a pre-flight stage of the adaptive
trainer: for each candidate dropout, a fresh model runs a short
reconstruction-only warmup (structure frozen) and its query-perturbation
sensitivity is measured on the structure split (the honest, out-of-sample
half when cross-fitting is active).  The argmax candidate wins; the main
adaptive run is then built with the winning dropout and warm-started from the
winner's warmup weights.

Only the two per-node MLP dropouts (value embedding ``mlp_per_node`` in
``ds_embed_S`` / ``ds_embed_X``) and the output-head dropout
(``output_mlp_dropout``) are swept — mirroring the dropout-sweep isolation;
every other dropout stays at its configured value.  The pre-flight epochs are
selection overhead and do NOT count against ``total_epoch_budget``.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from causaliT.utils.query_sensitivity import hsic_query_sensitivity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_mlp_dropout(config: Dict[str, Any], p: float) -> None:
    """Write dropout ``p`` into the swept locations of a (resolved) config.

    Targets exactly the per-node value-MLP embeddings (``embed:
    mlp_per_node``) in ``ds_embed_S`` / ``ds_embed_X`` and the output head
    (``output_mlp_dropout``); all other dropout knobs are left untouched.
    Plain-item assignment works for both dict and OmegaConf containers and
    overrides any interpolation (e.g. ``${experiment.dropout}``).
    """
    kwargs = config["model"]["kwargs"]
    for embed_key in ("ds_embed_S", "ds_embed_X"):
        for mod in kwargs[embed_key]["modules"]:
            if mod.get("embed") == "mlp_per_node":
                mod["kwargs"]["dropout"] = p
    kwargs["output_mlp_dropout"] = p


# ---------------------------------------------------------------------------
# Per-candidate warmup + sensitivity
# ---------------------------------------------------------------------------

def _recon_warmup(model, dm, warmup_epochs: int, cluster: bool) -> None:
    """Reconstruction-only warmup: train with the structure frozen.

    Uses the forecaster's ``training.freeze_structural_params`` flag (applied
    in ``on_fit_start``): with gradient routing the structural optimizer step
    is then a no-op, so this is exactly the reconstruct phase.
    """
    if hasattr(dm, "set_active_phase"):
        dm.set_active_phase("reconstruct")
    trainer = pl.Trainer(
        max_epochs=warmup_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=not cluster,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,  # selection needs no validation metrics
        # Deterministic warmup: the selection must be reproducible (same seed
        # -> same dropout chosen).  Safe because the caller sets
        # CUBLAS_WORKSPACE_CONFIG before any GEMM (see adaptive_trainer).
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)


def _structure_split_batches(dm, n_batches: int, device) -> List:
    """Up to ``n_batches`` (S, X) batches from the STRUCTURE split.

    With cross-fitting active this is the honest half (the reconstruct warmup
    never saw it); without cross-fitting it is the plain training set.
    """
    if hasattr(dm, "set_active_phase"):
        dm.set_active_phase("structure")
    batches: List = []
    for batch in dm.train_dataloader():
        s_b, x_b = batch[0], batch[1]
        batches.append((s_b.to(device), x_b.to(device)))
        if len(batches) >= n_batches:
            break
    return batches


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dropout_selection(
    config: Dict[str, Any],
    data_dir: str,
    dm,
    save_dir: str,
    cluster: bool,
    seed: int,
) -> Tuple[Optional[float], Optional[str]]:
    """Run the pre-flight dropout selection.

    Returns ``(best_dropout, winner_ckpt_path)`` — both ``None`` when the
    selection is skipped (guard triggered), in which case the caller keeps the
    configured dropout and no warm-start.
    """
    from causaliT.training.trainer import create_model_instance

    ds_cfg = config.get("adaptive_training", {}).get("dropout_selection", {}) or {}
    candidates: List[float] = [float(c) for c in ds_cfg.get("candidates", [])]
    warmup_epochs = int(ds_cfg.get("warmup_epochs", 99))
    n_pert = int(ds_cfg.get("n_pert", 10))
    n_batches = int(ds_cfg.get("n_batches", 5))
    eps = float(ds_cfg.get("eps", 1.0))

    # ---- Guards -----------------------------------------------------------
    if len(candidates) < 2:
        logger.warning(
            "[dropout_selection] fewer than 2 candidates (%s) - skipping.",
            candidates,
        )
        return None, None
    if not config["model"]["kwargs"].get("free_query_embedding", False):
        logger.warning(
            "[dropout_selection] free_query_embedding is off - the query "
            "perturbation probe is undefined.  Skipping the selection."
        )
        return None, None

    out_dir = Path(save_dir) / "stage_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[float, Dict[str, float]] = {}
    best_p: Optional[float] = None
    best_sens = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for p in candidates:
        cfg_p = copy.deepcopy(config)
        _set_mlp_dropout(cfg_p, p)
        # Identical init across candidates: same seed -> same weights.
        seed_everything(seed)
        model = create_model_instance(cfg_p, data_dir)
        model.freeze_structural_params = True  # recon-only warmup
        _recon_warmup(model, dm, warmup_epochs, cluster)

        device = next(model.parameters()).device
        batches = _structure_split_batches(dm, n_batches, device)
        base, sens, _ = hsic_query_sensitivity(
            model, batches, eps=eps, n_pert=n_pert, seed=seed
        )
        sens_mean = float(np.mean(sens))
        results[p] = {"hsic_base": base, "sensitivity": sens_mean,
                      "sensitivity_std": float(np.std(sens))}
        logger.info(
            "[dropout_selection] dropout=%g: HSIC=%.4e, sensitivity=%.4e +/- %.2e",
            p, base, sens_mean, float(np.std(sens)),
        )
        if sens_mean > best_sens:
            best_sens = sens_mean
            best_p = p
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert best_p is not None and best_state is not None

    # ---- Selection summary (lands in the SLURM / training log) ------------
    lines = [
        "[dropout_selection] sensitivity per candidate "
        f"(warmup_epochs={warmup_epochs}, n_pert={n_pert}, n_batches={n_batches}, "
        f"eps={eps}):",
    ]
    for p in candidates:
        r = results[p]
        marker = "  <-- SELECTED" if p == best_p else ""
        lines.append(
            f"  dropout={p:<5g} HSIC={r['hsic_base']:.4e} "
            f"sensitivity={r['sensitivity']:.4e} +/- {r['sensitivity_std']:.2e}"
            f"{marker}"
        )
    logger.info("\n".join(lines))

    winner_ckpt = out_dir / "dropout_selection_winner.ckpt"
    torch.save({"state_dict": best_state,
                "dropout_selection": {"dropout": best_p}}, str(winner_ckpt))

    report = {
        "candidates": candidates,
        "warmup_epochs": warmup_epochs,
        "n_pert": n_pert,
        "n_batches": n_batches,
        "eps": eps,
        "results": {str(p): r for p, r in results.items()},
        "best_dropout": best_p,
        "winner_checkpoint": str(winner_ckpt),
    }
    report_path = Path(save_dir) / "dropout_selection.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info(
        "[dropout_selection] winner: dropout=%g (sensitivity=%.4e) -> %s",
        best_p, best_sens, winner_ckpt,
    )
    if not cluster:
        print(f"  [dropout_selection] best dropout={best_p} "
              f"(sensitivity={best_sens:.3e}); report: {report_path}")
    return best_p, str(winner_ckpt)
