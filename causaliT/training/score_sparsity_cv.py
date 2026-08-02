"""
Score Sparsity Cross-Validation: Select λ_score via k-fold CV.

This module implements Stage 2 of the staged training pipeline.
After causal initialization has established a good structural prior,
this stage selects the optimal attention score LASSO regularization
(λ_score) via cross-validation.

Key design decisions:
- A **single** λ_score is used for both cross- and self-attention modules.
  This keeps the search 1D and tractable with k-fold CV.
- Each fold trains from the **causal_init checkpoint** (if available),
  ensuring the CV operates in the structurally-initialized landscape.
- Selection criteria: configurable via ``score_sparsity_selection_rule``:
  "min_hsic" (default, for causal models) or "min_recon" (for baselines).
  Both metrics are always reported in the summary for easy comparison.

Process:
1. For each λ_score candidate in the user-specified grid:
   a. Run k-fold cross-validation (short training from init checkpoint)
   b. Collect val_hsic_cross and val_loss_x per fold
   c. Average across folds
2. Select best λ_score by the configured selection rule
3. Save comprehensive summary with per-lambda, per-fold results

Pipeline position::

    Stage 0 (Calibration)   → λ_group, λ_hsic
    Stage 1 (Causal Init)   → causal_init checkpoint
    Stage 2 (this)           → λ_score  (via k-fold CV)
    Stage 3 (Main Training)  → final model (with λ_score applied)

Usage::

    best_lambda = run_score_sparsity_cv(
        config, data_dir, save_dir,
        starting_checkpoint=init_ckpt,
        cluster=False,
    )
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION METRICS CALLBACK
# =============================================================================

class ValidationMetricsTracker(pl.Callback):
    """
    Callback to capture validation HSIC and reconstruction loss at end of
    training.  Injected as ``extra_callbacks`` into ``train_single_fold``.

    After training completes, the Lightning trainer runs ``validate()``.
    This callback captures the resulting metrics so the caller can
    inspect them without parsing CSV logs.
    """

    def __init__(self):
        super().__init__()
        self.val_hsic_cross: Optional[float] = None
        self.val_hsic_self: Optional[float] = None
        self.val_recon: Optional[float] = None
        self.val_x_mae: Optional[float] = None
        self.val_loss: Optional[float] = None
        self.train_hsic_cross: Optional[float] = None
        self.train_recon: Optional[float] = None

    def on_train_epoch_end(self, trainer, pl_module):
        """Capture training-side metrics each epoch (keep latest)."""
        metrics = trainer.callback_metrics
        for key in ["train_hsic_cross", "train_hsic_reg", "train_hsic"]:
            if key in metrics:
                self.train_hsic_cross = float(metrics[key].item())
                break
        for key in ["train_loss_x", "train_recon", "train_loss"]:
            if key in metrics:
                self.train_recon = float(metrics[key].item())
                break

    def on_validation_end(self, trainer, pl_module):
        """Capture validation metrics after the final validate() call."""
        metrics = trainer.callback_metrics

        # Validation HSIC
        for key in ["val_hsic_cross", "val_hsic_reg", "val_hsic"]:
            if key in metrics:
                self.val_hsic_cross = float(metrics[key].item())
                break

        for key in ["val_hsic_self"]:
            if key in metrics:
                self.val_hsic_self = float(metrics[key].item())
                break

        # Validation reconstruction
        for key in ["val_loss_x", "val_recon", "val_loss"]:
            if key in metrics:
                self.val_recon = float(metrics[key].item())
                break

        # Validation MAE
        if "val_x_mae" in metrics:
            self.val_x_mae = float(metrics["val_x_mae"].item())

        # Total val loss
        if "val_loss" in metrics:
            self.val_loss = float(metrics["val_loss"].item())

    def to_dict(self) -> Dict[str, Any]:
        """Return all captured metrics as a dict."""
        return {
            "val_hsic_cross": self.val_hsic_cross,
            "val_hsic_self": self.val_hsic_self,
            "val_recon": self.val_recon,
            "val_x_mae": self.val_x_mae,
            "val_loss": self.val_loss,
            "train_hsic_cross": self.train_hsic_cross,
            "train_recon": self.train_recon,
        }


# =============================================================================
# DAG EXTRACTION (lightweight, in-memory)
# =============================================================================

def _extract_learned_dag(model: pl.LightningModule) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract learned DAG edge probabilities from a trained model (in-memory).

    This is a lightweight operation — it reads sigmoid(phi) directly from
    the model parameters. No forward pass or data loading required.

    For models without phi (e.g. pure ScaledDotSoftmax), returns None
    for those blocks.

    Returns:
        Dict with per-layer keys like "cross_L0", "self_L0", "cross_L1", etc.
        Each maps to a 2D numpy array of edge probabilities in [0, 1], or None.
        Also includes backward-compat keys "cross" and "self" (layer 0 or average).
    """
    from causaliT.evaluation.eval_funs.helpers.eval_lib import extract_phi_from_model

    # Determine architecture type from model class name
    arch_type = type(model).__name__

    phi_dict = extract_phi_from_model(model, arch_type)

    dag = {}

    # Collect per-layer keys
    layer_idx = 0
    while True:
        cross_key = f"decoder_cross_L{layer_idx}"
        self_key = f"decoder_L{layer_idx}"
        has_cross = phi_dict.get(cross_key) is not None
        has_self = phi_dict.get(self_key) is not None
        if not has_cross and not has_self:
            break
        dag[f"cross_L{layer_idx}"] = phi_dict.get(cross_key)
        dag[f"self_L{layer_idx}"] = phi_dict.get(self_key)
        layer_idx += 1

    n_layers = layer_idx

    # If no per-layer keys found, try backward-compat single-layer keys
    if n_layers == 0:
        cross_val = phi_dict.get("decoder_cross") or phi_dict.get("cross")
        self_val = phi_dict.get("decoder") or phi_dict.get("decoder_L0")
        dag["cross_L0"] = cross_val
        dag["self_L0"] = self_val
        n_layers = 1

    # Backward-compat summary keys (layer 0 for single-layer, average for multi)
    if n_layers == 1:
        dag["cross"] = dag.get("cross_L0")
        dag["self"] = dag.get("self_L0")
    else:
        cross_arrays = [dag[f"cross_L{i}"] for i in range(n_layers) if dag.get(f"cross_L{i}") is not None]
        self_arrays = [dag[f"self_L{i}"] for i in range(n_layers) if dag.get(f"self_L{i}") is not None]
        dag["cross"] = np.mean(cross_arrays, axis=0) if cross_arrays else None
        dag["self"] = np.mean(self_arrays, axis=0) if self_arrays else None

    dag["n_layers"] = n_layers

    return dag


# =============================================================================
# LASSO-PATH PLOTS
# =============================================================================

def _plot_lasso_path_edges(
    all_results: List[Dict[str, Any]],
    best_lambda: float,
    cv_dir: Path,
) -> None:
    """
    Plot LASSO-path of individual edge probabilities vs λ_score.

    Creates one row of subplots per decoder layer, each with cross-attention
    (S→X) and self-attention (X→X) panels. For single-layer models, this
    produces the same layout as before (1 row, 2 columns).

    Args:
        all_results:  Per-lambda result dicts (must include "best_fold_dag").
        best_lambda:  Selected λ_score (vertical dashed line).
        cv_dir:       Directory to save the plot.
    """
    import matplotlib.pyplot as plt

    # Collect edge paths from best fold per lambda
    lambdas = []
    dag_data_per_lambda = []

    for r in all_results:
        dag_data = r.get("best_fold_dag", {})
        if not dag_data:
            continue
        lambdas.append(r["lambda_score"])
        dag_data_per_lambda.append(dag_data)

    if not lambdas:
        logger.warning("No DAG data available for LASSO-path edge plot.")
        return

    # Determine number of layers from first available DAG data
    n_layers = dag_data_per_lambda[0].get("n_layers", 1)
    # Also check for per-layer keys
    if n_layers <= 1:
        for dag_data in dag_data_per_lambda:
            layer_idx = 0
            while f"cross_L{layer_idx}" in dag_data or f"self_L{layer_idx}" in dag_data:
                layer_idx += 1
            if layer_idx > n_layers:
                n_layers = layer_idx
    if n_layers == 0:
        n_layers = 1  # Fallback: use summary keys

    # For each layer, collect cross and self edge arrays
    def _get_edges(dag_data, att_type, layer_idx):
        """Get edge array for a given attention type and layer."""
        # Try per-layer key first
        key = f"{att_type}_L{layer_idx}"
        if dag_data.get(key) is not None:
            return np.array(dag_data[key])
        # For single-layer, fall back to summary key
        if layer_idx == 0 and dag_data.get(att_type) is not None:
            return np.array(dag_data[att_type])
        return None

    # Check which panels are needed per layer
    layer_has_cross = []
    layer_has_self = []
    for layer_idx in range(n_layers):
        has_cross = any(_get_edges(d, "cross", layer_idx) is not None for d in dag_data_per_lambda)
        has_self = any(_get_edges(d, "self", layer_idx) is not None for d in dag_data_per_lambda)
        layer_has_cross.append(has_cross)
        layer_has_self.append(has_self)

    n_cols = max(int(any(layer_has_cross)) + int(any(layer_has_self)), 1)
    n_rows = n_layers

    if n_cols == 0 or not any(layer_has_cross + layer_has_self):
        logger.warning("No phi/DAG probabilities found — skipping edge path plot.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    use_symlog = len(lambdas) > 2 and lambdas[-1] / max(lambdas[0], 1e-10) > 20
    min_lambda = min(lambdas)

    for layer_idx in range(n_layers):
        col_idx = 0

        # --- Cross-attention subplot ---
        if any(layer_has_cross):
            ax = axes[layer_idx, col_idx]
            col_idx += 1

            if layer_has_cross[layer_idx]:
                ref = next(_get_edges(d, "cross", layer_idx) for d in dag_data_per_lambda
                           if _get_edges(d, "cross", layer_idx) is not None)
                n_targets, n_sources = ref.shape

                for i in range(n_targets):
                    for j in range(n_sources):
                        vals, lams = [], []
                        for k, d in enumerate(dag_data_per_lambda):
                            e = _get_edges(d, "cross", layer_idx)
                            if e is not None:
                                vals.append(e[i, j])
                                lams.append(lambdas[k])
                        if vals:
                            ax.plot(lams, vals, marker="o", markersize=3,
                                    label=f"X{i+1}←S{j+1}", linewidth=1.2)

                ax.axvline(best_lambda, color="green", linestyle="--", alpha=0.7,
                           label=f"λ*={best_lambda}")
                ax.set_ylabel("Edge probability")
                layer_label = f" (Layer {layer_idx})" if n_layers > 1 else ""
                ax.set_title(f"Cross-attention (S→X){layer_label}")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=7, ncol=max(1, n_sources), loc="best")
                ax.grid(True, alpha=0.3)
                if use_symlog:
                    ax.set_xscale("symlog", linthresh=min(l for l in lambdas if l > 0) if any(l > 0 for l in lambdas) else 0.001)
                ax.set_xlim(left=min_lambda)
            else:
                ax.set_visible(False)

            ax.set_xlabel("λ_score")

        # --- Self-attention subplot ---
        if any(layer_has_self):
            ax = axes[layer_idx, col_idx]

            if layer_has_self[layer_idx]:
                ref = next(_get_edges(d, "self", layer_idx) for d in dag_data_per_lambda
                           if _get_edges(d, "self", layer_idx) is not None)
                n_nodes = ref.shape[0]

                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i == j:
                            continue
                        vals, lams = [], []
                        for k, d in enumerate(dag_data_per_lambda):
                            e = _get_edges(d, "self", layer_idx)
                            if e is not None:
                                vals.append(e[i, j])
                                lams.append(lambdas[k])
                        if vals:
                            ax.plot(lams, vals, marker="o", markersize=3,
                                    label=f"X{i+1}←X{j+1}", linewidth=1.2)

                ax.axvline(best_lambda, color="green", linestyle="--", alpha=0.7,
                           label=f"λ*={best_lambda}")
                ax.set_ylabel("Edge probability")
                layer_label = f" (Layer {layer_idx})" if n_layers > 1 else ""
                ax.set_title(f"Self-attention (X→X){layer_label}")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=7, ncol=max(1, n_nodes - 1), loc="best")
                ax.grid(True, alpha=0.3)
                if use_symlog:
                    ax.set_xscale("symlog", linthresh=min(l for l in lambdas if l > 0) if any(l > 0 for l in lambdas) else 0.001)
                ax.set_xlim(left=min_lambda)
            else:
                ax.set_visible(False)

            ax.set_xlabel("λ_score")

    plt.suptitle("LASSO Path: Edge Probabilities vs λ_score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cv_dir / "lasso_path_edges.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved lasso_path_edges.png to {cv_dir}")


def _plot_lasso_path_metrics(
    all_results: List[Dict[str, Any]],
    best_lambda: float,
    cv_dir: Path,
) -> None:
    """
    Plot validation HSIC and reconstruction loss vs λ_score.

    Two subplots: val HSIC (left) and val recon (right), each with
    mean ± std error bands across folds.

    Args:
        all_results:  Per-lambda result dicts.
        best_lambda:  Selected λ_score (vertical dashed line).
        cv_dir:       Directory to save the plot.
    """
    import matplotlib.pyplot as plt

    lambdas = [r["lambda_score"] for r in all_results]
    hsic_means = [r.get("mean_val_hsic") for r in all_results]
    hsic_stds = [r.get("std_val_hsic") for r in all_results]
    recon_means = [r.get("mean_val_recon") for r in all_results]
    recon_stds = [r.get("std_val_recon") for r in all_results]

    has_hsic = any(v is not None for v in hsic_means)
    has_recon = any(v is not None for v in recon_means)
    n_panels = int(has_hsic) + int(has_recon)

    if n_panels == 0:
        logger.warning("No metrics available for LASSO-path metrics plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    ax_idx = 0

    # Use symlog if lambda range is wide
    use_symlog = len(lambdas) > 2 and lambdas[-1] / max(lambdas[0], 1e-10) > 20
    min_lambda = min(lambdas)

    # --- HSIC subplot ---
    if has_hsic:
        ax = axes[0, ax_idx]
        ax_idx += 1
        lams_h = [l for l, v in zip(lambdas, hsic_means) if v is not None]
        means_h = [v for v in hsic_means if v is not None]
        stds_h = [s if s is not None else 0.0 for s, v in zip(hsic_stds, hsic_means) if v is not None]

        ax.plot(lams_h, means_h, "o-", color="tab:blue", linewidth=1.5, label="Mean val HSIC")
        ax.fill_between(
            lams_h,
            [m - s for m, s in zip(means_h, stds_h)],
            [m + s for m, s in zip(means_h, stds_h)],
            alpha=0.2, color="tab:blue",
        )
        ax.axvline(best_lambda, color="green", linestyle="--", alpha=0.7,
                   label=f"λ*={best_lambda}")
        ax.set_xlabel("λ_score")
        ax.set_ylabel("Validation HSIC")
        ax.set_title("Val HSIC vs λ_score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_symlog:
            ax.set_xscale("symlog", linthresh=min(l for l in lams_h if l > 0) if any(l > 0 for l in lams_h) else 0.001)
        ax.set_xlim(left=min_lambda)

    # --- Reconstruction subplot ---
    if has_recon:
        ax = axes[0, ax_idx]
        lams_r = [l for l, v in zip(lambdas, recon_means) if v is not None]
        means_r = [v for v in recon_means if v is not None]
        stds_r = [s if s is not None else 0.0 for s, v in zip(recon_stds, recon_means) if v is not None]

        ax.plot(lams_r, means_r, "o-", color="tab:orange", linewidth=1.5, label="Mean val recon")
        ax.fill_between(
            lams_r,
            [m - s for m, s in zip(means_r, stds_r)],
            [m + s for m, s in zip(means_r, stds_r)],
            alpha=0.2, color="tab:orange",
        )
        ax.axvline(best_lambda, color="green", linestyle="--", alpha=0.7,
                   label=f"λ*={best_lambda}")
        ax.set_xlabel("λ_score")
        ax.set_ylabel("Validation Recon Loss")
        ax.set_title("Val Reconstruction vs λ_score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_symlog:
            ax.set_xscale("symlog", linthresh=min(l for l in lams_r if l > 0) if any(l > 0 for l in lams_r) else 0.001)
        ax.set_xlim(left=min_lambda)

    plt.suptitle("Score Sparsity CV: Validation Metrics vs λ_score", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cv_dir / "lasso_path_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved lasso_path_metrics.png to {cv_dir}")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_cv_config(
    config: dict,
    lambda_score: float,
    epochs: int,
    k_fold: int,
) -> dict:
    """
    Build a config for one score-sparsity CV candidate.

    Changes applied:
    - ``training.lambda_cross_score_sparse`` = lambda_score
    - ``training.lambda_self_score_sparse``  = ratio · lambda_score
      where ``ratio = staged_training.lambda_self_to_cross_score_ratio``
      (default ``1.0`` for full backward compatibility). For attention
      stacks where the self branch carries a structural double-sigmoid
      handicap (Toeplitz; see ``docs/ATTENTION_MAGNITUDE_BALANCE.md``),
      a value of ``0.5`` calibrates the L1 weight to the per-branch
      magnitude.
    - ``training.max_epochs``                = epochs
    - ``training.k_fold``                    = k_fold
    - ``training.use_hsic_annealing``        = False  (constant HSIC during CV)
    - ``training.save_ckpt_every_n_epochs``  = epochs  (only last ckpt)

    Args:
        config:       Base configuration dict.
        lambda_score: Score sparsity lambda to test (interpreted as the
                      cross-attention λ).
        epochs:       Number of CV training epochs.
        k_fold:       Number of cross-validation folds.

    Returns:
        A deep copy of ``config`` with the above overrides applied.
    """
    config_cv = copy.deepcopy(config)
    ratio = float(
        config_cv.get("staged_training", {})
                 .get("lambda_self_to_cross_score_ratio", 1.0)
    )
    config_cv["training"]["lambda_cross_score_sparse"] = float(lambda_score)
    config_cv["training"]["lambda_self_score_sparse"] = float(ratio * lambda_score)
    config_cv["training"]["max_epochs"] = int(epochs)
    config_cv["training"]["k_fold"] = int(k_fold)
    config_cv["training"]["use_hsic_annealing"] = False
    config_cv["training"].setdefault("save_ckpt_every_n_epochs", epochs)
    # Override to avoid running post-training evaluations during CV
    config_cv.setdefault("evaluation", {})["functions"] = []
    return config_cv



def _load_weights_from_checkpoint(
    model: pl.LightningModule,
    checkpoint_path: str,
) -> None:
    """Load model weights from a Lightning checkpoint file (strict=False)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)


def _run_cv_for_single_lambda(
    config: dict,
    config_cv: dict,
    data_dir: str,
    lambda_dir: Path,
    lambda_score: float,
    starting_checkpoint: Optional[str],
    seed: int,
    cluster: bool,
) -> Dict[str, Any]:
    """
    Run k-fold CV for a single λ_score candidate.

    For each fold:
    1. Create a fresh model, load starting checkpoint if available.
    2. Train for ``score_sparsity_cv_epochs``.
    3. Capture validation HSIC and reconstruction loss.

    Args:
        config:               Original config (for model creation).
        config_cv:            CV config (with lambda_score, epochs, k_fold set).
        data_dir:             Data directory.
        lambda_dir:           Directory for this lambda's outputs.
        lambda_score:         The λ_score value being tested.
        starting_checkpoint:  Checkpoint to initialize from (causal_init or calibration).
        seed:                 Random seed.
        cluster:              Whether running on cluster.

    Returns:
        Dict with per-fold and aggregated metrics.
    """
    from causaliT.training.trainer import (
        create_model_instance,
        get_dataloader,
        train_single_fold,
        _make_fold_splits,
    )

    lambda_dir.mkdir(exist_ok=True, parents=True)

    seed_everything(seed)

    dm = get_dataloader(config_cv, data_dir, cluster=cluster, seed=seed)
    dm.prepare_data()

    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config_cv, dm, seed, data_dir=data_dir
    )

    k_folds = len(fold_splits)
    fold_results = []

    for fold, (train_local_idx, val_local_idx) in enumerate(fold_splits):
        # Reset seed for reproducible model initialization
        seed_everything(seed)
        model = create_model_instance(config_cv, data_dir)

        # Load starting checkpoint weights
        if starting_checkpoint is not None and os.path.exists(starting_checkpoint):
            _load_weights_from_checkpoint(model, starting_checkpoint)

        # Create the validation metrics tracker callback
        val_tracker = ValidationMetricsTracker()

        if not cluster:
            print(f"      Fold {fold + 1}/{k_folds}")

        train_single_fold(
            config=config_cv,
            model=model,
            dm=dm,
            fold=fold,
            train_local_idx=train_local_idx,
            val_local_idx=val_local_idx,
            test_idx=test_idx,
            train_val_idx=train_val_idx,
            save_dir=str(lambda_dir),
            trainable_params=0,
            cluster=cluster,
            extra_callbacks=[val_tracker],
        )

        fold_metrics = val_tracker.to_dict()
        fold_metrics["fold"] = fold

        # Extract learned DAG from the trained model (lightweight, no forward pass)
        try:
            dag = _extract_learned_dag(model)
            # Save backward-compat summary keys
            fold_metrics["dag_cross"] = dag["cross"].tolist() if dag.get("cross") is not None else None
            fold_metrics["dag_self"] = dag["self"].tolist() if dag.get("self") is not None else None
            # Save per-layer keys for multi-layer plotting
            fold_metrics["dag_n_layers"] = dag.get("n_layers", 1)
            for key, val in dag.items():
                if key.startswith(("cross_L", "self_L")) and val is not None:
                    fold_metrics[f"dag_{key}"] = val.tolist() if hasattr(val, 'tolist') else val
        except Exception as e:
            logger.debug(f"Could not extract DAG for fold {fold}: {e}")
            fold_metrics["dag_cross"] = None
            fold_metrics["dag_self"] = None

        fold_results.append(fold_metrics)

        if not cluster:
            hsic_str = f"{fold_metrics['val_hsic_cross']:.6f}" if fold_metrics["val_hsic_cross"] is not None else "N/A"
            recon_str = f"{fold_metrics['val_recon']:.6f}" if fold_metrics["val_recon"] is not None else "N/A"
            print(f"        val_hsic={hsic_str}, val_recon={recon_str}")

    # Aggregate across folds
    hsic_values = [r["val_hsic_cross"] for r in fold_results if r["val_hsic_cross"] is not None]
    recon_values = [r["val_recon"] for r in fold_results if r["val_recon"] is not None]
    mae_values = [r["val_x_mae"] for r in fold_results if r["val_x_mae"] is not None]

    result = {
        "lambda_score": lambda_score,
        "k_folds": k_folds,
        "per_fold": fold_results,
        # HSIC aggregation
        "mean_val_hsic": float(np.mean(hsic_values)) if hsic_values else None,
        "std_val_hsic": float(np.std(hsic_values)) if hsic_values else None,
        "min_val_hsic": float(np.min(hsic_values)) if hsic_values else None,
        "max_val_hsic": float(np.max(hsic_values)) if hsic_values else None,
        # Reconstruction aggregation
        "mean_val_recon": float(np.mean(recon_values)) if recon_values else None,
        "std_val_recon": float(np.std(recon_values)) if recon_values else None,
        "min_val_recon": float(np.min(recon_values)) if recon_values else None,
        "max_val_recon": float(np.max(recon_values)) if recon_values else None,
        # MAE aggregation
        "mean_val_x_mae": float(np.mean(mae_values)) if mae_values else None,
        "std_val_x_mae": float(np.std(mae_values)) if mae_values else None,
    }

    # Pick best fold's DAG for LASSO-path edge plots
    # Best fold = lowest val_hsic_cross (if available), else lowest val_recon
    folds_with_dag = [r for r in fold_results if r.get("dag_cross") is not None or r.get("dag_self") is not None]
    if folds_with_dag:
        folds_with_hsic = [r for r in folds_with_dag if r.get("val_hsic_cross") is not None]
        folds_with_recon = [r for r in folds_with_dag if r.get("val_recon") is not None]
        if folds_with_hsic:
            best_fold = min(folds_with_hsic, key=lambda r: r["val_hsic_cross"])
        elif folds_with_recon:
            best_fold = min(folds_with_recon, key=lambda r: r["val_recon"])
        else:
            best_fold = folds_with_dag[0]
        best_fold_dag = {
            "cross": best_fold.get("dag_cross"),
            "self": best_fold.get("dag_self"),
            "fold": best_fold["fold"],
            "n_layers": best_fold.get("dag_n_layers", 1),
        }
        # Include per-layer data
        for key in best_fold:
            if key.startswith("dag_cross_L") or key.startswith("dag_self_L"):
                # Strip "dag_" prefix to get "cross_L0", "self_L0", etc.
                plot_key = key[4:]  # "dag_cross_L0" -> "cross_L0"
                best_fold_dag[plot_key] = best_fold[key]
        result["best_fold_dag"] = best_fold_dag
    else:
        result["best_fold_dag"] = {}

    # Save per-lambda summary
    with open(lambda_dir / "cv_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _select_best_lambda(
    all_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Select best λ_score by two criteria: min HSIC and min reconstruction.

    Args:
        all_results: List of per-lambda result dicts from _run_cv_for_single_lambda.

    Returns:
        (selection_by_hsic, selection_by_recon) — each a dict with:
            lambda_score, mean_val_hsic/recon, fold_index (for best fold).
    """
    # Selection by HSIC
    valid_hsic = [r for r in all_results if r["mean_val_hsic"] is not None]
    if valid_hsic:
        best_hsic = min(valid_hsic, key=lambda r: r["mean_val_hsic"])
        selection_by_hsic = {
            "lambda_score": best_hsic["lambda_score"],
            "mean_val_hsic": best_hsic["mean_val_hsic"],
            "std_val_hsic": best_hsic["std_val_hsic"],
            "mean_val_recon": best_hsic["mean_val_recon"],
            "std_val_recon": best_hsic["std_val_recon"],
        }
    else:
        selection_by_hsic = {
            "lambda_score": None,
            "note": "No valid HSIC values across any lambda candidate.",
        }

    # Selection by reconstruction loss
    valid_recon = [r for r in all_results if r["mean_val_recon"] is not None]
    if valid_recon:
        best_recon = min(valid_recon, key=lambda r: r["mean_val_recon"])
        selection_by_recon = {
            "lambda_score": best_recon["lambda_score"],
            "mean_val_recon": best_recon["mean_val_recon"],
            "std_val_recon": best_recon["std_val_recon"],
            "mean_val_hsic": best_recon["mean_val_hsic"],
            "std_val_hsic": best_recon["std_val_hsic"],
        }
    else:
        selection_by_recon = {
            "lambda_score": None,
            "note": "No valid reconstruction values across any lambda candidate.",
        }

    return selection_by_hsic, selection_by_recon


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_score_sparsity_cv(
    config: dict,
    data_dir: str,
    save_dir: str,
    starting_checkpoint: Optional[str] = None,
    seed: int = 42,
    cluster: bool = False,
) -> float:
    """
    Run score sparsity cross-validation to select optimal λ_score.

    This is the main entry point for Stage 2 (Score Sparsity CV).

    Process:
    1. Read lambda candidates from ``staged_training.score_sparsity_lambda_candidates``.
    2. For each lambda candidate, run k-fold CV from the starting checkpoint.
    3. Select best lambda by lowest mean val HSIC (primary) and lowest mean
       val recon (secondary — always reported for comparison).
    4. Save comprehensive summary.

    Args:
        config:               Configuration dict with ``staged_training`` section.
        data_dir:             Root data directory.
        save_dir:             Parent save directory; a ``score_sparsity_cv/`` subfolder
                              is created here.
        starting_checkpoint:  Checkpoint to start each fold from (typically from
                              causal_init or calibration).
        seed:                 Random seed.
        cluster:              Whether running on cluster.

    Returns:
        The selected best λ_score (float), chosen by lowest mean val HSIC.
        Falls back to lowest mean val recon if HSIC is not available.
    """
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset

    cv_dir = Path(save_dir) / "score_sparsity_cv"
    cv_dir.mkdir(exist_ok=True, parents=True)

    staged_config = config.get("staged_training", {})

    # Read CV parameters
    lambda_candidates = staged_config.get(
        "score_sparsity_lambda_candidates",
        [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
    )
    cv_epochs = int(staged_config.get("score_sparsity_cv_epochs", 20))
    cv_folds = int(staged_config.get("score_sparsity_cv_folds", 5))
    selection_rule = staged_config.get("score_sparsity_selection_rule", "min_hsic")

    # Ensure lambda candidates is a list of floats
    lambda_candidates = [float(l) for l in lambda_candidates]

    if not cluster:
        print(f"\n{'='*70}")
        print("SCORE SPARSITY CV: Selecting λ_score via cross-validation")
        print(f"{'='*70}")
        print(f"  Lambda candidates: {lambda_candidates}")
        print(f"  CV folds: {cv_folds}")
        print(f"  CV epochs per fold: {cv_epochs}")
        print(f"  Starting checkpoint: {starting_checkpoint or 'fresh'}")
        print(f"  Total training runs: {len(lambda_candidates)} × {cv_folds} = {len(lambda_candidates) * cv_folds}")

    # Ensure sequence lengths are populated
    config = populate_seq_lengths_from_dataset(config, data_dir)

    # =========================================================================
    # RUN CV FOR EACH LAMBDA CANDIDATE
    # =========================================================================
    all_results = []

    for i, lambda_score in enumerate(lambda_candidates):
        if not cluster:
            print(f"\n    [{i+1}/{len(lambda_candidates)}] λ_score = {lambda_score}")

        lambda_dir = cv_dir / f"lambda_{lambda_score:.4f}"

        config_cv = _build_cv_config(
            config=config,
            lambda_score=lambda_score,
            epochs=cv_epochs,
            k_fold=cv_folds,
        )

        result = _run_cv_for_single_lambda(
            config=config,
            config_cv=config_cv,
            data_dir=data_dir,
            lambda_dir=lambda_dir,
            lambda_score=lambda_score,
            starting_checkpoint=starting_checkpoint,
            seed=seed,
            cluster=cluster,
        )

        all_results.append(result)

        if not cluster:
            hsic_str = f"{result['mean_val_hsic']:.6f}" if result["mean_val_hsic"] is not None else "N/A"
            recon_str = f"{result['mean_val_recon']:.6f}" if result["mean_val_recon"] is not None else "N/A"
            print(f"      → mean_val_hsic={hsic_str}, mean_val_recon={recon_str}")

    # =========================================================================
    # SELECT BEST LAMBDA
    # =========================================================================
    selection_by_hsic, selection_by_recon = _select_best_lambda(all_results)

    # Primary selection based on configured rule
    if selection_rule == "min_recon":
        # Baseline models: select by reconstruction loss
        if selection_by_recon.get("lambda_score") is not None:
            best_lambda = selection_by_recon["lambda_score"]
            primary_criterion = "min_val_recon"
        elif selection_by_hsic.get("lambda_score") is not None:
            best_lambda = selection_by_hsic["lambda_score"]
            primary_criterion = "min_val_hsic (fallback, no valid recon)"
        else:
            best_lambda = 0.0
            primary_criterion = "fallback_no_valid_metrics"
    else:
        # Causal models (default): select by HSIC
        if selection_by_hsic.get("lambda_score") is not None:
            best_lambda = selection_by_hsic["lambda_score"]
            primary_criterion = "min_val_hsic"
        elif selection_by_recon.get("lambda_score") is not None:
            best_lambda = selection_by_recon["lambda_score"]
            primary_criterion = "min_val_recon (fallback, no valid HSIC)"
        else:
            best_lambda = 0.0
            primary_criterion = "fallback_no_valid_metrics"

    if primary_criterion == "fallback_no_valid_metrics":
        logger.warning(
            "Score sparsity CV: no valid metrics found. "
            "Falling back to lambda_score=0.0."
        )

    # =========================================================================
    # SAVE COMPREHENSIVE SUMMARY
    # =========================================================================
    summary = {
        "best_lambda_score": best_lambda,
        "primary_criterion": primary_criterion,
        "selection_by_hsic": selection_by_hsic,
        "selection_by_recon": selection_by_recon,
        "cv_config": {
            "lambda_candidates": lambda_candidates,
            "cv_epochs": cv_epochs,
            "cv_folds": cv_folds,
            "seed": seed,
            "starting_checkpoint": starting_checkpoint,
        },
        "per_lambda_results": all_results,
    }

    summary_path = cv_dir / "score_sparsity_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # =========================================================================
    # LASSO-PATH PLOTS
    # =========================================================================
    try:
        _plot_lasso_path_edges(all_results, best_lambda, cv_dir)
        _plot_lasso_path_metrics(all_results, best_lambda, cv_dir)
    except Exception as e:
        logger.warning(f"Could not generate LASSO-path plots: {e}")
        if not cluster:
            print(f"  Warning: LASSO-path plots failed: {e}")

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    if not cluster:
        print(f"\n{'='*70}")
        print("SCORE SPARSITY CV COMPLETE")
        print(f"{'='*70}")

        # Summary table
        print(f"\n  {'λ_score':>10}  {'mean_val_hsic':>14}  {'std_val_hsic':>13}  {'mean_val_recon':>15}  {'std_val_recon':>14}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*13}  {'-'*15}  {'-'*14}")
        for r in all_results:
            ls = f"{r['lambda_score']:.4f}"
            hsic_m = f"{r['mean_val_hsic']:.6f}" if r["mean_val_hsic"] is not None else "N/A"
            hsic_s = f"{r['std_val_hsic']:.6f}" if r["std_val_hsic"] is not None else "N/A"
            recon_m = f"{r['mean_val_recon']:.6f}" if r["mean_val_recon"] is not None else "N/A"
            recon_s = f"{r['std_val_recon']:.6f}" if r["std_val_recon"] is not None else "N/A"
            marker = " ←" if r["lambda_score"] == best_lambda else ""
            print(f"  {ls:>10}  {hsic_m:>14}  {hsic_s:>13}  {recon_m:>15}  {recon_s:>14}{marker}")

        print(f"\n  Selected λ_score = {best_lambda} (by {primary_criterion})")

        print(f"\n  Best by HSIC:  λ={selection_by_hsic.get('lambda_score', 'N/A')}")
        if selection_by_hsic.get("mean_val_hsic") is not None:
            print(f"    mean_val_hsic = {selection_by_hsic['mean_val_hsic']:.6f} ± {selection_by_hsic['std_val_hsic']:.6f}")
        print(f"  Best by Recon: λ={selection_by_recon.get('lambda_score', 'N/A')}")
        if selection_by_recon.get("mean_val_recon") is not None:
            print(f"    mean_val_recon = {selection_by_recon['mean_val_recon']:.6f} ± {selection_by_recon['std_val_recon']:.6f}")

        print(f"\n  Summary saved to: {summary_path}")
        print(f"{'='*70}\n")

    return best_lambda
