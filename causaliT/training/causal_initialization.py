"""
Causal Initialization Stage: Initialize model toward causal structure.

This module implements Stage 1 of the staged training pipeline.
The goal is to pre-train the model with an HSIC-dominated loss function,
initializing attention patterns and weights toward the causal structure
before standard fitting-focused training begins.

Key insight: At random initialization, the model has no preference for any
particular causal structure.  By training first with HSIC >> Recon loss,
we break this symmetry and guide the model toward the true causal mechanism.

Features:
- Multi-seed selection: Try N random seeds and pick the one with lowest HSIC.
  Different initial weights lead to different local minima in HSIC landscape;
  this evolutionary-style selection finds favorable starting points.
- Pre/post DAG evaluation: Compute soft Hamming distance and phi decisiveness
  before and after initialization for diagnostic comparison.
- HSIC variance analysis: Log HSIC across all seeds to quantify how sensitive
  the initialization is to the random seed.

Process:
1. For each seed candidate:
   a. Create fresh model and load calibration weights (if any)
   b. Apply boosted λ_hsic = λ_hsic_calibrated * boost_factor
   c. Train for ``causal_init_epochs`` using ``train_single_fold``
   d. Evaluate final HSIC on trained model
2. Select the seed with lowest final HSIC cross value
3. Compute pre/post DAG metrics for diagnostic logging
4. Save comprehensive summary with seed variance analysis

Usage:
    checkpoint_path = run_causal_initialization(config, data_dir, save_dir)
"""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf

from causaliT.training.calibration import _find_last_checkpoint

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS LOGGER CALLBACK
# =============================================================================

class CausalInitProgressLogger(Callback):
    """
    Callback to log progress during causal initialization.

    Injected as ``extra_callbacks`` into ``train_single_fold``.  Tracks HSIC
    values and reconstruction loss so callers can inspect the trajectory.
    """

    def __init__(self):
        super().__init__()
        self.hsic_values: List[float] = []
        self.recon_values: List[float] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics

        hsic = None
        for key in ["train_hsic_cross", "train_hsic_reg", "train_hsic"]:
            if key in metrics:
                hsic = metrics[key].item()
                break

        recon = None
        for key in ["train_loss_x", "train_recon", "train_loss"]:
            if key in metrics:
                recon = metrics[key].item()
                break

        if hsic is not None:
            self.hsic_values.append(hsic)
        if recon is not None:
            self.recon_values.append(recon)

        epoch = trainer.current_epoch + 1
        hsic_str = f"{hsic:.6f}" if hsic is not None else "N/A"
        recon_str = f"{recon:.6f}" if recon is not None else "N/A"
        print(f"    Epoch {epoch}: HSIC = {hsic_str}, Recon = {recon_str}")


class DAGMetricsCallback(Callback):
    """
    Callback that captures DAG metrics at the start and end of training.

    Since ``get_dag_probabilities()`` requires populated internal attention
    state (running averages, buffers), we cannot call it on a fresh model
    before any forward pass.  Instead we capture:
    - **pre_metrics** after epoch 0 (earliest point with valid attention state)
    - **post_metrics** after training ends (``on_train_end``)

    This runs *inside* the trainer lifecycle so it is compatible with
    ``deterministic=True`` and does not contaminate RNG state.

    Args:
        config:   Configuration dictionary (for architecture type + dataset).
        data_dir: Root data directory (for loading true DAG masks).
    """

    def __init__(self, config: dict, data_dir: str):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.pre_metrics: Optional[Dict[str, Any]] = None
        self.post_metrics: Optional[Dict[str, Any]] = None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        # Capture after the first epoch = earliest valid attention state
        if trainer.current_epoch == 0 and self.pre_metrics is None:
            try:
                self.pre_metrics = evaluate_dag_from_model(
                    pl_module, self.config, self.data_dir
                )
            except Exception as e:
                logger.debug(f"DAGMetricsCallback pre_metrics failed: {e}")

    def on_train_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        try:
            self.post_metrics = evaluate_dag_from_model(
                pl_module, self.config, self.data_dir
            )
        except Exception as e:
            logger.debug(f"DAGMetricsCallback post_metrics failed: {e}")


# =============================================================================
# DAG EVALUATION UTILITY (Lightweight, in-memory)
# =============================================================================

def evaluate_dag_from_model(
    model: pl.LightningModule,
    config: dict,
    data_dir: str,
) -> Dict[str, Any]:
    """
    Lightweight DAG structure evaluation from a model in memory.

    Extracts phi tensors from the model's attention layers and computes:
    - Phi decisiveness (how far edge probabilities are from 0.5)
    - Soft Hamming distance to true DAG (if ground truth available)
    - Phi statistics (mean, std, min, max of sigmoid(phi))

    This is designed to be called quickly before/after causal initialization
    without the full eval_attention_scores ceremony.

    Args:
        model:    LightningModule with a `.model` attribute containing attention layers.
        config:   Configuration dictionary (needs data.dataset for ground truth lookup).
        data_dir: Root data directory for loading true DAG masks.

    Returns:
        Dict with keys:
            - phi_cross: sigmoid(phi) for cross-attention as numpy array (or None)
            - phi_self: sigmoid(phi) for self-attention as numpy array (or None)
            - phi_cross_decisiveness: mean |sigmoid(phi) - 0.5| for cross (higher = more decisive)
            - phi_self_decisiveness: mean |sigmoid(phi) - 0.5| for self (higher = more decisive)
            - phi_cross_stats: {mean, std, min, max} of sigmoid(phi) cross
            - phi_self_stats: {mean, std, min, max} of sigmoid(phi) self
            - soft_hamming_cross: soft Hamming to true DAG for cross (or None)
            - soft_hamming_self: soft Hamming to true DAG for self (or None)
    """
    from causaliT.evaluation.eval_funs.eval_lib import (
        extract_phi_from_model,
        get_architecture_type,
    )

    metrics = {}

    model.eval()

    # Reuse the proven extraction from eval_lib (handles all architectures).
    # extract_phi_from_model uses get_dag_probabilities() internally,
    # returning numpy arrays already in [0, 1].
    # NOTE: This requires that internal attention state has been populated
    # by at least one forward pass.  Use DAGMetricsCallback (not standalone
    # calls) to ensure this is called inside the trainer lifecycle.
    architecture_type = get_architecture_type(config)
    phi_dict = extract_phi_from_model(model, architecture_type)

    # Standard key mapping (eval_lib handles architecture-specific keys)
    prob_cross = phi_dict.get("decoder_cross") or phi_dict.get("cross")
    prob_self = phi_dict.get("decoder")

    # Fallback: when no explicit DAG parameterization (dag_mask=None),
    # use batch_att_mean from the last forward pass.  In SVFA mode the
    # structure embedding deterministically controls attention, so the
    # mean attention scores *are* the soft DAG.
    if prob_cross is None:
        prob_cross = _get_batch_att_mean_from_model(model, "cross")
    if prob_self is None:
        prob_self = _get_batch_att_mean_from_model(model, "self")

    # Process cross-attention DAG
    if prob_cross is not None:
        metrics["phi_cross"] = prob_cross
        metrics["phi_cross_decisiveness"] = float(np.mean(np.abs(prob_cross - 0.5)))
        metrics["phi_cross_stats"] = {
            "mean": float(np.mean(prob_cross)),
            "std": float(np.std(prob_cross)),
            "min": float(np.min(prob_cross)),
            "max": float(np.max(prob_cross)),
        }
    else:
        metrics["phi_cross"] = None
        metrics["phi_cross_decisiveness"] = None
        metrics["phi_cross_stats"] = None

    # Process self-attention DAG
    if prob_self is not None:
        metrics["phi_self"] = prob_self
        metrics["phi_self_decisiveness"] = float(np.mean(np.abs(prob_self - 0.5)))
        metrics["phi_self_stats"] = {
            "mean": float(np.mean(prob_self)),
            "std": float(np.std(prob_self)),
            "min": float(np.min(prob_self)),
            "max": float(np.max(prob_self)),
        }
    else:
        metrics["phi_self"] = None
        metrics["phi_self_decisiveness"] = None
        metrics["phi_self_stats"] = None

    # Load true DAG masks if available (informational only, NOT used for selection)
    dataset_name = config.get("data", {}).get("dataset")
    if dataset_name and data_dir:
        try:
            from causaliT.evaluation.eval_funs.eval_utils import (
                _compute_soft_hamming,
                _load_true_dag_mask,
            )
            datadir_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data",
            )
            # If data_dir is an actual path (not relative), use it directly
            if os.path.isdir(data_dir):
                datadir_path = data_dir

            # Cross-attention: S → X
            true_cross = _load_true_dag_mask(datadir_path, dataset_name, "dec_cross")
            if true_cross is not None and metrics["phi_cross"] is not None:
                if metrics["phi_cross"].shape == true_cross.shape:
                    metrics["soft_hamming_cross"] = float(
                        _compute_soft_hamming(metrics["phi_cross"], true_cross)
                    )
                else:
                    metrics["soft_hamming_cross"] = None
            else:
                metrics["soft_hamming_cross"] = None

            # Self-attention: X → X
            true_self = _load_true_dag_mask(datadir_path, dataset_name, "dec_self")
            if true_self is not None and metrics["phi_self"] is not None:
                if metrics["phi_self"].shape == true_self.shape:
                    metrics["soft_hamming_self"] = float(
                        _compute_soft_hamming(metrics["phi_self"], true_self)
                    )
                else:
                    metrics["soft_hamming_self"] = None
            else:
                metrics["soft_hamming_self"] = None

        except Exception as e:
            logger.warning(f"Could not compute soft Hamming (non-critical): {e}")
            metrics["soft_hamming_cross"] = None
            metrics["soft_hamming_self"] = None
    else:
        metrics["soft_hamming_cross"] = None
        metrics["soft_hamming_self"] = None

    return metrics


def _format_dag_metrics(metrics: Dict[str, Any], label: str) -> str:
    """Format DAG metrics for console printing."""
    lines = [f"  {label}:"]
    if metrics.get("phi_cross_decisiveness") is not None:
        lines.append(f"    Cross-att phi decisiveness: {metrics['phi_cross_decisiveness']:.4f}")
    if metrics.get("phi_self_decisiveness") is not None:
        lines.append(f"    Self-att  phi decisiveness: {metrics['phi_self_decisiveness']:.4f}")
    if metrics.get("soft_hamming_cross") is not None:
        lines.append(f"    Soft Hamming (cross, S→X):  {metrics['soft_hamming_cross']:.4f}")
    if metrics.get("soft_hamming_self") is not None:
        lines.append(f"    Soft Hamming (self, X→X):   {metrics['soft_hamming_self']:.4f}")
    if len(lines) == 1:
        lines.append("    (no phi available)")
    return "\n".join(lines)


def _metrics_to_serializable(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert DAG metrics dict to JSON-serializable form (drop numpy arrays)."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            continue  # skip large arrays
        elif isinstance(v, dict):
            out[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif v is None:
            out[k] = None
        else:
            out[k] = v
    return out


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _get_batch_att_mean_from_model(
    model: pl.LightningModule,
    attention_type: str,
) -> Optional[np.ndarray]:
    """
    Extract ``batch_att_mean`` from a model's inner attention layer.

    This is the fallback for attention types without explicit DAG
    parameterization (``dag_mask=None``).  In SVFA mode the structure
    embedding deterministically controls attention, so the batch-averaged
    attention scores *are* the soft DAG.

    ``batch_att_mean`` is set by ``CausalCrossAttention``, ``LieAttention``,
    and ``PhiSoftMax`` during every forward pass.  It is only available
    after at least one forward pass (i.e. inside the trainer lifecycle).

    Values are normalized to [0, 1] via min-max scaling for comparability
    with the binary ground-truth DAG.

    Args:
        model:          LightningModule with ``.model`` attribute.
        attention_type: ``"cross"`` or ``"self"``.

    Returns:
        numpy array in [0, 1], or None if not available.
    """
    try:
        inner_model = model.model

        # Navigate to the inner attention module
        if hasattr(inner_model, "decoder"):
            decoder = inner_model.decoder
        elif hasattr(inner_model, "decoder1"):
            decoder = inner_model.decoder1
        else:
            return None

        layer = decoder.layers[0]

        if attention_type == "cross":
            inner = layer.global_cross_attention.inner_attention
        else:
            inner = layer.global_self_attention.inner_attention

        att_mean = getattr(inner, "batch_att_mean", None)
        if att_mean is None:
            return None

        arr = att_mean.detach().cpu().numpy()

        # Normalize to [0, 1] via min-max (attention scores may be outside [0,1])
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 1e-8:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.full_like(arr, 0.5)

        return arr

    except (AttributeError, IndexError) as e:
        logger.debug(f"Could not extract batch_att_mean ({attention_type}): {e}")
        return None


def _load_weights_from_checkpoint(
    model: pl.LightningModule,
    checkpoint_path: str,
) -> None:
    """
    Load model weights from a Lightning checkpoint file.

    Uses ``strict=False`` so that minor architecture differences do not
    prevent loading.  Compatible with PyTorch 2.6+ (``weights_only=False``).

    Args:
        model:           LightningModule whose parameters will be updated.
        checkpoint_path: Path to the ``.ckpt`` file.
    """
    print(f"\n  Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Warning: Missing keys:    {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"    Warning: Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("  ✓ Weights loaded successfully")


def _build_causal_init_config(
    config: dict,
    epochs: int,
    lambda_hsic_cross_init: float,
    lambda_hsic_self_init: float,
    lambda_group: Optional[float],
) -> dict:
    """
    Build the config for causal initialization training.

    Changes applied:
    - ``training.lambda_hsic_cross``    = lambda_hsic_cross_init  (boosted)
    - ``training.lambda_hsic_self``     = lambda_hsic_self_init   (boosted)
    - ``training.lambda_hsic``          = lambda_hsic_cross_init  (legacy key)
    - ``training.use_hsic_annealing``   = False  (constant high HSIC)
    - ``training.max_epochs``           = epochs
    - ``training.k_fold``               = 1       (single fold)
    - ``training.save_ckpt_every_n_epochs`` guaranteed to exist
    - ``training.lambda_group_l1``      = lambda_group (if not None)

    Args:
        config:                  Base configuration dict.
        epochs:                  Number of causal-init training epochs.
        lambda_hsic_cross_init:  Boosted HSIC cross-attention coefficient.
        lambda_hsic_self_init:   Boosted HSIC self-attention coefficient.
        lambda_group:            Group-L1 coefficient from calibration (optional).

    Returns:
        A deep copy of ``config`` with the above overrides applied.
    """
    config_init = copy.deepcopy(config)
    config_init["training"]["lambda_hsic_cross"] = float(lambda_hsic_cross_init)
    config_init["training"]["lambda_hsic_self"] = float(lambda_hsic_self_init)
    config_init["training"]["lambda_hsic"] = float(lambda_hsic_cross_init)  # legacy
    config_init["training"]["use_hsic_annealing"] = False
    config_init["training"]["max_epochs"] = int(epochs)
    config_init["training"]["k_fold"] = 1
    config_init["training"].setdefault("save_ckpt_every_n_epochs", epochs)
    if lambda_group is not None:
        config_init["training"]["lambda_group_l1"] = float(lambda_group)
    return config_init


# =============================================================================
# SINGLE-SEED TRAINING HELPER
# =============================================================================

def _run_single_seed_init(
    config: dict,
    config_init: dict,
    data_dir: str,
    init_dir: Path,
    starting_checkpoint: Optional[str],
    seed: int,
    seed_index: int,
    n_seeds: int,
    cluster: bool = False,
) -> Dict[str, Any]:
    """
    Run causal initialization for a single seed.

    Creates a fresh model, optionally loads calibration weights, trains
    with HSIC-dominated loss, and returns the result.

    Args:
        config:               Original config (for model creation).
        config_init:          Causal init config (with boosted HSIC).
        data_dir:             Data directory.
        init_dir:             Parent directory for saving this seed's output.
        starting_checkpoint:  Optional calibration checkpoint to load weights from.
        seed:                 The seed value to use.
        seed_index:           Index of this seed (0-based, for display).
        n_seeds:              Total number of seeds being tried.

    Returns:
        Dict with seed results including HSIC values, checkpoint path, etc.
    """
    from causaliT.training.trainer import (
        create_model_instance,
        get_dataloader,
        train_single_fold,
        _make_fold_splits,
    )

    seed_label = f"seed_{seed}"
    seed_dir = init_dir / seed_label if n_seeds > 1 else init_dir

    if n_seeds > 1:
        seed_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n  --- Seed {seed_index + 1}/{n_seeds} (seed={seed}) ---")

    seed_everything(seed)
    model = create_model_instance(config_init, data_dir)

    if starting_checkpoint is not None:
        _load_weights_from_checkpoint(model, starting_checkpoint)

    # Freeze output MLP during causal init (structure-only learning phase).
    # The MLP head should NOT adapt during HSIC-dominated training because it
    # could compensate for incorrect causal structure by being expressive enough.
    # requires_grad is NOT saved in checkpoints, so main training will have
    # the MLP unfrozen by default when loading this checkpoint.
    freeze_mlp = config.get("staged_training", {}).get("causal_init_freeze_output_mlp", True)
    if freeze_mlp:
        inner = model.model if hasattr(model, "model") else model
        if hasattr(inner, "freeze_forecaster"):
            inner.freeze_forecaster()
            print("    ✓ Output MLP frozen for causal init (structure-only learning)")
        elif hasattr(inner, "freeze_output_head"):
            inner.freeze_output_head()
            print("    ✓ Output head frozen for causal init (structure-only learning)")

    dm = get_dataloader(config_init, data_dir, cluster=cluster, seed=seed)
    dm.prepare_data()

    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config_init, dm, seed, data_dir=data_dir
    )
    train_local_idx, val_local_idx = fold_splits[0]

    # Callbacks: HSIC tracker + DAG metrics (pre/post captured inside trainer)
    progress_logger = CausalInitProgressLogger()
    dag_callback = DAGMetricsCallback(config, data_dir)

    train_single_fold(
        config=config_init,
        model=model,
        dm=dm,
        fold=0,
        train_local_idx=train_local_idx,
        val_local_idx=val_local_idx,
        test_idx=test_idx,
        train_val_idx=train_val_idx,
        save_dir=str(seed_dir),
        trainable_params=0,
        cluster=cluster,
        extra_callbacks=[progress_logger, dag_callback],
    )

    # Locate checkpoint
    checkpoint_path = _find_last_checkpoint(seed_dir)
    if checkpoint_path is None:
        logger.warning(
            "Causal init seed %d: no checkpoint found in %s/k_0/checkpoints.",
            seed, seed_dir,
        )
        checkpoint_path = ""

    # Retrieve DAG metrics from callback (populated inside trainer lifecycle)
    pre_metrics = dag_callback.pre_metrics or {}
    post_metrics = dag_callback.post_metrics or {}

    # Get final HSIC values
    final_hsic_cross = progress_logger.hsic_values[-1] if progress_logger.hsic_values else None
    final_recon = progress_logger.recon_values[-1] if progress_logger.recon_values else None

    result = {
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "final_hsic_cross": final_hsic_cross,
        "final_recon": final_recon,
        "hsic_history": progress_logger.hsic_values,
        "recon_history": progress_logger.recon_values,
        "pre_init_metrics": _metrics_to_serializable(pre_metrics),
        "post_init_metrics": _metrics_to_serializable(post_metrics),
    }

    # Print seed summary
    hsic_str = f"{final_hsic_cross:.6f}" if final_hsic_cross is not None else "N/A"
    recon_str = f"{final_recon:.6f}" if final_recon is not None else "N/A"
    print(f"    Final HSIC: {hsic_str}, Recon: {recon_str}")
    print(_format_dag_metrics(pre_metrics, "Pre-init DAG (after epoch 0)"))
    print(_format_dag_metrics(post_metrics, "Post-init DAG"))

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_causal_initialization(
    config: dict,
    data_dir: str,
    save_dir: str,
    starting_checkpoint: Optional[str] = None,
    hsic_cross_multiplier: Optional[float] = None,
    hsic_self_multiplier: Optional[float] = None,
    seed: int = 42,
    cluster: bool = False,
) -> str:
    """
    Run causal initialization stage with optional multi-seed selection.

    This is the main entry point for Stage 1 (Causal Initialization).

    Process:
    1. Compute boosted HSIC coefficients using calibration multipliers and
       ``causal_init_hsic_multiplier`` from the config.
    2. Build a k=1 short-run config using ``_build_causal_init_config``.
    3. For each seed candidate (1 to ``causal_init_n_seeds``):
       a. Create fresh model, load calibration weights if provided.
       b. Evaluate DAG metrics before training.
       c. Train with HSIC-dominated loss.
       d. Evaluate DAG metrics after training.
       e. Record final HSIC value.
    4. Select the seed with lowest final HSIC cross value.
    5. Log seed variance analysis and pre/post DAG comparison.
    6. Return path to the best seed's checkpoint.

    Multi-seed selection rationale:
        Different random seeds lead to different local minima in the HSIC
        landscape. Because the initialization has few epochs, the final
        structure is heavily influenced by the starting point. Trying
        multiple seeds and selecting the best is analogous to evolutionary
        selection of favorable starting configurations.

    Args:
        config:               Configuration dictionary with ``staged_training``.
        data_dir:             Root data directory.
        save_dir:             Parent save directory; a ``causal_init/`` subfolder
                              is created here.
        starting_checkpoint:  Optional checkpoint to start from (from calibration).
        hsic_cross_multiplier: Multiplier for λ_hsic_cross (from calibration).
        hsic_self_multiplier:  Multiplier for λ_hsic_self (from calibration).
        seed:                 Base random seed.

    Returns:
        Absolute path string to the best causal init checkpoint.
    """
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset

    init_dir = Path(save_dir) / "causal_init"
    init_dir.mkdir(exist_ok=True, parents=True)

    staged_config = config.get("staged_training", {})
    init_epochs = staged_config.get("causal_init_epochs", 20)
    hsic_boost_factor = staged_config.get("causal_init_hsic_multiplier", 10.0)
    lambda_group = staged_config.get("lambda_group_l1", None)
    n_seeds = staged_config.get("causal_init_n_seeds", 1)

    # Ensure n_seeds is at least 1
    n_seeds = max(1, int(n_seeds))

    # Base HSIC weights from the (already-calibrated) training config
    base_hsic_cross = config["training"].get(
        "lambda_hsic_cross", config["training"].get("lambda_hsic", 0.1)
    )
    base_hsic_self = config["training"].get("lambda_hsic_self", 0.0)

    cross_mult = hsic_cross_multiplier if hsic_cross_multiplier is not None else 1.0
    self_mult = hsic_self_multiplier if hsic_self_multiplier is not None else 1.0

    lambda_hsic_cross_init = base_hsic_cross * cross_mult * hsic_boost_factor
    lambda_hsic_self_init = base_hsic_self * self_mult * hsic_boost_factor

    # Score sparsity lambdas pass through (selected via sweep before causal init)
    lambda_cross_score = config["training"].get("lambda_cross_score_sparse", 0.0)
    lambda_self_score = config["training"].get("lambda_self_score_sparse", 0.0)

    if not cluster:
        print(f"\n{'='*70}")
        print("CAUSAL INITIALIZATION: Training with HSIC-dominated loss")
        print(f"{'='*70}")
        print(f"  Epochs: {init_epochs}")
        print(f"  HSIC boost factor: {hsic_boost_factor}")
        print(f"  Number of seed candidates: {n_seeds}")
        print(f"\n  Cross-attention (S→X) HSIC:")
        print(f"    Base λ_hsic_cross     = {base_hsic_cross}")
        print(f"    Calibration multiplier = {cross_mult:.3f}")
        print(f"    λ_hsic_cross_init     = {lambda_hsic_cross_init:.4f}")
        if base_hsic_self > 0:
            print(f"\n  Self-attention (X→X) HSIC:")
            print(f"    Base λ_hsic_self      = {base_hsic_self}")
            print(f"    Calibration multiplier = {self_mult:.3f}")
            print(f"    λ_hsic_self_init      = {lambda_hsic_self_init:.4f}")
        else:
            print(f"\n  Self-attention (X→X) HSIC: disabled (λ_hsic_self = 0)")
        if lambda_cross_score > 0 or lambda_self_score > 0:
            print(f"\n  Score sparsity (from sweep):")
            print(f"    λ_cross_score_sparse  = {lambda_cross_score}")
            print(f"    λ_self_score_sparse   = {lambda_self_score}")
        if lambda_group is not None:
            print(f"\n  λ_group (from calibration): {lambda_group:.2e}")
        if starting_checkpoint:
            print(f"\n  Starting from checkpoint: {starting_checkpoint}")
        else:
            print(f"\n  Starting from fresh initialization")

    # Build config for causal init
    config_init = _build_causal_init_config(
        config=config,
        epochs=int(init_epochs),
        lambda_hsic_cross_init=lambda_hsic_cross_init,
        lambda_hsic_self_init=lambda_hsic_self_init,
        lambda_group=lambda_group,
    )
    # Score sparsity passes through from the original config
    config_init["training"]["lambda_cross_score_sparse"] = float(lambda_cross_score)
    config_init["training"]["lambda_self_score_sparse"] = float(lambda_self_score)

    # =========================================================================
    # MULTI-SEED LOOP
    # =========================================================================

    # Generate seed candidates: base seed, base+1, base+2, ...
    seed_candidates = [seed + i for i in range(n_seeds)]

    all_seed_results = []

    if not cluster:
        print(f"\n  Starting causal initialization training…")

    for i, candidate_seed in enumerate(seed_candidates):
        result = _run_single_seed_init(
            config=config,
            config_init=config_init,
            data_dir=data_dir,
            init_dir=init_dir,
            starting_checkpoint=starting_checkpoint,
            seed=candidate_seed,
            seed_index=i,
            n_seeds=n_seeds,
            cluster=cluster,
        )
        all_seed_results.append(result)

    # =========================================================================
    # SEED SELECTION
    # =========================================================================

    # Select the seed with the lowest final HSIC cross value
    valid_results = [r for r in all_seed_results if r["final_hsic_cross"] is not None]

    if not valid_results:
        logger.warning("No valid HSIC values found across seeds. Using first seed.")
        best_result = all_seed_results[0]
    elif len(valid_results) == 1:
        best_result = valid_results[0]
    else:
        best_result = min(valid_results, key=lambda r: r["final_hsic_cross"])

    best_seed = best_result["seed"]
    checkpoint_path = best_result["checkpoint_path"]

    # =========================================================================
    # HSIC VARIANCE ANALYSIS
    # =========================================================================

    hsic_values_all = [r["final_hsic_cross"] for r in all_seed_results
                       if r["final_hsic_cross"] is not None]

    if len(hsic_values_all) >= 2:
        hsic_array = np.array(hsic_values_all)
        hsic_variance = {
            "mean": float(np.mean(hsic_array)),
            "std": float(np.std(hsic_array)),
            "min": float(np.min(hsic_array)),
            "max": float(np.max(hsic_array)),
            "range": float(np.max(hsic_array) - np.min(hsic_array)),
            "cv": float(np.std(hsic_array) / (np.mean(hsic_array) + 1e-10)),  # coeff of variation
            "n_seeds": len(hsic_values_all),
        }
    elif len(hsic_values_all) == 1:
        hsic_variance = {
            "mean": hsic_values_all[0],
            "std": 0.0,
            "min": hsic_values_all[0],
            "max": hsic_values_all[0],
            "range": 0.0,
            "cv": 0.0,
            "n_seeds": 1,
        }
    else:
        hsic_variance = None

    # =========================================================================
    # PRE/POST DAG COMPARISON (from the best seed)
    # =========================================================================

    pre_init_metrics = best_result.get("pre_init_metrics", {})
    post_init_metrics = best_result.get("post_init_metrics", {})

    # Compute deltas for logging
    dag_deltas = {}
    for key in ["soft_hamming_cross", "soft_hamming_self",
                 "phi_cross_decisiveness", "phi_self_decisiveness"]:
        pre_val = pre_init_metrics.get(key)
        post_val = post_init_metrics.get(key)
        if pre_val is not None and post_val is not None:
            dag_deltas[f"delta_{key}"] = post_val - pre_val

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    # Prepare seed results for JSON (remove numpy arrays, keep scalars)
    seed_results_serializable = []
    for r in all_seed_results:
        seed_results_serializable.append({
            "seed": r["seed"],
            "final_hsic_cross": r["final_hsic_cross"],
            "final_recon": r["final_recon"],
            "checkpoint_path": r["checkpoint_path"],
            "hsic_history": r["hsic_history"],
            "recon_history": r["recon_history"],
            "pre_init_metrics": r["pre_init_metrics"],
            "post_init_metrics": r["post_init_metrics"],
        })

    summary = {
        "epochs": int(init_epochs),
        "n_seeds": n_seeds,
        "lambda_hsic_cross_init": lambda_hsic_cross_init,
        "lambda_hsic_self_init": lambda_hsic_self_init,
        "base_hsic_cross": base_hsic_cross,
        "base_hsic_self": base_hsic_self,
        "cross_multiplier": cross_mult,
        "self_multiplier": self_mult,
        "hsic_boost_factor": hsic_boost_factor,
        "lambda_group_l1": lambda_group,
        "lambda_cross_score_sparse": lambda_cross_score,
        "lambda_self_score_sparse": lambda_self_score,
        "starting_checkpoint": starting_checkpoint,
        # Best seed info
        "best_seed": best_seed,
        "selection_criterion": "final_hsic_cross",
        "final_checkpoint": checkpoint_path,
        "final_hsic": best_result["final_hsic_cross"],
        "final_recon": best_result["final_recon"],
        # Pre/post DAG comparison (best seed)
        "pre_init_metrics": pre_init_metrics,
        "post_init_metrics": post_init_metrics,
        "dag_deltas": dag_deltas,
        # Seed variance analysis
        "hsic_variance": hsic_variance,
        # All seed results
        "seed_results": seed_results_serializable,
    }

    with open(init_dir / "causal_init_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # =========================================================================
    # PRINT FINAL REPORT
    # =========================================================================

    if not cluster:
        print(f"\n{'='*70}")
        print("CAUSAL INITIALIZATION COMPLETE")
        print(f"{'='*70}")

        if n_seeds > 1:
            print(f"\n  Multi-seed selection ({n_seeds} seeds):")
            print(f"    Best seed: {best_seed}")
            print(f"    Selection criterion: lowest final HSIC cross")
            if hsic_variance:
                print(f"\n  HSIC variance across seeds:")
                print(f"    Mean:  {hsic_variance['mean']:.6f}")
                print(f"    Std:   {hsic_variance['std']:.6f}")
                print(f"    Range: [{hsic_variance['min']:.6f}, {hsic_variance['max']:.6f}]")
                print(f"    CV:    {hsic_variance['cv']:.4f}")
                if hsic_variance['cv'] > 0.3:
                    print(f"    ⚠ High coefficient of variation! Seed selection is very impactful.")
                elif hsic_variance['cv'] > 0.1:
                    print(f"    → Moderate seed sensitivity. Multi-seed selection is beneficial.")
                else:
                    print(f"    → Low seed sensitivity. Single seed may be sufficient.")

        if best_result["final_hsic_cross"] is not None:
            print(f"\n  Final HSIC (cross):  {best_result['final_hsic_cross']:.6f}")
        if best_result["final_recon"] is not None:
            print(f"  Final Recon:         {best_result['final_recon']:.6f}")

        # Print pre/post comparison
        print(f"\n  DAG Structure Comparison (best seed):")
        print(_format_dag_metrics(pre_init_metrics, "Before init"))
        print(_format_dag_metrics(post_init_metrics, "After init"))

        if dag_deltas:
            print(f"\n  Deltas:")
            for k, v in dag_deltas.items():
                direction = "↓" if v < 0 else "↑" if v > 0 else "="
                print(f"    {k}: {v:+.4f} {direction}")

        print(f"\n  Checkpoint: {checkpoint_path}")
        print(f"{'='*70}\n")

    # Generate diagnostic plot (non-critical; wrapped in try/except)
    try:
        _plot_causal_init_diagnostic(all_seed_results, best_seed, init_dir)
    except Exception as e:
        logger.warning(f"Could not generate diagnostic plot: {e}")

    return checkpoint_path


# =============================================================================
# DIAGNOSTIC PLOT
# =============================================================================

def _plot_causal_init_diagnostic(
    all_seed_results: List[Dict[str, Any]],
    best_seed: int,
    init_dir: Path,
) -> None:
    """
    Generate a diagnostic plot summarizing causal initialization results.

    Panels (adaptive layout):
    - Panel A: HSIC trajectories over epochs (all seeds, best highlighted)
    - Panel B: Pre/Post soft Hamming comparison (best seed bar chart)
    - Panel C: HSIC vs SHD scatter (only when n_seeds > 1)
               Answers: "Does lower HSIC correlate with lower SHD?"

    Saved to ``init_dir / causal_init_diagnostic.png`` and closed.

    Args:
        all_seed_results: List of per-seed result dicts from the multi-seed loop.
        best_seed:        The seed value selected as best.
        init_dir:         Directory to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe for cluster/CI)
    import matplotlib.pyplot as plt

    n_seeds = len(all_seed_results)
    has_shd = any(
        r.get("post_init_metrics", {}).get("soft_hamming_cross") is not None
        for r in all_seed_results
    )
    show_scatter = n_seeds > 1 and has_shd

    n_panels = 2 + int(show_scatter)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # ---- Panel A: HSIC trajectories over epochs ----
    ax = axes[0]
    for r in all_seed_results:
        hsic_hist = r.get("hsic_history", [])
        if not hsic_hist:
            continue
        epochs = list(range(1, len(hsic_hist) + 1))
        is_best = r["seed"] == best_seed
        ax.plot(
            epochs, hsic_hist,
            color="tab:red" if is_best else "tab:gray",
            linewidth=2.0 if is_best else 0.8,
            alpha=1.0 if is_best else 0.5,
            label=f"seed {r['seed']}" if is_best else None,
            zorder=10 if is_best else 1,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("HSIC (cross)")
    ax.set_title("A. HSIC Trajectories")
    if n_seeds > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel B: Pre/Post SHD comparison (best seed) ----
    ax = axes[1]
    best_r = next((r for r in all_seed_results if r["seed"] == best_seed), all_seed_results[0])
    pre = best_r.get("pre_init_metrics", {})
    post = best_r.get("post_init_metrics", {})

    metrics_to_plot = []
    labels = []
    for metric_key, label in [
        ("soft_hamming_cross", "SHD Cross\n(S→X)"),
        ("soft_hamming_self", "SHD Self\n(X→X)"),
        ("phi_cross_decisiveness", "Decisiveness\nCross"),
        ("phi_self_decisiveness", "Decisiveness\nSelf"),
    ]:
        pre_val = pre.get(metric_key)
        post_val = post.get(metric_key)
        if pre_val is not None and post_val is not None:
            metrics_to_plot.append((pre_val, post_val))
            labels.append(label)

    if metrics_to_plot:
        x_pos = np.arange(len(labels))
        width = 0.35
        pre_vals = [m[0] for m in metrics_to_plot]
        post_vals = [m[1] for m in metrics_to_plot]
        bars_pre = ax.bar(x_pos - width / 2, pre_vals, width, label="Pre-init", color="tab:blue", alpha=0.7)
        bars_post = ax.bar(x_pos + width / 2, post_vals, width, label="Post-init", color="tab:orange", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        # Annotate deltas
        for i, (pv, qv) in enumerate(zip(pre_vals, post_vals)):
            delta = qv - pv
            arrow = "↓" if delta < 0 else "↑"
            ax.text(i, max(pv, qv) + 0.02, f"{delta:+.3f}{arrow}", ha="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No SHD/decisiveness\ndata available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_title("B. Pre vs Post Init (best seed)")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel C: HSIC vs SHD scatter (multi-seed only) ----
    if show_scatter:
        ax = axes[2]
        hsic_vals = []
        shd_vals = []
        seeds = []
        for r in all_seed_results:
            h = r.get("final_hsic_cross")
            s = r.get("post_init_metrics", {}).get("soft_hamming_cross")
            if h is not None and s is not None:
                hsic_vals.append(h)
                shd_vals.append(s)
                seeds.append(r["seed"])

        if len(hsic_vals) >= 2:
            hsic_arr = np.array(hsic_vals)
            shd_arr = np.array(shd_vals)

            # Scatter: all seeds
            is_best_mask = np.array([s == best_seed for s in seeds])
            ax.scatter(hsic_arr[~is_best_mask], shd_arr[~is_best_mask],
                       c="tab:gray", s=50, alpha=0.7, label="Other seeds")
            ax.scatter(hsic_arr[is_best_mask], shd_arr[is_best_mask],
                       c="tab:red", s=100, marker="*", zorder=10, label=f"Best (seed {best_seed})")

            # Correlation
            if len(hsic_vals) >= 3:
                corr = np.corrcoef(hsic_arr, shd_arr)[0, 1]
                ax.set_xlabel(f"Final HSIC (cross)\n[Pearson r = {corr:.3f}]")
            else:
                ax.set_xlabel("Final HSIC (cross)")

            ax.set_ylabel("Soft Hamming (cross)")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Not enough data\nfor scatter", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("C. Does HSIC → SHD?")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Causal Initialization Diagnostic", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(init_dir / "causal_init_diagnostic.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved diagnostic plot: {init_dir / 'causal_init_diagnostic.png'}")


# =============================================================================
# OPTIONAL VERIFICATION
# =============================================================================

def verify_causal_init_effectiveness(
    config: dict,
    data_dir: str,
    save_dir: str,
    checkpoint_path: str,
) -> dict:
    """
    Verify the effectiveness of causal initialization.

    Checks:
    1. Attention patterns have structure (not uniform)
    2. HSIC is lower than at random init
    3. Predictions are not degenerate (have variance)

    Args:
        config:          Configuration dictionary.
        data_dir:        Root data directory.
        save_dir:        Save directory for verification results.
        checkpoint_path: Path to causal init checkpoint.

    Returns:
        Dict with verification metrics.
    """
    from causaliT.training.trainer import create_model_instance, get_dataloader

    model = create_model_instance(config, data_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dm = get_dataloader(
        config, data_dir, cluster=False, seed=config["training"].get("seed", 42)
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))

    if len(batch) == 3:
        S, X, Y = batch
    else:
        S, X = batch

    device = next(model.parameters()).device
    S, X = S.to(device), X.to(device)

    with torch.no_grad():
        val_idx = model.val_idx
        x_blanked = X.clone()
        x_blanked[:, :, val_idx] = 0.0
        pred_x, attention_weights, masks, entropies = model.model.forward(
            source_tensor=S,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=model.get_hard_masks() if hasattr(model, "get_hard_masks") else None,
        )

    x_target = X[:, :, val_idx]
    pred_variance = pred_x.var().item()
    residuals = x_target.squeeze() - pred_x.squeeze()
    residual_variance = residuals.var().item()

    att_entropy = None
    if attention_weights and attention_weights[0] is not None:
        att = attention_weights[0][0]
        att_entropy = -(att * (att + 1e-10).log()).sum(dim=-1).mean().item()

    metrics = {
        "prediction_variance": pred_variance,
        "attention_entropy": att_entropy,
        "residual_variance": residual_variance,
        "is_degenerate": pred_variance < 1e-6,
    }

    verify_path = Path(save_dir) / "causal_init" / "verification.json"
    with open(verify_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
