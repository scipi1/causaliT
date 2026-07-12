"""
ANM Staged Trainer: Flexible multi-stage training for Partial ANM Regression experiments.

Implements the Subsequent Structure-Reconstruct schedule from:
    docs/ideas/PARTIAL_ANM_REGRESSION.md

Each stage is defined by a flat dict of config overrides in
``config['anm_training']['stages']``.  Stages share the same data module and
fold splits (consistent train/val partition across the full alternating
schedule).  Checkpoints are chained: each stage warm-starts from the last
checkpoint of the previous stage.

Key differences from ``staged_trainer.py``:
- Arbitrary number of stages (not a fixed pipeline)
- Per-stage parameter freezing (structural or reconstruction)
- Per-stage BKD dropout override implementing the dropout curriculum (H3)
- Per-stage evaluation: DAG metrics and score-margin at stage end and/or
  every N epochs within a stage (H1/H2/H3 diagnostics)

Architecture compatibility — freeze flags:
    When ``use_gradient_routing=True``:
        ``freeze_structural_params=True`` → structural parameters get
        ``requires_grad_(False)`` in ``on_fit_start``.
        ``freeze_reconstruction_params=True`` → reconstruction parameters get
        ``requires_grad_(False)`` in ``on_fit_start``.

    When ``use_gradient_routing=False``:
        Freezing falls back to *loss-level gating*:
        - ``freeze_structural_params`` → set lambda_hsic_cross/self = 0.0
        - ``freeze_reconstruction_params`` → set lambda_recon = 0.0
        This is architecture-agnostic but less precise (zero-weighted gradients
        still flow through the shared graph; parameters are not truly frozen).
        TODO: per-architecture requires_grad freezing for non-routed models
              could be added in ``_build_stage_config`` when needed.

Example ``config['anm_training']`` block::

    anm_training:
      starting_checkpoint: null      # optional warm-start before stage 0
      stages:
        # H1: reconstruction warmup — measure residual-HSIC before structure opt
        - name: recon_warmup
          max_epochs: 30
          lambda_hsic_cross: 0.0
          lambda_hsic_self:  0.0
          lambda_recon: 1.0
          freeze_structural_params: true
          batch_key_dropout_p: 0.8
          eval_every_n_epochs: 5
          eval_dag: true

        # H2/H3: structure phase at current p level
        - name: struct_phase_1
          max_epochs: 20
          lambda_hsic_cross: 0.1
          lambda_recon: 0.0
          freeze_reconstruction_params: true
          batch_key_dropout_p: 0.8
          eval_every_n_epochs: 5
          eval_dag: true

        # H5: structure phase with gate-bias drift
        - name: struct_bias_drift
          max_epochs: 30
          lambda_hsic_cross: 0.1
          freeze_reconstruction_params: true
          batch_key_dropout_p: 0.4
          use_gate_bias_annealing: true
          gate_bias_start: 0.0
          gate_bias_end: -20.0
          gate_bias_anneal_epochs: 30
          eval_dag: true

        # Final joint fine-tuning
        - name: joint_finetune
          max_epochs: 10
          batch_key_dropout_p: 0.0
          eval_dag: true
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


def _to_plain_container(obj: Any) -> Any:
    """
    Convert OmegaConf containers to resolved plain Python dict/list objects.

    ANM stage specs often come from YAML as ``DictConfig`` / ``ListConfig``.
    Those are mapping-like but do not satisfy ``isinstance(x, dict)``, which
    previously caused nested keys such as ``evaluation.functions`` to be
    silently ignored.  Resolving here also materialises interpolations such as
    ``batch_key_dropout_p: ${experiment.batch_key_dropout}``.
    """
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _get_stage_eval_functions(stage_spec: dict) -> List[str]:
    """Return the post-stage evaluation function list from a stage spec."""
    evaluation = _to_plain_container(stage_spec.get("evaluation", {}) or {})
    if not isinstance(evaluation, dict):
        return []
    functions = _to_plain_container(evaluation.get("functions", []) or [])
    if isinstance(functions, str):
        return [functions]
    if isinstance(functions, (list, tuple)):
        return [str(fn) for fn in functions if fn]
    return []


def _save_stage_config(stage_config: dict, stage_dir: Path) -> str:
    """
    Save the fully-resolved per-stage config inside the stage directory.

    Evaluation functions load checkpoints from the stage directory.  They must
    also load the *same* config that produced that checkpoint, including stage
    overrides (loss weights, freezing flags, BKD p, epochs) and populated
    dataset-derived fields (sequence lengths, feature indices).  Therefore each
    stage persists its resolved config as ``<stage_dir>/config.yaml``.
    """
    stage_config_path = stage_dir / "config.yaml"
    cfg = OmegaConf.create(_to_plain_container(stage_config))
    OmegaConf.save(config=cfg, f=str(stage_config_path), resolve=True)
    return str(stage_config_path)


# =============================================================================
# STAGE EVALUATION CALLBACK
# =============================================================================

class StageEvalCallback(Callback):
    """
    Per-stage evaluation callback for ANM staged training.

    Fires at the end of each stage (``on_train_end``) and optionally every N
    epochs within a stage (``on_train_epoch_end``).  Captures:

    - DAG metrics via ``evaluate_dag_from_model``:
        phi decisiveness (how far edge probabilities are from 0.5),
        soft Hamming distance to the true DAG.
    - Score margin: ``score(true edges) - score(false edges)`` for both
      cross and self attention — the primary H2/H3 diagnostic.

    Args:
        stage_name:          Identifier for the current stage (in log tags).
        stage_idx:           Index of the current stage (0-based).
        config:              Full configuration dict.
        data_dir:            Root data directory (for loading true DAG masks).
        eval_every_n_epochs: Mid-stage snapshot frequency (0 = only at end).
    """

    def __init__(
        self,
        stage_name: str,
        stage_idx: int,
        config: dict,
        data_dir: str,
        eval_every_n_epochs: int = 0,
    ):
        super().__init__()
        self.stage_name = stage_name
        self.stage_idx = stage_idx
        self.config = config
        self.data_dir = data_dir
        self.eval_every_n_epochs = eval_every_n_epochs

        # Accumulated snapshots (mid-stage) and final result
        self.epoch_snapshots: List[Dict[str, Any]] = []
        self.final_metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _capture(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        label: str,
    ) -> Dict[str, Any]:
        """Capture DAG + score-margin diagnostics from the current model.

        Runs *inside* the training loop (``on_train_epoch_end``).  Any eval
        helper that flips the module into ``eval()`` and forgets to restore it
        would corrupt the NEXT training epoch — the root cause of the one-time
        HSIC "jump" at ``stage_start + eval_every_n_epochs`` observed for
        HardConcreteCrossAttention (its train forward is stochastic while its
        eval forward is a deterministic MAP gate).  We therefore snapshot the
        training mode here and restore it in a ``finally`` block as a
        defense-in-depth guard, in addition to the fix inside
        ``evaluate_dag_from_model``.
        """
        from causaliT.training.causal_initialization import evaluate_dag_from_model

        was_training = pl_module.training
        try:
            dag_metrics: Dict[str, Any] = {}
            try:
                dag_metrics = evaluate_dag_from_model(
                    pl_module, self.config, self.data_dir
                )
            except Exception as exc:
                logger.debug(
                    f"StageEvalCallback ({self.stage_name}): "
                    f"evaluate_dag_from_model failed: {exc}"
                )

            score_margin = _compute_score_margin(
                pl_module, self.config, self.data_dir
            )
        finally:
            if was_training:
                pl_module.train()


        result: Dict[str, Any] = {
            "stage": self.stage_name,
            "stage_idx": self.stage_idx,
            "epoch": trainer.current_epoch,
            "label": label,
            "score_margin_cross": score_margin.get("cross"),
            "score_margin_self": score_margin.get("self"),
        }
        # Attach scalar DAG metrics (skip numpy arrays — too large for JSON)
        for k, v in dag_metrics.items():
            if not isinstance(v, np.ndarray):
                result[k] = v

        return result

    # ------------------------------------------------------------------
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.eval_every_n_epochs <= 0:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every_n_epochs == 0:
            snap = self._capture(trainer, pl_module, label=f"epoch_{epoch}")
            self.epoch_snapshots.append(snap)

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.final_metrics = self._capture(
            trainer, pl_module, label="stage_end"
        )


# =============================================================================
# SCORE MARGIN HELPER
# =============================================================================

def _compute_score_margin(
    pl_module: pl.LightningModule,
    config: dict,
    data_dir: str,
) -> Dict[str, Optional[float]]:
    """
    Compute ``score(true edges) - score(false edges)`` for cross and self.

    Uses the batch-averaged score tensor stored on the inner attention module
    after the last forward pass.  Returns ``None`` for each attention type
    when the true DAG mask or the score tensor is unavailable.

    This is the primary diagnostic for H2 (does alternating improve edge
    separation?) and H3 (does the curriculum improve margin over stages?).

    Returns:
        Dict with keys ``"cross"`` and ``"self"``, each a float or None.
    """
    result: Dict[str, Optional[float]] = {"cross": None, "self": None}
    try:
        inner_model = getattr(pl_module, "model", pl_module)
        dataset_name = config.get("data", {}).get("dataset", "")
        if not dataset_name or not data_dir:
            return result

        from causaliT.evaluation.eval_funs.eval_utils import _load_true_dag_mask

        # ------------------------------------------------------------------
        # AttentionSelectorLayer branch
        # Single combined cross-attention block; no ``decoder`` attribute.
        # The attention module lives at ``inner_model.attention`` and its
        # inner_attention exposes a combined (L_X, L_S+L_X) score tensor.
        # We split it into the S→X (cross) and X→X (self) sub-matrices and
        # compute separate margins against the respective GT DAG masks.
        # ------------------------------------------------------------------
        if hasattr(inner_model, "attention") and not hasattr(inner_model, "decoder"):
            combined_inner = inner_model.attention.inner_attention
            # Accept both public and private attribute names for robustness.
            combined_score_t = getattr(combined_inner, "score_tensor_for_sparsity", None)
            if combined_score_t is None:
                combined_score_t = getattr(
                    combined_inner, "_score_tensor_for_sparsity", None
                )
            if combined_score_t is not None:
                scores_np = combined_score_t.detach().cpu().numpy()
                S_seq_len = getattr(inner_model, "S_seq_len", None)
                if S_seq_len is not None and scores_np.ndim == 2:
                    cross_scores_np = scores_np[:, :S_seq_len]    # (L_X, L_S)
                    self_scores_np  = scores_np[:, S_seq_len:]    # (L_X, L_X)

                    # --- S→X (cross) margin ---
                    true_cross = _load_true_dag_mask(
                        data_dir, dataset_name, "dec_cross"
                    )
                    if (
                        true_cross is not None
                        and cross_scores_np.shape == true_cross.shape
                    ):
                        true_mask  = true_cross.astype(bool)
                        false_mask = ~true_mask
                        if true_mask.any() and false_mask.any():
                            result["cross"] = float(
                                cross_scores_np[true_mask].mean()
                                - cross_scores_np[false_mask].mean()
                            )

                    # --- X→X (self) margin ---
                    true_self = _load_true_dag_mask(
                        data_dir, dataset_name, "dec_self"
                    )
                    if (
                        true_self is not None
                        and self_scores_np.shape == true_self.shape
                    ):
                        true_mask  = true_self.astype(bool)
                        false_mask = ~true_mask
                        np.fill_diagonal(false_mask, False)  # no self-loops
                        if true_mask.any() and false_mask.any():
                            result["self"] = float(
                                self_scores_np[true_mask].mean()
                                - self_scores_np[false_mask].mean()
                            )
            return result

        # ------------------------------------------------------------------
        # SingleCausalLayer (and related) branch
        # Separate cross- and self-attention blocks inside ``decoder.layers``.
        # ------------------------------------------------------------------
        if not hasattr(inner_model, "decoder"):
            return result

        layer = inner_model.decoder.layers[0]

        # --- Cross-attention score margin ---
        true_cross = _load_true_dag_mask(data_dir, dataset_name, "dec_cross")
        cross_inner = layer.global_cross_attention.inner_attention
        cross_score_t = getattr(cross_inner, "_score_tensor_for_sparsity", None)
        if cross_score_t is not None and true_cross is not None:
            scores_np = cross_score_t.detach().cpu().numpy()
            if scores_np.shape == true_cross.shape:
                true_mask = true_cross.astype(bool)
                false_mask = ~true_mask
                if true_mask.any() and false_mask.any():
                    result["cross"] = float(
                        scores_np[true_mask].mean() - scores_np[false_mask].mean()
                    )

        # --- Self-attention score margin ---
        true_self = _load_true_dag_mask(data_dir, dataset_name, "dec_self")
        self_inner = layer.global_self_attention.inner_attention
        # ToeplitzAttention exposes P_edge_for_reg; others use _score_tensor_for_sparsity
        self_score_t = getattr(self_inner, "P_edge_for_reg", None)
        if self_score_t is None:
            self_score_t = getattr(self_inner, "_score_tensor_for_sparsity", None)
        if self_score_t is not None and true_self is not None:
            scores_np = self_score_t.detach().cpu().numpy()
            if scores_np.ndim == 2 and scores_np.shape == true_self.shape:
                true_mask = true_self.astype(bool)
                false_mask = ~true_mask
                np.fill_diagonal(false_mask, False)   # exclude diagonal (no self-loops)
                if true_mask.any() and false_mask.any():
                    result["self"] = float(
                        scores_np[true_mask].mean() - scores_np[false_mask].mean()
                    )

    except Exception as exc:
        logger.debug(f"_compute_score_margin failed (non-critical): {exc}")

    return result


# =============================================================================
# STAGE CONFIG BUILDER
# =============================================================================

# Training-level keys that a stage spec may override directly.
# Keys are intentionally additive — adding a key here makes it eligible for
# direct forwarding into the ``training`` config for ALL architectures.
# Architecture-specific keys (e.g. ``lambda_hsic`` for AttentionSelectorLayer)
# are translated by ``_build_stage_config`` after this loop.
_TRAINING_OVERRIDE_KEYS = frozenset({
    "max_epochs",
    # SingleCausalLayer / related: separate cross and self HSIC weights
    "lambda_hsic_cross",
    "lambda_hsic_self",
    "lambda_recon",
    # Convex-mix reconstruction weight on the structural pathway
    # (SingleCausalForecaster + AttentionSelectorForecaster).
    "lambda_struct_recon",
    "lambda_self_score_sparse",

    "lambda_cross_score_sparse",
    # AttentionSelectorLayer: unified HSIC weight and score sparsity weight
    "lambda_hsic",
    "lambda_score_sparse",
    # L0 regularization (HardConcreteCrossAttention) — per-stage override,
    # e.g. lambda_l0=0 in warmup stage, swept value in structural/joint stages.
    "lambda_l0",
    "use_gradient_routing",
    # Freeze flags — processed with fallback logic below
    "freeze_structural_params",
    "freeze_reconstruction_params",
    # Gate bias annealing (H5)
    "use_gate_bias_annealing",
    "gate_bias_start",
    "gate_bias_end",
    "gate_bias_anneal_epochs",
    # Optimizer overrides
    "lr",
    "structural_lr",
    # Checkpoint frequency
    "save_ckpt_every_n_epochs",
})

# Keys consumed by the orchestrator only — not forwarded to training config
_ORCHESTRATOR_ONLY_KEYS = frozenset({
    "name",
    "eval_every_n_epochs",
    "eval_dag",
    "batch_key_dropout_p",
    # Cross-fit / bilevel data-split selector: which training subset this stage
    # trains on ("recon" | "struct" | "full").  Consumed by the orchestrator to
    # pick the per-stage train indices; not forwarded to the training config.
    "data_split",
    # Per-stage post-training evaluation — dispatched by the orchestrator after
    # train_single_fold returns; not a training-config key.
    "evaluation",
    # Per-stage model-constructor kwargs (e.g. init_tau, batch_key_dropout).
    # Applied to config['model']['kwargs'] in _build_stage_config.
    "model_kwargs_overrides",
})


def _build_stage_config(
    base_config: dict,
    stage_spec: dict,
    stage_idx: int,
) -> dict:
    """
    Build a per-stage training config by overlaying ``stage_spec`` onto
    a deep copy of ``base_config``.

    Always forces ``k_fold=1`` (stages run on a single fold) and ensures
    ``save_ckpt_every_n_epochs`` is set so a checkpoint is always written.

    Freeze flag logic:
        ``freeze_structural_params=True`` with ``use_gradient_routing=True``:
            Passed through to ``training`` config — ``SingleCausalForecaster.
            on_fit_start`` will call ``requires_grad_(False)`` on structural
            params (Q/K, structure embeddings, attention internals).

        ``freeze_structural_params=True`` with ``use_gradient_routing=False``:
            Falls back to loss-level gating: sets ``lambda_hsic_cross=0`` and
            ``lambda_hsic_self=0``.  The flag is cleared so the forecaster
            does not attempt to use ``classify_parameters`` on a non-routed model.
            TODO: Extend with per-module freezing for specific non-routed
                  architectures (e.g. locking ``query_projection`` only).

        ``freeze_reconstruction_params=True`` with ``use_gradient_routing=True``:
            Sets ``training.freeze_reconstruction_params=True`` so the forecaster
            calls ``requires_grad_(False)`` on value/FF/MLP parameters.

        ``freeze_reconstruction_params=True`` with ``use_gradient_routing=False``:
            Falls back to loss-level gating: sets ``lambda_recon=0.0``.

    BKD curriculum (batch_key_dropout_p):
        Overrides ``model.kwargs.batch_key_dropout`` (p_init) and
        ``batch_key_dropout_p_final`` (set to the same value so there is no
        within-stage annealing).  ``batch_key_dropout_annealing_batches`` is
        cleared (None) so the BKD step-counter annealing is disabled for this
        stage.  The stage's fixed p directly controls parent-set coverage.

    Args:
        base_config: Full base configuration dict (deep-copied, not modified).
        stage_spec:  Flat dict of stage-specific overrides.
        stage_idx:   0-based stage index (for log messages only).

    Returns:
        A deep copy of base_config with all stage overrides applied.
    """
    cfg = copy.deepcopy(base_config)
    tc = cfg["training"]
    model_obj = cfg.get("model", {}).get("model_object", "")

    # --- Direct training-level overrides ---
    for key in _TRAINING_OVERRIDE_KEYS:
        if key in stage_spec:
            tc[key] = stage_spec[key]

    # --- AttentionSelectorLayer key translation ---
    # AttentionSelectorForecaster uses unified ``lambda_hsic`` and
    # ``lambda_score_sparse`` (no separate cross/self split).  When a stage
    # spec was written with SingleCausalLayer key names and no explicit unified
    # key, translate automatically so the override takes effect.
    if model_obj == "AttentionSelectorLayer":
        if "lambda_hsic_cross" in stage_spec and "lambda_hsic" not in stage_spec:
            tc["lambda_hsic"] = stage_spec["lambda_hsic_cross"]
            logger.debug(
                "Stage %d: translated lambda_hsic_cross → lambda_hsic for "
                "AttentionSelectorLayer.",
                stage_idx,
            )
        if (
            "lambda_cross_score_sparse" in stage_spec
            and "lambda_score_sparse" not in stage_spec
        ):
            tc["lambda_score_sparse"] = stage_spec["lambda_cross_score_sparse"]
            logger.debug(
                "Stage %d: translated lambda_cross_score_sparse → lambda_score_sparse "
                "for AttentionSelectorLayer.",
                stage_idx,
            )

    # --- Freeze flag fallback when gradient routing is off ---
    use_gr = tc.get("use_gradient_routing", False)

    if stage_spec.get("freeze_structural_params", False) and not use_gr:
        # TODO: per-architecture hard freezing could go here for specific models
        logger.info(
            "Stage %d: freeze_structural_params=True but use_gradient_routing=False "
            "— falling back to HSIC loss-gating (lambda_hsic_cross/self = 0.0).",
            stage_idx,
        )
        tc["lambda_hsic_cross"] = 0.0
        tc["lambda_hsic_self"] = 0.0
        # AttentionSelectorLayer reads the unified key — zero it too.
        if model_obj == "AttentionSelectorLayer":
            tc["lambda_hsic"] = 0.0
        tc["freeze_structural_params"] = False   # don't try requires_grad route

    if stage_spec.get("freeze_reconstruction_params", False) and not use_gr:
        # TODO: per-architecture hard freezing could go here
        logger.info(
            "Stage %d: freeze_reconstruction_params=True but use_gradient_routing=False "
            "— falling back to recon loss-gating (lambda_recon = 0.0).",
            stage_idx,
        )
        tc["lambda_recon"] = 0.0
        tc["freeze_reconstruction_params"] = False

    # --- BKD dropout curriculum: fixed p per stage, no within-stage annealing ---
    bkd_p = stage_spec.get("batch_key_dropout_p", None)
    if bkd_p is not None:
        bkd_p = float(bkd_p)
        model_kwargs = cfg.get("model", {}).get("kwargs", {})
        if model_kwargs is not None:
            model_kwargs["batch_key_dropout"] = bkd_p
            # p_init == p_final → no annealing within the stage
            model_kwargs["batch_key_dropout_p_final"] = bkd_p
            # Disable step-counter annealing
            model_kwargs["batch_key_dropout_annealing_batches"] = None

    # --- Arbitrary model-constructor kwargs overrides ---
    # Overrides ``config['model']['kwargs']`` directly so that any model
    # constructor kwarg can be varied per stage without adding a dedicated
    # handling block here.
    #
    # Typical use-cases:
    #   init_tau          — attention temperature (plain Python float; NOT saved
    #                       in checkpoint state_dict, so the value from the freshly
    #                       constructed stage model always survives checkpoint loading)
    #   batch_key_dropout — BKD initial drop probability (plain float; also safe)
    #   init_gate_bias    — initial Toeplitz gate bias (initial value only; the
    #                       learnable ``gate_bias`` Parameter IS in state_dict and
    #                       WILL be restored from checkpoint — use with care)
    #
    # Note: keys in ``model_kwargs_overrides`` take precedence over the
    # ``batch_key_dropout_p`` shorthand applied in the block above, since this
    # block runs last.
    #
    # Note on AttentionSelectorLayer: ``batch_key_dropout`` is now supported
    # and can be passed freely via ``model_kwargs_overrides`` (or via the
    # ``batch_key_dropout_p`` shorthand above).
    mkw_overrides = stage_spec.get("model_kwargs_overrides", None)
    if mkw_overrides:
        mkw_overrides = _to_plain_container(mkw_overrides)
        if isinstance(mkw_overrides, dict) and mkw_overrides:
            model_kwargs = cfg.get("model", {}).get("kwargs")
            if model_kwargs is not None:
                for key, val in mkw_overrides.items():
                    model_kwargs[key] = val
                    logger.debug(
                        "Stage %d: model_kwargs_overrides[%s] = %s",
                        stage_idx, key, val,
                    )

    # --- Always single fold ---
    tc["k_fold"] = 1

    # --- Ensure a checkpoint is always saved ---
    if tc.get("save_ckpt_every_n_epochs") is None:
        tc["save_ckpt_every_n_epochs"] = tc.get("max_epochs", 100)

    return cfg


# =============================================================================
# CHECKPOINT FINDER
# =============================================================================

def _find_stage_checkpoint(stage_dir: Path) -> Optional[str]:
    """
    Find the last checkpoint written by ``train_single_fold`` in a stage dir.

    Standard path: ``<stage_dir>/k_0/checkpoints/last.ckpt``.
    Fallback: any ``.ckpt`` file in the checkpoints directory (alphabetical last).
    """
    last_ckpt = stage_dir / "k_0" / "checkpoints" / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    ckpt_dir = stage_dir / "k_0" / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            return str(ckpts[-1])

    return None


# =============================================================================
# CROSS-FIT DATA SPLIT HELPER
# =============================================================================

def _partition_train_indices(
    train_local_idx: np.ndarray,
    ratio: float,
    seed: int,
) -> tuple:
    """
    Split ``train_local_idx`` into two disjoint subsets (recon, struct).

    Deterministic given ``seed`` so reruns are reproducible.  ``ratio`` is the
    fraction of samples assigned to the reconstruction subset; the remainder go
    to the structure subset.  Both subsets are guaranteed non-empty.

    This implements the DML / DARTS-style honest cross-fit: the reconstruction
    regressor is fit on one subset and residual-based structural signals (HSIC)
    are computed on the other, so residuals used for structure learning are
    out-of-sample w.r.t. the reconstruction fit.

    Args:
        train_local_idx: 1-D array of local training indices to partition.
        ratio:           Fraction assigned to the recon subset.  Values in the
                         open interval (0, 1) produce a genuine disjoint split;
                         values <= 0 or >= 1 deactivate cross-fitting (both
                         subsets = the full training set).
        seed:            RNG seed for the deterministic permutation.

    Returns:
        (recon_idx, struct_idx): two disjoint sub-arrays of ``train_local_idx``.
    """
    train_local_idx = np.asarray(train_local_idx)

    # --- Cross-fit deactivation: ratio outside (0, 1) -> use full set for both ---
    # A ratio of exactly 0 or 1 (or out-of-range) is not a valid disjoint split;
    # treat it as an explicit "turn cross-fitting off" signal and return the full
    # training set for both subsets (identical to every stage using
    # ``data_split: full``).
    if not (0.0 < ratio < 1.0):
        return train_local_idx, train_local_idx

    n = len(train_local_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(ratio * n))
    # Keep both subsets non-empty
    cut = max(1, min(n - 1, cut))
    recon_idx = train_local_idx[perm[:cut]]
    struct_idx = train_local_idx[perm[cut:]]
    return recon_idx, struct_idx


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def anm_alternating_trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str = "NA",
    debug: bool = False,
    best: bool = False,
) -> pd.DataFrame:
    """
    ANM alternating trainer: runs the Subsequent Structure-Reconstruct schedule.

    Reads stage specifications from ``config['anm_training']['stages']``.
    Each stage spec is a flat dict; see module docstring and ``_build_stage_config``
    for the full list of supported keys.

    Data splits are computed once and shared across all stages, ensuring
    consistent train/val partitioning throughout the alternating schedule.

    Stage checkpoints are chained: stage k warm-starts from the last.ckpt of
    stage k-1.  If a stage produces no checkpoint (e.g. 0 training epochs), the
    previous checkpoint is reused and a warning is logged.

    Args:
        config:          Full configuration dict (``anm_training.stages`` required).
        data_dir:        Root data directory.
        save_dir:        Parent save directory.  Each stage writes to
                         ``<save_dir>/anm_stages/{idx:02d}_{name}/``.
        cluster:         Suppress progress bar / use 1-GPU mode.
        experiment_tag:  Passed to ``train_single_fold`` for the per-run manifest.
        debug:           Enable anomaly detection, memory logger, etc.
        best:            If True, collect best-checkpoint metrics per stage.

    Returns:
        pd.DataFrame: One row per stage with training metrics and DAG diagnostics.
    """
    from causaliT.training.trainer import (
        get_dataloader,
        _make_fold_splits,
        create_model_instance,
        train_single_fold,
    )
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset

    # -------------------------------------------------------------------------
    anm_cfg = config.get("anm_training", {})
    # Convert OmegaConf containers to resolved plain Python containers so
    # nested blocks such as ``evaluation.functions`` behave like normal dicts.
    stages = _to_plain_container(anm_cfg.get("stages", []))
    if not stages:
        raise ValueError(
            "config['anm_training']['stages'] is empty or missing. "
            "Define at least one stage spec dict."
        )

    seed = config["training"].get("seed", 42)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Populate sequence lengths from dataset metadata (needed by model builder)
    config = populate_seq_lengths_from_dataset(config, data_dir)

    # -------------------------------------------------------------------------
    # Build shared data module and fold splits (used by all stages)
    # -------------------------------------------------------------------------
    dm = get_dataloader(config, data_dir, cluster, seed)
    dm.prepare_data()
    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config, dm, seed, data_dir=data_dir
    )
    # All stages use fold 0
    train_local_idx, val_local_idx = fold_splits[0]

    # -------------------------------------------------------------------------
    # Cross-fit / bilevel data splits (computed once, shared across all stages)
    # -------------------------------------------------------------------------
    # Partition the training indices into two disjoint subsets so that
    # reconstruction-dominant and structure-dominant stages can train on
    # DIFFERENT data (DML/DARTS-style honesty: residual-HSIC is evaluated on
    # samples the reconstruction fit never saw).  Validation stays SHARED so
    # stage-to-stage DAG metrics remain comparable.  Each stage picks its subset
    # via the ``data_split`` spec key ("recon" | "struct" | "full"); "full"
    # (default) uses the entire training set = fully backward compatible.
    data_split_ratio = float(anm_cfg.get("data_split_ratio", 0.5))
    split_recon, split_struct = _partition_train_indices(
        train_local_idx, data_split_ratio, seed
    )
    stage_splits: Dict[str, np.ndarray] = {
        "recon": split_recon,
        "struct": split_struct,
        "full": train_local_idx,
    }
    if not cluster:
        print(
            f"  Cross-fit data splits (ratio={data_split_ratio}): "
            f"recon={len(split_recon)}, struct={len(split_struct)}, "
            f"full={len(train_local_idx)}; val shared={len(val_local_idx)}"
        )

    stages_parent_dir = Path(save_dir) / "anm_stages"
    stages_parent_dir.mkdir(parents=True, exist_ok=True)

    # Optional warm-start before stage 0
    starting_ckpt: Optional[str] = _to_plain_container(
        anm_cfg.get("starting_checkpoint", None)
    )

    # -------------------------------------------------------------------------
    # Stage loop
    # -------------------------------------------------------------------------
    all_stage_rows: List[Dict[str, Any]] = []
    all_stage_summaries: List[Dict[str, Any]] = []

    # Cumulative epoch offset: PL's epoch counter is NOT reset between stages
    # (resume_ckpt restores it).  We must therefore set max_epochs to the
    # running total so that PL runs exactly stage_local_epochs new epochs per
    # stage.  E.g. for four 200-epoch stages: 200 / 400 / 600 / 800.
    cumulative_epoch_offset: int = 0

    for stage_idx, stage_spec in enumerate(stages):
        stage_spec = _to_plain_container(stage_spec)
        if not isinstance(stage_spec, dict):
            raise TypeError(
                f"ANM stage {stage_idx} must be a mapping/dict, "
                f"got {type(stage_spec).__name__}: {stage_spec!r}"
            )
        stage_name = stage_spec.get("name", f"stage_{stage_idx:02d}")
        stage_dir = stages_parent_dir / f"{stage_idx:02d}_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Select this stage's training subset (cross-fit data split).
        stage_data_split = str(stage_spec.get("data_split", "full")).lower()
        if stage_data_split not in stage_splits:
            raise ValueError(
                f"ANM stage {stage_idx} ({stage_name}): invalid data_split "
                f"{stage_data_split!r}. Expected one of {sorted(stage_splits)}."
            )
        stage_train_idx = stage_splits[stage_data_split]

        # Local epoch count from stage spec (or fall back to base config)
        stage_local_epochs: int = int(
            stage_spec.get("max_epochs", config["training"].get("max_epochs", 100))
        )
        # Cumulative max_epochs passed to PL so the epoch counter continues
        # from where the previous stage left off
        cumulative_max_epochs: int = cumulative_epoch_offset + stage_local_epochs

        # Determine checkpoint mode:
        #   stage 0  → warm_start (weights only) from the optional pre-existing ckpt
        #   stage 1+ → resume (weights + optimizer state + epoch counter)
        # Note: resume_ckpt=None for stage 0 when no starting_checkpoint is set
        # in anm_training config; that is already the correct "fresh start" behaviour.
        is_first_stage: bool = (stage_idx == 0)
        stage_ckpt_mode: str = "warm_start" if is_first_stage else "resume"

        if not cluster:
            print("\n" + "=" * 70)
            print(f"ANM STAGE {stage_idx}: {stage_name}  "
                  f"(global epochs {cumulative_epoch_offset}–{cumulative_max_epochs - 1})")
            print("=" * 70)
            _print_stage_header(stage_spec, starting_ckpt, ckpt_mode=stage_ckpt_mode)

        # Build stage config and override max_epochs with the cumulative total
        stage_config = _build_stage_config(config, stage_spec, stage_idx)
        stage_config["training"]["max_epochs"] = cumulative_max_epochs
        stage_config_path = _save_stage_config(stage_config, stage_dir)

        # Model: always created fresh with the current stage config so that
        # freeze flags and lambda overrides are in effect.  When resume_ckpt is
        # used, PL will overwrite the random weights with the checkpoint weights
        # before training begins; the model object is just the architecture holder.
        seed_everything(seed)
        model = create_model_instance(stage_config, data_dir)

        # Stage evaluation callback
        do_eval = stage_spec.get("eval_dag", True)
        eval_every = int(stage_spec.get("eval_every_n_epochs", 0)) if do_eval else 0
        stage_eval_cb: Optional[StageEvalCallback] = (
            StageEvalCallback(
                stage_name=stage_name,
                stage_idx=stage_idx,
                config=config,
                data_dir=data_dir,
                eval_every_n_epochs=eval_every,
            )
            if do_eval else None
        )
        extra_cbs = [stage_eval_cb] if stage_eval_cb is not None else []

        # Train the stage
        fold_metrics = train_single_fold(
            config=stage_config,
            model=model,
            dm=dm,
            fold=0,
            train_local_idx=stage_train_idx,
            val_local_idx=val_local_idx,
            test_idx=test_idx,
            train_val_idx=train_val_idx,
            save_dir=str(stage_dir),
            trainable_params=0,
            cluster=cluster,
            # Stages 1+: resume restores weights + optimizer state + epoch counter.
            # Stage 0:   warm_start loads weights only (optimizer starts fresh),
            #            which is the correct behaviour when kicking off from an
            #            optional pre-existing checkpoint that may differ in
            #            training configuration.
            resume_ckpt=starting_ckpt if not is_first_stage else None,
            warm_start_ckpt=starting_ckpt if is_first_stage else None,
            experiment_tag=f"{experiment_tag}_{stage_name}",
            debug=debug,
            best=best,
            extra_callbacks=extra_cbs,
        )

        # Advance cumulative epoch offset for the next stage
        cumulative_epoch_offset = cumulative_max_epochs

        # Chain checkpoints
        new_ckpt = _find_stage_checkpoint(stage_dir)
        if new_ckpt is None:
            logger.warning(
                "Stage %d (%s): no checkpoint found in %s. "
                "Next stage will reuse the previous checkpoint.",
                stage_idx, stage_name, stage_dir,
            )
        else:
            starting_ckpt = new_ckpt
            logger.info(
                "Stage %d (%s): checkpoint for next stage → %s",
                stage_idx, stage_name, starting_ckpt,
            )
            if not cluster:
                print(f"  ✓ Next-stage checkpoint: {starting_ckpt}")

        # ---------------------------------------------------------------
        # Per-stage post-training evaluation (H1/H2/H3/H4/H5 diagnostics)
        #
        # If the stage spec contains an ``evaluation: {functions: [...]}``
        # block, run those functions now against the stage's output directory.
        # The last stage can therefore list the classical eval functions
        # (eval_attention_scores, eval_interventions, …) just like the
        # standard trainer does after a full training run.
        #
        # Errors in evaluation never abort the pipeline — they are logged
        # and the stage loop continues normally.
        # ---------------------------------------------------------------
        stage_eval_fns: List[str] = _get_stage_eval_functions(stage_spec)
        if stage_eval_fns:
            if not cluster:
                print(
                    f"\n  [{stage_name}] Running {len(stage_eval_fns)} "
                    f"post-stage evaluation(s): {stage_eval_fns}"
                )
            try:
                from causaliT.evaluation.eval_funs.eval_funs_wraps import (
                    run_evaluations_from_config,
                )
                run_evaluations_from_config(
                    experiment=str(stage_dir),
                    datadir_path=data_dir,
                    show_plots=False,
                    functions=stage_eval_fns,
                )
            except Exception as exc:
                logger.warning(
                    "Stage %d (%s): post-stage evaluation failed (non-critical): %s",
                    stage_idx, stage_name, exc,
                )

        # Collect stage row
        row: Dict[str, Any] = {
            "stage_idx": stage_idx,
            "stage_name": stage_name,
            "data_split": stage_data_split,
            "n_train_samples": int(len(stage_train_idx)),
            "checkpoint": starting_ckpt or "",
            "stage_config": stage_config_path,
        }
        for k, v in fold_metrics.items():
            row[k] = v.item() if isinstance(v, torch.Tensor) else v

        # Attach final DAG diagnostics
        if stage_eval_cb is not None and stage_eval_cb.final_metrics:
            fm = stage_eval_cb.final_metrics
            for diag_key in (
                "phi_cross_decisiveness",
                "phi_self_decisiveness",
                "soft_hamming_cross",
                "soft_hamming_self",
                "score_margin_cross",
                "score_margin_self",
            ):
                row[f"dag_{diag_key}"] = fm.get(diag_key)

        all_stage_rows.append(row)

        # Build JSON-serializable stage summary
        stage_summary: Dict[str, Any] = {
            "stage_idx": stage_idx,
            "stage_name": stage_name,
            "data_split": stage_data_split,
            "n_train_samples": int(len(stage_train_idx)),
            "global_epoch_start": cumulative_epoch_offset - stage_local_epochs,
            "global_epoch_end": cumulative_epoch_offset - 1,
            "checkpoint_mode": stage_ckpt_mode,
            "stage_spec": _to_plain_container(stage_spec),
            "stage_config": stage_config_path,
            "checkpoint": starting_ckpt or "",
            "final_dag_metrics": (
                {
                    k: v
                    for k, v in stage_eval_cb.final_metrics.items()
                    if not isinstance(v, np.ndarray)
                }
                if stage_eval_cb is not None and stage_eval_cb.final_metrics
                else {}
            ),
            "epoch_snapshots": (
                stage_eval_cb.epoch_snapshots if stage_eval_cb is not None else []
            ),
        }
        all_stage_summaries.append(stage_summary)

        if not cluster and stage_eval_cb is not None and stage_eval_cb.final_metrics:
            _print_dag_metrics(stage_eval_cb.final_metrics, stage_name)

    # -------------------------------------------------------------------------
    # Save full summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "experiment_tag": experiment_tag,
        "n_stages": len(stages),
        "final_checkpoint": starting_ckpt or "",
        "stages": all_stage_summaries,
    }
    summary_path = Path(save_dir) / "anm_training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    if not cluster:
        print("\n" + "=" * 70)
        print("ANM ALTERNATING TRAINING COMPLETE")
        print(f"  {len(stages)} stage(s) completed")
        print(f"  Final checkpoint: {starting_ckpt}")
        print(f"  Summary: {summary_path}")
        print("=" * 70)

    df = pd.DataFrame(all_stage_rows)
    return df


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def run_anm_trainer_from_config(
    config_path: str,
    data_dir: str,
    save_dir: str,
    cluster: bool = False,
    experiment_tag: str = "NA",
) -> pd.DataFrame:
    """Run ANM alternating training directly from a YAML config path."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    return anm_alternating_trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
    )


# =============================================================================
# HELPERS
# =============================================================================

def _print_stage_header(
    stage_spec: dict,
    starting_ckpt: Optional[str],
    ckpt_mode: str = "warm_start",
) -> None:
    """Print a concise stage configuration summary."""
    def _fmt(key, default="(inherited)"):
        return stage_spec.get(key, default)

    print(f"  epochs:        {_fmt('max_epochs')}")
    print(
        f"  lambda_hsic:   cross={_fmt('lambda_hsic_cross')}  "
        f"self={_fmt('lambda_hsic_self')}"
    )
    print(f"  lambda_recon:  {_fmt('lambda_recon')}")
    print(f"  BKD p:         {_fmt('batch_key_dropout_p')}")
    print(f"  freeze_struct: {stage_spec.get('freeze_structural_params', False)}")
    print(f"  freeze_recon:  {stage_spec.get('freeze_reconstruction_params', False)}")
    print(f"  data_split:    {stage_spec.get('data_split', 'full')}")
    if stage_spec.get("use_gate_bias_annealing"):
        print(
            f"  gate_bias:     {stage_spec.get('gate_bias_start', 0.0)} → "
            f"{stage_spec.get('gate_bias_end', -20.0)} "
            f"over {_fmt('gate_bias_anneal_epochs', '?')} epochs"
        )
    mkw = stage_spec.get("model_kwargs_overrides")
    if mkw:
        for k, v in (_to_plain_container(mkw) or {}).items():
            print(f"  model_kwarg:   {k} = {v}")
    if starting_ckpt:
        if ckpt_mode == "resume":
            print(f"  resume from:   {starting_ckpt}  (weights + optimizer state)")
        else:
            print(f"  warm start:    {starting_ckpt}  (weights only)")
    else:
        print("  warm start:    (fresh initialization)")


def _print_dag_metrics(metrics: dict, stage_name: str) -> None:
    """Print scalar DAG diagnostics after a stage completes."""
    print(f"\n  [{stage_name}] DAG diagnostics:")
    for key in (
        "phi_cross_decisiveness",
        "phi_self_decisiveness",
        "soft_hamming_cross",
        "soft_hamming_self",
        "score_margin_cross",
        "score_margin_self",
    ):
        val = metrics.get(key)
        if val is not None:
            print(f"    {key}: {val:.4f}")


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return str(obj)
