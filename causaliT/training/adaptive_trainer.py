"""
Adaptive Trainer: metric-driven alternating Structure/Reconstruct training.

Motivation
----------
``anm_staged_trainer.py`` runs a *rigid* schedule: each stage trains for a fixed
epoch budget, then a fresh ``pl.Trainer`` is created, a new model is built, and
the previous stage's checkpoint is reloaded from disk (``resume_ckpt`` restores
weights + optimizer state + epoch counter).  The data module is shared, but the
per-stage ``fit()`` / checkpoint serialize–deserialize cycle is pure overhead,
and — more importantly — a fixed epoch budget is almost never the right place to
switch phases.  Empirically, structure optimisation continues *past* the point
where the (frozen) reconstruction is still faithful, so the residuals used for
structural signals degrade.

This module implements an **adaptive, in-memory** alternative:

- A single ``pl.Trainer.fit()`` call (one model, one optimizer, one data module).
- A :class:`PhaseController` callback that switches between two mutually
  exclusive phases at validation boundaries by toggling ``requires_grad`` on the
  gradient-routing parameter groups:

    * **reconstruct** — train ``_reconstruction_params`` only (structure frozen).
      Re-adapts the predictor to the *current* structure.  Stops on a validation
      reconstruction plateau (rate-of-improvement) or an epoch-budget cap.

    * **structure** — train ``_structural_params`` only (reconstruction frozen).
      Keeps learning structure against a *frozen, currently-good* predictor.  As
      structure drifts, the frozen predictor goes stale and ``val_x_mae`` rises.
      Stops when ``val_x_mae`` exceeds the per-phase best by a configurable
      fraction (default 20%) sustained for ``drop_patience`` validation epochs;
      OR when the structural signal itself (``val_hsic``) stops improving for
      ``hsic_patience`` validation epochs (an early switch that frees budget for
      an earlier reconstruction update instead of wasting epochs on a stalled
      HSIC); OR a safety epoch cap.

The schedule alternates reconstruct ↔ structure, starting (by default) with a
reconstruction warmup so structure always begins from a good predictor.  The run
is bounded by a **single resource budget** — the global epoch budget
(``total_epoch_budget`` → ``pl.Trainer.max_epochs``) — and does as many
reconstruct/structure cycles as fit within it.  ``max_cycles`` is an *optional*
safety guard (default: unbounded), not a budget: it only exists to cap
degenerate fast cycling (phases that exit almost immediately, each incurring a
checkpoint + DAG-diagnostics write).


Requirements
------------
The controller performs a *true* freeze via ``requires_grad_(False)`` on the
pre-classified parameter groups, so it requires ``use_gradient_routing=True``
(both ``SingleCausalForecaster`` and ``AttentionSelectorForecaster`` expose
``_structural_params`` / ``_reconstruction_params`` in that mode).

Example ``config['adaptive_training']`` block::

    adaptive_training:
      total_epoch_budget: 800          # THE budget: global cap = pl.Trainer max_epochs
      start_phase: reconstruct         # warm up the predictor first
      max_cycles: null                 # optional safety cap on # cycles
                                       # (null = unbounded: as many as fit the budget)

      starting_checkpoint: null        # optional warm-start (weights only)
      reset_optimizer_state_on_switch: false
      monitor: val_x_mae               # metric driving both triggers
      eval_dag: true                   # capture DAG diagnostics at each switch
      data_split_ratio: null           # cross-fit: fraction of train samples for
                                       # the reconstruct phase (null = off)
      run_final_evaluations: true      # run the standard post-training evaluation
                                       # suite on <save_dir> once the fit ends
                                       # (the functions themselves are selected by
                                       # the top-level ``evaluation.functions``)

      reconstruct:
        max_epochs: 100                # per-phase safety cap
        min_epochs: 0                  # floor: suppress plateau exit before N epochs
                                       # (applies to every reconstruct phase)
        warmup_min_epochs: 0           # larger floor for the INITIAL warmup phase
                                       # (falls back to min_epochs when unset)
        plateau_patience: 5            # stop after N val epochs w/o rel. improvement
        plateau_min_delta: 1.0e-4      # relative improvement threshold

      structure:
        max_epochs: 200                # per-phase safety cap
        lambda_hsic_cross: 0.1
        lambda_hsic_self: 0.0
        drop_pct: 0.20                 # switch when monitor rises 20% over phase best
        drop_patience: 5
        hsic_monitor: val_hsic         # structural signal watched for a plateau
        hsic_patience: 0               # switch after N val epochs w/o HSIC improvement
                                       # (0 = disabled; no behaviour change)
        hsic_min_delta: 1.0e-4         # relative HSIC improvement threshold
        min_epochs: 0                  # floor: suppress BOTH structure early-exits
                                       # (drop AND HSIC plateau) before N epochs
                                       # (max_epochs still wins)
"""


import copy
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from causaliT.training.callbacks import KFoldResultsTracker

# Reuse plain-container coercion, score-margin helper, cross-fit partitioner and
# JSON serializer from the rigid trainer so the two trainers stay consistent.
from causaliT.training.anm_staged_trainer import (
    _to_plain_container,
    _compute_score_margin,
    _json_default,
    _partition_train_indices,
)

logger = logging.getLogger(__name__)

# Numeric encoding of the active phase so it can be logged as a CSV metric
# alongside the loss curves (strings cannot be logged via ``self.log``).
# reconstruct → 0, structure → 1.
_PHASE_CODE = {"reconstruct": 0, "structure": 1}


# =============================================================================
# PHASE CONTROLLER CALLBACK
# =============================================================================


class PhaseController(Callback):
    """
    Metric-driven state machine alternating reconstruct ↔ structure phases.

    The controller mutates the *in-memory* module at validation boundaries;
    nothing is reloaded from disk.  Freezing is a true ``requires_grad`` freeze
    on the gradient-routing parameter groups, so exactly one group trains per
    phase.

    Cross-fitting (optional):
        When ``dm`` and ``stage_splits`` are supplied, the controller swaps the
        data module's training subset at each phase switch — the reconstruct
        phase trains on the ``reconstruct`` subset and the structure phase on
        the disjoint ``structure`` subset (DML/DARTS-style honesty: residual-HSIC
        is measured out-of-sample w.r.t. the reconstruction fit).  This requires
        the ``pl.Trainer`` to be created with ``reload_dataloaders_every_n_epochs=1``
        so Lightning re-queries ``dm.train_dataloader()`` after each switch.

    Args:
        config:          Full configuration dict (``adaptive_training`` block read).
        data_dir:        Root data directory (for DAG diagnostics).
        save_dir:        Parent save directory (transition checkpoints go under
                         ``<save_dir>/stage_checkpoints/``).
        cluster:         Suppress console prints when True.
        dm:              Data module to swap training subsets on (cross-fitting).
                         ``None`` disables cross-fitting.
        stage_splits:    Mapping ``{"reconstruct": idx, "structure": idx}`` of the
                         per-phase local training indices.  ``None`` disables
                         cross-fitting.
        val_local_idx:   Shared validation indices (kept constant across phases).
        test_idx:        Test indices (kept constant across phases).
    """

    def __init__(
        self,
        config: dict,
        data_dir: str,
        save_dir: str,
        cluster: bool,
        dm=None,
        stage_splits: Optional[Dict[str, np.ndarray]] = None,
        val_local_idx=None,
        test_idx=None,
    ):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.cluster = cluster

        # --- Cross-fit data-swap state ---
        self.dm = dm
        self.stage_splits = stage_splits
        self.val_local_idx = val_local_idx
        self.test_idx = test_idx
        self.cross_fitting: bool = dm is not None and bool(stage_splits)


        ad = _to_plain_container(config.get("adaptive_training", {})) or {}
        self.adaptive_cfg: Dict[str, Any] = ad

        self.monitor: str = str(ad.get("monitor", "val_x_mae"))
        self.start_phase: str = str(ad.get("start_phase", "reconstruct")).lower()
        # ``max_cycles`` is NOT a budget — ``total_epoch_budget`` (→ Trainer
        # max_epochs) is the single authoritative resource cap.  ``max_cycles``
        # is an OPTIONAL safety guard against degenerate fast cycling (phases
        # that exit almost immediately, each incurring a checkpoint + DAG
        # diagnostics write).  Default ``None`` ⇒ unbounded: the run does as
        # many reconstruct/structure cycles as fit within the epoch budget.
        _mc = ad.get("max_cycles", None)
        self.max_cycles: Optional[int] = None if _mc is None else int(_mc)

        self.reset_opt_on_switch: bool = bool(
            ad.get("reset_optimizer_state_on_switch", False)
        )
        self.eval_dag: bool = bool(ad.get("eval_dag", True))

        self.recon_cfg: Dict[str, Any] = _to_plain_container(ad.get("reconstruct", {})) or {}
        self.struct_cfg: Dict[str, Any] = _to_plain_container(ad.get("structure", {})) or {}

        # Reconstruct-phase triggers
        self.recon_max_epochs: int = int(self.recon_cfg.get("max_epochs", 100))
        self.plateau_patience: int = int(self.recon_cfg.get("plateau_patience", 5))
        self.plateau_min_delta: float = float(self.recon_cfg.get("plateau_min_delta", 1e-4))
        # Minimum-epoch floors: suppress the plateau-based early exit until the
        # phase has run at least this many epochs.  ``warmup_min_epochs`` applies
        # only to the INITIAL warmup reconstruct phase (phase_index 0 when the run
        # starts on reconstruct); ``min_epochs`` applies to every later reconstruct
        # phase.  ``warmup_min_epochs`` falls back to ``min_epochs`` when unset.
        self.recon_min_epochs: int = int(self.recon_cfg.get("min_epochs", 0))
        self.recon_warmup_min_epochs: int = int(
            self.recon_cfg.get("warmup_min_epochs", self.recon_min_epochs)
        )


        # Structure-phase triggers
        self.struct_max_epochs: int = int(self.struct_cfg.get("max_epochs", 200))
        self.drop_pct: float = float(self.struct_cfg.get("drop_pct", 0.20))
        self.drop_patience: int = int(self.struct_cfg.get("drop_patience", 5))
        # HSIC-plateau early switch: watch the structural signal (``hsic_monitor``,
        # lower is better) and switch back to reconstruct when it stops improving.
        # ``hsic_patience == 0`` disables it entirely (no behaviour change).  A
        # ``min_epochs`` floor suppresses the early exit for the first N epochs of
        # every structure phase; the ``max_epochs`` safety cap still takes
        # precedence over the floor.
        self.struct_hsic_monitor: str = str(
            self.struct_cfg.get("hsic_monitor", "val_hsic")
        )
        self.struct_hsic_patience: int = int(self.struct_cfg.get("hsic_patience", 0))
        self.struct_hsic_min_delta: float = float(
            self.struct_cfg.get("hsic_min_delta", 1e-4)
        )
        self.struct_min_epochs: int = int(self.struct_cfg.get("min_epochs", 0))


        # Model object (for per-arch lambda translation)
        self.model_obj: str = config.get("model", {}).get("model_object", "")

        # Output dir for transition checkpoints
        self.out_dir = Path(save_dir) / "stage_checkpoints"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # --- Runtime state ---
        self.current_phase: str = self.start_phase
        self._phase_start_epoch: int = 0
        self._phase_best: float = float("inf")
        self._plateau_counter: int = 0   # consecutive no-improve epochs (recon)
        self._drop_counter: int = 0      # consecutive over-threshold epochs (struct)
        self._hsic_best: float = float("inf")   # best (lowest) HSIC this struct phase
        self._hsic_plateau_counter: int = 0     # consecutive no-improve epochs (HSIC)
        self._cycle_count: int = 0       # completed structure phases
        self._phase_index: int = 0       # 0-based phase counter across the run


        # Records
        self.transitions: List[Dict[str, Any]] = []
        self.phase_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Phase application
    # ------------------------------------------------------------------
    def _resolve_param_groups(self, pl_module: pl.LightningModule):
        struct = getattr(pl_module, "_structural_params", None)
        recon = getattr(pl_module, "_reconstruction_params", None)
        if struct is None or recon is None:
            raise RuntimeError(
                "PhaseController requires use_gradient_routing=True so that the "
                "forecaster exposes _structural_params / _reconstruction_params. "
                "Set training.use_gradient_routing: true in the config."
            )
        return struct, recon

    def _apply_lambdas(self, pl_module: pl.LightningModule, lambdas: Dict[str, Any]) -> None:
        """Set loss-weight attributes on the module, translating per-arch names."""
        for key, val in lambdas.items():
            if not str(key).startswith("lambda"):
                continue
            fval = float(val)
            if self.model_obj == "AttentionSelectorLayer" and key == "lambda_hsic_cross":
                # Unified HSIC weight for AttentionSelectorLayer
                setattr(pl_module, "lambda_hsic", fval)
            elif hasattr(pl_module, key):
                setattr(pl_module, key, fval)

    def _apply_phase(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                     phase: str) -> None:
        struct_params, recon_params = self._resolve_param_groups(pl_module)

        if phase == "reconstruct":
            for p in struct_params:
                p.requires_grad_(False)
            for p in recon_params:
                p.requires_grad_(True)
        elif phase == "structure":
            for p in recon_params:
                p.requires_grad_(False)
            for p in struct_params:
                p.requires_grad_(True)
            # Apply structure-phase loss weights (e.g. lambda_hsic_cross)
            self._apply_lambdas(pl_module, self.struct_cfg)
        else:
            raise ValueError(f"Unknown phase {phase!r}")

        # Optionally clear stale optimizer moment estimates at the switch.
        # Optimizer.state must remain a defaultdict(dict); a plain {} would
        # break the ``self.state[p]`` access pattern inside optimizer.step().
        if self.reset_opt_on_switch:
            for opt in trainer.optimizers:
                opt.state = defaultdict(dict)

        # Cross-fit: point the data module at this phase's training subset.
        # The pl.Trainer is created with reload_dataloaders_every_n_epochs=1, so
        # the next epoch re-queries dm.train_dataloader() and picks it up.
        n_subset = self._swap_train_subset(phase)

        self.current_phase = phase
        self._phase_start_epoch = trainer.current_epoch
        self._phase_best = float("inf")
        self._plateau_counter = 0
        self._drop_counter = 0
        self._hsic_best = float("inf")
        self._hsic_plateau_counter = 0


        # Always emit to the Python logger so the active stage is visible in
        # cluster log files (where console ``print`` is suppressed).
        logger.info(
            "[adaptive] entering phase '%s' @ global_epoch=%d "
            "(phase_index=%d, cycle=%d%s)",
            phase, trainer.current_epoch, self._phase_index, self._cycle_count,
            f", n_train={n_subset}" if n_subset is not None else "",
        )

        if not self.cluster:
            subset_msg = f" (n_train={n_subset})" if n_subset is not None else ""
            print(f"  [adaptive] -> phase '{phase}' at global epoch "
                  f"{trainer.current_epoch}{subset_msg}")


    def _swap_train_subset(self, phase: str) -> Optional[int]:
        """
        Point the data module at ``phase``'s cross-fit training subset.

        Returns the subset size (for logging), or ``None`` when cross-fitting is
        disabled or the phase has no dedicated subset.  Validation/test indices
        are kept constant so stage-to-stage metrics remain comparable.
        """
        if not self.cross_fitting or self.dm is None:
            return None
        # Preferred path: the datamodule owns the phase→subset mapping.
        if hasattr(self.dm, "set_active_phase"):
            return self.dm.set_active_phase(phase)
        # Fallback for datamodules without the stage-split API.
        if self.stage_splits is None:
            return None
        subset = self.stage_splits.get(phase)
        if subset is None:
            return None
        self.dm.update_idx(
            train_idx=subset,
            val_idx=self.val_local_idx,
            test_idx=self.test_idx,
        )
        return int(len(subset))




    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Apply the initial phase after optimizers exist and after the module's
        # own on_fit_start (which honours config freeze flags — left False here).
        self._apply_phase(trainer, pl_module, self.start_phase)

    def _capture_dag(self, trainer, pl_module, label: str) -> Dict[str, Any]:
        """Capture DAG diagnostics + score margin without corrupting train mode."""
        result: Dict[str, Any] = {
            "phase": self.current_phase,
            "phase_index": self._phase_index,
            "epoch": trainer.current_epoch,
            "label": label,
        }
        if not self.eval_dag:
            return result

        was_training = pl_module.training
        try:
            try:
                from causaliT.training.causal_initialization import (
                    evaluate_dag_from_model,
                )
                dag_metrics = evaluate_dag_from_model(
                    pl_module, self.config, self.data_dir
                )
                for k, v in dag_metrics.items():
                    if not isinstance(v, np.ndarray):
                        result[k] = v
            except Exception as exc:
                logger.debug(f"PhaseController: evaluate_dag_from_model failed: {exc}")

            margin = _compute_score_margin(pl_module, self.config, self.data_dir)
            result["score_margin_cross"] = margin.get("cross")
            result["score_margin_self"] = margin.get("self")
        finally:
            if was_training:
                pl_module.train()
        return result

    def _record_transition(self, trainer, pl_module, reason: str,
                           from_phase: str, to_phase: str, monitor_val: float) -> None:
        diag = self._capture_dag(trainer, pl_module, label=f"end_{from_phase}")
        ckpt_path = self.out_dir / (
            f"phase_{self._phase_index:02d}_{from_phase}_end.ckpt"
        )
        try:
            trainer.save_checkpoint(str(ckpt_path))
        except Exception as exc:
            logger.warning(f"PhaseController: failed to save checkpoint: {exc}")

        record = {
            "phase_index": self._phase_index,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "reason": reason,
            "global_epoch": trainer.current_epoch,
            "phase_epochs": trainer.current_epoch - self._phase_start_epoch + 1,
            "monitor": self.monitor,
            "monitor_value": float(monitor_val),
            "phase_best": (None if self._phase_best == float("inf")
                           else float(self._phase_best)),
            "checkpoint": str(ckpt_path),
            "dag_diagnostics": diag,
        }
        self.transitions.append(record)
        self.phase_rows.append({
            "phase_index": self._phase_index,
            "phase": from_phase,
            "end_reason": reason,
            "global_epoch_end": trainer.current_epoch,
            "phase_epochs": record["phase_epochs"],
            f"end_{self.monitor}": float(monitor_val),
            **{f"dag_{k}": v for k, v in diag.items()
               if k not in ("phase", "phase_index", "epoch", "label")},
        })

        if not self.cluster:
            print(f"  [adaptive] transition ({reason}): {from_phase} -> {to_phase} "
                  f"| {self.monitor}={monitor_val:.5f}")


    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = float(metrics[self.monitor])
        if not np.isfinite(current):
            return

        phase_epochs = trainer.current_epoch - self._phase_start_epoch + 1

        # Log the active stage as numeric CSV metrics so the phase can be
        # aligned with the loss curves (0 = reconstruct, 1 = structure).
        pl_module.log(
            "adaptive_phase",
            float(_PHASE_CODE.get(self.current_phase, -1)),
            on_step=False, on_epoch=True,
        )
        pl_module.log(
            "adaptive_phase_epochs", float(phase_epochs),
            on_step=False, on_epoch=True,
        )
        pl_module.log(
            "adaptive_cycle", float(self._cycle_count),
            on_step=False, on_epoch=True,
        )


        # ---------------- Reconstruct phase: plateau / budget ----------------
        if self.current_phase == "reconstruct":
            # Relative improvement check
            if current <= self._phase_best * (1.0 - self.plateau_min_delta):
                self._phase_best = current
                self._plateau_counter = 0
            else:
                if current < self._phase_best:
                    self._phase_best = current
                self._plateau_counter += 1

            # Minimum-epoch floor: the plateau early-exit is suppressed until the
            # phase has run at least ``effective_min`` epochs.  The initial warmup
            # reconstruct phase (phase_index 0 with start_phase=reconstruct) uses
            # the larger ``warmup_min_epochs`` floor to guarantee a fully-formed
            # predictor before any structure learning begins.  The ``max_epochs``
            # safety cap always takes precedence over the floor.
            is_warmup = self._phase_index == 0 and self.start_phase == "reconstruct"
            effective_min = (
                self.recon_warmup_min_epochs if is_warmup else self.recon_min_epochs
            )
            min_epochs_reached = phase_epochs >= effective_min

            plateaued = (
                self._plateau_counter >= self.plateau_patience
                and min_epochs_reached
            )
            budget_hit = phase_epochs >= self.recon_max_epochs

            if plateaued or budget_hit:
                reason = "recon_plateau" if plateaued else "recon_budget"

                self._record_transition(
                    trainer, pl_module, reason,
                    from_phase="reconstruct", to_phase="structure",
                    monitor_val=current,
                )
                self._phase_index += 1
                self._apply_phase(trainer, pl_module, "structure")

        # ---------- Structure phase: drop / HSIC plateau / budget ----------
        elif self.current_phase == "structure":
            if current < self._phase_best:
                self._phase_best = current

            threshold = self._phase_best * (1.0 + self.drop_pct)
            if current > threshold:
                self._drop_counter += 1
            else:
                self._drop_counter = 0

            # HSIC-plateau tracking: watch the structural signal (lower is
            # better) and count consecutive validation epochs without a relative
            # improvement.  Counting is disabled when ``hsic_patience == 0`` or
            # the metric is unavailable / non-finite this epoch.
            hsic_counter_ready = False
            hsic_val = metrics.get(self.struct_hsic_monitor)
            if self.struct_hsic_patience > 0 and hsic_val is not None:
                hsic_current = float(hsic_val)
                if np.isfinite(hsic_current):
                    if hsic_current <= self._hsic_best * (1.0 - self.struct_hsic_min_delta):
                        self._hsic_best = hsic_current
                        self._hsic_plateau_counter = 0
                    else:
                        if hsic_current < self._hsic_best:
                            self._hsic_best = hsic_current
                        self._hsic_plateau_counter += 1
                    hsic_counter_ready = (
                        self._hsic_plateau_counter >= self.struct_hsic_patience
                    )

            pl_module.log(
                "adaptive_hsic_plateau", float(self._hsic_plateau_counter),
                on_step=False, on_epoch=True,
            )

            # ``min_epochs`` is a symmetric floor for the whole structure phase:
            # it suppresses BOTH early-exit triggers (the stale-predictor drop and
            # the HSIC plateau) until the phase has run at least this many epochs,
            # so early structure-learning latency does not cause a premature
            # switch.  The counters keep accumulating during the floor window, so
            # a pending exit fires the moment the floor clears.  The ``max_epochs``
            # safety cap always takes precedence over the floor.
            min_epochs_reached = phase_epochs >= self.struct_min_epochs

            dropped = (
                self._drop_counter >= self.drop_patience and min_epochs_reached
            )
            hsic_plateaued = hsic_counter_ready and min_epochs_reached
            budget_hit = phase_epochs >= self.struct_max_epochs


            if dropped or hsic_plateaued or budget_hit:
                # Reason precedence: a stale predictor (drop) first, then a
                # stalled structural signal (HSIC plateau), then the safety cap.
                if dropped:
                    reason = "struct_drop"
                elif hsic_plateaued:
                    reason = "struct_hsic_plateau"
                else:
                    reason = "struct_budget"
                self._cycle_count += 1


                # Optional safety guard: stop only when an explicit max_cycles is
                # configured and has been reached.  When ``max_cycles is None``
                # (default) the run is bounded solely by ``total_epoch_budget``
                # (Trainer max_epochs) — i.e. it does as many cycles as fit.
                if self.max_cycles is not None and self._cycle_count >= self.max_cycles:

                    self._record_transition(
                        trainer, pl_module, f"{reason}_final",
                        from_phase="structure", to_phase="stop",
                        monitor_val=current,
                    )
                    self._phase_index += 1
                    if not self.cluster:
                        print(f"  [adaptive] max_cycles={self.max_cycles} reached "
                              f"- stopping.")

                    trainer.should_stop = True
                    return

                self._record_transition(
                    trainer, pl_module, reason,
                    from_phase="structure", to_phase="reconstruct",
                    monitor_val=current,
                )
                self._phase_index += 1
                self._apply_phase(trainer, pl_module, "reconstruct")


# =============================================================================
# OUTPUT-LAYOUT HELPERS (evaluation-suite compatibility)
# =============================================================================

def _save_config_snapshot(config: dict, save_dir: str) -> Optional[str]:
    """
    Persist the *resolved* run config as ``<save_dir>/config.yaml``.

    The evaluation suite (``eval_attention_scores`` / ``eval_interventions``)
    locates the experiment config by globbing ``config*.yaml`` in the experiment
    root and taking the first hit.  A run launched through the CLI already has a
    hand-written ``config_*.yaml`` there; the sweeper writes ``config.yaml``
    before training.  In both cases a file exists and we must NOT add a second
    candidate (it would make the "first hit" ambiguous), so this is a no-op then.

    When nothing is present (e.g. ``save_dir`` is a fresh scratch directory), we
    write the *resolved* config — sequence lengths populated from the dataset,
    ``k_fold=1`` and the adaptive epoch budget applied — so the run is
    self-describing for offline evaluation.  Mirrors
    ``anm_staged_trainer._save_stage_config``.

    Returns the path written, or ``None`` when a config was already present.
    """
    existing = glob.glob(str(Path(save_dir) / "config*.yaml"))
    if existing:
        return None

    config_path = Path(save_dir) / "config.yaml"
    try:
        cfg = OmegaConf.create(_to_plain_container(config))
        OmegaConf.save(config=cfg, f=str(config_path), resolve=True)
    except Exception as exc:
        logger.warning(
            "adaptive_trainer: failed to write config snapshot to %s: %s",
            config_path, exc,
        )
        return None
    return str(config_path)


def _write_kfold_summary(save_dir: str, fold_metrics: dict) -> None:
    """
    Write ``<save_dir>/kfold_summary.json`` for the single adaptive fold.

    The default evaluation suite maintains this file
    (``fix_kfold_summary`` / ``enrich_kfold_summary``) and the experiments
    manifest reads its aggregated statistics.  ``trainer()`` produces it via
    ``KFoldResultsTracker``; the adaptive run does the same for its one fold so
    both trainers leave an identical artefact set behind.

    ``fold_metrics`` is copied before the private ``_best_checkpoint_path`` key
    is popped, so the caller's dict (used for the adaptive summary JSON) is left
    untouched.
    """
    metrics = dict(fold_metrics)
    best_ckpt_path = metrics.pop("_best_checkpoint_path", None)
    try:
        tracker = KFoldResultsTracker(str(save_dir), k_folds=1)
        tracker.add_fold_result(0, metrics, best_ckpt_path)
    except Exception as exc:
        logger.warning("adaptive_trainer: failed to write kfold_summary.json: %s", exc)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def adaptive_trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str = "NA",
    debug: bool = False,
    best: bool = False,
) -> pd.DataFrame:
    """
    Adaptive alternating trainer: metric-driven Structure/Reconstruct schedule.

    Runs a single in-memory ``pl.Trainer.fit()`` with a :class:`PhaseController`
    callback that switches phases based on ``config['adaptive_training']``.

    Args:
        config:          Full configuration dict (``adaptive_training`` required,
                         ``training.use_gradient_routing`` must be True).
        data_dir:        Root data directory.
        save_dir:        Parent save directory.  Training output goes directly
                         under ``<save_dir>/k_0/`` (same layout as ``trainer()``)
                         and phase-transition checkpoints under
                         ``<save_dir>/stage_checkpoints/``.
        cluster:         Suppress progress bar / use 1-GPU mode.
        experiment_tag:  Passed to ``train_single_fold`` for the run manifest.
        debug:           Enable anomaly detection, memory logger, etc.
        best:            If True, collect best-checkpoint metrics.

    Returns:
        pd.DataFrame: One row per completed phase with end metrics and DAG
        diagnostics.
    """
    from causaliT.training.trainer import (
        get_dataloader,
        _make_fold_splits,
        create_model_instance,
        train_single_fold,
        _run_post_training_evaluations,
    )
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset

    ad_cfg = _to_plain_container(config.get("adaptive_training", {})) or {}
    if not ad_cfg:
        raise ValueError(
            "config['adaptive_training'] is empty or missing. Define the "
            "adaptive schedule block (see module docstring)."
        )

    if not config["training"].get("use_gradient_routing", False):
        raise ValueError(
            "adaptive_trainer requires training.use_gradient_routing=True so "
            "that structural/reconstruction parameter groups can be frozen "
            "independently. Enable it in the config."
        )

    seed = config["training"].get("seed", 42)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    config = populate_seq_lengths_from_dataset(config, data_dir)

    # --- Shared data module and fold splits (single fold) ---
    dm = get_dataloader(config, data_dir, cluster, seed)
    dm.prepare_data()

    # Force single-fold behaviour for the adaptive run.
    config = copy.deepcopy(config)
    config["training"]["k_fold"] = 1
    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config, dm, seed, data_dir=data_dir
    )
    train_local_idx, val_local_idx = fold_splits[0]

    # --- Global epoch budget → pl.Trainer max_epochs ---
    total_budget = int(ad_cfg.get("total_epoch_budget",
                                  config["training"].get("max_epochs", 800)))
    config["training"]["max_epochs"] = total_budget
    if config["training"].get("save_ckpt_every_n_epochs") is None:
        config["training"]["save_ckpt_every_n_epochs"] = total_budget

    # Training output goes straight into save_dir (train_single_fold appends the
    # ``k_{fold}`` subfolder), matching the layout produced by ``trainer()``.
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Optional warm-start (weights only)
    starting_ckpt: Optional[str] = _to_plain_container(
        ad_cfg.get("starting_checkpoint", None)
    )

    # --- Cross-fitting (optional) --------------------------------------------
    # When ``data_split_ratio`` is set (in the open interval (0, 1)), partition
    # the fold's training indices into two disjoint subsets: the reconstruct
    # phase trains on ``recon`` and the structure phase on ``struct`` (DML/DARTS
    # honest cross-fit — residual-HSIC is out-of-sample w.r.t. the reconstruction
    # fit).  ``None`` / out-of-range disables it (both phases use the full set).
    data_split_ratio = ad_cfg.get("data_split_ratio", None)
    stage_splits: Optional[Dict[str, np.ndarray]] = None
    reload_every_n = 0
    active_split_ratio: Optional[float] = None
    start_phase = str(ad_cfg.get("start_phase", "reconstruct")).lower()

    if data_split_ratio is not None and 0.0 < float(data_split_ratio) < 1.0:
        active_split_ratio = float(data_split_ratio)
        recon_idx, struct_idx = _partition_train_indices(
            train_local_idx, active_split_ratio, seed
        )

        stage_splits = {"reconstruct": recon_idx, "structure": struct_idx}
        # The datamodule OWNS the phase→subset mapping; the controller only
        # requests a phase by name (dm.set_active_phase).  val/test are held
        # constant so stage-to-stage metrics stay comparable.
        dm.set_stage_splits(stage_splits, val_idx=val_local_idx, test_idx=test_idx)
        # Start the shared fit on the subset of the starting phase so the first
        # epoch already trains on the correct partition.
        train_local_idx = stage_splits.get(start_phase, train_local_idx)
        # Lightning must re-query dm.train_dataloader() after each phase switch,
        # so reload the train dataloader every epoch.  Reloading every epoch
        # respawns the ENTIRE worker pool each epoch: on Windows (spawn) each
        # worker re-imports the package and re-copies the dataset tensors, which
        # crashes the session at the first epoch boundary; on Linux (fork) it is
        # cheaper but still leaks memory over long runs.  The dataset is an
        # in-memory TensorDataset (batches are index-selects on RAM tensors, no
        # I/O to overlap), so workers add pure overhead and no throughput —
        # force single-process loading (num_workers=0) on the reload path.
        # Correct and faster on BOTH Windows and the cluster.
        reload_every_n = 1
        dm.num_workers = 0
        dm.persistent_workers = False


        if not cluster:
            print(
                f"  Cross-fit data splits (ratio={data_split_ratio}): "
                f"reconstruct={len(recon_idx)}, structure={len(struct_idx)}"
            )

    # --- Build model once ---
    seed_everything(seed)
    model = create_model_instance(config, data_dir)

    # --- Phase controller ---
    controller = PhaseController(
        config=config, data_dir=data_dir, save_dir=save_dir, cluster=cluster,
        dm=dm if stage_splits is not None else None,
        stage_splits=stage_splits,
        val_local_idx=val_local_idx,
        test_idx=test_idx,
    )


    if not cluster:
        print("\n" + "=" * 70)
        print("ADAPTIVE ALTERNATING TRAINING")
        print(f"  total_epoch_budget : {total_budget}")
        print(f"  start_phase        : {controller.start_phase}")
        print(f"  monitor            : {controller.monitor}")
        print(f"  structure trigger  : +{controller.drop_pct:.0%} for "
              f"{controller.drop_patience} epochs (cap {controller.struct_max_epochs})")
        print(f"  structure floor    : min_epochs {controller.struct_min_epochs} "
              f"(suppresses drop + HSIC early-exits)")
        if controller.struct_hsic_patience > 0:
            print(f"  structure HSIC exit: {controller.struct_hsic_monitor} "
                  f"plateau patience {controller.struct_hsic_patience}")


        print(f"  reconstruct trigger: plateau patience "
              f"{controller.plateau_patience} (cap {controller.recon_max_epochs})")
        print(f"  reconstruct floor  : min_epochs {controller.recon_min_epochs}, "
              f"warmup_min_epochs {controller.recon_warmup_min_epochs}")
        print(f"  max_cycles         : "
              f"{'unbounded (epoch-budget only)' if controller.max_cycles is None else controller.max_cycles}")


        print("=" * 70)

    # --- Single in-memory fit ---
    fold_metrics = train_single_fold(
        config=config,
        model=model,
        dm=dm,
        fold=0,
        train_local_idx=train_local_idx,
        val_local_idx=val_local_idx,
        test_idx=test_idx,
        train_val_idx=train_val_idx,
        save_dir=str(save_dir),
        trainable_params=0,
        cluster=cluster,
        resume_ckpt=None,
        warm_start_ckpt=starting_ckpt,
        experiment_tag=f"{experiment_tag}_adaptive",
        debug=debug,
        best=best,
        extra_callbacks=[controller],
        reload_dataloaders_every_n_epochs=reload_every_n,
    )

    # --- Evaluation-suite artefacts --------------------------------------------
    # The default evaluation suite (causaliT/evaluation/eval_funs) expects the
    # standard experiment layout: a ``config*.yaml`` in the experiment root, one
    # ``k_*`` fold folder with checkpoints (produced by train_single_fold) and a
    # ``kfold_summary.json`` it fixes/enriches.  Emit the two root-level files
    # here so an adaptive run is indistinguishable from a ``trainer()`` run as
    # far as evaluation is concerned.
    _write_kfold_summary(save_dir, fold_metrics)
    _save_config_snapshot(config, save_dir)

    # --- Summary JSON ---
    summary = {
        "experiment_tag": experiment_tag,
        "total_epoch_budget": total_budget,
        "start_phase": controller.start_phase,
        "monitor": controller.monitor,
        "cross_fitting": stage_splits is not None,
        "data_split_ratio": active_split_ratio,
        "n_train_reconstruct": (int(len(stage_splits["reconstruct"]))
                                if stage_splits is not None else None),
        "n_train_structure": (int(len(stage_splits["structure"]))
                              if stage_splits is not None else None),
        "n_transitions": len(controller.transitions),

        "n_cycles": controller._cycle_count,
        "final_metrics": {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in fold_metrics.items()
        },
        "transitions": controller.transitions,
    }
    summary_path = Path(save_dir) / "adaptive_training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    if not cluster:
        print("\n" + "=" * 70)
        print("ADAPTIVE TRAINING COMPLETE")
        print(f"  transitions : {len(controller.transitions)}")
        print(f"  cycles      : {controller._cycle_count}")
        print(f"  summary     : {summary_path}")
        print("=" * 70)

    # --- Post-training evaluations --------------------------------------------
    # Same dispatcher (and same failure isolation) as ``trainer()``: WHICH
    # functions run is controlled by ``config['evaluation']['functions']``;
    # ``adaptive_training.run_final_evaluations: false`` skips the step entirely
    # (useful for long sweeps that evaluate all arms in one later pass).
    if bool(ad_cfg.get("run_final_evaluations", True)):
        _run_post_training_evaluations(config, str(save_dir), data_dir)

    df = pd.DataFrame(controller.phase_rows)
    return df


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def run_adaptive_trainer_from_config(
    config_path: str,
    data_dir: str,
    save_dir: str,
    cluster: bool = False,
    experiment_tag: str = "NA",
) -> pd.DataFrame:
    """Run adaptive alternating training directly from a YAML config path."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    return adaptive_trainer(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
    )
