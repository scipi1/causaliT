"""
Adaptive Trainer: metric-driven alternating Structure/Reconstruct training.

Motivation
----------
The predecessor (the now-removed ANM staged trainer) ran a *rigid* schedule:
each stage trained for a fixed epoch budget, then a fresh ``pl.Trainer`` was
created, a new model built, and the previous stage's checkpoint reloaded from
disk.  The data module was shared, but the per-stage ``fit()`` / checkpoint
serialize-deserialize cycle was pure overhead, and — more importantly — a fixed
epoch budget is almost never the right place to switch phases.  Empirically,
structure optimisation continues *past* the point where the (frozen)
reconstruction is still faithful, so the residuals used for structural signals
degrade.

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

An optional **final reconstruction-only phase** (``final_reconstruct``) can be
appended AFTER the alternating schedule: the structural parameters stay frozen
at their learned values, the cross-fit data split is disabled (the full
training set is used), and the predictor is refined against the frozen
structure - empirically useful for polishing the reconstruction before the
model is used to estimate the ATE.  Its epochs are ADDED ON TOP of
``total_epoch_budget`` (``pl.Trainer.max_epochs = total_epoch_budget +
final_reconstruct.max_epochs``), so the alternating schedule keeps its full
budget.  The phase exits on a validation reconstruction plateau (same
rate-of-improvement trigger as the reconstruct phase) or its own epoch cap,
then the run stops.


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
      swap_splits: false               # exchange the recon/structure subsets after
                                       # each completed cycle (needs data_split_ratio);
                                       # structure always stays disjoint from the
                                       # previous reconstruct split
      run_final_evaluations: true      # run the standard post-training evaluation
                                       # suite on <save_dir> once the fit ends
                                       # (the functions themselves are selected by
                                       # the top-level ``evaluation.functions``)

      final_reconstruct:               # OPTIONAL final reconstruction-only phase
        enabled: false                 # appended AFTER the alternating schedule
        max_epochs: 100                # (Trainer max_epochs = total_epoch_budget
                                       # + these epochs).  Structure stays frozen,
                                       # cross-fit split OFF (full training set).
        min_epochs: 0                  # floor: suppress plateau exit before N epochs
        plateau_patience: 5            # stop after N val epochs w/o rel. improvement
                                       # (falls back to the reconstruct values)
        plateau_min_delta: 1.0e-4      # relative improvement threshold (ditto)

      # Prior-softmax gain (GainSoftmax modules; see
      # causaliT/core/modules/gain_softmax.py).  The interpolation weight lambda
      # ramps 0 -> gain_lambda_final DURING the alternating schedule, driven by
      # the GLOBAL epoch (phase-agnostic): the gate's role morphs from the
      # multiplicative weight to the softmax support while the trainer keeps
      # alternating reconstruct <-> structure.  No separate final phase.
      gain_lambda_start: null          # global epoch where the turn-on begins
                                       # (null -> 0.5 * total_epoch_budget)
      gain_lambda_ramp: 0              # epochs to ramp 0 -> final (0 = jump)
      gain_lambda_final: 1.0           # ramp target (1.0 = full prior-softmax)

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
        # HSIC-progress gates: disarm a regularizer entirely once the run-best
        # HSIC has stalled for <reg>_gate_patience consecutive structure phases
        # (re-armed on improvement).  Both gates share the run-best HSIC
        # progress signal but close independently at their own patience.
        l0_gate_on_hsic: false         # gate the L0 weight lambda_l0
        kappa_gate_on_hsic: false      # gate the NOTEARS weight kappa
"""


import copy
import glob
import json
import logging
import os
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

# Plain-container coercion, score-margin helper, cross-fit partitioner and JSON
# serializer shared with the rest of the staged-training utilities.
from causaliT.training.staging_utils import (
    _to_plain_container,
    _compute_score_margin,
    _json_default,
    _partition_train_indices,
)

logger = logging.getLogger(__name__)

# Numeric encoding of the active phase so it can be logged as a CSV metric
# alongside the loss curves (strings cannot be logged via ``self.log``).
# reconstruct -> 0, structure -> 1, final_reconstruct -> 2.
_PHASE_CODE = {"reconstruct": 0, "structure": 1, "final_reconstruct": 2}


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

        With ``swap_splits`` enabled the two subsets are additionally exchanged
        after each completed recon+structure cycle (recon_1(I_1), struct_1(I_2),
        recon_2(I_2), struct_2(I_1), ...).  Within every recon->struct pairing
        the subsets stay disjoint — structure never reuses the split of the
        reconstruction that preceded it — so the honesty property is preserved
        while each sample serves both roles across the run.

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

        # Cross-fit split swapping: exchange the reconstruct/structure training
        # subsets at every completed recon+structure cycle (i.e. at each
        # structure -> reconstruct transition).  Within every recon->struct
        # pairing the two subsets stay disjoint, so residual-HSIC remains
        # out-of-sample w.r.t. the reconstruction fit (DML/DARTS honesty
        # preserved) while each sample serves both roles across the run.
        # Requires cross-fitting; forced off (with a warning) when no split is
        # active.
        self.swap_splits: bool = bool(ad.get("swap_splits", False))
        if self.swap_splits and not self.cross_fitting:
            logger.warning(
                "[adaptive] swap_splits=true but cross-fitting is disabled "
                "(data_split_ratio not in (0, 1)) - ignoring swap_splits."
            )
            self.swap_splits = False

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

        # HSIC-progress gates ("regularizers trim after HSIC ranks"): an armed
        # regularizer is applied only while the structural signal
        # (``hsic_monitor``, lower is better) still improves at RUN level.
        # Once ``<reg>_gate_patience`` consecutive structure phases have failed
        # to beat the run-best HSIC by ``<reg>_gate_min_delta`` (relative), the
        # regularizer's coefficient is applied as 0 on structure-phase entry,
        # so it can never act as the SOLE structure force (e.g. uniform gate
        # deflation once HSIC is exhausted, or NOTEARS driving structure into
        # the ill region).  A later improvement re-arms it.  Both gates share
        # the same run-best HSIC progress signal and stall counter, but close
        # independently at their own patience.  ``<reg>_gate_on_hsic: false``
        # (default) is a no-op (backward-compatible).
        self.l0_gate_on_hsic: bool = bool(self.struct_cfg.get("l0_gate_on_hsic", False))
        self.l0_gate_patience: int = int(self.struct_cfg.get("l0_gate_patience", 1))
        self.l0_gate_min_delta: float = float(
            self.struct_cfg.get("l0_gate_min_delta", self.struct_hsic_min_delta)
        )
        # NOTEARS gate: same mechanism applied to the acyclicity weight kappa.
        self.kappa_gate_on_hsic: bool = bool(
            self.struct_cfg.get("kappa_gate_on_hsic", False)
        )
        self.kappa_gate_patience: int = int(
            self.struct_cfg.get("kappa_gate_patience", 1)
        )
        self.kappa_gate_min_delta: float = float(
            self.struct_cfg.get("kappa_gate_min_delta", self.struct_hsic_min_delta)
        )

        # Final reconstruction-only phase (optional, APPENDED after the
        # alternating schedule): structure stays frozen, cross-fit split OFF
        # (full training set).  Refines the predictor against the frozen,
        # learned structure before the model is used for ATE estimation.
        self.final_cfg: Dict[str, Any] = _to_plain_container(
            ad.get("final_reconstruct", {})
        ) or {}
        self.final_enabled: bool = bool(self.final_cfg.get("enabled", False))
        self.final_max_epochs: int = int(self.final_cfg.get("max_epochs", 100))
        self.final_min_epochs: int = int(self.final_cfg.get("min_epochs", 0))
        # Plateau triggers fall back to the reconstruct-phase values so the
        # final phase behaves like a regular reconstruct phase unless
        # explicitly overridden.
        self.final_plateau_patience: int = int(
            self.final_cfg.get("plateau_patience", self.plateau_patience)
        )
        self.final_plateau_min_delta: float = float(
            self.final_cfg.get("plateau_min_delta", self.plateau_min_delta)
        )
        # Prior-softmax gain schedule (GainSoftmax modules; see
        # causaliT/core/modules/gain_softmax.py).  The interpolation weight
        # lambda ramps 0 -> gain_lambda_final DURING the alternating schedule,
        # driven by the GLOBAL epoch (phase-agnostic).  No-op when the model
        # owns no gain module (``set_gain_lambda`` returns 0).
        #   gain_lambda_start:  global epoch where the turn-on begins
        #                       (None -> 0.5 * total_epoch_budget)
        #   gain_lambda_ramp:   epochs to ramp 0 -> final (0 = jump)
        #   gain_lambda_final:  ramp target (1.0 = full prior-softmax)
        _gain_start = ad.get("gain_lambda_start", None)
        _total_budget = ad.get("total_epoch_budget", None)
        if _gain_start is None and _total_budget is not None:
            _gain_start = int(0.5 * int(_total_budget))
        self.gain_lambda_start: Optional[int] = (
            None if _gain_start is None else int(_gain_start)
        )
        self.gain_lambda_ramp: int = int(ad.get("gain_lambda_ramp", 0))
        self.gain_lambda_final: float = float(ad.get("gain_lambda_final", 1.0))
        if self.final_enabled and self.final_max_epochs <= 0:
            logger.warning(
                "[adaptive] final_reconstruct.enabled=true but max_epochs=%d "
                "<= 0 - disabling the final phase.",
                self.final_max_epochs,
            )
            self.final_enabled = False


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
        self._struct_phase_count: int = 0  # STARTED structure phases (1 = first)
        # HSIC-progress gate run state (used only when a ``*_gate_on_hsic``).
        self._l0_base: Optional[float] = None  # configured lambda_l0 (lazy)
        self._l0_active: bool = True
        self._kappa_base: Optional[float] = None  # configured kappa (lazy)
        self._kappa_active: bool = True
        self._hsic_run_best: float = float("inf")
        self._hsic_stall_cycles: int = 0
        self._hsic_phase_best_gate: float = float("inf")
        # Split key the ACTIVE phase is actually training on (may differ from
        # the phase name when ``swap_splits`` swaps the subsets each cycle).
        # ``None`` until the first cross-fit swap / when cross-fitting is off.
        self._active_split_key: Optional[str] = None



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
        """Set loss-weight attributes on the module, translating per-arch names.

        Recognised keys: ``lambda_*`` plus ``kappa`` (the NOTEARS weight, the
        only non-lambda loss coefficient) — the latter lets the structure
        phase and the HSIC-progress gate steer NOTEARS too.
        """
        for key, val in lambdas.items():
            if not (str(key).startswith("lambda") or str(key) == "kappa"):
                continue
            fval = float(val)
            if self.model_obj == "AttentionSelectorLayer" and key == "lambda_hsic_cross":
                # Unified HSIC weight for AttentionSelectorLayer
                setattr(pl_module, "lambda_hsic", fval)
            elif hasattr(pl_module, key):
                setattr(pl_module, key, fval)

    # Descendant-excluding HSIC knobs that may be overridden PER PHASE.
    # See causaliT.utils.descendant_mask and
    # AttentionSelectorForecaster._build_hsic_descendant_mask.
    #
    # Why per phase: the mask is derived from the adjacency that the masked HSIC
    # is itself training, so it carries a self-confirmation risk — turning it on
    # while the graph is still ~random can delete exactly the parent signal we
    # are looking for and lock in a wrong orientation.  The adaptive schedule
    # gives a much better curriculum handle than a raw epoch count: leave it OFF
    # for the first structure phase(s) and switch it ON once the predictor and
    # adjacency have co-adapted.  Only keys actually PRESENT in the phase block
    # are touched, so omitting them keeps the ``training:``-level value.
    _DESCENDANT_MASK_KEYS = (
        "hsic_exclude_descendants",
        "hsic_descendant_threshold",
        "hsic_descendant_hops",
        "hsic_descendant_exclude_self",
        "hsic_descendant_weight",
        "hsic_descendant_warmup_epochs",
        "hsic_descendant_min_kept_frac",
        "hsic_descendant_ema",
    )

    def _gated_struct_cfg(self, pl_module: pl.LightningModule) -> Dict[str, Any]:
        """Structure-phase loss weights with the HSIC-progress gates applied.

        "Regularizers trim after HSIC ranks": once the run-level structural
        signal has failed to improve for ``<reg>_gate_patience`` consecutive
        structure phases, that regularizer's coefficient is applied as 0 so it
        never acts as the SOLE structure force.  L0 (``l0_gate_on_hsic``) and
        NOTEARS (``kappa_gate_on_hsic``) share the same run-best HSIC progress
        signal but close independently at their own patience.  No-op (the raw
        ``struct_cfg``) when both gates are off.
        """
        if not (self.l0_gate_on_hsic or self.kappa_gate_on_hsic):
            return self.struct_cfg
        cfg = dict(self.struct_cfg)
        if self.l0_gate_on_hsic:
            if self._l0_base is None:
                # Base = the configured structure-phase value, falling back to
                # the module's current (training-level) lambda_l0.
                self._l0_base = float(self.struct_cfg.get(
                    "lambda_l0", getattr(pl_module, "lambda_l0", 0.0)))
            cfg["lambda_l0"] = self._l0_base if self._l0_active else 0.0
            if not self._l0_active:
                logger.info(
                    "[adaptive] L0 gate CLOSED (HSIC stalled for %d structure "
                    "phase(s)): lambda_l0 applied as 0.0 (base=%.3g).",
                    self._hsic_stall_cycles, self._l0_base,
                )
        if self.kappa_gate_on_hsic:
            if self._kappa_base is None:
                # Base = the configured structure-phase value, falling back to
                # the module's current (training-level) kappa.
                self._kappa_base = float(self.struct_cfg.get(
                    "kappa", getattr(pl_module, "kappa", 0.0)))
            cfg["kappa"] = self._kappa_base if self._kappa_active else 0.0
            if not self._kappa_active:
                logger.info(
                    "[adaptive] NOTEARS gate CLOSED (HSIC stalled for %d "
                    "structure phase(s)): kappa applied as 0.0 (base=%.3g).",
                    self._hsic_stall_cycles, self._kappa_base,
                )
        return cfg

    def _apply_descendant_mask_cfg(
        self, pl_module: pl.LightningModule, phase_cfg: Dict[str, Any]
    ) -> None:
        """Apply per-phase descendant-HSIC overrides (no-op when unspecified).

        ``hsic_descendant_hops`` is intentionally allowed to be ``None`` (=full
        transitive closure), so it is cast separately from the numeric keys.

        The ``hsic_descendant_warmup_epochs`` value set here is interpreted
        relative to the anchor installed by
        :meth:`_apply_descendant_warmup_anchor` — see that method for why a raw
        global-epoch threshold is meaningless under the adaptive schedule.
        """
        if not any(k in phase_cfg for k in self._DESCENDANT_MASK_KEYS):
            return
        if not hasattr(pl_module, "hsic_exclude_descendants"):
            logger.warning(
                "[adaptive] descendant-HSIC keys present in the phase config but "
                "the forecaster (%s) does not support them — ignoring.",
                type(pl_module).__name__,
            )
            return

        applied: Dict[str, Any] = {}
        for key in self._DESCENDANT_MASK_KEYS:
            if key not in phase_cfg:
                continue
            raw = phase_cfg[key]
            if key == "hsic_descendant_hops":
                val: Any = None if raw is None else int(raw)
            elif key in ("hsic_exclude_descendants", "hsic_descendant_exclude_self"):
                val = bool(raw)
            else:
                val = float(raw)
            setattr(pl_module, key, val)
            applied[key] = val

        # Changing the mask invalidates any smoothed adjacency accumulated under
        # the previous phase's settings, so drop the EMA state at the switch.
        if hasattr(pl_module, "_descendant_ema_score"):
            setattr(pl_module, "_descendant_ema_score", None)

        logger.info("[adaptive] descendant-HSIC overrides applied: %s", applied)

    def _apply_descendant_warmup_anchor(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Charge the descendant-mask warmup to the FIRST structure phase only.

        The forecaster counts ``hsic_descendant_warmup_epochs`` from
        ``_descendant_warmup_anchor`` (default 0 = start of the run).  Under the
        adaptive schedule everything runs in ONE ``fit()``, so ``current_epoch``
        is global and a raw threshold is the wrong quantity twice over:

        * it is typically ALREADY EXPIRED when structure first starts — with a
          reconstruct warmup of 100–200 epochs, a 50-epoch guard has long
          elapsed, so the mask goes live on the very first structural step, on a
          still-random adjacency.  That is exactly the self-confirmation risk the
          warmup exists to prevent;
        * and being a one-shot global comparison, it can never re-arm, so it
          cannot express "delay within each phase" either.

        Anchoring it to the first epoch of the FIRST structure phase makes the
        warmup count the epochs that actually TRAIN the structure.  Later
        structure phases set the anchor to ``None`` ("already served"), so the
        delay is paid once instead of being re-paid every cycle — the adjacency
        is no longer random by then, which is what the guard protects against.
        """
        if not hasattr(pl_module, "_descendant_warmup_anchor"):
            return

        if self._struct_phase_count <= 1:
            anchor: Optional[int] = int(trainer.current_epoch)
            warmup = int(getattr(pl_module, "hsic_descendant_warmup_epochs", 0) or 0)
            if warmup > 0:
                logger.info(
                    "[adaptive] descendant-HSIC warmup anchored to the FIRST "
                    "structure phase: %d epoch(s) from global_epoch=%d "
                    "(masking becomes active at global_epoch=%d).",
                    warmup, anchor, anchor + warmup,
                )
        else:
            # Warmup already served in the first structure phase.
            anchor = None

        setattr(pl_module, "_descendant_warmup_anchor", anchor)

    @staticmethod
    def _apply_fanin_phase(pl_module: pl.LightningModule, phase: str) -> None:
        """Tell the fan-in squeeze whether this phase counts as structural time.

        ``query_norm_log_scale`` is a STRUCTURAL parameter, so the squeeze must
        be clocked in structure epochs: a reconstruct phase freezes it, and
        letting the global epoch drive the schedule would spend the whole
        anneal window while nothing structural moves (same failure as the
        descendant-HSIC warmup above).  No-op when the forecaster predates the
        feature or the prior is off.
        """
        schedule = getattr(pl_module, "fanin_schedule", None)
        if schedule is None:
            return
        schedule.in_structure_phase = (phase == "structure")

    # ------------------------------------------------------------------
    # Prior-softmax gain schedule (GainSoftmax modules)
    # ------------------------------------------------------------------
    @staticmethod
    def _set_gain_lambda(pl_module: pl.LightningModule, value: float) -> int:
        """Set the gain interpolation weight on the model's GainSoftmax modules.

        The forecaster wraps the architecture as ``pl_module.model``; the
        architecture exposes ``set_gain_lambda`` (AttentionSelectorLayer).
        No-op (returns 0) for models without the prior-softmax gain.
        """
        setter = getattr(getattr(pl_module, "model", None), "set_gain_lambda", None)
        if setter is None:
            return 0
        return int(setter(value))

    def _update_gain_lambda(self, trainer: pl.Trainer,
                            pl_module: pl.LightningModule) -> None:
        """Ramp the gain lambda over the alternating schedule (global epoch).

        lambda = clamp((epoch - gain_lambda_start) / gain_lambda_ramp, 0, 1)
                 * gain_lambda_final.
        Before ``gain_lambda_start`` lambda is 0 (the gate-only baseline); a
        ramp of 0 jumps to the target at the start epoch.  No-op when the
        model owns no GainSoftmax module (``_set_gain_lambda`` returns 0) or no
        start is configured.
        """
        if self.gain_lambda_start is None:
            return
        epoch = int(trainer.current_epoch)
        if epoch < self.gain_lambda_start:
            lam = 0.0
        elif self.gain_lambda_ramp <= 0:
            lam = self.gain_lambda_final
        else:
            frac = (epoch - self.gain_lambda_start) / float(self.gain_lambda_ramp)
            lam = min(1.0, frac) * self.gain_lambda_final
        n = self._set_gain_lambda(pl_module, lam)
        if n > 0:
            pl_module.log(
                "gain_lambda", float(lam), on_step=False, on_epoch=True
            )

    def _apply_phase(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                     phase: str) -> None:
        struct_params, recon_params = self._resolve_param_groups(pl_module)
        self._apply_fanin_phase(pl_module, phase)

        # Nodewise query update: the gradient landscape changes at every
        # phase switch (theta_R moved all through the recon phase), so the
        # SNR evidence never crosses a boundary unless configured otherwise.
        if getattr(pl_module, "nodewise_reset_every_stage", False):
            pl_module.nodewise_reset_stats()


        if phase in ("reconstruct", "final_reconstruct"):
            for p in struct_params:
                p.requires_grad_(False)
            for p in recon_params:
                p.requires_grad_(True)
            # Descendant-HSIC overrides are honoured in every phase so a run can
            # e.g. keep the mask off during reconstruct and on during structure.
            # The final phase inherits the reconstruct block's settings and lets
            # its own block override individual keys.
            mask_cfg = (
                self.recon_cfg if phase == "reconstruct"
                else {**self.recon_cfg, **self.final_cfg}
            )
            self._apply_descendant_mask_cfg(pl_module, mask_cfg)
        elif phase == "structure":
            for p in recon_params:
                p.requires_grad_(False)
            for p in struct_params:
                p.requires_grad_(True)
            # Apply structure-phase loss weights (e.g. lambda_hsic_cross), with
            # lambda_l0 disarmed when the HSIC-progress gate is closed.
            self._apply_lambdas(pl_module, self._gated_struct_cfg(pl_module))
            # ...and the structure-phase descendant-exclusion settings.
            self._struct_phase_count += 1
            self._apply_descendant_mask_cfg(pl_module, self.struct_cfg)
            # Anchor AFTER the overrides so the warmup value logged/compared is
            # the one this phase will actually use.
            self._apply_descendant_warmup_anchor(trainer, pl_module)
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
        self._hsic_phase_best_gate = float("inf")


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


    def _resolve_split_key(self, phase: str) -> str:
        """Map a phase to its cross-fit split key, swapping per completed cycle.

        With ``swap_splits`` enabled the reconstruct/structure subsets are
        exchanged at every completed recon+structure cycle.  The parity of
        ``_cycle_count`` (completed structure phases) equals the number of swaps
        so far: it is incremented at the end of each structure phase, just
        before the following reconstruct phase is applied, so an odd count makes
        both alternating phases request the *other* subset.  Within every
        recon->struct pairing ``_cycle_count`` is constant, so the two phases
        always train on disjoint subsets — structure never reuses the split of
        the reconstruction that preceded it.  ``final_reconstruct`` (full
        training set) is never swapped.
        """
        if (
            self.swap_splits
            and phase in ("reconstruct", "structure")
            and self._cycle_count % 2 == 1
        ):
            return "structure" if phase == "reconstruct" else "reconstruct"
        return phase

    def _swap_train_subset(self, phase: str) -> Optional[int]:
        """
        Point the data module at ``phase``'s cross-fit training subset.

        The requested split key is resolved through :meth:`_resolve_split_key`,
        so with ``swap_splits`` enabled a phase may be pointed at the *other*
        phase's subset (the datamodule's static mapping itself is never
        mutated).  Returns the subset size (for logging), or ``None`` when
        cross-fitting is disabled or the phase has no dedicated subset.
        Validation/test indices are kept constant so stage-to-stage metrics
        remain comparable.
        """
        if not self.cross_fitting or self.dm is None:
            return None
        key = self._resolve_split_key(phase)
        self._active_split_key = key
        # Preferred path: the datamodule owns the phase→subset mapping.
        if hasattr(self.dm, "set_active_phase"):
            return self.dm.set_active_phase(key)
        # Fallback for datamodules without the stage-split API.
        if self.stage_splits is None:
            return None
        subset = self.stage_splits.get(key)
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
            # Cross-fit subset the ENDING phase actually trained on (equals the
            # phase name unless swap_splits exchanged the subsets this cycle;
            # None when cross-fitting is disabled).
            "train_split": self._active_split_key,
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
            "train_split": self._active_split_key,
            **{f"dag_{k}": v for k, v in diag.items()
               if k not in ("phase", "phase_index", "epoch", "label")},
        })

        if not self.cluster:
            print(f"  [adaptive] transition ({reason}): {from_phase} -> {to_phase} "
                  f"| {self.monitor}={monitor_val:.5f}")


    def _final_boundary_reached(self, trainer: pl.Trainer) -> bool:
        """True once the alternating schedule's epoch budget is exhausted.

        The final phase is APPENDED on top of ``total_epoch_budget``
        (``Trainer.max_epochs = total_epoch_budget + final_max_epochs``), so
        the alternating schedule owns epochs ``0 .. max_epochs -
        final_max_epochs - 1`` (0-based).  Switching at the validation
        boundary of the last owned epoch hands every remaining epoch to the
        final phase.  With sparse validation (``check_val_every_n_epoch > 1``)
        the switch fires at the first validation epoch past the boundary.
        """
        max_epochs = getattr(trainer, "max_epochs", None)
        if max_epochs is None:
            return False
        alternating_budget = int(max_epochs) - self.final_max_epochs
        return trainer.current_epoch >= alternating_budget - 1

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

        # ---------- Prior-softmax gain ramp (phase-agnostic) ----------
        # lambda ramps 0 -> gain_lambda_final over the alternating schedule,
        # driven by the global epoch: the gate's role morphs from the
        # multiplicative weight to the softmax support while the trainer keeps
        # alternating reconstruct <-> structure.  Runs in EVERY phase (and in
        # the optional final phase), before the phase dispatch.
        self._update_gain_lambda(trainer, pl_module)

        # ---------- Final reconstruction-only phase: entry trigger ----------
        # The alternating schedule owns the first ``total_epoch_budget`` epochs;
        # the optional final phase is appended on top of them.  Switch to it at
        # the first validation boundary on/after the alternating budget is
        # exhausted, from whichever phase is currently active.
        if (
            self.final_enabled
            and self.current_phase != "final_reconstruct"
            and self._final_boundary_reached(trainer)
        ):
            self._record_transition(
                trainer, pl_module, "alternating_budget",
                from_phase=self.current_phase, to_phase="final_reconstruct",
                monitor_val=current,
            )
            self._phase_index += 1
            self._apply_phase(trainer, pl_module, "final_reconstruct")
            return

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

            # HSIC-progress gates: track the phase-best HSIC whenever the
            # metric is available (the gates' progress signal), INDEPENDENT of
            # the plateau early-exit counters (which are disabled at
            # ``hsic_patience == 0``), and expose the gate states as CSV
            # metrics so post-mortems can see when each regularizer was armed.
            if (self.l0_gate_on_hsic or self.kappa_gate_on_hsic) and hsic_val is not None:
                hsic_current = float(hsic_val)
                if np.isfinite(hsic_current):
                    self._hsic_phase_best_gate = min(
                        self._hsic_phase_best_gate, hsic_current
                    )
            pl_module.log(
                "adaptive_l0_active", float(self._l0_active),
                on_step=False, on_epoch=True,
            )
            if self.kappa_gate_on_hsic:
                pl_module.log(
                    "adaptive_kappa_active", float(self._kappa_active),
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

                # HSIC-progress gates: score this structure phase against the
                # RUN-best HSIC.  An improvement re-arms BOTH gates; otherwise
                # the shared stall counter increments and each gate closes at
                # its own patience (the regularizer is disarmed from the NEXT
                # structure phase on, in ``_gated_struct_cfg``).  The
                # improvement threshold is the smallest min_delta of the
                # enabled gates (the progress signal is shared).
                if self.l0_gate_on_hsic or self.kappa_gate_on_hsic:
                    gate_min_delta = min(
                        d for on, d in (
                            (self.l0_gate_on_hsic, self.l0_gate_min_delta),
                            (self.kappa_gate_on_hsic, self.kappa_gate_min_delta),
                        ) if on
                    )
                    phase_best = self._hsic_phase_best_gate
                    if np.isfinite(phase_best):
                        if phase_best <= self._hsic_run_best * (1.0 - gate_min_delta):
                            self._hsic_run_best = phase_best
                            self._hsic_stall_cycles = 0
                            if not self._l0_active or not self._kappa_active:
                                logger.info(
                                    "[adaptive] HSIC-progress gates RE-ARMED: "
                                    "%s improved to %.6g (new run best).",
                                    self.struct_hsic_monitor, phase_best,
                                )
                            self._l0_active = True
                            self._kappa_active = True
                        else:
                            self._hsic_stall_cycles += 1
                            if self._hsic_stall_cycles >= self.l0_gate_patience:
                                self._l0_active = False
                            if self._hsic_stall_cycles >= self.kappa_gate_patience:
                                self._kappa_active = False


                # Optional safety guard: stop only when an explicit max_cycles is
                # configured and has been reached.  When ``max_cycles is None``
                # (default) the run is bounded solely by ``total_epoch_budget``
                # (Trainer max_epochs) — i.e. it does as many cycles as fit.
                if self.max_cycles is not None and self._cycle_count >= self.max_cycles:

                    # With the final phase enabled, do not stop yet: refine the
                    # reconstruction against the frozen structure on the full
                    # training set first (the final phase's own epoch cap still
                    # guarantees termination).
                    if self.final_enabled:
                        self._record_transition(
                            trainer, pl_module, f"{reason}_final",
                            from_phase="structure",
                            to_phase="final_reconstruct",
                            monitor_val=current,
                        )
                        self._phase_index += 1
                        if not self.cluster:
                            print(f"  [adaptive] max_cycles={self.max_cycles} "
                                  f"reached - entering final reconstruct phase.")
                        self._apply_phase(trainer, pl_module, "final_reconstruct")
                        return

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

        # --------- Final reconstruction-only phase: plateau / budget ---------
        elif self.current_phase == "final_reconstruct":
            # Same rate-of-improvement plateau logic as the reconstruct phase,
            # with the final block's own patience / min_delta (which fall back
            # to the reconstruct values) and its own min-epoch floor.  The
            # ``max_epochs`` cap always takes precedence over the floor.
            if current <= self._phase_best * (1.0 - self.final_plateau_min_delta):
                self._phase_best = current
                self._plateau_counter = 0
            else:
                if current < self._phase_best:
                    self._phase_best = current
                self._plateau_counter += 1

            min_epochs_reached = phase_epochs >= self.final_min_epochs
            plateaued = (
                self._plateau_counter >= self.final_plateau_patience
                and min_epochs_reached
            )
            budget_hit = phase_epochs >= self.final_max_epochs

            if plateaued or budget_hit:
                reason = (
                    "final_recon_plateau" if plateaued else "final_recon_budget"
                )
                self._record_transition(
                    trainer, pl_module, reason,
                    from_phase="final_reconstruct", to_phase="stop",
                    monitor_val=current,
                )
                self._phase_index += 1
                if not self.cluster:
                    print(f"  [adaptive] final reconstruct phase ended ({reason}) "
                          f"- stopping.")
                trainer.should_stop = True
                return


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
    self-describing for offline evaluation.

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
    # CuBLAS deterministic workspace: the main run's train_single_fold uses
    # pl.Trainer(deterministic=True) (torch.use_deterministic_algorithms), and
    # on CUDA >= 10.2 every CuBLAS GEMM then requires a fixed workspace via
    # this env var.  Set it at the very top: the optional dropout-selection
    # pre-flight below already initializes CuBLAS in this process, so the var
    # must be present before ANY GEMM, not just before train_single_fold.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from causaliT.training.trainer import (
        get_dataloader,
        _make_fold_splits,
        create_model_instance,
        train_single_fold,
        _run_post_training_evaluations,
        resolve_seeds,
    )
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset
    from causaliT.training.experiment_control import update_config

    ad_cfg = _to_plain_container(config.get("adaptive_training", {})) or {}
    if not ad_cfg:
        raise ValueError(
            "config['adaptive_training'] is empty or missing. Define the "
            "adaptive schedule block (see module docstring)."
        )

    # Resolve multiplier-derived fields (experiment.d_ff / d_qk) so configs with
    # ``d_ff: null`` / ``d_qk: null`` also work when adaptive_trainer is called
    # directly (the sweeper and the CLI's find_yml_files already do this; the
    # call is idempotent - it only fills nulls).
    config = update_config(config)

    if not config["training"].get("use_gradient_routing", False):
        raise ValueError(
            "adaptive_trainer requires training.use_gradient_routing=True so "
            "that structural/reconstruction parameter groups can be frozen "
            "independently. Enable it in the config."
        )

    # Model seed (weight init) and data seed (splits) are resolved separately;
    # data_seed defaults to seed, so legacy configs are unaffected.  The DAG
    # sweep pins data_seed to the DAG seed so the split stays FIXED while the
    # model seed varies -> per-edge stability across initializations.
    seed, data_seed = resolve_seeds(config)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    config = populate_seq_lengths_from_dataset(config, data_dir)

    # --- Shared data module and fold splits (single fold) ---
    dm = get_dataloader(config, data_dir, cluster, data_seed)
    dm.prepare_data()

    # Force single-fold behaviour for the adaptive run.
    config = copy.deepcopy(config)
    config["training"]["k_fold"] = 1
    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config, dm, data_seed, data_dir=data_dir
    )
    train_local_idx, val_local_idx = fold_splits[0]

    # --- Global epoch budget -> pl.Trainer max_epochs ---
    total_budget = int(ad_cfg.get("total_epoch_budget",
                                  config["training"].get("max_epochs", 800)))

    # Optional final reconstruction-only phase: its epochs are APPENDED on top
    # of the alternating budget, so the alternating schedule keeps all
    # ``total_budget`` epochs and the refinement runs afterwards (structure
    # frozen, cross-fit split off -> full training set).
    final_cfg = _to_plain_container(ad_cfg.get("final_reconstruct", {})) or {}
    final_enabled = bool(final_cfg.get("enabled", False))
    final_max_epochs = int(final_cfg.get("max_epochs", 100))
    if final_enabled and final_max_epochs <= 0:
        logger.warning(
            "adaptive_trainer: final_reconstruct.enabled=true but max_epochs=%d "
            "<= 0 - disabling the final phase.", final_max_epochs,
        )
        final_enabled = False
    if final_enabled:
        config["training"]["max_epochs"] = total_budget + final_max_epochs
    else:
        config["training"]["max_epochs"] = total_budget
    if config["training"].get("save_ckpt_every_n_epochs") is None:
        config["training"]["save_ckpt_every_n_epochs"] = (
            config["training"]["max_epochs"]
        )

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
    # Exchange the recon/structure subsets after each completed cycle (the
    # controller validates it against cross-fitting being active).
    swap_splits = bool(ad_cfg.get("swap_splits", False))

    if data_split_ratio is not None and 0.0 < float(data_split_ratio) < 1.0:
        active_split_ratio = float(data_split_ratio)
        recon_idx, struct_idx = _partition_train_indices(
            train_local_idx, active_split_ratio, data_seed
        )

        stage_splits = {"reconstruct": recon_idx, "structure": struct_idx}
        # Final reconstruction-only phase: data split OFF.  Register the FULL
        # fold training indices under the final phase's key so the controller
        # swaps back to the complete training set when the phase starts
        # (train_local_idx still holds the full fold indices here; it is
        # narrowed to the starting phase's subset only below).
        if final_enabled:
            stage_splits["final_reconstruct"] = np.asarray(train_local_idx)
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
                f"{', swapped after each cycle' if swap_splits else ''}"
            )

    # --- Pre-flight dropout selection (optional) -----------------------------
    # Selects the per-node MLP dropout by maximizing the query-perturbation
    # sensitivity of the train HSIC after a short reconstruction-only warmup
    # per candidate (see causaliT/training/dropout_selection.py).  The winning
    # dropout is written into the resolved config (so the main model is built
    # with it) and the main run warm-starts from the winner's warmup weights.
    # Pre-flight epochs are selection overhead and do NOT count against
    # total_epoch_budget.
    ds_cfg = _to_plain_container(ad_cfg.get("dropout_selection", {})) or {}
    if bool(ds_cfg.get("enabled", False)):
        from causaliT.training.dropout_selection import (
            run_dropout_selection,
            _set_mlp_dropout,
        )
        best_dropout, winner_ckpt = run_dropout_selection(
            config=config, data_dir=data_dir, dm=dm, save_dir=save_dir,
            cluster=cluster, seed=seed,
        )
        if best_dropout is not None:
            _set_mlp_dropout(config, best_dropout)
            if starting_ckpt is not None:
                logger.warning(
                    "adaptive_trainer: dropout_selection overrides the "
                    "configured starting_checkpoint (%s) with the winner's "
                    "warmup weights (%s).", starting_ckpt, winner_ckpt,
                )
            starting_ckpt = winner_ckpt

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
        if controller.final_enabled:
            print(f"  final reconstruct  : +{controller.final_max_epochs} epochs "
                  f"appended (structure frozen, full training set; "
                  f"plateau patience {controller.final_plateau_patience})")


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
        "swap_splits": controller.swap_splits,
        "n_train_reconstruct": (int(len(stage_splits["reconstruct"]))
                                if stage_splits is not None else None),
        "n_train_structure": (int(len(stage_splits["structure"]))
                              if stage_splits is not None else None),
        "n_transitions": len(controller.transitions),

        "n_cycles": controller._cycle_count,
        "final_reconstruct": {
            "enabled": controller.final_enabled,
            "max_epochs": (controller.final_max_epochs
                           if controller.final_enabled else None),
            "ran": any(r.get("phase") == "final_reconstruct"
                       for r in controller.phase_rows),
        },
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
