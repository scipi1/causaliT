"""
AttentionSelectorForecaster: PyTorch Lightning wrapper for AttentionSelectorLayer.

Research objective
==================
Test whether attention over value-blanked queries and actual-value keys/values
can recover causal parent sets from observational data when trained with MSE
reconstruction + HSIC independence regularization + score sparsity.

Two node topologies (``model.kwargs.homogeneous_nodes``)
=======================================================
SPLIT mode (``homogeneous_nodes=False``, the default) keeps the **S/X prior**:
S nodes are exogenous parents (keys/values only) and X nodes are the only
children (queries).  Two attention blocks (S->X cross + X->X self) are
re-concatenated by the layer into ONE posterior::

    attention  (B, L_X, L_S + L_X)     pred / target  (B, L_X, .) / (B, L_X)
      - columns 0 .. L_S-1   -> learned S->X edges
      - columns L_S .. end   -> learned X->X edges (diagonal = 0 by mask)

HOMOGENEOUS mode (``homogeneous_nodes=True``) DROPS that prior: ``[S ; X]`` is
ONE set of ``N = L_S + L_X`` nodes and every node is simultaneously a
value-blanked **query** (candidate child) and an actual-value **key/value**
(candidate parent).  There is exactly ONE square block, built from
``self_attention_type`` (the cross ``attention_type`` is IGNORED), hence::

    attention  (B, N, N)               pred / target  (B, N, .) / (B, N)

Everything below applies to both layouts; the mode-specific differences are:

* ``forward`` builds ``s_blanked`` (S with its value column zeroed) and hands it
  to ``model.forward_with_actual`` -- mandatory in homogeneous mode;
* ``_step`` targets ``cat([S_values, X_values], dim=1)`` -> ``(B, N)`` instead of
  the X values alone, so the MSE, the torchmetrics, the residuals and the ANM
  diagnostics all follow the N-row layout;
* the oracle mask is assembled as the square ``(N, N)`` GT adjacency (the S rows
  are all-zero: by dataset convention nothing points into a source);
* NOTEARS runs on the FULL square score tensor (see 3. below);
* the HSIC candidate-parent set is the target itself (already all N nodes).

Design differences from SingleCausalForecaster
===============================================
1. **One combined posterior** -- see the two topologies above.  Downstream code
   never sees two separate tensors: ``split_attention()`` (shape-aware) recovers
   the canonical ``(L_X, L_S)`` / ``(L_X, L_X)`` DAG blocks in BOTH modes, and
   ``split_attention_blocks()`` additionally exposes the X->S / S->S blocks that
   exist only when S nodes are children too.

2. **Unified HSIC over combined [S, X] source**.
   `source = cat([S_values, X_values], dim=1)` is passed to
   `hsic_cross_per_pair`, which computes HSIC(source_j, res_i) for all
   (i, j) pairs in one call.  No lambda weighting between S and X parts:
   the combined loss naturally penalizes dependence from any source.

3. **NOTEARS acyclicity** (``training.kappa``).  In SPLIT mode it is applied to
   the **X->X sub-block** of the score tensor (columns ``S_seq_len:``), a square
   ``(L_X, L_X)`` directed edge matrix; the S->X block is bipartite and
   inherently acyclic, so no term is added there.  In HOMOGENEOUS mode the FULL
   ``(N, N)`` score tensor already IS the square directed edge matrix over all
   nodes -- and S->S / X->S cycles are now expressible -- so NOTEARS is applied
   to it in full.  With ``use_gradient_routing=True`` the NOTEARS penalty rides
   on the structural pathway (same as HSIC), updating Q/K projections and
   structural embeddings.

4. **Gradient routing** works unchanged: query_projection and key_projection
   are structural params; value_projection, out_projection, FFN, forecaster
   are reconstruction params.  The classify_parameters() function identifies
   them by name without any modification (it keys on the ``query_embed``
   PREFIX, so the homogeneous S-side query table routes structural too).

Logged metrics
==============
- train/val_loss_x         : MSE reconstruction loss
- train/val_x_mae/rmse/r2  : Reconstruction metrics.  NOTE ``x_r2`` is POOLED
                             over every node (and, in homogeneous mode, over the
                             exogenous S rows whose correct R2 is ~0), so it is
                             NOT a fit quality.  Read ``x_r2_macro`` instead.
- train/val_x_r2_endo      : pooled R2 on the ENDOGENOUS rows only (== x_r2 in
                             split mode)
- train/val_x_r2_macro     : mean PER-NODE R2 over the endogenous rows --
                             pooling-free, this is the fit-quality number
- train/val_x_r2_src       : pooled R2 on the S rows (homogeneous mode only).
                             A source is only predictable from its descendants,
                             so a HIGH value flags ANTI-CAUSAL use of the
                             posterior.

- train/val_score_sparse   : L1 sparsity on attention weights
- train/val_hsic           : HSIC regularization value
- train/val_hsic_reg       : Weighted HSIC regularization term
- train/val_struct_recon_reg: Reconstruction injected into the structural loss
                             (lambda_struct_recon * loss_x); 0 unless > 0.
- train/val_group_l1       : Group-L1 embedding regularization

- train/val_notears        : NOTEARS acyclicity penalty (X->X sub-block in split
                             mode, full (N, N) matrix in homogeneous mode)
- train_interf_cos_<block> : (diagnostic) per-structural-block cosine
                             similarity between the L0 and HSIC gradients.
                             Only logged when the attention exposes a
                             differentiable L0 gate -- i.e.
                             HardConcreteCrossAttention or GatedCrossAttention
                             -- and lambda_l0>0, lambda_hsic>0, and
                             ``training.log_l0_hsic_interference=True``.
                             In homogeneous mode the gate is read off
                             ``self_attention_type`` (the type that actually
                             builds the single block).


Attention splitting for evaluation
====================================
After training, use ``split_attention(A)`` (on the forecaster or on the layer)
to get, in BOTH modes:
    att_sx  (B, L_X, L_S)  -- S->X attention (compare to S->X ground truth)
    att_xx  (B, L_X, L_X)  -- X->X attention (compare to X->X ground truth)
Then threshold and compute SHD.

For homogeneous-mode diagnostics the forecaster also forwards:
    ``split_attention_blocks(A)`` -- all four blocks, including ``x_to_s`` and
        ``s_to_s`` (``None`` in split mode, where S is never a child);
    ``source_scores(A)``          -- per-node incoming-edge mass; LOW means the
        node is likely a SOURCE, i.e. it RECOVERS the S/X partition that
        homogeneous mode no longer assumes.
"""

import inspect
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from os.path import join


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.utils import load_dag_masks, corrupt_dag_masks
from causaliT.utils.hsic_utils import (
    hsic_cross_per_pair,
    hsic_row_means,
    hsic_attention_weighted,
    hsic_attention_softmax,
    row_entropy_stats,
    hsic_null_calibration,
    bayes_multiplier,
    _median_bandwidth,
)
from causaliT.utils.descendant_mask import (
    build_hsic_pair_mask,
    build_hsic_pair_mask_budgeted,
)
from causaliT.utils.query_norm import (
    FaninPriorSchedule,
    collect_query_norm_penalty,
    query_norm_stats,
)


from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.nodewise_update import NodewiseQuerySelector
from causaliT.training.centroid_commit import CentroidCommitController
from causaliT.training.interference_utils import (
    build_interference_blocks,
    compute_l0_hsic_interference,
)
from causaliT.training.gradient_surgery import pcgrad_reconcile

logger = logging.getLogger(__name__)


class AttentionSelectorForecaster(pl.LightningModule):

    """
    Lightning wrapper for AttentionSelectorLayer.

    Args:
        config:   Configuration dictionary (data, model, training sections).
        data_dir: Path to the dataset directory.  Required when
                  ``training.use_hard_masks=True`` so that GT DAG mask CSV
                  files can be loaded and (optionally) corrupted for the
                  wrong-DAG oracle experiment.
    """

    def __init__(self, config: dict, data_dir: str = None):
        super().__init__()

        self.config = config

        # Build model
        self.model = AttentionSelectorLayer(**config["model"]["kwargs"])

        # ----------------------------------------------------------------
        # Query centroid initialisation (see AttentionSelectorLayer
        # .init_query_at_key_centroid).  Value-modulated key embeddings need
        # real data, so the write is deferred to the FIRST training batch.
        # ``_query_centroid_init_done`` is a plain (non-persistent) flag so it
        # does NOT enter the state_dict; on resume it is re-armed to True in
        # on_load_checkpoint whenever a trained query embedding is present, so
        # we never clobber a learned query on warm-start.
        # ----------------------------------------------------------------
        self._query_centroid_init = bool(
            config["model"]["kwargs"].get("query_centroid_init", False)
        )
        self._query_centroid_init_done = False
        # Known-edges query prior (overwrites the init for the listed nodes,
        # optionally frozen).  Same lazy first-batch mechanism as the centroid
        # init; see AttentionSelectorLayer.init_queries_from_parents.
        self._query_parents_prior = config["model"]["kwargs"].get(
            "query_parents_prior", None
        )
        # Source-nodes query prior (zero + freeze the query of the listed
        # nodes, so they can never be children).  Same lazy first-batch
        # mechanism as the parents prior; see
        # AttentionSelectorLayer.init_source_queries_zero.
        self._query_source_prior = config["model"]["kwargs"].get(
            "query_source_prior", None
        )


        # Data indices
        self.val_idx = config["data"]["val_idx"]

        self.S_seq_len = config["data"]["S_seq_len"]
        self.X_seq_len = config["data"]["X_seq_len"]

        # ------------------------------------------------------------------
        # Node-topology mode (mirrors AttentionSelectorLayer).
        #   False (default) Ã¢â€ â€™ SPLIT: only the L_X variables are children; the
        #       posterior is (B, L_X, L_S+L_X) and the target is the X values.
        #   True            Ã¢â€ â€™ HOMOGENEOUS: the S/X prior is dropped, all
        #       N = L_S + L_X nodes are simultaneously blanked queries and
        #       actual-value keys.  The posterior is the square (B, N, N)
        #       directed adjacency and the target is cat([S_values, X_values]).
        # ------------------------------------------------------------------
        self.homogeneous_nodes = bool(
            config["model"]["kwargs"].get("homogeneous_nodes", False)
        )
        self.N = self.S_seq_len + self.X_seq_len


        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(
                f"Unsupported loss_fn: {config['training']['loss_fn']}.  "
                f"AttentionSelectorForecaster only supports 'mse'."
            )

        # ----------------------------------------------------------------
        # Reconstruction loss weight
        # ----------------------------------------------------------------
        self.lambda_recon = float(config["training"].get("lambda_recon", 1.0))

        # ----------------------------------------------------------------
        # Structural reconstruction mixing (convex mix on the structural
        # pathway).  Mirrors SingleCausalForecaster.lambda_struct_recon.
        #
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg + acyclic_reg + l0_reg
        #
        # alpha = lambda_struct_recon in [0, 1]:
        #   * 0.0 Ã¢â€ â€™ pure HSIC structural stream (original behaviour).
        #   * >0  Ã¢â€ â€™ re-inject a controlled dose of reconstruction signal into
        #           the STRUCTURAL parameters (Q/K, structural embeddings) that
        #           gradient routing otherwise severs.  Motivated by the
        #           observation that causal parents must also be predictive,
        #           aligning the method with fit/likelihood-driven differentiable
        #           causal discovery (NOTEARS/DAG-GNN/GraN-DAG/DCDI).
        #
        # Only meaningful with use_gradient_routing=True: without routing the
        # reconstruction loss already updates every parameter via total_loss,
        # so the mix (which lives only in _last_loss_components["loss_structural"])
        # has no effect on the automatic-optimisation path.
        # ----------------------------------------------------------------
        self.lambda_struct_recon = float(
            config["training"].get("lambda_struct_recon", 0.0)
        )
        if not (0.0 <= self.lambda_struct_recon <= 1.0):
            raise ValueError(
                f"lambda_struct_recon must be in [0, 1], got {self.lambda_struct_recon}"
            )


        # ----------------------------------------------------------------
        # Score sparsity (L1 on attention weights)
        # ----------------------------------------------------------------
        self.lambda_score_sparse = config["training"].get("lambda_score_sparse", 0.0)

        # ----------------------------------------------------------------
        # HSIC regularization (unified: HSIC over combined [S, X] source)
        # ----------------------------------------------------------------
        self.lambda_hsic = config["training"].get("lambda_hsic", 0.0)
        # Per-row HSIC diagnostics (bilevel commit groundwork, Phase 0 of
        # docs/ideas/BILEVEL_CENTROID_COMMIT.md): log {stage}_hsic_row_{i},
        # the node-responsible HSIC term mean_j HSIC(source_j, res_i).  Only
        # supported for the plain (non attention-weighted) HSIC branch.
        self.log_hsic_rows = bool(config["training"].get("log_hsic_rows", False))
        self._last_hsic_row_means = None
        # Raw PRE-WEIGHTING pair HSIC matrix (NaN = excluded pair), detached,
        # stashed every step when log_hsic_rows materialises it.  Read by the
        # HSICClassMetrics callback (hsic_class/* diagnostics).
        self._last_hsic_pair_mat = None
        # ---- HSIC aggregation selector ---------------------------------
        #   plain            -- unweighted mean over pairs (hsic_cross_per_pair)
        #   attw             -- every pair weighted by the attention posterior
        #   attw_descendants -- HYBRID: non-descendant pairs UNWEIGHTED, only
        #                       descendant pairs attention-weighted.  Removes the
        #                       degenerate ``att -> 0`` solution by construction
        #                       (a true parent's HSIC is paid whatever its
        #                       weight), while leaving the escape open exactly
        #                       where collapse is the correct answer.
        #   attw_softmax     -- SOFTMAX COMPETITION: pair weights are the
        #                       row-wise softmax of the gate scores (self-edge
        #                       excluded pre-softmax).  Removes the ``att -> 0``
        #                       trivial solution by construction (each row sums
        #                       to 1) and REPLACES the descendant mask entirely:
        #                       sparsity/descendant rejection emerges from the
        #                       within-row weight competition.
        #
        # ``use_attention_weighted_hsic`` is the LEGACY alias, still honoured so
        # every existing config keeps working: True -> attw, False -> plain.
        _agg = config["training"].get("hsic_aggregation", None)
        _legacy = config["training"].get("use_attention_weighted_hsic", None)
        _valid_agg = ("plain", "attw", "attw_descendants", "attw_softmax")
        if _agg is None:
            self.hsic_aggregation = "attw" if bool(_legacy) else "plain"
        else:
            self.hsic_aggregation = str(_agg)
            if self.hsic_aggregation not in _valid_agg:
                raise ValueError(
                    f"hsic_aggregation must be one of {_valid_agg}, "
                    f"got {self.hsic_aggregation!r}."
                )
            if _legacy is not None:
                _implied = "attw" if bool(_legacy) else "plain"
                if _implied != self.hsic_aggregation and not (
                    bool(_legacy) and self.hsic_aggregation in (
                        "attw_descendants", "attw_softmax"
                    )
                ):
                    raise ValueError(
                        f"Conflicting HSIC aggregation settings: "
                        f"hsic_aggregation={self.hsic_aggregation!r} but the legacy "
                        f"use_attention_weighted_hsic={_legacy!r} implies "
                        f"{_implied!r}.  Set only one of them."
                    )
        # All weighted variants share the attention-weighted code path.
        self.use_attention_weighted_hsic = self.hsic_aggregation in (
            "attw", "attw_descendants", "attw_softmax"
        )
        self.hsic_weight_descendants_only = (
            self.hsic_aggregation == "attw_descendants"
        )
        self.hsic_softmax = self.hsic_aggregation == "attw_softmax"
        # Pair-weight construction for the softmax-competition aggregation:
        # "posterior" (default) renormalises the gate posterior row-wise
        # (softmax over log p: a CLOSED gate gets weight exactly 0 and
        # descendant pressure is routed onto the antisymmetric direction
        # gate); "softmax_logits" is the legacy mode where the [0, 1]
        # posterior is used directly as a logit (a zero gate still carries
        # weight 1/Z -- see hsic_utils.hsic_attention_softmax).
        self.hsic_pair_weight_mode = config["training"].get(
            "hsic_pair_weight_mode", "posterior"
        )
        if self.hsic_pair_weight_mode not in (
            "posterior", "softmax_logits", "evidence_max"
        ):
            raise ValueError(
                f"training.hsic_pair_weight_mode must be 'posterior', "
                f"'softmax_logits' or 'evidence_max', got "
                f"{self.hsic_pair_weight_mode!r}"
            )
        # Tilt temperature for evidence_max: "auto" = 1.4826*MAD of the pair
        # HSIC matrix (adapts sharpness to the HSIC noise level); or a fixed
        # positive float.
        self.hsic_tilt_tau = config["training"].get("hsic_tilt_tau", "auto")
        self._last_desc_weight_frac = 0.0

        # ---- HSIC cross-fitting -----------------------------------------
        # Evaluate the independence statistic on a DISJOINT, PERMANENT fold of
        # the training set: HSIC measured on the same rows the regressor just
        # fitted is optimistically biased (the fit absorbs sample-specific
        # noise, so the residual looks more independent than it is).
        #
        # Fold membership is by sample identity (datamodule.set_hsic_cross_fit),
        # never by position within a batch -- the train loader shuffles every
        # epoch, so a positional split would reassign samples each epoch and the
        # separation would dissolve entirely.
        #
        # Cost: a SECOND forward pass per step (fold B), i.e. ~2x compute.
        self.hsic_cross_fit = bool(
            config["training"].get("hsic_cross_fit", False)
        )
        self.hsic_cross_fit_ratio = float(
            config["training"].get("hsic_cross_fit_ratio", 0.5)
        )
        self._xfit_iter = None      # fold-B iterator, refreshed on exhaustion
        self._xfit_loader = None
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        self.hsic_mode = config["training"].get("hsic_mode", "biased")
        self.nhsic_epsilon = config["training"].get("nhsic_epsilon", 0.01)
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")
        self.hsic_bandwidth_multipliers = config["training"].get(
            "hsic_bandwidth_multipliers", None
        )
        # Per-phase bandwidth freeze (phase blocks set ``hsic_freeze_bandwidth:
        # true``, typically structure): on the first train batch of such a
        # phase, latch the per-variable median-heuristic bandwidths of the
        # source and the residuals and run the phase with them FIXED.  RBF +
        # per-batch median heuristic is exactly scale-equivariant, so the
        # adaptive default divides the residual-magnitude channel out of both
        # the HSIC metric (flat-looking curves) and its gradient (no pressure
        # toward smaller residuals).  Freezing restores both; see
        # experiments/6_INVESTIGATIONS/HSIC_OPT_2/diagnostics/frozen_bandwidth_test.py.
        self.hsic_freeze_bandwidth = bool(
            config["training"].get("hsic_freeze_bandwidth", False)
        )
        self._hsic_bw_frozen_sigmas = None   # (sig_src, sig_res) once latched
        self._hsic_bw_saved = None           # (sigma, adaptive) before freezing
        # BKD dropped-key exclusion: mask HSIC pairs whose source key was
        # dropped by batch-consistent key dropout this batch.  A dropped
        # parent cannot be regressed out of the residual, so the pair term
        # is irreducible within the batch - a p-dependent noise floor plus
        # a biased gradient into kept correlate edges.  Default False keeps
        # the legacy all-sources loss.  See _step / _build_bkd_keep_mask.
        self.hsic_bkd_exclude_dropped = bool(
            config["training"].get("hsic_bkd_exclude_dropped", False)
        )

        # LOO conditional-HSIC gate (docs/ideas/CONDITIONAL_HSIC_COUNTERPROPOSAL.md):
        # per-edge leave-one-out Bayes multiplier gamma, computed from masked
        # EVAL-mode forward passes (BKD is gated on self.training, so it is
        # inactive during measurement) and applied as a DETACHED multiplicative
        # gate on the per-pair HSIC weights.  Default False keeps the legacy
        # behaviour exactly.
        self.use_loo_gamma_gate = bool(
            config["training"].get("use_loo_gamma_gate", False)
        )
        self.loo_gamma_topk = config["training"].get("loo_gamma_topk", None)
        if self.loo_gamma_topk is not None:
            self.loo_gamma_topk = int(self.loo_gamma_topk)
        self.loo_gamma_ema = float(config["training"].get("loo_gamma_ema", 0.9))
        self.loo_gamma_refresh = int(config["training"].get("loo_gamma_refresh", 1))
        self.loo_gamma_permutations = int(
            config["training"].get("loo_gamma_permutations", 50)
        )

        # ----------------------------------------------------------------
        # Descendant-excluding HSIC (see causaliT.utils.descendant_mask)
        #
        # Under an ANM the residual r_i = e_i is independent of every
        # NON-descendant of i, but NECESSARILY dependent on its descendants and
        # on X_i itself.  Averaging those pairs into the HSIC term means the
        # TRUE DAG is not a minimiser of the structural loss: the only way to
        # shrink a descendant term is to stop having r_i = e_i, i.e. to regress
        # X_i on its own descendants Ã¢â‚¬â€ the documented "attends to descendants"
        # failure mode.  Excluding them restores consistency.
        #
        # The exclusion set is read off the LEARNED adjacency (the directed
        # posterior of the self-attention block), so it is only meaningful for
        # attention types that expose a direction-aware posterior.  Defaults
        # leave the loss byte-identical to the pre-feature behaviour.
        # ----------------------------------------------------------------
        self.hsic_exclude_descendants = bool(
            config["training"].get("hsic_exclude_descendants", False)
        )
        self.hsic_descendant_threshold = float(
            config["training"].get("hsic_descendant_threshold", 0.5)
        )
        _hops = config["training"].get("hsic_descendant_hops", None)
        self.hsic_descendant_hops = None if _hops is None else int(_hops)
        self.hsic_descendant_exclude_self = bool(
            config["training"].get("hsic_descendant_exclude_self", True)
        )
        self.hsic_descendant_weight = float(
            config["training"].get("hsic_descendant_weight", 0.0)
        )
        self.hsic_descendant_warmup_epochs = int(
            config["training"].get("hsic_descendant_warmup_epochs", 0)
        )
        self.hsic_descendant_min_kept_frac = float(
            config["training"].get("hsic_descendant_min_kept_frac", 0.25)
        )
        self.hsic_descendant_ema = float(
            config["training"].get("hsic_descendant_ema", 0.0)
        )
        if not (0.0 <= self.hsic_descendant_ema < 1.0):
            raise ValueError(
                f"hsic_descendant_ema must be in [0, 1), got {self.hsic_descendant_ema}"
            )
        # Budgeted variant (see descendant_mask.build_hsic_pair_mask_budgeted):
        # instead of hardening the posterior at ``hsic_descendant_threshold``
        # and hoping for a DAG, rank the pairs by a SOFT descendant score (the
        # detached posterior's fuzzy transitive closure) and exclude the top
        # ``hsic_descendant_budget_frac`` Ã¢â‚¬â€ per child row when
        # ``hsic_descendant_per_row`` Ã¢â‚¬â€ so the cap itself is the collapse guard
        # and the mask triggers every step.
        self.hsic_descendant_mode = str(
            config["training"].get("hsic_descendant_mode", "threshold")
        )
        if self.hsic_descendant_mode not in ("threshold", "budget"):
            raise ValueError(
                f"hsic_descendant_mode must be 'threshold' or 'budget', got "
                f"{self.hsic_descendant_mode!r}."
            )
        self.hsic_descendant_budget_frac = float(
            config["training"].get("hsic_descendant_budget_frac", 0.25)
        )
        self.hsic_descendant_per_row = bool(
            config["training"].get("hsic_descendant_per_row", True)
        )
        self.hsic_descendant_tnorm = str(
            config["training"].get("hsic_descendant_tnorm", "min")
        )
        # Self-attention block type: only a direction-aware posterior can tell
        # descendants from ancestors.  In homogeneous mode the single square
        # block IS the self-attention type; in split mode it drives the XÃ¢â€ â€™X
        # columns.  Anything else disables the feature (with one warning).
        self._self_attention_type = str(
            config["model"]["kwargs"].get("self_attention_type", "") or ""
        )
        self._descendant_mask_supported = (
            self._self_attention_type in self._DIRECTED_SELF_ATTENTION_TYPES
        )
        if self.hsic_exclude_descendants and not self._descendant_mask_supported:
            logger.warning(
                "training.hsic_exclude_descendants=True but self_attention_type=%r "
                "does not expose a DIRECTED edge posterior (supported: %s). "
                "Descendant exclusion is DISABLED Ã¢â‚¬â€ without an orientation the "
                "descendant set is undefined.",
                self._self_attention_type,
                ", ".join(self._DIRECTED_SELF_ATTENTION_TYPES),
            )
        # Epoch from which ``hsic_descendant_warmup_epochs`` is counted.
        #
        #   * ``0`` (default)  -> warmup counted from the START OF THE RUN.  This
        #     is the right semantics for the plain/staged trainers, where the
        #     structural loss is active from epoch 0, and it reproduces the
        #     original global-epoch behaviour exactly.
        #   * ``k``            -> warmup counted from global epoch ``k``.  The
        #     adaptive trainer sets this to the first epoch of the FIRST
        #     structure phase: under adaptive training everything runs in a
        #     single ``fit()``, so ``current_epoch`` is global and a raw
        #     threshold would be silently expired by the (long) reconstruct
        #     warmup phase before a single structural step had run.
        #   * ``None``         -> warmup already served, never delay again (set
        #     by the adaptive trainer on the second and later structure phases,
        #     so the warmup is a one-off and not re-paid every cycle).
        self._descendant_warmup_anchor: Optional[int] = 0
        # Running diagnostics (surfaced as logged metrics each step).
        self._descendant_ema_score: Optional[torch.Tensor] = None
        self._last_hsic_desc_kept_frac = 1.0
        self._last_hsic_desc_cyclic = False
        self._last_hsic_bkd_kept_frac = 1.0
        # LOO gamma gate state (detached, EMA-smoothed across refreshes).
        self._loo_gamma_cache: Optional[torch.Tensor] = None
        self._loo_gamma_step = 0
        self._last_loo_gamma_mean = 1.0
        self._last_loo_gamma_min = 1.0


        # ----------------------------------------------------------------
        # Group-L1 regularization (L2,1 norm on embedding columns)
        # ----------------------------------------------------------------
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 0.0)

        # ----------------------------------------------------------------
        # L0 regularization (HardConcreteCrossAttention only)
        # ----------------------------------------------------------------
        self.lambda_l0 = float(config["training"].get("lambda_l0", 0.0))

        # ----------------------------------------------------------------
        # Query-norm over-spend penalty (learnable per-node budget).  Charges
        # ``relu(M_i - target)^2`` on the STRUCTURAL loss only (see
        # causaliT.utils.query_norm); 0.0 (default) leaves behaviour unchanged.
        # ----------------------------------------------------------------
        self.lambda_query_norm = float(config["training"].get("lambda_query_norm", 0.0))

        # ----------------------------------------------------------------
        # Fan-in prior (experiment.fanin_prior, in EDGES).  Anneals the
        # over-spend target from mu=1 down to mu=sqrt(K*/N) over STRUCTURE
        # epochs, which by Lemma 1 caps how many parents a row can hold at the
        # target posterior.  Inert unless fanin_prior is set, so the default is
        # bit-identical to the pre-feature behaviour.  See
        # docs/experimental_elaborations/QUERY_NORM_CAPACITY_AND_FANIN_PRIOR.md.
        # ----------------------------------------------------------------
        self.fanin_schedule = FaninPriorSchedule(config, n_keys=self.N)

        # ----------------------------------------------------------------
        # L0 Ã¢â€ â€ HSIC gradient-interference logging (diagnostic).

        # When enabled AND the attention is HardConcreteCrossAttention AND
        # both lambda_l0 > 0 and lambda_hsic > 0, we log the per-block cosine
        # similarity between the L0 gradient and the HSIC gradient.  Negative
        # cosine Ã¢â€¡â€™ the two objectives push the structural parameters in
        # opposing directions (interference); positive Ã¢â€¡â€™ aligned.
        #
        # The two objectives share the structural pathway (Q/K projections and
        # structural embeddings), because the L0 penalty is a function of
        # log_alpha = QK^T/sqrt(E) and HSIC back-props through the attention
        # output, so this cosine localises where they conflict.
        # ----------------------------------------------------------------
        self.log_l0_hsic_interference = bool(
            config["training"].get("log_l0_hsic_interference", False)
        )
        self.interference_log_every_n_epochs = int(
            config["training"].get("interference_log_every_n_epochs", 1)
        )
        # Effective attention type for the interference gate.  In homogeneous
        # mode the cross ``attention_type`` is IGNORED by the architecture Ã¢â‚¬â€ the
        # single square block is built from ``self_attention_type`` Ã¢â‚¬â€ so that is
        # the type whose L0 gate the probe would see.
        self._attention_type = (
            config["model"]["kwargs"].get("self_attention_type", "")
            if self.homogeneous_nodes
            else config["model"]["kwargs"].get("attention_type", "")
        )
        # Cached block Ã¢â€ â€™ parameter-list mapping (built lazily on first use so
        # it reflects any requires_grad freezing applied in on_fit_start).
        self._interference_blocks: Optional[Dict[str, list]] = None
        # Stash for the two reg tensors so training_step can probe them while
        # the autograd graph is still alive.
        self._last_hsic_reg: Optional[torch.Tensor] = None
        # Structural pair weights (descendant x LOO, no BKD) stashed by _step
        # for the bilevel probe and the unrolled shadow (None when unused).
        self._last_probe_pair_mask: Optional[torch.Tensor] = None
        self._last_l0_reg: Optional[torch.Tensor] = None

        # ----------------------------------------------------------------
        # Acyclicity regularization (NOTEARS) Ã¢â‚¬â€ XÃ¢â€ â€™X sub-block only
        # Applied to the square (L_X, L_X) portion of the combined score
        # tensor (columns S_seq_len:).  The SÃ¢â€ â€™X block is bipartite and
        # inherently acyclic, so NOTEARS is not needed there.
        # Set kappa > 0 to activate; kappa=0.0 is the default (off).
        # ----------------------------------------------------------------
        self.kappa = float(config["training"].get("kappa", 0.0))
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")

        # Acyclicity functional applied to the (nonnegative) gate-posterior
        # score matrix.  "notears" (default, backward compatible):
        # h(A) = tr(exp(A Ã¢Å â„¢ A)) - d (Zheng et al., 2018).  "logdet" (DAGMA-
        # style, Bello et al., 2022): h(A) = -log det(sI - A) + d*log s.
        # Because the gate posterior has entries in (0, 1), the adaptive
        # scale s = max-row-sum(A) guarantees sI - A nonsingular at EVERY
        # iterate (rho(A) <= max row sum), removing the determinant
        # singularity that destabilises log-det on unbounded weights.
        # "nilpotent": h(A) = sum_{k=1..d} tr(A^k) Ã¢â‚¬â€ an exact acyclicity
        # characterisation for nonnegative A (all terms >= 0, zero iff the
        # graph has no closed walk of any length <= d).
        self.acyclicity_fn = str(
            config["training"].get("acyclicity_fn", "notears")
        )
        if self.acyclicity_fn not in ("notears", "logdet", "nilpotent"):
            raise ValueError(
                f"acyclicity_fn must be one of 'notears' | 'logdet' | "
                f"'nilpotent', got {self.acyclicity_fn!r}"
            )
        # DAGMA shift for the log-det backend: "adaptive" (default) uses the
        # (stop-grad) max row sum of the current score matrix; a positive
        # float fixes s globally (pure DAGMA uses s = 1).
        _acy_s = config["training"].get("acyclicity_s", "adaptive")
        if isinstance(_acy_s, str):
            if _acy_s != "adaptive":
                raise ValueError(
                    f"acyclicity_s must be 'adaptive' or a positive float, "
                    f"got {_acy_s!r}"
                )
            self.acyclicity_s: object = "adaptive"
        else:
            _acy_s = float(_acy_s)
            if _acy_s <= 0.0:
                raise ValueError(
                    f"acyclicity_s must be positive, got {_acy_s}"
                )
            self.acyclicity_s = _acy_s

        # ----------------------------------------------------------------
        # Augmented-Lagrangian acyclicity constraint (canonical NOTEARS
        # protocol, Zheng et al., 2018; see vendor/notears/linear.py):
        #
        #   L += alpha * h(W) + (rho/2) * h(W)^2
        #
        # with per-epoch dual ascent on alpha and NOTEARS-rule rho
        # escalation (rho <- rho*mult whenever the EMA of h fails to shrink
        # to <= 1/4 of its previous value).  All acyclicity backends are
        # >= 0 by construction, so the equality constraint h(W) = 0 needs
        # no relu.  Unlike the fixed kappa soft penalty, h(W) is DRIVEN to
        # ~0 (h_tol): the learned graph is asymptotically acyclic rather
        # than merely biased.  Mutually exclusive with kappa and
        # kappa_max_hsic_pct (the dynamic dual coefficient replaces the
        # fixed weight and its HSIC-anchored cap).
        # ----------------------------------------------------------------
        ac_cfg = config["training"].get("acyclicity_constraint", None) or {}
        self.acyclicity_constraint_enabled = bool(ac_cfg.get("enabled", False))
        if self.acyclicity_constraint_enabled:
            conflicting = []
            if self.kappa > 0.0:
                conflicting.append("kappa")
            # NOTE: kappa_max_hsic_pct is parsed LATER in __init__ (the
            # safeguard block), so read it from the config here.
            if float(config["training"].get("kappa_max_hsic_pct", 0.0)) > 0.0:
                conflicting.append("kappa_max_hsic_pct")
            if conflicting:
                raise ValueError(
                    "training.acyclicity_constraint.enabled=True is mutually "
                    f"exclusive with {conflicting}: the dual ascent replaces "
                    "the fixed kappa weight (and its HSIC-anchored cap). "
                    "Set them to 0 in the config."
                )
        self.acy_dual_lr = float(ac_cfg.get("dual_lr", 1.0))
        if self.acy_dual_lr <= 0.0:
            raise ValueError(
                "acyclicity_constraint.dual_lr must be > 0, got "
                f"{self.acy_dual_lr}"
            )
        self.acy_dual_max = float(ac_cfg.get("dual_max", 1.0e4))
        self.acy_rho_init = float(ac_cfg.get("rho_init", 1.0))
        if self.acy_rho_init < 0.0:
            raise ValueError(
                "acyclicity_constraint.rho_init must be >= 0, got "
                f"{self.acy_rho_init}"
            )
        self.acy_rho_mult = float(ac_cfg.get("rho_mult", 10.0))
        self.acy_rho_max = float(ac_cfg.get("rho_max", 1.0e8))
        self.acy_h_tol = float(ac_cfg.get("h_tol", 1.0e-8))
        if self.acy_h_tol < 0.0:
            raise ValueError(
                "acyclicity_constraint.h_tol must be >= 0, got "
                f"{self.acy_h_tol}"
            )
        self.acy_ema_decay = float(ac_cfg.get("ema", 0.9))
        if not (0.0 <= self.acy_ema_decay < 1.0):
            raise ValueError(
                "acyclicity_constraint.ema must be in [0, 1), got "
                f"{self.acy_ema_decay}"
            )
        # Dual state (plain floats; persisted via on_save_checkpoint).
        self._acy_dual_lambda = float(ac_cfg.get("dual_init", 0.0))
        if self._acy_dual_lambda < 0.0:
            raise ValueError(
                "acyclicity_constraint.dual_init must be >= 0, got "
                f"{self._acy_dual_lambda}"
            )
        self._acy_rho = self.acy_rho_init
        self._acy_ema: Optional[float] = None
        self._acy_prev_violation: Optional[float] = None

        # ----------------------------------------------------------------
        # Structural-regularizer safeguard (NOTEARS / L0 <= pct * HSIC).
        #
        # When the train HSIC goes flat ("diluted"), a fixed kappa /
        # lambda_l0 can dominate the structural pathway and drive structure
        # learning on its own (the NOTEARS-driven ill region).  The
        # safeguard caps each coefficient per step so the weighted term
        # entering the loss never exceeds a fixed fraction of the weighted
        # HSIC term:
        #   kappa_eff     = min(kappa,     pct * hsic_ref / (h(A) + eps))
        #   lambda_l0_eff = min(lambda_l0, pct * hsic_ref / (l0  + eps))
        # with hsic_ref an EMA of hsic_reg (detached, train batches only).
        # The rescaling is a detached per-step constant, so each regularizer
        # keeps its gradient direction; only its magnitude is capped.  When
        # HSIC -> 0 the capped terms fade out with it (HSIC stays the
        # primary structural driver).  0.0 (default) disables each cap
        # (backward compatible).
        # ----------------------------------------------------------------
        self.kappa_max_hsic_pct = float(
            config["training"].get("kappa_max_hsic_pct", 0.0)
        )
        self.lambda_l0_max_hsic_pct = float(
            config["training"].get("lambda_l0_max_hsic_pct", 0.0)
        )
        self.hsic_safeguard_ema = float(
            config["training"].get("hsic_safeguard_ema", 0.9)
        )
        if not (0.0 <= self.hsic_safeguard_ema < 1.0):
            raise ValueError(
                f"hsic_safeguard_ema must be in [0, 1), got {self.hsic_safeguard_ema}"
            )
        self._hsic_reg_ema: Optional[float] = None

        # ----------------------------------------------------------------
        # MSE-vs-acyclicity safeguard (single-optimizer DAGMA-like regime).
        #
        # Without gradient routing the plain MSE happily predicts a source
        # from its DESCENDANTS (anti-causal edges also reduce the MSE),
        # actively fighting the acyclicity term.  PCGrad-style surgery is
        # not applicable here (it was built for the structural-params
        # partition), so this cap applies the *scalar* safeguard pattern of
        # ``kappa_max_hsic_pct`` with the roles inverted: the effective
        # reconstruction weight is capped per step so the weighted MSE
        # entering the loss never exceeds a fixed fraction of the EMA of
        # the WEIGHTED acyclic term:
        #
        #   lambda_recon_eff = min(lambda_recon, pct * acy_ref / (mse + eps))
        #
        # with acy_ref an EMA of acyclic_reg (detached, train batches
        # only).  The rescaling is a detached per-step constant, so the MSE
        # keeps its gradient direction; only its magnitude is capped.  The
        # cap therefore lets the acyclic term dominate while it is
        # violated.  Because the reference shrinks with h, the cap would
        # strangle the MSE once the graph is (near-)acyclic; the optional
        # ``mse_cap_release_tol`` releases the cap (lambda_recon restored)
        # once the acyclic EMA reference falls at/below the tolerance
        # (e.g. set it to the ALM h_tol).  0.0 (default pct) disables the
        # cap (backward compatible).  Mutually exclusive with
        # use_gradient_routing and gradient_surgery (those paths decompose
        # the loss into per-term backward passes and do not use the capped
        # total_loss).
        # ----------------------------------------------------------------
        self.mse_max_acyclic_pct = float(
            config["training"].get("mse_max_acyclic_pct", 0.0)
        )
        if self.mse_max_acyclic_pct < 0.0:
            raise ValueError(
                f"mse_max_acyclic_pct must be non-negative, got "
                f"{self.mse_max_acyclic_pct}"
            )
        self.mse_cap_release_tol = float(
            config["training"].get("mse_cap_release_tol", 0.0)
        )
        if self.mse_cap_release_tol < 0.0:
            raise ValueError(
                f"mse_cap_release_tol must be non-negative, got "
                f"{self.mse_cap_release_tol}"
            )
        # L0-vs-acyclicity cap: same scalar-safeguard pattern as the MSE cap,
        # applied to the HardConcrete L0 weight.  Motivation: in HSIC-free
        # arms (lambda_hsic=0) a FIXED lambda_l0 is unopposed -- every open
        # gate receives constant closing pressure and, integrated over a long
        # run, the posterior collapses to the empty graph (train_l0_penalty
        # decays monotonically to ~0).  Anchoring to the acyclic reference
        # makes the L0 pressure fade as h -> 0, so the gates FREEZE once the
        # graph is acyclic instead of being ground down:
        #   lambda_l0_eff = min(lambda_l0, pct * acy_ref / (l0 + eps))
        # sharing the SAME acy_ref (and mse_cap_release_tol release) as the
        # MSE cap.  Mutually exclusive with the HSIC-anchored cap
        # ``lambda_l0_max_hsic_pct`` (one anchor per coefficient).  0.0
        # (default) disables it (backward compatible).
        self.l0_max_acyclic_pct = float(
            config["training"].get("l0_max_acyclic_pct", 0.0)
        )
        if self.l0_max_acyclic_pct < 0.0:
            raise ValueError(
                f"l0_max_acyclic_pct must be non-negative, got "
                f"{self.l0_max_acyclic_pct}"
            )
        if self.l0_max_acyclic_pct > 0.0 and self.lambda_l0_max_hsic_pct > 0.0:
            raise ValueError(
                "training.l0_max_acyclic_pct and training.lambda_l0_max_hsic_pct "
                "are mutually exclusive: both cap lambda_l0, anchored to "
                "different references (acyclic EMA vs HSIC EMA).  Pick one."
            )
        self._acyclic_reg_ema: Optional[float] = None


        # ----------------------------------------------------------------
        # Gradient routing (dual optimizer: structural vs reconstruction)
        # ----------------------------------------------------------------
        self.use_gradient_routing = config["training"].get("use_gradient_routing", False)
        if self.use_gradient_routing:
            self.automatic_optimization = False
            structural_params, reconstruction_params = classify_parameters(
                self.model, verbose=True
            )
            self._structural_params = structural_params
            self._reconstruction_params = reconstruction_params

        # ----------------------------------------------------------------
        # Gradient surgery (PCGrad): per-block projection of the L0 / NOTEARS
        # gradients against the HSIC gradient, keeping only the component of
        # each regularizer that is non-destructive for HSIC (Yu et al.,
        # NeurIPS 2020).  Requires gradient routing (a dedicated structural
        # backward to operate on).  See causaliT/training/gradient_surgery.py.
        # ----------------------------------------------------------------
        self.gradient_surgery = bool(
            config["training"].get("gradient_surgery", False)
        )
        if self.gradient_surgery and not self.use_gradient_routing:
            # Joint PCGrad: single optimizer over ALL parameters, but manual
            # optimization so the per-term autograd.grad calls (recon / HSIC /
            # L0 / NOTEARS / rest) can run and be reconciled before the step
            # (see _joint_pcgrad_step).  Same decomposition as the routing
            # path, extended beyond the structural-params partition.
            self.automatic_optimization = False

        if self.mse_max_acyclic_pct > 0.0 or self.l0_max_acyclic_pct > 0.0:
            if self.use_gradient_routing or self.gradient_surgery:
                raise ValueError(
                    "training.mse_max_acyclic_pct / l0_max_acyclic_pct > 0 is "
                    "mutually exclusive with use_gradient_routing and "
                    "gradient_surgery: the caps act on the single-optimizer "
                    "total_loss, which those paths bypass with per-term "
                    "backward passes."
                )


        # ----------------------------------------------------------------
        # HSIC as a CONSTRAINT (Lagrangian / augmented Lagrangian) instead of
        # a fixed-weight supervised penalty (``training.hsic_constraint``).
        #
        #   min_thetaS  L0 + NOTEARS        s.t.  HSIC(thetaS) <= tolerance
        #
        #   L = L0 + NOTEARS + lam*(HSIC - eps) + (rho/2)*relu(HSIC - eps)^2
        #
        # with per-epoch dual ascent on lam (projected to [0, dual_max]) and
        # optional NOTEARS-style rho escalation, both driven by an EMA of the
        # RAW train HSIC.  Unlike the fixed ``lambda_hsic`` penalty, the dual
        # variable grows while the constraint is violated, so the pressure on
        # the structural pathway does NOT vanish as HSIC shrinks.
        #
        # This REVERSES the HSIC vs L0/NOTEARS relationship: the structural
        # regularizers are the primal objective and HSIC is the monitor.  It
        # is therefore mutually exclusive with every safeguard built for the
        # opposite regime (PCGrad ``gradient_surgery``, the HSIC-relative
        # caps ``*_max_hsic_pct``, and a fixed ``lambda_hsic > 0`` weight).
        # ----------------------------------------------------------------
        hc_cfg = config["training"].get("hsic_constraint", None) or {}
        self.hsic_constraint_enabled = bool(hc_cfg.get("enabled", False))
        self.hsic_constraint_source = str(hc_cfg.get("source", "hsic"))
        if self.hsic_constraint_enabled:
            conflicting = []
            if self.gradient_surgery:
                conflicting.append("gradient_surgery")
            if self.kappa_max_hsic_pct > 0.0:
                conflicting.append("kappa_max_hsic_pct")
            if self.lambda_l0_max_hsic_pct > 0.0:
                conflicting.append("lambda_l0_max_hsic_pct")
            if float(self.lambda_hsic) > 0.0:
                conflicting.append("lambda_hsic")
            if conflicting:
                raise ValueError(
                    "training.hsic_constraint.enabled=True is mutually "
                    "exclusive with the HSIC-supervision safeguards "
                    f"{conflicting} (the constraint formulation makes HSIC "
                    "the monitored constraint of the L0+NOTEARS objective; "
                    "these options implement the reversed relationship). "
                    "Disable them in the config."
                )
            for legacy in ("use_hsic_annealing", "use_causal_init"):
                if config["training"].get(legacy, False):
                    logger.warning(
                        "training.%s is set but hsic_constraint is enabled: "
                        "the fixed-weight HSIC schedule it drives is inactive "
                        "(lambda_hsic must be 0); the constraint dual ascent "
                        "replaces it.",
                        legacy,
                    )
            if self.hsic_constraint_source not in ("hsic", "oracle_shd"):
                raise ValueError(
                    "hsic_constraint.source must be 'hsic' or 'oracle_shd', "
                    f"got {self.hsic_constraint_source!r}"
                )
            if self.hsic_constraint_source == "oracle_shd":
                # Leakage guard: the GT must NEVER reach the forward pass.
                # ``oracle_combined_mask`` is intersected into the attention
                # hard mask whenever it is not None (regardless of the oracle
                # flag), so both GT-consuming modes must be off; the GT is
                # loaded into a dedicated ``oracle_shd_gt`` buffer.
                if config["training"].get("use_hard_masks", False):
                    raise ValueError(
                        "hsic_constraint.source='oracle_shd' requires "
                        "use_hard_masks=False (the loaded mask is intersected "
                        "into the attention hard mask even without oracle "
                        "mode, which would leak the GT into the forward pass)."
                    )
                if config["training"].get("use_oracle_attention", False):
                    raise ValueError(
                        "hsic_constraint.source='oracle_shd' requires "
                        "use_oracle_attention=False (GT leakage into the "
                        "forward pass)."
                    )
            self.hsic_tol = float(hc_cfg.get("tolerance", 0.0))
            if self.hsic_tol < 0.0:
                raise ValueError(
                    f"hsic_constraint.tolerance must be >= 0, got {self.hsic_tol}"
                )
            self.hsic_dual_lr = float(hc_cfg.get("dual_lr", 1.0))
            if self.hsic_dual_lr <= 0.0:
                raise ValueError(
                    f"hsic_constraint.dual_lr must be > 0, got {self.hsic_dual_lr}"
                )
            self.hsic_dual_max = float(hc_cfg.get("dual_max", 1000.0))
            self.hsic_rho_init = float(hc_cfg.get("rho_init", 0.0))
            if self.hsic_rho_init < 0.0:
                raise ValueError(
                    f"hsic_constraint.rho_init must be >= 0, got {self.hsic_rho_init}"
                )
            self.hsic_rho_mult = float(hc_cfg.get("rho_mult", 2.0))
            self.hsic_rho_max = float(hc_cfg.get("rho_max", 1e6))
            self.hsic_constraint_ema_decay = float(hc_cfg.get("ema", 0.9))
            if not (0.0 <= self.hsic_constraint_ema_decay < 1.0):
                raise ValueError(
                    "hsic_constraint.ema must be in [0, 1), got "
                    f"{self.hsic_constraint_ema_decay}"
                )
            # Dual state (plain floats; persisted via on_save_checkpoint).
            self._hsic_dual_lambda = float(hc_cfg.get("dual_init", 0.0))
            if self._hsic_dual_lambda < 0.0:
                raise ValueError(
                    "hsic_constraint.dual_init must be >= 0, got "
                    f"{self._hsic_dual_lambda}"
                )
            self._hsic_rho = self.hsic_rho_init
            self._hsic_constraint_ema: Optional[float] = None
            self._hsic_constraint_prev_violation: Optional[float] = None


            # --- Per-rung tolerance calibration + dual-ascent control -------
            # ``calibrate_tolerance``: at every structure-phase entry, estimate
            # the permutation null of the (biased) HSIC on the CURRENT regime
            # (freshly-frozen recon residuals, current BKD rung, structure
            # fold) and set ``tolerance = quantile(null) * margin``.  The
            # static ``tolerance`` then only serves until the first
            # calibration fires.
            self.hsic_calibrate_tolerance = bool(
                hc_cfg.get("calibrate_tolerance", False)
            )
            self.hsic_calib_batches = int(hc_cfg.get("calibration_batches", 8))
            self.hsic_calib_perms = int(hc_cfg.get("calibration_permutations", 4))
            self.hsic_calib_quantile = float(
                hc_cfg.get("calibration_quantile", 0.99)
            )
            if not (0.0 < self.hsic_calib_quantile < 1.0):
                raise ValueError(
                    "hsic_constraint.calibration_quantile must be in (0, 1), "
                    f"got {self.hsic_calib_quantile}"
                )
            self.hsic_calib_margin = float(hc_cfg.get("calibration_margin", 2.0))
            if self.hsic_calib_margin <= 0.0:
                raise ValueError(
                    "hsic_constraint.calibration_margin must be > 0, got "
                    f"{self.hsic_calib_margin}"
                )
            self.hsic_calib_seed = int(hc_cfg.get("calibration_seed", 20240817))
            # Asymmetric dual learning rate: downward moves (constraint
            # satisfied) may use a different rate than upward ones so the
            # accumulated pressure can actually release.
            self.hsic_dual_lr_down = float(
                hc_cfg.get("dual_lr_down", self.hsic_dual_lr)
            )
            if self.hsic_dual_lr_down <= 0.0:
                raise ValueError(
                    "hsic_constraint.dual_lr_down must be > 0, got "
                    f"{self.hsic_dual_lr_down}"
                )
            # Low-rung pause: while the BKD key budget is below the graph's
            # plausible max in-degree the independence constraint is
            # unreachable BY CONSTRUCTION (residuals necessarily depend on
            # un-kept true parents), so dual ascent integrates pure
            # unreachable pressure.  ``dual_pause_below_keys > 0`` skips the
            # ascent at rungs with fewer keys (0 = disabled).
            self.hsic_dual_pause_below_keys = int(
                hc_cfg.get("dual_pause_below_keys", 0)
            )
            # Under the adaptive trainer, only structure phases integrate the
            # constraint: the reconstruct phase changes the residual regime
            # with FROZEN gates, so its violation is not actionable by the
            # structural stream.  No-op for non-adaptive runs (the phase
            # hook is never called and the flag stays True).
            self.hsic_dual_structure_only = bool(
                hc_cfg.get("dual_ascent_structure_only", True)
            )
            # Run state.
            self._hsic_dual_active: bool = True
            self._hsic_rung_keys: Optional[int] = None
            self._hsic_null_samples: list = []
            self._hsic_null_calib_remaining: int = 0
            self._hsic_null_calib_round: int = 0

        # ----------------------------------------------------------------
        # Node-wise (per-query) winner-take-all structural update.  Each
        # structural step updates only the ``topk`` query nodes whose gradient
        # has the strongest SNR evidence (EMA t-statistic); all other query
        # rows and their optimizer state are reverted after the step, so any
        # optimizer works unchanged.  Requires gradient routing (a dedicated
        # structural optimizer).  See causaliT/training/nodewise_update.py.
        # ----------------------------------------------------------------
        nw_cfg = config["training"].get("nodewise_update", None) or {}
        self.nodewise_enabled = bool(nw_cfg.get("enabled", False))
        self.nodewise_reset_every_stage = bool(
            nw_cfg.get("reset_every_stage", True)
        )
        self._nodewise: Optional[NodewiseQuerySelector] = None
        if self.nodewise_enabled:
            if not self.use_gradient_routing:
                raise ValueError(
                    "training.nodewise_update.enabled requires "
                    "use_gradient_routing=True (the nodewise gate acts on the "
                    "structural optimizer step)."
                )
            query_params = [
                t.embedding.weight
                for t in (getattr(self.model, "query_embed_S", None),
                          getattr(self.model, "query_embed_X", None))
                if t is not None
            ]
            if not query_params:
                raise ValueError(
                    "training.nodewise_update.enabled requires free query "
                    "embeddings (query_embed_S / query_embed_X)."
                )
            norm_params = [
                ia.query_norm_log_scale
                for ia in (getattr(getattr(self.model, "attention", None),
                                   "inner_attention", None),
                           getattr(getattr(self.model, "self_attention", None),
                                   "inner_attention", None))
                if getattr(ia, "query_norm_log_scale", None) is not None
            ]
            norm_param = norm_params[0] if norm_params else None
            self._nodewise = NodewiseQuerySelector(
                query_params=query_params,
                norm_param=norm_param,
                topk=int(nw_cfg.get("topk", 1)),
                selection=str(nw_cfg.get("selection", "snr")),
            )
            logger.info(
                "Nodewise query update enabled: topk=%d over %d nodes, "
                "selection=%s, reset_every_stage=%s",
                self._nodewise.topk, self._nodewise.n_nodes,
                self._nodewise.selection, self.nodewise_reset_every_stage,
            )

        # ----------------------------------------------------------------
        # Centroid-commit query dynamics (quantized queries with an evidence
        # shadow).  The query embedding weights hold COMMITTED key-subset
        # centroids; per-node shadows accumulate the leaked HSIC gradient and
        # a node re-commits when its shadow's exact nearest-centroid
        # projection changes subset.  See causaliT/training/centroid_commit.py.
        # ----------------------------------------------------------------
        cc_cfg = config["training"].get("centroid_commit", None) or {}
        self.centroid_commit_enabled = bool(cc_cfg.get("enabled", False))
        if (
            self.centroid_commit_enabled
            and self._query_parents_prior
            and any(s.get("fixed", False) for s in self._query_parents_prior.values())
        ):
            raise ValueError(
                "query_parents_prior with fixed=true is incompatible with "
                "centroid_commit: commit events rewrite the query rows "
                "outside the optimizer and would defeat the freeze."
            )
        if self.centroid_commit_enabled and self._query_source_prior:
            raise ValueError(
                "query_source_prior is incompatible with centroid_commit: "
                "commit events rewrite the query rows outside the optimizer "
                "and would defeat the source freeze."
            )

        self._commit_source = str(cc_cfg.get("shadow_source", "hsic"))
        if self._commit_source not in ("hsic", "structural", "hsic_unrolled"):
            raise ValueError(
                f"centroid_commit.shadow_source must be 'hsic', "
                f"'structural' or 'hsic_unrolled', got {self._commit_source!r}"
            )
        # Second-order (DARTS) shadow evidence: the shadow integrates the
        # destination-state HSIC gradient instead of the frozen-theta_R one.
        ur = (config["training"].get("unrolled", None)
              or cc_cfg.get("unrolled", None) or {})
        self._unrolled_inner_lr = ur.get("inner_lr", None)  # None -> recon lr
        self._unrolled_fd_eps = float(ur.get("fd_epsilon", 0.01))
        self._unrolled_every = max(1, int(ur.get("every", 1)))
        self._unrolled_step_count = 0

        # ----------------------------------------------------------------
        # Bi-level (DARTS second-order) STRUCTURAL gradient
        # (``training.structural_grad: hsic_unrolled``).  The structural
        # optimizer steps on the destination-state HSIC gradient: a virtual
        # reconstruction refit on fold B followed by the HSIC gradient at the
        # refit point, minus the finite-difference mixed-Hessian correction
        # (all passes on fold B when ``hsic_cross_fit`` is on).  The default
        # ``hsic`` keeps the first-order frozen-theta_R gradient.
        # ----------------------------------------------------------------
        self.structural_grad = str(
            config["training"].get("structural_grad", "hsic")
        )
        if self.structural_grad not in ("hsic", "hsic_unrolled"):
            raise ValueError(
                f"training.structural_grad must be 'hsic' or "
                f"'hsic_unrolled', got {self.structural_grad!r}"
            )
        if self.structural_grad == "hsic_unrolled":
            if not self.use_gradient_routing:
                raise ValueError(
                    "training.structural_grad='hsic_unrolled' requires "
                    "use_gradient_routing=True (a dedicated structural "
                    "optimizer step)."
                )
            if self.gradient_surgery:
                raise ValueError(
                    "training.structural_grad='hsic_unrolled' is mutually "
                    "exclusive with gradient_surgery (PCGrad operates on the "
                    "first-order per-term decomposition)."
                )
            if self.hsic_constraint_enabled:
                raise ValueError(
                    "training.structural_grad='hsic_unrolled' is not supported "
                    "with hsic_constraint (the dual-ascent scaling of the HSIC "
                    "term has no unrolled counterpart yet)."
                )
        self._commit: Optional[CentroidCommitController] = None
        if self.centroid_commit_enabled:
            if self._nodewise is not None:
                raise ValueError(
                    "centroid_commit and nodewise_update are alternative query "
                    "update rules; enable only one."
                )
            self.model.enable_centroid_commit()
            tables = [t for t in (self.model.query_embed_S,
                                  self.model.query_embed_X) if t is not None]
            # Lazy callable: the frame can be REPLACED by load_state_dict
            # (warm start / checkpoint resume) after this constructor runs.
            K = lambda: torch.cat([self.model.orth_embed_S.frame,
                                   self.model.orth_embed_X.frame]).detach()
            norm_params = [
                ia.query_norm_log_scale
                for ia in (getattr(getattr(self.model, "attention", None),
                                   "inner_attention", None),
                           getattr(getattr(self.model, "self_attention", None),
                                   "inner_attention", None))
                if getattr(ia, "query_norm_log_scale", None) is not None
            ]
            self._commit = CentroidCommitController(
                tables=tables,
                K=K,
                norm_param=norm_params[0] if norm_params else None,
                evidence_lr=float(cc_cfg.get("evidence_lr", 10.0)),
                evidence_leak=float(cc_cfg.get("evidence_leak", 0.95)),
                reset_m_on_commit=str(cc_cfg.get("reset_m_on_commit", "one")),
                prior_rho=float(cc_cfg.get("prior_rho", 0.0)),
                winner_take_all=bool(cc_cfg.get("winner_take_all", False)),
                commit_margin=float(cc_cfg.get("commit_margin", 0.0)),
                min_snr=float(cc_cfg.get("min_snr", 0.0)),
            )
            logger.info(
                "Centroid-commit query dynamics enabled: %d nodes, "
                "evidence_lr=%.3g, leak=%.3g, reset_m=%s, prior_rho=%.3g, "
                "wta=%s, commit_margin=%.3g, min_snr=%.3g",
                self._commit.n_nodes, self._commit.eta, self._commit.beta,
                self._commit.reset_m_on_commit, self._commit.prior_rho,
                self._commit.winner_take_all, self._commit.commit_margin,
                self._commit.min_snr,
            )

        # ------------------------------------------------------------------
        # Bilevel commit gate (Phase 2, docs/ideas/BILEVEL_CENTROID_COMMIT.md)
        # Eligible commits are DEFERRED and accepted only if the node's HSIC
        # row drops after a paired k-step reconstruction refit on validation
        # batches; rejected candidates are tabooed until the shadow moves on.
        # ------------------------------------------------------------------
        bg = cc_cfg.get("bilevel_gate", None) or {}
        self._gate_enabled = bool(bg.get("enabled", False))
        self._gate_k_inner = int(bg.get("k_inner", 15))
        self._gate_inner_optimizer = bg.get("inner_optimizer", None)
        self._gate_inner_lr = bg.get("inner_lr", None)
        self._gate_inner_wd = bg.get("inner_weight_decay", None)
        self._gate_margin = float(bg.get("accept_margin", 0.0))
        self._gate_max_val = int(bg.get("max_val_batches", 8))
        self._val_probe_cache: list = []
        if self._gate_enabled:
            if not self.centroid_commit_enabled:
                raise ValueError(
                    "centroid_commit.bilevel_gate requires "
                    "centroid_commit.enabled: true."
                )
            logger.info(
                "Bilevel commit gate enabled: k_inner=%d, inner_lr=%s, "
                "accept_margin=%.3g, max_val_batches=%d",
                self._gate_k_inner, self._gate_inner_lr, self._gate_margin,
                self._gate_max_val,
            )

        # ----------------------------------------------------------------
        # Parameter freezing for alternating structure/reconstruct phases
        # Set by the training config (config['training']['freeze_*_params']).
        # Requires use_gradient_routing=True; otherwise the config loader
        # falls back to loss-level gating and leaves these False.
        # Applied in on_fit_start so that warm-started weights can be loaded
        # first and frozen second (requires_grad is not saved in checkpoints).
        # ----------------------------------------------------------------
        self.freeze_structural_params = bool(
            config["training"].get("freeze_structural_params", False)
        )
        self.freeze_reconstruction_params = bool(
            config["training"].get("freeze_reconstruction_params", False)
        )

        # ----------------------------------------------------------------
        # Oracle mode
        # When use_oracle_attention=True the forecaster bypasses QK^T and
        # feeds the GT DAG hard mask (combined SÃ¢â€ â€™X Ã¢â‚¬â€“ XÃ¢â€ â€™X) directly as the
        # attention weight matrix so that only the value/FFN/MLP head is
        # trained from the reconstruction loss.
        # Requires use_hard_masks=True; validated below.
        # ----------------------------------------------------------------
        self.use_oracle = config["training"].get("use_oracle_attention", False)

        # ----------------------------------------------------------------
        # Hard mask configuration
        # Mirrors SingleCausalForecaster / NoiseAwareCausalForecaster.
        # ----------------------------------------------------------------
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False

        if self.use_oracle and not self.use_hard_masks:
            raise ValueError(
                "training.use_oracle_attention=True requires "
                "training.use_hard_masks=True.  The oracle uses the loaded "
                "GT DAG combined mask as the attention weight matrix."
            )

        # Wrong-DAG oracle controls Ã¢â‚¬â€ same semantics as SingleCausalForecaster.
        # seed in {None, 0} OR both SHDs == 0  Ã¢â€ â€™  no corruption.
        self.hard_masks_corruption_seed = config["training"].get(
            "hard_masks_corruption_seed", None
        )
        self.cross_control_shd = int(
            config["training"].get("cross_control_shd", 0) or 0
        )
        self.self_control_shd = int(
            config["training"].get("self_control_shd", 0) or 0
        )
        self.hard_masks_preserve_sparsity = bool(
            config["training"].get("hard_masks_preserve_sparsity", False)
        )
        # Filled in by _load_combined_oracle_mask when corruption is applied.
        self.hard_mask_corruption_info: Optional[Dict[str, dict]] = None

        # Load and build the combined oracle mask if masks are enabled.
        if self.use_hard_masks and data_dir is not None:
            self._load_combined_oracle_mask(config, data_dir)
        elif self.use_hard_masks and data_dir is None:
            print(
                "Warning: training.use_hard_masks=True but data_dir was not "
                "provided to AttentionSelectorForecaster.  Hard masks will "
                "not be loaded.  Pass data_dir via create_model_instance."
            )

        # Oracle-SHD constraint: GT adjacency for the constraint monitor,
        # loaded into a dedicated buffer that NEVER reaches the forward pass.
        if self.hsic_constraint_source == "oracle_shd":
            if data_dir is None:
                # Eval/notebook loading path: the GT buffer is only needed to
                # TRAIN with the constraint; skip with a warning instead of
                # raising so load_from_checkpoint works without data_dir.
                logger.warning(
                    "hsic_constraint.source='oracle_shd' but data_dir is "
                    "None: GT buffer not loaded (fine for evaluation; "
                    "training with the constraint would fail)."
                )
            else:
                self._load_oracle_shd_gt(config, data_dir)

        self.save_hyperparameters(config)

        # Metrics
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()

        # --- Fit metrics that are actually interpretable -------------------
        # ``r2_x`` is POOLED over every node and sample (``reshape(-1)``), so
        # (a) in homogeneous mode it includes the exogenous S rows, whose
        # causally-correct R2 is ~0, and (b) between-node variance leaks into
        # SStot, making the value depend on the relative variances of the
        # variables rather than on fit quality alone.  ``r2_x`` is kept
        # UNCHANGED (it is in every historical metrics.csv) and the three keys
        # below are added next to it:
        #   x_r2_endo  pooled R2 over the ENDOGENOUS rows only (== x_r2 in
        #              split mode, where the target already excludes S)
        #   x_r2_macro mean of PER-NODE R2 over the endogenous rows; each node
        #              is normalised by its own variance, so the value is
        #              pooling-free and invariant to per-node rescaling
        #   x_r2_src   pooled R2 over the S rows (homogeneous mode only).  This
        #              is a DIAGNOSTIC, not noise: an exogenous source can only
        #              be predicted from its own descendants, so a high value
        #              means the posterior is being used ANTI-CAUSALLY.
        self.r2_x_endo = tm.R2Score()
        # torchmetrics compatibility: ``num_outputs`` was deprecated in 1.5.0
        # and REMOVED in 1.6.0, where R2Score infers the output count from the
        # input shape instead.  Passing it to >= 1.6 raises
        # "Unexpected keyword arguments: num_outputs" (it falls through to
        # Metric.__init__).  requirements.txt pins 1.0.3, but cluster envs
        # routinely carry a newer build, so probe the signature rather than the
        # version string and stay correct on both.
        _r2_macro_kwargs: Dict[str, Any] = {"multioutput": "uniform_average"}
        if "num_outputs" in inspect.signature(tm.R2Score.__init__).parameters:
            _r2_macro_kwargs["num_outputs"] = self.X_seq_len   # torchmetrics < 1.6
        self.r2_x_macro = tm.R2Score(**_r2_macro_kwargs)
        self.r2_x_src = (
            tm.R2Score() if self.homogeneous_nodes else None
        )


    # ------------------------------------------------------------------
    # Hard mask loading
    # ------------------------------------------------------------------

    def _load_combined_oracle_mask(self, config: dict, data_dir: str):
        """
        Load GT DAG mask CSVs, optionally corrupt them, and register the
        combined (L_X, L_S+L_X) oracle mask as a Lightning buffer.

        The combined oracle mask concatenates:
            dec_cross  (L_X, L_S)  Ã¢â‚¬â€ SÃ¢â€ â€™X GT edges
            dec_self   (L_X, L_X)  Ã¢â‚¬â€ XÃ¢â€ â€™X GT edges
        along dim=1 to produce (L_X, L_S+L_X), matching the shape of
        AttentionSelectorLayer.combined_mask.
        """
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files is None:
            print(
                "Warning: use_hard_masks=True but no hard_mask_files "
                "specified in training config.  Oracle mask not loaded."
            )
            return

        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)

        masks = load_dag_masks(dataset_dir, mask_files, device="cpu")
        if masks is None:
            print("Warning: No DAG mask files found.  Oracle mask not loaded.")
            return

        # Optional wrong-DAG oracle corruption
        corruption_info = None
        if (
            self.hard_masks_corruption_seed not in (None, 0)
            and (self.cross_control_shd > 0 or self.self_control_shd > 0)
        ):
            X_len = int(self.config["data"]["X_seq_len"])
            masks, corruption_info = corrupt_dag_masks(
                masks,
                seed=self.hard_masks_corruption_seed,
                cross_shd=self.cross_control_shd,
                self_shd=self.self_control_shd,
                X_len=X_len,
                preserve_sparsity=self.hard_masks_preserve_sparsity,
            )
            print(
                f"Ã¢Å“â€œ Oracle masks CORRUPTED "
                f"(seed={int(self.hard_masks_corruption_seed)}, "
                f"cross_shd={self.cross_control_shd}, "
                f"self_shd={self.self_control_shd}, "
                f"preserve_sparsity={self.hard_masks_preserve_sparsity})"
                f" Ã¢â‚¬â€ wrong-DAG oracle."
            )
            for _name, _info in corruption_info.items():
                cyc = _info.get("has_cycles")
                cyc_str = (
                    "N/A" if cyc is None else ("Ã¢Å¡Â  cycles" if cyc else "Ã¢Å“â€œ acyclic")
                )
                fb = " [fallback all-edges-wrong]" if _info.get("fallback_used") else ""
                print(
                    f"    - {_name}: shd_req={_info['shd_requested']}, "
                    f"shd_real={_info['shd_realised']}, "
                    f"k_true={_info['num_true_edges']}, "
                    f"pool={_info['eligible_pool_size']}, "
                    f"{cyc_str}{fb}"
                )
        self.hard_mask_corruption_info = corruption_info

        # Build combined (L_X, L_S+L_X) mask from dec_cross and dec_self
        cross_mask = masks.get("dec_cross", None)
        self_mask = masks.get("dec_self", None)

        if cross_mask is None or self_mask is None:
            print(
                "Warning: Expected 'dec_cross' and 'dec_self' in hard_mask_files "
                "but one or both are missing.  Oracle mask not registered."
            )
            return

        if self.homogeneous_nodes:
            # Homogeneous mode: the model expects the SQUARE (N, N) GT
            # adjacency because every node is a child.  Rows 0..L_S-1 are the
            # S children Ã¢â‚¬â€ sources have no parents in this dataset family, so
            # those rows are all-zero Ã¢â‚¬â€ and rows L_S..N-1 carry the X children's
            # parents as [dec_cross | dec_self].
            combined = torch.zeros(self.N, self.N, dtype=cross_mask.dtype)
            combined[self.S_seq_len :, : self.S_seq_len] = cross_mask
            combined[self.S_seq_len :, self.S_seq_len :] = self_mask
        else:
            # Split mode: concatenate [SÃ¢â€ â€™X part | XÃ¢â€ â€™X part] Ã¢â€ â€™ (L_X, L_S + L_X)
            combined = torch.cat([cross_mask, self_mask], dim=1)

        self.register_buffer("oracle_combined_mask", combined)
        self._hard_masks_loaded = True
        print(
            f"Ã¢Å“â€œ Oracle combined mask built: shape {combined.shape} "
            f"(cross {cross_mask.shape} Ã¢â‚¬â€“ self {self_mask.shape}"
            f"{', homogeneous square layout' if self.homogeneous_nodes else ''})"
        )

    def _load_oracle_shd_gt(self, config: dict, data_dir: str):
        """Load the GT DAG adjacency for the oracle-SHD constraint.

        Registered as the dedicated ``oracle_shd_gt`` buffer, which is NEVER
        passed to ``forward`` (the __init__ validation hard-errors when
        use_hard_masks / use_oracle_attention are on, so no GT can leak into
        the attention).  Same layout convention as ``oracle_combined_mask``:
        square (N, N) in homogeneous mode, (L_X, L_S+L_X) in split mode;
        entry [i, j] = 1 iff j is a parent of i.
        """
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files is None:
            raise ValueError(
                "hsic_constraint.source='oracle_shd' requires "
                "training.hard_mask_files (dec_cross / dec_self GT CSVs)."
            )
        dataset_dir = join(data_dir, config["data"]["dataset"])
        masks = load_dag_masks(dataset_dir, mask_files, device="cpu")
        if masks is None:
            raise ValueError(
                f"oracle_shd: no DAG mask files found in {dataset_dir}."
            )
        cross_mask = masks.get("dec_cross", None)
        self_mask = masks.get("dec_self", None)
        if cross_mask is None or self_mask is None:
            raise ValueError(
                "oracle_shd: expected 'dec_cross' and 'dec_self' masks."
            )
        if self.homogeneous_nodes:
            gt = torch.zeros(self.N, self.N, dtype=cross_mask.dtype)
            gt[self.S_seq_len :, : self.S_seq_len] = cross_mask
            gt[self.S_seq_len :, self.S_seq_len :] = self_mask
        else:
            gt = torch.cat([cross_mask, self_mask], dim=1)
        self.register_buffer("oracle_shd_gt", gt)
        print(
            f"[oracle_shd] GT adjacency loaded: shape {tuple(gt.shape)} "
            f"({int(gt.sum())} edges)"
        )


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data_source: torch.Tensor,
        data_intermediate: torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            data_source: S tensor, shape (B, L_S, features).
            data_intermediate: X tensor with actual values, shape (B, L_X, features).
                The value column is blanked internally for the query path.

        Returns:
            pred_x:            (B, L_X, 1) predictions Ã¢â‚¬â€ (B, N, 1) when
                               ``homogeneous_nodes=True`` (S nodes are children
                               too and therefore reconstructed as well).
            attention_weights: (B, L_X, L_S + L_X) combined attention matrix Ã¢â‚¬â€
                               square (B, N, N) when ``homogeneous_nodes=True``.
            entropy:           Attention entropy.
        """
        # Blank value column for the query path
        x_blanked = data_intermediate.clone()
        x_blanked[:, :, self.val_idx] = 0.0

        # Homogeneous mode: the S nodes are queries as well, so they need their
        # own value-blanked copy.  ``forward_with_actual`` raises ValueError if
        # this is missing, and ignores it in split mode.
        s_blanked = None
        if self.homogeneous_nodes:
            s_blanked = data_source.clone()
            s_blanked[:, :, self.val_idx] = 0.0


        # Retrieve the GT oracle mask when hard masks are loaded and oracle is
        # active. Gating on apply_hard_masks mirrors SingleCausalForecaster:
        # if hard masks are disabled (e.g. evaluation w/o GT), oracle falls
        # back to the structural mask so the learned attention is used instead.
        apply_hard_masks = self.use_hard_masks and self._hard_masks_loaded
        oracle = self.use_oracle and apply_hard_masks
        oracle_mask = (
            getattr(self, "oracle_combined_mask", None)
            if apply_hard_masks else None
        )

        return self.model.forward_with_actual(
            source_tensor=data_source,
            x_blanked=x_blanked,
            x_actual=data_intermediate,
            oracle=oracle,
            oracle_combined_mask=oracle_mask,
            s_blanked=s_blanked,
        )
        # Note: forward_with_actual returns (pred_x, attention_weights, entropy, l0_penalty).
        # All four values are passed through so that _step can access l0_penalty.

    # ------------------------------------------------------------------
    # Generative forward (interventional roll-out)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def causal_predict(
        self,
        data_source: torch.Tensor,
        x_init: torch.Tensor,
        clamp: Optional[Dict[int, float]] = None,
        residual_pool: Optional[torch.Tensor] = None,
        n_iter: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Generative forward pass: predict X by iterating the model to a fixed
        point instead of teacher-forcing it with observed X values.

        One forward pass computes ``f_i(pa(i))`` with the CURRENT X values as
        keys/values; the predictions become the X values of the next round::

            X^(k)_i = f_i(S, X^(k-1)) + 1[i not in D] * e_i      (i = 1..L_X)
            X^(k)_j = d_j                                          (j in D)

        where ``D`` is the clamped do-set (``clamp``) and ``e`` is a residual
        vector drawn ONCE per batch row from ``residual_pool`` and re-added
        every round (variant B of docs/documentation/ATE_INTERVENTIONAL_ROLLOUT.md;
        ``residual_pool=None`` gives the deterministic variant A).

        For an acyclic learned graph each round propagates final values one
        topological layer further, so the iterate stops EXACTLY after at most
        ``L_X`` rounds; a non-zero final ``rollout_delta`` flags a cyclic
        learned graph, whose interventional semantics are undefined.

        Clamping a slot every round IS the graph mutilation: the clamped node
        no longer depends on its parents, while downstream nodes keep seeing
        its intervened value as key/value.

        Args:
            data_source:   S tensor (B, L_S, F), normalized; already intervened
                           by the caller for S-side interventions.
            x_init:        Initial X state (B, L_X, F); its value column is
                           overwritten by the iteration, its index column is
                           preserved (the model needs the variable indices).
            clamp:         {x_position: normalized_value} applied every round.
                           ``None`` or ``{}`` = observational (generative) run.
            residual_pool: (N_pool, L_X) normalized residuals
                           ``x_i - f_i(pa(i))`` collected teacher-forced on
                           held-out data.  One row-index vector is drawn per
                           call (from ``generator``), so the SAME noise is
                           re-added each round and the fixed point converges.
            n_iter:        Max rounds; default ``L_X`` (sufficient for any DAG).
            generator:     torch.Generator for the residual draws.  Pass
                           identically-seeded generators across treated and
                           baseline runs for common random numbers.

        Returns:
            x_final:       (B, L_X, F) converged X state (value + index cols).
            n_iter_used:   Rounds actually run (early stop on convergence).
            rollout_delta: max |X^(K) - X^(K-1)| over the value column; ~0 for
                           an acyclic learned graph.
        """
        clamp = clamp or {}
        B, L_X, F = x_init.shape
        if n_iter is None:
            n_iter = L_X

        device = x_init.device
        val = self.val_idx

        # Draw one residual vector per batch row (constant across rounds).
        e = None
        if residual_pool is not None:
            pool = residual_pool.to(device=device, dtype=x_init.dtype)
            idx = torch.randint(pool.shape[0], (B,), generator=generator, device="cpu")
            e = pool[idx.to(pool.device)].to(device)          # (B, L_X)
            if clamp:
                # Clamped nodes are mutilated: no noise on the do-set.
                keep = torch.ones(L_X, device=device, dtype=e.dtype)
                for pos in clamp:
                    keep[pos] = 0.0
                e = e * keep.unsqueeze(0)

        x = x_init.clone()
        delta = float("inf")
        rounds = 0
        for k in range(n_iter):
            pred = self.forward(data_source, x)[0]            # (B, L_X, 1) or (B, N, 1)
            if self.homogeneous_nodes:
                pred = pred[:, self.S_seq_len :, :]           # keep the X rows
            x_new = x.clone()
            x_new[:, :, val] = pred.squeeze(-1)
            if e is not None:
                x_new[:, :, val] = x_new[:, :, val] + e
            for pos, value in clamp.items():
                x_new[:, pos, val] = float(value)
            delta = float((x_new[:, :, val] - x[:, :, val]).abs().max())
            x = x_new
            rounds = k + 1
            if delta == 0.0:
                break

        return x, rounds, delta

    # ------------------------------------------------------------------
    # Common step
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str = "train"):
        # Unpack Ã¢â‚¬â€ support (S, X) and (S, X, Y)
        S = batch[0]
        X = batch[1]

        x_val = X[:, :, self.val_idx]           # (B, L_X)  ground truth values

        # Homogeneous mode: the model reconstructs ALL N nodes, so the target
        # is cat([S_values, X_values]) Ã¢â€ â€™ (B, N).  Everything downstream that
        # consumes ``x_target`` / ``pred_x`` (MSE, metrics, HSIC residuals,
        # ANM diagnostics) then operates on N rows automatically.
        if self.homogeneous_nodes:
            x_val = torch.cat([S[:, :, self.val_idx], x_val], dim=1)   # (B, N)

        # Forward Ã¢â‚¬â€ returns (pred_x, attention_weights, aux_dict)
        pred_x, attention_weights, aux = self.forward(S, X)

        # Soft adjacency for structure diagnostics (PeriodicDAGMetrics).
        # THE ONLY valid source on this architecture: phi/dag_mask are
        # deprecated and ``batch_att_mean`` is never assigned any more, so the
        # attention posterior produced by THIS forward pass -- driven by the
        # structural embeddings -- IS the learned graph.  Detached and stored
        # on the module; never touched by the loss.
        if attention_weights is not None and attention_weights.dim() == 3:
            self._last_att_mean = attention_weights.detach().mean(dim=0)
        entropy    = aux.get("entropy")    if isinstance(aux, dict) else aux
        l0_penalty = aux.get("l0_penalty") if isinstance(aux, dict) else None

        # Reconstruction loss
        x_target = torch.nan_to_num(x_val)
        mse_per_elem = self.loss_fn(pred_x.squeeze(), x_target.squeeze())
        loss_x = mse_per_elem.mean()

        # ----------------------------------------------------------------
        # Score sparsity (L1 on raw attention weights)
        # CausalCrossAttention exposes score_tensor_for_sparsity = attention matrix
        # ----------------------------------------------------------------
        # Unified score tensor for the sparsity / NOTEARS terms.  In split mode
        # (self_attention_type set) this concatenates the SÃ¢â€ â€™X cross gate
        # posterior with the direction-aware XÃ¢â€ â€™X GatedSelfAttention posterior,
        # so the (L_X, L_S+L_X) layout is identical to single mode.  Falls back
        # to the legacy inner-attention attribute for older checkpoints/models.
        get_score = getattr(self.model, "get_score_tensor_for_sparsity", None)
        if callable(get_score):
            score_tensor = get_score()
        else:
            inner_att = self.model.attention.inner_attention
            score_tensor = getattr(inner_att, "score_tensor_for_sparsity", None)

        if score_tensor is not None:
            score_sparse_value = score_tensor.abs().mean()
        else:
            # Fallback: entropy of attention weights
            if entropy is not None:
                score_sparse_value = entropy.mean()
            else:
                score_sparse_value = torch.tensor(0.0, device=X.device)

        score_sparsity_reg = self.lambda_score_sparse * score_sparse_value

        # ----------------------------------------------------------------
        # HSIC regularization
        # Unified over combined source = [S_values, X_values]
        # HSIC(source_j, res_i) for all (i, j) pairs
        # ----------------------------------------------------------------
        # CROSS-FITTING: on train steps, recompute the residual on a DISJOINT
        # fold (a second forward pass) so the independence statistic is measured
        # on samples this step's reconstruction gradient did not fit.  Falls
        # back silently to the in-batch residual when disabled or unavailable
        # (e.g. validation, or the fold loader was never installed).
        xfit = self._next_cross_fit_batch() if stage == "train" else None
        if xfit is not None:
            S_h, X_h = xfit[0], xfit[1]
            x_val_h = X_h[:, :, self.val_idx]
            if self.homogeneous_nodes:
                x_val_h = torch.cat([S_h[:, :, self.val_idx], x_val_h], dim=1)
            pred_h = self.forward(S_h, X_h)[0]
            target_h = torch.nan_to_num(x_val_h)
            residuals = target_h.squeeze() - pred_h.squeeze()
            if self.homogeneous_nodes:
                combined_source = target_h.squeeze()
            else:
                combined_source = torch.cat(
                    [S_h[:, :, self.val_idx], target_h.squeeze()], dim=1
                )
        else:
            residuals = x_target.squeeze() - pred_x.squeeze()    # (B, L_X)

            # Candidate-parent values must be paired with the residual of every
            # child row.  In homogeneous mode ``x_target`` ALREADY is
            # [S_values | X_values] (all N nodes), so it is the candidate set
            # itself; in split mode the S values must be prepended.
            if self.homogeneous_nodes:
                combined_source = x_target.squeeze()                    # (B, N)
            else:
                s_values = S[:, :, self.val_idx]          # (B, L_S)
                x_values = x_target.squeeze()             # (B, L_X)
                # Concatenate all potential parent values:
                # [S_1,...,S_{L_S}, X_1,...,X_{L_X}]
                combined_source = torch.cat([s_values, x_values], dim=1)  # (B, L_S+L_X)

        # --- Per-phase bandwidth freeze (lazy latch on the first train batch) --
        if stage == "train" and self.hsic_freeze_bandwidth:
            if self._hsic_bw_frozen_sigmas is None:
                self.freeze_hsic_bandwidth(combined_source, residuals)

        # --- Descendant exclusion -------------------------------------------
        # Under an ANM the residual r_i = e_i is independent of every
        # NON-descendant of i, but NECESSARILY dependent on its descendants
        # (and on X_i itself).  Penalising those pairs biases the objective
        # away from the true DAG and rewards attending to descendants, so we
        # drop them from the average.  The mask is derived from the learned
        # adjacency and is always DETACHED (a differentiable mask would let the
        # model create an edge purely to delete its own penalty term).
        hsic_pair_mask, hsic_desc_kept_frac, hsic_desc_cyclic = self._build_hsic_descendant_mask(
            score_tensor
        )
        # Bilevel machinery: the probe/virtual-pass pair weights are the
        # STRUCTURAL mask only (descendant x LOO) Ã¢â‚¬â€ BKD excluded, since probes
        # run with key dropout off.  Captured before the BKD multiply below.
        _probe_mask_base = hsic_pair_mask

        # --- BKD dropped-key exclusion ------------------------------------
        # Keys dropped by batch-consistent key dropout were NOT in the
        # estimation set, so for a dropped true parent j the pair
        # HSIC(res_i, source_j) > 0 is irreducible within the batch (its
        # aggregation weight AND gradient are zeroed by the keep mask).
        # Excluding dropped pairs removes a p-dependent offset, batch
        # variance and a shortcut-biased gradient from the structural loss.
        # Validation runs without BKD, so ``val_hsic`` stays the
        # all-sources reference curve.
        bkd_keep_mask = self._build_bkd_keep_mask(
            n_targets=residuals.shape[-1],
            n_sources=combined_source.shape[-1],
            device=combined_source.device,
        )
        if bkd_keep_mask is not None:
            hsic_pair_mask = (
                bkd_keep_mask
                if hsic_pair_mask is None
                else hsic_pair_mask * bkd_keep_mask
            )
        self._last_hsic_bkd_kept_frac = (
            float(bkd_keep_mask[0].mean().item())
            if bkd_keep_mask is not None
            else 1.0
        )

        # --- LOO conditional-HSIC gate (counter-proposal Ã‚Â§5, option (c)) ------
        # Detached per-edge multiplier applied to the pair weights BEFORE the
        # HSIC average: conditionally-redundant pairs are downweighted out of
        # the marginal mean; load-bearing pairs keep full weight.  Same
        # (n_targets, n_sources) layout as the pair mask / attention matrix.
        loo_gamma = None
        if self.use_loo_gamma_gate:
            loo_gamma = self._maybe_update_loo_gamma(S, X, x_target, stage)
        if loo_gamma is not None:
            loo_gamma = loo_gamma.to(combined_source.device)
            if self.use_attention_weighted_hsic:
                pass  # applied to att_mean in the branch below
            elif hsic_pair_mask is not None:
                if hsic_pair_mask.shape == loo_gamma.shape:
                    hsic_pair_mask = hsic_pair_mask * loo_gamma
                else:
                    logger.warning(
                        "LOO gamma shape %s != pair-mask shape %s; skipping gate.",
                        tuple(loo_gamma.shape), tuple(hsic_pair_mask.shape),
                    )
                    loo_gamma = None
            else:
                hsic_pair_mask = loo_gamma

        # Stash the probe/virtual-pass pair weights (descendant x LOO, no BKD)
        # for the bilevel gate and the unrolled shadow.  Train stage only;
        # None when the attention-weighted HSIC variant is in use (the probe
        # then measures the plain per-pair rows, which the gate docs note).
        if stage == "train" and (
                self._gate_enabled or self._commit_source == "hsic_unrolled"):
            if self.use_attention_weighted_hsic:
                self._last_probe_pair_mask = None
            elif loo_gamma is not None and not self.use_attention_weighted_hsic:
                self._last_probe_pair_mask = (
                    loo_gamma.detach() if _probe_mask_base is None
                    else (_probe_mask_base * loo_gamma).detach()
                )
            else:
                self._last_probe_pair_mask = (
                    None if _probe_mask_base is None
                    else _probe_mask_base.detach()
                )

        pair_mask_empty = hsic_pair_mask is not None and not bool(
            (hsic_pair_mask != 0).any()
        )
        if pair_mask_empty:
            # Every pair excluded this batch (e.g. BKD dropped all keys):
            # no learnable HSIC signal - contribute an exact zero rather
            # than a NaN from the empty weighted mean.
            hsic_value = torch.zeros((), device=combined_source.device)
            self._last_hsic_row_means = None
        elif self.use_attention_weighted_hsic:
            # Attention-weighted HSIC: weight each (child, source) pair by the
            # batch-mean attention weight att[child, source].  The attention
            # matrix is (B, n_targets, n_sources) Ã¢â‚¬â€ (B, N, N) in homogeneous
            # mode, (B, L_X, L_S+L_X) in split mode Ã¢â‚¬â€ matching the HSIC pair
            # matrix layout exactly.  Descendant masking is NOT applied here:
            # the attention weight itself is the pair weight.
            att_mean = attention_weights.mean(dim=0)  # (n_targets, n_sources)
            if bkd_keep_mask is not None:
                # Dropped sources get zero pair weight (pair_mask is unused
                # by the attention-weighted variant).
                att_mean = att_mean * bkd_keep_mask
                if self.hsic_softmax:
                    # Softmax treats the scores as LOGITS: a dropped source
                    # must be -inf (weight exactly 0), not 0 (weight ~ 1/Z).
                    att_mean = att_mean.masked_fill(
                        bkd_keep_mask == 0, float("-inf")
                    )
            if loo_gamma is not None:
                att_mean = att_mean * loo_gamma

            if self.hsic_softmax:
                # SOFTMAX COMPETITION: row-wise softmax pair weights, self-edge
                # excluded inside the aggregation, NO descendant mask.
                hsic_out = hsic_attention_softmax(
                    source_values=combined_source,
                    residuals=residuals,
                    attention_weights=att_mean,
                    sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon,
                    source_kernel=self.hsic_kernel_source,
                    bandwidth_multipliers=self.hsic_bandwidth_multipliers,
                    return_matrix=True,   # pair matrix is computed anyway;
                    # stashed detached for the hsic_class/* diagnostics
                    # (log_hsic_rows below only gates the row-means logging)
                    # Split mode: target i's own value sits at column
                    # i + S_seq_len of the combined [S ; X] source matrix.
                    diagonal_offset=(
                        0 if self.homogeneous_nodes else self.S_seq_len
                    ),
                    pair_weight_mode=self.hsic_pair_weight_mode,
                    tilt_tau=self.hsic_tilt_tau,
                    return_weights=True,
                )
                hsic_value, hsic_mat, pair_w = hsic_out
                self._last_hsic_pair_mat = (
                    hsic_mat.detach() if hsic_mat is not None else None
                )
                # Entropy of the ACTUAL pair weights used by the aggregation
                # (detached): H ~ ln(K) = near-uniform competition (legacy
                # [0,1]-logit mode is pinned there by construction); falling
                # H / eff_competitors -> true in-degree is the signature of
                # the structure competition resolving.
                _ent = row_entropy_stats(pair_w)
                self.log(f"{stage}_hsic_att_entropy_mean", _ent["mean"],
                         on_step=False, on_epoch=True)
                self.log(f"{stage}_hsic_att_entropy_min", _ent["min"],
                         on_step=False, on_epoch=True)
                self.log(f"{stage}_hsic_att_entropy_max", _ent["max"],
                         on_step=False, on_epoch=True)
                self.log(f"{stage}_hsic_att_entropy_norm", _ent["norm_mean"],
                         on_step=False, on_epoch=True)
                self.log(f"{stage}_hsic_att_eff_competitors",
                         _ent["eff_competitors_mean"],
                         on_step=False, on_epoch=True)
                # Pre-normalisation row mass: 1.0 for row-stochastic modes;
                # under evidence_max (leader = 1) it tracks how much
                # subordinate weight survives - the dilution gauge.
                self.log(f"{stage}_hsic_att_row_mass", _ent["row_mass_mean"],
                         on_step=False, on_epoch=True)
                if self.log_hsic_rows:
                    # Node-responsible rows with the SAME pair weights as the
                    # scalar aggregation -- the (detached) pair weights, so
                    # the rows sum-decompose the logged HSIC.
                    self._last_hsic_row_means = hsic_row_means(
                        hsic_mat.detach(),
                        pair_mask=pair_w,
                    )
                else:
                    self._last_hsic_row_means = None
            else:
                # HYBRID aggregation: weight ONLY descendant pairs.  The mask is
                # derived from the DIRECTED score tensor (the asymmetric/skew term
                # that carries the orientation), never from the gated posterior, and
                # is always detached -- see _build_descendant_weight_mask.
                desc_w_mask = None
                if self.hsic_weight_descendants_only:
                    desc_w_mask = self._build_descendant_weight_mask(score_tensor)

                hsic_out = hsic_attention_weighted(
                    source_values=combined_source,
                    residuals=residuals,
                    attention_weights=att_mean,
                    sigma=self.hsic_sigma,
                    # HYBRID mode MUST drop the diagonal: HSIC(X_i, r_i) is
                    # irreducible and is NOT a descendant pair, so it would enter
                    # unweighted as a large constant and drown the signal.
                    exclude_diagonal=self.hsic_weight_descendants_only,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon,
                    source_kernel=self.hsic_kernel_source,
                    bandwidth_multipliers=self.hsic_bandwidth_multipliers,
                    return_matrix=True,   # pair matrix is computed anyway;
                    # stashed detached for the hsic_class/* diagnostics
                    # (log_hsic_rows below only gates the row-means logging)
                    descendant_mask=desc_w_mask,
                )
                hsic_value, hsic_mat = hsic_out
                self._last_hsic_pair_mat = hsic_mat.detach()
                if self.log_hsic_rows:
                    # Node-responsible rows with the SAME pair weights as the
                    # scalar aggregation -- here the (detached) attention posterior
                    # itself, so the rows sum-decompose the logged HSIC.  Detached:
                    # logging only, never part of the loss.
                    self._last_hsic_row_means = hsic_row_means(
                        hsic_mat.detach(), pair_mask=att_mean.detach()
                    )
                else:
                    self._last_hsic_row_means = None
        else:
            hsic_out = hsic_cross_per_pair(
                combined_source,
                residuals,
                sigma=self.hsic_sigma,
                adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                mode=self.hsic_mode,
                nhsic_epsilon=self.nhsic_epsilon,
                source_kernel=self.hsic_kernel_source,
                bandwidth_multipliers=self.hsic_bandwidth_multipliers,
                pair_mask=hsic_pair_mask,
                return_matrix=self.log_hsic_rows,
            )
            if self.log_hsic_rows:
                hsic_value, hsic_mat = hsic_out
                # Node-responsible rows: mean_j HSIC(source_j, res_i) with the
                # same pair weights as the scalar aggregation.  Detached:
                # logging only, never part of the loss.
                self._last_hsic_row_means = hsic_row_means(
                    hsic_mat.detach(), pair_mask=hsic_pair_mask
                )
            else:
                hsic_value = hsic_out
        # Constraint source switch: the monitored quantity is the HSIC
        # (default) or, with source='oracle_shd', the EXPECTED SHD to the GT
        # DAG under the directed gate posterior (mean per-pair
        # misclassification; differentiable through the same Q/K structural
        # pathway as L0).  HSIC is still computed and logged either way --
        # under oracle_shd it is a pure EVALUATION metric.
        constraint_value = hsic_value
        if (
            self.hsic_constraint_enabled
            and self.hsic_constraint_source == "oracle_shd"
        ):
            post = attention_weights.mean(dim=0)
            gt_buf = getattr(self, "oracle_shd_gt", None)
            if gt_buf is None:
                raise ValueError(
                    "hsic_constraint.source='oracle_shd': GT buffer not "
                    "loaded (data_dir was None at init)."
                )
            gt = gt_buf.to(dtype=post.dtype)
            if post.shape != gt.shape:
                raise ValueError(
                    f"oracle_shd: posterior {tuple(post.shape)} != GT "
                    f"{tuple(gt.shape)}"
                )
            constraint_value = (gt * (1.0 - post) + (1.0 - gt) * post).mean()
            self.log(f"{stage}_oracle_shd", constraint_value,
                     on_step=False, on_epoch=True)
        if self.hsic_constraint_enabled:
            # Lagrangian / augmented-Lagrangian constraint term:
            #   lam * (HSIC - eps) + (rho/2) * relu(HSIC - eps)^2
            # d/dHSIC = lam + rho * relu(HSIC - eps): the dual lam supplies
            # pressure that does NOT vanish as HSIC -> eps (unlike the fixed
            # lambda_hsic penalty, whose gradient dies with the signal).
            hsic_violation = constraint_value - self.hsic_tol
            hsic_reg = (
                self._hsic_dual_lambda * hsic_violation
                + 0.5 * self._hsic_rho * torch.clamp(hsic_violation, min=0.0) ** 2
            )
            # EMA of the RAW HSIC (train batches only, detached) drives the
            # per-epoch dual ascent in on_train_epoch_end.  Mirrors the
            # _hsic_safeguard_ref EMA pattern.
            if stage == "train" and self._hsic_dual_active:
                v = float(constraint_value.detach())
                if self._hsic_constraint_ema is None:
                    self._hsic_constraint_ema = v
                else:
                    d = self.hsic_constraint_ema_decay
                    self._hsic_constraint_ema = (
                        d * self._hsic_constraint_ema + (1.0 - d) * v
                    )
        else:
            hsic_reg = self.lambda_hsic * hsic_value

        # --- Per-rung tolerance calibration (permutation null) -------------
        # Armed by hsic_constraint_on_phase_switch(phase=structure); consumes
        # the first ``calibration_batches`` train batches of the structure
        # phase (same fold, same rung, same residuals the constraint sees).
        if (
            self.hsic_constraint_enabled
            and stage == "train"
            and self._hsic_null_calib_remaining > 0
        ):
            self._collect_hsic_null_sample(
                combined_source=combined_source,
                residuals=residuals,
                attention_weights=attention_weights,
                bkd_keep_mask=bkd_keep_mask,
                hsic_pair_mask=hsic_pair_mask,
            )

        if hsic_pair_mask is not None:
            self._last_hsic_desc_kept_frac = hsic_desc_kept_frac
            self._last_hsic_desc_cyclic = hsic_desc_cyclic
        else:
            self._last_hsic_desc_kept_frac = 1.0
            self._last_hsic_desc_cyclic = False

        # Safeguard reference: EMA of the weighted HSIC term (detached),
        # updated on train batches only.  None when both caps are disabled.
        hsic_ref = self._hsic_safeguard_ref(hsic_reg, stage)

        # ----------------------------------------------------------------
        # Group-L1 regularization (L2,1 norm on embedding columns)
        # ----------------------------------------------------------------
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_reg = self.lambda_group_l1 * group_l1_loss

        # ----------------------------------------------------------------
        # Acyclicity regularization (NOTEARS) Ã¢â‚¬â€ XÃ¢â€ â€™X sub-block only
        # Extract the square (L_X, L_X) directed edge matrix from the
        # combined (L_X, L_S+L_X) score tensor by slicing columns S_seq_len:.
        # The score tensor is 2-D (batch-mean, head-averaged) for single-head
        # CausalCrossAttention.  Multi-head tensors (dim != 2) are skipped.
        # ----------------------------------------------------------------
        # In homogeneous mode the score tensor IS the square (N, N) directed
        # adjacency over all nodes, so NOTEARS applies to the FULL matrix (the
        # column slice would be a meaningless sub-block there).
        acy_constrained = self.acyclicity_constraint_enabled
        if ((self.kappa > 0.0 or acy_constrained)
                and score_tensor is not None and score_tensor.dim() == 2):
            A_cyc = (
                score_tensor                      # (N, N)
                if self.homogeneous_nodes
                else score_tensor[:, self.S_seq_len:]   # (L_X, L_X)
            )
            notears_raw = self._acyclicity_penalty(A_cyc)
            if acy_constrained:
                # Augmented Lagrangian: alpha * h + (rho/2) * h^2 (canonical
                # NOTEARS protocol).  h >= 0 for every backend, so the
                # equality constraint h = 0 needs no relu.
                kappa_eff = self.kappa   # 0.0; kept for logging continuity
                acyclic_reg = (
                    self._acy_dual_lambda * notears_raw
                    + 0.5 * self._acy_rho * notears_raw ** 2
                )
                # EMA of the RAW h (train batches only, detached) drives the
                # per-epoch dual ascent in on_train_epoch_end.
                if stage == "train":
                    v = float(notears_raw.detach())
                    if self._acy_ema is None:
                        self._acy_ema = v
                    else:
                        dcy = self.acy_ema_decay
                        self._acy_ema = dcy * self._acy_ema + (1.0 - dcy) * v
            else:
                kappa_eff = self._cap_reg_coeff(
                    self.kappa, notears_raw, self.kappa_max_hsic_pct, hsic_ref
                )
                acyclic_reg = kappa_eff * notears_raw
        else:
            kappa_eff = self.kappa
            acyclic_reg = torch.tensor(0.0, device=X.device)

        # MSE-vs-acyclicity cap: throttle the reconstruction weight so the
        # weighted MSE cannot exceed mse_max_acyclic_pct * (EMA of the
        # weighted acyclic term).  Released (lambda_recon restored) once the
        # acyclic EMA falls at/below mse_cap_release_tol -- when the graph is
        # (near-)acyclic there is no conflict left to arbitrate.
        acy_ref = self._acyclic_safeguard_ref(acyclic_reg, stage)
        if acy_ref is not None and acy_ref <= self.mse_cap_release_tol:
            acy_ref = None  # release: acyclicity (near-)satisfied
        lambda_recon_eff = self._cap_reg_coeff(
            self.lambda_recon, loss_x, self.mse_max_acyclic_pct, acy_ref
        )


        # ----------------------------------------------------------------
        # L0 regularization (non-zero only for HardConcreteCrossAttention)
        # l0_penalty is the expected number of active edges = sum P(z_ij > 0)
        # ----------------------------------------------------------------
        # NOTE: the weighted term ``l0_reg`` (which enters the loss) is gated by
        # ``lambda_l0`` so a zero strength contributes nothing to the gradient.
        # However ``l0_penalty`` (the *measured* expected active-gate count) must
        # be logged UNCONDITIONALLY: at ``lambda_l0 == 0`` the gate is fully dense,
        # so the true penalty is at its MAXIMUM (~n_edges), not zero. Overwriting
        # it with 0.0 (as the old code did) produced a misleading sparsity
        # dose-response where the no-L0 baseline appeared perfectly sparse.
        if l0_penalty is None:
            # Non-HardConcrete attentions do not expose an L0 penalty at all.
            l0_penalty = torch.tensor(0.0, device=X.device)
        if self.lambda_l0 > 0.0:
            if self.l0_max_acyclic_pct > 0.0:
                # Acyclicity-anchored cap (HSIC-free arms): the L0 pressure
                # fades as h -> 0, so the gates freeze on an acyclic graph
                # instead of collapsing to the empty one.
                lambda_l0_eff = self._cap_reg_coeff(
                    self.lambda_l0, l0_penalty, self.l0_max_acyclic_pct, acy_ref
                )
            else:
                lambda_l0_eff = self._cap_reg_coeff(
                    self.lambda_l0, l0_penalty, self.lambda_l0_max_hsic_pct, hsic_ref
                )
            l0_reg = lambda_l0_eff * l0_penalty
        else:
            lambda_l0_eff = self.lambda_l0
            l0_reg = torch.tensor(0.0, device=X.device)

        # ----------------------------------------------------------------
        # Total loss
        # ----------------------------------------------------------------
        total_loss = (
            lambda_recon_eff * loss_x
            + score_sparsity_reg
            + hsic_reg
            + group_l1_reg
            + acyclic_reg
            + l0_reg
        )

        # Store for gradient routing.
        # NOTEARS rides on the structural pathway (same as HSIC): its gradient
        # flows through the Q/K score matrix back to Q/K projections and
        # structural embeddings, leaving V/FFN/MLP untouched.
        # L0 also rides on the structural pathway: P(z>0) = sigmoid(log_alpha - offset)
        # and log_alpha = QK^T/sqrt(E), so gradients flow through Q/K.
        #
        # Convex mix on the HSIC/reconstruction split of the structural pathway
        # (mirrors SingleCausalForecaster):
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg + acyclic_reg + l0_reg
        # alpha = lambda_struct_recon.  At alpha=0 this is identical to the
        # original pure-HSIC structural stream.  The alpha * loss_x term reuses
        # the already-computed reconstruction loss and the retained autograd
        # graph, so its gradient flows to the STRUCTURAL params through the
        # attention weights with no extra forward/backward pass.  The
        # reconstruction params keep their pure-recon gradients via the
        # save/restore logic in training_step, so theta_R is unaffected.
        # Query-norm over-spend penalty (structural pathway only): each child's
        # learnable budget M_i is charged relu(M_i - target)^2, summed over
        # nodes (deduped across tied cross/self blocks).
        qn_penalty = collect_query_norm_penalty(self.model)
        if qn_penalty is None:
            qn_penalty = torch.tensor(0.0, device=X.device)
        qn_reg = self.lambda_query_norm * qn_penalty

        # Without gradient routing there IS no structural loss stream: the
        # plain ``training_step`` branch back-propagates ``total_loss`` alone,
        # so a penalty that only rides on ``loss_structural`` would be logged
        # but contribute exactly zero gradient (the learnable query-norm budget
        # would then be free to grow without bound).  Add it to ``total_loss``
        # on that path only -- routed runs keep their byte-identical behaviour
        # because they never back-propagate ``total_loss``.
        if not self.use_gradient_routing:
            total_loss = total_loss + qn_reg

        alpha = self.lambda_struct_recon
        struct_recon_reg = alpha * loss_x
        self._last_loss_components = {
            "loss_recon": loss_x,
            "loss_structural": (
                (1.0 - alpha) * hsic_reg
                + struct_recon_reg
                + score_sparsity_reg + group_l1_reg + acyclic_reg + l0_reg
                + qn_reg
            ),
        }

        # Keep references to the individual reg terms (graph still attached) so
        # training_step can probe L0 Ã¢â€ â€ HSIC gradient interference before the
        # real backward runs.
        self._last_hsic_reg = hsic_reg
        self._last_l0_reg = l0_reg
        # Separate terms for PCGrad gradient surgery (gradient-routing path):
        # HSIC enters loss_structural as (1 - alpha) * hsic_reg; the L0 and
        # NOTEARS terms are projected against it per block, and everything
        # else (struct-recon mix, score sparsity, group L1, query norm) is
        # bundled as the untouched "rest" term.
        self._last_acyclic_reg = acyclic_reg
        self._last_struct_hsic_term = (1.0 - alpha) * hsic_reg
        self._last_struct_rest = (
            struct_recon_reg + score_sparsity_reg + group_l1_reg + qn_reg
        )

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"))
        self.log(f"{stage}_score_sparse", score_sparse_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_reg, on_step=False, on_epoch=True)
        if self.hsic_weight_descendants_only:
            # Fraction of pairs currently classed as descendants, i.e. the only
            # pairs whose attention receives gradient.  Watch for drift: -> 0
            # means the hybrid has degenerated to the unweighted objective,
            # -> 1 means it has degenerated to plain attention weighting.
            self.log(f"{stage}_desc_weight_frac",
                     float(self._last_desc_weight_frac),
                     on_step=False, on_epoch=True)
        # Per-row (node-responsible) HSIC: mean over sources per target node.
        # NaN rows (fully excluded this batch) are skipped.
        if self.log_hsic_rows and self._last_hsic_row_means is not None:
            for _i, _v in enumerate(self._last_hsic_row_means):
                if not torch.isnan(_v):
                    self.log(f"{stage}_hsic_row_{_i}", _v,
                             on_step=False, on_epoch=True)
        # Descendant-exclusion diagnostics.  NOTE: with masking active the
        # ``{stage}_hsic`` value above is a MASKED mean, so its normalisation
        # set drifts as the learned graph changes and it is NOT comparable
        # across epochs.  ``kept_frac`` tells you how much of the pair set is
        # still contributing (a collapse toward 0 kills the structural signal),
        # and ``cyclic`` flags a thresholded graph containing a cycle, which
        # inflates the descendant closure.
        if self.hsic_exclude_descendants:
            self.log(
                f"{stage}_hsic_desc_kept_frac",
                float(self._last_hsic_desc_kept_frac),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{stage}_hsic_desc_cyclic",
                float(self._last_hsic_desc_cyclic),
                on_step=False, on_epoch=True,
            )
        if self.use_loo_gamma_gate:
            self.log(
                f"{stage}_loo_gamma_mean",
                float(self._last_loo_gamma_mean),
                on_step=False, on_epoch=True,
            )
            self.log(
                f"{stage}_loo_gamma_min",
                float(self._last_loo_gamma_min),
                on_step=False, on_epoch=True,
            )
        if self.hsic_bkd_exclude_dropped:
            self.log(
                f"{stage}_hsic_bkd_kept_frac",
                float(self._last_hsic_bkd_kept_frac),
                on_step=False, on_epoch=True,
            )
        # Structural-pathway reconstruction term (alpha * loss_x).  Non-zero
        # only when lambda_struct_recon > 0; lets eval/monitoring see how much
        # reconstruction signal is shaping the structural parameters.
        self.log(f"{stage}_struct_recon_reg", struct_recon_reg, on_step=False, on_epoch=True)
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)
        # Query-norm diagnostics: weighted penalty + mean / max budget M_i.
        self.log(f"{stage}_query_norm_reg", qn_reg, on_step=False, on_epoch=True)
        mean_M, max_M = query_norm_stats(self.model)
        if mean_M is not None:
            self.log("query_norm/mean_M", mean_M, on_step=False, on_epoch=True)
            self.log("query_norm/max_M", max_M, on_step=False, on_epoch=True)


        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred_x.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val" and name == "mae"))

        self._log_r2_variants(pred_x, x_target, stage)


        if effective_dims is not None:
            self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)

        # NOTEARS acyclicity (auto-discovered by eval_training.py via "notears" key)
        self.log(f"{stage}_notears", acyclic_reg, on_step=False, on_epoch=True)
        # Safeguard diagnostics: the EFFECTIVE coefficients applied this step
        # (< base value => the HSIC-relative cap is active).
        if self.kappa_max_hsic_pct > 0.0:
            self.log(f"{stage}_kappa_eff", float(kappa_eff), on_step=False, on_epoch=True)
        if self.mse_max_acyclic_pct > 0.0:
            self.log(f"{stage}_lambda_recon_eff", float(lambda_recon_eff), on_step=False, on_epoch=True)



        # L0 penalty (expected number of active edges, non-zero only for
        # HardConcreteCrossAttention; logged as 0.0 for all other attention types)
        self.log(f"{stage}_l0_penalty", l0_penalty, on_step=False, on_epoch=True)
        self.log(f"{stage}_l0_reg", l0_reg, on_step=False, on_epoch=True)
        if self.lambda_l0_max_hsic_pct > 0.0 or self.l0_max_acyclic_pct > 0.0:
            self.log(f"{stage}_lambda_l0_eff", float(lambda_l0_eff), on_step=False, on_epoch=True)

        if stage == "val":
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss, pred_x, X

    # ------------------------------------------------------------------
    # Interpretable fit metrics
    # ------------------------------------------------------------------

    def _log_r2_variants(
        self,
        pred_x: torch.Tensor,
        x_target: torch.Tensor,
        stage: str,
    ) -> None:
        """Log ``x_r2_endo`` / ``x_r2_macro`` / ``x_r2_src`` next to ``x_r2``.

        ``x_r2`` (logged by the caller) is pooled over EVERY node and sample, so
        in homogeneous mode it averages the endogenous rows with the exogenous S
        rows, whose causally-correct R2 is ~0 -- the metric then has a ceiling
        far below 1 and cannot be read as a fit quality.  Pooling also mixes
        between-node variance into SStot, so the value depends on the relative
        variances of the variables.  Hence:

        * ``x_r2_endo``  pooled R2 on the endogenous rows only (identical to
          ``x_r2`` in split mode, where the target has no S rows);
        * ``x_r2_macro`` mean of the PER-NODE R2 over the endogenous rows -- each
          node normalised by its own variance, so pooling-free and invariant to
          per-node rescaling.  This is the number to read as "fit quality";
        * ``x_r2_src``   pooled R2 on the S rows (homogeneous mode only).  A
          source can only be predicted from its own descendants, so a HIGH value
          is a positive diagnostic of ANTI-CAUSAL use of the posterior.

        ``x_r2`` itself is never redefined: it appears in every historical
        ``metrics.csv`` and in the sweep/Optuna plumbing.
        """
        # (B, n_rows) view of predictions and targets.
        pred = pred_x.reshape(x_target.shape[0], -1)
        targ = x_target.reshape(x_target.shape[0], -1)

        if self.homogeneous_nodes:
            pred_endo, targ_endo = pred[:, self.S_seq_len:], targ[:, self.S_seq_len:]
            pred_src, targ_src = pred[:, : self.S_seq_len], targ[:, : self.S_seq_len]
        else:
            pred_endo, targ_endo = pred, targ
            pred_src, targ_src = None, None

        self.log(
            f"{stage}_x_r2_endo",
            self.r2_x_endo(pred_endo.reshape(-1), targ_endo.reshape(-1)),
            on_step=False, on_epoch=True,
        )
        # R2Score(num_outputs>1) needs a (B, n_outputs) 2-D input and at least
        # two samples per output; a degenerate batch would raise, so guard it.
        if pred_endo.shape[0] > 1 and pred_endo.shape[1] == self.X_seq_len:
            self.log(
                f"{stage}_x_r2_macro",
                self.r2_x_macro(pred_endo, targ_endo),
                on_step=False, on_epoch=True,
            )
        if self.r2_x_src is not None and pred_src is not None and targ_src is not None:
            self.log(
                f"{stage}_x_r2_src",
                self.r2_x_src(pred_src.reshape(-1), targ_src.reshape(-1)),
                on_step=False, on_epoch=True,
            )


    # ------------------------------------------------------------------
    # L0 Ã¢â€ â€ HSIC gradient-interference diagnostic
    # ------------------------------------------------------------------


    # Attention types that expose a differentiable L0 penalty on the structure
    # gate (aux["l0_penalty"]), for which the L0 Ã¢â€ â€ HSIC interference probe is
    # meaningful.  Both drive the Hard-Concrete gate logit off the structural
    # query/key pair, so their gradients flow to the same structural params.
    _INTERFERENCE_ATTENTION_TYPES = (
        "HardConcreteCrossAttention",
        "GatedCrossAttention",
        # Homogeneous mode: the single square structural block IS the gated
        # self-attention, and it exposes aux["l0_penalty"] (undirected
        # skeleton edge count over the strictly-upper triangle).
        "GatedSelfAttention",
    )

    def _interference_enabled(self) -> bool:
        """Whether the L0 Ã¢â€ â€ HSIC interference diagnostic should run."""
        return (
            self.log_l0_hsic_interference
            and self._attention_type in self._INTERFERENCE_ATTENTION_TYPES
            and float(self.lambda_l0) > 0.0
            and float(self.lambda_hsic) > 0.0
        )


    def _maybe_log_interference(self, batch_idx: int):
        """Log per-block cosine similarity between the L0 and HSIC gradients.

        Guarded so it runs only for the first batch of an epoch, on the
        configured epoch cadence, and only when the diagnostic is enabled.

        Uses ``torch.autograd.grad(..., retain_graph=True)`` (inside
        :func:`compute_l0_hsic_interference`), which returns the gradients as
        tensors WITHOUT writing to ``.grad``.  Consequently the subsequent
        real backward (automatic optimisation) or the gradient-routing dual
        backward is left completely unaffected.
        """
        if not self._interference_enabled():
            return
        if batch_idx != 0:
            return
        every = max(1, int(self.interference_log_every_n_epochs))
        if (self.current_epoch % every) != 0:
            return
        if self._last_hsic_reg is None or self._last_l0_reg is None:
            return

        # Build the block Ã¢â€ â€™ parameter mapping lazily.  Rebuilt if it came back
        # empty last time (e.g. structural params were frozen for this stage).
        if not self._interference_blocks:
            self._interference_blocks = build_interference_blocks(self.model)
        blocks = self._interference_blocks
        if not blocks:
            return

        try:
            cos_by_block = compute_l0_hsic_interference(
                model=self.model,
                hsic_reg=self._last_hsic_reg,
                l0_reg=self._last_l0_reg,
                blocks=blocks,
            )
        except RuntimeError as exc:
            # Autograd may fail if the graph was already freed (e.g. a prior
            # backward without retain_graph).  Never let the diagnostic break
            # training Ã¢â‚¬â€ just skip this step.
            logger.warning(
                "L0Ã¢â€ â€HSIC interference probe skipped (autograd error): %s", exc
            )
            return

        # Skip NaN blocks: a NaN cosine means one objective's gradient is
        # entirely zero in that block (pure reconstruction blocks receive no
        # L0 gradient).  Logging only the non-NaN blocks auto-focuses the
        # metric set on the structural pathway (Q/K + embeddings) where the
        # L0 Ã¢â€ â€ HSIC interference actually happens.  We simultaneously collect
        # per-block cosines into a summary so the conflict is human-readable in
        # the console / log file (not just as scattered CSV columns).
        overall_cos = float("nan")
        block_cos: Dict[str, float] = {}
        for block_name, cos in cos_by_block.items():
            if math.isnan(cos):
                continue
            self.log(
                f"train_interf_cos_{block_name}",
                float(cos),
                on_step=False,
                on_epoch=True,
            )
            if block_name == "overall":
                overall_cos = float(cos)
            else:
                block_cos[block_name] = float(cos)

        # --- Human-readable summary of the L0 Ã¢â€ â€ HSIC gradient conflict ---
        # cos < 0 Ã¢â€¡â€™ the L0 (sparsity) and HSIC (independence) gradients push
        # the shared structural parameters in opposing directions in that
        # block, i.e. the two objectives are in direct conflict there.
        if block_cos:
            conflicting = {b: c for b, c in block_cos.items() if c < 0.0}
            n_blocks = len(block_cos)
            n_conflict = len(conflicting)
            # Sort blocks from most-conflicting (most negative) to most-aligned
            worst = sorted(block_cos.items(), key=lambda kv: kv[1])
            detail = ", ".join(f"{b}={c:+.3f}" for b, c in worst)
            overall_str = (
                f"{overall_cos:+.3f}" if not math.isnan(overall_cos) else "n/a"
            )
            logger.info(
                "[L0Ã¢â€ â€HSIC interference] epoch=%d | overall_cos=%s | "
                "conflicting_blocks=%d/%d | per-block: %s",
                int(self.current_epoch),
                overall_str,
                n_conflict,
                n_blocks,
                detail,
            )


    # ------------------------------------------------------------------
    # Descendant-excluding HSIC mask
    # ------------------------------------------------------------------

    # Self-attention types whose score tensor is a DIRECTED edge posterior
    # (existence gate x direction gate), from which "descendant" is well
    # defined.  Symmetric / undirected scores cannot orient an edge, so the
    # descendant set would be meaningless there.
    _DIRECTED_SELF_ATTENTION_TYPES = ("GatedSelfAttention",)

    @staticmethod
    def _module_bkd_keep(mod) -> Optional[torch.Tensor]:
        # Per-key keep mask from either BKD implementation (None if absent).
        # Inline-BKD modules (GatedCrossAttention / GatedSelfAttention /
        # CommutatorSelfAttention) expose ``last_bkd_keep`` (float, 1 = kept);
        # the CausalCrossAttention family wraps a BatchConsistentKeyDropout
        # sub-module exposing ``_last_key_mask`` (bool, True = kept).  Both
        # are None in eval mode / inactive phases, which propagates as
        # "no BKD exclusion this step".
        if mod is None:
            return None
        keep = getattr(mod, "last_bkd_keep", None)
        if keep is not None:
            return keep
        bkd = getattr(mod, "batch_key_dropout", None)
        if bkd is not None:
            mask = getattr(bkd, "_last_key_mask", None)
            if mask is not None:
                return mask.to(torch.float32)
        return None

    def _build_bkd_keep_mask(
        self, n_targets: int, n_sources: int, device
    ) -> Optional[torch.Tensor]:
        # ``(n_targets, n_sources)`` 0/1 pair mask of keys NOT dropped by BKD.
        # Returns None when the feature is off or no attention module applied
        # BKD this step (the HSIC pair set is then unchanged).  Split mode
        # maps the cross-attention keep mask onto the S columns and the
        # self-attention keep mask onto the X columns; a source whose module
        # has no BKD defaults to kept.  Homogeneous mode AND-combines the
        # available masks: a pair is excluded only when the key was dropped
        # in every module that could route it into the estimate.
        if not self.hsic_bkd_exclude_dropped:
            return None
        inner_cross = getattr(
            getattr(self.model, "attention", None), "inner_attention", None
        )
        inner_self = getattr(
            getattr(self.model, "self_attention", None), "inner_attention", None
        )
        cross_keep = self._module_bkd_keep(inner_cross)
        self_keep = self._module_bkd_keep(inner_self)
        if cross_keep is None and self_keep is None:
            return None
        if self.homogeneous_nodes:
            keeps = [k for k in (cross_keep, self_keep) if k is not None]
            col = keeps[0].to(device=device, dtype=torch.float32)
            for k in keeps[1:]:
                col = torch.minimum(col, k.to(device=device, dtype=torch.float32))
        else:
            L_S = int(self.S_seq_len)
            cross_part = (
                cross_keep.to(device=device, dtype=torch.float32)
                if cross_keep is not None
                else torch.ones(L_S, device=device)
            )
            n_x = max(int(n_sources) - L_S, 0)
            self_part = (
                self_keep.to(device=device, dtype=torch.float32)
                if self_keep is not None
                else torch.ones(n_x, device=device)
            )
            col = torch.cat([cross_part, self_part])
        col = col.detach()
        if col.numel() != n_sources:
            logger.warning(
                "BKD keep-mask size %d != HSIC n_sources %d; skipping BKD "
                "dropped-key exclusion this step.",
                col.numel(), n_sources,
            )
            return None
        return col.unsqueeze(0).expand(n_targets, n_sources)

    # ------------------------------------------------------------------
    # LOO conditional-HSIC gate (docs/ideas/CONDITIONAL_HSIC_COUNTERPROPOSAL.md)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _loo_measure_forward(self, S, X, mask):
        """Eval-mode forward for LOO measurement.

        BKD is gated on ``self.training``, so under ``model.eval()`` the
        measurement passes see the full candidate set with no key dropout Ã¢â‚¬â€
        the two worlds (baseline vs. masked) then differ ONLY in the masked
        edge, as the Delta contrast requires.  ``mask`` is an
        ``(n_targets, n_sources)`` 1=allowed / 0=forbidden combined mask
        routed to ``forward_with_actual`` as ``oracle_combined_mask`` (it is
        intersected with the structural mask, so forbidden pairs stay
        forbidden), or None for the baseline pass.
        """
        x_blanked = X.clone()
        x_blanked[:, :, self.val_idx] = 0.0
        s_blanked = None
        if self.homogeneous_nodes:
            s_blanked = S.clone()
            s_blanked[:, :, self.val_idx] = 0.0
        out = self.model.forward_with_actual(
            source_tensor=S,
            x_blanked=x_blanked,
            x_actual=X,
            oracle=False,
            oracle_combined_mask=mask,
            s_blanked=s_blanked,
        )
        return out[0].squeeze()  # pred_x (B, n_targets)

    @torch.no_grad()
    def _compute_loo_gamma(self, S, X, x_target):
        """Per-edge Bayes multiplier gamma, shape (n_targets, n_sources).

        For each candidate source column i, one masked forward pass yields
        the leave-one-out residual eps^{-i}; gamma-null-calibrated HSIC of
        (X_i, eps^{+}) and (X_i, eps^{-i}) feeds ``bayes_multiplier`` with
        prior P_m = current (detached) score tensor.  Rows that would become
        fully masked by dropping column i are left unmasked and keep
        gamma = 1 (edge i is the row's only allowed key Ã¢â‚¬â€ no measurement is
        possible there).

        Cost: one kernel build per calibration call, so O(T x S) kernel
        matrices per refresh Ã¢â‚¬â€ control with ``loo_gamma_topk`` /
        ``loo_gamma_refresh`` / ``loo_gamma_permutations``.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            pred_plus = self._loo_measure_forward(S, X, None)
            resid_plus = x_target.squeeze() - pred_plus          # (B, T)
            if self.homogeneous_nodes:
                combined = x_target.squeeze()                    # (B, N)
            else:
                combined = torch.cat(
                    [S[:, :, self.val_idx], x_target.squeeze()], dim=1
                )                                                # (B, L_S+T)
            n_targets = resid_plus.shape[-1]
            n_sources = combined.shape[-1]
            device = combined.device

            gamma = torch.ones(n_targets, n_sources, device=device)

            # Prior P_m: current structural score tensor (detached), else 0.5.
            p_m = torch.full((n_targets, n_sources), 0.5, device=device)
            get_score = getattr(self.model, "get_score_tensor_for_sparsity", None)
            if callable(get_score):
                s = get_score()
                if s is not None and tuple(s.shape) == (n_targets, n_sources):
                    p_m = s.detach().clamp(0.0, 1.0)

            return self._loo_gamma_loop(
                S, X, x_target, resid_plus, combined, p_m, device
            )
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def _loo_gamma_loop(self, S, X, x_target, resid_plus, combined, p_m, device):
        """Masked-forward loop of ``_compute_loo_gamma`` (split for size)."""
        n_targets = resid_plus.shape[-1]
        n_sources = combined.shape[-1]
        gamma = torch.ones(n_targets, n_sources, device=device)

        # Candidate columns: top-k per row by P_m, or all (full LOO).
        cols_per_row = None
        if self.loo_gamma_topk is not None and self.loo_gamma_topk < n_sources:
            cols_per_row = torch.topk(p_m, k=self.loo_gamma_topk, dim=1).indices

        base_mask = torch.ones(n_targets, n_sources, device=device)
        measured_cols = (
            sorted(set(torch.unique(cols_per_row).tolist()))
            if cols_per_row is not None
            else list(range(n_sources))
        )
        for i in measured_cols:
            mask = base_mask.clone()
            mask[:, i] = 0.0
            dead_rows = mask.sum(dim=1) == 0
            if bool(dead_rows.any()):
                mask[dead_rows, i] = 1.0  # no measurement possible there
            pred_minus = self._loo_measure_forward(S, X, mask)
            resid_minus = x_target.squeeze() - pred_minus
            for j in range(n_targets):
                if cols_per_row is not None and not bool(
                    (cols_per_row[j] == i).any()
                ):
                    continue
                if bool(dead_rows[j]):
                    continue
                cal_plus = hsic_null_calibration(
                    combined[:, i], resid_plus[:, j],
                    adaptive_bandwidth=True,
                    n_permutations=self.loo_gamma_permutations,
                )
                cal_minus = hsic_null_calibration(
                    combined[:, i], resid_minus[:, j],
                    adaptive_bandwidth=True,
                    n_permutations=self.loo_gamma_permutations,
                )
                gamma[j, i] = bayes_multiplier(
                    cal_plus["log_q0"], cal_minus["log_q0"], p_m[j, i]
                ).to(device)
        return gamma

    def _maybe_update_loo_gamma(self, S, X, x_target, stage: str):
        """Return the detached (n_targets, n_sources) gamma gate, or None.

        Recomputes on train steps every ``loo_gamma_refresh`` steps and
        EMA-smooths; validation/test reuse the cached EMA.  Any failure is
        logged and degrades to no gating (matching the descendant-mask
        robustness pattern).
        """
        if stage == "train":
            self._loo_gamma_step += 1
            if self._loo_gamma_step % self.loo_gamma_refresh == 0:
                try:
                    new = self._compute_loo_gamma(S, X, x_target)
                except Exception as exc:  # never break training on the gate
                    logger.warning(
                        "LOO gamma computation failed (%s); gating skipped "
                        "for this refresh.", exc,
                    )
                    new = None
                if new is not None:
                    prev = self._loo_gamma_cache
                    if prev is None or prev.shape != new.shape:
                        self._loo_gamma_cache = new
                    else:
                        a = self.loo_gamma_ema
                        self._loo_gamma_cache = (
                            a * prev.to(new.device) + (1.0 - a) * new
                        )
        g = self._loo_gamma_cache
        if g is None:
            return None
        self._last_loo_gamma_mean = float(g.mean())
        self._last_loo_gamma_min = float(g.min())
        return g

    def _next_cross_fit_batch(self):
        """Next fold-B batch (moved to the module device), or ``None``.

        Fold B is an independent, permanently-disjoint subset of the training
        set (see ``ProcessDataModule.set_hsic_cross_fit``).  Its loader is
        cycled independently of the Lightning train loader: the two folds differ
        in size and shuffle order, so the iterator is simply restarted whenever
        it is exhausted rather than being tied to the epoch boundary.

        Returns ``None`` (silently falling back to the in-batch residual) when
        cross-fitting is off or no fold loader is installed -- e.g. under a
        trainer that never called ``set_hsic_cross_fit``.
        """
        if not self.hsic_cross_fit:
            return None

        if self._xfit_loader is None:
            dm = getattr(self.trainer, "datamodule", None) if self._trainer else None
            get = getattr(dm, "hsic_cross_fit_dataloaders", None) if dm else None
            loaders = get() if callable(get) else None
            if loaders is None:
                if not getattr(self, "_warned_no_xfit", False):
                    self._warned_no_xfit = True
                    logger.warning(
                        "hsic_cross_fit=True but the datamodule exposes no "
                        "cross-fit folds; falling back to the in-batch residual "
                        "(the HSIC is then measured on the fitted samples)."
                    )
                return None
            self._xfit_loader = loaders[1]      # fold B

        try:
            if self._xfit_iter is None:
                self._xfit_iter = iter(self._xfit_loader)
            batch = next(self._xfit_iter)
        except StopIteration:
            self._xfit_iter = iter(self._xfit_loader)
            batch = next(self._xfit_iter)

        return [
            b.to(self.device) if torch.is_tensor(b) else b for b in batch
        ]

    def _build_descendant_weight_mask(
        self,
        score_tensor: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Detached 0/1 mask marking DESCENDANT pairs, for hybrid aggregation.

        ``1`` where source ``j`` is a descendant of target ``i`` (those pairs get
        attention-weighted), ``0`` elsewhere (those enter unweighted).  This is
        the exact complement of ``build_hsic_pair_mask``'s "kept" set, so the
        closure logic -- hardening, transitive closure, two-cycle resolution --
        is shared rather than re-derived.

        The source is the DIRECTED score tensor: its asymmetric (skew) term is
        what carries edge orientation, and unlike the gated posterior it does
        not shrink as attention collapses, so the mask cannot be dissolved by
        the very collapse it is meant to police.

        Returns ``None`` when no mask can be built (no/multi-head score tensor),
        in which case the caller degrades to plain attention weighting.
        """
        if score_tensor is None or not isinstance(score_tensor, torch.Tensor):
            return None
        if score_tensor.dim() != 2:
            return None

        try:
            keep_mask, kept_frac, _cyclic = build_hsic_pair_mask(
                score_tensor=score_tensor.detach(),
                s_seq_len=self.S_seq_len,
                homogeneous_nodes=self.homogeneous_nodes,
                threshold=self.hsic_descendant_threshold,
                hops=self.hsic_descendant_hops,
                # The diagonal is handled by ``exclude_diagonal`` in the
                # aggregation itself; keep it out of the descendant set.
                exclude_self=False,
                excluded_weight=0.0,
            )
        except (ValueError, RuntimeError) as e:   # pragma: no cover - guard
            logger.warning(f"hybrid descendant mask unavailable: {e}")
            return None

        # keep_mask is 1 on NON-descendants -> invert to get the descendant set.
        desc = (1.0 - keep_mask).detach()
        self._last_desc_weight_frac = float(1.0 - kept_frac)
        return desc

    def _build_hsic_descendant_mask(
        self,
        score_tensor: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], float, bool]:
        """Build the detached HSIC pair mask that excludes ``Desc(i) U {i}``.

        Returns ``(mask, kept_frac, is_cyclic)``.  ``mask is None`` means "no
        masking this step" Ã¢â‚¬â€ the HSIC term then falls back to the plain mean, so
        every guard below degrades gracefully to the pre-feature behaviour.

        Guards, in order:
          1. feature off, or the self-attention block has no directed posterior;
          2. no score tensor available, or it is not 2-D (multi-head);
          3. warmup: the learned adjacency is still ~random early on, and
             masking on a wrong graph would delete exactly the parent signal we
             are trying to find (self-confirmation risk).  The countdown starts
             at ``self._descendant_warmup_anchor`` rather than at epoch 0, so
             the adaptive trainer can anchor it to the first epoch of the FIRST
             structure phase Ã¢â‚¬â€ the epochs that actually train the structure Ã¢â‚¬â€
             instead of having it silently expire during the (long) reconstruct
             warmup.  ``anchor is None`` means the warmup was already served;
          4. collapse: if a dense adjacency makes the closure swallow almost
             every pair, the structural gradient would silently vanish, so we
             fall back to no masking and log it.
        """
        if not (self.hsic_exclude_descendants and self._descendant_mask_supported):
            return None, 1.0, False

        if score_tensor is None or not isinstance(score_tensor, torch.Tensor):
            return None, 1.0, False
        if score_tensor.dim() != 2:
            # Multi-head posteriors have no single (target, source) adjacency.
            return None, 1.0, False

        anchor = self._descendant_warmup_anchor
        if (
            anchor is not None
            and (self.current_epoch - anchor) < self.hsic_descendant_warmup_epochs
        ):
            return None, 1.0, False

        score = score_tensor.detach()

        # Optional EMA smoothing: stops the mask from thrashing between batches
        # when edge posteriors sit near the threshold.
        if self.hsic_descendant_ema > 0.0:
            beta = self.hsic_descendant_ema
            prev = self._descendant_ema_score
            if prev is None or prev.shape != score.shape:
                self._descendant_ema_score = score.clone()
            else:
                self._descendant_ema_score = (
                    beta * prev.to(score.device) + (1.0 - beta) * score
                )
            score = self._descendant_ema_score

        try:
            if self.hsic_descendant_mode == "budget":
                # Budgeted variant: rank the pairs by the soft descendant score
                # and exclude the top budget_frac.  The cap IS the collapse
                # guard, so the mask triggers every step.
                mask, kept_frac, is_cyclic = build_hsic_pair_mask_budgeted(
                    score_tensor=score,
                    s_seq_len=self.S_seq_len,
                    homogeneous_nodes=self.homogeneous_nodes,
                    budget_frac=self.hsic_descendant_budget_frac,
                    per_row=self.hsic_descendant_per_row,
                    exclude_self=self.hsic_descendant_exclude_self,
                    excluded_weight=self.hsic_descendant_weight,
                    tnorm=self.hsic_descendant_tnorm,
                    hops=self.hsic_descendant_hops,
                )
            else:
                mask, kept_frac, is_cyclic = build_hsic_pair_mask(
                    score_tensor=score,
                    s_seq_len=self.S_seq_len,
                    homogeneous_nodes=self.homogeneous_nodes,
                    threshold=self.hsic_descendant_threshold,
                    hops=self.hsic_descendant_hops,
                    exclude_self=self.hsic_descendant_exclude_self,
                    excluded_weight=self.hsic_descendant_weight,
                )
        except ValueError as exc:
            # Shape inconsistency: never break training over a diagnostic mask.
            logger.warning(
                "Descendant HSIC mask skipped (%s). Falling back to unmasked HSIC.",
                exc,
            )
            return None, 1.0, False

        # The collapse guard is a THRESHOLD-mode concern: in budget mode the
        # cap itself bounds the exclusion, so the guard would fire spuriously.
        if (
            self.hsic_descendant_mode == "threshold"
            and kept_frac < self.hsic_descendant_min_kept_frac
        ):
            logger.warning(
                "Descendant HSIC mask would keep only %.1f%% of pairs "
                "(< min_kept_frac=%.1f%%) Ã¢â‚¬â€ the learned graph is too dense and "
                "the structural signal would collapse. Falling back to unmasked "
                "HSIC for this step.",
                100.0 * kept_frac,
                100.0 * self.hsic_descendant_min_kept_frac,
            )
            return None, kept_frac, is_cyclic

        return mask, kept_frac, is_cyclic

    # ------------------------------------------------------------------
    # Acyclicity helper (mirrors SingleCausalForecaster)
    # ------------------------------------------------------------------

    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity penalty h(A) = tr(exp(A Ã¢Å â„¢ A)) - d.

        Zero iff A induces a directed acyclic graph (Zheng et al., 2018).
        Applied to the XÃ¢â€ â€™X sub-block (square, L_X Ãƒâ€” L_X) of the combined
        score tensor.  Caller must ensure A is 2-D (shape (L_X, L_X)).
        """
        d = A.shape[-1]
        return torch.trace(torch.matrix_exp(A * A)) - d

    @staticmethod
    def _logdet_acyclicity(A: torch.Tensor, s: object = "adaptive"):
        """DAGMA-style log-det penalty h(A) = -log det(sI - A) + d*log s
        (Bello et al., 2022).

        ``A`` is the NONNEGATIVE gate-posterior score matrix (entries in
        (0, 1)), so no elementwise squaring is applied (in NOTEARS the
        elementwise square only exists to make signed linear weights
        nonnegative; on a posterior it would attenuate exactly the
        uncertain edges).

        ``s`` selects the DAGMA shift:

        * ``"adaptive"``: s = max_i sum_j A_ij (STOP-GRAD) + eps.  Since
          rho(A) <= max row sum for any matrix, sI - A is guaranteed
          nonsingular at every iterate -- the determinant singularity /
          NaN region of fixed-s log-det is unreachable.  With
          h = -log det(I - A/s) and rho(A/s) < 1, h >= 0 with equality
          iff A is nilpotent (DAG).
        * positive float: fixed global shift (pure DAGMA uses s = 1).
          Raises if sI - A is not positive-determinant.

        Caller must ensure A is 2-D (shape (d, d)).
        """
        d = A.shape[-1]
        if isinstance(s, str):
            assert s == "adaptive"
            # Stop-grad: s is a per-step conditioning constant, not part of
            # the penalty landscape.  The eps margin keeps sI - A strictly
            # nonsingular even in the (measure-zero) Perron-equality case
            # and makes h(0) = 0 exact.
            s_val = A.detach().sum(dim=-1).max() + 1e-4
        else:
            s_val = torch.as_tensor(float(s), dtype=A.dtype, device=A.device)
        eye = torch.eye(d, dtype=A.dtype, device=A.device)
        sign, logabsdet = torch.linalg.slogdet(s_val * eye - A)
        sign_f = float(sign.detach())
        if not torch.isfinite(logabsdet.detach()) or sign_f <= 0.0:
            raise FloatingPointError(
                f"log-det acyclicity: det(sI - A) non-positive "
                f"(sign={sign_f}); rho(A) >= s.  Use "
                f"acyclicity_s='adaptive' or a larger fixed shift."
            )
        return -logabsdet + d * torch.log(s_val)

    @staticmethod
    def _nilpotent_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """Exact nilpotency penalty h(A) = sum_{k=1..d} tr(A^k).

        For NONNEGATIVE A every term is >= 0, and tr(A^k) = 0 iff the graph
        has no closed walk of length k; a d-node graph is acyclic iff it has
        no closed walk of any length <= d.  Hence h(A) = 0 iff A induces a
        DAG -- an exact characterisation (no factorial down-weighting of
        long cycles as in NOTEARS, no determinant as in log-det).  Computed
        by explicit powers; cheap at the node counts in use.

        Caller must ensure A is 2-D (shape (d, d)).
        """
        d = A.shape[-1]
        h = torch.zeros((), dtype=A.dtype, device=A.device)
        Ak = A
        for _ in range(d):
            h = h + torch.trace(Ak)
            Ak = Ak @ A
        return h

    def _acyclicity_penalty(self, A: torch.Tensor) -> torch.Tensor:
        """Dispatch to the configured acyclicity functional
        (``training.acyclicity_fn``: notears | logdet | nilpotent)."""
        if self.acyclicity_fn == "logdet":
            return self._logdet_acyclicity(A, self.acyclicity_s)
        if self.acyclicity_fn == "nilpotent":
            return self._nilpotent_acyclicity(A)
        return self._notears_acyclicity(A)


    # ------------------------------------------------------------------
    # Structural-regularizer safeguard helpers
    # ------------------------------------------------------------------

    def freeze_hsic_bandwidth(self, combined_source, residuals) -> None:
        """Latch per-variable median-heuristic bandwidths for this phase.

        Called lazily on the first train batch of a phase whose config sets
        ``hsic_freeze_bandwidth: true``.  Saves the current
        ``(hsic_sigma, hsic_adaptive_bandwidth)`` so
        :meth:`restore_hsic_bandwidth` can undo the freeze at phase exit.
        """
        from causaliT.utils.hsic_utils import _median_bandwidth
        with torch.no_grad():
            sig_src = torch.stack([_median_bandwidth(combined_source[:, i].detach())
                                   for i in range(combined_source.shape[1])])
            sig_res = torch.stack([_median_bandwidth(residuals[:, j].detach())
                                   for j in range(residuals.shape[1])])
        if self._hsic_bw_saved is None:
            self._hsic_bw_saved = (self.hsic_sigma, self.hsic_adaptive_bandwidth)
        self.hsic_sigma = (sig_src, sig_res)
        self.hsic_adaptive_bandwidth = False
        self._hsic_bw_frozen_sigmas = (sig_src, sig_res)
        self.log("struct/hsic_sigma_src_med", float(sig_src.median()),
                 on_step=False, on_epoch=True)
        self.log("struct/hsic_sigma_res_med", float(sig_res.median()),
                 on_step=False, on_epoch=True)
        logger.info(
            "HSIC bandwidth frozen for this phase: median sigma_src=%.4g, "
            "median sigma_res=%.4g.", float(sig_src.median()),
            float(sig_res.median()),
        )

    def restore_hsic_bandwidth(self) -> None:
        """Undo the per-phase bandwidth freeze (phase exit)."""
        if self._hsic_bw_saved is not None:
            self.hsic_sigma, self.hsic_adaptive_bandwidth = self._hsic_bw_saved
            self._hsic_bw_saved = None
        self._hsic_bw_frozen_sigmas = None

    def _hsic_safeguard_ref(
        self, hsic_reg: torch.Tensor, stage: str
    ) -> Optional[float]:
        """Reference value (EMA of the weighted HSIC term) for the reg caps.

        Updated on train batches only (detached scalar), so the val/test
        logging applies the same caps the training steps used.  With
        ``hsic_safeguard_ema == 0`` the reference is the instantaneous
        per-batch value.  Returns ``None`` when both caps are disabled.
        """
        if self.kappa_max_hsic_pct <= 0.0 and self.lambda_l0_max_hsic_pct <= 0.0:
            return None
        val = float(hsic_reg.detach())
        if stage == "train":
            if self._hsic_reg_ema is None:
                self._hsic_reg_ema = val
            else:
                d = self.hsic_safeguard_ema
                self._hsic_reg_ema = d * self._hsic_reg_ema + (1.0 - d) * val
            return self._hsic_reg_ema
        # val/test: never update.  Instantaneous mode uses the current batch;
        # EMA mode reuses the running reference from the train batches.
        if self.hsic_safeguard_ema == 0.0 or self._hsic_reg_ema is None:
            return val
        return self._hsic_reg_ema
    def _acyclic_safeguard_ref(
        self, acyclic_reg: torch.Tensor, stage: str
    ) -> Optional[float]:
        """Reference value (EMA of the weighted acyclic term) for the MSE cap.

        Mirrors :meth:`_hsic_safeguard_ref` with the roles inverted: the
        acyclic term is the reference the reconstruction loss is capped
        against.  Updated on train batches only (detached scalar), so the
        val/test logging applies the same cap the training steps used.
        With ``hsic_safeguard_ema == 0`` the reference is the instantaneous
        per-batch value.  Returns ``None`` when the MSE cap is disabled.
        """
        if self.mse_max_acyclic_pct <= 0.0 and self.l0_max_acyclic_pct <= 0.0:
            return None
        val = float(acyclic_reg.detach())
        if stage == "train":
            if self._acyclic_reg_ema is None:
                self._acyclic_reg_ema = val
            else:
                d = self.hsic_safeguard_ema
                self._acyclic_reg_ema = d * self._acyclic_reg_ema + (1.0 - d) * val
            return self._acyclic_reg_ema
        # val/test: never update.  Instantaneous mode uses the current batch;
        # EMA mode reuses the running reference from the train batches.
        if self.hsic_safeguard_ema == 0.0 or self._acyclic_reg_ema is None:
            return val
        return self._acyclic_reg_ema



    @staticmethod
    def _cap_reg_coeff(
        base_coeff: float,
        raw_term: torch.Tensor,
        max_pct: float,
        hsic_ref: Optional[float],
    ) -> float:
        """Cap a regularizer coefficient so its term stays <= max_pct * hsic_ref.

        Returns a detached float, so the regularizer's gradient direction is
        preserved and only its magnitude is capped.  No-op (returns
        ``base_coeff``) when the cap is disabled, the base coefficient is
        zero, the reference is unavailable, or the raw term is non-positive
        or non-finite.
        """
        if max_pct <= 0.0 or base_coeff <= 0.0 or hsic_ref is None:
            return base_coeff
        raw = float(raw_term.detach())
        if not math.isfinite(raw) or raw <= 0.0:
            return base_coeff
        return min(base_coeff, max_pct * hsic_ref / raw)

    def hsic_constraint_on_phase_switch(
        self,
        phase: Optional[str] = None,
        bkd_min_keys: Optional[int] = None,
    ) -> None:
        """Reset per-regime constraint memory at an adaptive phase boundary.

        Called by the adaptive trainer at every phase switch: the BKD rung and
        the cross-fit fold change with the phase, so the HSIC EMA and the
        previous-violation memory are not comparable across the boundary.
        The dual variables (lambda, rho) deliberately PERSIST -- they
        accumulate constraint evidence over the whole run.

        ``phase`` / ``bkd_min_keys`` (optional, backward compatible):

        * with ``dual_ascent_structure_only`` the dual ascent + EMA are gated
          to structure phases (a reconstruct phase changes the residual
          regime with FROZEN gates -- its violation is not actionable by the
          structural stream);
        * at structure-phase entry with ``calibrate_tolerance``, the
          permutation-null calibration is (re)armed for the current rung.
        """
        self._hsic_constraint_ema = None
        self._hsic_constraint_prev_violation = None
        self._hsic_rung_keys = None if bkd_min_keys is None else int(bkd_min_keys)
        self._hsic_dual_active = (
            (not self.hsic_dual_structure_only) or phase in (None, "structure")
        )
        if (
            phase == "structure"
            and self.hsic_calibrate_tolerance
            and self.hsic_constraint_source == "hsic"
        ):
            self._hsic_null_samples = []
            self._hsic_null_calib_remaining = self.hsic_calib_batches
            self._hsic_null_calib_round += 1
            logger.info(
                "[hsic-constraint] tolerance calibration armed: %d batches x "
                "%d permutations at structure-phase entry (rung keys=%s).",
                self.hsic_calib_batches, self.hsic_calib_perms,
                str(self._hsic_rung_keys),
            )

    def _update_hsic_dual(self) -> None:
        """Per-epoch dual ascent for the HSIC constraint (Lagrangian mode).

        Driven by the EMA of the RAW train HSIC accumulated in ``_step``:

        * dual ascent: ``lam <- clip(lam + lr*(ema - eps), 0, dual_max)`` with
          ``lr = dual_lr`` on violation (upward) and ``lr = dual_lr_down`` on
          satisfaction (downward), so accumulated pressure can release at its
          own rate once the (possibly per-rung calibrated) tolerance is met;
        * NOTEARS-style rho escalation (only when ``rho_init > 0``, i.e. the
          augmented Lagrangian is active): ``rho <- min(rho*rho_mult,
          rho_max)`` whenever the violation fails to decay to <= 1/4 of its
          previous value.

        The dual state persists across epochs (and checkpoints), so lambda
        accumulates pressure while the constraint stays violated and stops
        growing once the structure satisfies HSIC <= eps.

        Gating (adaptive runs): no ascent in phases where the dual is
        inactive (``dual_ascent_structure_only``) or at low BKD rungs where
        the constraint is unreachable by construction
        (``dual_pause_below_keys``).  State and tolerance are still logged
        every epoch for continuity of the metric curves.
        """
        if self._hsic_constraint_ema is None:
            return
        violation = self._hsic_constraint_ema - self.hsic_tol
        paused = (
            not self._hsic_dual_active
            or (
                self.hsic_dual_pause_below_keys > 0
                and self._hsic_rung_keys is not None
                and self._hsic_rung_keys < self.hsic_dual_pause_below_keys
            )
        )
        if paused:
            self.log("hsic/dual_paused", 1.0, on_step=False, on_epoch=True)
        else:
            lr = self.hsic_dual_lr if violation > 0.0 else self.hsic_dual_lr_down
            new_lambda = self._hsic_dual_lambda + lr * violation
            self._hsic_dual_lambda = min(
                self.hsic_dual_max, max(0.0, new_lambda)
            )
        if (
            not paused
            and self._hsic_rho > 0.0
            and self.hsic_rho_mult > 1.0
            and violation > 0.0
            and self._hsic_constraint_prev_violation is not None
            and violation > 0.25 * self._hsic_constraint_prev_violation
        ):
            self._hsic_rho = min(
                self.hsic_rho_max, self._hsic_rho * self.hsic_rho_mult
            )
        self._hsic_constraint_prev_violation = violation
        self.log("hsic/tolerance", self.hsic_tol, on_step=False, on_epoch=True)
        self.log("hsic/dual_lambda", self._hsic_dual_lambda,
                 on_step=False, on_epoch=True)
        self.log("hsic/rho", self._hsic_rho, on_step=False, on_epoch=True)
        self.log("hsic/constraint_ema", self._hsic_constraint_ema,
                 on_step=False, on_epoch=True)
        self.log("hsic/constraint_violation", violation,
                 on_step=False, on_epoch=True)

    def _update_acyclicity_dual(self) -> None:
        """Per-epoch dual ascent / rho escalation for the acyclicity ALM.

        Driven by the EMA of the RAW h(W) accumulated in ``_step``:

        * dual ascent: ``lam <- clip(lam + dual_lr * violation, 0, dual_max)``
          with ``violation = ema - h_tol`` (negative violation releases the
          accumulated pressure at the same rate, as in _update_hsic_dual);
        * NOTEARS-style rho escalation: ``rho <- min(rho*rho_mult, rho_max)``
          whenever the violation fails to decay to <= 1/4 of its previous
          value (vendor/notears/linear.py: ``h_new > 0.25 * h``).

        Effectively frozen once the EMA violation drops to ~0 (converged:
        h(W) <= h_tol).
        """
        if self._acy_ema is None:
            return
        violation = self._acy_ema - self.acy_h_tol
        self._acy_dual_lambda = min(
            self.acy_dual_max,
            max(0.0, self._acy_dual_lambda + self.acy_dual_lr * violation),
        )
        if (
            violation > 0.0
            and self._acy_rho > 0.0
            and self.acy_rho_mult > 1.0
            and self._acy_prev_violation is not None
            and violation > 0.25 * self._acy_prev_violation
        ):
            self._acy_rho = min(
                self.acy_rho_max, self._acy_rho * self.acy_rho_mult
            )
        self._acy_prev_violation = violation
        self.log("acyclicity/dual_lambda", self._acy_dual_lambda,
                 on_step=False, on_epoch=True)
        self.log("acyclicity/rho", self._acy_rho, on_step=False, on_epoch=True)
        self.log("acyclicity/ema", self._acy_ema, on_step=False, on_epoch=True)
        self.log("acyclicity/violation", violation,
                 on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Per-rung HSIC tolerance calibration (permutation null)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect_hsic_null_sample(
        self,
        combined_source: torch.Tensor,
        residuals: torch.Tensor,
        attention_weights: Optional[torch.Tensor],
        bkd_keep_mask: Optional[torch.Tensor],
        hsic_pair_mask: Optional[torch.Tensor],
    ) -> None:
        """Accumulate permutation-null HSIC replicates from ONE train batch.

        The residuals are permuted across the batch dimension (destroying any
        residual/source dependence) and re-aggregated through the SAME path
        the constraint uses (softmax / attention-weighted / masked mean) with
        the SAME detached pair weights and BKD keep mask, so the null matches
        the current rung regime.  Everything is detached; training RNG is
        untouched (dedicated generator seeded per calibration round+batch).
        """
        hsic_kw = dict(
            sigma=self.hsic_sigma,
            adaptive_bandwidth=self.hsic_adaptive_bandwidth,
            mode=self.hsic_mode,
            nhsic_epsilon=self.nhsic_epsilon,
            source_kernel=self.hsic_kernel_source,
            bandwidth_multipliers=self.hsic_bandwidth_multipliers,
        )
        try:
            gen = torch.Generator()
            gen.manual_seed(
                self.hsic_calib_seed
                + 1009 * self._hsic_null_calib_round
                + self._hsic_null_calib_remaining
            )
            B = residuals.shape[0]
            att_mean = None
            if self.use_attention_weighted_hsic and attention_weights is not None:
                att_mean = attention_weights.detach().mean(dim=0)
                if bkd_keep_mask is not None:
                    att_mean = att_mean * bkd_keep_mask
                    if self.hsic_softmax:
                        # Dropped keys are -inf logits (weight exactly 0),
                        # matching the train aggregation.
                        att_mean = att_mean.masked_fill(
                            bkd_keep_mask == 0, float("-inf")
                        )
            for _ in range(self.hsic_calib_perms):
                perm = torch.randperm(B, generator=gen).to(residuals.device)
                res_p = residuals.detach()[perm]
                if self.use_attention_weighted_hsic and att_mean is not None:
                    if self.hsic_softmax:
                        v = hsic_attention_softmax(
                            source_values=combined_source.detach(),
                            residuals=res_p,
                            attention_weights=att_mean,
                            return_matrix=False,
                            diagonal_offset=(
                                0 if self.homogeneous_nodes else self.S_seq_len
                            ),
                            pair_weight_mode=self.hsic_pair_weight_mode,
                            tilt_tau=self.hsic_tilt_tau,
                            **hsic_kw,
                        )
                    else:
                        v = hsic_attention_weighted(
                            source_values=combined_source.detach(),
                            residuals=res_p,
                            attention_weights=att_mean,
                            exclude_diagonal=self.hsic_weight_descendants_only,
                            return_matrix=False,
                            descendant_mask=None,
                            **hsic_kw,
                        )
                else:
                    v = hsic_cross_per_pair(
                        combined_source.detach(),
                        res_p,
                        pair_mask=hsic_pair_mask,
                        return_matrix=False,
                        **hsic_kw,
                    )
                fv = float(v)
                if fv == fv:  # NaN guard
                    self._hsic_null_samples.append(fv)
        except Exception as exc:  # never break training on calibration
            logger.warning("HSIC null calibration sample skipped: %s", exc)
        self._hsic_null_calib_remaining -= 1
        if self._hsic_null_calib_remaining <= 0:
            self._finalize_hsic_null_calibration()

    def _finalize_hsic_null_calibration(self) -> None:
        """Set ``hsic_tol`` from the accumulated permutation-null samples.

        ``tolerance = quantile(null, q) * margin``.  Falls back to keeping the
        configured tolerance (with a warning) when too few finite samples were
        collected.
        """
        import numpy as np
        samples = np.asarray(self._hsic_null_samples, dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        self._hsic_null_samples = []
        if samples.size < 4:
            logger.warning(
                "HSIC null calibration: only %d finite samples - keeping "
                "tolerance=%.3e.", samples.size, self.hsic_tol,
            )
            return
        null_mean = float(samples.mean())
        null_q = float(np.quantile(samples, self.hsic_calib_quantile))
        old_tol = self.hsic_tol
        self.hsic_tol = null_q * self.hsic_calib_margin
        self.log("hsic/null_mean", null_mean, on_step=False, on_epoch=True)
        self.log("hsic/null_quantile", null_q, on_step=False, on_epoch=True)
        self.log("hsic/tolerance", self.hsic_tol, on_step=False, on_epoch=True)
        logger.info(
            "[hsic-constraint] rung tolerance calibrated: %.3e -> %.3e "
            "(null mean %.3e, q%.2f %.3e, margin %.2f, n=%d, rung keys=%s).",
            old_tol, self.hsic_tol, null_mean, self.hsic_calib_quantile,
            null_q, self.hsic_calib_margin, samples.size,
            str(self._hsic_rung_keys),
        )

    # ------------------------------------------------------------------
    # Group-L1 (identical to SingleCausalForecaster implementation)
    # ------------------------------------------------------------------

    def _compute_group_l1(self):
        """Compute L2,1 norm on embedding columns (group sparsity)."""
        if self.lambda_group_l1 == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device), None

        total_l21 = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        effective_dims = 0

        for name, param in self.model.named_parameters():
            if "nn_embedding" in name and "weight" in name:
                # param shape: (num_embeddings, embedding_dim)
                col_norms = param.norm(dim=0)   # (embedding_dim,)
                total_l21 = total_l21 + col_norms.sum()
                effective_dims += (col_norms > 1e-6).float().sum().item()
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device), None

        return total_l21, torch.tensor(effective_dims / count)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _maybe_init_query_centroid(self, batch) -> None:
        """Lazily initialise the free query embeddings on the first batch.

        Runs at most once, on the first training batch, when
        ``query_centroid_init=True``, ``query_parents_prior`` and/or
        ``query_source_prior`` is set.  Deferred to the first batch because
        value-modulated key embeddings need real data to define the key
        frame.  ORDER: the centroid/default initialisation first, then the
        parents prior OVERWRITES the listed nodes (and re-snapshots the
        fixed rows at the prior values), then the source prior ZEROES and
        freezes the source rows.
        """
        if self._query_centroid_init_done:
            return
        if (
            not self._query_centroid_init
            and not self._query_parents_prior
            and not self._query_source_prior
        ):
            return
        if getattr(self.model, "query_embed_X", None) is None:
            # Nothing to initialise (free_query_embedding disabled); latch off.
            self._query_centroid_init_done = True
            return
        S, X = batch[0], batch[1]
        if self._query_centroid_init:
            self.model.init_query_at_key_centroid(S, X)
            logger.info(
                "Initialised X query embedding at the key centroid "
                "(query_centroid_init=True; all queries start from the same "
                "point)."
            )
        if self._query_parents_prior:
            n_fixed = self.model.init_queries_from_parents(
                self._query_parents_prior, S, X
            )
            logger.info(
                "Applied query parents prior to %d node(s) (%d frozen): %s",
                len(self._query_parents_prior),
                n_fixed,
                {int(c): s for c, s in self._query_parents_prior.items()},
            )
        if self._query_source_prior:
            n_frozen_src = self.model.init_source_queries_zero(
                self._query_source_prior
            )
            logger.info(
                "Applied query source prior to %d node(s) (%d frozen): %s",
                len(self._query_source_prior),
                n_frozen_src,
                {int(n): s for n, s in self._query_source_prior.items()},
            )

        self._query_centroid_init_done = True
        # Centroid-commit: the shadow starts at the same point (assignment =
        # the full key set, the "select-all" hypothesis).
        self.model.sync_commit_shadows()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """Re-assert prior-frozen query rows after EVERY optimizer step.

        Covers the automatic-optimization path (use_gradient_routing=False):
        the manual path re-asserts right after ``opt_struct.step()``, but on
        the joint-loss path Lightning steps internally, so without this hook
        decoupled weight decay / gradient noise would drift the frozen rows.
        No-op when no query row is frozen.
        """
        self.model.reassert_frozen_query_rows()

    def on_train_epoch_start(self):
        """Advance the fan-in squeeze and write ``mu(t)`` onto every module.

        No-op unless ``experiment.fanin_prior`` is set.  The write precedes the
        clock increment, so the first epoch always sees ``mu(0) = 1`` and the
        centroid initialisation is untouched.
        """
        self.fanin_schedule.on_epoch_start(self.model)
        for name, value in self.fanin_schedule.metrics(self.model).items():
            self.log(name, value, on_step=False, on_epoch=True)

    def _structural_backward(self, loss_structural: torch.Tensor) -> None:
        """Structural backward, with optional PCGrad gradient surgery.

        When ``training.gradient_surgery`` is enabled, the single fused
        backward on ``loss_structural`` is replaced by per-term
        ``torch.autograd.grad`` calls over the structural parameters:

        * the HSIC term (``(1 - alpha) * hsic_reg``) is the reference;
        * the L0 and NOTEARS terms are projected per block against it
          (conflicting components removed, see
          :func:`causaliT.training.gradient_surgery.pcgrad_reconcile`);
        * the remaining structural terms (struct-recon mix, score sparsity,
          group L1, query norm) are bundled untouched.

        The combined gradient is written directly into ``p.grad`` of the
        structural params, so downstream logic (nodewise WTA masking /
        snapshot, ``opt_struct.step()``) is unaffected.  With surgery disabled
        this is exactly the original ``manual_backward(loss_structural)``.
        """
        if not self.gradient_surgery:
            self.manual_backward(loss_structural)
            return

        hsic_term = self._last_struct_hsic_term
        targets: Dict[str, torch.Tensor] = {}
        if self._last_l0_reg is not None and self._last_l0_reg.requires_grad:
            targets["l0"] = self._last_l0_reg
        if (
            self._last_acyclic_reg is not None
            and self._last_acyclic_reg.requires_grad
        ):
            targets["notears"] = self._last_acyclic_reg

        if (
            hsic_term is None
            or not hsic_term.requires_grad
            or not targets
        ):
            # Nothing to reconcile (e.g. lambda_hsic == 0 or both regs off):
            # fall back to the plain fused structural backward.
            self.manual_backward(loss_structural)
            return

        # Per-block grouping restricted to the structural params (the only
        # ones the structural optimizer steps).
        if not self._interference_blocks:
            self._interference_blocks = build_interference_blocks(self.model)
        struct_ids = {id(p) for p in self._structural_params}
        blocks = {
            name: [p for p in plist if id(p) in struct_ids]
            for name, plist in self._interference_blocks.items()
        }
        blocks = {name: plist for name, plist in blocks.items() if plist}
        all_params = [p for plist in blocks.values() for p in plist]
        if not all_params:
            # No trainable structural params in this phase (e.g. warmup with
            # theta_S frozen): nothing to reconcile -> plain fused backward.
            self.manual_backward(loss_structural)
            return

        g_hsic = torch.autograd.grad(
            hsic_term, all_params, retain_graph=True, allow_unused=True
        )
        g_targets = {
            name: torch.autograd.grad(
                term, all_params, retain_graph=True, allow_unused=True
            )
            for name, term in targets.items()
        }
        # Last consumer of the graph -> no retain.
        g_rest = torch.autograd.grad(
            self._last_struct_rest, all_params,
            retain_graph=False, allow_unused=True,
        )

        projected, metrics = pcgrad_reconcile(g_hsic, g_targets, blocks, all_params)

        for i, p in enumerate(all_params):
            parts = [g_hsic[i], g_rest[i]]
            parts += [projected[name][i] for name in targets]
            parts = [g for g in parts if g is not None]
            if parts:
                p.grad = torch.stack([g.detach() for g in parts]).sum(dim=0)

        # Always-on surgery metrics (epoch-mean aggregates).
        for key, val in metrics.items():
            if val == val:  # skip NaN (no valid block for that target)
                self.log(f"surgery/{key}", val, on_step=False, on_epoch=True)

    def _joint_pcgrad_step(self) -> None:
        """Joint-optimizer PCGrad step (``use_gradient_routing=False``).

        Same per-term decomposition as :meth:`_structural_backward` but over
        ALL trainable parameters (the single optimizer steps everything, and
        the recon / HSIC / regularizer conflicts live in shared parameters
        too): the gradient written into ``p.grad`` is

            g_recon + g_hsic + g_rest + sum(PCGrad-projected g_reg)

        with the HSIC term as the reference and L0 / NOTEARS projected per
        interference block.  Falls back to a plain fused backward of the
        total loss when there is nothing to reconcile (no HSIC signal or
        both regularizers off).  NOTE: manual optimization is active here,
        so LR schedulers / gradient clipping configured through Lightning
        are NOT applied (this arm uses neither).
        """
        opt = self.optimizers()
        loss_recon = self._last_loss_components["loss_recon"]
        loss_structural = self._last_loss_components["loss_structural"]
        hsic_term = self._last_struct_hsic_term
        targets: Dict[str, torch.Tensor] = {}
        if self._last_l0_reg is not None and self._last_l0_reg.requires_grad:
            targets["l0"] = self._last_l0_reg
        if (
            self._last_acyclic_reg is not None
            and self._last_acyclic_reg.requires_grad
        ):
            targets["notears"] = self._last_acyclic_reg

        opt.zero_grad()
        if hsic_term is None or not hsic_term.requires_grad or not targets:
            # Nothing to reconcile: plain fused backward of the total loss.
            self.manual_backward(loss_recon + loss_structural)
            opt.step()
            return

        # Per-block grouping over ALL trainable parameters (joint mode:
        # conflicts are not restricted to the structural partition).
        if not self._interference_blocks:
            self._interference_blocks = build_interference_blocks(self.model)
        blocks = {
            name: plist
            for name, plist in self._interference_blocks.items()
            if plist
        }
        all_params = [p for p in self.parameters() if p.requires_grad]

        g_recon = torch.autograd.grad(
            loss_recon, all_params, retain_graph=True, allow_unused=True
        )
        g_hsic = torch.autograd.grad(
            hsic_term, all_params, retain_graph=True, allow_unused=True
        )
        g_targets = {
            name: torch.autograd.grad(
                term, all_params, retain_graph=True, allow_unused=True
            )
            for name, term in targets.items()
        }
        # Last consumer of the graph -> no retain.
        g_rest = torch.autograd.grad(
            self._last_struct_rest, all_params,
            retain_graph=False, allow_unused=True,
        )

        projected, metrics = pcgrad_reconcile(g_hsic, g_targets, blocks, all_params)

        for i, p in enumerate(all_params):
            parts = [g_recon[i], g_hsic[i], g_rest[i]]
            parts += [projected[name][i] for name in targets]
            parts = [g for g in parts if g is not None]
            if parts:
                p.grad = torch.stack([g.detach() for g in parts]).sum(dim=0)

        # Always-on surgery metrics (epoch-mean aggregates).
        for key, val in metrics.items():
            if val == val:  # skip NaN (no valid block for that target)
                self.log(f"surgery/{key}", val, on_step=False, on_epoch=True)

        opt.step()

    def training_step(self, batch, batch_idx):
        # One-off: place every X query at the key centroid before the first step.
        self._maybe_init_query_centroid(batch)

        if self.use_gradient_routing:

            # --- Manual optimization with dual backward ---
            # Both backward passes must complete BEFORE any optimizer step,
            # otherwise in-place parameter updates invalidate the computation graph.
            opt_recon, opt_struct = self.optimizers()

            # Single forward pass + loss computation (shared)
            total_loss, _, _ = self._step(batch=batch, stage="train")
            loss_recon = self._last_loss_components["loss_recon"]
            loss_structural = self._last_loss_components["loss_structural"]

            # Diagnostic: probe L0 Ã¢â€ â€ HSIC gradient interference on the live
            # graph BEFORE any zero_grad / backward.  Uses autograd.grad with
            # retain_graph=True and never touches .grad, so the dual backward
            # below is unaffected.
            self._maybe_log_interference(batch_idx)

            # Zero all gradients
            opt_recon.zero_grad()
            opt_struct.zero_grad()

            # Backward 1: recon loss (retain graph for second backward)
            self.manual_backward(loss_recon, retain_graph=True)

            # Save recon gradients for reconstruction params
            _saved_recon_grads = {}
            for p in self._reconstruction_params:
                if p.grad is not None:
                    _saved_recon_grads[id(p)] = p.grad.clone()

            # Zero all gradients
            self.zero_grad()

            # Centroid-commit: capture the HSIC-only shadow evidence BEFORE
            # the structural backward consumes the graph, and drop any recon
            # gradient the STE path put on the shadows.  Skipped when the
            # structural params are frozen (reconstruct phase).
            cc_active = (
                self._commit is not None
                and self.model.query_embed_X.embedding.weight.requires_grad
            )
            cc_grads = None
            if cc_active:
                for t in self._commit.tables:
                    if t.shadow.grad is not None:
                        t.shadow.grad = None
                if self._commit_source in ("hsic", "hsic_unrolled"):
                    cc_grads = self._shadow_evidence(batch)

            # Backward 2: structural loss (graph consumed).  With
            # training.gradient_surgery=True this applies PCGrad per block to
            # the L0 / NOTEARS terms against the HSIC gradient instead of a
            # fused backward.  With training.structural_grad='hsic_unrolled'
            # the HSIC gradient is the DARTS second-order (bi-level) one,
            # computed on fold B when cross-fitting is active.
            if self.structural_grad == "hsic_unrolled":
                xfit = self._next_cross_fit_batch()
                self._bi_level_step(xfit if xfit is not None else batch)
            else:
                self._structural_backward(loss_structural)

            # Restore recon grads on reconstruction params
            for p in self._reconstruction_params:
                if id(p) in _saved_recon_grads:
                    p.grad = _saved_recon_grads[id(p)]

            # Nodewise winner-take-all: select the top-SNR query nodes, mask
            # the other rows, and snapshot them (params + optimizer state) so
            # the structural optimizer step can be reverted row-wise.
            nw_snap = None
            if self._nodewise is not None:
                selected = self._nodewise.select()
                if selected is not None:  # None = no structural grads (recon)
                    self._nodewise_log_step(selected)
                    if len(selected) < self._nodewise.n_nodes:
                        self._nodewise.mask_grads(selected)
                        nw_snap = self._nodewise.snapshot(opt_struct, selected)

            # Now step both optimizers (graph fully consumed, safe)
            opt_recon.step()
            opt_struct.step()

            # Re-assert prior-frozen query rows: decoupled weight decay and
            # structural gradient noise act even on zero-gradient rows.
            self.model.reassert_frozen_query_rows()

            if nw_snap is not None:
                NodewiseQuerySelector.restore(nw_snap)

            # Centroid-commit: evidence update + commit checks (consumes and
            # clears the shadow grads).  With the bilevel gate, eligible
            # commits are deferred and resolved by paired refit probes.
            if cc_active:
                n_new = self._commit.step(cc_grads, defer=self._gate_enabled)
                self.log("struct/commit_count", float(n_new), on_step=False,
                         on_epoch=True, reduce_fx="sum")
                for c in self._commit.last_commits:
                    self.log("struct/commit_margin", float(c["margin"]),
                             on_step=False, on_epoch=True, reduce_fx="max")
                if self._gate_enabled:
                    self._run_bilevel_gate()

            return total_loss
        else:
            total_loss, _, _ = self._step(batch, stage="train")
            # Diagnostic: probe interference on the live graph before the
            # backward (retain_graph=True in the probe keeps the graph intact).
            self._maybe_log_interference(batch_idx)
            if not self.gradient_surgery:
                # Lightning runs its automatic backward on total_loss.
                return total_loss
            # Joint PCGrad: manual optimization (automatic_optimization was
            # disabled in __init__), per-term grads reconciled before the step.
            self._joint_pcgrad_step()
            return total_loss

    # ------------------------------------------------------------------
    # Nodewise query update: diagnostics + phase-switch reset
    # ------------------------------------------------------------------
    def _nodewise_log_step(self, selected):
        """Per-step logging for the nodewise gate (epoch-mean aggregates)."""
        nw = self._nodewise
        if nw is None:
            return
        if nw.selection == "norm":
            self.log("struct/nodewise_max_gnorm", float(nw.last_max_stat),
                     on_step=False, on_epoch=True)
        else:
            snr = nw.current_snr()
            self.log("struct/nodewise_max_snr", float(snr.max()),
                     on_step=False, on_epoch=True)
        gate_fired = len(selected) == 0
        self.log("struct/nodewise_gate_fired", float(gate_fired),
                 on_step=False, on_epoch=True)

    def nodewise_reset_stats(self):
        """Clear the SNR evidence (called by the adaptive trainer at every
        phase switch when ``nodewise_update.reset_every_stage`` is true)."""
        if self._nodewise is not None:
            self._nodewise.reset_stats()

    def on_train_epoch_end(self):
        if self._commit is not None:
            sizes = self._commit.assignment_sizes().float()
            self.log("struct/commit_mean_subset", float(sizes.mean()),
                     on_step=False, on_epoch=True)
            self.log_dict(
                {f"struct/commit_subset_{i}": float(sizes[i])
                 for i in range(self._commit.n_nodes)},
                on_step=False, on_epoch=True,
            )
        if self._nodewise is not None:
            nw = self._nodewise
            if nw.n_steps > 0:
                frac = nw.sel_counts.double() / nw.n_steps
                self.log_dict(
                    {f"struct/nodewise_selfrac_{i}": float(frac[i])
                     for i in range(nw.n_nodes)},
                    on_step=False, on_epoch=True,
                )
            nw.reset_epoch_diagnostics()
        if self.hsic_constraint_enabled:
            self._update_hsic_dual()
        if self.acyclicity_constraint_enabled:
            self._update_acyclicity_dual()
        super().on_train_epoch_end()

    # ------------------------------------------------------------------
    # Second-order (DARTS) shadow evidence: hsic_unrolled
    # ------------------------------------------------------------------
    def _lean_pred(self, S, X, overrides=None):
        """One forward's prediction; with ``overrides`` the reconstruction
        params are substituted (``torch.func.functional_call``) instead of
        mutated Ã¢â‚¬â€ the module and its version counters stay untouched."""
        if overrides is None:
            return self.forward(data_source=S, data_intermediate=X)[0]
        from torch.func import functional_call
        return functional_call(
            self, overrides, (), {"data_source": S, "data_intermediate": X},
        )[0]

    def _lean_hsic(self, S, X, overrides=None):
        """HSIC term of ONE grad-enabled forward, using the stashed
        structural pair mask (descendant x LOO, no BKD) and, when latched,
        the phase's frozen bandwidths."""
        pred = self._lean_pred(S, X, overrides)
        x_val = X[:, :, self.val_idx]
        if self.homogeneous_nodes:
            x_val = torch.cat([S[:, :, self.val_idx], x_val], dim=1)
        x_target = torch.nan_to_num(x_val)
        residuals = x_target.squeeze() - pred.squeeze()
        combined = (x_target.squeeze() if self.homogeneous_nodes else
                    torch.cat([S[:, :, self.val_idx], x_target.squeeze()], dim=1))
        if self._hsic_bw_frozen_sigmas is not None:
            sigma, adaptive = self._hsic_bw_frozen_sigmas, False
        else:
            sigma, adaptive = self.hsic_sigma, self.hsic_adaptive_bandwidth
        return hsic_cross_per_pair(
            combined, residuals, sigma=sigma, adaptive_bandwidth=adaptive,
            mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon,
            source_kernel=self.hsic_kernel_source,
            bandwidth_multipliers=self.hsic_bandwidth_multipliers,
            pair_mask=self._last_probe_pair_mask,
        )

    def _lean_recon(self, S, X, overrides=None):
        """MSE of ONE grad-enabled forward (same target layout as _step)."""
        pred = self._lean_pred(S, X, overrides)
        x_val = X[:, :, self.val_idx]
        if self.homogeneous_nodes:
            x_val = torch.cat([S[:, :, self.val_idx], x_val], dim=1)
        x_target = torch.nan_to_num(x_val)
        return torch.nn.functional.mse_loss(pred.squeeze(), x_target.squeeze())

    def _shadow_evidence(self, batch):
        """Shadow gradient for the commit controller: first-order HSIC
        (``hsic``) or the DARTS second-order destination-state gradient
        (``hsic_unrolled``).  Must be called with the main graph alive
        (before the structural backward).  With ``unrolled.every = m > 1``
        the unrolled gradient is computed every m-th call and the
        first-order term on off-steps."""
        if self._commit_source == "hsic_unrolled":
            self._unrolled_step_count += 1
            if self._unrolled_step_count % self._unrolled_every == 0:
                return self._unrolled_shadow_grads(batch)
        return torch.autograd.grad(
            self._last_hsic_reg,
            [t.shadow for t in self._commit.tables],
            retain_graph=True, allow_unused=True,
        )

    def _unrolled_shadow_grads(self, batch):
        """DARTS second-order shadow gradient for the commit controller.

        Thin wrapper over :meth:`_darts_second_order_grads` with the commit
        shadow tensors as targets.
        """
        return self._darts_second_order_grads(
            batch[0], batch[1], [t.shadow for t in self._commit.tables]
        )

    def _darts_second_order_grads(self, S, X, targets):
        """DARTS second-order gradient w.r.t. ``targets``
        (docs/ideas/BILEVEL_CENTROID_COMMIT.md).

            theta_R' = theta_R - eta * grad_{theta_R} L_recon   (virtual refit)
            g_q = grad_shadow HSIC(theta_R')
                - eta * [grad_shadow L_recon(theta_R + eps*v)
                         - grad_shadow L_recon(theta_R - eps*v)] / (2*eps)
            v = grad_{theta_R'} HSIC

        ALL passes (including the theta_R gradient of step 1) use
        ``torch.func.functional_call`` with substituted parameter dicts:
        theta_R is NEVER mutated in place (an in-place swap would bump
        parameter version counters and invalidate the retained main graph for
        the structural backward that follows).  Step 1 differentiates a lean
        pass w.r.t. detached copies rather than the main graph w.r.t. the live
        parameters, because in the structure phase the adaptive controller
        freezes theta_R (``requires_grad_(False)``) and autograd would raise
        "One of the differentiated Tensors does not require grad".
        BKD is off during the lean passes (the probe mask excludes the BKD
        factor) and every lean pass shares one forked RNG seed, so stochastic
        gates/dropout are paired across passes and the global training RNG
        stream is not advanced.

        Args:
            S, X:    batch tensors (fold B when cross-fitting is active).
            targets: tensors to differentiate w.r.t. (commit shadows, or the
                     structural parameters for the bi-level structural step).

        Returns a list aligned with ``targets`` (entries may be ``None``).
        """
        shadows = list(targets)
        recon_params = getattr(self, "_reconstruction_params", None)
        if recon_params is None:
            # No gradient routing (single optimizer): classify on the fly.
            _, recon_params = classify_parameters(self.model, verbose=False)
        recon_ids = {id(p) for p in recon_params}
        base = {n: p for n, p in self.named_parameters()
                if id(p) in recon_ids}
        names = list(base)
        eta = (self._unrolled_inner_lr if self._unrolled_inner_lr is not None
               else float(self.config["training"].get("lr", 1e-3)))
        eps = self._unrolled_fd_eps
        device = base[names[0]].device

        bkd_mods = [m for m in self.model.modules()
                    if hasattr(m, "set_bkd_phase_active")]
        bkd_state = [getattr(m, "_bkd_phase_active", True) for m in bkd_mods]

        def _lean(fn, overrides):
            with torch.random.fork_rng(
                    devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(20240913)
                return fn(S, X, overrides)

        try:
            for m in bkd_mods:
                m.set_bkd_phase_active(False)

            # 1. theta_R gradient of the recon loss.  The MAIN graph cannot
            # be differentiated w.r.t. theta_R here: in the structure phase
            # the phase controller freezes theta_R (requires_grad_(False)),
            # so autograd.grad on the live parameters raises "One of the
            # differentiated Tensors does not require grad".  Differentiate
            # a LEAN pass w.r.t. detached, grad-enabled COPIES substituted
            # via functional_call instead (same mechanism as the virtual /
            # perturbed passes below): the live parameters and their version
            # counters stay untouched, so the retained main graph used by the
            # structural backward that follows remains valid.
            theta = {n: base[n].detach().requires_grad_(True) for n in names}
            g_R = torch.autograd.grad(
                _lean(self._lean_recon, theta),
                [theta[n] for n in names], allow_unused=True)
            g_R = [torch.zeros_like(theta[n]) if g is None else g.detach()
                   for g, n in zip(g_R, names)]

            # 2. Virtual refit: HSIC gradient at the destination state.  The
            # perturbed theta_R' are fresh leaves so v = grad_{theta_R'} HSIC
            # is defined at the virtual point.
            pert = {n: (theta[n] - eta * g).detach().requires_grad_(True)
                    for n, g in zip(names, g_R)}
            g_all = torch.autograd.grad(
                _lean(self._lean_hsic, pert),
                shadows + [pert[n] for n in names], allow_unused=True)
            g_q = list(g_all[:len(shadows)])
            v = [torch.zeros_like(pert[n]) if g is None else g.detach()
                 for g, n in zip(g_all[len(shadows):], names)]

            # 3. Finite-difference mixed Hessian: d/dshadow L_recon along v.
            fd = []
            for sign in (+1.0, -1.0):
                fd_pert = {n: (theta[n] + sign * eps * vi).detach()
                           for n, vi in zip(names, v)}
                g = torch.autograd.grad(_lean(self._lean_recon, fd_pert),
                                        shadows, allow_unused=True)
                fd.append([None if gi is None else gi.detach() for gi in g])

            out = []
            for gq, gp, gm in zip(g_q, fd[0], fd[1]):
                if gq is None:
                    out.append(None)
                    continue
                corr = torch.zeros_like(gq) if gp is None or gm is None \
                    else (gp - gm) * (eta / (2.0 * eps))
                out.append(gq.detach() - corr)
            return out
        finally:
            for m, s in zip(bkd_mods, bkd_state):
                m.set_bkd_phase_active(s)

    # ------------------------------------------------------------------
    # Bi-level (DARTS second-order) structural step
    # ------------------------------------------------------------------
    def _bi_level_step(self, batch) -> None:
        """Structural gradient via the DARTS second-order rule, written
        directly into ``p.grad`` of the structural parameters (the HSIC part
        from :meth:`_darts_second_order_grads` on ``batch`` Ã¢â‚¬â€ fold B when
        cross-fitting is active Ã¢â‚¬â€ plus the non-HSIC structural terms
        differentiated from the live main graph).

        Called from ``training_step`` in place of :meth:`_structural_backward`
        when ``training.structural_grad == 'hsic_unrolled'``.  The gradient
        scaling matches the first-order stream: the structural HSIC term is
        ``(1 - lambda_struct_recon) * lambda_hsic * HSIC`` and ``_lean_hsic``
        returns the raw (unweighted) HSIC, so the unrolled gradient is scaled
        by ``lambda_hsic * (1 - lambda_struct_recon)``.

        The main graph is consumed by differentiating the remaining
        structural terms (L0 / NOTEARS / struct-recon mix / sparsity / query
        norm); the HSIC branch of the graph is intentionally NOT back-
        propagated Ã¢â‚¬â€ its gradient is replaced by the unrolled one Ã¢â‚¬â€ and its
        buffers are released when the next ``_step`` overwrites the stashed
        tensors.
        """
        struct = [p for p in self._structural_params if p.requires_grad]
        if not struct:
            return
        g_hsic = self._darts_second_order_grads(batch[0], batch[1], struct)
        scale = self.lambda_hsic * (1.0 - self.lambda_struct_recon)
        for p, g in zip(struct, g_hsic):
            if g is not None:
                p.grad = (scale * g).to(p.dtype)

        # Non-HSIC structural terms from the live graph (consumes it).
        rest_terms = [
            t for t in (self._last_l0_reg, self._last_acyclic_reg,
                        self._last_struct_rest)
            if t is not None and torch.is_tensor(t) and t.requires_grad
        ]
        if not rest_terms:
            return
        g_rest = torch.autograd.grad(
            sum(rest_terms), struct, retain_graph=False, allow_unused=True
        )
        for p, g in zip(struct, g_rest):
            if g is None:
                continue
            p.grad = g.detach() if p.grad is None else p.grad + g.detach()

    # ------------------------------------------------------------------
    # Bilevel commit gate (Phase 2, docs/ideas/BILEVEL_CENTROID_COMMIT.md)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Bilevel commit gate (Phase 2, docs/ideas/BILEVEL_CENTROID_COMMIT.md)
    # ------------------------------------------------------------------
    def _run_bilevel_gate(self) -> None:
        """Probe and resolve the deferred commit candidates of this step.

        Each pending candidacy is a snapshot (subset fixed at trigger time).
        The probe compares the node's HSIC row after a paired k-step
        reconstruction refit (incumbent vs candidate) on cached validation
        batches; acceptance commits, rejection taboos the candidate subset.
        Without cached val batches the candidacy is DROPPED (not tabooed) -
        the shadow persists, so eligibility re-triggers on the next step.
        """
        from causaliT.training.bilevel_probe import paired_refit_probe
        from causaliT.training.centroid_commit import centroid_of

        cc = self._commit
        pending, cc.pending = cc.pending, []
        if not pending:
            return
        if not self._val_probe_cache:
            logger.info("bilevel gate: %d candidacy(ies) dropped, no cached "
                        "val batches yet", len(pending))
            self.log("struct/gate_deferred", float(len(pending)),
                     on_step=False, on_epoch=True, reduce_fx="sum")
            return
        for entry in pending:
            vec = centroid_of(cc.K, entry["cand"])
            res = paired_refit_probe(
                self,
                {entry["gi"]: vec},
                self._val_probe_cache,
                k_inner=self._gate_k_inner,
                inner_optimizer=self._gate_inner_optimizer,
                inner_lr=self._gate_inner_lr,
                inner_weight_decay=self._gate_inner_wd,
                accept_margin=self._gate_margin,
                seed=self.global_step,
                pair_mask=self._last_probe_pair_mask,
            )
            accepted = res.accepted[entry["gi"]]
            cc.finalize(entry, accepted)
            self.log("struct/gate_accepted", float(accepted),
                     on_step=False, on_epoch=True, reduce_fx="sum")
            self.log("struct/gate_rejected", float(not accepted),
                     on_step=False, on_epoch=True, reduce_fx="sum")
            self.log("struct/gate_delta", res.deltas[entry["gi"]],
                     on_step=False, on_epoch=True)
            # Constant-shift tripwire: a large MSE gain with delta ~ 0 means
            # the move is invisible to HSIC (docs/ideas/BILEVEL_CENTROID_COMMIT.md).
            self.log("struct/gate_dmse", res.mse_deltas[entry["gi"]],
                     on_step=False, on_epoch=True)
        self.log("struct/gate_taboos",
                 float(sum(len(s) for s in cc.taboos)),
                 on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        # Bilevel gate: cache the first max_val_batches val batches (detached)
        # for the paired refit probes.  Filled once; also covers the initial
        # sanity-check pass so probes can fire from epoch 0.
        if (self._gate_enabled
                and len(self._val_probe_cache) < self._gate_max_val):
            self._val_probe_cache.append(
                (batch[0].detach(), batch[1].detach()))
        total_loss, _, _ = self._step(batch, stage="val")
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss, _, _ = self._step(batch, stage="test")
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer(s) via the shared optimizer factory.

        With gradient routing, two optimizers are created (recon first, then
        structural -- matches the training_step unpack).  The structural
        optimizer is configured independently from the reconstruction one via
        ``structural_optimizer``, ``structural_lr``, ``structural_weight_decay``
        and ``structural_optimizer_kwargs`` (e.g. ``{momentum: 0.9,
        nesterov: true}`` for SGD); null/missing values fall back to the
        reconstruction settings (``optimizer``, ``lr``, ``weight_decay``,
        ``optimizer_kwargs``).
        """
        from causaliT.training.optimizer_factory import (
            make_optimizer,
            get_recon_optimizer_config,
            get_structural_optimizer_config,
        )

        tc = self.config["training"]

        if self.use_gradient_routing:
            recon_cfg = get_recon_optimizer_config(tc)
            struct_cfg = get_structural_optimizer_config(tc)
            opt_recon = make_optimizer(self._reconstruction_params, **recon_cfg)
            opt_struct = make_optimizer(self._structural_params, **struct_cfg)
            return [opt_recon, opt_struct]   # recon first -> matches training_step unpack
        else:
            recon_cfg = get_recon_optimizer_config(tc)
            return make_optimizer(list(self.model.parameters()), **recon_cfg)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist the centroid-commit controller state (taboos, counters)."""
        if self._commit is not None:
            checkpoint["centroid_commit"] = self._commit.state_dict()
        if self.hsic_constraint_enabled:
            # Dual state must survive checkpoint resume / staged warm starts:
            # lambda accumulates constraint violation over the WHOLE run.
            checkpoint["hsic_constraint"] = {
                "dual_lambda": self._hsic_dual_lambda,
                "rho": self._hsic_rho,
                "ema": self._hsic_constraint_ema,
                "prev_violation": self._hsic_constraint_prev_violation,
                "tolerance": self.hsic_tol,
            }
        if self.acyclicity_constraint_enabled:
            # Dual state must survive checkpoint resume: lambda and rho
            # accumulate constraint violation over the WHOLE run.
            checkpoint["acyclicity_constraint"] = {
                "dual_lambda": self._acy_dual_lambda,
                "rho": self._acy_rho,
                "ema": self._acy_ema,
                "prev_violation": self._acy_prev_violation,
            }
        if self.mse_max_acyclic_pct > 0.0 or self.l0_max_acyclic_pct > 0.0:
            # The acyclic-reference EMA must survive checkpoint resume so the
            # MSE/L0 caps continue from the same reference.
            checkpoint["mse_acyclic_cap"] = {"ema": self._acyclic_reg_ema}



    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Strip BKD state-dict keys that don't exist in the current model.

        When warm-starting across phases whose ``batch_key_dropout``
        configuration differs (e.g. stage 1 has BKD enabled, stage 2 has
        ``batch_key_dropout=null``), the saved checkpoint may contain keys
        such as::

            model.attention.inner_attention.batch_key_dropout._step_count

        that are absent from the freshly-constructed stage model.  Leaving
        them in the checkpoint causes PyTorch Lightning to raise a
        ``RuntimeError: unexpected key(s) in state_dict`` when it calls
        ``load_state_dict(strict=True)``.

        This hook removes any key whose prefix matches
        ``*.batch_key_dropout.*`` and that is absent from the current
        model's ``state_dict``.  All other unexpected keys are left
        untouched so that genuine architecture mismatches still surface
        as hard errors.

        Note: the symmetric case (stage without BKD warm-starting a stage
        with BKD) produces *missing* keys for ``batch_key_dropout.*``
        entries.  Those are benign Ã¢â‚¬â€ PyTorch initialises missing buffers
        from the module constructor, which is exactly what we want (the
        step counter resets to 0 at each new stage).  However PL strict
        loading would still reject them, so we also drop keys present in
        the *current* model but absent from the checkpoint when they
        belong to ``batch_key_dropout.*``.
        """
        if "state_dict" not in checkpoint:
            return

        # Restore the centroid-commit controller state (taboos, counters).
        if "centroid_commit" in checkpoint and self._commit is not None:
            self._commit.load_state_dict(checkpoint["centroid_commit"])

        # Re-arm the centroid-init latch: if the checkpoint already carries a
        # (trained) X query embedding, mark the one-off init as DONE so a warm-
        # start / resume never overwrites the learned query with the centroid.
        if any(
            k.endswith("query_embed_X.embedding.weight")
            for k in checkpoint["state_dict"]
        ):
            self._query_centroid_init_done = True

        # ----------------------------------------------------------------
        # Oracle combined mask (hard-mask / cheater runs).
        #
        # At training time ``data_dir`` is available and __init__ registers
        # the ``oracle_combined_mask`` buffer (GT DAG mask, optionally
        # corrupted), so it is saved in the checkpoint.  At evaluation time
        # ``load_from_checkpoint`` constructs the model WITHOUT ``data_dir``
        # and the buffer is never registered, so PL's strict load_state_dict
        # raises "Unexpected key(s) in state_dict: oracle_combined_mask"
        # before any predictor-side fallback can rebuild the mask.
        #
        # Register the buffer straight from the checkpoint tensor: the values
        # ARE the training-time mask, and setting ``_hard_masks_loaded``
        # keeps the mask APPLIED in forward -- a hard-masked run must be
        # evaluated WITH its mask, otherwise it silently stops cheating.
        # ----------------------------------------------------------------
        _om_key = "oracle_combined_mask"
        if _om_key in checkpoint["state_dict"] and not hasattr(self, _om_key):
            self.register_buffer(_om_key, checkpoint["state_dict"][_om_key])
            self._hard_masks_loaded = True
            logger.info(
                "on_load_checkpoint: registered '%s' buffer from checkpoint "
                "(shape %s); hard masks remain active.",
                _om_key,
                tuple(checkpoint["state_dict"][_om_key].shape),
            )

        current_keys = set(self.state_dict().keys())

        ckpt_keys = set(checkpoint["state_dict"].keys())

        # Keys in checkpoint that the current model doesn't have
        unexpected = {
            k for k in (ckpt_keys - current_keys)
            if "batch_key_dropout" in k
        }
        # Keys the current model has but the checkpoint doesn't
        # (only for batch_key_dropout Ã¢â‚¬â€ handled by popping from state_dict
        # here so we can fill them from the constructor default below)
        missing_bkd = {
            k for k in (current_keys - ckpt_keys)
            if "batch_key_dropout" in k
        }

        if unexpected:
            for key in unexpected:
                del checkpoint["state_dict"][key]
            import logging
            logging.getLogger(__name__).warning(
                "on_load_checkpoint: removed %d unexpected batch_key_dropout "
                "key(s) from checkpoint state_dict (BKD presence changed "
                "between stages): %s",
                len(unexpected),
                sorted(unexpected),
            )

        if missing_bkd:
            # Fill missing BKD keys from the current model so that
            # strict loading doesn't complain about missing keys either.
            current_sd = self.state_dict()
            for key in missing_bkd:
                checkpoint["state_dict"][key] = current_sd[key]
            import logging
            logging.getLogger(__name__).warning(
                "on_load_checkpoint: filled %d missing batch_key_dropout "
                "key(s) from current model (BKD absent in checkpoint, "
                "present in current stage): %s",
                len(missing_bkd),
                sorted(missing_bkd),
            )

        # Symmetric oracle-mask case: the current model carries the buffer
        # (constructed WITH data_dir) but the checkpoint predates it.  Fill
        # from the current model so strict loading succeeds and the freshly
        # rebuilt mask values are kept.
        missing_om = {
            k for k in (current_keys - ckpt_keys) if k == _om_key
        }
        if missing_om:
            current_sd = self.state_dict()
            for key in missing_om:
                checkpoint["state_dict"][key] = current_sd[key]
            logger.warning(
                "on_load_checkpoint: filled missing '%s' key from the "
                "current model (checkpoint predates the oracle mask buffer).",
                _om_key,
            )

        # ----------------------------------------------------------------
        # Per-node value-MLP dropout migration.
        #
        # ``mlp_per_node_emb`` gained a Dropout/Identity layer after the hidden
        # activation (dropout > 0 -> nn.Dropout, else nn.Identity), shifting
        # the second Linear of every per-node MLP from ``mlps.<i>.2`` to
        # ``mlps.<i>.3``.  Checkpoints written before that change carry the old
        # ``.2.`` keys; Dropout/Identity is parameter-free, so a pure key rename
        # is an exact migration.  Applied only when the current model expects
        # ``.3.`` and the checkpoint still has ``.2.`` (new checkpoints are
        # untouched).
        # ----------------------------------------------------------------
        rename = {}
        for k in ckpt_keys:
            if ".embedding.mlps." not in k:
                continue
            head, _, param = k.rpartition(".")
            if param not in ("weight", "bias") or not head.endswith(".2"):
                continue
            new_k = head[:-2] + ".3." + param
            if new_k in current_keys and k not in current_keys and new_k not in ckpt_keys:
                rename[k] = new_k
        if rename:
            for old_k, new_k in rename.items():
                checkpoint["state_dict"][new_k] = checkpoint["state_dict"].pop(old_k)
            logger.warning(
                "on_load_checkpoint: migrated %d per-node value-MLP key(s) from "
                "mlps.*.2 to mlps.*.3 (checkpoint predates the dropout layer in "
                "mlp_per_node_emb).",
                len(rename),
            )

        # Eval-loading path: the model skipped registering ``oracle_shd_gt``
        # (data_dir=None); drop the checkpoint's copy so strict loading works.
        if "oracle_shd_gt" not in current_keys:
            checkpoint["state_dict"].pop("oracle_shd_gt", None)

        # Restore the HSIC-constraint dual state (absent in checkpoints that
        # predate the feature -> keep the freshly-initialised values).
        if self.hsic_constraint_enabled:
            saved = checkpoint.get("hsic_constraint", None)
            if saved is not None:
                self._hsic_dual_lambda = float(saved.get("dual_lambda", 0.0))
                self._hsic_rho = float(saved.get("rho", self.hsic_rho_init))
                self._hsic_constraint_ema = saved.get("ema", None)
                self._hsic_constraint_prev_violation = saved.get(
                    "prev_violation", None
                )
                # Calibrated tolerances persist (they are re-calibrated at
                # the next structure-phase entry when calibrate_tolerance
                # is on, so this mainly keeps resumed runs consistent).
                self.hsic_tol = float(saved.get("tolerance", self.hsic_tol))

        # Restore the acyclicity-constraint dual state (absent in checkpoints
        # that predate the feature -> keep the freshly-initialised values).
        if self.acyclicity_constraint_enabled:
            saved = checkpoint.get("acyclicity_constraint", None)
            if saved is not None:
                self._acy_dual_lambda = float(saved.get("dual_lambda", 0.0))
                self._acy_rho = float(saved.get("rho", self.acy_rho_init))
                self._acy_ema = saved.get("ema", None)
                self._acy_prev_violation = saved.get("prev_violation", None)

        # Restore the MSE-cap acyclic-reference EMA (absent in checkpoints
        # that predate the feature -> keep the freshly-initialised None).
        if self.mse_max_acyclic_pct > 0.0 or self.l0_max_acyclic_pct > 0.0:
            saved = checkpoint.get("mse_acyclic_cap", None)
            if saved is not None:
                self._acyclic_reg_ema = saved.get("ema", None)

    def on_fit_start(self):
        """
        Phase-level parameter freezing.

        Called by Lightning at the start of each ``trainer.fit()`` call.
        Freezes structural or reconstruction parameters when the corresponding
        flag is set in ``config['training']``.

        Only active when ``use_gradient_routing=True`` (param groups exist).
        When gradient routing is off there are no parameter groups to freeze,
        so both flags are expected to stay ``False`` and the loss weights do
        the gating instead.

        ``requires_grad`` is **not** persisted in checkpoints, so each new
        stage's warm-started model starts fully unfrozen and this hook re-
        applies the correct constraint.
        """
        if self.freeze_structural_params and self.use_gradient_routing:
            for p in self._structural_params:
                p.requires_grad_(False)
            print("  [phase] Structural parameters frozen (requires_grad=False).")
        if self.freeze_reconstruction_params and self.use_gradient_routing:
            for p in self._reconstruction_params:
                p.requires_grad_(False)
            print("  [phase] Reconstruction parameters frozen (requires_grad=False).")

        # Rebuild the interference block mapping so it reflects the current
        # requires_grad state for this stage.
        self._interference_blocks = None

        # Arm the permutation-null tolerance calibration at fit start.  Under
        # the adaptive trainer this is re-armed at every structure-phase
        # entry (the warmup-regime estimate made here is then simply unused:
        # dual ascent is structure-gated there).  Under the STATIC trainer
        # there are no phase switches, so this is the one and only
        # calibration - collected from the first ``calibration_batches``
        # train batches, matching the regime the constraint will train in.
        if (
            self.hsic_constraint_enabled
            and self.hsic_calibrate_tolerance
            and self.hsic_constraint_source == "hsic"
        ):
            self._hsic_null_samples = []
            self._hsic_null_calib_remaining = self.hsic_calib_batches
            self._hsic_null_calib_round += 1
            logger.info(
                "[hsic-constraint] fit-start tolerance calibration armed: "
                "%d batches x %d permutations.",
                self.hsic_calib_batches, self.hsic_calib_perms,
            )

    # ------------------------------------------------------------------
    # Convenience: expose split attention for post-hoc evaluation
    # ------------------------------------------------------------------


    def get_split_attention(
        self,
        data_source: torch.Tensor,
        data_intermediate: torch.Tensor,
    ):
        """
        Run forward and return split SÃ¢â€ â€™X and XÃ¢â€ â€™X attention matrices.

        Works in BOTH node topologies: ``AttentionSelectorLayer.split_attention``
        is shape-aware, so a square homogeneous ``(B, N, N)`` posterior is first
        row-sliced to the X children before the columns are split.

        Returns:
            att_sx: (B, L_X, L_S)  Ã¢â‚¬â€ SÃ¢â€ â€™X learned edges
            att_xx: (B, L_X, L_X)  Ã¢â‚¬â€ XÃ¢â€ â€™X learned edges (diagonal = 0)
        """
        with torch.no_grad():
            _, attention_weights, _ = self.forward(data_source, data_intermediate)
        return self.model.split_attention(attention_weights)

    # -- Homogeneous-mode diagnostics (thin pass-throughs to the layer) -----
    # Kept on the forecaster so notebooks/eval code can reach them without
    # touching ``forecaster.model`` and without re-deriving the L_S / L_X split.

    def split_attention_blocks(self, attention: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """All FOUR sub-blocks of the posterior (see the layer's docstring).

        Returns a dict with ``s_to_x`` / ``x_to_x`` always present, plus
        ``x_to_s`` / ``s_to_s`` which are ``None`` in split mode (there, S nodes
        are never children, so those rows simply do not exist).
        """
        return self.model.split_attention_blocks(attention)

    def source_scores(self, attention: torch.Tensor) -> torch.Tensor:
        """Per-node incoming-edge mass over the ``N`` nodes (LOW Ã¢â€¡â€™ likely a source).

        In homogeneous mode this is the quantity that tells us whether the model
        RE-DISCOVERED the S/X partition it was not given: true exogenous sources
        should end up with (near-)zero incoming attention.
        """
        return self.model.source_scores(attention)

    def get_diagnostic_blocks(
        self,
        data_source: torch.Tensor,
        data_intermediate: torch.Tensor,
    ):
        """Convenience: one forward Ã¢â€ â€™ (all four blocks, per-node source scores)."""
        with torch.no_grad():
            _, attention_weights, _ = self.forward(data_source, data_intermediate)
            blocks = self.model.split_attention_blocks(attention_weights)
            scores = self.model.source_scores(attention_weights)
        return blocks, scores
