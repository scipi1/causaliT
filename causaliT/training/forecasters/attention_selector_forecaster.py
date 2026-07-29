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
- train/val_x_mae/rmse/r2  : Reconstruction metrics
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

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
from os.path import join


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.utils import load_dag_masks, corrupt_dag_masks
from causaliT.utils.hsic_utils import hsic_cross_per_pair
from causaliT.utils.query_norm import collect_query_norm_penalty, query_norm_stats
from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.interference_utils import (
    build_interference_blocks,
    compute_l0_hsic_interference,
)

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

        # Data indices
        self.val_idx = config["data"]["val_idx"]

        self.S_seq_len = config["data"]["S_seq_len"]
        self.X_seq_len = config["data"]["X_seq_len"]

        # ------------------------------------------------------------------
        # Node-topology mode (mirrors AttentionSelectorLayer).
        #   False (default) → SPLIT: only the L_X variables are children; the
        #       posterior is (B, L_X, L_S+L_X) and the target is the X values.
        #   True            → HOMOGENEOUS: the S/X prior is dropped, all
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
        #   * 0.0 → pure HSIC structural stream (original behaviour).
        #   * >0  → re-inject a controlled dose of reconstruction signal into
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
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        self.hsic_mode = config["training"].get("hsic_mode", "biased")
        self.nhsic_epsilon = config["training"].get("nhsic_epsilon", 0.01)
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")

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
        # L0 ↔ HSIC gradient-interference logging (diagnostic).
        # When enabled AND the attention is HardConcreteCrossAttention AND
        # both lambda_l0 > 0 and lambda_hsic > 0, we log the per-block cosine
        # similarity between the L0 gradient and the HSIC gradient.  Negative
        # cosine ⇒ the two objectives push the structural parameters in
        # opposing directions (interference); positive ⇒ aligned.
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
        # mode the cross ``attention_type`` is IGNORED by the architecture — the
        # single square block is built from ``self_attention_type`` — so that is
        # the type whose L0 gate the probe would see.
        self._attention_type = (
            config["model"]["kwargs"].get("self_attention_type", "")
            if self.homogeneous_nodes
            else config["model"]["kwargs"].get("attention_type", "")
        )
        # Cached block → parameter-list mapping (built lazily on first use so
        # it reflects any requires_grad freezing applied in on_fit_start).
        self._interference_blocks: Optional[Dict[str, list]] = None
        # Stash for the two reg tensors so training_step can probe them while
        # the autograd graph is still alive.
        self._last_hsic_reg: Optional[torch.Tensor] = None
        self._last_l0_reg: Optional[torch.Tensor] = None

        # ----------------------------------------------------------------
        # Acyclicity regularization (NOTEARS) — X→X sub-block only
        # Applied to the square (L_X, L_X) portion of the combined score
        # tensor (columns S_seq_len:).  The S→X block is bipartite and
        # inherently acyclic, so NOTEARS is not needed there.
        # Set kappa > 0 to activate; kappa=0.0 is the default (off).
        # ----------------------------------------------------------------
        self.kappa = float(config["training"].get("kappa", 0.0))
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")

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
        # Parameter freezing for ANM alternating stages
        # Set by anm_staged_trainer._build_stage_config per stage.
        # Requires use_gradient_routing=True; otherwise _build_stage_config
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
        # feeds the GT DAG hard mask (combined S→X ‖ X→X) directly as the
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

        # Wrong-DAG oracle controls — same semantics as SingleCausalForecaster.
        # seed in {None, 0} OR both SHDs == 0  →  no corruption.
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

        self.save_hyperparameters(config)

        # Metrics
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()

    # ------------------------------------------------------------------
    # Hard mask loading
    # ------------------------------------------------------------------

    def _load_combined_oracle_mask(self, config: dict, data_dir: str):
        """
        Load GT DAG mask CSVs, optionally corrupt them, and register the
        combined (L_X, L_S+L_X) oracle mask as a Lightning buffer.

        The combined oracle mask concatenates:
            dec_cross  (L_X, L_S)  — S→X GT edges
            dec_self   (L_X, L_X)  — X→X GT edges
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
                f"✓ Oracle masks CORRUPTED "
                f"(seed={int(self.hard_masks_corruption_seed)}, "
                f"cross_shd={self.cross_control_shd}, "
                f"self_shd={self.self_control_shd}, "
                f"preserve_sparsity={self.hard_masks_preserve_sparsity})"
                f" — wrong-DAG oracle."
            )
            for _name, _info in corruption_info.items():
                cyc = _info.get("has_cycles")
                cyc_str = (
                    "N/A" if cyc is None else ("⚠ cycles" if cyc else "✓ acyclic")
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
            # S children — sources have no parents in this dataset family, so
            # those rows are all-zero — and rows L_S..N-1 carry the X children's
            # parents as [dec_cross | dec_self].
            combined = torch.zeros(self.N, self.N, dtype=cross_mask.dtype)
            combined[self.S_seq_len :, : self.S_seq_len] = cross_mask
            combined[self.S_seq_len :, self.S_seq_len :] = self_mask
        else:
            # Split mode: concatenate [S→X part | X→X part] → (L_X, L_S + L_X)
            combined = torch.cat([cross_mask, self_mask], dim=1)

        self.register_buffer("oracle_combined_mask", combined)
        self._hard_masks_loaded = True
        print(
            f"✓ Oracle combined mask built: shape {combined.shape} "
            f"(cross {cross_mask.shape} ‖ self {self_mask.shape}"
            f"{', homogeneous square layout' if self.homogeneous_nodes else ''})"
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
            pred_x:            (B, L_X, 1) predictions — (B, N, 1) when
                               ``homogeneous_nodes=True`` (S nodes are children
                               too and therefore reconstructed as well).
            attention_weights: (B, L_X, L_S + L_X) combined attention matrix —
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
    # Common step
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str = "train"):
        # Unpack — support (S, X) and (S, X, Y)
        S = batch[0]
        X = batch[1]

        x_val = X[:, :, self.val_idx]           # (B, L_X)  ground truth values

        # Homogeneous mode: the model reconstructs ALL N nodes, so the target
        # is cat([S_values, X_values]) → (B, N).  Everything downstream that
        # consumes ``x_target`` / ``pred_x`` (MSE, metrics, HSIC residuals,
        # ANM diagnostics) then operates on N rows automatically.
        if self.homogeneous_nodes:
            x_val = torch.cat([S[:, :, self.val_idx], x_val], dim=1)   # (B, N)

        # Forward — returns (pred_x, attention_weights, aux_dict)
        pred_x, attention_weights, aux = self.forward(S, X)
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
        # (self_attention_type set) this concatenates the S→X cross gate
        # posterior with the direction-aware X→X GatedSelfAttention posterior,
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

        hsic_value = hsic_cross_per_pair(
            combined_source,
            residuals,
            sigma=self.hsic_sigma,
            adaptive_bandwidth=self.hsic_adaptive_bandwidth,
            mode=self.hsic_mode,
            nhsic_epsilon=self.nhsic_epsilon,
            source_kernel=self.hsic_kernel_source,
        )
        hsic_reg = self.lambda_hsic * hsic_value

        # ----------------------------------------------------------------
        # Group-L1 regularization (L2,1 norm on embedding columns)
        # ----------------------------------------------------------------
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_reg = self.lambda_group_l1 * group_l1_loss

        # ----------------------------------------------------------------
        # Acyclicity regularization (NOTEARS) — X→X sub-block only
        # Extract the square (L_X, L_X) directed edge matrix from the
        # combined (L_X, L_S+L_X) score tensor by slicing columns S_seq_len:.
        # The score tensor is 2-D (batch-mean, head-averaged) for single-head
        # CausalCrossAttention.  Multi-head tensors (dim != 2) are skipped.
        # ----------------------------------------------------------------
        # In homogeneous mode the score tensor IS the square (N, N) directed
        # adjacency over all nodes, so NOTEARS applies to the FULL matrix (the
        # column slice would be a meaningless sub-block there).
        if self.kappa > 0.0 and score_tensor is not None and score_tensor.dim() == 2:
            A_cyc = (
                score_tensor                      # (N, N)
                if self.homogeneous_nodes
                else score_tensor[:, self.S_seq_len:]   # (L_X, L_X)
            )
            acyclic_reg = self.kappa * self._notears_acyclicity(A_cyc)
        else:
            acyclic_reg = torch.tensor(0.0, device=X.device)

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
            l0_reg = self.lambda_l0 * l0_penalty
        else:
            l0_reg = torch.tensor(0.0, device=X.device)

        # ----------------------------------------------------------------
        # Total loss
        # ----------------------------------------------------------------
        total_loss = (
            self.lambda_recon * loss_x
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
        # training_step can probe L0 ↔ HSIC gradient interference before the
        # real backward runs.
        self._last_hsic_reg = hsic_reg
        self._last_l0_reg = l0_reg

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"))
        self.log(f"{stage}_score_sparse", score_sparse_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_reg, on_step=False, on_epoch=True)
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

        if effective_dims is not None:
            self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)

        # NOTEARS acyclicity (auto-discovered by eval_training.py via "notears" key)
        self.log(f"{stage}_notears", acyclic_reg, on_step=False, on_epoch=True)

        # L0 penalty (expected number of active edges, non-zero only for
        # HardConcreteCrossAttention; logged as 0.0 for all other attention types)
        self.log(f"{stage}_l0_penalty", l0_penalty, on_step=False, on_epoch=True)
        self.log(f"{stage}_l0_reg", l0_reg, on_step=False, on_epoch=True)

        if stage == "val":
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss, pred_x, X

    # ------------------------------------------------------------------
    # L0 ↔ HSIC gradient-interference diagnostic
    # ------------------------------------------------------------------

    # Attention types that expose a differentiable L0 penalty on the structure
    # gate (aux["l0_penalty"]), for which the L0 ↔ HSIC interference probe is
    # meaningful.  Both drive the Hard-Concrete gate logit off the structural
    # query/key pair, so their gradients flow to the same structural params.
    _INTERFERENCE_ATTENTION_TYPES = (
        "HardConcreteCrossAttention",
        "GatedCrossAttention",
    )

    def _interference_enabled(self) -> bool:
        """Whether the L0 ↔ HSIC interference diagnostic should run."""
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

        # Build the block → parameter mapping lazily.  Rebuilt if it came back
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
            # training — just skip this step.
            logger.warning(
                "L0↔HSIC interference probe skipped (autograd error): %s", exc
            )
            return

        # Skip NaN blocks: a NaN cosine means one objective's gradient is
        # entirely zero in that block (pure reconstruction blocks receive no
        # L0 gradient).  Logging only the non-NaN blocks auto-focuses the
        # metric set on the structural pathway (Q/K + embeddings) where the
        # L0 ↔ HSIC interference actually happens.  We simultaneously collect
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

        # --- Human-readable summary of the L0 ↔ HSIC gradient conflict ---
        # cos < 0 ⇒ the L0 (sparsity) and HSIC (independence) gradients push
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
                "[L0↔HSIC interference] epoch=%d | overall_cos=%s | "
                "conflicting_blocks=%d/%d | per-block: %s",
                int(self.current_epoch),
                overall_str,
                n_conflict,
                n_blocks,
                detail,
            )


    # ------------------------------------------------------------------
    # Acyclicity helper (mirrors SingleCausalForecaster)
    # ------------------------------------------------------------------

    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity penalty h(A) = tr(exp(A ⊙ A)) - d.

        Zero iff A induces a directed acyclic graph (Zheng et al., 2018).
        Applied to the X→X sub-block (square, L_X × L_X) of the combined
        score tensor.  Caller must ensure A is 2-D (shape (L_X, L_X)).
        """
        d = A.shape[-1]
        return torch.trace(torch.matrix_exp(A * A)) - d

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
        """Lazily initialise the free X query embedding at the key centroid.

        Runs at most once, on the first training batch, when
        ``query_centroid_init=True``.  Deferred to the first batch because
        value-modulated key embeddings need real data to define the centroid
        (see AttentionSelectorLayer.init_query_at_key_centroid).
        """
        if not self._query_centroid_init or self._query_centroid_init_done:
            return
        if getattr(self.model, "query_embed_X", None) is None:
            # Nothing to initialise (free_query_embedding disabled); latch off.
            self._query_centroid_init_done = True
            return
        S, X = batch[0], batch[1]
        self.model.init_query_at_key_centroid(S, X)
        self._query_centroid_init_done = True
        logger.info(
            "Initialised X query embedding at the key centroid "
            "(query_centroid_init=True; all queries start from the same point)."
        )

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

            # Diagnostic: probe L0 ↔ HSIC gradient interference on the live
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

            # Backward 2: structural loss (graph consumed)
            self.manual_backward(loss_structural)

            # Restore recon grads on reconstruction params
            for p in self._reconstruction_params:
                if id(p) in _saved_recon_grads:
                    p.grad = _saved_recon_grads[id(p)]

            # Now step both optimizers (graph fully consumed, safe)
            opt_recon.step()
            opt_struct.step()

            return total_loss
        else:
            total_loss, _, _ = self._step(batch, stage="train")
            # Diagnostic: probe interference on the live graph before Lightning
            # runs its automatic backward on total_loss (retain_graph=True in
            # the probe keeps the graph intact for that backward).
            self._maybe_log_interference(batch_idx)
            return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, _, _ = self._step(batch, stage="val")
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss, _, _ = self._step(batch, stage="test")
        return total_loss

    def configure_optimizers(self):
        lr = self.config["training"].get("lr", 1e-3)
        weight_decay = self.config["training"].get("weight_decay", 0.01)
        optimizer_name = self.config["training"].get("optimizer", "adamw").lower()
        # Extra kwargs forwarded to the optimizer constructor, e.g.
        # {momentum: 0.9, nesterov: true} for SGD.
        optimizer_kwargs = self.config["training"].get("optimizer_kwargs", {}) or {}

        def _make_optimizer(params):
            if optimizer_name == "adamw":
                return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "adam":
                return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "sgd":
                return torch.optim.SGD(
                    params, lr=lr, weight_decay=weight_decay, **optimizer_kwargs
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if self.use_gradient_routing:
            opt_recon = _make_optimizer(self._reconstruction_params)
            opt_struct = _make_optimizer(self._structural_params)
            return [opt_recon, opt_struct]   # recon first → matches training_step unpack
        else:
            return _make_optimizer(self.model.parameters())

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Strip BKD state-dict keys that don't exist in the current model.

        When warm-starting across ANM stages whose ``batch_key_dropout``
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
        entries.  Those are benign — PyTorch initialises missing buffers
        from the module constructor, which is exactly what we want (the
        step counter resets to 0 at each new stage).  However PL strict
        loading would still reject them, so we also drop keys present in
        the *current* model but absent from the checkpoint when they
        belong to ``batch_key_dropout.*``.
        """
        if "state_dict" not in checkpoint:
            return

        # Re-arm the centroid-init latch: if the checkpoint already carries a
        # (trained) X query embedding, mark the one-off init as DONE so a warm-
        # start / resume never overwrites the learned query with the centroid.
        if any(
            k.endswith("query_embed_X.embedding.weight")
            for k in checkpoint["state_dict"]
        ):
            self._query_centroid_init_done = True

        current_keys = set(self.state_dict().keys())

        ckpt_keys = set(checkpoint["state_dict"].keys())

        # Keys in checkpoint that the current model doesn't have
        unexpected = {
            k for k in (ckpt_keys - current_keys)
            if "batch_key_dropout" in k
        }
        # Keys the current model has but the checkpoint doesn't
        # (only for batch_key_dropout — handled by popping from state_dict
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

    def on_fit_start(self):
        """
        ANM stage-level parameter freezing.

        Called by Lightning at the start of each ``trainer.fit()`` call.
        Freezes structural or reconstruction parameters when the corresponding
        flag is set by ``anm_staged_trainer._build_stage_config``.

        Only active when ``use_gradient_routing=True`` (param groups exist).
        When gradient routing is off, ``_build_stage_config`` already falls
        back to loss-level gating and leaves both flags ``False``.

        ``requires_grad`` is **not** persisted in checkpoints, so each new
        stage's warm-started model starts fully unfrozen and this hook re-
        applies the correct constraint.
        """
        if self.freeze_structural_params and self.use_gradient_routing:
            for p in self._structural_params:
                p.requires_grad_(False)
            print("  [ANM stage] Structural parameters frozen (requires_grad=False).")
        if self.freeze_reconstruction_params and self.use_gradient_routing:
            for p in self._reconstruction_params:
                p.requires_grad_(False)
            print("  [ANM stage] Reconstruction parameters frozen (requires_grad=False).")

        # Rebuild the interference block mapping so it reflects the current
        # requires_grad state for this stage.
        self._interference_blocks = None

    # ------------------------------------------------------------------
    # Convenience: expose split attention for post-hoc evaluation
    # ------------------------------------------------------------------


    def get_split_attention(
        self,
        data_source: torch.Tensor,
        data_intermediate: torch.Tensor,
    ):
        """
        Run forward and return split S→X and X→X attention matrices.

        Works in BOTH node topologies: ``AttentionSelectorLayer.split_attention``
        is shape-aware, so a square homogeneous ``(B, N, N)`` posterior is first
        row-sliced to the X children before the columns are split.

        Returns:
            att_sx: (B, L_X, L_S)  — S→X learned edges
            att_xx: (B, L_X, L_X)  — X→X learned edges (diagonal = 0)
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
        """Per-node incoming-edge mass over the ``N`` nodes (LOW ⇒ likely a source).

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
        """Convenience: one forward → (all four blocks, per-node source scores)."""
        with torch.no_grad():
            _, attention_weights, _ = self.forward(data_source, data_intermediate)
            blocks = self.model.split_attention_blocks(attention_weights)
            scores = self.model.source_scores(attention_weights)
        return blocks, scores
