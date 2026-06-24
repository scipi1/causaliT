"""
AttentionSelectorForecaster: PyTorch Lightning wrapper for AttentionSelectorLayer.

Research objective
==================
Test whether a single cross-attention block — with X queries (blanked) and
[S_actual, X_actual] as keys/values — can recover causal parent sets from
observational data when trained with MSE reconstruction + HSIC independence
regularization + score sparsity.

Design differences from SingleCausalForecaster
===============================================
1. **Single combined attention** (no separate cross/self blocks).
   The attention matrix has shape (B, L_X, L_S + L_X):
     - columns 0 .. L_S-1   → learned S→X edges
     - columns L_S .. end   → learned X→X edges (diagonal = 0 by mask)

2. **Unified HSIC over combined [S, X] source**.
   `source = cat([S_values, X_values], dim=1)` is passed to
   `hsic_cross_per_pair`, which computes HSIC(source_j, res_i) for all
   (i, j) pairs in one call.  No lambda weighting between S and X parts:
   the combined loss naturally penalizes dependence from any source.

3. **NOTEARS acyclicity** (``training.kappa``) — applied to the **X→X
   sub-block** of the combined score tensor (columns ``S_seq_len:``),
   which is a square ``(L_X, L_X)`` directed edge matrix.  The S→X
   block is bipartite and inherently acyclic, so no NOTEARS term is
   added there.  With ``use_gradient_routing=True`` the NOTEARS penalty
   rides on the structural pathway (same as HSIC), updating Q/K
   projections and structural embeddings.

4. **Gradient routing** works unchanged: query_projection and key_projection
   are structural params; value_projection, out_projection, FFN, forecaster
   are reconstruction params.  The classify_parameters() function identifies
   them by name without any modification.

Logged metrics
==============
- train/val_loss_x         : MSE reconstruction loss
- train/val_x_mae/rmse/r2  : Reconstruction metrics
- train/val_score_sparse   : L1 sparsity on attention weights
- train/val_hsic           : HSIC regularization value
- train/val_hsic_reg       : Weighted HSIC regularization term
- train/val_group_l1       : Group-L1 embedding regularization
- train/val_notears        : NOTEARS acyclicity penalty on X→X sub-block

Attention splitting for evaluation
====================================
After training, use model.split_attention(A) to get:
    att_sx  (B, L_X, L_S)  — S→X attention (compare to S→X ground truth)
    att_xx  (B, L_X, L_X)  — X→X attention (compare to X→X ground truth)
Then threshold and compute SHD.
"""

import json
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
from causaliT.training.gradient_routing import classify_parameters


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

        # Data indices
        self.val_idx = config["data"]["val_idx"]
        self.S_seq_len = config["data"]["S_seq_len"]
        self.X_seq_len = config["data"]["X_seq_len"]

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

        # Concatenate: [S→X part | X→X part] → (L_X, L_S + L_X)
        combined = torch.cat([cross_mask, self_mask], dim=1)
        self.register_buffer("oracle_combined_mask", combined)
        self._hard_masks_loaded = True
        print(
            f"✓ Oracle combined mask built: shape {combined.shape} "
            f"(cross {cross_mask.shape} ‖ self {self_mask.shape})"
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
            pred_x:            (B, L_X, 1) predictions.
            attention_weights: (B, L_X, L_S + L_X) combined attention matrix.
            entropy:           Attention entropy.
        """
        # Blank value column for the query path
        x_blanked = data_intermediate.clone()
        x_blanked[:, :, self.val_idx] = 0.0

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

        s_values = S[:, :, self.val_idx]          # (B, L_S)
        x_values = x_target.squeeze()             # (B, L_X)

        # Concatenate all potential parent values: [S_1,...,S_{L_S}, X_1,...,X_{L_X}]
        combined_source = torch.cat([s_values, x_values], dim=1)   # (B, L_S + L_X)

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
        if self.kappa > 0.0 and score_tensor is not None and score_tensor.dim() == 2:
            A_xx = score_tensor[:, self.S_seq_len:]   # (L_X, L_X)
            acyclic_reg = self.kappa * self._notears_acyclicity(A_xx)
        else:
            acyclic_reg = torch.tensor(0.0, device=X.device)

        # ----------------------------------------------------------------
        # L0 regularization (non-zero only for HardConcreteCrossAttention)
        # l0_penalty is the expected number of active edges = sum P(z_ij > 0)
        # ----------------------------------------------------------------
        if self.lambda_l0 > 0.0 and l0_penalty is not None:
            l0_reg = self.lambda_l0 * l0_penalty
        else:
            l0_reg = torch.tensor(0.0, device=X.device)
            l0_penalty = torch.tensor(0.0, device=X.device)

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
        self._last_loss_components = {
            "loss_recon": loss_x,
            "loss_structural": hsic_reg + score_sparsity_reg + group_l1_reg + acyclic_reg + l0_reg,
        }

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"))
        self.log(f"{stage}_score_sparse", score_sparse_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_reg, on_step=False, on_epoch=True)
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)

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

    def training_step(self, batch, batch_idx):
        if self.use_gradient_routing:
            # --- Manual optimization with dual backward ---
            # Both backward passes must complete BEFORE any optimizer step,
            # otherwise in-place parameter updates invalidate the computation graph.
            opt_recon, opt_struct = self.optimizers()

            # Single forward pass + loss computation (shared)
            total_loss, _, _ = self._step(batch=batch, stage="train")
            loss_recon = self._last_loss_components["loss_recon"]
            loss_structural = self._last_loss_components["loss_structural"]

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

        def _make_optimizer(params):
            if optimizer_name == "adamw":
                return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "adam":
                return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if self.use_gradient_routing:
            opt_recon = _make_optimizer(self._reconstruction_params)
            opt_struct = _make_optimizer(self._structural_params)
            return [opt_recon, opt_struct]   # recon first → matches training_step unpack
        else:
            return _make_optimizer(self.model.parameters())

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

        Returns:
            att_sx: (B, L_X, L_S)  — S→X learned edges
            att_xx: (B, L_X, L_X)  — X→X learned edges (diagonal = 0)
        """
        with torch.no_grad():
            _, attention_weights, _ = self.forward(data_source, data_intermediate)
        return self.model.split_attention(attention_weights)
