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

3. **No NOTEARS** acyclicity term (the block is a bipartite cross-attention,
   not a self-attention over X).  If needed, it can be applied to the
   X→X sub-matrix in a follow-up experiment.

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
from causaliT.utils.hsic_utils import hsic_cross_per_pair
from causaliT.training.gradient_routing import classify_parameters


class AttentionSelectorForecaster(pl.LightningModule):
    """
    Lightning wrapper for AttentionSelectorLayer.

    Args:
        config: Configuration dictionary (data, model, training sections).
    """

    def __init__(self, config: dict):
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
        # Oracle mode (bypasses QK^T, uses hard mask directly)
        # ----------------------------------------------------------------
        self.use_oracle = config["training"].get("use_oracle_attention", False)

        self.save_hyperparameters(config)

        # Metrics
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()

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

        return self.model.forward_with_actual(
            source_tensor=data_source,
            x_blanked=x_blanked,
            x_actual=data_intermediate,
            oracle=self.use_oracle,
        )

    # ------------------------------------------------------------------
    # Common step
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str = "train"):
        # Unpack — support (S, X) and (S, X, Y)
        S = batch[0]
        X = batch[1]

        x_val = X[:, :, self.val_idx]           # (B, L_X)  ground truth values

        # Forward
        pred_x, attention_weights, entropy = self.forward(S, X)
        
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
        # Total loss
        # ----------------------------------------------------------------
        total_loss = (
            self.lambda_recon * loss_x
            + score_sparsity_reg
            + hsic_reg
            + group_l1_reg
        )

        # Store for gradient routing
        self._last_loss_components = {
            "loss_recon": loss_x,
            "loss_structural": hsic_reg + score_sparsity_reg + group_l1_reg,
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

        if stage == "val":
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss, pred_x, X

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
