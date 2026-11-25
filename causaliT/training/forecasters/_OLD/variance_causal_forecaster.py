"""
VarianceCausalForecaster: PyTorch Lightning wrapper for VarianceCausalLayer.

Key differences from NoiseAwareCausalForecaster
------------------------------------------------
- No noise injection: forward pass is always deterministic.
- No sigma_R: variance is fully explained by alpha^2 @ sigma_A^2.
- New optional loss term:  lambda_cov * ResidualCovarianceLoss
      matches empirical residual covariance to Wright's formula:
      Sigma_model = alpha @ diag(sigma_A^2) @ alpha.T
- sigma_A prior regularization kept.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.variance_causal import VarianceCausalLayer
from causaliT.core.modules.noise_layers import GaussianNLLLoss
from causaliT.core.modules.variance_layers import ResidualCovarianceLoss
from causaliT.core.utils import load_dag_masks, corrupt_dag_masks
from causaliT.utils.hsic_utils import (
    hsic_per_token, hsic_per_x_pair,
    hsic_attention_weighted, hsic_cross_per_pair,
)
from causaliT.training.gradient_routing import classify_parameters


class VarianceCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for VarianceCausalLayer.

    Training loss:
        L = Gaussian_NLL(mu, log_var_analytical)
          + lambda_cov  * ResidualCovarianceLoss    [optional]
          + lambda_hsic_cross * HSIC(S, residuals)  [optional]
          + lambda_hsic_self  * HSIC(X, residuals)  [optional]
          + lambda_self_score_sparse * score_sparsity
          + lambda_cross_score_sparse * score_sparsity
          + lambda_noise_prior * || log(sigma_A) - log(prior_sigma_A) ||^2
          + lambda_group_l1  * L2,1(embedding)

    Args:
        config: Full config dict.
        data_dir: Optional data directory for loading hard masks.
    """

    def __init__(self, config: dict, data_dir: str = None):
        super().__init__()

        self.config = config
        self.model = VarianceCausalLayer(**config["model"]["kwargs"])

        # Gaussian NLL loss
        self.nll_loss = GaussianNLLLoss(
            eps=config["training"].get("nll_eps", 1e-6),
            reduction='none',
            full=config["training"].get("nll_full", False)
        )

        # Covariance matching loss
        self.lambda_cov = float(config["training"].get("lambda_cov", 0.0))
        if self.lambda_cov > 0:
            self.cov_loss_fn = ResidualCovarianceLoss(
                normalize=config["training"].get("normalize_cov_loss", False)
            )
        else:
            self.cov_loss_fn = None

        # Data indices
        self.val_idx = config["data"]["val_idx"]

        # Score sparsity
        self.lambda_self_score_sparse = config["training"].get("lambda_self_score_sparse", 0.0)
        self.lambda_cross_score_sparse = config["training"].get("lambda_cross_score_sparse", 0.0)

        # HSIC
        self.lambda_hsic_cross = config["training"].get("lambda_hsic_cross",
                                     config["training"].get("lambda_hsic", 0.0))
        self.lambda_hsic_self = config["training"].get("lambda_hsic_self", 0.0)
        self.use_attention_weighted_hsic = config["training"].get("use_attention_weighted_hsic", False)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        self.hsic_mode = config["training"].get("hsic_mode", "biased")
        self.nhsic_epsilon = config["training"].get("nhsic_epsilon", 0.01)
        self.normalize_hsic_by_loss = config["training"].get("normalize_hsic_by_loss", False)
        self.hsic_cross_mode = config["training"].get("hsic_cross_mode", "averaged")
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")

        # Noise prior (sigma_A only — no sigma_R)
        self.lambda_noise_prior = config["training"].get("lambda_noise_prior", 0.0)
        self.prior_sigma_A = config["training"].get("prior_sigma_A", 0.1)

        # Group L1
        self.lambda_group_l1 = config["training"].get("lambda_group_l1",
                                   config["training"].get("lambda_embed_l1", 0.0))

        # Structural loss recon anchor
        self.lambda_struct_recon = float(config["training"].get("lambda_struct_recon", 0.0))
        if not (0.0 <= self.lambda_struct_recon <= 1.0):
            raise ValueError(f"lambda_struct_recon must be in [0, 1], got {self.lambda_struct_recon}")

        # Gradient norm logging
        self.log_gradient_norms = config["training"].get("log_gradient_norms", False)

        # HSIC annealing
        self.use_hsic_annealing = config["training"].get("use_hsic_annealing", False)
        self.hsic_anneal_epochs = config["training"].get("hsic_anneal_epochs", None)
        self.hsic_lambda_cross_start = config["training"].get("hsic_lambda_cross_start", self.lambda_hsic_cross)
        self.hsic_lambda_cross_end = config["training"].get("hsic_lambda_cross_end", 0.0)
        self.hsic_lambda_self_start = config["training"].get("hsic_lambda_self_start", self.lambda_hsic_self)
        self.hsic_lambda_self_end = config["training"].get("hsic_lambda_self_end", 0.0)

        # Group L1 annealing
        self._group_l1_anneal_start = config["training"].get("lambda_group_l1_anneal_start_value", None)
        self._group_l1_anneal_idle = config["training"].get("lambda_group_l1_anneal_idle_epochs", 0)
        self._group_l1_anneal_transient = config["training"].get("lambda_group_l1_anneal_transient_epochs", 0)
        self._use_group_l1_annealing = (
            self._group_l1_anneal_start is not None
            and self._group_l1_anneal_start != self.lambda_group_l1
        )
        if self._use_group_l1_annealing:
            self._group_l1_final = self.lambda_group_l1
            self.lambda_group_l1 = float(self._group_l1_anneal_start)

        # Gradient routing
        self.use_gradient_routing = config["training"].get("use_gradient_routing", False)
        if self.use_gradient_routing:
            self.automatic_optimization = False
            structural_params, reconstruction_params = classify_parameters(self.model, verbose=True)
            self._structural_params = structural_params
            self._reconstruction_params = reconstruction_params

        # Parameter freezing
        self.freeze_structural_params = bool(config["training"].get("freeze_structural_params", False))
        self.freeze_reconstruction_params = bool(config["training"].get("freeze_reconstruction_params", False))

        # Hard masks
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None
        self.use_oracle_attention = config["training"].get("use_oracle_attention", False)
        if self.use_oracle_attention and not self.use_hard_masks:
            raise ValueError("use_oracle_attention=True requires use_hard_masks=True.")
        self.hard_masks_corruption_seed = config["training"].get("hard_masks_corruption_seed", None)
        self.cross_control_shd = int(config["training"].get("cross_control_shd", 0) or 0)
        self.self_control_shd = int(config["training"].get("self_control_shd", 0) or 0)
        self.hard_masks_preserve_sparsity = bool(config["training"].get("hard_masks_preserve_sparsity", False))
        self.hard_mask_corruption_info: Optional[Dict[str, dict]] = None

        if self.use_hard_masks:
            self._register_hard_mask_placeholders()
        if self.use_hard_masks and data_dir is not None:
            self._load_hard_masks(config, data_dir)

        self.save_hyperparameters(config)

        # Metrics
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()

    # ------------------------------------------------------------------
    # Hard mask helpers (identical to NoiseAwareCausalForecaster)
    # ------------------------------------------------------------------

    def _register_hard_mask_placeholders(self):
        S_len = self.config["data"]["S_seq_len"]
        X_len = self.config["data"]["X_seq_len"]
        self.register_buffer('hard_mask_dec_cross', torch.zeros(X_len, S_len))
        self.register_buffer('hard_mask_dec_self', torch.zeros(X_len, X_len))

    def _load_hard_masks(self, config: dict, data_dir: str):
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified.")
            return
        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)
        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')
        if masks is None:
            print("Warning: No hard masks loaded.")
            return
        corruption_info = None
        seed = self.hard_masks_corruption_seed
        if seed is not None and seed != 0 and (self.cross_control_shd > 0 or self.self_control_shd > 0):
            masks, corruption_info = corrupt_dag_masks(
                masks,
                cross_shd=self.cross_control_shd,
                self_shd=self.self_control_shd,
                seed=seed,
                preserve_sparsity=self.hard_masks_preserve_sparsity,
            )
        self.hard_mask_corruption_info = corruption_info
        if 'dec_cross' in masks:
            self.hard_mask_dec_cross = masks['dec_cross']
        if 'dec_self' in masks:
            self.hard_mask_dec_self = masks['dec_self']
        self._hard_masks_loaded = True

    def _get_hard_masks(self):
        if not self.use_hard_masks or not self._hard_masks_loaded:
            return None
        return {
            'dec_cross': self.hard_mask_dec_cross,
            'dec_self': self.hard_mask_dec_self,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data_source: torch.Tensor,
        data_intermediate: torch.Tensor,
    ):
        """Deterministic forward — no inject_noise flag needed."""
        hard_masks = self._get_hard_masks()

        # Blank value dimension of X
        X_blanked = data_intermediate.clone()
        X_blanked[:, :, self.val_idx] = 0.0

        oracle = self.use_oracle_attention
        if oracle:
            mu, log_var, attention_weights, masks, entropies = self.model(
                source_tensor=data_source,
                intermediate_tensor_blanked=X_blanked,
                hard_masks=hard_masks,
            )
        else:
            mu, log_var, attention_weights, masks, entropies = self.model(
                source_tensor=data_source,
                intermediate_tensor_blanked=X_blanked,
                hard_masks=hard_masks,
            )

        return mu, log_var, attention_weights, masks, entropies

    # ------------------------------------------------------------------
    # _step
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str):
        if len(batch) == 3:
            S, X, _ = batch
        else:
            S, X = batch

        x_val = X[:, :, self.val_idx]

        # Forward (always deterministic)
        mu, log_var, attention_weights, masks, entropies = self.forward(
            data_source=S, data_intermediate=X
        )

        dec_cross_att, dec_self_att = attention_weights
        dec_cross_ent, dec_self_ent = entropies

        # ===== Gaussian NLL =====
        x_target = torch.nan_to_num(x_val)
        nll_per_elem = self.nll_loss(mu.squeeze(), x_target.squeeze(), log_var.squeeze())
        loss_nll = nll_per_elem.mean()

        # ===== Covariance matching loss (optional) =====
        cov_loss = torch.tensor(0.0, device=X.device)
        if self.lambda_cov > 0 and self.cov_loss_fn is not None:
            residuals = (x_target - mu.squeeze(-1)).detach()
            alpha_last = dec_self_att[-1]
            sigma_A = self.model.intrinsic_noise.sigma_A
            cov_loss = self.lambda_cov * self.cov_loss_fn(residuals, alpha_last, sigma_A)

        # ===== Score sparsity =====
        n_layers = len(self.model.decoder.layers)
        total_self_score_sparse = torch.tensor(0.0, device=X.device)
        total_cross_score_sparse = torch.tensor(0.0, device=X.device)
        self_mode_used = "entropy"
        cross_mode_used = "entropy"

        def _compute_score_sparsity(inner_attention, entropy_value, device):
            score_tensor = getattr(inner_attention, 'score_tensor_for_sparsity', None)
            if score_tensor is not None:
                return score_tensor.abs().mean(), "l1"
            return entropy_value, "entropy"

        for layer_idx in range(n_layers):
            layer = self.model.decoder.layers[layer_idx]
            dec_self_inner = layer.global_self_attention.inner_attention
            dec_cross_inner = layer.global_cross_attention.inner_attention
            layer_self_ent = dec_self_ent[layer_idx].mean()
            layer_cross_ent = dec_cross_ent[layer_idx].mean()
            layer_self_sparse, self_mode_used = _compute_score_sparsity(
                dec_self_inner, layer_self_ent, X.device)
            layer_cross_sparse, cross_mode_used = _compute_score_sparsity(
                dec_cross_inner, layer_cross_ent, X.device)
            total_self_score_sparse = total_self_score_sparse + layer_self_sparse
            total_cross_score_sparse = total_cross_score_sparse + layer_cross_sparse

        avg_self_score_sparse = total_self_score_sparse / n_layers
        avg_cross_score_sparse = total_cross_score_sparse / n_layers
        score_sparsity_regularizer = (
            self.lambda_self_score_sparse * avg_self_score_sparse +
            self.lambda_cross_score_sparse * avg_cross_score_sparse
        )

        # ===== HSIC =====
        hsic_regularizer = 0.0
        hsic_cross_value = None
        hsic_self_value = None
        residuals_per_x = x_target.squeeze() - mu.squeeze()

        if self.normalize_hsic_by_loss:
            hsic_loss_scale = torch.abs(loss_nll).detach()
        else:
            hsic_loss_scale = 1.0

        if self.lambda_hsic_cross > 0:
            s_values = S[:, :, self.val_idx]
            if self.use_attention_weighted_hsic:
                att_cross_mean = dec_cross_att[0].mean(dim=0)
                hsic_cross_value = hsic_attention_weighted(
                    source_values=s_values, residuals=residuals_per_x,
                    attention_weights=att_cross_mean, sigma=self.hsic_sigma,
                    exclude_diagonal=False, adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon,
                    source_kernel=self.hsic_kernel_source)
            elif self.hsic_cross_mode == "per_variable" and residuals_per_x.dim() > 1:
                hsic_cross_value = hsic_cross_per_pair(
                    s_values, residuals_per_x, sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth, mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon, source_kernel=self.hsic_kernel_source)
            else:
                mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
                hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth, mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon, source_kernel=self.hsic_kernel_source)
            hsic_regularizer = hsic_regularizer + self.lambda_hsic_cross * hsic_loss_scale * hsic_cross_value

        if self.lambda_hsic_self > 0 and residuals_per_x.dim() > 1:
            x_values_for_hsic = x_target.squeeze()
            if self.use_attention_weighted_hsic:
                att_self_mean = dec_self_att[0].mean(dim=0)
                hsic_self_value = hsic_attention_weighted(
                    source_values=x_values_for_hsic, residuals=residuals_per_x,
                    attention_weights=att_self_mean, sigma=self.hsic_sigma,
                    exclude_diagonal=True, adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon)
            else:
                hsic_self_value = hsic_per_x_pair(x_values_for_hsic, residuals_per_x,
                    sigma=self.hsic_sigma, adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon)
            hsic_regularizer = hsic_regularizer + self.lambda_hsic_self * hsic_loss_scale * hsic_self_value

        # Always compute HSIC for logging
        if hsic_cross_value is None:
            s_values = S[:, :, self.val_idx]
            if self.hsic_cross_mode == "per_variable" and residuals_per_x.dim() > 1:
                hsic_cross_value = hsic_cross_per_pair(
                    s_values, residuals_per_x, sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth, mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon, source_kernel=self.hsic_kernel_source)
            else:
                mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
                hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth, mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon, source_kernel=self.hsic_kernel_source)
        if hsic_self_value is None and residuals_per_x.dim() > 1:
            hsic_self_value = hsic_per_x_pair(x_target.squeeze(), residuals_per_x,
                sigma=self.hsic_sigma, adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon)

        # ===== Noise prior (single σ per node) =====
        noise_prior_regularizer = 0.0
        if self.lambda_noise_prior > 0:
            sigma = self.model.intrinsic_noise.sigma
            log_ratio = torch.log(sigma) - torch.log(
                torch.tensor(self.prior_sigma_A, device=sigma.device))
            noise_prior_regularizer = self.lambda_noise_prior * (log_ratio ** 2).mean()

        # ===== Group L1 =====
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_regularizer = self.lambda_group_l1 * group_l1_loss

        # ===== Total loss =====
        total_loss = (
            loss_nll
            + cov_loss
            + score_sparsity_regularizer
            + hsic_regularizer
            + noise_prior_regularizer
            + group_l1_regularizer
        )

        alpha = self.lambda_struct_recon
        loss_recon_for_struct = loss_nll + noise_prior_regularizer
        self._last_loss_components = {
            "loss_recon": loss_recon_for_struct,
            "loss_structural": (
                (1.0 - alpha) * hsic_regularizer
                + alpha * loss_recon_for_struct
                + cov_loss
                + score_sparsity_regularizer
                + group_l1_regularizer
            ),
        }

        # ===== Logging =====
        self.log(f"{stage}_nll", loss_nll, on_step=False, on_epoch=True, prog_bar=(stage == "val"))

        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(mu.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val" and name == "mae"))

        var = torch.exp(log_var)
        self.log(f"{stage}_pred_var_mean", var.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_pred_var_std", var.std(), on_step=False, on_epoch=True)

        sigma = self.model.intrinsic_noise.sigma
        self.log(f"{stage}_sigma_mean", sigma.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_sigma_std", sigma.std(), on_step=False, on_epoch=True)

        if self.lambda_cov > 0:
            self.log(f"{stage}_cov_loss", cov_loss, on_step=False, on_epoch=True)

        if hsic_cross_value is not None:
            self.log(f"{stage}_hsic_cross", hsic_cross_value.mean() if hasattr(hsic_cross_value, 'mean') else hsic_cross_value,
                     on_step=False, on_epoch=True)
        if hsic_self_value is not None:
            self.log(f"{stage}_hsic_self", hsic_self_value, on_step=False, on_epoch=True)

        self.log(f"{stage}_score_sparse_self", avg_self_score_sparse, on_step=False, on_epoch=True)
        self.log(f"{stage}_score_sparse_cross", avg_cross_score_sparse, on_step=False, on_epoch=True)

        return total_loss, mu, log_var, X

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.use_gradient_routing:
            opt_recon, opt_struct = self.optimizers()
            loss, _, _, _ = self._step(batch=batch, stage="train")
            loss_recon = self._last_loss_components["loss_recon"]
            loss_structural = self._last_loss_components["loss_structural"]
            total_loss = loss_recon + loss_structural

            _saved_recon_grads = {}
            self.manual_backward(loss_recon, retain_graph=True)
            for p in self._reconstruction_params:
                if p.grad is not None:
                    _saved_recon_grads[id(p)] = p.grad.clone()
            self.zero_grad()
            self.manual_backward(loss_structural)
            for p in self._reconstruction_params:
                if id(p) in _saved_recon_grads:
                    p.grad = _saved_recon_grads[id(p)]
            # Manual gradient clipping (Trainer-level clip is bypassed when
            # automatic_optimization=False).
            clip_val = self.config["training"].get("gradient_clip_val", None)
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self._reconstruction_params, clip_val)
                torch.nn.utils.clip_grad_norm_(self._structural_params, clip_val)
            opt_recon.step()
            opt_struct.step()
            self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            loss, _, _, _ = self._step(batch=batch, stage="train")
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _, _ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        from causaliT.training.optimizer_factory import (
            make_optimizer, make_scheduler,
            get_recon_optimizer_config, get_structural_optimizer_config,
            get_structural_scheduler_config, get_gradient_noise_config,
        )
        tc = self.config["training"]
        max_epochs = tc.get("max_epochs", 1000)

        if self.use_gradient_routing:
            recon_cfg = get_recon_optimizer_config(tc)
            struct_cfg = get_structural_optimizer_config(tc)
            opt_recon = make_optimizer(self._reconstruction_params, **recon_cfg)
            opt_struct = make_optimizer(self._structural_params, **struct_cfg)

            noise_cfg = get_gradient_noise_config(tc)
            self._structural_grad_noise_std = noise_cfg["noise_std"]
            self._structural_grad_noise_decay = noise_cfg["noise_decay"]

            sched_cfg = get_structural_scheduler_config(tc)
            struct_scheduler = make_scheduler(opt_struct, **sched_cfg, max_epochs=max_epochs)

            if struct_scheduler is not None:
                return (
                    [opt_recon, opt_struct],
                    [
                        {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt_recon, lr_lambda=lambda e: 1.0)},
                        {"scheduler": struct_scheduler},
                    ],
                )
            return [opt_recon, opt_struct]
        else:
            recon_cfg = get_recon_optimizer_config(tc)
            optimizer = make_optimizer(self.parameters(), **recon_cfg)
            if tc.get("use_scheduler", False):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
            return optimizer

    def _compute_group_l1(self) -> tuple:
        device = next(self.model.embedding_X.parameters()).device
        l21_norm = torch.tensor(0.0, device=device)
        total_dims = 0
        active_dims = 0
        threshold = 1e-3
        for p in self.model.embedding_X.parameters():
            if p.requires_grad and p.dim() >= 2:
                W = p
                l2_per_col = W.norm(p=2, dim=0)
                l21_norm = l21_norm + l2_per_col.sum()
                total_dims += l2_per_col.numel()
                active_dims += (l2_per_col > threshold).sum().item()
        if total_dims > 0:
            l21_norm = l21_norm / total_dims
        effective_dims = torch.tensor(float(active_dims), device=device)
        return l21_norm, effective_dims

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------

    def predict(
        self,
        S: torch.Tensor,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, std) — fully deterministic."""
        self.eval()
        with torch.no_grad():
            mu, log_var, _, _, _ = self.forward(data_source=S, data_intermediate=X)
            std = torch.exp(0.5 * log_var)
        return mu, std

    def predict_with_intervals(
        self,
        S: torch.Tensor,
        X: torch.Tensor,
        confidence: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, lower, upper) confidence intervals."""
        import scipy.stats
        mu, std = self.predict(S, X)
        z = scipy.stats.norm.ppf((1 + confidence) / 2)
        return mu, mu - z * std, mu + z * std

    def get_noise_parameters(self) -> Dict[str, torch.Tensor]:
        return self.model.get_noise_parameters()

    def get_sigma_model(
        self, S: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """Return (L, L) Wright covariance matrix."""
        return self.model.get_sigma_model(S, X)
