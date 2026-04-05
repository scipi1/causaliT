"""
NoiseAwareCausalForecaster: PyTorch Lightning wrapper for noise-aware causal model.

This forecaster handles training, validation, and testing for the noise-aware
architecture with Gaussian NLL loss, enabling uncertainty quantification.

Key Features:
- Gaussian Negative Log-Likelihood loss for probabilistic training
- Ambient noise (σ_A) and reading noise (σ_R) are learned parameters
- Outputs full predictive distribution at inference time
- All regularizers from SingleCausalForecaster are supported

Loss Function:
    L = (x - μ)² / (2τ²) + log(τ) + regularizers

where τ² = σ_R² (reading noise variance) is learned per-node.

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
"""

from typing import Any, Dict, Optional, Tuple
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.noise_aware import NoiseAwareSingleCausalLayer
from causaliT.core.modules.noise_layers import GaussianNLLLoss
from causaliT.core.utils import load_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token, hsic_per_x_pair, hsic_attention_weighted
from causaliT.core.modules.extra_layers import dag_decisiveness_loss, dag_temperature_loss


class NoiseAwareCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for NoiseAwareSingleCausalLayer transformer model.
    
    This forecaster manages training for a noise-aware causal relationship: S → X
    with explicit modeling of ambient and reading noise.
    
    Features:
    - Gaussian NLL loss for uncertainty-aware training
    - Per-node learnable noise parameters (σ_A, σ_R)
    - Entropy and acyclicity regularization support
    - Hard mask support for enforcing ground-truth DAG structure
    - Full predictive distribution output at inference
    
    Training Objective:
        L_NLL = (x - μ)² / (2τ²) + log(τ)
        L_total = L_NLL + λ_entropy * H + λ_acyclic * R + ...
    
    The log(τ) term naturally penalizes unnecessarily large variance,
    preventing the model from explaining everything as noise.
    
    Args:
        config: Configuration dictionary containing model, training, and data settings
        data_dir: Optional data directory for loading hard masks
    """
    
    def __init__(self, config: dict, data_dir: str = None):
        super().__init__()
        
        self.config = config
        self.model = NoiseAwareSingleCausalLayer(**config["model"]["kwargs"])
        
        # Gaussian NLL loss
        self.nll_loss = GaussianNLLLoss(
            eps=config["training"].get("nll_eps", 1e-6),
            reduction='none',  # Compute per-element for logging
            full=config["training"].get("nll_full", False)
        )
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # Regularizers (same as SingleCausalForecaster)
        self.kappa = config["training"].get("kappa", 0)  # Acyclicity regularization
        
        # DAG Sparsity regularization - L1 penalty on edge probabilities (phi)
        self.lambda_sparse = config["training"].get("lambda_sparse", 0)
        self.lambda_sparse_cross = config["training"].get("lambda_sparse_cross", None)
        if self.lambda_sparse_cross is None:
            self.lambda_sparse_cross = self.lambda_sparse
        
        # =====================================================================
        # UNIFIED SCORE SPARSITY REGULARIZATION
        # =====================================================================
        self.lambda_self_score_sparse = config["training"].get("lambda_self_score_sparse", 0.0)
        self.lambda_cross_score_sparse = config["training"].get("lambda_cross_score_sparse", 0.0)
        self.self_sparsity_regularizer = config["training"].get("self_sparsity_regularizer", "l1")
        self.cross_sparsity_regularizer = config["training"].get("cross_sparsity_regularizer", "entropy")
        
        # Track if fallback was triggered (for warning once)
        self._self_sparsity_fallback_warned = False
        self._cross_sparsity_fallback_warned = False
        
        # =====================================================================
        # HSIC REGULARIZATION
        # =====================================================================
        self.lambda_hsic_cross = config["training"].get("lambda_hsic_cross", 
                                     config["training"].get("lambda_hsic", 0.0))
        self.lambda_hsic_self = config["training"].get("lambda_hsic_self", 0.0)
        self.use_attention_weighted_hsic = config["training"].get("use_attention_weighted_hsic", False)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.normalize_hsic_by_loss = config["training"].get("normalize_hsic_by_loss", False)
        
        # KL divergence prior regularization
        self.lambda_kl = config["training"].get("lambda_kl", 1.0)
        self.adaptive_z_scaling = config["training"].get("adaptive_z_scaling", True)
        
        # DAG decisiveness regularization
        self.lambda_decisive = config["training"].get("lambda_decisive", 0)
        self.lambda_decisive_cross = config["training"].get("lambda_decisive_cross", None)
        self.lambda_tau = config["training"].get("lambda_tau", 0)
        self.target_tau = config["training"].get("target_tau", 0.1)
        if self.lambda_decisive_cross is None:
            self.lambda_decisive_cross = self.lambda_decisive
        
        # Noise-specific regularization
        self.lambda_noise_prior = config["training"].get("lambda_noise_prior", 0.0)
        self.prior_sigma_A = config["training"].get("prior_sigma_A", 0.01)
        self.prior_sigma_R = config["training"].get("prior_sigma_R", 0.05)
        
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 
                                   config["training"].get("lambda_embed_l1", 0.0))
        
        # GRADIENT NORM LOGGING (for calibration stage) - only config-controlled logging option
        self.log_gradient_norms = config["training"].get("log_gradient_norms", False)
        
        # =====================================================================
        # ANNEALING CONFIGURATION
        # =====================================================================
        
        # 1. Gumbel-Softmax Temperature Annealing (tau_gs)
        self.use_tau_gs_annealing = config["training"].get("use_tau_gs_annealing", False)
        self.tau_gs_start = config["training"].get("tau_gs_start", 2.0)
        self.tau_gs_end = config["training"].get("tau_gs_end", 0.2)
        self.tau_gs_anneal_epochs = config["training"].get("tau_gs_anneal_epochs", None)
        
        # 2. Toeplitz Activation Temperature Annealing (tau_gate, tau_dir)
        self.use_tau_act_annealing = config["training"].get("use_tau_act_annealing", False)
        self.tau_gate_start = config["training"].get("tau_gate_start", 1.0)
        self.tau_gate_end = config["training"].get("tau_gate_end", 0.2)
        self.tau_dir_start = config["training"].get("tau_dir_start", 0.5)
        self.tau_dir_end = config["training"].get("tau_dir_end", 0.1)
        self.tau_act_anneal_epochs = config["training"].get("tau_act_anneal_epochs", None)
        
        # 3. HSIC Annealing - independent annealing for cross and self
        # Each HSIC term has its own start/end values for full flexibility
        self.use_hsic_annealing = config["training"].get("use_hsic_annealing", False)
        self.hsic_anneal_epochs = config["training"].get("hsic_anneal_epochs", None)
        
        # Separate start/end values for cross-attention HSIC
        # Default: start from current lambda_hsic_cross, anneal to 0
        self.hsic_lambda_cross_start = config["training"].get("hsic_lambda_cross_start", self.lambda_hsic_cross)
        self.hsic_lambda_cross_end = config["training"].get("hsic_lambda_cross_end", 0.0)
        
        # Separate start/end values for self-attention HSIC
        # Default: start from current lambda_hsic_self, anneal to 0
        self.hsic_lambda_self_start = config["training"].get("hsic_lambda_self_start", self.lambda_hsic_self)
        self.hsic_lambda_self_end = config["training"].get("hsic_lambda_self_end", 0.0)
        
        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None
        
        if self.use_hard_masks:
            self._register_hard_mask_placeholders()
        
        if self.use_hard_masks and data_dir is not None:
            self._load_hard_masks(config, data_dir)
        
        self.save_hyperparameters(config)
        
        # Metrics for X reconstruction
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()
    
    def _register_hard_mask_placeholders(self):
        """Register placeholder buffers for hard masks."""
        S_len = self.config["data"]["S_seq_len"]
        X_len = self.config["data"]["X_seq_len"]
        
        self.register_buffer('hard_mask_dec_cross', torch.zeros(X_len, S_len))
        self.register_buffer('hard_mask_dec_self', torch.zeros(X_len, X_len))
    
    def _load_hard_masks(self, config: dict, data_dir: str):
        """Load hard masks from data directory based on config."""
        mask_files = config["training"].get("hard_mask_files", None)
        
        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified in config.")
            return
        
        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)
        
        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')
        
        if masks is not None:
            self._hard_masks = masks
            self._hard_masks_loaded = True
            
            for name, mask in masks.items():
                self.register_buffer(f'hard_mask_{name}', mask)
            
            print(f"✓ Hard masks loaded and registered for training.")
        else:
            print("Warning: No hard masks were loaded.")
    
    def get_hard_masks(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get hard masks dictionary, retrieving from buffers."""
        if not self.use_hard_masks:
            return None
        
        masks = {}
        for name in ['dec_cross', 'dec_self']:
            buffer_name = f'hard_mask_{name}'
            if hasattr(self, buffer_name):
                masks[name] = getattr(self, buffer_name)
        
        return masks if masks else None
    
    def forward(
        self, 
        data_source: torch.Tensor, 
        data_intermediate: torch.Tensor,
        disable_hard_masks: bool = False,
        inject_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any]:
        """
        Forward pass through the model.
        
        Args:
            data_source: Source nodes (S)
            data_intermediate: Intermediate variables (X)
            disable_hard_masks: If True, disables hard masks even if model was trained with them.
            inject_noise: If True, inject ambient noise (training mode). If False, deterministic.
            
        Returns:
            mu: Predicted mean
            log_var: Predicted log-variance
            attention_weights: Attention weights from decoder
            masks: Masks for S, X
            entropies: Attention entropies from decoder
        """
        # Prepare intermediate input (blank X values)
        x_blanked = data_intermediate.clone()
        x_blanked[:, :, self.val_idx] = 0.0
        
        # Determine whether to use hard masks
        apply_hard_masks = self.use_hard_masks and not disable_hard_masks
        hard_masks = self.get_hard_masks() if apply_hard_masks else None
        
        # Model forward pass returns distribution parameters
        mu, log_var, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=hard_masks,
            inject_noise=inject_noise,
        )
        
        return mu, log_var, attention_weights, masks, entropies
    
    def _step(self, batch, stage: str = None):
        """
        Common step logic for train/val/test with Gaussian NLL loss.
        
        Args:
            batch: Tuple of (S, X) or (S, X, Y) tensors - Y is ignored if present
            stage: One of "train", "val", or "test"
            
        Returns:
            total_loss: Total loss including NLL and regularizers
            mu: Predicted mean
            log_var: Predicted log-variance
            X: Actual X values
        """
        # Unpack batch - handle both 2-element (S, X) and 3-element (S, X, Y) batches
        if len(batch) == 3:
            S, X, Y = batch  # Y is unused but captured for compatibility
        else:
            S, X = batch
            Y = None  # No target data
        
        # Extract actual values for loss computation
        x_val = X[:, :, self.val_idx]
        
        # Forward pass - inject noise only during training
        inject_noise = (stage == "train")
        mu, log_var, attention_weights, masks, entropies = self.forward(
            data_source=S,
            data_intermediate=X,
            inject_noise=inject_noise
        )
        
        # Unpack attention weights and entropies
        dec_cross_att, dec_self_att = attention_weights
        dec_cross_ent, dec_self_ent = entropies
        
        # Compute entropy values (always needed for potential sparsity regularization)
        dec_cross_ent_batch = torch.concat(dec_cross_ent, dim=0).mean()
        dec_self_ent_batch = torch.concat(dec_self_ent, dim=0).mean()
        
        # Get learned DAG parameters for acyclicity and prior regularization
        dec_self_inner = self.model.decoder.layers[0].global_self_attention.inner_attention
        dec_cross_inner = self.model.decoder.layers[0].global_cross_attention.inner_attention
        
        dec_self_phi = getattr(dec_self_inner, 'phi', None)
        dec_cross_phi = getattr(dec_cross_inner, 'phi', None)
        
        dec_self_runav_mean = getattr(dec_self_inner, 'runav_att_mean', None)
        dec_self_runav_snr = getattr(dec_self_inner, 'runav_att_snr', None)
        dec_cross_runav_mean = getattr(dec_cross_inner, 'runav_att_mean', None)
        dec_cross_runav_snr = getattr(dec_cross_inner, 'runav_att_snr', None)
        
        # =====================================================================
        # GAUSSIAN NLL LOSS (main loss)
        # L = (x - μ)² / (2τ²) + log(τ)
        # =====================================================================
        
        x_target = torch.nan_to_num(x_val)
        nll_per_elem = self.nll_loss(mu.squeeze(), x_target.squeeze(), log_var.squeeze())
        loss_nll = nll_per_elem.mean()
        
        # =====================================================================
        # REGULARIZERS (same as SingleCausalForecaster)
        # =====================================================================
        
        # Acyclicity regularizer (only for self-attention DAGs)
        if self.kappa > 0:
            acyclic_regularizer = 0.0
            if dec_self_phi is not None:
                if dec_self_phi.dim() != 2:
                    raise NotImplementedError(
                        f"Acyclicity regularization only supports single-head attention."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec_self_phi)
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # Prior regularizer
        def _get_prior_reg(phi, evidence, alpha, use_adaptive_scaling, lambda_kl):
            if phi is None or evidence is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            
            if use_adaptive_scaling and alpha is not None:
                alpha_abs = torch.abs(alpha)
            else:
                alpha_abs = 1.0
            
            kl = (alpha_abs * (p * (torch.log(p + _eps) - torch.log(p0 + _eps)) + 
                              (1 - p) * (torch.log(1 - p + _eps) - torch.log(1 - p0 + _eps)))).mean()
            return lambda_kl * kl
        
        prior_regularizer = (
            _get_prior_reg(dec_self_phi, dec_self_runav_mean, dec_self_runav_snr, 
                          self.adaptive_z_scaling, self.lambda_kl) + 
            _get_prior_reg(dec_cross_phi, dec_cross_runav_mean, dec_cross_runav_snr,
                          self.adaptive_z_scaling, self.lambda_kl)
        )
        
        # DAG Sparsity regularizer via _get_reg_dag
        def _get_reg_dag(phi):
            """L1 regularization on learned DAG (phi).
            
            Returns L1 norm: mean over rows.
            Shape: (L, S) -> mean()
            """
            if phi is None:
                return 0.0
            return torch.sigmoid(phi).mean()
        
        self_attention_sparsity = _get_reg_dag(dec_self_phi)
        cross_attention_sparsity = _get_reg_dag(dec_cross_phi)
        
        sparsity_regularizer = (
            self.lambda_sparse * self_attention_sparsity +
            self.lambda_sparse_cross * cross_attention_sparsity
        )
        
        # =====================================================================
        # UNIFIED SCORE SPARSITY REGULARIZATION
        # Mode selection: "l1" or "entropy", with automatic fallback for softmax attention
        # =====================================================================
        
        def _compute_score_sparsity(mode: str, inner_attention, entropy_value, device, is_self: bool):
            """
            Compute unified score sparsity regularization.
            
            Args:
                mode: "l1" or "entropy"
                inner_attention: Inner attention module with score_tensor_for_sparsity property
                entropy_value: Pre-computed attention entropy
                device: Device for tensors
                is_self: True for self-attention, False for cross-attention
                
            Returns:
                Tuple of (sparsity_value, actual_mode_used)
            """
            if mode == "l1":
                # Try L1 first
                score_tensor = getattr(inner_attention, 'score_tensor_for_sparsity', None)
                if score_tensor is not None:
                    return score_tensor.mean(), "l1"
                else:
                    # Fallback to entropy for softmax-based attention
                    if is_self and not self._self_sparsity_fallback_warned:
                        print("Warning: L1 sparsity unavailable for self-attention (softmax-based). Using entropy fallback.")
                        self._self_sparsity_fallback_warned = True
                    elif not is_self and not self._cross_sparsity_fallback_warned:
                        print("Warning: L1 sparsity unavailable for cross-attention (softmax-based). Using entropy fallback.")
                        self._cross_sparsity_fallback_warned = True
                    return entropy_value, "entropy"
            else:  # entropy
                return entropy_value, "entropy"
        
        # Self-attention score sparsity
        self_score_sparse, self_mode_used = _compute_score_sparsity(
            self.self_sparsity_regularizer, dec_self_inner, dec_self_ent_batch, X.device, is_self=True
        )
        
        # Cross-attention score sparsity
        cross_score_sparse, cross_mode_used = _compute_score_sparsity(
            self.cross_sparsity_regularizer, dec_cross_inner, dec_cross_ent_batch, X.device, is_self=False
        )
        
        # Total unified score sparsity regularizer
        score_sparsity_regularizer = (
            self.lambda_self_score_sparse * self_score_sparse +
            self.lambda_cross_score_sparse * cross_score_sparse
        )
        
        # =====================================================================
        # HSIC REGULARIZATION
        # Encourages independence between residuals and parents (S for cross, X for self)
        # - use_attention_weighted_hsic=False: uniform weighting (all edges equal)
        # - use_attention_weighted_hsic=True: weight by attention scores
        # - normalize_hsic_by_loss=True: scale HSIC by |loss_nll| for NLL training
        # =====================================================================
        hsic_regularizer = 0.0
        hsic_cross_value = None
        hsic_self_value = None
        
        # Compute loss normalization factor for NLL-aware HSIC
        # This ensures HSIC gradients scale with NLL gradients, keeping λ_hsic sensible
        # Without this, NLL's 1/σ² term amplifies gradients ~10,000-100,000x vs HSIC
        if self.normalize_hsic_by_loss:
            # Use |loss_nll| as scaling factor (detached to avoid second-order gradients)
            hsic_loss_scale = torch.abs(loss_nll).detach()
        else:
            hsic_loss_scale = 1.0
        
        # Always compute HSIC for logging
        residuals_per_x = x_target.squeeze() - mu.squeeze()
        
        if self.lambda_hsic_cross > 0 or self.lambda_hsic_self > 0:
            # S→X HSIC (cross-attention)
            if self.lambda_hsic_cross > 0:
                s_values = S[:, :, self.val_idx]  # (batch, seq_len_s)
                
                if self.use_attention_weighted_hsic:
                    # Attention-weighted: higher penalty for strongly attended edges
                    att_cross_mean = dec_cross_att[0].mean(dim=0)  # (seq_len_x, seq_len_s)
                    hsic_cross_value = hsic_attention_weighted(
                        source_values=s_values,
                        residuals=residuals_per_x,
                        attention_weights=att_cross_mean,
                        sigma=self.hsic_sigma,
                        exclude_diagonal=False
                    )
                else:
                    # Uniform weighting: use mean residuals for efficiency
                    mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
                    hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma)
                
                # Apply loss-normalized scaling: effective_loss = λ * |loss_nll| * hsic
                hsic_regularizer += self.lambda_hsic_cross * hsic_loss_scale * hsic_cross_value
            
            # X→X HSIC (self-attention)
            if self.lambda_hsic_self > 0:
                x_values_for_hsic = x_target.squeeze()  # (batch, seq_len_x)
                
                if residuals_per_x.dim() > 1:
                    if self.use_attention_weighted_hsic:
                        att_self_mean = dec_self_att[0].mean(dim=0)
                        hsic_self_value = hsic_attention_weighted(
                            source_values=x_values_for_hsic,
                            residuals=residuals_per_x,
                            attention_weights=att_self_mean,
                            sigma=self.hsic_sigma,
                            exclude_diagonal=True
                        )
                    else:
                        hsic_self_value = hsic_per_x_pair(x_values_for_hsic, residuals_per_x, sigma=self.hsic_sigma)
                    
                    hsic_regularizer += self.lambda_hsic_self * hsic_loss_scale * hsic_self_value
        
        # Always compute HSIC for logging (if not already computed)
        if hsic_cross_value is None:
            s_values = S[:, :, self.val_idx]
            mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
            hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma)
        if hsic_self_value is None and residuals_per_x.dim() > 1:
            hsic_self_value = hsic_per_x_pair(x_target.squeeze(), residuals_per_x, sigma=self.hsic_sigma)
        
        # DAG Decisiveness regularizer (always compute for logging)
        decisive_self_loss = torch.tensor(0.0, device=x_target.device)
        decisive_cross_loss = torch.tensor(0.0, device=x_target.device)
        tau_self_loss = torch.tensor(0.0, device=x_target.device)
        tau_cross_loss = torch.tensor(0.0, device=x_target.device)
        
        if self.lambda_decisive > 0 or self.lambda_tau > 0:
            if dec_self_phi is not None:
                log_tau_gs_self = getattr(dec_self_inner, 'log_tau_gs', None)
                tau_gs_self = torch.exp(log_tau_gs_self) if log_tau_gs_self is not None else None
                
                is_square = dec_self_phi.shape[-2] == dec_self_phi.shape[-1]
                decisive_self_loss = dag_decisiveness_loss(
                    dec_self_phi, tau=tau_gs_self, exclude_diagonal=is_square
                )
                
                if log_tau_gs_self is not None and self.lambda_tau > 0:
                    tau_self_loss = dag_temperature_loss(log_tau_gs_self, target_tau=self.target_tau)
            
            if dec_cross_phi is not None:
                log_tau_gs_cross = getattr(dec_cross_inner, 'log_tau_gs', None)
                tau_gs_cross = torch.exp(log_tau_gs_cross) if log_tau_gs_cross is not None else None
                
                decisive_cross_loss = dag_decisiveness_loss(
                    dec_cross_phi, tau=tau_gs_cross, exclude_diagonal=False
                )
                
                if log_tau_gs_cross is not None and self.lambda_tau > 0:
                    tau_cross_loss = dag_temperature_loss(log_tau_gs_cross, target_tau=self.target_tau)
        
        decisiveness_regularizer = (
            self.lambda_decisive * decisive_self_loss +
            self.lambda_decisive_cross * decisive_cross_loss +
            self.lambda_tau * (tau_self_loss + tau_cross_loss)
        )
        
        # =====================================================================
        # NOISE PRIOR REGULARIZER (optional, for identifiability)
        # KL divergence from prior: encourages σ_A, σ_R to stay near initial values
        # =====================================================================
        
        noise_prior_regularizer = 0.0
        if self.lambda_noise_prior > 0:
            sigma_A = self.model.ambient_noise.sigma_A
            sigma_R = self.model.output_head.sigma_R
            
            # Log-normal prior: penalize deviation from prior values
            # KL(σ || σ_prior) ≈ (log(σ) - log(σ_prior))² / 2
            noise_prior_regularizer = self.lambda_noise_prior * (
                ((torch.log(sigma_A) - torch.log(torch.tensor(self.prior_sigma_A, device=sigma_A.device))) ** 2).mean() +
                ((torch.log(sigma_R) - torch.log(torch.tensor(self.prior_sigma_R, device=sigma_R.device))) ** 2).mean()
            )
        
        # =====================================================================
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        # =====================================================================
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_regularizer = self.lambda_group_l1 * group_l1_loss
        
        # =====================================================================
        # GRADIENT AND UPDATE SIGNAL LOGGING (for two-step calibration)
        # =====================================================================
        # When enabled, computes BOTH:
        # 1. BASE GRADIENTS - for Step 1 (sparsity calibration via λ_group)
        # 2. UPDATE SIGNALS - for Step 2 (λ_hsic selection and verification)
        #
        # Two-Step Calibration Process:
        # Step 1: Find λ_group to make HSIC landscape non-flat
        #   - Use BASE gradient ratios (train_grad_ratio_*)
        #   - Target: bring base ratio toward 1.0 via sparsity
        #   - Output: λ_group_optimal, final_base_ratio
        #
        # Step 2: Select λ_hsic and verify balance
        #   - Set λ_hsic = final_base_ratio (makes update_ratio ≈ 1.0)
        #   - Verify with UPDATE ratios (train_update_ratio_*)
        #   - Output: suggested λ_hsic values
        #
        # Why both metrics?
        # - Base gradients reflect the HSIC landscape shape (sparsity effect)
        # - Update signals reflect the actual learning contribution (λ effect)
        # - Step 1 uses sparsity to shape the landscape
        # - Step 2 uses λ to balance the learning signals
        if self.log_gradient_norms and stage == "train":
            # Compute gradient norms for each loss component separately
            # Using torch.autograd.grad to compute gradients without accumulating
            
            def _compute_grad_norm(loss_tensor, model_params):
                """Compute Frobenius norm of gradients for a loss component."""
                if loss_tensor == 0.0 or not isinstance(loss_tensor, torch.Tensor):
                    return torch.tensor(0.0, device=x_target.device)
                
                grads = torch.autograd.grad(
                    loss_tensor, model_params, 
                    retain_graph=True, allow_unused=True, create_graph=False
                )
                total_norm = 0.0
                for g in grads:
                    if g is not None:
                        total_norm += g.norm(2).item() ** 2
                return torch.tensor(total_norm ** 0.5, device=x_target.device)
            
            # Get list of model parameters with gradients
            params_with_grad = [p for p in self.model.parameters() if p.requires_grad]
            
            # =================================================================
            # RECONSTRUCTION GRADIENT (same for base and update, implicit λ=1)
            # =================================================================
            recon_grad_norm = _compute_grad_norm(loss_nll, params_with_grad)
            self.log("train_recon_grad_norm", recon_grad_norm, on_step=False, on_epoch=True)
            
            # =================================================================
            # HSIC CROSS - BASE GRADIENT AND UPDATE SIGNAL
            # =================================================================
            hsic_cross_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_cross_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_cross_value is not None and isinstance(hsic_cross_value, torch.Tensor):
                # BASE gradient: ||∇ hsic_cross_value|| (no lambda)
                hsic_cross_base_grad = _compute_grad_norm(hsic_cross_value, params_with_grad)
                
                # UPDATE signal: λ * [scale] * ||∇ hsic_cross_value||
                hsic_cross_update_norm = self.lambda_hsic_cross * hsic_cross_base_grad
                if self.normalize_hsic_by_loss:
                    hsic_cross_update_norm = hsic_cross_update_norm * hsic_loss_scale
            
            # Log BASE gradient (for Step 1: sparsity calibration)
            self.log("train_hsic_cross_grad_norm", hsic_cross_base_grad, on_step=False, on_epoch=True)
            # Log UPDATE signal (for Step 2: verification)
            self.log("train_hsic_cross_update_norm", hsic_cross_update_norm, on_step=False, on_epoch=True)
            
            # =================================================================
            # HSIC SELF - BASE GRADIENT AND UPDATE SIGNAL
            # =================================================================
            hsic_self_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_self_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_self_value is not None and isinstance(hsic_self_value, torch.Tensor):
                # BASE gradient: ||∇ hsic_self_value|| (no lambda)
                hsic_self_base_grad = _compute_grad_norm(hsic_self_value, params_with_grad)
                
                # UPDATE signal: λ * [scale] * ||∇ hsic_self_value||
                hsic_self_update_norm = self.lambda_hsic_self * hsic_self_base_grad
                if self.normalize_hsic_by_loss:
                    hsic_self_update_norm = hsic_self_update_norm * hsic_loss_scale
            
            # Log BASE gradient (for Step 1: sparsity calibration)
            self.log("train_hsic_self_grad_norm", hsic_self_base_grad, on_step=False, on_epoch=True)
            # Log UPDATE signal (for Step 2: verification)
            self.log("train_hsic_self_update_norm", hsic_self_update_norm, on_step=False, on_epoch=True)
            
            # =================================================================
            # BASE GRADIENT RATIOS (for Step 1: sparsity calibration)
            # =================================================================
            # These ratios are independent of λ_hsic
            # Step 1 tries to bring these toward 1.0 via λ_group (sparsity)
            if hsic_cross_base_grad > 1e-10:
                ratio_cross = recon_grad_norm / hsic_cross_base_grad
                self.log("train_grad_ratio_cross", ratio_cross, on_step=False, on_epoch=True)
            
            if hsic_self_base_grad > 1e-10:
                ratio_self = recon_grad_norm / hsic_self_base_grad
                self.log("train_grad_ratio_self", ratio_self, on_step=False, on_epoch=True)
            
            # Min base ratio (for Step 1 convergence)
            base_ratios = []
            if hsic_cross_base_grad > 1e-10:
                base_ratios.append(float(recon_grad_norm / hsic_cross_base_grad))
            if hsic_self_base_grad > 1e-10:
                base_ratios.append(float(recon_grad_norm / hsic_self_base_grad))
            if base_ratios:
                self.log("train_grad_ratio_min", min(base_ratios), on_step=False, on_epoch=True)
            
            # =================================================================
            # UPDATE SIGNAL RATIOS (for Step 2: verification)
            # =================================================================
            # These ratios reflect actual learning balance with current λ values
            # After setting λ_hsic = base_ratio, update_ratio should be ≈ 1.0
            if hsic_cross_update_norm > 1e-10:
                update_ratio_cross = recon_grad_norm / hsic_cross_update_norm
                self.log("train_update_ratio_cross", update_ratio_cross, on_step=False, on_epoch=True)
            
            if hsic_self_update_norm > 1e-10:
                update_ratio_self = recon_grad_norm / hsic_self_update_norm
                self.log("train_update_ratio_self", update_ratio_self, on_step=False, on_epoch=True)
            
            # Min update ratio (for Step 2 verification)
            update_ratios = []
            if hsic_cross_update_norm > 1e-10:
                update_ratios.append(float(recon_grad_norm / hsic_cross_update_norm))
            if hsic_self_update_norm > 1e-10:
                update_ratios.append(float(recon_grad_norm / hsic_self_update_norm))
            if update_ratios:
                self.log("train_update_ratio_min", min(update_ratios), on_step=False, on_epoch=True)
        
        # =====================================================================
        # TOTAL LOSS
        # =====================================================================
        
        total_loss = (loss_nll + 
                     acyclic_regularizer +
                     prior_regularizer +
                     sparsity_regularizer +
                     score_sparsity_regularizer +
                     hsic_regularizer +
                     decisiveness_regularizer +
                     noise_prior_regularizer +
                     group_l1_regularizer)
        
        # =====================================================================
        # LOGGING - Always log everything
        # =====================================================================
        
        # Main loss
        self.log(f"{stage}_nll", loss_nll, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Reconstruction metrics
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(mu.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val" and name == "mae"))
        
        # Predicted variance
        var = torch.exp(log_var)
        self.log(f"{stage}_pred_var_mean", var.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_pred_var_std", var.std(), on_step=False, on_epoch=True)
        
        # Noise parameters
        sigma_A = self.model.ambient_noise.sigma_A
        sigma_R = self.model.output_head.sigma_R
        self.log(f"{stage}_sigma_A_mean", sigma_A.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_sigma_A_std", sigma_A.std(), on_step=False, on_epoch=True)
        self.log(f"{stage}_sigma_R_mean", sigma_R.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_sigma_R_std", sigma_R.std(), on_step=False, on_epoch=True)
        
        # Acyclicity
        self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True)
        
        # DAG sparsity
        self.log(f"{stage}_sparsity_self", self_attention_sparsity, on_step=False, on_epoch=True)
        self.log(f"{stage}_sparsity_cross", cross_attention_sparsity, on_step=False, on_epoch=True)
        
        # HSIC
        if hsic_cross_value is not None:
            self.log(f"{stage}_hsic_cross", hsic_cross_value, on_step=False, on_epoch=True)
        if hsic_self_value is not None:
            self.log(f"{stage}_hsic_self", hsic_self_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_regularizer, on_step=False, on_epoch=True)
        
        # Decisiveness
        self.log(f"{stage}_decisive_self", decisive_self_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_decisive_cross", decisive_cross_loss, on_step=False, on_epoch=True)
        
        # Group L1
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)
        
        return total_loss, mu, log_var, X
    
    @staticmethod
    def _linear_anneal(start: float, end: float, epoch: int, total_epochs: int) -> float:
        """Linear annealing from start to end over total_epochs."""
        progress = min(1.0, epoch / max(1, total_epochs))
        return start + progress * (end - start)
    
    def on_train_epoch_start(self):
        """Apply annealing schedules at the start of each training epoch."""
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs if self.trainer else 100
        
        dec_self_inner = self.model.decoder.layers[0].global_self_attention.inner_attention
        
        # 1. Gumbel-Softmax temperature annealing
        if self.use_tau_gs_annealing:
            anneal_epochs = self.tau_gs_anneal_epochs or max_epochs
            new_tau_gs = self._linear_anneal(self.tau_gs_start, self.tau_gs_end, epoch, anneal_epochs)
            
            log_tau_gs = getattr(dec_self_inner, 'log_tau_gs', None)
            if log_tau_gs is not None:
                with torch.no_grad():
                    log_tau_gs.copy_(torch.log(torch.tensor(new_tau_gs)))
            
            self.log("annealed_tau_gs", new_tau_gs, on_step=False, on_epoch=True)
        
        # 2. Toeplitz activation temperature annealing
        if self.use_tau_act_annealing:
            anneal_epochs = self.tau_act_anneal_epochs or max_epochs
            new_tau_gate = self._linear_anneal(self.tau_gate_start, self.tau_gate_end, epoch, anneal_epochs)
            new_tau_dir = self._linear_anneal(self.tau_dir_start, self.tau_dir_end, epoch, anneal_epochs)
            
            log_tau_gate = getattr(dec_self_inner, 'log_tau_gate', None)
            log_tau_dir = getattr(dec_self_inner, 'log_tau_dir', None)
            
            if log_tau_gate is not None:
                with torch.no_grad():
                    log_tau_gate.copy_(torch.log(torch.tensor(new_tau_gate)))
            if log_tau_dir is not None:
                with torch.no_grad():
                    log_tau_dir.copy_(torch.log(torch.tensor(new_tau_dir)))
            
            self.log("annealed_tau_gate", new_tau_gate, on_step=False, on_epoch=True)
            self.log("annealed_tau_dir", new_tau_dir, on_step=False, on_epoch=True)
        
        # 3. HSIC annealing - independent annealing for cross and self
        if self.use_hsic_annealing:
            anneal_epochs = self.hsic_anneal_epochs or max_epochs
            
            # Anneal cross-attention HSIC independently
            self.lambda_hsic_cross = self._linear_anneal(
                self.hsic_lambda_cross_start, self.hsic_lambda_cross_end, epoch, anneal_epochs
            )
            
            # Anneal self-attention HSIC independently
            self.lambda_hsic_self = self._linear_anneal(
                self.hsic_lambda_self_start, self.hsic_lambda_self_end, epoch, anneal_epochs
            )
            
            self.log("annealed_lambda_hsic_cross", self.lambda_hsic_cross, on_step=False, on_epoch=True)
            self.log("annealed_lambda_hsic_self", self.lambda_hsic_self, on_step=False, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, _, _, _ = self._step(batch=batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, _, _, _ = self._step(batch=batch, stage="test")
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with optional learning rate scheduler."""
        
        learning_rate = self.config["training"].get("lr", 1e-4)
        weight_decay = self.config["training"].get("weight_decay", 0.01)
        optimizer_type = self.config["training"].get("optimizer", "adamw").lower()
        
        if optimizer_type == "sgd":
            momentum = self.config["training"].get("momentum", 0.0)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        if self.config["training"].get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        
        return optimizer
    
    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity constraint."""
        d = A.shape[0]
        expm_A = torch.matrix_exp(torch.relu(A))
        return torch.trace(expm_A) - d
    
    def _compute_group_l1(self) -> tuple:
        """
        Compute Group L1 (L2,1 norm) on embedding columns - Group LASSO regularization.
        
        This regularizer encourages entire embedding dimensions (columns) to go to zero,
        effectively reducing d_model and creating an information bottleneck that makes
        the HSIC signal meaningful during causal structure learning.
        
        Unlike element-wise L1 (standard LASSO):
        - Element-wise L1: Different variables can use different sparse dimensions
        - Group L1 (L2,1): Entire dimensions are zeroed out, truly reducing capacity
        
        The L2,1 norm is computed as:
            ||W||_{2,1} = Σ_j ||W[:,j]||_2 = Σ_j sqrt(Σ_i W[i,j]²)
        
        - L2 within columns: Keeps values within a column grouped together
        - L1 over column norms: Induces sparsity at the column level (LASSO effect)
        
        Returns:
            Tuple of:
            - group_l1_loss: Normalized L2,1 norm of embedding weights
            - effective_dims: Number of embedding dimensions with ||col||_2 > threshold
        """
        device = next(self.model.embedding_X.parameters()).device
        l21_norm = torch.tensor(0.0, device=device)
        total_dims = 0
        active_dims = 0
        threshold = 1e-3  # Threshold for counting "active" dimensions
        
        # Get embedding weight matrices
        # For nn.Embedding, weight shape is (num_embeddings, d_model)
        # We want to regularize columns (dimensions) to encourage entire dimensions to be zero
        
        for p in self.model.embedding_X.parameters():
            if p.requires_grad and p.dim() >= 2:
                # Get weight matrix - shape is (num_embeddings, d_model) for Embedding
                # or (out_features, in_features) for Linear
                W = p
                
                # Compute L2 norm per column (dimension)
                # For Embedding: columns are the embedding dimensions
                # norm over dim=0 gives (d_model,) - one norm per dimension
                l2_per_col = W.norm(p=2, dim=0)  # (d_model,)
                
                # Sum of L2 norms (L2,1 norm) - this is the Group LASSO penalty
                l21_norm = l21_norm + l2_per_col.sum()
                
                # Count active dimensions
                total_dims += l2_per_col.numel()
                active_dims += (l2_per_col > threshold).sum().item()
        
        # Normalize by total dimensions to make λ scale-independent
        if total_dims > 0:
            l21_norm = l21_norm / total_dims
        
        # Effective dimensions (float for logging)
        effective_dims = torch.tensor(float(active_dims), device=device)
        
        return l21_norm, effective_dims
    
    # =========================================================================
    # INFERENCE UTILITIES
    # =========================================================================
    
    def predict(
        self, 
        S: torch.Tensor, 
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            S: Source tensor
            X: Intermediate tensor (with values to blank)
            
        Returns:
            mu: Predicted mean
            std: Predicted standard deviation
        """
        self.eval()
        with torch.no_grad():
            mu, log_var, _, _, _ = self.forward(S, X, inject_noise=False)
            std = torch.exp(0.5 * log_var)
        return mu, std
    
    def predict_with_intervals(
        self, 
        S: torch.Tensor, 
        X: torch.Tensor,
        confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence intervals.
        
        Args:
            S: Source tensor
            X: Intermediate tensor
            confidence: Confidence level (default 0.95)
            
        Returns:
            mu: Predicted mean
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
        """
        import scipy.stats
        
        mu, std = self.predict(S, X)
        z = scipy.stats.norm.ppf((1 + confidence) / 2)
        lower = mu - z * std
        upper = mu + z * std
        return mu, lower, upper
    
    def get_noise_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current noise parameter values."""
        return self.model.get_noise_parameters()
    
    def get_predictive_distribution(
        self, 
        S: torch.Tensor, 
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full predictive distribution parameters.
        
        Returns:
            mu: Mean tensor
            var: Variance tensor (τ² = σ_R²)
        """
        mu, std = self.predict(S, X)
        var = std ** 2
        return mu, var
