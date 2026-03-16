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
from causaliT.utils.hsic_utils import hsic_per_token
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
        
        # Logging configuration
        self.log_entropy = config["training"].get("log_entropy", False)
        self.log_acyclicity = config["training"].get("log_acyclicity", False)
        self.log_noise_params = config["training"].get("log_noise_params", True)
        
        # Regularizers (same as SingleCausalForecaster)
        self.lambda_entropy_self = config["training"].get("lambda_entropy_self", 0.0)
        self.lambda_entropy_cross = config["training"].get("lambda_entropy_cross", 0.0)
        self.kappa = config["training"].get("kappa", 0)  # Acyclicity regularization
        
        # Sparsity regularization
        self.lambda_sparse = config["training"].get("lambda_sparse", 0)
        self.lambda_sparse_cross = config["training"].get("lambda_sparse_cross", None)
        if self.lambda_sparse_cross is None:
            self.lambda_sparse_cross = self.lambda_sparse
        self.log_sparsity = config["training"].get("log_sparsity", False)
        
        # L1 regularization on attention scores
        self.lambda_l1_self_scores = config["training"].get("lambda_l1_self_scores", 0.0)
        self.lambda_l1_cross_scores = config["training"].get("lambda_l1_cross_scores", 0.0)
        self.log_l1_scores = config["training"].get("log_l1_scores", False)
        
        # HSIC regularization
        self.lambda_hsic = config["training"].get("lambda_hsic", 0)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.log_hsic = config["training"].get("log_hsic", False)
        
        # KL divergence prior regularization
        self.lambda_kl = config["training"].get("lambda_kl", 1.0)
        self.adaptive_z_scaling = config["training"].get("adaptive_z_scaling", True)
        
        # DAG decisiveness regularization
        self.lambda_decisive = config["training"].get("lambda_decisive", 0)
        self.lambda_decisive_cross = config["training"].get("lambda_decisive_cross", None)
        self.lambda_tau = config["training"].get("lambda_tau", 0)
        self.target_tau = config["training"].get("target_tau", 0.1)
        self.log_decisiveness = config["training"].get("log_decisiveness", False)
        if self.lambda_decisive_cross is None:
            self.lambda_decisive_cross = self.lambda_decisive
        
        # Noise-specific regularization (optional)
        # Can add prior on noise parameters if needed for identifiability
        self.lambda_noise_prior = config["training"].get("lambda_noise_prior", 0.0)
        self.prior_sigma_A = config["training"].get("prior_sigma_A", 0.01)
        self.prior_sigma_R = config["training"].get("prior_sigma_R", 0.05)
        
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
        
        # 3. HSIC Annealing (lambda_hsic decreases over training)
        self.use_hsic_annealing = config["training"].get("use_hsic_annealing", False)
        self.hsic_lambda_start = config["training"].get("hsic_lambda_start", 1.0)
        self.hsic_lambda_end = config["training"].get("hsic_lambda_end", 0.0)
        self.hsic_anneal_epochs = config["training"].get("hsic_anneal_epochs", None)
        
        # L1 regularization on Toeplitz gate probabilities
        self.lambda_l1_toeplitz_gate = config["training"].get("lambda_l1_toeplitz_gate", 0.0)
        
        # Logging for annealing (disabled by default)
        self.log_tau_annealing = config["training"].get("log_tau_annealing", False)
        self.log_hsic_annealing = config["training"].get("log_hsic_annealing", False)
        
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
        
        # Compute entropy regularization if needed
        if self.lambda_entropy_self > 0 or self.lambda_entropy_cross > 0 or self.log_entropy:
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
        
        # Entropy regularizer
        if self.lambda_entropy_self > 0 or self.lambda_entropy_cross > 0:
            entropy_regularizer = (
                self.lambda_entropy_self * dec_self_ent_batch +
                self.lambda_entropy_cross * dec_cross_ent_batch
            )
        else:
            entropy_regularizer = 0.0
        
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
        
        # Sparsity regularizer
        def _get_sparsity_reg(phi):
            if phi is None:
                return 0.0
            return torch.sigmoid(phi).mean()
        
        self_attention_sparsity = _get_sparsity_reg(dec_self_phi)
        cross_attention_sparsity = _get_sparsity_reg(dec_cross_phi)
        
        sparsity_regularizer = (
            self.lambda_sparse * self_attention_sparsity +
            self.lambda_sparse_cross * cross_attention_sparsity
        )
        
        # L1 on attention scores
        def _get_att_scores_l1(att_weights_list):
            if not att_weights_list:
                return 0.0
            return att_weights_list[-1].mean()
        
        l1_self_scores = _get_att_scores_l1(dec_self_att)
        l1_cross_scores = _get_att_scores_l1(dec_cross_att)
        
        l1_scores_regularizer = (
            self.lambda_l1_self_scores * l1_self_scores +
            self.lambda_l1_cross_scores * l1_cross_scores
        )
        
        # HSIC regularizer
        if self.lambda_hsic > 0 or self.log_hsic:
            residuals = x_target.squeeze() - mu.squeeze()
            if residuals.dim() > 1:
                mean_residuals = residuals.mean(dim=1)
            else:
                mean_residuals = residuals
            
            s_values = S[:, :, self.val_idx]
            hsic_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma)
            hsic_regularizer = self.lambda_hsic * hsic_value
        else:
            hsic_regularizer = 0.0
            hsic_value = None
        
        # DAG Decisiveness regularizer
        decisive_self_loss = torch.tensor(0.0, device=x_target.device)
        decisive_cross_loss = torch.tensor(0.0, device=x_target.device)
        tau_self_loss = torch.tensor(0.0, device=x_target.device)
        tau_cross_loss = torch.tensor(0.0, device=x_target.device)
        
        if self.lambda_decisive > 0 or self.lambda_tau > 0 or self.log_decisiveness:
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
        
        # L1 regularization on Toeplitz gate probabilities
        l1_toeplitz_gate = torch.tensor(0.0, device=x_target.device)
        if self.lambda_l1_toeplitz_gate > 0:
            gate_probs = getattr(dec_self_inner, 'gate_probs_for_reg', None)
            if gate_probs is not None:
                l1_toeplitz_gate = gate_probs.mean()
        l1_toeplitz_gate_reg = self.lambda_l1_toeplitz_gate * l1_toeplitz_gate
        
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
        # TOTAL LOSS
        # =====================================================================
        
        total_loss = (loss_nll + 
                     entropy_regularizer + 
                     acyclic_regularizer +
                     prior_regularizer +
                     sparsity_regularizer +
                     l1_scores_regularizer +
                     hsic_regularizer +
                     decisiveness_regularizer +
                     l1_toeplitz_gate_reg +
                     noise_prior_regularizer)
        
        # =====================================================================
        # LOGGING
        # =====================================================================
        
        # Log NLL loss
        self.log(f"{stage}_nll", loss_nll, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log reconstruction metrics (using mean prediction)
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(mu.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val" and name == "mae"))
        
        # Log predicted variance statistics
        var = torch.exp(log_var)
        self.log(f"{stage}_pred_var_mean", var.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}_pred_var_std", var.std(), on_step=False, on_epoch=True)
        
        # Log noise parameters
        if self.log_noise_params:
            sigma_A = self.model.ambient_noise.sigma_A
            sigma_R = self.model.output_head.sigma_R
            self.log(f"{stage}_sigma_A_mean", sigma_A.mean(), on_step=False, on_epoch=True)
            self.log(f"{stage}_sigma_A_std", sigma_A.std(), on_step=False, on_epoch=True)
            self.log(f"{stage}_sigma_R_mean", sigma_R.mean(), on_step=False, on_epoch=True)
            self.log(f"{stage}_sigma_R_std", sigma_R.std(), on_step=False, on_epoch=True)
        
        # Log entropies if requested
        if self.log_entropy:
            self.log(f"{stage}_dec_cross_entropy", dec_cross_ent_batch, on_step=False, on_epoch=True)
            self.log(f"{stage}_dec_self_entropy", dec_self_ent_batch, on_step=False, on_epoch=True)
        
        # Log acyclicity if requested
        if self.log_acyclicity:
            self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True)
        
        # Log sparsity if requested
        if self.log_sparsity:
            self.log(f"{stage}_sparsity_self", self_attention_sparsity, on_step=False, on_epoch=True)
            self.log(f"{stage}_sparsity_cross", cross_attention_sparsity, on_step=False, on_epoch=True)
        
        # Log L1 scores if requested
        if self.log_l1_scores:
            self.log(f"{stage}_l1_self_scores", l1_self_scores, on_step=False, on_epoch=True)
            self.log(f"{stage}_l1_cross_scores", l1_cross_scores, on_step=False, on_epoch=True)
        
        # Log HSIC if requested
        if self.log_hsic and hsic_value is not None:
            self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
        
        # Log decisiveness if requested
        if self.log_decisiveness:
            self.log(f"{stage}_decisive_self", decisive_self_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_decisive_cross", decisive_cross_loss, on_step=False, on_epoch=True)
        
        return total_loss, mu, log_var, X
    
    def on_train_epoch_start(self):
        """Apply annealing schedules at the start of each training epoch."""
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs if self.trainer else 100
        
        dec_self_inner = self.model.decoder.layers[0].global_self_attention.inner_attention
        
        # 1. Gumbel-Softmax temperature annealing
        if self.use_tau_gs_annealing:
            anneal_epochs = self.tau_gs_anneal_epochs or max_epochs
            progress = min(1.0, epoch / anneal_epochs)
            # Exponential annealing: tau = start * (end/start)^progress
            new_tau_gs = self.tau_gs_start * (self.tau_gs_end / self.tau_gs_start) ** progress
            new_log_tau_gs = torch.log(torch.tensor(new_tau_gs))
            
            log_tau_gs = getattr(dec_self_inner, 'log_tau_gs', None)
            if log_tau_gs is not None:
                with torch.no_grad():
                    log_tau_gs.copy_(new_log_tau_gs)
            
            if self.log_tau_annealing:
                self.log("annealed_tau_gs", new_tau_gs, on_step=False, on_epoch=True)
        
        # 2. Toeplitz activation temperature annealing
        if self.use_tau_act_annealing:
            anneal_epochs = self.tau_act_anneal_epochs or max_epochs
            progress = min(1.0, epoch / anneal_epochs)
            
            new_tau_gate = self.tau_gate_start * (self.tau_gate_end / self.tau_gate_start) ** progress
            new_tau_dir = self.tau_dir_start * (self.tau_dir_end / self.tau_dir_start) ** progress
            
            log_tau_gate = getattr(dec_self_inner, 'log_tau_gate', None)
            log_tau_dir = getattr(dec_self_inner, 'log_tau_dir', None)
            
            if log_tau_gate is not None:
                with torch.no_grad():
                    log_tau_gate.copy_(torch.log(torch.tensor(new_tau_gate)))
            if log_tau_dir is not None:
                with torch.no_grad():
                    log_tau_dir.copy_(torch.log(torch.tensor(new_tau_dir)))
            
            if self.log_tau_annealing:
                self.log("annealed_tau_gate", new_tau_gate, on_step=False, on_epoch=True)
                self.log("annealed_tau_dir", new_tau_dir, on_step=False, on_epoch=True)
        
        # 3. HSIC annealing (decreasing lambda)
        if self.use_hsic_annealing:
            anneal_epochs = self.hsic_anneal_epochs or max_epochs
            progress = min(1.0, epoch / anneal_epochs)
            # Linear annealing from start to end
            self.lambda_hsic = self.hsic_lambda_start + progress * (self.hsic_lambda_end - self.hsic_lambda_start)
            
            if self.log_hsic_annealing:
                self.log("annealed_lambda_hsic", self.lambda_hsic, on_step=False, on_epoch=True)
    
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
