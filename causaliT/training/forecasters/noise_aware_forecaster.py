"""
NoiseAwareCausalForecaster: PyTorch Lightning wrapper for noise-aware causal model.

This forecaster handles training, validation, and testing for the noise-aware
architecture with Gaussian NLL loss, enabling uncertainty quantification.

Key Features:
- Gaussian Negative Log-Likelihood loss for probabilistic training
- Ambient noise (σ_A) and reading noise (σ_R) are learned parameters
- Outputs full predictive distribution at inference time
- Score sparsity regularization across ALL decoder layers

Active regularizers:
- Score sparsity (L1/entropy on attention scores) — applied to ALL decoder layers
- HSIC (independence between residuals and parents)
- Group L1 (embedding bottleneck)
- Noise prior (optional, for identifiability)

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
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

from causaliT.core.architectures.noise_aware import NoiseAwareSingleCausalLayer
from causaliT.core.modules.noise_layers import GaussianNLLLoss
from causaliT.core.utils import load_dag_masks, corrupt_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token, hsic_per_x_pair, hsic_attention_weighted, hsic_cross_per_pair
from causaliT.training.gradient_routing import classify_parameters


class NoiseAwareCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for NoiseAwareSingleCausalLayer transformer model.
    
    This forecaster manages training for a noise-aware causal relationship: S → X
    with explicit modeling of ambient and reading noise.
    
    Features:
    - Gaussian NLL loss for uncertainty-aware training
    - Per-node learnable noise parameters (σ_A, σ_R)
    - Score sparsity regularization across ALL decoder layers
    - HSIC regularization for causal independence
    - Hard mask support for enforcing ground-truth DAG structure
    - Full predictive distribution output at inference
    
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
            reduction='none',
            full=config["training"].get("nll_full", False)
        )
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # =====================================================================
        # UNIFIED SCORE SPARSITY REGULARIZATION
        # Applied to ALL decoder layers (averaged)
        # =====================================================================
        self.lambda_self_score_sparse = config["training"].get("lambda_self_score_sparse", 0.0)
        self.lambda_cross_score_sparse = config["training"].get("lambda_cross_score_sparse", 0.0)
        # Config keys self_sparsity_regularizer / cross_sparsity_regularizer
        # are DEPRECATED and ignored — auto-detection is used instead.
        
        # =====================================================================
        # HSIC REGULARIZATION
        # =====================================================================
        self.lambda_hsic_cross = config["training"].get("lambda_hsic_cross", 
                                     config["training"].get("lambda_hsic", 0.0))
        self.lambda_hsic_self = config["training"].get("lambda_hsic_self", 0.0)
        self.use_attention_weighted_hsic = config["training"].get("use_attention_weighted_hsic", False)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        # HSIC estimator: "biased" (standard) or "normalized" (nHSIC, Ma et al. 2020)
        self.hsic_mode = config["training"].get("hsic_mode", "biased")
        self.nhsic_epsilon = config["training"].get("nhsic_epsilon", 0.01)
        self.normalize_hsic_by_loss = config["training"].get("normalize_hsic_by_loss", False)
        
        # HSIC cross-attention mode: "averaged" (legacy) or "per_variable" (edge-level)
        # "averaged": HSIC(S_i, mean(residuals)) — weak, diluted signal
        # "per_variable": HSIC(S_i, res_j) for all (i,j) pairs — edge-specific signal
        self.hsic_cross_mode = config["training"].get("hsic_cross_mode", "averaged")
        # Source kernel: "rbf" (default, for continuous S) or "dirac" (for discrete S)
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")
        
        # =====================================================================
        # NOISE-SPECIFIC REGULARIZATION
        # =====================================================================
        self.lambda_noise_prior = config["training"].get("lambda_noise_prior", 0.0)
        self.prior_sigma_A = config["training"].get("prior_sigma_A", 0.01)
        self.prior_sigma_R = config["training"].get("prior_sigma_R", 0.05)
        
        # =====================================================================
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        # =====================================================================
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 
                                   config["training"].get("lambda_embed_l1", 0.0))
        
        # =====================================================================
        # STRUCTURAL LOSS RECON ANCHOR (convex mix)
        # =====================================================================
        # alpha in [0, 1]: convex combination weight on the structural pathway.
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg
        # Reconstruction loss is unchanged (alpha leaks recon into structure only).
        # Default 0.0 → exact backward compatibility (pure HSIC).
        # Note: for the noise-aware forecaster, "loss_recon" is the NLL + noise prior
        # (mirrors what is routed via gradient routing).
        self.lambda_struct_recon = float(config["training"].get("lambda_struct_recon", 0.0))
        if not (0.0 <= self.lambda_struct_recon <= 1.0):
            raise ValueError(
                f"lambda_struct_recon must be in [0, 1], got {self.lambda_struct_recon}"
            )
        
        # =====================================================================
        # GRADIENT NORM LOGGING (for calibration stage)
        # =====================================================================
        self.log_gradient_norms = config["training"].get("log_gradient_norms", False)

        
        # =====================================================================
        # ANNEALING CONFIGURATION
        # =====================================================================
        # NOTE (temperatures): the tau annealers were REMOVED (see
        # SingleCausalForecaster and docs/documentation/ATTENTION_TEMPERATURES.md).
        # The Hard-Concrete gate temperatures are fixed, calculated constants;
        # the learnable log_tau_gate / log_tau_dir parameters targeted by the
        # legacy use_tau_act_annealing schedule no longer exist, and
        # use_tau_annealing only rewrote the constant tau float of the legacy
        # non-gated attentions.

        # 1. HSIC Annealing - independent annealing for cross and self
        self.use_hsic_annealing = config["training"].get("use_hsic_annealing", False)
        self.hsic_anneal_epochs = config["training"].get("hsic_anneal_epochs", None)
        self.hsic_lambda_cross_start = config["training"].get("hsic_lambda_cross_start", self.lambda_hsic_cross)
        self.hsic_lambda_cross_end = config["training"].get("hsic_lambda_cross_end", 0.0)
        self.hsic_lambda_self_start = config["training"].get("hsic_lambda_self_start", self.lambda_hsic_self)
        self.hsic_lambda_self_end = config["training"].get("hsic_lambda_self_end", 0.0)
        
        # 2. Group L1 Annealing — start high, decay to lambda_group_l1

        #    Schedule: [0, idle) = start_value; [idle, idle+transient) = linear decay; rest = lambda_group_l1
        self._group_l1_anneal_start = config["training"].get("lambda_group_l1_anneal_start_value", None)
        self._group_l1_anneal_idle = config["training"].get("lambda_group_l1_anneal_idle_epochs", 0)
        self._group_l1_anneal_transient = config["training"].get("lambda_group_l1_anneal_transient_epochs", 0)
        self._use_group_l1_annealing = (
            self._group_l1_anneal_start is not None
            and self._group_l1_anneal_start != self.lambda_group_l1
        )
        if self._use_group_l1_annealing:
            # Store the final target (what's in lambda_group_l1) and set current to start
            self._group_l1_final = self.lambda_group_l1
            self.lambda_group_l1 = float(self._group_l1_anneal_start)
        
        # =====================================================================
        # GRADIENT ROUTING (dual optimizer for structure vs reconstruction)
        # =====================================================================
        self.use_gradient_routing = config["training"].get("use_gradient_routing", False)
        if self.use_gradient_routing:
            self.automatic_optimization = False
            structural_params, reconstruction_params = classify_parameters(
                self.model, verbose=True
            )
            self._structural_params = structural_params
            self._reconstruction_params = reconstruction_params

        # =====================================================================
        # PARAMETER FREEZING FOR ALTERNATING STAGES (ANM experiments)
        # =====================================================================
        # Set by the training config (config['training']['freeze_*_params']).
        # Requires use_gradient_routing=True; otherwise the config loader
        # falls back to loss-level gating and leaves these False.
        # requires_grad is not saved in checkpoints, so each new stage starts
        # with all params unfrozen.
        self.freeze_structural_params = bool(
            config["training"].get("freeze_structural_params", False)
        )
        self.freeze_reconstruction_params = bool(
            config["training"].get("freeze_reconstruction_params", False)
        )

        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None

        # =====================================================================
        # ORACLE / MASK CORRUPTION CONFIGURATION
        # Mirrors SingleCausalForecaster — same config keys, same semantics.
        # hard_masks_corruption_seed in {None, 0}  OR  both SHDs == 0
        # disables corruption and the ground-truth masks are used unchanged.
        # See causaliT.core.utils.corrupt_dag_masks for full semantics.
        # =====================================================================
        self.use_oracle_attention = config["training"].get("use_oracle_attention", False)
        if self.use_oracle_attention and not self.use_hard_masks:
            raise ValueError(
                "training.use_oracle_attention=True requires training.use_hard_masks=True. "
                "The oracle uses the hard mask itself as the attention score matrix."
            )
        self.hard_masks_corruption_seed = config["training"].get("hard_masks_corruption_seed", None)
        self.cross_control_shd = int(config["training"].get("cross_control_shd", 0) or 0)
        self.self_control_shd  = int(config["training"].get("self_control_shd",  0) or 0)
        # When True, corruption is restricted to one-for-one edge swaps so the
        # corrupted mask retains the SAME number of edges as the ground truth.
        self.hard_masks_preserve_sparsity = bool(
            config["training"].get("hard_masks_preserve_sparsity", False)
        )
        # Filled in by _load_hard_masks when corruption is applied.
        self.hard_mask_corruption_info: Optional[Dict[str, dict]] = None

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
        """Load hard masks from data directory based on config.

        Mirrors SingleCausalForecaster._load_hard_masks:
        - Loads base masks from disk.
        - Applies corrupt_dag_masks() when hard_masks_corruption_seed is set
          and cross/self_control_shd > 0.
        - Stores corruption metadata in self.hard_mask_corruption_info for
          on_fit_start logging.
        """
        mask_files = config["training"].get("hard_mask_files", None)

        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified in config.")
            return

        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)

        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')

        if masks is None:
            print("Warning: No hard masks were loaded.")
            return

        # ------------------------------------------------------------------
        # Optional mask corruption (anti-oracle / SHD sweep)
        # Corruption is skipped when seed is None/0 or both SHDs are 0.
        # ------------------------------------------------------------------
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
                f"✓ Hard masks CORRUPTED "
                f"(seed={int(self.hard_masks_corruption_seed)}, "
                f"cross_shd={self.cross_control_shd}, "
                f"self_shd={self.self_control_shd}, "
                f"preserve_sparsity={self.hard_masks_preserve_sparsity})"
            )
            for _name, _info in corruption_info.items():
                cyc = _info.get("has_cycles")
                print(f"  [{_name}] actual_shd={_info.get('actual_shd')}  has_cycles={cyc}")
        else:
            print(
                f"✓ Hard masks loaded (no corruption: seed={self.hard_masks_corruption_seed}, "
                f"cross_shd={self.cross_control_shd}, self_shd={self.self_control_shd})"
            )

        self.hard_mask_corruption_info = corruption_info
        self._hard_masks = masks
        self._hard_masks_loaded = True

        for name, mask in masks.items():
            self.register_buffer(f'hard_mask_{name}', mask)

        print("✓ Hard masks registered for training.")
    
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
        # Unpack batch
        if len(batch) == 3:
            S, X, Y = batch
        else:
            S, X = batch
            Y = None
        
        # Extract actual values for loss computation
        x_val = X[:, :, self.val_idx]
        
        # Forward pass - inject noise only during training
        inject_noise = (stage == "train")
        mu, log_var, attention_weights, masks, entropies = self.forward(
            data_source=S,
            data_intermediate=X,
            inject_noise=inject_noise
        )
        
        # Unpack attention weights and entropies (lists, one element per decoder layer)
        dec_cross_att, dec_self_att = attention_weights
        dec_cross_ent, dec_self_ent = entropies
        
        # =====================================================================
        # GAUSSIAN NLL LOSS (main loss)
        # L = (x - μ)² / (2τ²) + log(τ)
        # =====================================================================
        x_target = torch.nan_to_num(x_val)
        nll_per_elem = self.nll_loss(mu.squeeze(), x_target.squeeze(), log_var.squeeze())
        loss_nll = nll_per_elem.mean()
        
        # =====================================================================
        # SCORE SPARSITY REGULARIZATION (all decoder layers)
        # Accumulates score sparsity across ALL layers and averages.
        # =====================================================================
        n_layers = len(self.model.decoder.layers)
        total_self_score_sparse = torch.tensor(0.0, device=X.device)
        total_cross_score_sparse = torch.tensor(0.0, device=X.device)
        self_mode_used = "entropy"
        cross_mode_used = "entropy"
        
        def _compute_score_sparsity(inner_attention, entropy_value, device):
            """
            Compute score sparsity regularization with auto-detection.
            
            Auto-selects the method based on what the attention module supports:
            - L1 on score_tensor_for_sparsity if available (non-softmax attention)
            - Entropy fallback for softmax-based attention (no score tensor)
            
            Config keys self_sparsity_regularizer / cross_sparsity_regularizer
            are DEPRECATED and ignored.
            """
            score_tensor = getattr(inner_attention, 'score_tensor_for_sparsity', None)
            if score_tensor is not None:
                # L1 norm: use absolute values since attention activations (e.g., GeLU(Tanh))
                # can produce negative values. Without .abs(), mean(A) can be negative.
                return score_tensor.abs().mean(), "l1"
            else:
                return entropy_value, "entropy"
        
        for layer_idx in range(n_layers):
            layer = self.model.decoder.layers[layer_idx]
            dec_self_inner = layer.global_self_attention.inner_attention
            dec_cross_inner = layer.global_cross_attention.inner_attention
            
            # Per-layer entropy (scalar)
            layer_self_ent = dec_self_ent[layer_idx].mean()
            layer_cross_ent = dec_cross_ent[layer_idx].mean()
            
            # Compute score sparsity for this layer (auto-detected)
            layer_self_sparse, self_mode_used = _compute_score_sparsity(
                dec_self_inner, layer_self_ent, X.device
            )
            layer_cross_sparse, cross_mode_used = _compute_score_sparsity(
                dec_cross_inner, layer_cross_ent, X.device
            )
            
            total_self_score_sparse = total_self_score_sparse + layer_self_sparse
            total_cross_score_sparse = total_cross_score_sparse + layer_cross_sparse
        
        # Average over layers
        avg_self_score_sparse = total_self_score_sparse / n_layers
        avg_cross_score_sparse = total_cross_score_sparse / n_layers
        
        score_sparsity_regularizer = (
            self.lambda_self_score_sparse * avg_self_score_sparse +
            self.lambda_cross_score_sparse * avg_cross_score_sparse
        )
        
        # =====================================================================
        # HSIC REGULARIZATION
        # =====================================================================
        hsic_regularizer = 0.0
        hsic_cross_value = None
        hsic_self_value = None
        
        # Compute loss normalization factor for NLL-aware HSIC
        if self.normalize_hsic_by_loss:
            hsic_loss_scale = torch.abs(loss_nll).detach()
        else:
            hsic_loss_scale = 1.0
        
        # Always compute HSIC for logging
        residuals_per_x = x_target.squeeze() - mu.squeeze()
        
        if self.lambda_hsic_cross > 0 or self.lambda_hsic_self > 0:
            # S→X HSIC (cross-attention)
            if self.lambda_hsic_cross > 0:
                s_values = S[:, :, self.val_idx]
                
                if self.use_attention_weighted_hsic:
                    att_cross_mean = dec_cross_att[0].mean(dim=0)
                    hsic_cross_value = hsic_attention_weighted(
                        source_values=s_values,
                        residuals=residuals_per_x,
                        attention_weights=att_cross_mean,
                        sigma=self.hsic_sigma,
                        exclude_diagonal=False,
                        adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                        mode=self.hsic_mode,
                        nhsic_epsilon=self.nhsic_epsilon,
                        source_kernel=self.hsic_kernel_source,
                    )
                elif self.hsic_cross_mode == "per_variable" and residuals_per_x.dim() > 1:
                    # Per-variable HSIC: HSIC(S_i, res_j) for all (i,j) pairs
                    # Provides edge-level gradient signal for cross-attention DAG learning
                    hsic_cross_value = hsic_cross_per_pair(
                        s_values, residuals_per_x,
                        sigma=self.hsic_sigma,
                        adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                        mode=self.hsic_mode,
                        nhsic_epsilon=self.nhsic_epsilon,
                        source_kernel=self.hsic_kernel_source,
                    )
                else:
                    # Legacy averaged HSIC: HSIC(S_i, mean(residuals)) — weak signal
                    mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
                    hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma,
                                                      adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                                                      mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon,
                                                      source_kernel=self.hsic_kernel_source)
                
                hsic_regularizer += self.lambda_hsic_cross * hsic_loss_scale * hsic_cross_value
            
            # X→X HSIC (self-attention)
            if self.lambda_hsic_self > 0:
                x_values_for_hsic = x_target.squeeze()
                
                if residuals_per_x.dim() > 1:
                    if self.use_attention_weighted_hsic:
                        att_self_mean = dec_self_att[0].mean(dim=0)
                        hsic_self_value = hsic_attention_weighted(
                            source_values=x_values_for_hsic,
                            residuals=residuals_per_x,
                            attention_weights=att_self_mean,
                            sigma=self.hsic_sigma,
                            exclude_diagonal=True,
                            adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                            mode=self.hsic_mode,
                            nhsic_epsilon=self.nhsic_epsilon,
                        )
                    else:
                        hsic_self_value = hsic_per_x_pair(x_values_for_hsic, residuals_per_x,
                                                          sigma=self.hsic_sigma,
                                                          adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                                                          mode=self.hsic_mode,
                                                          nhsic_epsilon=self.nhsic_epsilon)
                    
                    hsic_regularizer += self.lambda_hsic_self * hsic_loss_scale * hsic_self_value
        
        # Always compute HSIC for logging (if not already computed)
        if hsic_cross_value is None:
            s_values = S[:, :, self.val_idx]
            if self.hsic_cross_mode == "per_variable" and residuals_per_x.dim() > 1:
                hsic_cross_value = hsic_cross_per_pair(
                    s_values, residuals_per_x,
                    sigma=self.hsic_sigma,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                    mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon,
                    source_kernel=self.hsic_kernel_source,
                )
            else:
                mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
                hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma,
                                                  adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                                                  mode=self.hsic_mode, nhsic_epsilon=self.nhsic_epsilon,
                                                  source_kernel=self.hsic_kernel_source)
        if hsic_self_value is None and residuals_per_x.dim() > 1:
            hsic_self_value = hsic_per_x_pair(x_target.squeeze(), residuals_per_x,
                                              sigma=self.hsic_sigma,
                                              adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                                              mode=self.hsic_mode,
                                              nhsic_epsilon=self.nhsic_epsilon)
        
        # =====================================================================
        # NOISE PRIOR REGULARIZER (optional, for identifiability)
        # =====================================================================
        noise_prior_regularizer = 0.0
        if self.lambda_noise_prior > 0:
            sigma_A = self.model.ambient_noise.sigma_A
            sigma_R = self.model.output_head.sigma_R
            
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
        if self.log_gradient_norms and stage == "train":
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
            
            params_with_grad = [p for p in self.model.parameters() if p.requires_grad]
            
            # Reconstruction gradient
            recon_grad_norm = _compute_grad_norm(loss_nll, params_with_grad)
            self.log("train_recon_grad_norm", recon_grad_norm, on_step=False, on_epoch=True)
            
            # HSIC Cross
            hsic_cross_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_cross_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_cross_value is not None and isinstance(hsic_cross_value, torch.Tensor):
                hsic_cross_base_grad = _compute_grad_norm(hsic_cross_value, params_with_grad)
                hsic_cross_update_norm = self.lambda_hsic_cross * hsic_cross_base_grad
                if self.normalize_hsic_by_loss:
                    hsic_cross_update_norm = hsic_cross_update_norm * hsic_loss_scale
            
            self.log("train_hsic_cross_grad_norm", hsic_cross_base_grad, on_step=False, on_epoch=True)
            self.log("train_hsic_cross_update_norm", hsic_cross_update_norm, on_step=False, on_epoch=True)
            
            # HSIC Self
            hsic_self_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_self_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_self_value is not None and isinstance(hsic_self_value, torch.Tensor):
                hsic_self_base_grad = _compute_grad_norm(hsic_self_value, params_with_grad)
                hsic_self_update_norm = self.lambda_hsic_self * hsic_self_base_grad
                if self.normalize_hsic_by_loss:
                    hsic_self_update_norm = hsic_self_update_norm * hsic_loss_scale
            
            self.log("train_hsic_self_grad_norm", hsic_self_base_grad, on_step=False, on_epoch=True)
            self.log("train_hsic_self_update_norm", hsic_self_update_norm, on_step=False, on_epoch=True)
            
            # Base gradient ratios
            if hsic_cross_base_grad > 1e-10:
                self.log("train_grad_ratio_cross", recon_grad_norm / hsic_cross_base_grad, on_step=False, on_epoch=True)
            if hsic_self_base_grad > 1e-10:
                self.log("train_grad_ratio_self", recon_grad_norm / hsic_self_base_grad, on_step=False, on_epoch=True)
            
            base_ratios = []
            if hsic_cross_base_grad > 1e-10:
                base_ratios.append(float(recon_grad_norm / hsic_cross_base_grad))
            if hsic_self_base_grad > 1e-10:
                base_ratios.append(float(recon_grad_norm / hsic_self_base_grad))
            if base_ratios:
                self.log("train_grad_ratio_min", min(base_ratios), on_step=False, on_epoch=True)
            
            # Update signal ratios
            if hsic_cross_update_norm > 1e-10:
                self.log("train_update_ratio_cross", recon_grad_norm / hsic_cross_update_norm, on_step=False, on_epoch=True)
            if hsic_self_update_norm > 1e-10:
                self.log("train_update_ratio_self", recon_grad_norm / hsic_self_update_norm, on_step=False, on_epoch=True)
            
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
                     score_sparsity_regularizer +
                     hsic_regularizer +
                     noise_prior_regularizer +
                     group_l1_regularizer)
        
        # Store decomposed losses for gradient routing (used by training_step).
        # Convex mix on the structural pathway only (alpha leaks recon into struct):
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg
        # loss_recon (NLL + noise prior) is unchanged.
        alpha = self.lambda_struct_recon
        loss_recon_for_struct = loss_nll + noise_prior_regularizer
        self._last_loss_components = {
            "loss_recon": loss_recon_for_struct,
            "loss_structural": (
                (1.0 - alpha) * hsic_regularizer
                + alpha * loss_recon_for_struct
                + score_sparsity_regularizer
                + group_l1_regularizer
            ),
        }

        
        # =====================================================================
        # LOGGING
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
        
        # Score sparsity (averaged across layers)
        self.log(f"{stage}_self_score_sparse", avg_self_score_sparse, on_step=False, on_epoch=True)
        self.log(f"{stage}_cross_score_sparse", avg_cross_score_sparse, on_step=False, on_epoch=True)
        
        # HSIC
        if hsic_cross_value is not None:
            self.log(f"{stage}_hsic_cross", hsic_cross_value, on_step=False, on_epoch=True)
        if hsic_self_value is not None:
            self.log(f"{stage}_hsic_self", hsic_self_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_regularizer, on_step=False, on_epoch=True)
        
        # Group L1
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)
        
        return total_loss, mu, log_var, X
    
    @staticmethod
    def _linear_anneal(start: float, end: float, epoch: int, total_epochs: int) -> float:
        """Linear annealing from start to end over total_epochs."""
        progress = min(1.0, epoch / max(1, total_epochs))
        return start + progress * (end - start)

    def on_fit_start(self):
        """Stage-level parameter freezing + hard-mask dump.

        Parameter freezing:
            ANM alternating experiments: freeze structural or reconstruction
            params.  Only applies when use_gradient_routing=True (param groups
            exist).  When gradient_routing=False, _build_stage_config already
            fell back to loss-level gating and left these flags False.
            requires_grad is not saved in checkpoints, so each new stage starts
            with all params unfrozen.

        Hard-mask dump:
            Writes to <run_dir>/used_masks/:
            - dec_cross_used.csv, dec_self_used.csv  — binary mask arrays
            - mask_summary.json — corruption metadata

            Path resolution: CSVLogger.log_dir is "<save_dir_k>/logs/csv/version_X"
            so parents[2] = "<save_dir_k>".
        """
        # ANM alternating experiments: freeze structural or reconstruction params.
        # Only applies when use_gradient_routing=True (param groups exist).
        if self.freeze_structural_params and self.use_gradient_routing:
            for p in self._structural_params:
                p.requires_grad_(False)
            print("  [ANM stage] Structural parameters frozen (requires_grad=False).")
        if self.freeze_reconstruction_params and self.use_gradient_routing:
            for p in self._reconstruction_params:
                p.requires_grad_(False)
            print("  [ANM stage] Reconstruction parameters frozen (requires_grad=False).")

        if not self.use_hard_masks or not self._hard_masks_loaded:
            return

        try:
            log_dir = getattr(self.trainer, "log_dir", None)
            if log_dir is None:
                return
            out_dir = Path(log_dir).resolve().parents[2] / "used_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[on_fit_start] could not resolve run dir for mask dump: {e}")
            return

        masks = self.get_hard_masks() or {}
        for name, mask in masks.items():
            arr = mask.detach().cpu().numpy()
            np.savetxt(out_dir / f"{name}_used.csv", arr, fmt="%.6g", delimiter=",")

        # Build a small JSON summary (same schema as SingleCausalForecaster)
        summary: Dict[str, Any] = {
            "use_oracle_attention": bool(self.use_oracle_attention),
            "hard_masks_corruption_seed": self.hard_masks_corruption_seed,
            "cross_control_shd_requested": int(self.cross_control_shd),
            "self_control_shd_requested": int(self.self_control_shd),
            "hard_masks_preserve_sparsity": bool(self.hard_masks_preserve_sparsity),
            "corruption_applied": self.hard_mask_corruption_info is not None,
            "per_mask": {},
        }
        for name, mask in masks.items():
            arr = mask.detach().cpu().numpy()
            entry = {
                "shape": list(arr.shape),
                "num_active_edges": int((arr != 0).sum()),
                "density": float((arr != 0).mean()),
            }
            if (
                self.hard_mask_corruption_info is not None
                and name in self.hard_mask_corruption_info
            ):
                # Surface corrupt_dag_masks metadata verbatim
                entry["corruption"] = {
                    k: (v if not isinstance(v, (np.bool_, np.integer))
                        else (bool(v) if isinstance(v, np.bool_) else int(v)))
                    for k, v in self.hard_mask_corruption_info[name].items()
                }
            summary["per_mask"][name] = entry

        with open(out_dir / "mask_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[on_fit_start] wrote used masks to {out_dir}")

    def on_train_epoch_start(self):
        """Apply annealing schedules at the start of each training epoch."""
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs if self.trainer else 100

        # 1. HSIC annealing - independent for cross and self
        if self.use_hsic_annealing:
            anneal_epochs = self.hsic_anneal_epochs or max_epochs
            self.lambda_hsic_cross = self._linear_anneal(
                self.hsic_lambda_cross_start, self.hsic_lambda_cross_end, epoch, anneal_epochs
            )
            self.lambda_hsic_self = self._linear_anneal(
                self.hsic_lambda_self_start, self.hsic_lambda_self_end, epoch, anneal_epochs
            )
            
            self.log("annealed_lambda_hsic_cross", self.lambda_hsic_cross, on_step=False, on_epoch=True)
            self.log("annealed_lambda_hsic_self", self.lambda_hsic_self, on_step=False, on_epoch=True)
        
        # 2. Group L1 annealing — start high, decay to final value
        if self._use_group_l1_annealing:

            idle = self._group_l1_anneal_idle
            transient = self._group_l1_anneal_transient
            start_val = float(self._group_l1_anneal_start)
            final_val = self._group_l1_final
            
            if epoch < idle:
                self.lambda_group_l1 = start_val
            elif epoch < idle + transient:
                self.lambda_group_l1 = self._linear_anneal(
                    start_val, final_val, epoch - idle, transient
                )
            else:
                self.lambda_group_l1 = final_val
            
            self.log("annealed_lambda_group_l1", self.lambda_group_l1, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):

        """Training step with optional gradient routing."""

        if self.use_gradient_routing:
            opt_recon, opt_struct = self.optimizers()
            
            total_loss, _, _, _ = self._step(batch=batch, stage="train")
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
            
            # Inject gradient noise for structural params (Langevin-style exploration)
            noise_std = getattr(self, '_structural_grad_noise_std', 0.0)
            if noise_std > 0.0:
                decay = getattr(self, '_structural_grad_noise_decay', 1.0)
                current_std = noise_std * (decay ** self.current_epoch)
                for p in self._structural_params:
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * current_std)
                self.log("structural_grad_noise_std", current_std, on_step=False, on_epoch=True)
            
            # Now step both optimizers (graph fully consumed, safe)
            opt_recon.step()
            opt_struct.step()
            
            self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_loss_recon_routed", loss_recon, on_step=False, on_epoch=True)
            self.log("train_loss_structural_routed", loss_structural, on_step=False, on_epoch=True)
        else:
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
        """Configure optimizer(s) with optional gradient routing.
        
        Structure optimizer can be configured independently via:
            structural_optimizer, structural_lr, structural_weight_decay,
            structural_optimizer_kwargs, structural_scheduler,
            structural_scheduler_kwargs, structural_gradient_noise
        """
        from causaliT.training.optimizer_factory import (
            make_optimizer, make_scheduler,
            get_recon_optimizer_config, get_structural_optimizer_config,
            get_structural_scheduler_config, get_gradient_noise_config,
        )
        
        tc = self.config["training"]
        max_epochs = tc.get("max_epochs", 1000)
        
        if self.use_gradient_routing:
            # --- Dual optimizer mode ---
            recon_cfg = get_recon_optimizer_config(tc)
            struct_cfg = get_structural_optimizer_config(tc)
            
            opt_recon = make_optimizer(self._reconstruction_params, **recon_cfg)
            opt_struct = make_optimizer(self._structural_params, **struct_cfg)
            
            # Store gradient noise config for use in training_step
            noise_cfg = get_gradient_noise_config(tc)
            self._structural_grad_noise_std = noise_cfg["noise_std"]
            self._structural_grad_noise_decay = noise_cfg["noise_decay"]
            
            # Structural scheduler (optional)
            sched_cfg = get_structural_scheduler_config(tc)
            struct_scheduler = make_scheduler(
                opt_struct, **sched_cfg, max_epochs=max_epochs
            )
            
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
            # --- Standard single optimizer (backward compatible) ---
            recon_cfg = get_recon_optimizer_config(tc)
            optimizer = make_optimizer(self.parameters(), **recon_cfg)
            
            if tc.get("use_scheduler", False):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
                }
            
            return optimizer
    
    def _compute_group_l1(self) -> tuple:
        """
        Compute Group L1 (L2,1 norm) on embedding columns - Group LASSO regularization.
        
        Returns:
            Tuple of:
            - group_l1_loss: Normalized L2,1 norm of embedding weights
            - effective_dims: Number of embedding dimensions with ||col||_2 > threshold
        """
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
