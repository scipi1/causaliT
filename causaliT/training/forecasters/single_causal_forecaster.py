"""
SingleCausalForecaster: PyTorch Lightning wrapper for SingleCausalLayer model.

This forecaster handles training, validation, and testing for the single-decoder
architecture focusing on S → X causal learning.
"""

from typing import Any, Dict, Optional
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.utils import load_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token
from causaliT.core.modules.extra_layers import dag_decisiveness_loss, dag_temperature_loss


class SingleCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for SingleCausalLayer transformer model.
    
    This forecaster manages training for a single causal relationship: S → X
    
    Features:
    - Single loss computation (MSE for X only)
    - Entropy and acyclicity regularization support
    - Hard mask support for enforcing ground-truth DAG structure
    - Designed for staged learning experiments
    
    Args:
        config: Configuration dictionary containing model, training, and data settings
        data_dir: Optional data directory for loading hard masks
    """
    def __init__(self, config, data_dir: str = None):
        super().__init__()
        
        self.config = config
        
        # Build model kwargs, adding attention_bypass from top-level config if present
        # This allows sweeping attention_bypass at the model.attention_bypass level
        # (sweeper requires 2-level nesting: category.parameter)
        model_kwargs = dict(config["model"]["kwargs"])
        if "attention_bypass" in config.get("model", {}):
            model_kwargs["attention_bypass"] = config["model"]["attention_bypass"]
        
        self.model = SingleCausalLayer(**model_kwargs)
        
        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # Logging configuration
        self.log_entropy = config["training"].get("log_entropy", False)
        self.log_acyclicity = config["training"].get("log_acyclicity", False)
        
        # Regularizers
        # Entropy regularization - encourages focused attention (low entropy)
        # Higher lambda = more penalty for high entropy = more focused attention
        self.lambda_entropy_self = config["training"].get("lambda_entropy_self", 0.0)
        self.lambda_entropy_cross = config["training"].get("lambda_entropy_cross", 0.0)
        self.kappa = config["training"].get("kappa", 0)   # Acyclicity regularization
        
        # Sparsity regularization - L1 penalty on edge probabilities
        self.lambda_sparse = config["training"].get("lambda_sparse", 0)
        self.lambda_sparse_cross = config["training"].get("lambda_sparse_cross", None)
        
        if self.lambda_sparse_cross is None:
            self.lambda_sparse_cross = self.lambda_sparse
        
        # Logging for sparsity
        self.log_sparsity = config["training"].get("log_sparsity", False)
        
        # L1 regularization on attention SCORES (works for ANY attention type, including ScaledDotAttention)
        # These are independent parameters for fine-grained control over sparsity
        self.lambda_l1_self_scores = config["training"].get("lambda_l1_self_scores", 0.0)
        self.lambda_l1_cross_scores = config["training"].get("lambda_l1_cross_scores", 0.0)
        self.log_l1_scores = config["training"].get("log_l1_scores", False)
        
        # HSIC regularization - encourages independence between S and residuals
        # Lower HSIC values indicate better causal structure learning
        self.lambda_hsic = config["training"].get("lambda_hsic", 0)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.log_hsic = config["training"].get("log_hsic", False)
        
        # KL divergence prior regularization
        # lambda_kl: scalar weight for the KL prior term
        # adaptive_z_scaling: if True, use SNR-based adaptive confidence (alpha)
        #                     if False, set alpha=1 (uniform confidence)
        self.lambda_kl = config["training"].get("lambda_kl", 0.0)
        self.adaptive_z_scaling = config["training"].get("adaptive_z_scaling", True)
        
        # DAG decisiveness regularization - encourages decisive edge probabilities (away from 0.5)
        # This is particularly important for DAGMaskAntisym where sigmoid(0) = 0.5
        # lambda_decisive: Weight for decisiveness (binary entropy) loss
        # lambda_tau: Weight for temperature penalty (encourages lower tau for sharper masks)
        # target_tau: Target temperature for annealing (no penalty below this value)
        self.lambda_decisive = config["training"].get("lambda_decisive", 0)
        self.lambda_decisive_cross = config["training"].get("lambda_decisive_cross", None)
        self.lambda_tau = config["training"].get("lambda_tau", 0)
        self.target_tau = config["training"].get("target_tau", 0.1)
        self.log_decisiveness = config["training"].get("log_decisiveness", False)
        
        # Embedding L1 regularization - limits embedding capacity for causal capacity assessment
        # Higher lambda_embed_l1 → sparser/lower-capacity embeddings
        # Used in ANS (Attention Necessity Score) evaluation to determine if attention is necessary
        self.lambda_embed_l1 = config["training"].get("lambda_embed_l1", 0.0)
        self.log_embed_l1 = config["training"].get("log_embed_l1", False)
        
        if self.lambda_decisive_cross is None:
            self.lambda_decisive_cross = self.lambda_decisive
        
        # =====================================================================
        # ANNEALING CONFIGURATION (all disabled by default)
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
        
        # L1 regularization on Toeplitz gate probabilities (disabled by default)
        self.lambda_l1_toeplitz_gate = config["training"].get("lambda_l1_toeplitz_gate", 0.0)
        
        # Logging for annealing
        self.log_tau_annealing = config["training"].get("log_tau_annealing", False)
        self.log_hsic_annealing = config["training"].get("log_hsic_annealing", False)
        
        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None
        
        # Register placeholder buffers if hard masks are enabled
        if self.use_hard_masks:
            self._register_hard_mask_placeholders()
        
        # Load hard masks if enabled and data_dir provided
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
        
        # dec_cross: S → X cross-attention (X_len queries, S_len keys)
        self.register_buffer('hard_mask_dec_cross', torch.zeros(X_len, S_len))
        # dec_self: X self-attention (X_len x X_len)
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
    
    def forward(self, data_source: torch.Tensor, data_intermediate: torch.Tensor,
                disable_hard_masks: bool = False) -> Any:
        """
        Forward pass through the model.
        
        Args:
            data_source: Source nodes (S)
            data_intermediate: Intermediate variables (X)
            disable_hard_masks: If True, disables hard masks even if model was trained with them.
            
        Returns:
            pred_x: Predicted X from decoder
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
        
        # Model forward pass
        pred_x, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=hard_masks,
        )
        
        return pred_x, attention_weights, masks, entropies
    
    def _step(self, batch, stage: str = None):
        """
        Common step logic for train/val/test.
        
        Args:
            batch: Tuple of (S, X) or (S, X, Y) tensors - Y is ignored if present
            stage: One of "train", "val", or "test"
            
        Returns:
            total_loss: Loss from X prediction
            pred_x: Predicted X values
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
        
        # Forward pass
        pred_x, attention_weights, masks, entropies = self.forward(
            data_source=S,
            data_intermediate=X
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
        
        # phi - learned DAGs
        dec_self_phi = getattr(dec_self_inner, 'phi', None)
        dec_cross_phi = getattr(dec_cross_inner, 'phi', None)
        
        # Running averages (detached, used as priors)
        dec_self_runav_mean = getattr(dec_self_inner, 'runav_att_mean', None)
        dec_self_runav_snr = getattr(dec_self_inner, 'runav_att_snr', None)
        dec_cross_runav_mean = getattr(dec_cross_inner, 'runav_att_mean', None)
        dec_cross_runav_snr = getattr(dec_cross_inner, 'runav_att_snr', None)
        
        # Entropy regularizer - encourages focused attention (low entropy)
        # Higher entropy = more uniform attention = more exploration
        # Lower entropy = more focused attention = more decisive
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
                        f"Acyclicity regularization only supports single-head attention. "
                        f"Decoder self-attention phi has shape {dec_self_phi.shape}, expected 2D tensor."
                    )
                acyclic_regularizer += self._notears_acyclicity(dec_self_phi)
            
            acyclic_regularizer = self.kappa * acyclic_regularizer
        else:
            acyclic_regularizer = 0.0
        
        # Prior regularizer with configurable scaling
        def _get_prior_reg(phi, evidence, alpha, use_adaptive_scaling, lambda_kl):
            """
            KL divergence between learned phi and empirical evidence.
            
            Args:
                phi: Learned DAG parameters
                evidence: Running average of attention (prior)
                alpha: SNR-based confidence (adaptive scaling)
                use_adaptive_scaling: If True, use alpha; if False, use 1.0
                lambda_kl: Scalar weight for the KL term
            """
            if phi is None or evidence is None:
                return 0.0
            _eps = 1E-6
            p = torch.sigmoid(phi)
            p0 = torch.sigmoid(evidence)
            
            # Apply adaptive scaling only if enabled and alpha is available
            if use_adaptive_scaling and alpha is not None:
                alpha_abs = torch.abs(alpha)
            else:
                alpha_abs = 1.0
            
            kl = (alpha_abs * (p * (torch.log(p + _eps) - torch.log(p0 + _eps)) + 
                              (1 - p) * (torch.log(1 - p + _eps) - torch.log(1 - p0 + _eps)))).mean()
            return lambda_kl * kl
        
        # Self and cross-attention prior regularization
        prior_regularizer = (
            _get_prior_reg(dec_self_phi, dec_self_runav_mean, dec_self_runav_snr, 
                          self.adaptive_z_scaling, self.lambda_kl) + 
            _get_prior_reg(dec_cross_phi, dec_cross_runav_mean, dec_cross_runav_snr,
                          self.adaptive_z_scaling, self.lambda_kl)
        )
        
        # Sparsity regularizer
        def _get_sparsity_reg(phi):
            """L1-style sparsity penalty: sum of edge probabilities."""
            if phi is None:
                return 0.0
            return torch.sigmoid(phi).mean()
        
        self_attention_sparsity = _get_sparsity_reg(dec_self_phi)
        cross_attention_sparsity = _get_sparsity_reg(dec_cross_phi)
        
        sparsity_regularizer = (
            self.lambda_sparse * self_attention_sparsity +
            self.lambda_sparse_cross * cross_attention_sparsity
        )
        
        # L1 regularization on attention SCORES
        # NOTE: This is INEFFECTIVE for ScaledDotProduct attention because post-softmax
        # weights always sum to 1, making the mean approximately constant (1/seq_len).
        # For ScaledDotProduct, use lambda_entropy_* instead for sparsity.
        # This regularizer is kept for attention types with learnable phi parameters.
        def _get_att_scores_l1(att_weights_list):
            """L1 penalty on attention weights (post-softmax/activation).
            
            NOTE: Ineffective for ScaledDotProduct attention - use entropy regularization instead.
            
            Args:
                att_weights_list: List of attention weight tensors from each layer
                
            Returns:
                Mean L1 penalty across all layers
            """
            if not att_weights_list:
                return 0.0
            # Use the last layer's attention weights (most relevant for output)
            # att_weights: (B, L, S) or (B, H, L, S)
            return att_weights_list[-1].mean()
        
        # L1 on self-attention scores (dec_self)
        l1_self_scores = _get_att_scores_l1(dec_self_att)
        
        # L1 on cross-attention scores (dec_cross)
        l1_cross_scores = _get_att_scores_l1(dec_cross_att)
        
        # Total L1 scores regularizer
        l1_scores_regularizer = (
            self.lambda_l1_self_scores * l1_self_scores +
            self.lambda_l1_cross_scores * l1_cross_scores
        )
        
        # Compute loss for X
        x_target = torch.nan_to_num(x_val)
        mse_x_per_elem = self.loss_fn(pred_x.squeeze(), x_target.squeeze())
        loss_x = mse_x_per_elem.mean()
        
        # HSIC regularizer - encourages independence between S and residuals
        # Lower HSIC = less dependence = better causal structure learning
        if self.lambda_hsic > 0 or self.log_hsic:
            # Compute residuals: (batch, seq_len_y)
            residuals = x_target.squeeze() - pred_x.squeeze()
            
            # Mean residuals across sequence length: (batch,)
            if residuals.dim() > 1:
                mean_residuals = residuals.mean(dim=1)
            else:
                mean_residuals = residuals
            
            # Extract S values: (batch, seq_len_s)
            s_values = S[:, :, self.val_idx]
            
            # Compute HSIC between each S token and mean residuals
            hsic_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma)
            hsic_regularizer = self.lambda_hsic * hsic_value
        else:
            hsic_regularizer = 0.0
            hsic_value = None
        
        # DAG Decisiveness regularizer - encourages edge probabilities away from 0.5
        # This addresses the problem with antisymmetric DAG parameterization where
        # sigmoid(0) = 0.5, leading to indecisive edges
        decisive_self_loss = torch.tensor(0.0, device=x_target.device)
        decisive_cross_loss = torch.tensor(0.0, device=x_target.device)
        tau_self_loss = torch.tensor(0.0, device=x_target.device)
        tau_cross_loss = torch.tensor(0.0, device=x_target.device)
        
        if self.lambda_decisive > 0 or self.lambda_tau > 0 or self.log_decisiveness:
            # Self-attention decisiveness
            if dec_self_phi is not None:
                # Get Gumbel-Softmax temperature for self-attention (log_tau_gs is used for DAG mask)
                log_tau_gs_self = getattr(dec_self_inner, 'log_tau_gs', None)
                tau_gs_self = torch.exp(log_tau_gs_self) if log_tau_gs_self is not None else None
                
                # Exclude diagonal for self-attention (it's always 0 for antisymmetric)
                is_square = dec_self_phi.shape[-2] == dec_self_phi.shape[-1]
                decisive_self_loss = dag_decisiveness_loss(
                    dec_self_phi, tau=tau_gs_self, exclude_diagonal=is_square
                )
                
                # Temperature penalty for self-attention
                if log_tau_gs_self is not None and self.lambda_tau > 0:
                    tau_self_loss = dag_temperature_loss(log_tau_gs_self, target_tau=self.target_tau)
            
            # Cross-attention decisiveness
            if dec_cross_phi is not None:
                # Get Gumbel-Softmax temperature for cross-attention (log_tau_gs is used for DAG mask)
                log_tau_gs_cross = getattr(dec_cross_inner, 'log_tau_gs', None)
                tau_gs_cross = torch.exp(log_tau_gs_cross) if log_tau_gs_cross is not None else None
                
                # Cross-attention is not square, so no diagonal to exclude
                decisive_cross_loss = dag_decisiveness_loss(
                    dec_cross_phi, tau=tau_gs_cross, exclude_diagonal=False
                )
                
                # Temperature penalty for cross-attention
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
        
        # Embedding L1 regularizer - limits embedding capacity
        # Used for Attention Necessity Score (ANS) evaluation
        if self.lambda_embed_l1 > 0 or self.log_embed_l1:
            embed_l1_loss = self._compute_embedding_l1()
            embed_l1_regularizer = self.lambda_embed_l1 * embed_l1_loss
        else:
            embed_l1_loss = torch.tensor(0.0, device=x_target.device)
            embed_l1_regularizer = 0.0
        
        # Total loss
        total_loss = (loss_x + 
                     entropy_regularizer + 
                     acyclic_regularizer +
                     prior_regularizer +
                     sparsity_regularizer +
                     l1_scores_regularizer +
                     hsic_regularizer +
                     decisiveness_regularizer +
                     l1_toeplitz_gate_reg +
                     embed_l1_regularizer)
        
        # Log loss
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Log metrics for X reconstruction
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred_x.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val" and name == "mae"))
        
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
            self.log(f"{stage}_sparsity_total", sparsity_regularizer, on_step=False, on_epoch=True)
        
        # Log L1 scores regularization if requested
        if self.log_l1_scores:
            self.log(f"{stage}_l1_self_scores", l1_self_scores, on_step=False, on_epoch=True)
            self.log(f"{stage}_l1_cross_scores", l1_cross_scores, on_step=False, on_epoch=True)
        
        # Log HSIC if requested
        if self.log_hsic and hsic_value is not None:
            self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
            self.log(f"{stage}_hsic_reg", hsic_regularizer, on_step=False, on_epoch=True)
        
        # Log decisiveness if requested
        if self.log_decisiveness:
            self.log(f"{stage}_decisive_self", decisive_self_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_decisive_cross", decisive_cross_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_tau_self", tau_self_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_tau_cross", tau_cross_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_decisive_total", decisiveness_regularizer, on_step=False, on_epoch=True)
            # Log actual Gumbel-Softmax temperature values for monitoring
            # log_tau_gs is the Gumbel-Softmax temperature used for DAG mask sampling
            if dec_self_phi is not None:
                log_tau_gs_self = getattr(dec_self_inner, 'log_tau_gs', None)
                if log_tau_gs_self is not None:
                    self.log(f"{stage}_tau_gs_self_value", torch.exp(log_tau_gs_self), on_step=False, on_epoch=True)
            if dec_cross_phi is not None:
                log_tau_gs_cross = getattr(dec_cross_inner, 'log_tau_gs', None)
                if log_tau_gs_cross is not None:
                    self.log(f"{stage}_tau_gs_cross_value", torch.exp(log_tau_gs_cross), on_step=False, on_epoch=True)
        
        # Log embedding L1 if requested
        if self.log_embed_l1:
            self.log(f"{stage}_embed_l1", embed_l1_loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_embed_l1_reg", embed_l1_regularizer, on_step=False, on_epoch=True)
        
        return total_loss, pred_x, X
    
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
        loss, _, _ = self._step(batch=batch, stage="train")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, _, _ = self._step(batch=batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, _, _ = self._step(batch=batch, stage="test")
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
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adamw' or 'sgd'.")
        
        if self.config["training"].get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
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
        """
        NOTEARS acyclicity constraint.
        
        Args:
            A: (d, d) adjacency matrix (non-negative entries)
            
        Returns:
            Scalar acyclicity penalty
        """
        d = A.shape[0]
        expm_A = torch.matrix_exp(torch.relu(A))
        return torch.trace(expm_A) - d
    
    def _compute_embedding_l1(self) -> torch.Tensor:
        """
        Compute L1 norm of X embedding parameters.
        
        This is used for capacity control in Attention Necessity Score (ANS) evaluation.
        Higher L1 penalty → sparser embeddings → reduced capacity → forces attention to be useful.
        
        Only applies to learnable X embeddings (embedding_X), not to orthogonal S embeddings
        which are frozen by design.
        
        Returns:
            torch.Tensor: Sum of absolute values of all X embedding parameters, 
                         normalized by the number of parameters.
        """
        l1_sum = torch.tensor(0.0, device=next(self.model.embedding_X.parameters()).device)
        n_params = 0
        
        for p in self.model.embedding_X.parameters():
            if p.requires_grad:  # Only count trainable parameters
                l1_sum = l1_sum + p.abs().sum()
                n_params += p.numel()
        
        # Normalize by number of parameters to make λ scale-independent
        if n_params > 0:
            l1_sum = l1_sum / n_params
        
        return l1_sum
