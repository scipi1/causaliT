"""
SingleCausalForecaster: PyTorch Lightning wrapper for SingleCausalLayer model.

This forecaster handles training, validation, and testing for the single-decoder
architecture focusing on S → X causal learning.

Active regularizers:
- Score sparsity (L1/entropy on attention scores) — applied to ALL decoder layers
- HSIC (independence between residuals and parents)
- Group L1 (embedding bottleneck)

Deprecated (removed):
- Acyclicity (NOTEARS) — partial causal ordering prevents cycles by construction
- KL divergence prior — no explicit phi parametrization in SVFA
- DAG sparsity (L1 on phi) — no explicit phi parametrization in SVFA
- Decisiveness — no explicit phi parametrization in SVFA
"""

from typing import Any, Dict, Optional
from os.path import join

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.utils import load_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token, hsic_per_x_pair, hsic_attention_weighted


class SingleCausalForecaster(pl.LightningModule):
    """
    Lightning wrapper for SingleCausalLayer transformer model.
    
    This forecaster manages training for a single causal relationship: S → X
    
    Features:
    - Single loss computation (MSE for X only)
    - Score sparsity regularization (L1/entropy) across ALL decoder layers
    - HSIC regularization for causal independence
    - Group L1 for embedding bottleneck
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
        model_kwargs = dict(config["model"]["kwargs"])
        if "attention_bypass" in config.get("model", {}):
            model_kwargs["attention_bypass"] = config["model"]["attention_bypass"]
        
        self.model = SingleCausalLayer(**model_kwargs)
        
        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        
        # Data indices for blanking values
        self.val_idx = config["data"]["val_idx"]
        
        # =====================================================================
        # UNIFIED SCORE SPARSITY REGULARIZATION
        # Applied to ALL decoder layers (averaged)
        # =====================================================================
        self.lambda_self_score_sparse = config["training"].get("lambda_self_score_sparse", 0.0)
        self.lambda_cross_score_sparse = config["training"].get("lambda_cross_score_sparse", 0.0)
        
        # Mode selection: "l1" or "entropy"
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
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        
        # =====================================================================
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        # =====================================================================
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 
                                   config["training"].get("lambda_embed_l1", 0.0))
        
        # =====================================================================
        # GRADIENT NORM LOGGING (for calibration stage)
        # =====================================================================
        self.log_gradient_norms = config["training"].get("log_gradient_norms", False)
        
        # =====================================================================
        # ANNEALING CONFIGURATION
        # =====================================================================
        
        # 1. Toeplitz Activation Temperature Annealing (tau_gate, tau_dir)
        #    Applied to ALL decoder layers
        self.use_tau_act_annealing = config["training"].get("use_tau_act_annealing", False)
        self.tau_gate_start = config["training"].get("tau_gate_start", 1.0)
        self.tau_gate_end = config["training"].get("tau_gate_end", 0.2)
        self.tau_dir_start = config["training"].get("tau_dir_start", 0.5)
        self.tau_dir_end = config["training"].get("tau_dir_end", 0.1)
        self.tau_act_anneal_epochs = config["training"].get("tau_act_anneal_epochs", None)
        
        # 2. HSIC Annealing - independent annealing for cross and self
        self.use_hsic_annealing = config["training"].get("use_hsic_annealing", False)
        self.hsic_anneal_epochs = config["training"].get("hsic_anneal_epochs", None)
        self.hsic_lambda_cross_start = config["training"].get("hsic_lambda_cross_start", self.lambda_hsic_cross)
        self.hsic_lambda_cross_end = config["training"].get("hsic_lambda_cross_end", 0.0)
        self.hsic_lambda_self_start = config["training"].get("hsic_lambda_self_start", self.lambda_hsic_self)
        self.hsic_lambda_self_end = config["training"].get("hsic_lambda_self_end", 0.0)
        
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
        
        # Unpack attention weights and entropies (lists, one element per decoder layer)
        dec_cross_att, dec_self_att = attention_weights
        dec_cross_ent, dec_self_ent = entropies
        
        # Compute loss for X
        x_target = torch.nan_to_num(x_val)
        mse_x_per_elem = self.loss_fn(pred_x.squeeze(), x_target.squeeze())
        loss_x = mse_x_per_elem.mean()
        
        # =====================================================================
        # SCORE SPARSITY REGULARIZATION (all decoder layers)
        # Accumulates score sparsity across ALL layers and averages.
        # This ensures all layers are regularized, not just layer 0.
        # =====================================================================
        n_layers = len(self.model.decoder.layers)
        total_self_score_sparse = torch.tensor(0.0, device=X.device)
        total_cross_score_sparse = torch.tensor(0.0, device=X.device)
        self_mode_used = "entropy"
        cross_mode_used = "entropy"
        
        def _compute_score_sparsity(mode: str, inner_attention, entropy_value, device, is_self: bool):
            """
            Compute unified score sparsity regularization.
            
            Args:
                mode: "l1" or "entropy"
                inner_attention: Inner attention module with score_tensor_for_sparsity property
                entropy_value: Pre-computed attention entropy (scalar)
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
        
        for layer_idx in range(n_layers):
            layer = self.model.decoder.layers[layer_idx]
            dec_self_inner = layer.global_self_attention.inner_attention
            dec_cross_inner = layer.global_cross_attention.inner_attention
            
            # Per-layer entropy (scalar)
            layer_self_ent = dec_self_ent[layer_idx].mean()
            layer_cross_ent = dec_cross_ent[layer_idx].mean()
            
            # Compute score sparsity for this layer
            layer_self_sparse, self_mode_used = _compute_score_sparsity(
                self.self_sparsity_regularizer, dec_self_inner, layer_self_ent, X.device, is_self=True
            )
            layer_cross_sparse, cross_mode_used = _compute_score_sparsity(
                self.cross_sparsity_regularizer, dec_cross_inner, layer_cross_ent, X.device, is_self=False
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
        # Encourages independence between residuals and parents (S for cross, X for self)
        # =====================================================================
        hsic_regularizer = 0.0
        hsic_cross_value = None
        hsic_self_value = None
        
        # Compute per-X residuals: (batch, seq_len_x)
        residuals_per_x = x_target.squeeze() - pred_x.squeeze()
        
        # S→X HSIC (cross-attention)
        s_values = S[:, :, self.val_idx]  # (batch, seq_len_s)
        
        if self.use_attention_weighted_hsic:
            att_cross_mean = dec_cross_att[0].mean(dim=0)  # (seq_len_x, seq_len_s)
            hsic_cross_value = hsic_attention_weighted(
                source_values=s_values,
                residuals=residuals_per_x,
                attention_weights=att_cross_mean,
                sigma=self.hsic_sigma,
                exclude_diagonal=False,
                adaptive_bandwidth=self.hsic_adaptive_bandwidth,
            )
        else:
            mean_residuals = residuals_per_x.mean(dim=1) if residuals_per_x.dim() > 1 else residuals_per_x
            hsic_cross_value = hsic_per_token(s_values, mean_residuals, sigma=self.hsic_sigma,
                                              adaptive_bandwidth=self.hsic_adaptive_bandwidth)
        
        hsic_regularizer += self.lambda_hsic_cross * hsic_cross_value
        
        # X→X HSIC (self-attention)
        x_values_for_hsic = x_target.squeeze()  # (batch, seq_len_x)
        
        if residuals_per_x.dim() > 1:
            if self.use_attention_weighted_hsic:
                att_self_mean = dec_self_att[0].mean(dim=0)  # (seq_len_x, seq_len_x)
                hsic_self_value = hsic_attention_weighted(
                    source_values=x_values_for_hsic,
                    residuals=residuals_per_x,
                    attention_weights=att_self_mean,
                    sigma=self.hsic_sigma,
                    exclude_diagonal=True,
                    adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                )
            else:
                hsic_self_value = hsic_per_x_pair(x_values_for_hsic, residuals_per_x,
                                                  sigma=self.hsic_sigma,
                                                  adaptive_bandwidth=self.hsic_adaptive_bandwidth)
            
            hsic_regularizer += self.lambda_hsic_self * hsic_self_value
        
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
            recon_grad_norm = _compute_grad_norm(loss_x, params_with_grad)
            self.log("train_recon_grad_norm", recon_grad_norm, on_step=False, on_epoch=True)
            
            # HSIC Cross
            hsic_cross_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_cross_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_cross_value is not None and isinstance(hsic_cross_value, torch.Tensor):
                hsic_cross_base_grad = _compute_grad_norm(hsic_cross_value, params_with_grad)
                hsic_cross_update_norm = self.lambda_hsic_cross * hsic_cross_base_grad
            
            self.log("train_hsic_cross_grad_norm", hsic_cross_base_grad, on_step=False, on_epoch=True)
            self.log("train_hsic_cross_update_norm", hsic_cross_update_norm, on_step=False, on_epoch=True)
            
            # HSIC Self
            hsic_self_base_grad = torch.tensor(0.0, device=x_target.device)
            hsic_self_update_norm = torch.tensor(0.0, device=x_target.device)
            
            if hsic_self_value is not None and isinstance(hsic_self_value, torch.Tensor):
                hsic_self_base_grad = _compute_grad_norm(hsic_self_value, params_with_grad)
                hsic_self_update_norm = self.lambda_hsic_self * hsic_self_base_grad
            
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
        total_loss = (loss_x + 
                     score_sparsity_regularizer +
                     hsic_regularizer +
                     group_l1_regularizer)
        
        # =====================================================================
        # LOGGING
        # =====================================================================
        
        # Main loss
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        
        # Reconstruction metrics
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred_x.reshape(-1), x_target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True, prog_bar=(stage == "val" and name == "mae"))
        
        # Score sparsity (averaged across layers)
        self.log(f"{stage}_self_score_sparse", avg_self_score_sparse, on_step=False, on_epoch=True)
        self.log(f"{stage}_cross_score_sparse", avg_cross_score_sparse, on_step=False, on_epoch=True)
        self.log(f"{stage}_self_sparsity_mode", 1.0 if self_mode_used == "l1" else 0.0, on_step=False, on_epoch=True)
        self.log(f"{stage}_cross_sparsity_mode", 1.0 if cross_mode_used == "l1" else 0.0, on_step=False, on_epoch=True)
        
        # HSIC
        if hsic_cross_value is not None:
            self.log(f"{stage}_hsic_cross", hsic_cross_value, on_step=False, on_epoch=True)
        if hsic_self_value is not None:
            self.log(f"{stage}_hsic_self", hsic_self_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_regularizer, on_step=False, on_epoch=True)
        
        # Group L1
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_group_l1_reg", group_l1_regularizer, on_step=False, on_epoch=True)
        if effective_dims is not None:
            self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)
        
        return total_loss, pred_x, X
    
    @staticmethod
    def _linear_anneal(start: float, end: float, epoch: int, total_epochs: int) -> float:
        """Linear annealing from start to end over total_epochs."""
        progress = min(1.0, epoch / max(1, total_epochs))
        return start + progress * (end - start)
    
    def on_train_epoch_start(self):
        """Apply annealing schedules at the start of each training epoch."""
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs if self.trainer else 100
        
        # 1. Toeplitz activation temperature annealing (ALL layers)
        if self.use_tau_act_annealing:
            anneal_epochs = self.tau_act_anneal_epochs or max_epochs
            new_tau_gate = self._linear_anneal(self.tau_gate_start, self.tau_gate_end, epoch, anneal_epochs)
            new_tau_dir = self._linear_anneal(self.tau_dir_start, self.tau_dir_end, epoch, anneal_epochs)
            
            for layer in self.model.decoder.layers:
                dec_self_inner = layer.global_self_attention.inner_attention
                
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
        
        # 2. HSIC annealing - independent for cross and self
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
