"""
SingleCausalForecaster: PyTorch Lightning wrapper for SingleCausalLayer model.

This forecaster handles training, validation, and testing for the single-decoder
architecture focusing on S → X causal learning.

Active regularizers:
- Score sparsity (L1/entropy on attention scores) — applied to ALL decoder layers
- HSIC (independence between residuals and parents)
- Group L1 (embedding bottleneck)
- Acyclicity (NOTEARS, ``training.kappa``) — applied to the directed
  edge matrix of every decoder layer's self-attention. Closes the cycle
  hole inherent to ToeplitzAttention (which only suppresses 2-cycles by
  construction). Uses ``inner_attention.score_tensor_for_sparsity`` as
  the directed att matrix; falls back to ``inner_attention.phi``.

Deprecated (removed):
- KL divergence prior — no explicit phi parametrization in SVFA
- DAG sparsity (L1 on phi) — no explicit phi parametrization in SVFA
- Decisiveness — no explicit phi parametrization in SVFA
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

from causaliT.core.architectures.single_causal import SingleCausalLayer
from causaliT.core.utils import load_dag_masks, corrupt_dag_masks
from causaliT.utils.hsic_utils import hsic_per_token, hsic_per_x_pair, hsic_attention_weighted, hsic_cross_per_pair
from causaliT.training.gradient_routing import classify_parameters


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
        # Auto-detects L1 vs entropy based on attention type:
        #   - L1 (score_tensor_for_sparsity) for non-softmax attention
        #   - Entropy fallback for softmax-based attention only
        # Config keys self_sparsity_regularizer / cross_sparsity_regularizer
        # are DEPRECATED and ignored.
        # =====================================================================
        self.lambda_self_score_sparse = config["training"].get("lambda_self_score_sparse", 0.0)
        self.lambda_cross_score_sparse = config["training"].get("lambda_cross_score_sparse", 0.0)
        
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
        # HSIC cross-attention mode: "averaged" (legacy) or "per_variable" (edge-level)
        # "averaged": HSIC(S_i, mean(residuals)) — weak, diluted signal
        # "per_variable": HSIC(S_i, res_j) for all (i,j) pairs — edge-specific signal
        self.hsic_cross_mode = config["training"].get("hsic_cross_mode", "averaged")
        # Source kernel: "rbf" (default, for continuous S) or "dirac" (for discrete S)
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")
        
        # =====================================================================
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        # =====================================================================
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 
                                   config["training"].get("lambda_embed_l1", 0.0))

        # =====================================================================
        # ACYCLICITY REGULARIZATION (NOTEARS) — applied to self-attention only
        # =====================================================================
        # Closes the cycle hole inherent to ToeplitzAttention: while Toeplitz
        # guarantees att[i,j] + att[j,i] = sigmoid(S/τ) (suppressing 2-cycles),
        # nothing prevents 3-cycles or longer in X self-attention. NOTEARS
        # adds tr(exp(A ⊙ A)) − d, which is 0 iff A induces a DAG.
        # Applied to every decoder layer's self-attention directed edge matrix
        # (sourced from `score_tensor_for_sparsity`, falling back to `phi`).
        # Cross-attention is bipartite (S → X) and inherently acyclic, so no
        # NOTEARS term is added there.
        self.kappa = float(config["training"].get("kappa", 0.0))
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        
        # =====================================================================
        # RECONSTRUCTION-LOSS WEIGHT (knob to disable recon for HSIC-only runs)
        # =====================================================================
        # `lambda_recon` ∈ [0, ∞) scales the MSE reconstruction term in the
        # total loss. Default 1.0 = legacy behaviour. Setting to 0.0 (with
        # `use_gradient_routing=False` and a non-zero `lambda_hsic_*`) yields
        # an HSIC-only single-loss training, useful as a dependence-only
        # ablation against the joint MSE+HSIC training.
        self.lambda_recon = float(config["training"].get("lambda_recon", 1.0))
        if self.lambda_recon < 0.0:
            raise ValueError(
                f"lambda_recon must be non-negative, got {self.lambda_recon}"
            )

        # =====================================================================
        # STRUCTURAL LOSS RECON ANCHOR (convex mix)
        # =====================================================================
        # alpha in [0, 1]: convex combination weight on the structural pathway.
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg
        # Reconstruction loss is unchanged (alpha leaks recon into structure only).
        # Default 0.0 → exact backward compatibility (pure HSIC).
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
        
        # 2b. Unified attention temperature annealing (self + cross attentions)
        #     Mirrors the group-L1 anneal pattern.
        #     Schedule:
        #       [0, idle)                 -> tau = tau_start  (frozen)
        #       [idle, idle+transient)    -> tau linearly decays start -> end (frozen)
        #       [idle+transient, end)     -> tau is unfrozen and learnable from `end`
        #     Targets the following parameters when present on each attention module:
        #       - ToeplitzLieAttention   : `log_tau_gate`, `log_tau_dir`
        #     Note (iter_10+): ``ToeplitzAttention.tau`` and the cross-attentions'
        #     ``tau`` are now non-learnable Python floats (``init_tau``), so they
        #     are NOT touched by this annealer.

        #     freeze_tau_during_anneal=True: requires_grad disabled during idle+transient,
        #     re-enabled exactly once at the boundary.
        self.use_tau_annealing = config["training"].get("use_tau_annealing", False)
        self.tau_anneal_start = float(config["training"].get("tau_anneal_start", 1.0))
        self.tau_anneal_end = float(config["training"].get("tau_anneal_end", 0.2))
        self.tau_anneal_idle_epochs = int(config["training"].get("tau_anneal_idle_epochs", 0))
        self.tau_anneal_transient_epochs = int(
            config["training"].get("tau_anneal_transient_epochs", 0)
        )
        self.freeze_tau_during_anneal = bool(
            config["training"].get("freeze_tau_during_anneal", True)
        )
        # Tracks whether log_tau* params have been re-enabled (avoid repeated work)
        self._tau_unfrozen = False
        
        # 3. Group L1 Annealing — start high, decay to lambda_group_l1

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
            self.automatic_optimization = False  # Manual optimization for dual optimizer
            structural_params, reconstruction_params = classify_parameters(
                self.model, verbose=True
            )
            self._structural_params = structural_params
            self._reconstruction_params = reconstruction_params
        
        # Hard mask configuration
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        self._hard_masks = None

        # Oracle attention configuration
        # When True, the model bypasses QK^T entirely and uses the hard mask
        # directly as the attention score matrix for the values. The alignment
        # is given (not learned), and the model only learns the rest from the
        # reconstruction loss. Requires use_hard_masks=True.
        self.use_oracle_attention = config["training"].get("use_oracle_attention", False)
        if self.use_oracle_attention and not self.use_hard_masks:
            raise ValueError(
                "training.use_oracle_attention=True requires training.use_hard_masks=True. "
                "The oracle uses the hard mask itself as the attention score matrix."
            )

        # Wrong-DAG oracle controls (replaces the legacy `permute_hard_masks_seed`).
        # `hard_masks_corruption_seed` in {None, 0}  OR  both SHDs == 0
        # disable corruption and the ground-truth masks are used unchanged.
        # See `causaliT.core.utils.corrupt_dag_masks` for full semantics
        # (incl. the "shd > num_true_edges → all-edges-must-be-wrong" fallback).
        self.hard_masks_corruption_seed = config["training"].get("hard_masks_corruption_seed", None)
        self.cross_control_shd = int(config["training"].get("cross_control_shd", 0) or 0)
        self.self_control_shd = int(config["training"].get("self_control_shd", 0) or 0)
        # When True, corruption is restricted to one-for-one edge swaps so the
        # corrupted mask retains the SAME number of edges as the ground truth.
        # See `causaliT.core.utils.corrupt_dag_masks` for full semantics.
        self.hard_masks_preserve_sparsity = bool(
            config["training"].get("hard_masks_preserve_sparsity", False)
        )
        # Filled in by `_load_hard_masks` when corruption is applied.
        self.hard_mask_corruption_info: Optional[Dict[str, dict]] = None
        
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
            # Optional wrong-DAG oracle: corrupt each mask by exactly `shd`
            # entries (off-diagonal pool for self masks). Cycle status of the
            # corrupted self mask is detected and stashed for later logging.
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
                    f"preserve_sparsity={self.hard_masks_preserve_sparsity}) "
                    f"— wrong-DAG oracle."
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

        # Oracle mode follows the same gating as hard masks: if hard masks are
        # disabled (e.g. evaluation w/o GT), oracle is also disabled so the
        # forward pass falls back to the learned attention.
        oracle = self.use_oracle_attention and apply_hard_masks
        
        # Model forward pass
        pred_x, attention_weights, masks, entropies = self.model.forward(
            source_tensor=data_source,
            intermediate_tensor_blanked=x_blanked,
            hard_masks=hard_masks,
            oracle=oracle,
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
        
        def _compute_score_sparsity(inner_attention, entropy_value, device):
            """
            Compute score sparsity regularization with auto-detection.
            
            Auto-selects the method based on what the attention module supports:
            - L1 on score_tensor_for_sparsity if available (non-softmax attention)
            - Entropy fallback for softmax-based attention (no score tensor)
            
            Config keys self_sparsity_regularizer / cross_sparsity_regularizer
            are DEPRECATED and ignored.
            
            Args:
                inner_attention: Inner attention module (may have score_tensor_for_sparsity)
                entropy_value: Pre-computed attention entropy (scalar, used as fallback)
                device: Device for tensors
                
            Returns:
                Tuple of (sparsity_value, actual_mode_used)
            """
            # Auto-detect: always prefer L1 if available, fall back to entropy
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
        # Encourages independence between residuals and parents (S for cross, X for self)
        # =====================================================================
        hsic_regularizer = 0.0
        hsic_cross_value = None
        hsic_self_value = None
        
        # Compute per-X residuals: (batch, seq_len_x)
        residuals_per_x = x_target.squeeze() - pred_x.squeeze()
        
        # S→X HSIC (cross-attention)
        s_values = S[:, :, self.val_idx]  # (batch, seq_len_s)

        # =====================================================================
        # KERNEL DIAGNOSTICS: residual scale & median pairwise distances.
        # Cheap (subsampled) per-step diagnostics; aggregated to epoch by Lightning.
        # Used to diagnose HSIC kernel collapse:
        #   * fixed-σ:   residual_std ≪ σ  →  L kernel collapses to all-ones
        #   * adaptive:  σ tracks pairdist_med, so |σ_S − pairdist_med_X| informs scale-mismatch
        # Names:
        #   {stage}_residual_std         : std over all (batch, X) residual entries
        #   {stage}_residual_pairdist_med: mean over X dims of median |r_i − r_j| (subsampled)
        #   {stage}_s_pairdist_med       : same for S values (subsampled)
        # =====================================================================
        with torch.no_grad():
            res_det = residuals_per_x.detach()
            self.log(
                f"{stage}_residual_std",
                res_det.float().std(),
                on_step=False, on_epoch=True,
            )
            B = res_det.shape[0]
            n_sub = min(B, 256)
            if n_sub > 1:
                if B > n_sub:
                    idx = torch.randperm(B, device=res_det.device)[:n_sub]
                    res_sub = res_det[idx]
                    s_sub = s_values.detach()[idx]
                else:
                    res_sub = res_det
                    s_sub = s_values.detach()

                # Build the upper-triangular index mask on CPU because the
                # subsequent median is computed on CPU as well — see _med_pairdist.
                triu_mask_cpu = torch.triu(
                    torch.ones(n_sub, n_sub, dtype=torch.bool),
                    diagonal=1,
                )

                def _med_pairdist(t: torch.Tensor) -> torch.Tensor:
                    """Median |t_i − t_j| (i<j), averaged across feature dim if any.

                    Computed on CPU because `torch.median(dim=...)` is non-deterministic
                    on CUDA when ``torch.use_deterministic_algorithms(True)`` is set
                    (Lightning's default in this project) — it would otherwise raise
                    `RuntimeError: median CUDA with indices output does not have a
                    deterministic implementation`. Subsample is ≤256 rows so the CPU
                    detour is sub-millisecond.
                    """
                    t_cpu = t.detach().cpu()
                    if t_cpu.dim() == 1:
                        d = (t_cpu.unsqueeze(0) - t_cpu.unsqueeze(1)).abs()
                        return d[triu_mask_cpu].median()
                    # t: (n, F) → diffs: (n, n, F)
                    d = (t_cpu.unsqueeze(0) - t_cpu.unsqueeze(1)).abs()
                    triu_vals = d[triu_mask_cpu]  # (n*(n-1)/2, F)
                    return triu_vals.median(dim=0).values.mean()

                self.log(
                    f"{stage}_residual_pairdist_med",
                    _med_pairdist(res_sub),
                    on_step=False, on_epoch=True,
                )
                self.log(
                    f"{stage}_s_pairdist_med",
                    _med_pairdist(s_sub),
                    on_step=False, on_epoch=True,
                )
        
        if self.use_attention_weighted_hsic:
            att_cross_mean = dec_cross_att[0].mean(dim=0)  # (seq_len_x, seq_len_s)
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
                    mode=self.hsic_mode,
                    nhsic_epsilon=self.nhsic_epsilon,
                )
            else:
                hsic_self_value = hsic_per_x_pair(x_values_for_hsic, residuals_per_x,
                                                  sigma=self.hsic_sigma,
                                                  adaptive_bandwidth=self.hsic_adaptive_bandwidth,
                                                  mode=self.hsic_mode,
                                                  nhsic_epsilon=self.nhsic_epsilon)
            
            hsic_regularizer += self.lambda_hsic_self * hsic_self_value
        
        # =====================================================================
        # GROUP L1 REGULARIZATION (L2,1 norm on embedding columns)
        # =====================================================================
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_regularizer = self.lambda_group_l1 * group_l1_loss

        # =====================================================================
        # ACYCLICITY REGULARIZATION (NOTEARS) — self-attention only
        # =====================================================================
        # For each decoder layer's self-attention we read the directed edge
        # matrix A (shape (L, L), batch-mean) from
        #   inner.score_tensor_for_sparsity   (preferred — exposed by
        #                                      ToeplitzAttention as the
        #                                      directed att = sigmoid(S/τ)·sigmoid(A/τ))
        # falling back to inner.phi for legacy DAG-learning attentions.
        # h(A) = tr(exp(A ⊙ A)) − d   is 0 iff A induces a DAG.
        # We sum over layers and divide by the number of layers that
        # actually contributed a 2-D matrix (single-head only — for
        # multi-head the matrix becomes (H, L, L) and is skipped, mirroring
        # the constraint in stage_causal_forecaster).
        if self.kappa > 0.0:
            acyclic_regularizer = torch.tensor(0.0, device=X.device)
            n_acyclic_layers = 0
            for layer_idx in range(n_layers):
                layer = self.model.decoder.layers[layer_idx]
                inner = layer.global_self_attention.inner_attention
                A_self = getattr(inner, "score_tensor_for_sparsity", None)
                if A_self is None:
                    A_self = getattr(inner, "phi", None)
                if A_self is None or A_self.dim() != 2:
                    continue
                acyclic_regularizer = acyclic_regularizer + self._notears_acyclicity(A_self)
                n_acyclic_layers += 1
            if n_acyclic_layers > 0:
                acyclic_regularizer = (
                    self.kappa * acyclic_regularizer / n_acyclic_layers
                )
        else:
            acyclic_regularizer = torch.tensor(0.0, device=X.device)
        
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
        # `lambda_recon` (default 1.0) scales the recon term so that setting
        # it to 0 yields a pure-HSIC training objective. The MSE/MAE/R² metrics
        # still log the unscaled recon for monitoring.
        total_loss = (self.lambda_recon * loss_x +
                     score_sparsity_regularizer +
                     hsic_regularizer +
                     group_l1_regularizer +
                     acyclic_regularizer)
        
        # Store decomposed losses for gradient routing (used by training_step).
        # Convex mix on the structural pathway only:
        #   L_struct = (1 - alpha) * HSIC_reg + alpha * loss_recon
        #              + score_sparsity_reg + group_l1_reg + acyclic_reg
        # Reconstruction loss (loss_recon) is unchanged.
        # NOTEARS rides on the structural pathway since its only effect is on
        # the self-attention edge matrix (= structural params).
        alpha = self.lambda_struct_recon
        self._last_loss_components = {
            "loss_recon": loss_x,
            "loss_structural": (
                (1.0 - alpha) * hsic_regularizer
                + alpha * loss_x
                + score_sparsity_regularizer
                + group_l1_regularizer
                + acyclic_regularizer
            ),
        }

        
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

        # NOTEARS acyclicity (auto-discovered by eval_training.py via "notears" key)
        self.log(f"{stage}_notears", acyclic_regularizer, on_step=False, on_epoch=True)
        
        return total_loss, pred_x, X
    
    @staticmethod
    def _linear_anneal(start: float, end: float, epoch: int, total_epochs: int) -> float:
        """Linear annealing from start to end over total_epochs."""
        progress = min(1.0, epoch / max(1, total_epochs))
        return start + progress * (end - start)

    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS acyclicity penalty h(A) = tr(exp(A ⊙ A)) - d.

        Zero iff A induces a directed acyclic graph (Zheng et al., 2018).
        Single-head only — caller is responsible for ensuring A is 2-D
        (shape (L, L)). Mirrors ``stage_causal_forecaster._notears_acyclicity``.
        """
        d = A.shape[-1]
        return torch.trace(torch.matrix_exp(A * A)) - d
    
    # ------------------------------------------------------------------
    # Used-mask dump (fires once per fit)
    # ------------------------------------------------------------------
    def on_fit_start(self):
        """Persist the *actual* hard masks used in training to disk.

        Always runs when ``use_hard_masks=True``, regardless of corruption.
        Output goes to ``<run_dir_k>/used_masks/`` so that GT, corrupted, and
        any future programmatic mask is always reproducible from the run dir.

        Also writes a ``mask_summary.json`` with corruption metadata (if any).
        """
        if not self.use_hard_masks or not self._hard_masks_loaded:
            return

        # Resolve run directory: CSVLogger.log_dir is "<save_dir_k>/logs/csv/version_X"
        # so parents[2] = "<save_dir_k>".
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

        # Build a small json summary
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
        
        # 3. Group L1 annealing — start high, decay to final value
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
        
        # 4. Unified attention temperature annealing (self + cross)
        #    Schedule overrides log_tau* params during [0, idle+transient).
        #    After the schedule ends: requires_grad is re-enabled (Option A).
        if self.use_tau_annealing:
            idle = self.tau_anneal_idle_epochs
            transient = self.tau_anneal_transient_epochs
            tau_start = self.tau_anneal_start
            tau_end = self.tau_anneal_end
            schedule_end = idle + transient
            
            in_schedule = epoch < schedule_end
            
            if in_schedule:
                if epoch < idle:
                    new_tau = tau_start
                else:
                    new_tau = self._linear_anneal(
                        tau_start, tau_end, epoch - idle, transient
                    )
                
                # Walk all decoder layers, set tau on self + cross attention.
                # Two paths, applied side-by-side:
                #   (a) Learnable log_tau Parameters of ToeplitzLieAttention
                #       (``log_tau_gate`` / ``log_tau_dir``) — fill with the
                #       log of the new tau.
                #   (b) Constant Python-float ``tau`` attributes of
                #       ToeplitzAttention / CausalCrossAttention /
                #       SigmoidCrossAttention (introduced in iter_10) —
                #       overwrite the float in place. No autograd, no
                #       requires_grad games. iter_11+ uses this path to
                #       anneal the constant taus from beginning to end.
                tau_param_names = ("log_tau_gate", "log_tau_dir")
                log_new_tau = float(torch.log(torch.tensor(new_tau)))

                for layer in self.model.decoder.layers:
                    for inner in (layer.global_self_attention.inner_attention,
                                  layer.global_cross_attention.inner_attention):
                        # (a) learnable log_tau* Parameters
                        for pname in tau_param_names:
                            p = getattr(inner, pname, None)
                            if p is None:
                                continue
                            with torch.no_grad():
                                p.fill_(log_new_tau)
                            if self.freeze_tau_during_anneal:
                                p.requires_grad = False
                        # (b) constant float ``tau`` (iter_10+ attentions)
                        tau_attr = getattr(inner, "tau", None)
                        if tau_attr is not None and not isinstance(tau_attr, torch.Tensor):
                            inner.tau = float(new_tau)

                self.log("annealed_tau", new_tau, on_step=False, on_epoch=True)

            elif self.freeze_tau_during_anneal and not self._tau_unfrozen:
                # One-shot unfreeze at boundary (iter_10+ list — see above)
                tau_param_names = ("log_tau_gate", "log_tau_dir")

                for layer in self.model.decoder.layers:
                    for inner in (layer.global_self_attention.inner_attention,
                                  layer.global_cross_attention.inner_attention):
                        for pname in tau_param_names:
                            p = getattr(inner, pname, None)
                            if p is None:
                                continue
                            p.requires_grad = True
                self._tau_unfrozen = True
                self.log("annealed_tau", tau_end, on_step=False, on_epoch=True)
            else:
                self.log("annealed_tau", tau_end, on_step=False, on_epoch=True)

    
    def training_step(self, batch, batch_idx):
        """Training step with optional gradient routing.
        
        When gradient routing is enabled (use_gradient_routing=True):
        - Reconstruction loss updates only reconstruction parameters (θ_R)
        - Structural loss (HSIC + score sparsity + group L1) updates only structural parameters (θ_S)
        - Uses manual optimization with two separate backward passes
        
        When gradient routing is disabled (default):
        - Standard single-optimizer training with automatic optimization
        """
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
            # --- Standard automatic optimization ---
            total_loss, _, _ = self._step(batch=batch, stage="train")
            self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            return total_loss
    
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
        """Configure optimizer(s).
        
        When gradient routing is enabled, creates two optimizers:
        - opt_recon: updates reconstruction parameters (θ_R) with reconstruction loss
        - opt_struct: updates structural parameters (θ_S) with structural loss
        
        Structure optimizer can be configured independently via:
            structural_optimizer, structural_lr, structural_weight_decay,
            structural_optimizer_kwargs, structural_scheduler,
            structural_scheduler_kwargs, structural_gradient_noise
        
        When gradient routing is disabled (default), creates a single optimizer
        over all parameters (backward compatible).
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
                    optimizer, mode='min', factor=0.5, patience=10, verbose=True
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
