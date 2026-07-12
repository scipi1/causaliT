"""
SelfSelectorForecaster: PyTorch Lightning wrapper for SelfSelectorLayer.

Research objective
==================
Test whether a single direction-aware self-attention block can recover the WHOLE
directed acyclic graph over N = L_S + L_X variables from observational data —
WITHOUT assuming which variables are sources.  Every node is embedded once (a
single shared embedding over the concatenated [S, X] set), used both as a
value-blanked query (candidate child) and an actual-value key/value (candidate
parent), and the ``GatedSelfAttention`` block produces the full ``(N, N)``
directed posterior via the Toeplitz symmetric-existence x antisymmetric-direction
x reconstruction-gain factorisation.

Design differences from AttentionSelectorForecaster
===================================================
1. **Homogeneous N-node set.**  ``all = cat([S, X], dim=1)``.  X variable-ids are
   offset by ``x_var_id_offset`` (default ``S_seq_len``) so S and X share ONE
   contiguous id namespace for the single shared embedding table.

2. **All-N reconstruction.**  MSE is computed over ALL N node values, not just X.

3. **Full-matrix HSIC.**  HSIC(source_j, residual_i) over all (i, j) with the
   combined source = all N node values.

4. **NOTEARS over the FULL (N, N) score** — the whole graph must be acyclic,
   so the penalty is applied to the entire directed edge matrix (not just an
   X->X sub-block).

Gradient routing, L0 regularisation, score sparsity, group-L1, BKD checkpoint
handling and the L0<->HSIC interference diagnostic are reused unchanged: the
structural params are the Q/K projections + structural embeddings; the
reconstruction params are the ``gain_*`` identity tables + V/out/FFN/MLP head
(the ``gain_`` name prefix routes them correctly).

Evaluation
==========
``model.split_attention(A)`` returns the four GT-comparable blocks
(``s_to_x``, ``x_to_x``, ``x_to_s``, ``s_to_s``); ``model.source_scores(A)``
returns per-node incoming-edge mass (low => likely source).
"""

import logging
import math
from os.path import join
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm

from causaliT.core.architectures.self_selector import SelfSelectorLayer
from causaliT.core.utils import load_dag_masks
from causaliT.utils.hsic_utils import hsic_cross_per_pair
from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.interference_utils import (
    build_interference_blocks,
    compute_l0_hsic_interference,
)

logger = logging.getLogger(__name__)


class SelfSelectorForecaster(pl.LightningModule):
    """Lightning wrapper for SelfSelectorLayer (homogeneous whole-graph discovery)."""

    def __init__(self, config: dict, data_dir: str = None):
        super().__init__()

        self.config = config
        self.model = SelfSelectorLayer(**config["model"]["kwargs"])

        # Data indices
        self.val_idx = config["data"]["val_idx"]
        self.var_idx = config["data"].get("feature_indices", {}).get("variable", None)
        self.S_seq_len = int(config["data"]["S_seq_len"])
        self.X_seq_len = int(config["data"]["X_seq_len"])
        self.N = self.S_seq_len + self.X_seq_len

        # Offset applied to X variable-ids so S and X share one id namespace for
        # the single shared embedding.  Default: shift X by the number of S
        # variables (S ids 1..L_S -> X ids L_S+1..N).  Padded ids (0) are left
        # untouched.
        # ``.get(..., default)`` does NOT apply the default when the key is
        # present but explicitly ``null`` (the common template value), so coerce
        # a ``None`` offset to the S-count fallback here.
        _x_offset = config["training"].get("x_var_id_offset", None)
        self.x_var_id_offset = int(
            _x_offset if _x_offset is not None else self.S_seq_len
        )


        # Loss function
        if config["training"]["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise ValueError(
                f"Unsupported loss_fn: {config['training']['loss_fn']}.  "
                f"SelfSelectorForecaster only supports 'mse'."
            )

        # Loss weights / regularisation strengths
        self.lambda_recon = float(config["training"].get("lambda_recon", 1.0))
        self.lambda_struct_recon = float(
            config["training"].get("lambda_struct_recon", 0.0)
        )
        if not (0.0 <= self.lambda_struct_recon <= 1.0):
            raise ValueError(
                f"lambda_struct_recon must be in [0, 1], got {self.lambda_struct_recon}"
            )
        self.lambda_score_sparse = config["training"].get("lambda_score_sparse", 0.0)

        # HSIC
        self.lambda_hsic = config["training"].get("lambda_hsic", 0.0)
        self.hsic_sigma = config["training"].get("hsic_sigma", 1.0)
        self.hsic_adaptive_bandwidth = config["training"].get("hsic_adaptive_bandwidth", False)
        self.hsic_mode = config["training"].get("hsic_mode", "biased")
        self.nhsic_epsilon = config["training"].get("nhsic_epsilon", 0.01)
        self.hsic_kernel_source = config["training"].get("hsic_kernel_source", "rbf")

        # Group-L1
        self.lambda_group_l1 = config["training"].get("lambda_group_l1", 0.0)

        # L0
        self.lambda_l0 = float(config["training"].get("lambda_l0", 0.0))

        # L0 <-> HSIC interference diagnostic
        self.log_l0_hsic_interference = bool(
            config["training"].get("log_l0_hsic_interference", False)
        )
        self.interference_log_every_n_epochs = int(
            config["training"].get("interference_log_every_n_epochs", 1)
        )
        self._attention_type = config["model"]["kwargs"].get("attention_type", "")
        self._interference_blocks: Optional[Dict[str, list]] = None
        self._last_hsic_reg: Optional[torch.Tensor] = None
        self._last_l0_reg: Optional[torch.Tensor] = None

        # NOTEARS acyclicity over the FULL (N, N) score
        self.kappa = float(config["training"].get("kappa", 0.0))
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")

        # Gradient routing
        self.use_gradient_routing = config["training"].get("use_gradient_routing", False)
        if self.use_gradient_routing:
            self.automatic_optimization = False
            # Prefer the model's identity-based grouping: name-substring routing
            # (classify_parameters) misclassifies the self-selector's singular
            # ``orth_embed`` / ``query_embed`` and the ambiguously-named
            # ``embed_modules_list.<i>`` structural embedding.  parameter_groups()
            # partitions by MODULE REFERENCE and is authoritative here.
            if hasattr(self.model, "parameter_groups"):
                structural_params, reconstruction_params = self.model.parameter_groups()
            else:
                structural_params, reconstruction_params = classify_parameters(
                    self.model, verbose=True
                )
            self._structural_params = structural_params
            self._reconstruction_params = reconstruction_params

        # ANM stage freezing
        self.freeze_structural_params = bool(
            config["training"].get("freeze_structural_params", False)
        )
        self.freeze_reconstruction_params = bool(
            config["training"].get("freeze_reconstruction_params", False)
        )

        # Oracle mode + hard masks (full (N, N) GT adjacency)
        self.use_oracle = config["training"].get("use_oracle_attention", False)
        self.use_hard_masks = config["training"].get("use_hard_masks", False)
        self._hard_masks_loaded = False
        if self.use_oracle and not self.use_hard_masks:
            raise ValueError(
                "training.use_oracle_attention=True requires "
                "training.use_hard_masks=True."
            )
        if self.use_hard_masks and data_dir is not None:
            self._load_combined_oracle_mask(config, data_dir)
        elif self.use_hard_masks and data_dir is None:
            print(
                "Warning: training.use_hard_masks=True but data_dir was not "
                "provided to SelfSelectorForecaster.  Hard masks not loaded."
            )

        self.save_hyperparameters(config)

        # Metrics
        self.mae_x = tm.MeanAbsoluteError()
        self.rmse_x = tm.MeanSquaredError(squared=False)
        self.r2_x = tm.R2Score()

    # ------------------------------------------------------------------
    # Hard mask loading (full N x N)
    # ------------------------------------------------------------------

    def _load_combined_oracle_mask(self, config: dict, data_dir: str):
        """
        Build a full ``(N, N)`` oracle adjacency from the GT DAG mask CSVs.

        The standard SCM masks encode ``dec_cross`` (L_X, L_S) = S->X and
        ``dec_self`` (L_X, L_X) = X->X.  Assuming S are the true sources, the
        full adjacency (entry ``[i, j]`` = edge ``j -> i``) is::

            full[S:, :S] = dec_cross     (S -> X)
            full[S:, S:] = dec_self      (X -> X)
            full[:S, : ] = 0             (S have no parents)
        """
        mask_files = config["training"].get("hard_mask_files", None)
        if mask_files is None:
            print("Warning: use_hard_masks=True but no hard_mask_files specified.")
            return
        dataset_name = config["data"]["dataset"]
        dataset_dir = join(data_dir, dataset_name)
        masks = load_dag_masks(dataset_dir, mask_files, device="cpu")
        if masks is None:
            print("Warning: No DAG mask files found.  Oracle mask not loaded.")
            return
        cross_mask = masks.get("dec_cross", None)   # (L_X, L_S)
        self_mask = masks.get("dec_self", None)      # (L_X, L_X)
        if cross_mask is None or self_mask is None:
            print("Warning: Expected 'dec_cross' and 'dec_self' in hard_mask_files.")
            return
        S, N = self.S_seq_len, self.N
        full = torch.zeros(N, N)
        full[S:, :S] = cross_mask
        full[S:, S:] = self_mask
        self.register_buffer("oracle_full_mask", full)
        self._hard_masks_loaded = True
        print(f"✓ Oracle full mask built: shape {full.shape}")

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _assemble_nodes(self, S: torch.Tensor, X: torch.Tensor):
        """
        Concatenate S and X into a homogeneous ``(B, N, F)`` tensor and offset
        the X variable-id column so S and X share one id namespace for the
        single shared embedding.  Returns ``(all_actual, all_blanked)``.
        """
        X_off = X.clone()
        if self.var_idx is not None and self.x_var_id_offset != 0:
            ids = X_off[:, :, self.var_idx]
            # Leave padded ids (0) untouched; shift real ids by the offset.
            X_off[:, :, self.var_idx] = torch.where(
                ids > 0, ids + self.x_var_id_offset, ids
            )
        all_actual = torch.cat([S, X_off], dim=1)          # (B, N, F)
        all_blanked = all_actual.clone()
        all_blanked[:, :, self.val_idx] = 0.0
        return all_actual, all_blanked

    def forward(self, data_source: torch.Tensor, data_intermediate: torch.Tensor):
        all_actual, all_blanked = self._assemble_nodes(data_source, data_intermediate)
        apply_hard_masks = self.use_hard_masks and self._hard_masks_loaded
        oracle = self.use_oracle and apply_hard_masks
        oracle_mask = (
            getattr(self, "oracle_full_mask", None) if apply_hard_masks else None
        )
        return self.model.forward_with_actual(
            all_blanked=all_blanked,
            all_actual=all_actual,
            oracle=oracle,
            oracle_mask=oracle_mask,
        )

    # ------------------------------------------------------------------
    # Common step
    # ------------------------------------------------------------------

    def _step(self, batch, stage: str = "train"):
        S = batch[0]
        X = batch[1]

        # Ground-truth values for ALL N nodes (S first, then X).
        node_vals = torch.cat(
            [S[:, :, self.val_idx], X[:, :, self.val_idx]], dim=1
        )                                                   # (B, N)

        pred, attention_weights, aux = self.forward(S, X)
        entropy = aux.get("entropy") if isinstance(aux, dict) else aux
        l0_penalty = aux.get("l0_penalty") if isinstance(aux, dict) else None

        # Reconstruction loss over all N nodes
        target = torch.nan_to_num(node_vals)
        mse_per_elem = self.loss_fn(pred.squeeze(), target.squeeze())
        loss_x = mse_per_elem.mean()

        # Score sparsity (L1 on the directed posterior)
        inner_att = self.model.attention.inner_attention
        score_tensor = getattr(inner_att, "score_tensor_for_sparsity", None)
        if score_tensor is not None:
            score_sparse_value = score_tensor.abs().mean()
        elif entropy is not None:
            score_sparse_value = entropy.mean()
        else:
            score_sparse_value = torch.tensor(0.0, device=X.device)
        score_sparsity_reg = self.lambda_score_sparse * score_sparse_value

        # HSIC over the full N-node source
        residuals = target.squeeze() - pred.squeeze()       # (B, N)
        combined_source = target.squeeze()                  # (B, N) all node values
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

        # Group-L1
        group_l1_loss, effective_dims = self._compute_group_l1()
        group_l1_reg = self.lambda_group_l1 * group_l1_loss

        # NOTEARS over the FULL (N, N) score
        if self.kappa > 0.0 and score_tensor is not None and score_tensor.dim() == 2:
            acyclic_reg = self.kappa * self._notears_acyclicity(score_tensor)
        else:
            acyclic_reg = torch.tensor(0.0, device=X.device)

        # L0
        if l0_penalty is None:
            l0_penalty = torch.tensor(0.0, device=X.device)
        l0_reg = self.lambda_l0 * l0_penalty if self.lambda_l0 > 0.0 else torch.tensor(0.0, device=X.device)

        # Total loss
        total_loss = (
            self.lambda_recon * loss_x
            + score_sparsity_reg
            + hsic_reg
            + group_l1_reg
            + acyclic_reg
            + l0_reg
        )

        # Gradient-routing loss split (convex HSIC/recon mix on structural path)
        alpha = self.lambda_struct_recon
        struct_recon_reg = alpha * loss_x
        self._last_loss_components = {
            "loss_recon": loss_x,
            "loss_structural": (
                (1.0 - alpha) * hsic_reg
                + struct_recon_reg
                + score_sparsity_reg + group_l1_reg + acyclic_reg + l0_reg
            ),
        }
        self._last_hsic_reg = hsic_reg
        self._last_l0_reg = l0_reg

        # Logging
        self.log(f"{stage}_loss_x", loss_x, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"))
        self.log(f"{stage}_score_sparse", score_sparse_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic", hsic_value, on_step=False, on_epoch=True)
        self.log(f"{stage}_hsic_reg", hsic_reg, on_step=False, on_epoch=True)
        self.log(f"{stage}_struct_recon_reg", struct_recon_reg, on_step=False, on_epoch=True)
        self.log(f"{stage}_group_l1", group_l1_loss, on_step=False, on_epoch=True)
        for name, metric in [("mae", self.mae_x), ("rmse", self.rmse_x), ("r2", self.r2_x)]:
            metric_eval = metric(pred.reshape(-1), target.reshape(-1))
            self.log(f"{stage}_x_{name}", metric_eval, on_step=False, on_epoch=True,
                     prog_bar=(stage == "val" and name == "mae"))
        if effective_dims is not None:
            self.log(f"{stage}_effective_dims", effective_dims, on_step=False, on_epoch=True)
        self.log(f"{stage}_notears", acyclic_reg, on_step=False, on_epoch=True)
        self.log(f"{stage}_l0_penalty", l0_penalty, on_step=False, on_epoch=True)
        self.log(f"{stage}_l0_reg", l0_reg, on_step=False, on_epoch=True)
        if stage == "val":
            self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss, pred, X

    # ------------------------------------------------------------------
    # L0 <-> HSIC interference diagnostic
    # ------------------------------------------------------------------

    _INTERFERENCE_ATTENTION_TYPES = ("GatedSelfAttention",)

    def _interference_enabled(self) -> bool:
        return (
            self.log_l0_hsic_interference
            and self._attention_type in self._INTERFERENCE_ATTENTION_TYPES
            and float(self.lambda_l0) > 0.0
            and float(self.lambda_hsic) > 0.0
        )

    def _maybe_log_interference(self, batch_idx: int):
        if not self._interference_enabled():
            return
        if batch_idx != 0:
            return
        every = max(1, int(self.interference_log_every_n_epochs))
        if (self.current_epoch % every) != 0:
            return
        if self._last_hsic_reg is None or self._last_l0_reg is None:
            return
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
            logger.warning("L0<->HSIC interference probe skipped (autograd error): %s", exc)
            return
        for block_name, cos in cos_by_block.items():
            if math.isnan(cos):
                continue
            self.log(f"train_interf_cos_{block_name}", float(cos),
                     on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _notears_acyclicity(A: torch.Tensor) -> torch.Tensor:
        """NOTEARS penalty h(A) = tr(exp(A ⊙ A)) - d; zero iff A is a DAG."""
        d = A.shape[-1]
        return torch.trace(torch.matrix_exp(A * A)) - d

    def _compute_group_l1(self):
        if self.lambda_group_l1 == 0.0:
            return torch.tensor(0.0, device=next(self.parameters()).device), None
        total_l21 = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        effective_dims = 0
        for name, param in self.model.named_parameters():
            if "nn_embedding" in name and "weight" in name:
                col_norms = param.norm(dim=0)
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
            opt_recon, opt_struct = self.optimizers()
            total_loss, _, _ = self._step(batch=batch, stage="train")
            loss_recon = self._last_loss_components["loss_recon"]
            loss_structural = self._last_loss_components["loss_structural"]
            self._maybe_log_interference(batch_idx)
            opt_recon.zero_grad()
            opt_struct.zero_grad()
            self.manual_backward(loss_recon, retain_graph=True)
            _saved_recon_grads = {}
            for p in self._reconstruction_params:
                if p.grad is not None:
                    _saved_recon_grads[id(p)] = p.grad.clone()
            self.zero_grad()
            self.manual_backward(loss_structural)
            for p in self._reconstruction_params:
                if id(p) in _saved_recon_grads:
                    p.grad = _saved_recon_grads[id(p)]
            opt_recon.step()
            opt_struct.step()
            return total_loss
        else:
            total_loss, _, _ = self._step(batch, stage="train")
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
            return [opt_recon, opt_struct]
        return _make_optimizer(self.model.parameters())

    def on_fit_start(self):
        if self.freeze_structural_params and self.use_gradient_routing:
            for p in self._structural_params:
                p.requires_grad_(False)
            print("  [ANM stage] Structural parameters frozen (requires_grad=False).")
        if self.freeze_reconstruction_params and self.use_gradient_routing:
            for p in self._reconstruction_params:
                p.requires_grad_(False)
            print("  [ANM stage] Reconstruction parameters frozen (requires_grad=False).")
        self._interference_blocks = None

    # ------------------------------------------------------------------
    # Convenience: split attention for post-hoc evaluation
    # ------------------------------------------------------------------

    def get_split_attention(self, data_source: torch.Tensor, data_intermediate: torch.Tensor):
        """Run forward and return the four split blocks (dict) + source scores."""
        with torch.no_grad():
            _, attention_weights, _ = self.forward(data_source, data_intermediate)
        blocks = self.model.split_attention(attention_weights)
        blocks["source_scores"] = self.model.source_scores(attention_weights)
        return blocks
