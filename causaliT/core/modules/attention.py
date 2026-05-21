from os.path import dirname, abspath
import sys
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causaliT.core.modules.extra_layers import UniformAttentionMask, BatchConsistentKeyDropout
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear
from causaliT.utils.entropy_utils import register_attention_entropy, calculate_attention_entropy
from typing import List, Optional


# ---------------------------------------------------------------------------
# Backward-compat helper for state_dict loading
# ---------------------------------------------------------------------------
# In iter_10+, the cross-attention modules (`CausalCrossAttention`,
# `SigmoidCrossAttention`) and the self-attention `ToeplitzAttention` no
# longer carry the parameters ``log_gain`` / ``log_tau_act`` / ``log_tau``
# (and the scalar ``max_gain``). When loading checkpoints saved by earlier
# iterations these keys would otherwise be flagged as ``unexpected_keys``
# and (with ``strict=True`` loading) abort the run. This pre-hook silently
# drops them so older checkpoints can still be loaded for evaluation.
_LEGACY_ATTENTION_KEYS = ("log_gain", "log_tau_act", "log_tau", "max_gain")


def _drop_legacy_attention_keys(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    for key in _LEGACY_ATTENTION_KEYS:
        full = prefix + key
        state_dict.pop(full, None)


class CausalCrossAttention(nn.Module):
    """
    Causal Cross-Attention with ReLU(Tanh) activation.

    Uses ``ReLU(Tanh(scores / tau))`` as attention activation with a *constant*
    scalar temperature ``tau`` (non-learnable, not annealed). No external
    learnable DAG gate.

    Activation parameterization (iter_10+):
        ``att = ReLU(Tanh(scores / tau))`` where ``tau`` is a *constant* scalar
        (default 3.0). Pass ``init_tau`` to override.

    Args:
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        init_tau: Constant temperature for the activation (default: 3.0)
    """
    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        init_tau: float = 3.0,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(CausalCrossAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        # Constant activation temperature (non-learnable, not annealed).
        self.tau = float(init_tau)

        # Batch-consistent key dropout (None = disabled, use standard nn.Dropout).
        # blanking_value=0.0: applied post-activation (ReLU-Tanh output).
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=0.0,
            )
        else:
            self.batch_key_dropout = None

        # Drop legacy attention parameters when loading older checkpoints.
        self._register_load_state_dict_pre_hook(_drop_legacy_attention_keys)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        if oracle:
            is_multihead = query.dim() == 4
            if is_multihead:
                B, _, H, _ = query.shape
                _, S_len, _, _ = key.shape
                L = query.shape[1]
            else:
                B, L, _ = query.shape
                _, S_len, _ = key.shape
                H = 1
            V_out, A_out, ent = _oracle_attention_forward(
                value=value,
                hard_mask=hard_mask,
                is_multihead=is_multihead,
                batch_size=B,
                n_heads=H,
                query_seq_len=L,
                key_seq_len=S_len,
                entropy_enabled=self.entropy_enabled,
            )
            self._score_tensor_for_sparsity = A_out.detach().mean(dim=0)
            return V_out, A_out, ent

        is_multihead = query.dim() == 4
        # Shared-DAG / multi-head-V: Q,K are 3-D, V is 4-D.
        is_shared_mh_v = (query.dim() == 3 and value.dim() == 4)

        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = value.shape[2] if is_shared_mh_v else 1

        scale = 1.0 / sqrt(E)

        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)

        scores = scale * scores

        # Apply causal mask (additive, before activation)
        if pos is not None and causal_mask:
            H_score = H if is_multihead else 1
            M_causal = build_causal_mask(pos, n_heads=H_score)
            scores = scores + M_causal

        # ReLU(Tanh) activation with constant temperature
        att = F.relu(F.tanh(scores / self.tau))
        att = torch.nan_to_num(att, nan=0.0)

        # Apply hard mask if provided
        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            att = att * hard_mask_expanded

        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None

        A = self.dropout(att)
        if self.batch_key_dropout is not None:
            A = self.batch_key_dropout(A)
        self._score_tensor_for_sparsity = A.mean(dim=0)

        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        elif is_shared_mh_v:
            V = torch.einsum("bls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)

        return V.contiguous(), A, entropy

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, '_score_tensor_for_sparsity', None)


class SigmoidCrossAttention(nn.Module):
    """
    Causal Cross-Attention with sigmoid activation.

    Identical to ``CausalCrossAttention`` but uses ``sigmoid(scores / tau)``
    instead of ``ReLU(Tanh(scores / tau))``. Constant (non-learnable) temperature.

    Args:
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        init_tau: Constant temperature for the activation (default: 3.0)
    """
    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        init_tau: float = 3.0,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(SigmoidCrossAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        self.tau = float(init_tau)

        # Additive gate bias for dense→sparse annealing (H5 experiment).
        # Plain float (not a Parameter) so it is not part of the checkpoint
        # state and always resets to 0.0 on model creation.  Set externally
        # by the gate-bias annealer in ``on_train_epoch_start``.
        # Mirrors the ``gate_bias`` nn.Parameter on ToeplitzAttention; the
        # annealer handles both types via an isinstance check.
        self.gate_bias: float = 0.0

        # Batch-consistent key dropout (None = disabled).
        # blanking_value=0.0: applied post-activation (Sigmoid output).
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=0.0,
            )
        else:
            self.batch_key_dropout = None

        self._register_load_state_dict_pre_hook(_drop_legacy_attention_keys)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        if oracle:
            is_multihead = query.dim() == 4
            if is_multihead:
                B, _, H, _ = query.shape
                _, S_len, _, _ = key.shape
                L = query.shape[1]
            else:
                B, L, _ = query.shape
                _, S_len, _ = key.shape
                H = 1
            V_out, A_out, ent = _oracle_attention_forward(
                value=value,
                hard_mask=hard_mask,
                is_multihead=is_multihead,
                batch_size=B,
                n_heads=H,
                query_seq_len=L,
                key_seq_len=S_len,
                entropy_enabled=self.entropy_enabled,
            )
            self._score_tensor_for_sparsity = A_out.detach().mean(dim=0)
            return V_out, A_out, ent

        is_multihead = query.dim() == 4
        is_shared_mh_v = (query.dim() == 3 and value.dim() == 4)

        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = value.shape[2] if is_shared_mh_v else 1

        scale = 1.0 / sqrt(E)

        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)

        scores = scale * scores

        if pos is not None and causal_mask:
            H_score = H if is_multihead else 1
            M_causal = build_causal_mask(pos, n_heads=H_score)
            scores = scores + M_causal

        # Sigmoid activation with constant temperature and additive gate_bias
        # (plain float, default 0.0; set by H5 annealer to impose sparsity).
        att = torch.sigmoid(scores / self.tau + self.gate_bias)
        att = torch.nan_to_num(att, nan=0.0)

        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            att = att * hard_mask_expanded

        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None

        A = self.dropout(att)
        if self.batch_key_dropout is not None:
            A = self.batch_key_dropout(A)
            
        self._score_tensor_for_sparsity = A.mean(dim=0)

        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        elif is_shared_mh_v:
            V = torch.einsum("bls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)

        return V.contiguous(), A, entropy

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, '_score_tensor_for_sparsity', None)


class ToeplitzAttention(nn.Module):
    """
    Clean Toeplitz Attention for DAG learning.

    Decomposes attention scores into symmetric and antisymmetric parts using
    *orthogonal* projectors:

        S = (QK^T + KQ^T) / sqrt(2)   # Symmetric: edge existence
        A = (QK^T - KQ^T) / sqrt(2)   # Antisymmetric: direction split

    Computes attention as:
        att[i,j] = sigmoid(S[i,j] / tau + b_gate) * sigmoid(A[i,j] / tau)

    Temperature ``tau`` is a *constant* scalar (default 3.0, non-learnable, not annealed).

    Gate bias ``b_gate``: scalar bias on the symmetric (gate) sigmoid's
    pre-activation. Initialized strongly negative (default -15.0) so the gate is
    effectively closed at init. Whether the bias is learnable or fixed is
    controlled by ``gate_bias_trainable``.

    Args:
        attention_dropout: Dropout rate for attention weights
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        init_tau: Constant temperature (default: 3.0)
        init_gate_bias: Initial value of the gate bias (default: -15.0)
        gate_bias_trainable: If True (default), gate_bias is a learnable
            parameter updated during training. If False, it is registered as a
            fixed constant (frozen at init_gate_bias for the whole run).
        tau_min: Legacy kwarg (ignored)
        tau_max: Legacy kwarg (ignored)
    """
    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        init_tau: float = 3.0,
        init_gate_bias: float = -15.0,
        gate_bias_trainable: bool = True,
        tau_min: float = None,
        tau_max: float = None,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(ToeplitzAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        self.tau = float(init_tau)
        self.gate_bias = nn.Parameter(torch.tensor(float(init_gate_bias)))
        self.gate_bias.requires_grad = gate_bias_trainable

        # Batch-consistent key dropout (None = disabled).
        # blanking_value=0.0: applied post-activation (Toeplitz product output).
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=0.0,
            )
        else:
            self.batch_key_dropout = None

        self._register_load_state_dict_pre_hook(_drop_legacy_attention_keys)

    def get_dag_probabilities(self) -> torch.Tensor:
        if hasattr(self, 'last_att'):
            return self.last_att
        return None

    def get_edge_existence_probabilities(self) -> torch.Tensor:
        if hasattr(self, 'last_P_edge'):
            return self.last_P_edge
        return None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        if oracle:
            is_multihead = query.dim() == 4
            if is_multihead:
                B, _, H, _ = query.shape
                _, S_len, _, _ = key.shape
                L = query.shape[1]
            else:
                B, L, _ = query.shape
                _, S_len, _ = key.shape
                H = 1
            V_out, A_out, ent = _oracle_attention_forward(
                value=value,
                hard_mask=hard_mask,
                is_multihead=is_multihead,
                batch_size=B,
                n_heads=H,
                query_seq_len=L,
                key_seq_len=S_len,
                entropy_enabled=self.entropy_enabled,
            )
            mask_avg = A_out.detach().mean(dim=0)
            self.last_att = mask_avg
            self.last_P_edge = mask_avg
            self.last_S = torch.zeros_like(mask_avg)
            self.last_A = torch.zeros_like(mask_avg)
            self.P_edge_for_reg = A_out.mean(dim=0)
            return V_out, A_out, ent

        is_multihead = query.dim() == 4
        is_shared_mh_v = (query.dim() == 3 and value.dim() == 4)

        if is_multihead:
            B, L, H, E = query.shape
            _, S_len, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S_len, _ = key.shape
            H = value.shape[2] if is_shared_mh_v else 1

        scale = 1.0 / sqrt(E)

        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)

        scores = scale * scores

        _inv_sqrt2 = 1.0 / sqrt(2.0)
        S = (scores + scores.transpose(-1, -2)) / 2
        A = (scores - scores.transpose(-1, -2)) / 2

        tau = self.tau

        P_edge = torch.sigmoid(S / tau + self.gate_bias)
        d = torch.sigmoid(A / tau)

        att = P_edge * d

        # Zero out diagonal (no self-loops)
        if is_multihead:
            diag_mask = torch.eye(L, S_len, device=att.device, dtype=torch.bool)
            att = att.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), 0.0)
        else:
            diag_mask = torch.eye(L, S_len, device=att.device, dtype=torch.bool)
            att = att.masked_fill(diag_mask.unsqueeze(0), 0.0)

        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            att = att * hard_mask_expanded

        self.last_att = att.detach().mean(dim=0)
        self.last_P_edge = P_edge.detach().mean(dim=0)
        self.last_S = S.detach().mean(dim=0)
        self.last_A = A.detach().mean(dim=0)
        self.P_edge_for_reg = P_edge.mean(dim=0)

        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None

        A_out = self.dropout(att)
        A_out = torch.nan_to_num(A_out)
        if self.batch_key_dropout is not None:
            A_out = self.batch_key_dropout(A_out)
            
        if is_multihead:
            V_out = torch.einsum("bhls,bshd->blhd", A_out, value)
        elif is_shared_mh_v:
            V_out = torch.einsum("bls,bshd->blhd", A_out, value)
        else:
            V_out = torch.einsum("bls,bsd->bld", A_out, value)

        return V_out.contiguous(), A_out, entropy

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, 'P_edge_for_reg', None)


class ScaledDotAttention(nn.Module):
    """
    Simplified Scaled Dot-Product Attention.

    Hard mask is applied BEFORE softmax to ensure masked positions don't
    influence the softmax normalization (preventing information leakage).
    """
    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(ScaledDotAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        # Batch-consistent key dropout (None = disabled).
        # blanking_value=-inf: applied to pre-softmax scores so dropped keys
        # get zero probability after softmax and remaining weights renormalise.
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=float('-inf'),
            )
        else:
            self.batch_key_dropout = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        if oracle:
            raise NotImplementedError(
                "Oracle attention mode is not implemented for ScaledDotAttention. "
                "Use ToeplitzAttention (self) or CausalCrossAttention (cross) instead."
            )

        is_multihead = query.dim() == 4

        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1

        scale = 1.0 / sqrt(E)

        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)

        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal

        all_masked_rows = None

        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)

            all_masked_rows = (hard_mask_expanded.sum(dim=-1, keepdim=True) == 0)

            hard_mask_additive = torch.where(
                hard_mask_expanded == 0,
                torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype),
                torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            )

            if all_masked_rows.any():
                hard_mask_additive = torch.where(
                    all_masked_rows,
                    torch.tensor(0.0, device=scores.device, dtype=scores.dtype),
                    hard_mask_additive
                )

            scores = scores + hard_mask_additive

        # Batch-consistent key dropout on pre-softmax scores (-inf blanking).
        if self.batch_key_dropout is not None:
            scores = self.batch_key_dropout(scores)

        att = torch.softmax(scale * scores, dim=-1)

        if all_masked_rows is not None and all_masked_rows.any():
            att = att * (~all_masked_rows).float()

        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None

        A = self.dropout(att)

        if is_multihead:
            # Q/K are 4D (B, L, H, E) and V is 4D (B, S, H, d_head):
            # standard multi-head path.
            V = torch.einsum("bhls,bshd->blhd", A, value)
        elif value.dim() == 4:
            # Mixed SVFA case: Q/K have 1 structure head (3D, shape B×L×E / B×S×E)
            # but V has multiple value heads (4D, shape B×S×H×d_head).
            # A is 3D (B×L×S); broadcast the shared attention map across all V heads.
            V = torch.einsum("bls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)

        return V.contiguous(), A, entropy

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return None


class ScaledDotAttentionNAIM(nn.Module):
    """
    Scaled Dot-Product Attention with NAIM (Not All Is Missing) handling.

    This version includes special handling for missing data via mask_miss_k and mask_miss_q,
    using ReLU after softmax to handle the missing query mask.
    Reference: https://arxiv.org/abs/2407.11540

    NOTE: The hard_mask in this version is applied AFTER softmax, which can cause
    information leakage. Use ScaledDotAttention for proper causal masking.
    """
    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(ScaledDotAttentionNAIM, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        # Batch-consistent key dropout (None = disabled).
        # blanking_value=-inf: applied to pre-softmax scores.
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=float('-inf'),
            )
        else:
            self.batch_key_dropout = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        if oracle:
            raise NotImplementedError(
                "Oracle attention mode is not implemented for ScaledDotAttentionNAIM. "
                "Use ToeplitzAttention (self) or CausalCrossAttention (cross) instead."
            )

        is_multihead = query.dim() == 4

        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1

        scale = 1.0 / sqrt(E)

        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)

        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            scores = scores + M_causal

        if is_multihead:
            key_size = scores.size(-1)
            query_size = scores.size(-2)

            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, query_size).transpose(-1, -2)

            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, key_size)
        else:
            key_size = scores.size(-1)
            query_size = scores.size(-2)

            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.expand(-1, -1, query_size).transpose(-1, -2)

            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.expand(-1, -1, key_size)

        if mask_miss_k is not None:
            M_k = torch.zeros_like(scores).masked_fill_(mask_miss_k_expanded, -torch.inf)
        else:
            M_k = torch.zeros_like(scores)

        if mask_miss_q is not None:
            M_q = torch.zeros_like(scores).masked_fill_(mask_miss_q_expanded, -torch.inf)
        else:
            M_q = torch.zeros_like(scores)

        # Batch-consistent key dropout on pre-softmax scores (-inf blanking).
        if self.batch_key_dropout is not None:
            scores = self.batch_key_dropout(scores)

        att = torch.relu(torch.softmax(scale * (scores + M_k), dim=-1) + M_q)

        # Apply hard mask if provided
        # WARNING: Applied AFTER softmax - may cause information leakage
        if hard_mask is not None:
            if is_multihead:
                if hard_mask.dim() == 2:
                    hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
                else:
                    hard_mask = hard_mask.unsqueeze(0)
            else:
                hard_mask = hard_mask.unsqueeze(0)

            att = att * hard_mask

        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None

        A = torch.nan_to_num(self.dropout(att))

        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)

        return V.contiguous(), A, entropy


def _oracle_attention_forward(
    value: torch.Tensor,
    hard_mask: torch.Tensor,
    is_multihead: bool,
    batch_size: int,
    n_heads: int,
    query_seq_len: int,
    key_seq_len: int,
    entropy_enabled: bool = True,
):
    """
    Oracle attention forward: bypass QK^T entirely and use the hard mask directly
    as the attention score matrix for the values.
    """
    if hard_mask is None:
        raise ValueError(
            "Oracle attention mode requires hard_mask to be provided. "
            "Set training.use_hard_masks=true and provide hard_mask_files."
        )

    att = expand_hard_mask(hard_mask, is_multihead, batch_size).to(value.dtype)

    if is_multihead:
        if att.shape[0] == 1:
            att = att.expand(batch_size, *att.shape[1:])
        if att.shape[1] == 1:
            att = att.expand(att.shape[0], n_heads, *att.shape[2:])
        V_out = torch.einsum("bhls,bshd->blhd", att, value)
    else:
        if att.shape[0] == 1:
            att = att.expand(batch_size, *att.shape[1:])
        V_out = torch.einsum("bls,bsd->bld", att, value)

    if entropy_enabled:
        entropy = calculate_attention_entropy(att)
    else:
        entropy = None

    return V_out.contiguous(), att, entropy


def expand_hard_mask(hard_mask: torch.Tensor, is_multihead: bool, batch_size: int) -> torch.Tensor:
    """
    Expand hard_mask to match attention scores shape.

    Args:
        hard_mask: Mask tensor. Can be:
            - (L, S): static mask for all samples and heads
            - (H, L, S): per-head static mask (multihead only)
            - (B, L, S): per-sample in-context mask
            - (B, H, L, S): per-sample per-head mask (multihead only)
        is_multihead: Whether attention is multi-head
        batch_size: Batch size to detect if first dim is batch or heads

    Returns:
        Expanded mask matching scores shape: (B, L, S) or (B, H, L, S)
    """
    if is_multihead:
        if hard_mask.dim() == 2:
            hard_mask = hard_mask.unsqueeze(0).unsqueeze(0)
        elif hard_mask.dim() == 3:
            if hard_mask.shape[0] == batch_size:
                hard_mask = hard_mask.unsqueeze(1)
            else:
                hard_mask = hard_mask.unsqueeze(0)
    else:
        if hard_mask.dim() == 2:
            hard_mask = hard_mask.unsqueeze(0)
    return hard_mask


def build_causal_mask(p: torch.Tensor, n_heads: int = 1) -> torch.Tensor:
    """
    Args:
        p: (B, L, 1) tensor with the position of every token in the sequence.
        n_heads: number of attention heads

    Returns:
        For single head (n_heads=1): (B, L, L) mask M
        For multi-head (n_heads>1): (B, H, L, L) mask M
        with M[b, (h,) i, j] = -inf if p[b, j] > p[b, i], 0 otherwise.
    """
    p_flat = p.squeeze(-1)

    p_i = p_flat.unsqueeze(-1)
    p_j = p_flat.unsqueeze(-2)

    M = torch.zeros_like(p_i.expand(-1, -1, p_flat.size(-1)))
    M.masked_fill_(p_j > p_i, float("-inf"))

    if n_heads > 1:
        M = M.unsqueeze(1).expand(-1, n_heads, -1, -1)

    return M


def calculate_attention_entropy(att_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate entropy of attention weights.
    """
    att_clamped = torch.clamp(att_weights, min=eps)
    log_att = torch.log(att_clamped)
    entropy = -torch.sum(att_weights * log_att, dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer.

    Supports the following attention mechanisms:
        - ScaledDotAttention
        - ScaledDotAttentionNAIM
        - CausalCrossAttention  (ReLU(Tanh) activation, constant tau)
        - SigmoidCrossAttention (Sigmoid activation, constant tau)
        - ToeplitzAttention     (Toeplitz decomposition, constant tau + learnable gate bias)

    Shared Structure Mode:
        When ``shared_qk_inner`` is provided, the layer reuses the Q/K projections
        and inner attention from another layer instead of creating its own.

    Args:
        attention: Attention class
        d_model_queries: Dimension of query inputs
        d_model_keys: Dimension of key inputs
        d_model_values: Dimension of value inputs
        d_queries_keys: Dimension per head for Q/K projections
        n_heads: Number of attention heads
        mask_layer: Legacy mask layer (kept for compatibility)
        attention_dropout: Dropout rate for attention weights
        dropout_qkv: Dropout rate for Q/K/V projections
        register_entropy: Whether to register entropy for logging
        layer_name: Name for logging purposes
        query_seq_len: Query sequence length (informational)
        key_seq_len: Key sequence length (informational)
        key_projection_type: "linear" or "orthogonal"
        orthogonal_scale: Whether to include learnable scale in orthogonal projection
        orthogonal_init_scale: Initial scale value for orthogonal projection
        init_tau: Constant temperature for CausalCrossAttention / SigmoidCrossAttention
                  / ToeplitzAttention (non-learnable, not annealed). Default 3.0.
        init_gate_bias: Initial value of the gate bias in ToeplitzAttention (default: -15.0).
            Ignored for other attention types.
        gate_bias_trainable: If True (default), the gate bias is updated during training.
            If False, it is frozen at init_gate_bias. Only applies to ToeplitzAttention.
        shared_qk_inner: Optional dict with shared Q/K/inner_attention components.
        shared_dag_across_heads: When True (default), a single score (B,L,S) is
            shared across n_heads value channels. When False, legacy per-head scores.
        dual_value: When True, creates a second value path for SVFA dual-residual.
    """
    def __init__(
        self,
        attention: nn.Module,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        d_queries_keys: int,
        n_heads: int,
        mask_layer: nn.Module,
        attention_dropout: float,
        dropout_qkv: float,
        register_entropy: bool = False,
        layer_name: str = None,
        query_seq_len: int = None,
        key_seq_len: int = None,
        key_projection_type: str = "linear",
        orthogonal_scale: bool = True,
        orthogonal_init_scale: float = 1.0,
        init_tau: float = 3.0,
        init_gate_bias: float = -15.0,
        gate_bias_trainable: bool = True,
        shared_qk_inner: dict = None,
        shared_dag_across_heads: bool = True,
        dual_value: bool = False,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(AttentionLayer, self).__init__()

        self._uses_shared_structure = shared_qk_inner is not None
        self.shared_dag_across_heads = bool(shared_dag_across_heads)
        n_heads_struct = 1 if self.shared_dag_across_heads else n_heads
        self._n_heads_struct = n_heads_struct

        if shared_qk_inner is not None:
            # ===== SHARED MODE =====
            self.query_projection = shared_qk_inner["query_projection"]
            self.key_projection = shared_qk_inner["key_projection"]
            self.inner_attention = shared_qk_inner["inner_attention"]
            self.key_projection_type = getattr(
                shared_qk_inner.get("_source_layer", None),
                "key_projection_type", "linear"
            )
        else:
            # ===== STANDARD MODE =====

            # Create inner attention module
            ATTENTION_WITH_TAU = (CausalCrossAttention, SigmoidCrossAttention, ToeplitzAttention)
            if attention is ToeplitzAttention:
                # ToeplitzAttention also accepts gate bias settings
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    init_gate_bias=init_gate_bias,
                    gate_bias_trainable=gate_bias_trainable,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                )
            elif attention in ATTENTION_WITH_TAU:
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                )
            else:
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                )

            # Q and K projections
            self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads_struct)

            self.key_projection_type = key_projection_type
            if key_projection_type == "linear":
                self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads_struct)
            elif key_projection_type == "orthogonal":
                out_features = d_queries_keys * n_heads_struct
                in_features = d_model_keys
                if out_features < in_features:
                    raise ValueError(
                        f"Orthogonal key projection requires d_queries_keys * n_heads_struct >= d_model_keys, "
                        f"got {out_features} < {in_features}."
                    )
                self.key_projection = OrthogonalLinear(
                    in_features=in_features,
                    out_features=out_features,
                    use_scale=orthogonal_scale,
                    init_scale=orthogonal_init_scale
                )
            else:
                raise ValueError(
                    f"Invalid key_projection_type='{key_projection_type}'. "
                    f"Must be one of: 'linear', 'orthogonal'"
                )

        # V projection, output projection, and dropout are always per-layer
        self.value_projection = nn.Linear(d_model_values, d_model_values * n_heads)
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_values)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

        # Dual-value (SVFA dual-residual)
        self.dual_value = bool(dual_value)
        if self.dual_value:
            self.value_projection_struct = nn.Linear(d_model_keys, d_model_keys * n_heads)
            if n_heads > 1:
                self.out_projection_struct = nn.Linear(d_model_keys * n_heads, d_model_keys)
            else:
                self.out_projection_struct = None
            self._d_model_keys = d_model_keys
        else:
            self.value_projection_struct = None
            self.out_projection_struct = None

    def get_shared_qk_inner(self) -> dict:
        if self._uses_shared_structure:
            raise ValueError(
                "Cannot extract shared components from a layer that itself uses shared structure. "
                "Extract from the original (owning) layer instead."
            )
        return {
            "query_projection": self.query_projection,
            "key_projection": self.key_projection,
            "inner_attention": self.inner_attention,
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        hard_mask: torch.Tensor = None,
        oracle: bool = False,
        key_value: torch.Tensor = None,
    ):
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        H_struct = self._n_heads_struct

        if H_struct > 1:
            q = self.dropout_qkv(self.query_projection(query)).view(B, L, H_struct, -1)
            k = self.dropout_qkv(self.key_projection(key)).view(B, S, H_struct, -1)
        else:
            q = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            k = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)

        if H > 1:
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)

        out, attn, ent = self.inner_attention(
            query=q,
            key=k,
            value=v,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos=pos,
            causal_mask=causal_mask,
            hard_mask=hard_mask,
            oracle=oracle,
        )

        if H > 1:
            out = out.contiguous().view(B, L, -1)
            out = self.out_projection(out)
        else:
            out = out.view(B, L, -1)

        if self.dual_value:
            kv_source = key_value if key_value is not None else key

            if H > 1:
                v_struct = self.dropout_qkv(
                    self.value_projection_struct(kv_source)
                ).view(B, S, H, -1)
            else:
                v_struct = self.dropout_qkv(
                    self.value_projection_struct(kv_source)
                ).view(B, S, -1)

            # Reuse the attention weights already computed in the first pass.
            # Calling inner_attention a second time would:
            #   (a) redundantly recompute Q/K and all activation logic,
            #   (b) draw a fresh BatchConsistentKeyDropout mask, overwriting
            #       _last_key_mask with a different sample and causing the
            #       breakpoint to fire twice per step.
            # The two value paths intentionally share the same masked attention
            # matrix — that is the SVFA dual-residual design contract.
            if H > 1:
                if attn.dim() == 4:
                    # Both struct and value heads are multi-head (normal case)
                    out_struct = torch.einsum("bhls,bshd->blhd", attn, v_struct)
                else:
                    # n_heads_struct=1 (attn is 3-D), n_heads_value>1 (v_struct is 4-D):
                    # broadcast the single structure attention map across all value heads.
                    out_struct = torch.einsum("bls,bshd->blhd", attn, v_struct)
                out_struct = out_struct.contiguous().view(B, L, -1)
                if self.out_projection_struct is not None:
                    out_struct = self.out_projection_struct(out_struct)
            else:
                out_struct = torch.einsum("bls,bsd->bld", attn, v_struct)
                out_struct = out_struct.view(B, L, -1)

            return out, out_struct, attn, ent

        return out, attn, ent


def main():
    """Quick test for both single-head and multi-head attention"""

    bs = 1
    seq_len = 5
    d_model = 12
    d_queries_keys = 8
    x = torch.ones(bs, seq_len, d_model)
    x[0, 0, 0] = torch.nan

    print("Testing single-head attention (n_heads=1):")
    attention_single = AttentionLayer(
        attention=ScaledDotAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)

    out_single, score_single, ent = attention_single.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )

    print(f"Single-head - Output shape: {out_single.shape}, Score shape: {score_single.shape}")

    print("\nTesting multi-head attention (n_heads=4):")
    attention_multi = AttentionLayer(
        attention=ScaledDotAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys,
        n_heads=4,
        mask_layer=UniformAttentionMask(),
        attention_dropout=0,
        dropout_qkv=0)

    out_multi, score_multi, ent = attention_multi.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )

    print(f"Multi-head - Output shape: {out_multi.shape}, Score shape: {score_multi.shape}")

    print("\nTesting with causal mask:")
    pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()

    out_causal, score_causal, ent = attention_multi.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
    )
    print(f"Causal multi-head - Output shape: {out_causal.shape}, Score shape: {score_causal.shape}")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
