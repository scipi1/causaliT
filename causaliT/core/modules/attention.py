from os.path import dirname, abspath
import sys
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causaliT.core.modules.extra_layers import UniformAttentionMask, BatchConsistentKeyDropout
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear
from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.core.modules.commutator_self_attention import CommutatorSelfAttention

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


# ---------------------------------------------------------------------------
# aux_dict helpers
# ---------------------------------------------------------------------------

def _make_aux(entropy, l0_penalty=None):
    """Build the standard attention auxiliary dictionary.

    All attention modules return ``(V, A, aux_dict)`` where ``aux_dict``
    contains at least the keys ``"entropy"`` and ``"l0_penalty"``.  This
    factory ensures a consistent structure across all attention classes.

    Args:
        entropy: Attention entropy tensor (or None if disabled).
        l0_penalty: Scalar L0 regularization penalty (None for non-HC modules).

    Returns:
        dict with keys "entropy" and "l0_penalty".
    """
    return {"entropy": entropy, "l0_penalty": l0_penalty}


class CausalCrossAttention(nn.Module):
    """
    Causal Cross-Attention with ReLU(Tanh) activation.

    Uses ``ReLU(Tanh(scores / tau))`` as attention activation with a *constant*
    scalar temperature ``tau`` (non-learnable, not annealed). No external
    learnable DAG gate.

    Activation parameterization (iter_10+):
        ``att = ReLU(Tanh(scores / tau))`` where ``tau`` is a *constant* scalar
        (default 3.0). Pass ``init_tau`` to override.

    Returns ``(V, A, aux_dict)`` where ``aux_dict = {"entropy": ..., "l0_penalty": None}``.

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
        optuna_protocol: Optional[float] = None,
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

        # --- Optuna capacity-search protocol (constant-score override) --------
        # When set (``0`` or ``1``), the learned QK^T attention weights are
        # REPLACED by this constant on every allowed edge, in BOTH train and
        # eval.  This severs the query/key (structural) stream from the loss so
        # the reconstruction capacity of the value stream can be measured
        # independently of structure learning — mimicking the frozen-structure
        # warmup phase:
        #   optuna_protocol=0 → residual-only floor  (no variable mixing)
        #   optuna_protocol=1 → uniform mixing        (selection-insensitive;
        #                        pair with heavy batch_key_dropout for robustness)
        # ``None`` (default) disables the override entirely (normal behaviour).
        # The override is applied BEFORE the hard mask so the diagonal
        # self-loop constraint still prevents X_i from reading its own value.
        self.optuna_protocol: Optional[float] = (
            float(optuna_protocol) if optuna_protocol is not None else None
        )


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
            return V_out, A_out, _make_aux(ent)

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

        # --- Optuna capacity-search override (constant-score protocol) --------
        # Replace the learned QK^T attention weights with a constant so the
        # structural (query/key) stream is detached from the loss.  Applied in
        # BOTH train and eval, and BEFORE the hard mask so the diagonal
        # self-loop constraint still removes X_i -> X_i value leakage.
        # ``torch.full_like`` produces a leaf-free constant → no gradient flows
        # back into Q/K (structure is effectively frozen for free).
        if self.optuna_protocol is not None:
            att = torch.full_like(att, self.optuna_protocol)

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

        return V.contiguous(), A, _make_aux(entropy)

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, '_score_tensor_for_sparsity', None)


class SigmoidCrossAttention(nn.Module):
    """
    Causal Cross-Attention with sigmoid activation.

    Identical to ``CausalCrossAttention`` but uses ``sigmoid(scores / tau)``
    instead of ``ReLU(Tanh(scores / tau))``. Constant (non-learnable) temperature.

    Returns ``(V, A, aux_dict)`` where ``aux_dict = {"entropy": ..., "l0_penalty": None}``.

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
            return V_out, A_out, _make_aux(ent)

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

        return V.contiguous(), A, _make_aux(entropy)

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, '_score_tensor_for_sparsity', None)


class HardConcreteCrossAttention(nn.Module):
    """
    Causal Cross-Attention with Hard Concrete stochastic gates (L0 regularization).

    Inspired by Louizos, Welling, Kingma "Learning Sparse Neural Networks through
    L0 Regularization" (ICLR 2018, https://arxiv.org/abs/1712.01312).

    The scaled QK dot-product score is used directly as the log-alpha location
    parameter of the Binary Concrete distribution.  The attention weight IS the
    Hard Concrete gate z_ij ∈ [0,1] — no separate multiplicative gate.

    Forward (training, stochastic):
        log_alpha_ij = s_ij = (q_i · k_j) / sqrt(E)
        u_ij ~ Uniform(0, 1)
        s_tilde = sigmoid((log(u) - log(1-u) + log_alpha) / beta)
        s_bar   = s_tilde * (zeta - gamma) + gamma
        z_ij    = clip(s_bar, 0, 1)                    [Hard Concrete gate]

    Forward (eval / MAP, deterministic):
        z_ij = clip(sigmoid(log_alpha / beta) * (zeta - gamma) + gamma, 0, 1)

    L0 penalty (expected number of active edges, differentiable surrogate):
        P(z_ij > 0) = sigmoid(log_alpha_ij - beta * log(-gamma / zeta))
        l0_penalty  = sum_{(i,j) unmasked} P(z_ij > 0)

    The L0 penalty is included in aux_dict["l0_penalty"] and should be added to
    the training loss weighted by a small lambda_l0 coefficient.

    Full edge-existence posterior:
        self.last_p_edge_on  (L, S) or (H, L, S) — P(edge is ON) per (i,j).
        MAP causal graph: ``(module.last_p_edge_on > 0.5).float()``

    Returns ``(V, A, aux_dict)`` where
        ``aux_dict = {"entropy": ..., "l0_penalty": <scalar tensor>}``.

    Hard mask is supported to zero out structurally forbidden edges (e.g. the
    diagonal self-loop constraint in AttentionSelectorLayer). Masked positions
    are excluded from the L0 penalty.

    Oracle mode and causal mask are NOT supported (raise NotImplementedError).

    Args:
        attention_dropout: Dropout rate applied to z after sampling.
        register_entropy: Whether to compute and return attention entropy.
        layer_name: Name for logging purposes.
        init_tau: Binary Concrete temperature β (default 2/3 as in paper).
            Passed as ``init_tau`` for compatibility with AttentionLayer which
            uses that kwarg for all temperature-bearing attention classes.
        gamma: Stretch lower bound, must be < 0 (default −0.1 as in paper).
        zeta: Stretch upper bound, must be > 1 (default 1.1 as in paper).
        batch_key_dropout: Optional batch-consistent key dropout probability.
        batch_key_dropout_p_final: Final dropout probability after annealing.
        batch_key_dropout_annealing_batches: Batches over which to anneal dropout.
    """

    def __init__(
        self,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        init_tau: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super(HardConcreteCrossAttention, self).__init__()

        if gamma >= 0.0:
            raise ValueError(
                f"gamma must be strictly negative (stretch lower bound), got {gamma}"
            )
        if zeta <= 1.0:
            raise ValueError(
                f"zeta must be strictly greater than 1 (stretch upper bound), got {zeta}"
            )

        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        self.entropy_enabled = True

        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")

        # Hard Concrete parameters (all fixed, non-learnable).
        # init_tau mirrors the kwarg name used by other tau-bearing attention
        # classes so that AttentionLayer can instantiate this class uniformly.
        self.beta = float(init_tau)    # Binary Concrete temperature
        self.gamma = float(gamma)      # Stretch lower bound  (< 0)
        self.zeta = float(zeta)        # Stretch upper bound  (> 1)

        # Pre-compute the constant offset for the L0 penalty:
        #   P(z > 0) = sigmoid(log_alpha - beta * log(-gamma / zeta))
        self._l0_offset: float = float(self.beta * log(-self.gamma / self.zeta))

        # Batch-consistent key dropout (None = disabled).
        # blanking_value=0.0: applied post-sampling so dropped keys get z=0.
        if batch_key_dropout is not None:
            self.batch_key_dropout = BatchConsistentKeyDropout(
                p_init=batch_key_dropout,
                p_final=batch_key_dropout_p_final,
                annealing_batches=batch_key_dropout_annealing_batches,
                blanking_value=0.0,
            )
        else:
            self.batch_key_dropout = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        """
        Hard Concrete cross-attention forward pass.

        Args:
            query: (B, L, E) or (B, L, H, E) query tensor.
            key:   (B, S, E) or (B, S, H, E) key tensor.
            value: (B, S, E), (B, S, H, d) or (B, S, d) value tensor.
            mask_miss_k: Unused (kept for interface compatibility).
            mask_miss_q: Unused (kept for interface compatibility).
            pos:       Unused (causal_mask not supported; raises if True).
            causal_mask: Must be False; raises NotImplementedError otherwise.
            hard_mask: Optional binary mask (L, S) or (B, L, S) zeroing out
                forbidden edges. Masked positions are excluded from the L0
                penalty. Supported for diagonal / structural constraints.
            oracle: Must be False; raises NotImplementedError.

        Returns:
            V_out:    (B, L, E) or (B, L, H, d) value-aggregated output.
            A:        (B, L, S) or (B, H, L, S) Hard Concrete gates after dropout.
            aux_dict: {"entropy": <Tensor|None>, "l0_penalty": <scalar Tensor>}
        """
        if oracle:
            raise NotImplementedError(
                "Oracle mode is not supported by HardConcreteCrossAttention. "
                "Use CausalCrossAttention or ToeplitzAttention instead."
            )
        if causal_mask:
            raise NotImplementedError(
                "Causal masking (causal_mask=True) is not supported by "
                "HardConcreteCrossAttention. The Hard Concrete gate already "
                "encodes edge direction via the QK scores."
            )

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

        # ---- Scaled QK scores = log_alpha of the Binary Concrete ----
        if is_multihead:
            log_alpha = scale * torch.einsum("blhe,bshe->bhls", query, key)
        else:
            log_alpha = scale * torch.einsum("ble,bse->bls", query, key)

        # Guard against NaN inputs propagating from missing/masked tokens.
        # NaN log_alpha would corrupt both the sampled gate z and the L0
        # penalty.  Replace with 0.0 (→ P(z>0) = sigmoid(-offset) ≈ 0.09
        # with paper defaults), which is the maximally uncertain / sparse
        # default for positions whose score is undefined.
        log_alpha = torch.nan_to_num(log_alpha, nan=0.0)

        # ---- Hard Concrete sampling ----
        if self.training:
            # Stochastic path: u ~ Uniform(0, 1), clamped for numerical stability
            u = torch.zeros_like(log_alpha).uniform_().clamp_(1e-8, 1.0 - 1e-8)
            # Binary Concrete relaxation
            s_tilde = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + log_alpha) / self.beta
            )
            # Stretch to (gamma, zeta)
            s_bar = s_tilde * (self.zeta - self.gamma) + self.gamma
            # Hard-sigmoid clip → z ∈ {0} ∪ (0,1) ∪ {1}
            z = s_bar.clamp(0.0, 1.0)
        else:
            # Deterministic MAP path: mode of the Hard Concrete distribution
            z = (
                torch.sigmoid(log_alpha / self.beta) * (self.zeta - self.gamma)
                + self.gamma
            ).clamp(0.0, 1.0)

        # ---- L0 penalty: expected number of active (non-zero) gates ----
        # P(z_ij > 0) = sigmoid(log_alpha_ij - beta * log(-gamma / zeta))
        p_edge_on = torch.sigmoid(log_alpha - self._l0_offset)

        # Store batch-averaged posterior for external access (evaluation,
        # graph extraction, MAP thresholding at p_edge_on > 0.5).
        self.last_p_edge_on = p_edge_on.detach().mean(dim=0)

        # ---- Hard mask (structural / self-loop constraint) ----
        if hard_mask is not None:
            hard_mask_expanded = expand_hard_mask(hard_mask, is_multihead, B)
            z = z * hard_mask_expanded
            # Exclude masked positions from the L0 penalty
            p_edge_on_for_penalty = p_edge_on * hard_mask_expanded
        else:
            p_edge_on_for_penalty = p_edge_on

        # Scalar L0 penalty: sum of P(z_ij > 0) over all unmasked (i,j) positions.
        # Summed (not averaged) so the penalty scales with graph size, consistent
        # with the original L0 formulation.
        l0_penalty = p_edge_on_for_penalty.sum()

        # ---- Entropy (for logging / compatibility with existing pipeline) ----
        z_clean = torch.nan_to_num(z, nan=0.0)
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(z_clean)
        else:
            entropy = None

        # ---- Dropout ----
        A = self.dropout(z_clean)
        if self.batch_key_dropout is not None:
            A = self.batch_key_dropout(A)

        # Expose p_edge_on (batch-averaged) as the sparsity tensor.
        # Using the edge-existence probability rather than the gate z ensures
        # the sparsity metric is differentiable and interpretable even at eval
        # time (where z may be clipped to 0 or 1).
        self._score_tensor_for_sparsity = p_edge_on.detach().mean(dim=0)

        # ---- Value aggregation ----
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        elif is_shared_mh_v:
            V = torch.einsum("bls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)

        return V.contiguous(), A, _make_aux(entropy, l0_penalty)

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

    Returns ``(V, A, aux_dict)`` where ``aux_dict = {"entropy": ..., "l0_penalty": None}``.

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
            return V_out, A_out, _make_aux(ent)

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

        return V_out.contiguous(), A_out, _make_aux(entropy)

    @property
    def score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        return getattr(self, 'P_edge_for_reg', None)


class ScaledDotAttention(nn.Module):
    """
    Simplified Scaled Dot-Product Attention.

    Hard mask is applied BEFORE softmax to ensure masked positions don't
    influence the softmax normalization (preventing information leakage).

    Returns ``(V, A, aux_dict)`` where ``aux_dict = {"entropy": ..., "l0_penalty": None}``.
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
        # Value-structure QUERY injection: per-QUERY value term already
        # projected to the value output width (shape (B, L, d) or (B, L, H, d)).
        # Added as ``(sum_j A_ij) * value_query`` — the exact, memory-cheap
        # decomposition of concatenating the query identity into a linear,
        # bias-free W_V^q.
        value_query: torch.Tensor = None,
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

        return V.contiguous(), A, _make_aux(entropy)

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

    Returns ``(V, A, aux_dict)`` where ``aux_dict = {"entropy": ..., "l0_penalty": None}``.
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

        return V.contiguous(), A, _make_aux(entropy)


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

    Returns ``(V_out, att, entropy_tensor)`` — note: raw entropy tensor, NOT aux_dict.
    Callers that wrap this (the attention module forward methods) are responsible
    for wrapping the entropy into ``_make_aux(entropy)``.
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
        if value.dim() == 4:
            # shared_dag_across_heads: Q/K are 3-D (H_struct=1), V is 4-D (B, S, H, d)
            V_out = torch.einsum("bls,bshd->blhd", att, value)
        else:
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
        - CausalCrossAttention    (ReLU(Tanh) activation, constant tau)
        - SigmoidCrossAttention   (Sigmoid activation, constant tau)
        - ToeplitzAttention       (Toeplitz decomposition, constant tau + learnable gate bias)
        - HardConcreteCrossAttention  (Hard Concrete L0 gates, constant beta/gamma/zeta)

    All inner attention modules return ``(V, A, aux_dict)`` where
    ``aux_dict = {"entropy": <Tensor|None>, "l0_penalty": <Tensor|None>}``.
    ``AttentionLayer.forward()`` passes ``aux_dict`` through as the 3rd return
    value, so callers receive the full dictionary.

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
                  / ToeplitzAttention / HardConcreteCrossAttention (non-learnable).
                  Default 3.0; for HardConcreteCrossAttention the paper default is 2/3.
        init_gate_bias: Initial value of the gate bias in ToeplitzAttention (default: -15.0).
            Ignored for other attention types.
        gate_bias_trainable: If True (default), the gate bias is updated during training.
            If False, it is frozen at init_gate_bias. Only applies to ToeplitzAttention.
        init_gamma: Stretch lower bound for HardConcreteCrossAttention (default: -0.1).
            Ignored for other attention types.
        init_zeta: Stretch upper bound for HardConcreteCrossAttention (default: 1.1).
            Ignored for other attention types.
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
        init_gamma: float = -0.1,
        init_zeta: float = 1.1,
        # Additive init-balancing logit offset on the GatedCrossAttention
        # structure gate only (ignored by all other inner attentions).  See
        # GatedCrossAttention.init_edge_offset; ln 3 (~1.0986) lands the cross
        # S->X init edge probability at 0.25 to match a directed self edge.
        init_edge_offset: float = 0.0,
        gain_tau: float = 1.0,
        dir_tau: float = 2.0 / 3.0,

        # CommutatorSelfAttention direction-gate parametrisation (see that
        # module).  "qk" (default) keeps the antisymmetric-of-raw direction;
        # "skew_query" builds a learnable so(d) commutator on the query alone
        # (Option B).  ``direction_dim`` is derived internally from the
        # projected-query width; ``direction_rank`` sets the rank of Ω.
        direction_mode: str = "qk",
        direction_rank: Optional[int] = None,

        # When False, bypass the learnable reconstruction gain in the gated
        # attentions (GatedCrossAttention / GatedSelfAttention): the structure
        # gate becomes the final attention weight (A = z instead of A = z*g).
        use_gain: bool = True,
        # Centroid-collapse fix (GatedCrossAttention / GatedSelfAttention only):
        # L2-normalise the structural query and use a fixed sqrt(query_fanin_scale)
        # score scale in place of 1/sqrt(E).
        normalize_query: bool = False,
        query_fanin_scale: float = 1.0,
        # Learnable per-node query-norm multiplier (only meaningful with
        # ``normalize_query=True``); threaded into the gated inner attentions.
        # Each child owns ``M_i = exp(log_scale_i)`` (init ``query_norm_init_scale``)
        # scaling its unit query so it can ADAPTIVELY overspend the directional
        # budget when the structural signal pays for it; the structural loss
        # charges ``relu(M_i - query_norm_target)^2``.  The per-node count is the
        # number of query rows (children) = ``query_seq_len``.  Disabled (or
        # init_scale=1.0) reproduces the plain unit-norm cap.
        query_norm_learnable: bool = False,
        query_norm_init_scale: float = 1.0,
        query_norm_target: float = 1.0,
        shared_qk_inner: dict = None,


        # Externally-projected query.  When True, this layer does NOT build its
        # own structural ``query_projection`` (W_q).  Instead the caller passes
        # an ALREADY-PROJECTED query tensor of shape
        # ``(B, L, d_queries_keys * n_heads_struct)`` to ``forward`` and it is
        # fed straight to the inner attention (bypassing W_q and its dropout).
        # This lets several AttentionLayer blocks share ONE W_q owned by the
        # parent module (e.g. a cross S→X block and a self X→X block sharing the
        # same ``Q = emb(X) W_q``).  The key/value projections and the inner
        # attention remain per-layer.  Mutually exclusive with ``shared_qk_inner``.
        query_external: bool = False,

        # Externally-projected key.  Symmetric to ``query_external``: when True,
        # this layer does NOT build its own structural ``key_projection`` (W_K).
        # Instead the caller passes an ALREADY-PROJECTED key tensor of shape
        # ``(B, S, d_queries_keys * n_heads_struct)`` to ``forward`` and it is
        # fed straight to the inner attention (bypassing W_K and its dropout).
        # This lets a self X→X block reuse the SINGLE W_K owned by the parent's
        # cross S→X block, so S and X keys pass through the same (isometric)
        # projection and stay mutually orthogonal.  Mutually exclusive with
        # ``shared_qk_inner``.
        key_external: bool = False,

        # Remove the structural query/key projections entirely: read the query /
        # keys DIRECTLY from the (structural) embeddings, dropping W_q / W_K.
        # Implemented as an ``nn.Identity`` in place of the ``nn.Linear`` so all
        # forward paths (including the shared-projection self block) work
        # unchanged; the score becomes <e_i^q, e_j^k> on the raw embeddings.
        # Requires the embedding width to match the inner-attention Q/K width
        # (``d_queries_keys * n_heads_struct``) — true when ``d_qk == d_model``
        # and ``shared_dag_across_heads=True``.  Mutually superseded by the
        # ``*_external`` flags (a shared/external projection takes priority).
        # Motivation: the shared W_q couples every query and aligns them to the
        # keys, numbing the per-node query embeddings (see the SELF_ATTENTION
        # spurious-S3->X4 investigation); dropping it isolates the single-node
        # embedding gradient.
        remove_query_projection: bool = False,
        remove_key_projection: bool = False,
        # Freeze the structural query/key projections at initialisation
        # (``requires_grad=False``).  Because ``classify_parameters`` keeps only
        # ``requires_grad=True`` params, a frozen projection is dropped from the
        # structural optimiser group and the freeze PERSISTS across adaptive
        # phase switches.  No-op for external (None) or removed (Identity)
        # projections, which own no parameters.
        freeze_query_projection: bool = False,
        freeze_key_projection: bool = False,


        shared_dag_across_heads: bool = True,

        dual_value: bool = False,
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        optuna_protocol: Optional[float] = None,
        # Value-structure injection (source-node identity concatenated onto the
        # value stream before W_V).  When >0, the reconstruction value_projection
        # accepts ``d_model_values + value_structure_dim`` inputs and the caller
        # must pass a ``value_structure`` tensor of that trailing width to
        # ``forward``.  0 (default) preserves the original data-only value stream.
        value_structure_dim: int = 0,
        # Value-structure QUERY injection (target/child identity).  When >0 the
        # value additionally depends on the QUERY node:
        #     V_ij = W_V([v_j ; e_j])  +  W_V^q(e_i^q).
        # Because W_V is linear this decomposes into the existing source-shared
        # value PLUS an additive term contributed by the query identity, scaled
        # by the attention row-sum ``sum_j A_ij`` (computed inside the inner
        # attention, where the TRUE applied weights are known).  A dedicated
        # bias-free ``value_query_proj`` maps the query identity of width
        # ``value_structure_query_dim`` to the value output width; the caller
        # passes a per-QUERY ``value_structure_query`` tensor of that trailing
        # width to ``forward``.  Only the gated (GatedCrossAttention /
        # GatedSelfAttention) and ScaledDotAttention inner modules support the
        # additive query term.  0 (default) disables it.
        value_structure_query_dim: int = 0,
    ):
        super(AttentionLayer, self).__init__()




        self._uses_shared_structure = shared_qk_inner is not None
        self._query_external = bool(query_external)
        if self._query_external and self._uses_shared_structure:
            raise ValueError(
                "query_external=True is mutually exclusive with shared_qk_inner "
                "(the query projection is either externally owned OR shared via "
                "the shared_qk_inner bundle, not both)."
            )
        self._key_external = bool(key_external)
        if self._key_external and self._uses_shared_structure:
            raise ValueError(
                "key_external=True is mutually exclusive with shared_qk_inner "
                "(the key projection is either externally owned OR shared via "
                "the shared_qk_inner bundle, not both)."
            )

        # Remove-projection flags (read Q/K straight from the embeddings).  An
        # external / shared projection takes priority (nothing is built here in
        # that case), so removal only applies to a locally-owned projection.
        self._remove_query_projection = bool(remove_query_projection)
        self._remove_key_projection = bool(remove_key_projection)

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

            # Attention classes that accept init_tau
            ATTENTION_WITH_TAU = (
                CausalCrossAttention,
                SigmoidCrossAttention,
                ToeplitzAttention,
                HardConcreteCrossAttention,
            )

            # Create inner attention module
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
            elif attention is HardConcreteCrossAttention:
                # HardConcreteCrossAttention also needs gamma / zeta stretch params
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    gamma=init_gamma,
                    zeta=init_zeta,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                )
            elif attention is GatedCrossAttention:
                # GatedCrossAttention: HardConcrete structure gate x sigmoid
                # reconstruction gain.  Needs gamma/zeta (gate) + gain_tau.
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    gamma=init_gamma,
                    zeta=init_zeta,
                    # Init-balancing offset on the S->X existence gate (lands the
                    # cross init edge prob at sigmoid(-offset); ln 3 -> 0.25).
                    init_edge_offset=init_edge_offset,
                    gain_tau=gain_tau,
                    use_gain=use_gain,
                    normalize_query=normalize_query,
                    query_fanin_scale=query_fanin_scale,
                    query_norm_learnable=query_norm_learnable,
                    query_norm_init_scale=query_norm_init_scale,
                    query_norm_target=query_norm_target,
                    query_norm_num_nodes=query_seq_len,
                    batch_key_dropout=batch_key_dropout,

                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                    # Optuna capacity-search protocol: freeze the STRUCTURE gate
                    # at a constant (0/1) while the reconstruction gain learns.
                    optuna_protocol=optuna_protocol,
                )

            elif attention is CommutatorSelfAttention:
                # CommutatorSelfAttention: GCA-style asymmetric HardConcrete
                # existence gate on the raw alignment × antisymmetric direction
                # gate × sigmoid reconstruction gain.  The direction gate is
                # parametrised by ``direction_mode`` ("qk" antisymmetric-of-raw,
                # or "skew_query" learnable so(d) commutator on the query — the
                # ``direction_dim`` is the projected-query width).
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    gamma=init_gamma,
                    zeta=init_zeta,
                    dir_tau=dir_tau,
                    gain_tau=gain_tau,
                    use_gain=use_gain,
                    normalize_query=normalize_query,
                    query_fanin_scale=query_fanin_scale,
                    query_norm_learnable=query_norm_learnable,
                    query_norm_init_scale=query_norm_init_scale,
                    query_norm_target=query_norm_target,
                    query_norm_num_nodes=query_seq_len,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                    optuna_protocol=optuna_protocol,
                    direction_mode=direction_mode,

                    direction_dim=d_queries_keys * n_heads_struct,
                    direction_rank=direction_rank,
                )
            elif attention is GatedSelfAttention:
                # GatedSelfAttention: direction-aware selector.  Toeplitz split
                # of the structural score → symmetric HardConcrete existence gate
                # (gamma/zeta, init_tau) × antisymmetric coupled direction gate
                # (dir_tau) × sigmoid reconstruction gain (gain_tau).
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    gamma=init_gamma,
                    zeta=init_zeta,
                    dir_tau=dir_tau,
                    gain_tau=gain_tau,
                    use_gain=use_gain,
                    normalize_query=normalize_query,
                    query_fanin_scale=query_fanin_scale,
                    query_norm_learnable=query_norm_learnable,
                    query_norm_init_scale=query_norm_init_scale,
                    query_norm_target=query_norm_target,
                    query_norm_num_nodes=query_seq_len,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                    optuna_protocol=optuna_protocol,
                )
            elif attention is CausalCrossAttention:




                # CausalCrossAttention also accepts the Optuna capacity-search
                # protocol flag (constant-score override).
                self.inner_attention = attention(
                    attention_dropout=attention_dropout,
                    register_entropy=register_entropy,
                    layer_name=layer_name,
                    init_tau=init_tau,
                    batch_key_dropout=batch_key_dropout,
                    batch_key_dropout_p_final=batch_key_dropout_p_final,
                    batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                    optuna_protocol=optuna_protocol,
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

            # Q and K projections.  When ``query_external`` the query arrives
            # already projected (a parent block owns the shared W_q), so this
            # block must NOT create its own W_q — otherwise it would register an
            # unused parameter that never receives gradients.
            _q_out = d_queries_keys * n_heads_struct
            if self._query_external:
                self.query_projection = None
            elif self._remove_query_projection:
                # No W_q: the projected query IS the (structural) query embedding.
                # Requires the embedding width to equal the inner-attention query
                # width so the raw embedding can be scored directly.
                if d_model_queries != _q_out:
                    raise ValueError(
                        "remove_query_projection=True requires d_model_queries == "
                        f"d_queries_keys * n_heads_struct, got {d_model_queries} != "
                        f"{_q_out}.  Set d_qk so d_qk == d_model and keep "
                        "shared_dag_across_heads=True."
                    )
                self.query_projection = nn.Identity()
            else:
                self.query_projection = nn.Linear(d_model_queries, _q_out)


            self.key_projection_type = key_projection_type
            _k_out = d_queries_keys * n_heads_struct
            if self._key_external:
                # Externally-owned W_K: a parent block projects the key and passes
                # it in already projected.  Build no key_projection here so this
                # layer registers no unused parameter (mirrors query_external).
                self.key_projection = None
            elif self._remove_key_projection:
                # No W_K: keys are read straight from the (structural) embedding;
                # ``key_projection_type`` is intentionally bypassed.
                if d_model_keys != _k_out:
                    raise ValueError(
                        "remove_key_projection=True requires d_model_keys == "
                        f"d_queries_keys * n_heads_struct, got {d_model_keys} != "
                        f"{_k_out}."
                    )
                self.key_projection = nn.Identity()
            elif key_projection_type == "linear":

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

        # ---- Optional freeze of the structural Q/K projections --------------
        # Set requires_grad=False so ``classify_parameters`` (which keeps only
        # requires_grad=True params) drops them from the structural optimiser
        # group; the freeze then PERSISTS across adaptive phase switches.  No-op
        # for external (None) or removed (Identity) projections, which own no
        # parameters.
        if freeze_query_projection and self.query_projection is not None:
            for _p in self.query_projection.parameters():
                _p.requires_grad_(False)
        if freeze_key_projection and self.key_projection is not None:
            for _p in self.key_projection.parameters():
                _p.requires_grad_(False)

        # V projection, output projection, and dropout are always per-layer.

        # Value-structure injection: when ``value_structure_dim > 0`` the value
        # projection accepts the data value CONCATENATED with a source-node
        # identity code of that width (``V_j = W_V([v_j ; e_j])``), giving the
        # otherwise data-only value stream a per-source-node functional.  The
        # output width (``d_model_values * n_heads``) is unchanged.
        self.value_structure_dim = int(value_structure_dim)
        self.value_projection = nn.Linear(
            d_model_values + self.value_structure_dim, d_model_values * n_heads
        )
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_values)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

        # Value-structure QUERY injection.  A bias-free projection maps the
        # per-QUERY identity code (width ``value_structure_query_dim``) to the
        # value output width (``d_model_values * n_heads``).  The additive query
        # term ``(sum_j A_ij) * W_V^q(e_i^q)`` is applied INSIDE the inner
        # attention (which alone knows the true applied weights A).  The
        # projection is deliberately bias-free: the bias already lives in the
        # key/value path ``value_projection`` — see the linearity decomposition
        # in the constructor docstring.  Named ``value_query_proj`` (NOT
        # "query_projection") so the name-based gradient router classifies it as
        # a RECONSTRUCTION parameter.
        self.value_structure_query_dim = int(value_structure_query_dim)
        # Which inner attentions support the additive query term.
        self._value_query_capable = attention in (
            GatedCrossAttention,
            GatedSelfAttention,
            CommutatorSelfAttention,
            ScaledDotAttention,
        )
        if self.value_structure_query_dim > 0:
            if not self._value_query_capable:
                raise ValueError(
                    "value_structure_query_dim > 0 requires an inner attention "
                    "that supports the additive query term "
                    "(GatedCrossAttention, GatedSelfAttention, ScaledDotAttention); "
                    f"got {attention.__name__}."
                )
            self.value_query_proj = nn.Linear(
                self.value_structure_query_dim, d_model_values * n_heads, bias=False
            )
        else:
            self.value_query_proj = None



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

        # Reconstruction-gain projections (GatedCrossAttention only).
        # Named ``gain_q_proj`` / ``gain_k_proj`` — deliberately WITHOUT the
        # substrings "query_projection" / "key_projection" — so the name-based
        # gradient router classifies them as RECONSTRUCTION parameters (driven
        # by the MSE loss), keeping them disentangled from the structural gate.
        self._gated_gain = (
            attention in (GatedCrossAttention, GatedSelfAttention, CommutatorSelfAttention)
        ) and (shared_qk_inner is None)
        if self._gated_gain:
            if not self.shared_dag_across_heads:
                raise ValueError(
                    "GatedCrossAttention / GatedSelfAttention / "
                    "CommutatorSelfAttention require shared_dag_across_heads=True "
                    "(single structural head)."
                )

            self.gain_q_proj = nn.Linear(d_model_queries, d_queries_keys)
            self.gain_k_proj = nn.Linear(d_model_keys, d_queries_keys)
        else:
            self.gain_q_proj = None
            self.gain_k_proj = None

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
        gain_query: torch.Tensor = None,
        gain_key: torch.Tensor = None,
        value_structure: torch.Tensor = None,
        value_structure_query: torch.Tensor = None,
    ):
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        H_struct = self._n_heads_struct


        # External-query mode: ``query`` is ALREADY projected by a W_q owned by
        # the parent module (shape (B, L, d_queries_keys * H_struct)).  Feed it
        # straight through, bypassing this layer's (absent) query_projection and
        # its dropout — the parent is responsible for any query dropout.
        if H_struct > 1:
            if self._query_external:
                q = query.view(B, L, H_struct, -1)
            else:
                q = self.dropout_qkv(self.query_projection(query)).view(B, L, H_struct, -1)
            if self._key_external:
                k = key.view(B, S, H_struct, -1)
            else:
                k = self.dropout_qkv(self.key_projection(key)).view(B, S, H_struct, -1)
        else:
            if self._query_external:
                q = query.view(B, L, -1)
            else:
                q = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            if self._key_external:
                k = key.view(B, S, -1)
            else:
                k = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)



        # Value-structure injection: concatenate the per-source-node identity
        # code onto the data value BEFORE the (widened) reconstruction W_V, so
        # V_j = W_V([v_j ; e_j]).  Only active when the layer was built with
        # value_structure_dim > 0 AND a value_structure tensor is supplied.
        if self.value_structure_dim > 0 and value_structure is not None:
            value = torch.cat([value, value_structure], dim=-1)

        if H > 1:
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            v = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)

        # Value-structure QUERY injection: project the per-QUERY identity code
        # to the value output width and reshape to match the value-head layout.
        # The additive term ``(sum_j A_ij) * value_query`` is applied INSIDE the
        # inner attention (only there are the true applied weights A available).
        value_query = None
        if self.value_query_proj is not None and value_structure_query is not None:
            vq = self.value_query_proj(value_structure_query)  # (B, L, d_model_values * H)
            if H > 1:
                value_query = vq.view(B, L, H, -1)
            else:
                value_query = vq.view(B, L, -1)
        # Only pass ``value_query`` to inner attentions that support it; other
        # attention types do not accept the kwarg.
        vq_kwargs = {"value_query": value_query} if self._value_query_capable else {}

        if self._gated_gain:

            # Reconstruction-gain stream (GatedCrossAttention).  gain_query /
            # gain_key default to the structural query/key inputs when the
            # caller does not supply dedicated gain embeddings (shared mode).
            gq_in = gain_query if gain_query is not None else query
            gk_in = gain_key if gain_key is not None else key
            gq = self.dropout_qkv(self.gain_q_proj(gq_in)).view(B, L, -1)
            gk = self.dropout_qkv(self.gain_k_proj(gk_in)).view(B, S, -1)
            out, attn, aux = self.inner_attention(
                query=q,
                key=k,
                value=v,
                mask_miss_k=mask_miss_k,
                mask_miss_q=mask_miss_q,
                pos=pos,
                causal_mask=causal_mask,
                hard_mask=hard_mask,
                oracle=oracle,
                gain_query=gq,
                gain_key=gk,
                **vq_kwargs,
            )
        else:
            out, attn, aux = self.inner_attention(
                query=q,
                key=k,
                value=v,
                mask_miss_k=mask_miss_k,
                mask_miss_q=mask_miss_q,
                pos=pos,
                causal_mask=causal_mask,
                hard_mask=hard_mask,
                oracle=oracle,
                **vq_kwargs,
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

            return out, out_struct, attn, aux

        return out, attn, aux


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

    out_single, score_single, aux = attention_single.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False
    )

    print(f"Single-head - Output shape: {out_single.shape}, Score shape: {score_single.shape}")
    print(f"Single-head - aux keys: {list(aux.keys())}")

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

    out_multi, score_multi, aux = attention_multi.forward(
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

    out_causal, score_causal, aux = attention_multi.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
    )
    print(f"Causal multi-head - Output shape: {out_causal.shape}, Score shape: {score_causal.shape}")

    print("\nTesting HardConcreteCrossAttention (n_heads=1):")
    hc_attention = AttentionLayer(
        attention=HardConcreteCrossAttention,
        d_model_queries=d_model,
        d_model_keys=d_model,
        d_model_values=d_model,
        d_queries_keys=d_queries_keys,
        n_heads=1,
        mask_layer=None,
        attention_dropout=0,
        dropout_qkv=0,
        init_tau=2.0/3.0,
        init_gamma=-0.1,
        init_zeta=1.1,
    )
    hc_attention.train()
    out_hc, score_hc, aux_hc = hc_attention.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False,
    )
    print(f"HardConcrete (train) - Output: {out_hc.shape}, Attn: {score_hc.shape}, "
          f"l0_penalty={aux_hc['l0_penalty']:.3f}")
    hc_attention.eval()
    out_hc_e, score_hc_e, aux_hc_e = hc_attention.forward(
        query=x,
        key=x,
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=None,
        causal_mask=False,
    )
    print(f"HardConcrete (eval)  - Output: {out_hc_e.shape}, Attn: {score_hc_e.shape}, "
          f"l0_penalty={aux_hc_e['l0_penalty']:.3f}")
    print(f"p_edge_on posterior: min={hc_attention.inner_attention.last_p_edge_on.min():.3f}, "
          f"max={hc_attention.inner_attention.last_p_edge_on.max():.3f}")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
