"""
VarianceCausalDecoderLayer — deterministic SVFA dual-residual decoder.

This is the decoder layer for the VarianceCausalLayer architecture
(docs/documentation/NOISE_AWARE_2.md).

Key differences from NoiseAwareDualResDecoderLayer:
- NO noise injection: the ambient noise step is entirely absent.
- No inject_noise parameter anywhere.
- The forward pass is fully deterministic.
- Both attention blocks still update X_struct via dual residual (dual_value=True).

Forward pass per layer
----------------------
    Cross-attention  (dual_value=True):
        Q              = LN_struct(X_struct)
        K              = ext_struct
        V_val          = ext_val
        V_struct_src   = ext_struct           (key_value kwarg)
        out_val, out_struct = CrossAtt(...)
        H_det    = X_val    + dropout(out_val)      ← value residual
        X_struct = X_struct + dropout(out_struct)   ← structural residual

    Self-attention  (dual_value=True):
        Q              = LN_struct(X_struct)
        K              = LN_struct(X_struct)
        V_val          = LN(H_det)
        V_struct_src   = LN_struct(X_struct)  (key_value kwarg)
        out_val, out_struct = SelfAtt(...)
        U        = H_det   + dropout(out_val)       ← value residual
        X_struct = X_struct + dropout(out_struct)   ← structural residual

    Feedforward (value stream only):
        X_val_out = U + FF(LN(U))

The self-attention weights α (returned as self_att) are used by the
model's forward() to compute analytical variance:
    Var(X_i) = Σ_j α_{ij}² · σ_A[j]²
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules.extra_layers import Normalization


class VarianceCausalDecoderLayer(nn.Module):
    """
    Deterministic SVFA dual-residual decoder layer for VarianceCausalLayer.

    Both cross- and self-attention blocks update BOTH the value stream (X_val)
    and the structure stream (X_struct) via separate residual connections,
    exactly as in NoiseAwareDualResDecoderLayer — but without any noise
    injection.

    This requires attention layers built with dual_value=True.

    Args:
        global_cross_attention: Cross-attention module (dual_value=True).
        global_self_attention:  Self-attention module (dual_value=True).
        d_model_dec: Model dimension.
        d_ff: Feedforward hidden dimension.
        dropout_ff: Feedforward dropout rate.
        dropout_attn_out: Attention output dropout rate.
        activation: "relu" or "gelu".
        norm: Normalization type (passed to Normalization).
        shared_struct_norms: Optional dict with keys "norm1_struct" /
            "norm2_struct" for sharing norms across layers when
            share_structure_across_layers=True.
    """

    def __init__(
        self,
        global_cross_attention: nn.Module,
        global_self_attention: nn.Module,
        d_model_dec: int,
        d_ff: int,
        dropout_ff: float,
        dropout_attn_out: float,
        activation: str,
        norm: str,
        shared_struct_norms: dict = None,
    ):
        super().__init__()

        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention

        # Sanity-check: dual_value must be enabled on both attentions.
        for label, att in (
            ("cross", global_cross_attention),
            ("self", global_self_attention),
        ):
            if not getattr(att, "dual_value", False):
                raise ValueError(
                    f"VarianceCausalDecoderLayer requires {label}-attention "
                    f"to be created with dual_value=True (got dual_value=False)."
                )

        # Value-stream pre-norms (cross, self, FFN).
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)

        # Structure-stream pre-norms.  Optionally shared across layers.
        if shared_struct_norms is not None:
            self.norm1_struct = shared_struct_norms["norm1_struct"]
            self.norm2_struct = shared_struct_norms["norm2_struct"]
        else:
            self.norm1_struct = Normalization(method=norm, d_model=d_model_dec)
            self.norm2_struct = Normalization(method=norm, d_model=d_model_dec)

        # Feedforward (value stream only).
        self.linear1 = nn.Linear(d_model_dec, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model_dec, bias=True)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        X: Tuple[torch.Tensor, torch.Tensor],
        external_context,
        self_mask_miss_k: torch.Tensor,
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor,
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool,
        cross_hard_mask: Optional[torch.Tensor] = None,
        self_hard_mask: Optional[torch.Tensor] = None,
    ):
        """
        Deterministic forward pass: dual-residual, no noise.

        Args:
            X: Tuple (X_struct, X_val), each (B, L, d_model).
            external_context: Single tensor or tuple (ext_struct, ext_val).
            self_mask_miss_k:  Missing-value mask for self-attention keys.
            self_mask_miss_q:  Missing-value mask for self-attention queries.
            cross_mask_miss_k: Missing-value mask for cross-attention keys.
            cross_mask_miss_q: Missing-value mask for cross-attention queries.
            dec_input_pos: Positional info forwarded to self-attention.
            causal_mask: Whether to apply causal masking in self-attention.
            cross_hard_mask: Optional hard mask for cross-attention.
            self_hard_mask:  Optional hard mask for self-attention.

        Returns:
            ((X_struct, X_val), cross_att, self_att, cross_ent, self_ent)
            where self_att (B, L, L) is the self-attention weight matrix used
            downstream for analytical variance computation.
        """
        if not isinstance(X, tuple):
            raise ValueError(
                "VarianceCausalDecoderLayer requires SVFA input "
                "(tuple X=(X_struct, X_val)); got a single tensor."
            )
        X_struct, X_val = X

        not_cross_mask_miss_q = (
            ~cross_mask_miss_q if cross_mask_miss_q is not None else None
        )
        not_self_mask_miss_q = (
            ~self_mask_miss_q if self_mask_miss_q is not None else None
        )

        # Unpack external context.
        if isinstance(external_context, tuple):
            ext_struct, ext_val = external_context
        else:
            ext_struct = external_context
            ext_val = external_context

        # ===== Step 1: Cross-attention (dual_value=True) =====
        X_struct_norm = self.norm1_struct(X_struct, not_cross_mask_miss_q)

        cross_out_val, cross_out_struct, cross_att, cross_ent = self.global_cross_attention(
            query=X_struct_norm,    # Q from structure stream
            key=ext_struct,         # K from external structure
            value=ext_val,          # V_val from external value
            key_value=ext_struct,   # V_struct source: external structure
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos=None,
            causal_mask=False,
            hard_mask=cross_hard_mask,
        )

        # Dual residual: update BOTH streams.
        H_det = X_val + self.dropout_attn_out(cross_out_val)           # value path
        X_struct = X_struct + self.dropout_attn_out(cross_out_struct)  # structural path

        # ===== Step 2: Self-attention (dual_value=True) =====
        # NOTE: No noise injection here — H_det goes directly into self-attention.
        X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
        H_norm = self.norm2(H_det, not_self_mask_miss_q)

        self_out_val, self_out_struct, self_att, self_ent = self.global_self_attention(
            query=X_struct_norm,    # Q from structure stream
            key=X_struct_norm,      # K from structure stream
            value=H_norm,           # V_val: deterministic hidden state
            key_value=X_struct_norm,  # V_struct source: structure stream
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
        )

        # Dual residual.
        U = H_det + self.dropout_attn_out(self_out_val)
        X_struct = X_struct + self.dropout_attn_out(self_out_struct)

        # ===== Step 3: Feedforward (value stream only) =====
        U_norm = self.norm3(U, not_self_mask_miss_q)
        U_ff = self.dropout_ff(self.activation(self.linear1(U_norm)))
        U_ff = self.dropout_ff(self.linear2(U_ff))
        X_val_out = U + U_ff

        return (X_struct, X_val_out), cross_att, self_att, cross_ent, self_ent


class VarianceCausalDecoder(nn.Module):
    """
    Stack of VarianceCausalDecoderLayer modules.

    Processes input through multiple deterministic dual-residual layers.
    Both X_struct and X_val evolve through layers.  Final normalisation
    is applied to the value stream only.
    """

    def __init__(
        self,
        decoder_layers: list,
        norm_layer: nn.Module,
        emb_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self,
        X: Tuple[torch.Tensor, torch.Tensor],
        external_context,
        self_mask_miss_k: torch.Tensor,
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor,
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool,
        cross_hard_mask: torch.Tensor = None,
        self_hard_mask: torch.Tensor = None,
    ):
        """
        Forward through all layers.

        Returns:
            (X, cross_att_list, self_att_list, cross_ent_list, self_ent_list)
        """
        if not isinstance(X, tuple):
            raise ValueError(
                "VarianceCausalDecoder requires SVFA input "
                "(tuple X=(X_struct, X_val))."
            )

        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None

        X_struct, X_val = X
        X_struct = self.emb_dropout(X_struct)
        X_val = self.emb_dropout(X_val)
        X = (X_struct, X_val)

        cross_att_list, self_att_list = [], []
        cross_ent_list, self_ent_list = [], []

        for layer in self.layers:
            X, cross_att, self_att, cross_ent, self_ent = layer(
                X=X,
                external_context=external_context,
                self_mask_miss_k=self_mask_miss_k,
                self_mask_miss_q=self_mask_miss_q,
                cross_mask_miss_k=cross_mask_miss_k,
                cross_mask_miss_q=cross_mask_miss_q,
                dec_input_pos=dec_input_pos,
                causal_mask=causal_mask,
                cross_hard_mask=cross_hard_mask,
                self_hard_mask=self_hard_mask,
            )
            cross_att_list.append(cross_att)
            self_att_list.append(self_att)
            cross_ent_list.append(cross_ent)
            self_ent_list.append(self_ent)

        # Final normalisation on value stream only.
        if self.norm_layer is not None:
            X_struct, X_val = X
            X_val = self.norm_layer(X_val, not_mask)
            X = (X_struct, X_val)

        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
