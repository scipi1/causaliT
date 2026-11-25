"""Decoder for NoiseAwareSingleCausalLayerRes (noise-aware SVFA dual-residual).

Mirror of ``causaliT.core.architectures.noise_aware.decoder`` but BOTH
attention blocks apply a residual connection on the STRUCTURE stream as well
as the value stream, exactly as in
``causaliT.core.architectures.single_causal_res.decoder``.

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

    Noise injection:
        H = AmbientNoise(H_det)               ← unchanged from noise_aware

    Self-attention  (dual_value=True):
        Q              = LN_struct(X_struct)
        K              = LN_struct(X_struct)
        V_val          = LN(H)                (noisy hidden state)
        V_struct_src   = LN_struct(X_struct)  (key_value kwarg)
        out_val, out_struct = SelfAtt(...)
        U        = H       + dropout(out_val)      ← value residual
        X_struct = X_struct + dropout(out_struct)  ← structural residual

    Feedforward (value stream only):
        X_val_out = U + FF(LN(U))

Notes
-----
* X_struct evolves through layers just like X_val, enabling the structural
  embedding to be progressively refined by successive attention patterns.
* Noise is injected on the VALUE path only (H_det → H); X_struct is not
  noisy — the Q/K scores remain deterministic as in the base noise_aware
  architecture.
* The structural value matrices (``value_projection_struct`` and
  ``out_projection_struct``) are routed to the structural-loss optimizer
  group via the patterns in ``causaliT/training/gradient_routing.py``.
* No FFN is applied to X_struct (value-stream reconstruction operation only).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules.extra_layers import Normalization
from causaliT.core.modules.noise_layers import AmbientNoiseLayer


class NoiseAwareDualResDecoderLayer(nn.Module):
    """Noise-aware decoder layer with SVFA dual residuals.

    Combines the ambient-noise injection from
    ``NoiseAwareReversedDecoderLayer`` with the dual-residual structural
    update from ``DualResidualDecoderLayer``:

    * Both attentions must be created with ``dual_value=True``.
    * After each attention block the STRUCTURE stream is updated via a
      residual on the structural output (``out_struct``).
    * Between cross- and self-attention, ambient noise is injected on the
      value path (H_det → H).  X_struct is never noised.

    Args:
        global_cross_attention: Cross-attention module (dual_value=True).
        global_self_attention: Self-attention module (dual_value=True).
        ambient_noise_layer: Shared ``AmbientNoiseLayer`` instance.
        d_model_dec: Model dimension.
        d_ff: Feedforward hidden dimension.
        dropout_ff: Feedforward dropout.
        dropout_attn_out: Attention output dropout.
        activation: ``"relu"`` or ``"gelu"``.
        norm: Normalization type string (passed to ``Normalization``).
        shared_struct_norms: Optional dict with keys ``"norm1_struct"`` /
            ``"norm2_struct"`` for sharing norms across layers when
            ``share_structure_across_layers=True``.
    """

    def __init__(
        self,
        global_cross_attention: nn.Module,
        global_self_attention: nn.Module,
        ambient_noise_layer: AmbientNoiseLayer,
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
        self.ambient_noise = ambient_noise_layer

        # Sanity-check: dual_value must be enabled on both attentions.
        for label, att in (
            ("cross", global_cross_attention),
            ("self", global_self_attention),
        ):
            if not getattr(att, "dual_value", False):
                raise ValueError(
                    f"NoiseAwareDualResDecoderLayer requires {label}-attention "
                    f"to be created with dual_value=True (got dual_value=False)."
                )

        # Value-stream pre-norms (cross, self-noisy-H, FFN).
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
        inject_noise: bool = True,
    ):
        """Forward pass: dual-residual + noise-aware.

        Args:
            X: Tuple ``(X_struct, X_val)``, each (B, L, d_model).
            external_context: Single tensor or tuple ``(ext_struct, ext_val)``.
            self_mask_miss_k: Missing-value mask for self-attention keys.
            self_mask_miss_q: Missing-value mask for self-attention queries.
            cross_mask_miss_k: Missing-value mask for cross-attention keys.
            cross_mask_miss_q: Missing-value mask for cross-attention queries.
            dec_input_pos: Positional info forwarded to self-attention.
            causal_mask: Whether to apply causal masking in self-attention.
            cross_hard_mask: Optional hard mask for cross-attention.
            self_hard_mask: Optional hard mask for self-attention.
            inject_noise: If True, inject ambient noise (default True).

        Returns:
            ``((X_struct, X_val), cross_att, self_att, cross_ent, self_ent)``
        """
        if not isinstance(X, tuple):
            raise ValueError(
                "NoiseAwareDualResDecoderLayer requires SVFA input "
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
            query=X_struct_norm,    # Q from structure
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
        H_det = X_val + self.dropout_attn_out(cross_out_val)        # value path
        X_struct = X_struct + self.dropout_attn_out(cross_out_struct)  # structural path

        # ===== Step 2: Ambient noise injection (value path only) =====
        H = self.ambient_noise(H_det, inject_noise=inject_noise)

        # ===== Step 3: Self-attention (dual_value=True) =====
        X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
        H_norm = self.norm2(H, not_self_mask_miss_q)

        self_out_val, self_out_struct, self_att, self_ent = self.global_self_attention(
            query=X_struct_norm,    # Q from structure
            key=X_struct_norm,      # K from structure
            value=H_norm,           # V_val: NOISY hidden state
            key_value=X_struct_norm,  # V_struct source: structure stream
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
        )

        # Dual residual.
        U = H + self.dropout_attn_out(self_out_val)
        X_struct = X_struct + self.dropout_attn_out(self_out_struct)

        # ===== Step 4: Feedforward (value stream only) =====
        U_norm = self.norm3(U, not_self_mask_miss_q)
        U_ff = self.dropout_ff(self.activation(self.linear1(U_norm)))
        U_ff = self.dropout_ff(self.linear2(U_ff))
        X_val_out = U + U_ff

        return (X_struct, X_val_out), cross_att, self_att, cross_ent, self_ent


class NoiseAwareDualResDecoder(nn.Module):
    """Stack of ``NoiseAwareDualResDecoderLayer`` modules."""

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
        inject_noise: bool = True,
    ):
        """Forward through all layers.

        Args:
            X: Tuple ``(X_struct, X_val)`` — SVFA embeddings.
            external_context: External context for cross-attention.
            inject_noise: Whether to inject ambient noise in each layer.
            (remaining args forwarded to each layer unchanged)

        Returns:
            ``(X, cross_att_list, self_att_list, cross_ent_list, self_ent_list)``
        """
        if not isinstance(X, tuple):
            raise ValueError(
                "NoiseAwareDualResDecoder requires SVFA input "
                "(tuple X=(X_struct, X_val))."
            )

        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None

        X_struct, X_val = X
        X_struct = self.emb_dropout(X_struct)
        X_val = self.emb_dropout(X_val)
        X = (X_struct, X_val)

        cross_att_list, self_att_list = [], []
        cross_ent_list, self_ent_list = [], []

        for decoder_layer in self.layers:
            X, cross_att, self_att, cross_ent, self_ent = decoder_layer(
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
                inject_noise=inject_noise,
            )
            cross_att_list.append(cross_att)
            self_att_list.append(self_att)
            cross_ent_list.append(cross_ent)
            self_ent_list.append(self_ent)

        # Final normalization on value stream only.
        if self.norm_layer is not None:
            X_struct, X_val = X
            X_val = self.norm_layer(X_val, not_mask)
            X = (X_struct, X_val)

        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
