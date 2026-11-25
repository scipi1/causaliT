"""Decoder for SingleCausalLayerRes (SVFA dual-residual).

Mirror of ``causaliT.core.architectures.stage_causal.decoder`` but the
inner layer applies a residual connection to BOTH streams of the SVFA
factorization:

    Cross-attention:
        Q     = LN(X_struct)
        K, KV = ext_struct
        V     = ext_val
        out_val, out_struct = AttentionLayer(...)         # dual_value=True
        X_val    <- X_val    + dropout(out_val)
        X_struct <- X_struct + dropout(out_struct)

    Self-attention:
        Q, K, KV = LN(X_struct)
        V        = LN(X_val)
        out_val, out_struct = AttentionLayer(...)         # dual_value=True
        X_val    <- X_val    + dropout(out_val)
        X_struct <- X_struct + dropout(out_struct)

    Feedforward:
        X_val <- X_val + FFN(LN(X_val))     # value stream only

Notes
-----
* The structural residual is the only path through which the structure
  embedding is updated layer-by-layer, so each decoder layer can re-use
  Q/K projections to produce a *different* attention pattern as X_struct
  evolves.
* The structural value matrices ``value_projection_struct`` and
  ``out_projection_struct`` (created inside ``AttentionLayer`` when
  ``dual_value=True``) are routed to the structural-loss optimizer group
  via the patterns added in ``causaliT/training/gradient_routing.py``.
* No FFN is applied to ``X_struct`` — the FFN remains a value-stream
  reconstruction operation.

The decoder is otherwise a drop-in replacement for ``ReversedDecoder``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules.extra_layers import Normalization


class DualResidualDecoderLayer(nn.Module):
    """Reversed decoder layer with SVFA dual residuals (X_struct + X_val).

    Forward order: cross-attention -> self-attention -> feedforward.

    Both attentions use ``AttentionLayer(dual_value=True)`` so that the
    SAME Q,K-derived score produces two outputs: one acting on V_val
    (value stream, reconstruction-driven) and one acting on V_struct
    (structure stream, structural-loss-driven via gradient routing).
    """

    def __init__(
        self,
        global_cross_attention,
        global_self_attention,
        d_model_dec,
        activation,
        norm,
        d_ff,
        dropout_ff,
        dropout_attn_out,
        shared_struct_norms: dict = None,
    ):
        super().__init__()

        # Attention modules initialized in the parent model. Both MUST have
        # ``dual_value=True`` so that they return a 4-tuple.
        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention

        # Sanity-check: dual_value must be enabled on both attentions.
        for name, att in (
            ("cross", global_cross_attention),
            ("self", global_self_attention),
        ):
            if not getattr(att, "dual_value", False):
                raise ValueError(
                    f"DualResidualDecoderLayer requires {name}-attention to "
                    f"be created with dual_value=True (got dual_value=False)."
                )

        # Value-stream pre-norms (one per attention block + one for the FFN).
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)

        # Structure-stream pre-norms. Optionally shared across layers
        # (matches the existing ReversedDecoderLayer convention so layered
        # SVFA models with ``share_structure_across_layers=True`` continue
        # to enforce identical DAGs across layers).
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
        X,
        external_context,
        self_mask_miss_k: torch.Tensor,
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor,
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool,
        cross_hard_mask: torch.Tensor = None,
        self_hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        """Forward pass with SVFA dual residuals.

        Args:
            X: Tuple ``(X_struct, X_val)``, each of shape (B, L, d_model).
            external_context: Either a single tensor (used for both K, V) or
                a tuple ``(ext_struct, ext_val)``.
            ...

        Returns:
            ``((X_struct, X_val), cross_att, self_att, cross_ent, self_ent)``
        """
        if not isinstance(X, tuple):
            raise ValueError(
                "DualResidualDecoderLayer requires SVFA input "
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

        # ===== Step 1: Cross-attention =====
        X_struct_norm = self.norm1_struct(X_struct, not_cross_mask_miss_q)

        cross_out_val, cross_out_struct, cross_att, cross_ent = self.global_cross_attention(
            query=X_struct_norm,        # Q from structure
            key=ext_struct,             # K from external structure
            value=ext_val,              # V_val from external value
            key_value=ext_struct,       # V_struct source: same as K (external structure)
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos=None,
            causal_mask=False,
            hard_mask=cross_hard_mask,
            oracle=oracle,
        )

        # Dual residual: update BOTH streams.
        X_val = X_val + self.dropout_attn_out(cross_out_val)
        X_struct = X_struct + self.dropout_attn_out(cross_out_struct)

        # ===== Step 2: Self-attention =====
        X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
        X_val_norm = self.norm2(X_val, not_self_mask_miss_q)

        self_out_val, self_out_struct, self_att, self_ent = self.global_self_attention(
            query=X_struct_norm,        # Q from structure
            key=X_struct_norm,          # K from structure
            value=X_val_norm,           # V_val from value stream
            key_value=X_struct_norm,    # V_struct source: structure stream
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
            oracle=oracle,
        )

        X_val = X_val + self.dropout_attn_out(self_out_val)
        X_struct = X_struct + self.dropout_attn_out(self_out_struct)

        # ===== Step 3: Feedforward (value stream only) =====
        X_val_norm = self.norm3(X_val, not_self_mask_miss_q)
        X_val_ff = self.dropout_ff(self.activation(self.linear1(X_val_norm)))
        X_val_ff = self.dropout_ff(self.linear2(X_val_ff))
        X_val = X_val + X_val_ff

        return (X_struct, X_val), cross_att, self_att, cross_ent, self_ent


class DualResidualDecoder(nn.Module):
    """Stack of ``DualResidualDecoderLayer`` modules (SVFA dual residual)."""

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
        # Hard-code SVFA semantics: this decoder is dual-residual SVFA-only.
        self.factorization = "svfa"

    def forward(
        self,
        X,
        external_context,
        self_mask_miss_k: torch.Tensor,
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor,
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool,
        cross_hard_mask: torch.Tensor = None,
        self_hard_mask: torch.Tensor = None,
        oracle: bool = False,
    ):
        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None

        if not isinstance(X, tuple):
            raise ValueError(
                "DualResidualDecoder requires SVFA input "
                "(tuple X=(X_struct, X_val))."
            )
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
                oracle=oracle,
            )
            cross_att_list.append(cross_att)
            self_att_list.append(self_att)
            cross_ent_list.append(cross_ent)
            self_ent_list.append(self_ent)

        # Final normalization (value stream only — keeps parity with
        # ReversedDecoder).
        if self.norm_layer is not None:
            X_struct, X_val = X
            X_val = self.norm_layer(X_val, not_mask)
            X = (X_struct, X_val)

        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
