"""
Reversed Decoder implementation for StageCausaliT.

Key difference from standard decoder: Cross-attention comes BEFORE self-attention.
Standard: Self → Cross → FF
Reversed: Cross → Self → FF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules.extra_layers import Normalization


class ReversedDecoderLayer(nn.Module):
    """
    Decoder layer with REVERSED attention order.
    
    Architecture:
    1. Cross-attention (queries from input, keys/values from external source)
    2. Self-attention (queries, keys, values all from previous layer)
    3. Feedforward network
    
    This enables the decoder to first attend to external context (e.g., source nodes)
    before performing self-attention on the combined representation.
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
    ):
        super(ReversedDecoderLayer, self).__init__()
        
        # Attention modules initialized in the parent model
        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention
        
        # Normalization layers
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)

        # Feedforward layers (linear)
        self.linear1 = nn.Linear(in_features=d_model_dec, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_dec, bias=True)
        
        # Dropouts and activation
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self,
        X: torch.Tensor,
        external_context: torch.Tensor,
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
        Forward pass with REVERSED attention order.
        
        Args:
            X: Input tensor (queries for both attentions)
            external_context: External context (keys/values for cross-attention)
            self_mask_miss_k: Missing value mask for self-attention keys
            self_mask_miss_q: Missing value mask for self-attention queries
            cross_mask_miss_k: Missing value mask for cross-attention keys
            cross_mask_miss_q: Missing value mask for cross-attention queries
            dec_input_pos: Positional information for decoder input
            causal_mask: Whether to apply causal masking
            cross_hard_mask: Optional hard mask for cross-attention (L_q, L_k), values in [0,1]
            self_hard_mask: Optional hard mask for self-attention (L, L), values in [0,1]
            
        Returns:
            decoder_out: Output tensor after all operations
            cross_att: Cross-attention weights
            self_att: Self-attention weights
            cross_ent: Cross-attention entropy
            self_ent: Self-attention entropy
        """
        
        not_cross_mask_miss_q = ~cross_mask_miss_q if cross_mask_miss_q is not None else None
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        # Step 1: Cross-attention (FIRST)
        X1 = self.norm1(X, not_cross_mask_miss_q)
        
        X1, cross_att, cross_ent = self.global_cross_attention(
            query=X1,
            key=external_context,
            value=external_context,
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos=None,
            causal_mask=False,
            hard_mask=cross_hard_mask,
        )
        
        X2 = X + self.dropout_attn_out(X1)
        
        # Step 2: Self-attention (SECOND)
        X3 = self.norm2(X2, not_self_mask_miss_q)
        
        X3, self_att, self_ent = self.global_self_attention(
            query=X3,
            key=X3,
            value=X3,
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
        )
        
        X4 = X2 + self.dropout_attn_out(X3)
        
        # Step 3: Feedforward
        X5 = self.norm3(X4, not_self_mask_miss_q)
        
        X5 = self.dropout_ff(self.activation(self.linear1(X5)))
        X5 = self.dropout_ff(self.linear2(X5))
        
        # Final residual connection
        decoder_out = X4 + X5
        
        return decoder_out, cross_att, self_att, cross_ent, self_ent


class ReversedDecoder(nn.Module):
    """
    Stack of ReversedDecoderLayer modules.
    
    This decoder processes input through multiple layers, each with:
    1. Cross-attention to external context
    2. Self-attention on internal representations
    3. Feedforward transformation
    """
    def __init__(
        self,
        decoder_layers: list,
        norm_layer: nn.Module,
        emb_dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)
    
    def forward(
        self,
        X: torch.Tensor,
        external_context: torch.Tensor,
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
        Forward pass through all decoder layers.
        
        Args:
            X: Input tensor
            external_context: External context for cross-attention
            self_mask_miss_k: Self-attention key mask
            self_mask_miss_q: Self-attention query mask
            cross_mask_miss_k: Cross-attention key mask
            cross_mask_miss_q: Cross-attention query mask
            dec_input_pos: Positional information
            causal_mask: Whether to use causal masking
            cross_hard_mask: Optional hard mask for cross-attention (L_q, L_k), values in [0,1]
            self_hard_mask: Optional hard mask for self-attention (L, L), values in [0,1]
            
        Returns:
            X: Output tensor after all layers
            cross_att_list: List of cross-attention weights
            self_att_list: List of self-attention weights
            cross_ent_list: List of cross-attention entropies
            self_ent_list: List of self-attention entropies
        """
        
        X = self.emb_dropout(X)
        
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
            )
            
            cross_att_list.append(cross_att)
            self_att_list.append(self_att)
            cross_ent_list.append(cross_ent)
            self_ent_list.append(self_ent)
        
        if self.norm_layer is not None:
            not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
            X = self.norm_layer(X, not_self_mask_miss_q)
        
        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
