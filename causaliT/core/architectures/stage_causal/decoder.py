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
    
    SVFA Mode (factorization="svfa"):
    - X is tuple (X_struct, X_val)
    - external_context can be single tensor or tuple (ext_struct, ext_val)
    - Cross-attention: Q from X_struct, K from ext (or ext_struct), V from ext (or ext_val)
    - Self-attention: Q, K from X_struct, V from X_val
    - Only X_val is updated; X_struct passes through unchanged
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
        factorization: str = "standard"
    ):
        super(ReversedDecoderLayer, self).__init__()
        
        # Attention modules initialized in the parent model
        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention
        self.factorization = factorization
        
        # Normalization layers
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)
        
        # For SVFA: separate normalization for structure embeddings
        if factorization == "svfa":
            self.norm1_struct = Normalization(method=norm, d_model=d_model_dec)
            self.norm2_struct = Normalization(method=norm, d_model=d_model_dec)

        # Feedforward layers (linear)
        self.linear1 = nn.Linear(in_features=d_model_dec, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_dec, bias=True)
        
        # Dropouts and activation
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
    ):
        """
        Forward pass with REVERSED attention order.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (X_struct, X_val) each (B, L, d_model)
            external_context: In standard mode: tensor (B, S, d_model)
                             In SVFA mode: single tensor or tuple (ext_struct, ext_val)
            self_mask_miss_k: Missing value mask for self-attention keys
            self_mask_miss_q: Missing value mask for self-attention queries
            cross_mask_miss_k: Missing value mask for cross-attention keys
            cross_mask_miss_q: Missing value mask for cross-attention queries
            dec_input_pos: Positional information for decoder input
            causal_mask: Whether to apply causal masking
            cross_hard_mask: Optional hard mask for cross-attention (L_q, L_k), values in [0,1]
            self_hard_mask: Optional hard mask for self-attention (L, L), values in [0,1]
            
        Returns:
            In standard mode: (decoder_out, cross_att, self_att, cross_ent, self_ent)
            In SVFA mode: ((X_struct, X_val), cross_att, self_att, cross_ent, self_ent)
        """
        
        not_cross_mask_miss_q = ~cross_mask_miss_q if cross_mask_miss_q is not None else None
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        if self.factorization == "svfa":
            # SVFA mode: unpack X tuple
            X_struct, X_val = X
            
            # Handle external_context - can be single tensor or tuple
            if isinstance(external_context, tuple):
                ext_struct, ext_val = external_context
            else:
                # External context is single tensor (e.g., OrthogonalMaskEmbedding)
                # Use same tensor for both K and V
                ext_struct = external_context
                ext_val = external_context
            
            # === Step 1: Cross-attention (FIRST) ===
            # Q from X_struct, K from ext_struct, V from ext_val
            X_struct_norm = self.norm1_struct(X_struct, not_cross_mask_miss_q)
            X_val_norm = self.norm1(X_val, not_cross_mask_miss_q)
            
            cross_attn_out, cross_att, cross_ent = self.global_cross_attention(
                query=X_struct_norm,        # Q: structure
                key=ext_struct,             # K: external structure (or full embedding)
                value=ext_val,              # V: external value (or full embedding)
                mask_miss_k=cross_mask_miss_k,
                mask_miss_q=cross_mask_miss_q,
                pos=None,
                causal_mask=False,
                hard_mask=cross_hard_mask,
            )
            
            # Residual on VALUE only
            X_val = X_val + self.dropout_attn_out(cross_attn_out)
            
            # === Step 2: Self-attention (SECOND) ===
            # Q, K from X_struct, V from X_val
            X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
            X_val_norm = self.norm2(X_val, not_self_mask_miss_q)
            
            self_attn_out, self_att, self_ent = self.global_self_attention(
                query=X_struct_norm,        # Q: structure
                key=X_struct_norm,          # K: structure
                value=X_val_norm,           # V: value
                mask_miss_k=self_mask_miss_k,
                mask_miss_q=self_mask_miss_q,
                pos=dec_input_pos,
                causal_mask=causal_mask,
                hard_mask=self_hard_mask,
            )
            
            # Residual on VALUE only
            X_val = X_val + self.dropout_attn_out(self_attn_out)
            
            # === Step 3: Feedforward ===
            X_val_norm = self.norm3(X_val, not_self_mask_miss_q)
            X_val_ff = self.dropout_ff(self.activation(self.linear1(X_val_norm)))
            X_val_ff = self.dropout_ff(self.linear2(X_val_ff))
            
            # Final residual on value
            X_val = X_val + X_val_ff
            
            # Structure passes through unchanged
            return (X_struct, X_val), cross_att, self_att, cross_ent, self_ent
        
        else:
            # Standard mode
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
    
    SVFA Mode (factorization="svfa"):
    - X is tuple (X_struct, X_val)
    - Only X_val is updated; X_struct passes through unchanged
    """
    def __init__(
        self,
        decoder_layers: list,
        norm_layer: nn.Module,
        emb_dropout: float,
        factorization: str = "standard"
    ):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.factorization = factorization
    
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
    ):
        """
        Forward pass through all decoder layers.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (X_struct, X_val) each (B, L, d_model)
            external_context: External context for cross-attention
                             Can be tensor or tuple in SVFA mode
            self_mask_miss_k: Self-attention key mask
            self_mask_miss_q: Self-attention query mask
            cross_mask_miss_k: Cross-attention key mask
            cross_mask_miss_q: Cross-attention query mask
            dec_input_pos: Positional information
            causal_mask: Whether to use causal masking
            cross_hard_mask: Optional hard mask for cross-attention (L_q, L_k), values in [0,1]
            self_hard_mask: Optional hard mask for self-attention (L, L), values in [0,1]
            
        Returns:
            In standard mode: (X, cross_att_list, self_att_list, cross_ent_list, self_ent_list)
            In SVFA mode: ((X_struct, X_val), cross_att_list, self_att_list, cross_ent_list, self_ent_list)
        """
        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        # Apply embedding dropout
        if self.factorization == "svfa":
            X_struct, X_val = X
            X_struct = self.emb_dropout(X_struct)
            X_val = self.emb_dropout(X_val)
            X = (X_struct, X_val)
        else:
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
        
        # Apply final normalization
        if self.norm_layer is not None:
            if self.factorization == "svfa":
                X_struct, X_val = X
                X_val = self.norm_layer(X_val, not_mask)
                X = (X_struct, X_val)
            else:
                X = self.norm_layer(X, not_mask)
        
        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
