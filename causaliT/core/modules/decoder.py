import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath, join
import sys
# sys.path.append(dirname(abspath(__file__)))
from causaliT.core.modules.extra_layers import (
    Normalization,
    UniformAttentionMask
)

# TODO move somewhere else
from datetime import datetime

# ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
# DUMP_DIR = join(ROOT_DIR,"dump")

# Get the current date and time
def time_name(filename):
    now = datetime.now()

    # Format it as YYYYMMDD_HHMMSS (or adjust as needed)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Create a filename with the timestamp
    return filename+f"_{timestamp}"


class DecoderLayer(nn.Module):
    """
    Decoder layer with support for SVFA (Structure-Value Factorized Attention).
    
    In standard mode:
        - X and enc_out are single tensors
        
    In SVFA mode:
        - X is tuple (dec_struct, dec_val) for decoder embeddings
        - enc_out is tuple (enc_struct, enc_val) from encoder
        - Self-attention: Q, K from dec_struct; V from dec_val
        - Cross-attention: Q from dec_struct; K from enc_struct; V from enc_val
        - Only dec_val is updated; dec_struct passes through unchanged
    """
    def __init__(
        self,
        global_self_attention,
        global_cross_attention,
        d_model_dec,
        # d_yt, #(??) #TODO don't need them?
        # d_yc, #(??)
        activation,
        norm,
        d_ff,
        dropout_ff,
        dropout_attn_out,
        factorization: str = "standard"
        ):
        super(DecoderLayer, self).__init__()
        
        # global attention is initialized in the `model.py` module
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        self.factorization = factorization
        
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)

        # For SVFA: separate normalization for structure embeddings
        if factorization == "svfa":
            self.norm1_struct = Normalization(method=norm, d_model=d_model_dec)
            self.norm2_struct = Normalization(method=norm, d_model=d_model_dec)

        # output convolutions or linear
        self.conv1 = nn.Conv1d(in_channels=d_model_dec, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model_dec, kernel_size=1)
        self.linear1 = nn.Linear(in_features=d_model_dec, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_dec, bias=True)
        
        
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        
    def forward(
        self, X, 
        enc_out, 
        self_mask_miss_k: torch.Tensor, 
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor, 
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool
        ):
        """
        Forward pass through decoder layer.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (dec_struct, dec_val) each (B, L, d_model)
            enc_out: In standard mode: tensor (B, S, d_model)
                     In SVFA mode: tuple (enc_struct, enc_val) each (B, S, d_model)
                     
        Returns:
            In standard mode: (decoder_out, self_att, cross_att, self_ent, cross_ent)
            In SVFA mode: ((dec_struct, dec_val), self_att, cross_att, self_ent, cross_ent)
        """
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        if self.factorization == "svfa":
            # SVFA mode: unpack tuples
            dec_struct, dec_val = X
            enc_struct, enc_val = enc_out
            
            # === Self-attention ===
            # Normalize decoder embeddings
            dec_struct_norm = self.norm1_struct(dec_struct, not_self_mask_miss_q)
            dec_val_norm = self.norm1(dec_val, not_self_mask_miss_q)
            
            # Self-attention: Q, K from dec_struct; V from dec_val
            self_attn_out, self_att, self_ent = self.global_self_attention(
                query=dec_struct_norm,      # Q: decoder structure
                key=dec_struct_norm,        # K: decoder structure
                value=dec_val_norm,         # V: decoder value
                mask_miss_k=self_mask_miss_k,
                mask_miss_q=self_mask_miss_q,
                pos=dec_input_pos,
                causal_mask=causal_mask
            )
            
            # Residual on decoder VALUE only
            dec_val = dec_val + self.dropout_attn_out(self_attn_out)
            
            # === Cross-attention ===
            # Normalize for cross-attention
            dec_struct_norm = self.norm2_struct(dec_struct, not_self_mask_miss_q)
            dec_val_norm = self.norm2(dec_val, not_self_mask_miss_q)
            
            # Cross-attention: Q from dec_struct; K from enc_struct; V from enc_val
            cross_attn_out, cross_att, cross_ent = self.global_cross_attention(
                query=dec_struct_norm,      # Q: decoder structure
                key=enc_struct,             # K: encoder structure
                value=enc_val,              # V: encoder value
                mask_miss_k=cross_mask_miss_k,
                mask_miss_q=cross_mask_miss_q,
                pos=None,
                causal_mask=False
            )
            
            # Residual on decoder VALUE only
            dec_val = dec_val + self.dropout_attn_out(cross_attn_out)
            
            # === Feed-forward ===
            dec_val_norm = self.norm3(dec_val, not_self_mask_miss_q)
            dec_val_ff = self.dropout_ff(self.activation(self.linear1(dec_val_norm)))
            dec_val_ff = self.dropout_ff(self.linear2(dec_val_ff))
            
            # Final residual on value embedding
            dec_val = dec_val + dec_val_ff
            
            # Structure embedding passes through unchanged
            return (dec_struct, dec_val), self_att, cross_att, self_ent, cross_ent
        
        else:
            # Standard mode
            X1 = self.norm1(X, not_self_mask_miss_q)
            
            X1, self_att, self_ent = self.global_self_attention(
                query=X1,
                key=X1,
                value=X1,
                mask_miss_k=self_mask_miss_k,
                mask_miss_q=self_mask_miss_q,
                pos=dec_input_pos,
                causal_mask=causal_mask
            )
            
            X2 = X + self.dropout_attn_out(X1)
            
            X3 = self.norm2(X2, not_self_mask_miss_q)
            
            X3, cross_att, cross_ent = self.global_cross_attention(
                query=X3,
                key=enc_out,
                value=enc_out,
                mask_miss_k=cross_mask_miss_k,
                mask_miss_q=cross_mask_miss_q,
                pos=None,
                causal_mask=False
            )
            
            X4 = X2 + self.dropout_attn_out(X3)

            X5 = self.norm3(X4, not_self_mask_miss_q)
            
            # feedforward layers (linear)
            X5 = self.dropout_ff(self.activation(self.linear1(X5)))
            X5 = self.dropout_ff(self.linear2(X5))
            
            # final res connection
            decoder_out = X4 + X5

            return decoder_out, self_att, cross_att, self_ent, cross_ent
    
    
class Decoder(nn.Module):
    """
    Decoder with support for SVFA (Structure-Value Factorized Attention).
    
    In SVFA mode:
        - X is tuple (dec_struct, dec_val) for decoder embeddings
        - enc_out is tuple (enc_struct, enc_val) from encoder
        - Only dec_val is updated; dec_struct passes through unchanged
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
        self, X, 
        enc_out, 
        self_mask_miss_k: torch.Tensor, 
        self_mask_miss_q: torch.Tensor,
        cross_mask_miss_k: torch.Tensor, 
        cross_mask_miss_q: torch.Tensor,
        dec_input_pos: torch.Tensor,
        causal_mask: bool
        ):
        """
        Forward pass through decoder.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (dec_struct, dec_val) each (B, L, d_model)
            enc_out: In standard mode: tensor (B, S, d_model)
                     In SVFA mode: tuple (enc_struct, enc_val) each (B, S, d_model)
                     
        Returns:
            In standard mode: (decoder_out, self_att_list, cross_att_list, self_ent_list, cross_ent_list)
            In SVFA mode: ((dec_struct, dec_val), self_att_list, cross_att_list, self_ent_list, cross_ent_list)
        """
        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        # Apply embedding dropout
        if self.factorization == "svfa":
            dec_struct, dec_val = X
            dec_struct = self.emb_dropout(dec_struct)
            dec_val = self.emb_dropout(dec_val)
            X = (dec_struct, dec_val)
        else:
            X = self.emb_dropout(X)

        self_att_list, cross_att_list = [], []
        self_enc_list, cross_enc_list = [], []
        
        for _, decoder_layer in enumerate(self.layers):
            
            X, self_att, cross_att, self_enc, cross_enc = decoder_layer(
                X=X, 
                enc_out=enc_out, 
                self_mask_miss_k=self_mask_miss_k, 
                self_mask_miss_q=self_mask_miss_q,
                cross_mask_miss_k=cross_mask_miss_k, 
                cross_mask_miss_q=cross_mask_miss_q,
                dec_input_pos=dec_input_pos,
                causal_mask=causal_mask
            )
            
            self_att_list.append(self_att)
            cross_att_list.append(cross_att)
            self_enc_list.append(self_enc)
            cross_enc_list.append(cross_enc)

        # Apply final normalization
        if self.norm_layer is not None:
            if self.factorization == "svfa":
                dec_struct, dec_val = X
                dec_val = self.norm_layer(dec_val, not_mask)
                X = (dec_struct, dec_val)
            else:
                X = self.norm_layer(X, not_mask)

        return X, self_att_list, cross_att_list, self_enc_list, cross_enc_list
