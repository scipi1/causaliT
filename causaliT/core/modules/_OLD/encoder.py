import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath
import sys
# sys.path.append(dirname(abspath(__file__)))
from causaliT.core.modules.extra_layers import Normalization



class EncoderLayer(nn.Module):
    """
    Encoder layer with support for SVFA (Structure-Value Factorized Attention).
    
    In standard mode:
        - X is a single tensor, used for Q, K, V and residual connections
        
    In SVFA mode:
        - X is a tuple (emb_struct, emb_val)
        - emb_struct: Passed as query and key to attention (structural alignment)
        - emb_val: Passed as value to attention and used for residual connections
        - Only emb_val is updated; emb_struct passes through unchanged
    """
    def __init__(
        self,
        global_attention,
        d_model_enc,  
        activation,         
        norm, 
        d_ff,                 
        dropout_ff,            
        dropout_attn_out,
        factorization: str = "standard"
        ):
        super().__init__()
        
        # global attention is initialized in the `model.py` module
        self.global_attention = global_attention
        self.factorization = factorization
        
        # normalization
        self.norm1 = Normalization(method=norm, d_model=d_model_enc)
        self.norm2 = Normalization(method=norm, d_model=d_model_enc)
        
        # For SVFA: separate normalization for structure embeddings
        if factorization == "svfa":
            self.norm1_struct = Normalization(method=norm, d_model=d_model_enc)
        
        # output convolutions or linear
        self.conv1 = nn.Conv1d(in_channels=d_model_enc, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model_enc, kernel_size=1)
        self.linear1 = nn.Linear(in_features=d_model_enc, out_features=d_ff, bias=True)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model_enc, bias=True)
        
        # dropouts and activation
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        #self.time_windows = time_windows                #??
        #self.time_window_offset = time_window_offset    #??
        #self.d_yc = d_yc                                # for local attention

    def forward(
        self, 
        X, 
        mask_miss_k: torch.Tensor, 
        mask_miss_q: torch.Tensor, 
        enc_input_pos: torch.Tensor,
        causal_mask: bool):
        """
        Forward pass through encoder layer.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (emb_struct, emb_val) each (B, L, d_model)
            mask_miss_k: Missing key mask
            mask_miss_q: Missing query mask  
            enc_input_pos: Positional encoding
            causal_mask: Whether to apply causal masking
            
        Returns:
            In standard mode: (encoder_out, attn, ent)
            In SVFA mode: ((emb_struct, updated_emb_val), attn, ent)
        """
        not_mask = ~mask_miss_q if mask_miss_q is not None else None
        
        if self.factorization == "svfa":
            # SVFA mode: X is tuple (emb_struct, emb_val)
            emb_struct, emb_val = X
            
            # Normalize both embeddings
            emb_struct_norm = self.norm1_struct(emb_struct, not_mask)
            emb_val_norm = self.norm1(emb_val, not_mask)
            
            # Self-attention with SVFA factorization:
            # Q, K from structure embedding; V from value embedding
            attn_out, attn, ent = self.global_attention(
                query=emb_struct_norm,      # Q: structure
                key=emb_struct_norm,        # K: structure  
                value=emb_val_norm,         # V: value
                mask_miss_k=mask_miss_k,
                mask_miss_q=mask_miss_q,
                pos=enc_input_pos,
                causal_mask=causal_mask,
            )
            
            # Residual on VALUE embedding only (sum apples with apples)
            emb_val = emb_val + self.dropout_attn_out(attn_out)
            
            # Feed-forward on value embedding only
            emb_val_norm = self.norm2(emb_val, not_mask)
            emb_val_ff = self.dropout_ff(self.activation(self.linear1(emb_val_norm)))
            emb_val_ff = self.dropout_ff(self.linear2(emb_val_ff))
            
            # Final residual on value embedding
            emb_val = emb_val + emb_val_ff
            
            # Structure embedding passes through unchanged
            return (emb_struct, emb_val), attn, ent
        
        else:
            # Standard mode: uses pre-norm Transformer architecture
            X1 = self.norm1(X, not_mask)
            
            # self-attention queries=keys=values=X
            X1, attn, ent = self.global_attention(
                query=X1,
                key=X1,
                value=X1,
                mask_miss_k=mask_miss_k,
                mask_miss_q=mask_miss_q,
                pos=enc_input_pos,
                causal_mask=causal_mask,
            )                    
            
            # resnet
            X = X + self.dropout_attn_out(X1)
            
            X1 = self.norm2(X, not_mask)
            
            # feedforward layers (linear)
            X1 = self.dropout_ff(self.activation(self.linear1(X1)))
            X1 = self.dropout_ff(self.linear2(X1))
            
            # final res connection
            encoder_out = X + X1
            
            return encoder_out, attn, ent
    
    
class Encoder(nn.Module):
    """
    Encoder with support for SVFA (Structure-Value Factorized Attention).
    
    In SVFA mode, X is a tuple (emb_struct, emb_val) that passes through all layers.
    The structure embedding is unchanged; only the value embedding is updated.
    """
    def __init__(
        self,
        encoder_layers: list,
        norm_layer: nn.Module,
        emb_dropout: float,
        factorization: str = "standard"
    ):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.factorization = factorization

    def forward(
        self, 
        X, 
        mask_miss_k: torch.Tensor, 
        mask_miss_q: torch.Tensor,
        enc_input_pos: torch.Tensor,
        causal_mask: bool):
        """
        Forward pass through encoder.
        
        Args:
            X: In standard mode: tensor (B, L, d_model)
               In SVFA mode: tuple (emb_struct, emb_val) each (B, L, d_model)
               
        Returns:
            In standard mode: (encoder_out, attn_list, ent_list)
            In SVFA mode: ((emb_struct, emb_val), attn_list, ent_list)
        """
        not_mask = ~mask_miss_q if mask_miss_q is not None else None
        
        # Apply embedding dropout
        if self.factorization == "svfa":
            emb_struct, emb_val = X
            emb_struct = self.emb_dropout(emb_struct)
            emb_val = self.emb_dropout(emb_val)
            X = (emb_struct, emb_val)
        else:
            X = self.emb_dropout(X)

        attn_list, ent_list = [], []
        
        for _, encoder_layer in enumerate(self.layers):
            X, attn, ent = encoder_layer(
                X=X, 
                mask_miss_k=mask_miss_k, 
                mask_miss_q=mask_miss_q, 
                enc_input_pos=enc_input_pos,
                causal_mask=causal_mask) 
            
            attn_list.append(attn)
            ent_list.append(ent)
            
        # Apply final normalization
        if self.norm_layer is not None:
            if self.factorization == "svfa":
                emb_struct, emb_val = X
                emb_val = self.norm_layer(emb_val, not_mask)
                X = (emb_struct, emb_val)
            else:
                X = self.norm_layer(X, not_mask)

        return X, attn_list, ent_list
