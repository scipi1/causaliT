import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import dirname, abspath
import sys
# sys.path.append(dirname(abspath(__file__)))
from proT.core.modules.extra_layers import Normalization



class EncoderLayer(nn.Module):
    def __init__(
        self,
        global_attention,
        d_model_enc,  
        activation,         
        norm, 
        d_ff,                 
        dropout_ff,            
        dropout_attn_out
        ):
        super().__init__()
        
        # global attention is initialized in the `model.py` module
        self.global_attention = global_attention
        
        # normalization
        self.norm1 = Normalization(method=norm, d_model=d_model_enc)
        self.norm2 = Normalization(method=norm, d_model=d_model_enc)
        
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
        X: torch.Tensor, 
        mask_miss_k: torch.Tensor, 
        mask_miss_q: torch.Tensor, 
        enc_input_pos: torch.Tensor,
        causal_mask: bool):
        
        # uses pre-norm Transformer architecture
        
        X1 = self.norm1(X, ~mask_miss_q)
        
        
        # self-attention queries=keys=values=X
        X1, attn, ent = self.global_attention(
            query=X1,
            key=X1,
            value=X1,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos = enc_input_pos,
            causal_mask=causal_mask,
            )                    
        
        # resnet
        X = X + self.dropout_attn_out(X1)
        
        X1 = self.norm2(X, ~mask_miss_q)
        
        
        # feedforward layers (done here as 1x1 convs)
        # X1 = self.dropout_ff(self.activation(self.conv1(X1.transpose(-1, 1))))
        # X1 = self.dropout_ff(self.conv2(X1).transpose(-1, 1))
        
        
        # feedforward layers (linear)
        X1 = self.dropout_ff(self.activation(self.linear1(X1)))
        X1 = self.dropout_ff(self.linear2(X1))
        
        # final res connection
        encoder_out = X + X1
        
        return encoder_out, attn, ent
    
    
class Encoder(nn.Module):
    def __init__(
        self,
        encoder_layers: int,
        norm_layer: nn.Module,
        emb_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self, 
        X: torch.Tensor, 
        mask_miss_k: torch.Tensor, 
        mask_miss_q: torch.Tensor,
        enc_input_pos: torch.Tensor,
        causal_mask: bool):
        
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
            
        if self.norm_layer is not None:
            X = self.norm_layer(X, ~mask_miss_q)

        return X, attn_list, ent_list
