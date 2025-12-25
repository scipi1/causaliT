from os.path import dirname, abspath
import sys
# ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(ROOT_DIR)
from math import sqrt, log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from proT.core.modules.extra_layers import UniformAttentionMask
from proT.utils.entropy_utils import register_attention_entropy, calculate_attention_entropy
from typing import List



class LieAttention(nn.Module):
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(LieAttention, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        ):
        
        # Handle both single-head (3D) and multi-head (4D) tensors
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # convert to commutator
        scores = scores - scores.transpose(-1,-2)
        
        
        #att = torch.softmax(scale * (scores), dim=-1)
        att = F.gelu(scale * scores)
        # Attention entropy hook - register entropy before dropout
        # if self.register_entropy:
        #     register_attention_entropy(self.layer_name, att)
        
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
            
            
        A = torch.nan_to_num(self.dropout(att))
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        
        return V.contiguous(), A, entropy





class ScaledDotAttention(nn.Module):
    def __init__(self, mask_layer: nn.Module, attention_dropout: float, register_entropy: bool, layer_name: str):
        
        super(ScaledDotAttention, self).__init__()
        
        self.mask_layer = mask_layer
        self.dropout = nn.Dropout(attention_dropout)
        self.register_entropy = register_entropy
        self.layer_name = layer_name
        
        self.entropy_enabled = True
        
        if register_entropy and layer_name is None:
            raise ValueError("If register_entropy is True, layer_name must be provided.")
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        ):
        
        # Handle both single-head (3D) and multi-head (4D) tensors
        is_multihead = query.dim() == 4
        
        if is_multihead:
            B, L, H, E = query.shape
            _, S, _, _ = key.shape
        else:
            B, L, E = query.shape
            _, S, _ = key.shape
            H = 1
        
        scale = 1.0 / sqrt(E)
        
        # Compute attention scores
        if is_multihead:
            scores = torch.einsum("blhe,bshe->bhls", query, key)
        else:
            scores = torch.einsum("ble,bse->bls", query, key)
        
        # Apply causal mask
        if pos is not None and causal_mask:
            M_causal = build_causal_mask(pos, n_heads=H)
            
            scores = scores + M_causal
        
        # Apply missing data masks
        # masking missing value with -inf to force the softmax to zero
        # (reference https://arxiv.org/abs/2407.11540)
        
        if is_multihead:
            # For multi-head: scores shape is (B, H, L, S)
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
            # Expand masks to (B, H, L, S)
            if mask_miss_k is not None:
                mask_miss_k_expanded = mask_miss_k.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, query_size).transpose(-1, -2)
            
            if mask_miss_q is not None:
                mask_miss_q_expanded = mask_miss_q.unsqueeze(1).expand(-1, H, -1, -1).expand(-1, -1, -1, key_size)
        else:
            # For single-head: scores shape is (B, L, S)
            
            key_size = scores.size(-1)  # S
            query_size = scores.size(-2)  # L
            
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
        
        att = torch.relu(torch.softmax(scale * (scores + M_k), dim=-1) + M_q)
        
        # Attention entropy hook - register entropy before dropout
        # if self.register_entropy:
        #     register_attention_entropy(self.layer_name, att)
        
        if self.entropy_enabled:
            entropy = calculate_attention_entropy(att)
        else:
            entropy = None
            
            
        A = torch.nan_to_num(self.dropout(att))
        
        # Compute output values
        if is_multihead:
            V = torch.einsum("bhls,bshd->blhd", A, value)
        else:
            V = torch.einsum("bls,bsd->bld", A, value)
        
        return V.contiguous(), A, entropy


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
    p_flat = p.squeeze(-1)      # shape (B, L)

    p_i = p_flat.unsqueeze(-1)  # shape (B, L, 1)
    p_j = p_flat.unsqueeze(-2)  # shape (B, 1, L)

    # Build the additive mask (same dtype/device as `p`)
    M = torch.zeros_like(p_i.expand(-1, -1, p_flat.size(-1)))
    M.masked_fill_(p_j > p_i, float("-inf"))
    
    # Expand for multi-head if needed
    if n_heads > 1:
        M = M.unsqueeze(1).expand(-1, n_heads, -1, -1)  # (B, H, L, L)
    
    return M


def calculate_attention_entropy(att_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate entropy of attention weights.
    
    Args:
        att_weights: Attention weights tensor
                    - Multi-head: (B, H, L, S) 
                    - Single-head: (B, L, S)
        eps: Small value to avoid log(0)
    
    Returns:
        Entropy tensor:
        - Multi-head: (B, H, L) - entropy for each query position in each head
        - Single-head: (B, L) - entropy for each query position
    """
    # Clamp to avoid log(0)
    att_clamped = torch.clamp(att_weights, min=eps)
    
    # Calculate entropy: -sum(p * log(p)) along the key dimension (last dimension)
    log_att = torch.log(att_clamped)
    entropy = -torch.sum(att_weights * log_att, dim=-1)
    
    # Handle NaN values that might arise from 0 * log(0)
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy


        
        
class AttentionLayer(nn.Module):
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
        layer_name: str = None
        ):
        
        super(AttentionLayer, self).__init__()
        
        self.inner_attention = attention(
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            register_entropy=register_entropy,
            layer_name=layer_name
            )
        self.query_projection = nn.Linear(d_model_queries, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model_keys, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model_keys, d_model_values * n_heads)
        self.out_projection = nn.Linear(d_model_values * n_heads, d_model_queries)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask_miss_k: torch.Tensor,
        mask_miss_q: torch.Tensor,
        pos: torch.Tensor,
        causal_mask: bool,
        ):
        
        B, L, _ = query.shape
        _, S, _ = key.shape
        H = self.n_heads
        
        # Apply projections and reshape for multi-head attention
        if H > 1:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, H, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, H, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, H, -1)
        else:
            query = self.dropout_qkv(self.query_projection(query)).view(B, L, -1)
            key = self.dropout_qkv(self.key_projection(key)).view(B, S, -1)
            value = self.dropout_qkv(self.value_projection(value)).view(B, S, -1)
            
        out, attn, ent = self.inner_attention(
            query=query,
            key=key,
            value=value,
            mask_miss_k=mask_miss_k,
            mask_miss_q=mask_miss_q,
            pos=pos,
            causal_mask=causal_mask,
            )
        
        # Reshape output and apply final projection if multi-head
        if H > 1:
            # out shape is (B, L, H, d_v) -> reshape to (B, L, H*d_v)
            out = out.view(B, L, -1)
            out = self.out_projection(out)
        else:
            # out shape is already (B, L, d_v)
            out = out.view(B, L, -1)
        
        return out, attn, ent
    
    
    
def main():
    """Quick test for both single-head and multi-head attention

    Returns:
        _tuple(torch.Tensor): attention output and score
    """
    
    bs = 1
    seq_len = 5
    d_model = 12
    d_queries_keys = 8
    mask = [False, True, True, False, False]
    x = torch.ones(bs,seq_len,d_model)
    x[0,0,0] = torch.nan
    
    print("Testing single-head attention (n_heads=1):")
    attention_single = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
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
    
    print(f"Single-head - Output shape: {out_single.shape}, Score shape: {score_single.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    print("\nTesting multi-head attention (n_heads=4):")
    attention_multi = AttentionLayer(
        attention=ScaledDotAttention, 
        d_model_queries = d_model,
        d_model_keys= d_model,
        d_model_values= d_model,
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
    
    print(f"Multi-head - Output shape: {out_multi.shape}, Score shape: {score_multi.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    
    # Test with causal mask and position
    print("\nTesting with causal mask:")
    pos = torch.arange(seq_len).unsqueeze(0).unsqueeze(-1).float()  # (1, 5, 1)
    
    out_causal, score_causal, ent = attention_multi.forward(
        query=x, 
        key=x, 
        value=x,
        mask_miss_k=None,
        mask_miss_q=None,
        pos=pos,
        causal_mask=True
        )
    
    print(f"Causal multi-head - Output shape: {out_causal.shape}, Score shape: {score_causal.shape}, Entropy shape: {ent.shape if ent is not None else 'None'}")
    print("All tests completed successfully!")
    
    
if __name__ == "__main__":
    main()
