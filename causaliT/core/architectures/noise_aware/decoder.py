"""
Noise-Aware Reversed Decoder implementation.

This module extends the ReversedDecoder to inject ambient noise between
cross-attention and self-attention, implementing the noise-aware causal mixing.

Architecture (per layer):
1. Cross-attention: H_det = CrossAtt(X_struct, S, S)  [deterministic]
2. Ambient noise:   H = H_det + σ_A * ε              [noise injection]
3. Self-attention:  U = SelfAtt(X_struct, X_struct, H) [noisy value mixing]
4. Feedforward:     output = FF(U)

Key difference from ReversedDecoderLayer:
- Noise is injected AFTER cross-attention, BEFORE self-attention
- Self-attention uses noisy H as values (not X_val)
- This ensures causal mixing operates on noisy physical states

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules.extra_layers import Normalization
from causaliT.core.modules.noise_layers import AmbientNoiseLayer


class NoiseAwareReversedDecoderLayer(nn.Module):
    """
    Decoder layer with noise injection for SVFA mode.
    
    This layer implements the noise-aware causal mechanism:
    1. Cross-attention produces deterministic hidden state H_det
    2. Ambient noise is injected: H = H_det + σ_A * ε
    3. Self-attention mixes the noisy physical states
    4. Feedforward processes the mixed representation
    
    The key insight is that noise is injected BEFORE self-attention mixing,
    so downstream nodes receive and propagate upstream uncertainty through
    the learned attention weights.
    
    Architecture (SVFA mode only):
    -----------------------------
    Input: X = (X_struct, X_val), external_context = S
    
    1. Cross-attention:
       - Q from X_struct, K/V from S
       - H_det = X_val + dropout(CrossAtt(...))  [deterministic hidden state]
    
    2. Ambient noise injection:
       - H = AmbientNoise(H_det)  [H = H_det + σ_A * ε during training]
    
    3. Self-attention (on noisy states):
       - Q, K from X_struct (structure determines mixing)
       - V from H (noisy physical states are mixed)
       - U = H + dropout(SelfAtt(...))  [aggregated noisy states]
    
    4. Feedforward:
       - output = U + dropout(FF(U))
    
    Args:
        global_cross_attention: Cross-attention module
        global_self_attention: Self-attention module
        ambient_noise_layer: AmbientNoiseLayer for noise injection
        d_model_dec: Model dimension
        d_ff: Feedforward dimension
        dropout_ff: Feedforward dropout rate
        dropout_attn_out: Attention output dropout rate
        activation: Activation function ('relu' or 'gelu')
        norm: Normalization method
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
    ):
        super().__init__()
        
        # Attention modules
        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention
        
        # Ambient noise layer (shared reference from parent model)
        self.ambient_noise = ambient_noise_layer
        
        # Normalization layers for value path
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)
        
        # Separate normalization for structure embeddings (SVFA)
        self.norm1_struct = Normalization(method=norm, d_model=d_model_dec)
        self.norm2_struct = Normalization(method=norm, d_model=d_model_dec)
        
        # Feedforward layers
        self.linear1 = nn.Linear(d_model_dec, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model_dec, bias=True)
        
        # Dropouts and activation
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(
        self,
        X: Tuple[torch.Tensor, torch.Tensor],
        external_context: torch.Tensor,
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
        """
        Forward pass with noise injection between cross and self attention.
        
        Args:
            X: Tuple (X_struct, X_val) - SVFA embeddings
               - X_struct: (B, L, d_model) structure embedding for Q, K
               - X_val: (B, L, d_model) value embedding
            external_context: (B, S, d_model) external context (S embedding)
            self_mask_miss_k: Missing value mask for self-attention keys
            self_mask_miss_q: Missing value mask for self-attention queries
            cross_mask_miss_k: Missing value mask for cross-attention keys
            cross_mask_miss_q: Missing value mask for cross-attention queries
            dec_input_pos: Positional information for decoder input
            causal_mask: Whether to apply causal masking
            cross_hard_mask: Optional hard mask for cross-attention
            self_hard_mask: Optional hard mask for self-attention
            inject_noise: If True, inject ambient noise. Default True.
            
        Returns:
            (X_struct, X_val): Updated SVFA embeddings (X_struct unchanged)
            cross_att: Cross-attention weights
            self_att: Self-attention weights
            cross_ent: Cross-attention entropy
            self_ent: Self-attention entropy
        """
        # Compute negated masks for normalization
        not_cross_mask_miss_q = ~cross_mask_miss_q if cross_mask_miss_q is not None else None
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        # Unpack SVFA tuple
        X_struct, X_val = X
        
        # Handle external_context - can be single tensor or tuple
        if isinstance(external_context, tuple):
            ext_struct, ext_val = external_context
        else:
            # External context is single tensor (e.g., OrthogonalMaskEmbedding)
            # Use same tensor for both K and V
            ext_struct = external_context
            ext_val = external_context
        
        # =====================================================================
        # STEP 1: Cross-attention (FIRST) - produces H_det
        # Q from X_struct, K from ext_struct, V from ext_val
        # =====================================================================
        
        X_struct_norm = self.norm1_struct(X_struct, not_cross_mask_miss_q)
        X_val_norm = self.norm1(X_val, not_cross_mask_miss_q)
        
        cross_attn_out, cross_att, cross_ent = self.global_cross_attention(
            query=X_struct_norm,     # Q: structure embedding
            key=ext_struct,          # K: external structure (S)
            value=ext_val,           # V: external value (S)
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos=None,
            causal_mask=False,
            hard_mask=cross_hard_mask,
        )
        
        # Residual connection on VALUE path → H_det (deterministic hidden state)
        H_det = X_val + self.dropout_attn_out(cross_attn_out)
        
        # =====================================================================
        # STEP 2: Ambient noise injection → H = H_det + σ_A * ε
        # This is the KEY difference from ReversedDecoderLayer
        # =====================================================================
        
        H = self.ambient_noise(H_det, inject_noise=inject_noise)
        
        # =====================================================================
        # STEP 3: Self-attention (SECOND) - mixes noisy physical states
        # Q, K from X_struct (structure determines mixing)
        # V from H (noisy physical states are what gets mixed!)
        # =====================================================================
        
        X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
        H_norm = self.norm2(H, not_self_mask_miss_q)
        
        self_attn_out, self_att, self_ent = self.global_self_attention(
            query=X_struct_norm,     # Q: structure embedding
            key=X_struct_norm,       # K: structure embedding
            value=H_norm,            # V: NOISY hidden state (not X_val!)
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
        )
        
        # Residual connection → U (aggregated noisy states)
        U = H + self.dropout_attn_out(self_attn_out)
        
        # =====================================================================
        # STEP 4: Feedforward
        # =====================================================================
        
        U_norm = self.norm3(U, not_self_mask_miss_q)
        U_ff = self.dropout_ff(self.activation(self.linear1(U_norm)))
        U_ff = self.dropout_ff(self.linear2(U_ff))
        
        # Final residual on value path
        X_val_out = U + U_ff
        
        # Structure passes through unchanged
        return (X_struct, X_val_out), cross_att, self_att, cross_ent, self_ent


class NoiseAwareReversedDecoderLayerV2(nn.Module):
    """
    Alternative decoder layer with noise injection in value projection.
    
    This variant injects noise INSIDE the self-attention value projection:
    - V_noisy = W_v(H_det) + σ_A * ε
    - Then self-attention aggregates: U = α @ V_noisy
    
    This is an alternative to NoiseAwareReversedDecoderLayer where noise
    is injected BEFORE value projection.
    
    Design Choice: This version adds noise AFTER the value projection,
    which may be useful if the noise should be in the projected space
    rather than the embedding space.
    
    Note: This is kept as an alternative implementation for experimentation.
    The default NoiseAwareReversedDecoderLayer is recommended.
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
    ):
        super().__init__()
        
        # Same structure as NoiseAwareReversedDecoderLayer
        self.global_cross_attention = global_cross_attention
        self.global_self_attention = global_self_attention
        self.ambient_noise = ambient_noise_layer
        
        self.norm1 = Normalization(method=norm, d_model=d_model_dec)
        self.norm2 = Normalization(method=norm, d_model=d_model_dec)
        self.norm3 = Normalization(method=norm, d_model=d_model_dec)
        self.norm1_struct = Normalization(method=norm, d_model=d_model_dec)
        self.norm2_struct = Normalization(method=norm, d_model=d_model_dec)
        
        self.linear1 = nn.Linear(d_model_dec, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, d_model_dec, bias=True)
        
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Flag to identify this variant
        self.noise_in_projection = True
    
    def forward(
        self,
        X: Tuple[torch.Tensor, torch.Tensor],
        external_context: torch.Tensor,
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
        """
        Forward pass with noise injection after value projection.
        
        The difference from NoiseAwareReversedDecoderLayer:
        - Noise is added AFTER normalization, effectively in the "value" input
          to self-attention rather than before normalization.
        
        This subtle difference may affect how noise interacts with layer norm.
        """
        not_cross_mask_miss_q = ~cross_mask_miss_q if cross_mask_miss_q is not None else None
        not_self_mask_miss_q = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        X_struct, X_val = X
        
        if isinstance(external_context, tuple):
            ext_struct, ext_val = external_context
        else:
            ext_struct = external_context
            ext_val = external_context
        
        # Step 1: Cross-attention
        X_struct_norm = self.norm1_struct(X_struct, not_cross_mask_miss_q)
        X_val_norm = self.norm1(X_val, not_cross_mask_miss_q)
        
        cross_attn_out, cross_att, cross_ent = self.global_cross_attention(
            query=X_struct_norm,
            key=ext_struct,
            value=ext_val,
            mask_miss_k=cross_mask_miss_k,
            mask_miss_q=cross_mask_miss_q,
            pos=None,
            causal_mask=False,
            hard_mask=cross_hard_mask,
        )
        
        H_det = X_val + self.dropout_attn_out(cross_attn_out)
        
        # Step 2 & 3: Normalize, then add noise, then self-attention
        X_struct_norm = self.norm2_struct(X_struct, not_self_mask_miss_q)
        H_det_norm = self.norm2(H_det, not_self_mask_miss_q)
        
        # Inject noise AFTER normalization (in projected space)
        H_norm = self.ambient_noise(H_det_norm, inject_noise=inject_noise)
        
        self_attn_out, self_att, self_ent = self.global_self_attention(
            query=X_struct_norm,
            key=X_struct_norm,
            value=H_norm,  # Noisy normalized value
            mask_miss_k=self_mask_miss_k,
            mask_miss_q=self_mask_miss_q,
            pos=dec_input_pos,
            causal_mask=causal_mask,
            hard_mask=self_hard_mask,
        )
        
        U = H_det + self.dropout_attn_out(self_attn_out)
        
        # Step 4: Feedforward
        U_norm = self.norm3(U, not_self_mask_miss_q)
        U_ff = self.dropout_ff(self.activation(self.linear1(U_norm)))
        U_ff = self.dropout_ff(self.linear2(U_ff))
        
        X_val_out = U + U_ff
        
        return (X_struct, X_val_out), cross_att, self_att, cross_ent, self_ent
