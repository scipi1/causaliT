"""
StageCausaliT: Multi-stage causal transformer with dual reversed decoders.

Architecture:
- Stage 1 (Decoder 1): S → X reconstruction
- Stage 2 (Decoder 2): X → Y prediction

Key features:
- Reversed attention order (cross → self → FF)
- Shared embedding/de-embedding across all variables
- Teacher forcing support
- Cascaded loss computation
"""

import sys
import warnings
from os.path import dirname, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules import (
    LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer,
    ModularEmbedding,
    Normalization, UniformAttentionMask
)
from causaliT.core.architectures.stage_causal.decoder import (
    ReversedDecoder, ReversedDecoderLayer
)


class StageCausaliT(nn.Module):
    """
    StageCausaliT: Multi-stage causal transformer architecture.
    
    This model processes data through two cascaded decoder stages:
    1. Stage 1: Source (S) → Intermediate (X) reconstruction
    2. Stage 2: Intermediate (X) → Target (Y) prediction
    
    Each decoder uses reversed attention order (cross-attention before self-attention).
    All variables share the same embedding and de-embedding transformations for consistency.
    
    Required data shapes: (BATCH_SIZE, sequence_length, variables)
    """
    def __init__(
        self,
        model: str,
        
        # Shared embedding configuration
        ds_embed_shared,
        comps_embed_shared,
        
        # Attention configuration for both decoders
        dec1_cross_attention_type,
        dec1_cross_mask_type,
        dec1_self_attention_type,
        dec1_self_mask_type,
        dec2_cross_attention_type,
        dec2_cross_mask_type,
        dec2_self_attention_type,
        dec2_self_mask_type,
        n_heads: int,
        
        # Causal masking
        dec1_causal_mask: bool,
        dec2_causal_mask: bool,
        
        # Dropout rates
        dropout_emb: float,
        dropout_attn_out: float,
        dropout_ff: float,
        dec1_cross_dropout_qkv: float,
        dec1_cross_attention_dropout: float,
        dec1_self_dropout_qkv: float,
        dec1_self_attention_dropout: float,
        dec2_cross_dropout_qkv: float,
        dec2_cross_attention_dropout: float,
        dec2_self_dropout_qkv: float,
        dec2_self_attention_dropout: float,
        
        # Model architecture
        d1_layers: int,  # Number of layers in decoder 1
        d2_layers: int,  # Number of layers in decoder 2
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        
        # Model dimensions
        out_dim: int,
        d_ff: int,
        d_model: int,  # Shared model dimension
        d_qk: int,
        
        # Sequence lengths for attention initialization
        S_seq_len: int = None,
        X_seq_len: int = None,
        Y_seq_len: int = None,
    ):
        super().__init__()
        
        # Store configuration
        self.model_name = model
        self.dec1_causal_mask = dec1_causal_mask
        self.dec2_causal_mask = dec2_causal_mask
        
        # Shared embedding system for all variables (S, X, Y)
        self.shared_embedding = ModularEmbedding(
            ds_embed=ds_embed_shared,
            comps=comps_embed_shared,
            device=device
        )
        
        # Shared attention configuration
        attn_shared_kwargs = {
            "n_heads": n_heads,
            "d_queries_keys": d_qk,
        }
        
        # Decoder 1 attention configurations
        attn_dec1_cross_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec1_cross_attention_type,
            "mask_type": dec1_cross_mask_type,
            "dropout_qkv": dec1_cross_dropout_qkv,
            "attention_dropout": dec1_cross_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec1_cross_att",
            "query_seq_len": X_seq_len,
            "key_seq_len": S_seq_len
        }
        
        attn_dec1_self_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec1_self_attention_type,
            "mask_type": dec1_self_mask_type,
            "dropout_qkv": dec1_self_dropout_qkv,
            "attention_dropout": dec1_self_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec1_self_att",
            "query_seq_len": X_seq_len,
            "key_seq_len": X_seq_len
        }
        
        # Decoder 2 attention configurations
        attn_dec2_cross_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec2_cross_attention_type,
            "mask_type": dec2_cross_mask_type,
            "dropout_qkv": dec2_cross_dropout_qkv,
            "attention_dropout": dec2_cross_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec2_cross_att",
            "query_seq_len": Y_seq_len,
            "key_seq_len": X_seq_len
        }
        
        attn_dec2_self_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec2_self_attention_type,
            "mask_type": dec2_self_mask_type,
            "dropout_qkv": dec2_self_dropout_qkv,
            "attention_dropout": dec2_self_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec2_self_att",
            "query_seq_len": Y_seq_len,
            "key_seq_len": Y_seq_len
        }
        
        # Build Decoder 1 (S → X)
        self.decoder1 = ReversedDecoder(
            decoder_layers=[
                ReversedDecoderLayer(
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec1_cross_kwargs)),
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec1_self_kwargs)),
                    d_model_dec=d_model,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                ) for _ in range(d1_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb
        )
        
        # Build Decoder 2 (X → Y)
        self.decoder2 = ReversedDecoder(
            decoder_layers=[
                ReversedDecoderLayer(
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec2_cross_kwargs)),
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec2_self_kwargs)),
                    d_model_dec=d_model,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                ) for _ in range(d2_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb
        )
        
        # Shared de-embedding head (forecaster)
        self.forecaster = nn.Linear(d_model, out_dim, bias=False)
    
    def forward(
        self,
        source_tensor,
        intermediate_tensor_blanked,
        intermediate_tensor_actual,
        target_tensor,
        use_teacher_forcing: bool = False,
        hard_masks: dict = None,
    ):
        """
        Forward pass through both decoder stages.
        
        Args:
            source_tensor: Source nodes (S), shape (B, S_seq_len, features)
            intermediate_tensor_blanked: Intermediate variables (X) with values blanked, 
                                         shape (B, X_seq_len, features). Always used for decoder 1.
            intermediate_tensor_actual: Intermediate variables (X) with actual values,
                                        shape (B, X_seq_len, features). Used for decoder 2 with teacher forcing.
            target_tensor: Target variables (Y, blanked), shape (B, Y_seq_len, features)
            use_teacher_forcing: If True, use actual X for decoder 2 instead of predicted X
            hard_masks: Optional dict of hard masks for attention. Keys:
                        - 'dec1_cross': mask for decoder 1 cross-attention (X_len, S_len)
                        - 'dec1_self': mask for decoder 1 self-attention (X_len, X_len)
                        - 'dec2_cross': mask for decoder 2 cross-attention (Y_len, X_len)
                        - 'dec2_self': mask for decoder 2 self-attention (Y_len, Y_len)
                        Values in [0, 1], where 1 = attention allowed.
            
        Returns:
            pred_x: Predicted X from decoder 1
            pred_y: Predicted Y from decoder 2
            attention_weights: Tuple of attention weights from both decoders
            masks: Tuple of masks for S, X, Y
            entropies: Tuple of attention entropies from both decoders
        """
        
        # Extract hard masks if provided
        dec1_cross_hard = None
        dec1_self_hard = None
        dec2_cross_hard = None
        dec2_self_hard = None
        
        if hard_masks is not None:
            dec1_cross_hard = hard_masks.get('dec1_cross', None)
            dec1_self_hard = hard_masks.get('dec1_self', None)
            dec2_cross_hard = hard_masks.get('dec2_cross', None)
            dec2_self_hard = hard_masks.get('dec2_self', None)
        
        # ===== Stage 1: Source → Intermediate (S → X) =====
        
        # Embed source nodes
        s_embedded = self.shared_embedding(X=source_tensor)
        s_input_pos = self.shared_embedding.pass_var(X=source_tensor)
        s_mask = self.shared_embedding.get_mask(X=source_tensor)
        
        # Embed intermediate (always use blanked X for decoder 1)
        x_embedded = self.shared_embedding(X=intermediate_tensor_blanked)
        x_input_pos = self.shared_embedding.pass_var(X=intermediate_tensor_blanked)
        x_mask = self.shared_embedding.get_mask(X=intermediate_tensor_blanked)
        
        # Pass through Decoder 1: X queries, S keys/values
        dec1_out, dec1_cross_att, dec1_self_att, dec1_cross_ent, dec1_self_ent = self.decoder1(
            X=x_embedded,
            external_context=s_embedded,
            self_mask_miss_k=x_mask,
            self_mask_miss_q=x_mask,
            cross_mask_miss_k=s_mask,
            cross_mask_miss_q=x_mask,
            dec_input_pos=x_input_pos,
            causal_mask=self.dec1_causal_mask,
            cross_hard_mask=dec1_cross_hard,
            self_hard_mask=dec1_self_hard,
        )
        
        # De-embed to get predicted X
        pred_x = self.forecaster(dec1_out)
        
        # ===== Stage 2: Intermediate → Target (X → Y) =====
        
        # Teacher forcing logic: decide whether to use actual or predicted X
        if use_teacher_forcing:
            # Use actual X values (ground truth) for decoder 2
            x_for_dec2 = intermediate_tensor_actual
        else:
            # Use predicted X from decoder 1 for decoder 2
            # Start from blanked tensor and fill in predicted values
            x_for_dec2 = intermediate_tensor_blanked.clone()
            x_for_dec2[:, :, 0] = pred_x.squeeze(-1)
        
        # Re-embed X for decoder 2 (using shared embedding)
        x_for_dec2_embedded = self.shared_embedding(X=x_for_dec2)
        x_for_dec2_pos = self.shared_embedding.pass_var(X=x_for_dec2)
        x_for_dec2_mask = self.shared_embedding.get_mask(X=x_for_dec2)
        
        # Embed target (blanked Y)
        y_embedded = self.shared_embedding(X=target_tensor)
        y_input_pos = self.shared_embedding.pass_var(X=target_tensor)
        y_mask = self.shared_embedding.get_mask(X=target_tensor)
        
        # Pass through Decoder 2: Y queries, X keys/values
        dec2_out, dec2_cross_att, dec2_self_att, dec2_cross_ent, dec2_self_ent = self.decoder2(
            X=y_embedded,
            external_context=x_for_dec2_embedded,
            self_mask_miss_k=y_mask,
            self_mask_miss_q=y_mask,
            cross_mask_miss_k=x_for_dec2_mask,
            cross_mask_miss_q=y_mask,
            dec_input_pos=y_input_pos,
            causal_mask=self.dec2_causal_mask,
            cross_hard_mask=dec2_cross_hard,
            self_hard_mask=dec2_self_hard,
        )
        
        # De-embed to get predicted Y
        pred_y = self.forecaster(dec2_out)
        
        # Collect outputs
        attention_weights = (dec1_cross_att, dec1_self_att, dec2_cross_att, dec2_self_att)
        masks = (s_mask, x_mask, y_mask)
        entropies = (dec1_cross_ent, dec1_self_ent, dec2_cross_ent, dec2_self_ent)
        
        return pred_x, pred_y, attention_weights, masks, entropies
    
    def _attn(
        self,
        d_model_queries: int,
        d_model_keys: int,
        d_model_values: int,
        n_heads: int,
        d_queries_keys: int,
        attention_type: str,
        mask_type: str,
        dropout_qkv: float,
        attention_dropout: float,
        register_entropy: bool,
        layer_name: str,
        query_seq_len: int = None,
        key_seq_len: int = None
    ):
        """
        Create an attention layer with specified configuration.
        
        Args:
            d_model_queries: Dimension of query projections
            d_model_keys: Dimension of key projections
            d_model_values: Dimension of value projections
            n_heads: Number of attention heads
            d_queries_keys: Dimension of queries and keys per head
            attention_type: Type of attention mechanism ("ScaledDotProduct" or "LieAttention")
            mask_type: Type of masking ("Uniform" or None)
            dropout_qkv: Dropout rate for query/key/value projections
            attention_dropout: Dropout rate for attention weights
            register_entropy: Whether to register attention entropy
            layer_name: Name for entropy registration
            query_seq_len: Query sequence length (for Lie attention initialization)
            key_seq_len: Key sequence length (for Lie attention initialization)
            
        Returns:
            AttentionLayer: Configured attention layer
        """
        
        # Choose attention type
        assert attention_type in ["ScaledDotProduct", "LieAttention", "CausalCrossAttention", "PhiSoftMax"]
        
        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention
        elif attention_type == "LieAttention":
            attention_module = LieAttention
        elif attention_type == "CausalCrossAttention":
            attention_module = CausalCrossAttention
        elif attention_type == "PhiSoftMax":
            attention_module = PhiSoftMax
        
        # Choose mask type
        mask_layer = None
        if mask_type is not None:
            if mask_type == "Uniform":
                mask_layer = UniformAttentionMask()
        
        att = AttentionLayer(
            attention=attention_module,
            d_model_queries=d_model_queries,
            d_model_keys=d_model_keys,
            d_model_values=d_model_values,
            d_queries_keys=d_queries_keys,
            n_heads=n_heads,
            mask_layer=mask_layer,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv,
            register_entropy=register_entropy,
            layer_name=layer_name,
            query_seq_len=query_seq_len,
            key_seq_len=key_seq_len
        )
        
        return att
