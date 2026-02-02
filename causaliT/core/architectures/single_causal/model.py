"""
SingleCausalLayer: Simplified single-decoder model for S → X causal learning.

Architecture:
- Single Decoder: S → X reconstruction using reversed attention order

Key features:
- Reversed attention order (cross → self → FF)
- Orthogonal embedding for S (frozen) - ensures independent attention scores
- Standard learnable embedding for X
- Single stage for focused causal structure learning

Embedding design:
- S uses OrthogonalMaskEmbedding: value_embed(v) ⊙ binary_mask
- This ensures ⟨emb(Sᵢ), emb(Sⱼ)⟩ = 0 for i ≠ j
- X uses standard ModularEmbedding (learnable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules import (
    LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer,
    ModularEmbedding, OrthogonalMaskEmbedding,
    Normalization, UniformAttentionMask
)
from causaliT.core.architectures.stage_causal.decoder import (
    ReversedDecoder, ReversedDecoderLayer
)


class SingleCausalLayer(nn.Module):
    """
    SingleCausalLayer: Simplified single-decoder causal transformer.
    
    This model focuses on learning a single causal relationship: S → X
    It uses the same reversed attention architecture as StageCausaliT's decoder 1,
    but without the second decoder stage.
    
    Embedding:
    - S (source): OrthogonalMaskEmbedding - frozen orthogonal basis
    - X (intermediate): ModularEmbedding - learnable
    
    Required data shapes: (BATCH_SIZE, sequence_length, features)
    """
    def __init__(
        self,
        model: str,
        
        # S embedding configuration (orthogonal)
        ds_embed_S: dict,
        
        # X embedding configuration (standard)
        ds_embed_X: dict,
        comps_embed_X: str,
        
        # Attention configuration for decoder
        dec_cross_attention_type,
        dec_cross_mask_type,
        dec_self_attention_type,
        dec_self_mask_type,
        n_heads: int,
        
        # Causal masking
        dec_causal_mask: bool,
        
        # Dropout rates
        dropout_emb: float,
        dropout_attn_out: float,
        dropout_ff: float,
        dec_cross_dropout_qkv: float,
        dec_cross_attention_dropout: float,
        dec_self_dropout_qkv: float,
        dec_self_attention_dropout: float,
        
        # Model architecture
        dec_layers: int,
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        
        # Model dimensions
        out_dim: int,
        d_ff: int,
        d_model: int,
        d_qk: int,
        
        # Sequence lengths for attention initialization
        S_seq_len: int,
        X_seq_len: int,
    ):
        super().__init__()
        
        # Store configuration
        self.model_name = model
        self.dec_causal_mask = dec_causal_mask
        self.d_model = d_model
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        
        # Orthogonal embedding for S (frozen by default)
        self.embedding_S = OrthogonalMaskEmbedding(
            num_variables=ds_embed_S["num_variables"],
            d_model=d_model,
            value_input_dim=ds_embed_S.get("value_input_dim", 1),
            value_idx=ds_embed_S["value_idx"],
            var_idx=ds_embed_S["var_idx"],
            var_id_offset=ds_embed_S.get("var_id_offset", 1),  # Default 1 (1-indexed var IDs)
            freeze=ds_embed_S.get("freeze", True),
            device=device
        )
        
        # Standard embedding for X (learnable)
        self.embedding_X = ModularEmbedding(
            ds_embed=ds_embed_X,
            comps=comps_embed_X,
            device=device
        )
        
        # =====================================================================
        # ATTENTION CONFIGURATION
        # =====================================================================
        
        attn_shared_kwargs = {
            "n_heads": n_heads,
            "d_queries_keys": d_qk,
        }
        
        # Decoder cross-attention configuration (S → X)
        attn_dec_cross_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec_cross_attention_type,
            "mask_type": dec_cross_mask_type,
            "dropout_qkv": dec_cross_dropout_qkv,
            "attention_dropout": dec_cross_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec_cross_att",
            "query_seq_len": X_seq_len,
            "key_seq_len": S_seq_len
        }
        
        # Decoder self-attention configuration (X ← X)
        attn_dec_self_kwargs = {
            "d_model_queries": d_model,
            "d_model_keys": d_model,
            "d_model_values": d_model,
            "attention_type": dec_self_attention_type,
            "mask_type": dec_self_mask_type,
            "dropout_qkv": dec_self_dropout_qkv,
            "attention_dropout": dec_self_attention_dropout,
            "register_entropy": True,
            "layer_name": "dec_self_att",
            "query_seq_len": X_seq_len,
            "key_seq_len": X_seq_len
        }
        
        # =====================================================================
        # DECODER
        # =====================================================================
        
        self.decoder = ReversedDecoder(
            decoder_layers=[
                ReversedDecoderLayer(
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec_cross_kwargs)),
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec_self_kwargs)),
                    d_model_dec=d_model,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                ) for _ in range(dec_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb
        )
        
        # De-embedding head (forecaster)
        self.forecaster = nn.Linear(d_model, out_dim, bias=False)
    
    def forward(
        self,
        source_tensor,
        intermediate_tensor_blanked,
        hard_masks: dict = None,
    ):
        """
        Forward pass through the single decoder.
        
        Args:
            source_tensor: Source nodes (S), shape (B, S_seq_len, features)
            intermediate_tensor_blanked: Intermediate variables (X) with values blanked, 
                                         shape (B, X_seq_len, features)
            hard_masks: Optional dict of hard masks for attention. Keys:
                        - 'dec_cross': mask for decoder cross-attention (X_len, S_len)
                        - 'dec_self': mask for decoder self-attention (X_len, X_len)
            
        Returns:
            pred_x: Predicted X from decoder
            attention_weights: Tuple of (cross_att, self_att)
            masks: Tuple of (s_mask, x_mask)
            entropies: Tuple of (cross_ent, self_ent)
        """
        
        # Extract hard masks if provided
        dec_cross_hard = None
        dec_self_hard = None
        
        if hard_masks is not None:
            dec_cross_hard = hard_masks.get('dec_cross', None)
            dec_self_hard = hard_masks.get('dec_self', None)
        
        # ===== EMBEDDING =====
        
        # Orthogonal embedding for S (frozen)
        s_embedded = self.embedding_S(source_tensor)
        s_mask = self.embedding_S.get_mask(source_tensor)
        
        # Standard embedding for X (learnable)
        x_embedded = self.embedding_X(X=intermediate_tensor_blanked)
        x_input_pos = self.embedding_X.pass_var(X=intermediate_tensor_blanked)
        x_mask = self.embedding_X.get_mask(X=intermediate_tensor_blanked)
        
        # ===== DECODER: Source → Intermediate (S → X) =====
        
        dec_out, dec_cross_att, dec_self_att, dec_cross_ent, dec_self_ent = self.decoder(
            X=x_embedded,
            external_context=s_embedded,
            self_mask_miss_k=x_mask,
            self_mask_miss_q=x_mask,
            cross_mask_miss_k=s_mask,
            cross_mask_miss_q=x_mask,
            dec_input_pos=x_input_pos,
            causal_mask=self.dec_causal_mask,
            cross_hard_mask=dec_cross_hard,
            self_hard_mask=dec_self_hard,
        )
        
        # De-embed to get predicted X
        pred_x = self.forecaster(dec_out)
        
        # Collect outputs
        attention_weights = (dec_cross_att, dec_self_att)
        masks = (s_mask, x_mask)
        entropies = (dec_cross_ent, dec_self_ent)
        
        return pred_x, attention_weights, masks, entropies
    
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
        query_seq_len: int,
        key_seq_len: int
    ):
        """Create an attention layer with specified configuration."""
        
        assert attention_type in ["ScaledDotProduct", "LieAttention", "CausalCrossAttention", "PhiSoftMax"]
        
        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention
        elif attention_type == "LieAttention":
            attention_module = LieAttention
        elif attention_type == "CausalCrossAttention":
            attention_module = CausalCrossAttention
        elif attention_type == "PhiSoftMax":
            attention_module = PhiSoftMax
        
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
    
    # =========================================================================
    # FREEZING UTILITIES
    # =========================================================================
    
    def freeze_embedding_S(self):
        """Freeze S embedding."""
        for param in self.embedding_S.parameters():
            param.requires_grad = False
    
    def unfreeze_embedding_S(self):
        """Unfreeze S embedding."""
        self.embedding_S.unfreeze()
    
    def freeze_embedding_X(self):
        """Freeze X embedding."""
        for param in self.embedding_X.parameters():
            param.requires_grad = False
    
    def unfreeze_embedding_X(self):
        """Unfreeze X embedding."""
        for param in self.embedding_X.parameters():
            param.requires_grad = True
    
    def freeze_decoder(self):
        """Freeze decoder layers."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def unfreeze_decoder(self):
        """Unfreeze decoder layers."""
        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def freeze_forecaster(self):
        """Freeze forecaster (de-embedding)."""
        for param in self.forecaster.parameters():
            param.requires_grad = False
    
    def unfreeze_forecaster(self):
        """Unfreeze forecaster."""
        for param in self.forecaster.parameters():
            param.requires_grad = True
    
    def get_embedding_info(self):
        """Return info about embedding configuration."""
        return {
            "S_embedding": repr(self.embedding_S),
            "S_frozen": not any(p.requires_grad for p in self.embedding_S.parameters()),
            "d_model": self.d_model,
        }
