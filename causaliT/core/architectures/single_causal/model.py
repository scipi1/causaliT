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
    LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer,ToeplitzLieAttention,
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
    
    SVFA Mode (factorization="svfa"):
    - X embedding returns tuple (X_struct, X_val)
    - S embedding remains single tensor (orthogonal basis)
    - Cross-attention: Q from X_struct, K/V from S
    - Self-attention: Q, K from X_struct, V from X_val
    - Only X_val is used for forecasting
    
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
        
        # S embedding composition (optional - if provided, use ModularEmbedding for S)
        comps_embed_S: str = None,
        
        # SVFA: factorization mode ("standard" or "svfa")
        factorization: str = "standard",
        
        # DAG parameterization for self-attention: "independent", "antisymmetric", or "gated"
        # "antisymmetric" enforces P(i→j) + P(j→i) = 1, preventing bidirectional edges
        # "gated" adds symmetric gate + antisymmetric direction (requires square attention)
        dag_parameterization_self: str = "independent",
        
        # DAG parameterization for cross-attention: must be "independent"
        # Cross-attention is non-square (X queries, S keys), so only "independent" is valid
        dag_parameterization_cross: str = "independent",
        
        # Attention bypass mode for Attention Necessity Score (ANS) evaluation
        # When True, replaces learned attention with uniform attention
        # This tests if the model can fit data using only embeddings + MLP
        attention_bypass: bool = False,
        
        # Key projection type for preserving embedding orthogonality
        # - "linear": Standard unconstrained linear projection (default)
        # - "orthogonal": Orthogonal projection (rotation + optional scaling)
        # For cross-attention (S keys): use "orthogonal" to preserve S orthogonality
        # For self-attention (X keys): use "linear" (X is not orthogonal)
        key_projection_type_cross: str = "linear",
        key_projection_type_self: str = "linear",
        orthogonal_scale: bool = True,
        orthogonal_init_scale: float = 1.0,
    ):
        super().__init__()
        
        # Store configuration
        self.model_name = model
        self.dec_causal_mask = dec_causal_mask
        self.d_model = d_model
        self.factorization = factorization
        self.dag_parameterization_self = dag_parameterization_self
        self.dag_parameterization_cross = dag_parameterization_cross
        self.attention_bypass = attention_bypass
        
        # Store sequence lengths for attention bypass
        self.S_seq_len = S_seq_len
        self.X_seq_len = X_seq_len
        
        # Store key projection parameters
        self.key_projection_type_cross = key_projection_type_cross
        self.key_projection_type_self = key_projection_type_self
        self.orthogonal_scale = orthogonal_scale
        self.orthogonal_init_scale = orthogonal_init_scale
        
        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        
        # Store comps_embed_S to determine embedding type
        self.comps_embed_S = comps_embed_S
        
        # S embedding: use ModularEmbedding if comps_embed_S is provided, else OrthogonalMaskEmbedding
        if comps_embed_S is not None:
            # Learnable embedding for S (same style as X)
            self.embedding_S = ModularEmbedding(
                ds_embed=ds_embed_S,
                comps=comps_embed_S,
                device=device
            )
        else:
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
        # S keys are orthogonal, so use key_projection_type_cross (can be "orthogonal")
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
            "key_seq_len": S_seq_len,
            "dag_parameterization": dag_parameterization_cross,  # Non-square: must be "independent"
            "key_projection_type": key_projection_type_cross,    # Orthogonal for S keys
            "orthogonal_scale": orthogonal_scale,
            "orthogonal_init_scale": orthogonal_init_scale,
        }
        
        # Decoder self-attention configuration (X ← X)
        # X keys are not orthogonal, so use key_projection_type_self (typically "linear")
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
            "key_seq_len": X_seq_len,
            "dag_parameterization": dag_parameterization_self,  # Square: can use any
            "key_projection_type": key_projection_type_self,    # Linear for X keys
            "orthogonal_scale": orthogonal_scale,
            "orthogonal_init_scale": orthogonal_init_scale,
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
                    factorization=factorization,
                ) for _ in range(dec_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
            factorization=factorization
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
        
        # S embedding - depends on comps_embed_S
        if self.comps_embed_S is not None:
            # ModularEmbedding for S (learnable, returns tuple when SVFA)
            s_embedded = self.embedding_S(X=source_tensor)
            s_mask = self.embedding_S.get_mask(X=source_tensor)
        else:
            # OrthogonalMaskEmbedding for S (frozen, returns single tensor)
            s_embedded = self.embedding_S(source_tensor)
            s_mask = self.embedding_S.get_mask(source_tensor)
        
        # Standard embedding for X (learnable)
        x_embedded = self.embedding_X(X=intermediate_tensor_blanked)
        x_input_pos = self.embedding_X.pass_var(X=intermediate_tensor_blanked)
        x_mask = self.embedding_X.get_mask(X=intermediate_tensor_blanked)
        
        # ===== ATTENTION BYPASS MODE =====
        # When attention_bypass=True, replace learned attention with uniform attention
        # This is used for ANS (Attention Necessity Score) evaluation to test if
        # the model can fit data using only embeddings + MLP (no learned attention)
        if self.attention_bypass:
            dec_out, dec_cross_att, dec_self_att, dec_cross_ent, dec_self_ent = self._forward_bypass(
                x_embedded=x_embedded,
                s_embedded=s_embedded,
                x_mask=x_mask,
                s_mask=s_mask,
            )
        else:
            # ===== DECODER: Source → Intermediate (S → X) =====
            # In SVFA mode: x_embedded is tuple (X_struct, X_val), s_embedded is single tensor
            # Decoder will return tuple in SVFA mode
            
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
        # In SVFA mode: extract value embedding from tuple for forecasting
        if self.factorization == "svfa":
            _, x_val = dec_out
            pred_x = self.forecaster(x_val)
        else:
            pred_x = self.forecaster(dec_out)
        
        # Collect outputs
        attention_weights = (dec_cross_att, dec_self_att)
        masks = (s_mask, x_mask)
        entropies = (dec_cross_ent, dec_self_ent)
        
        return pred_x, attention_weights, masks, entropies
    
    def _forward_bypass(
        self,
        x_embedded,
        s_embedded,
        x_mask,
        s_mask,
    ):
        """
        Forward pass with attention bypass (uniform attention).
        
        This method replaces learned attention with uniform attention for ANS evaluation.
        It preserves the same architecture (embeddings → attention aggregation → FF → forecaster)
        but uses uniform attention weights instead of learned ones.
        
        Uniform attention means each query attends equally to all keys:
        - Cross-attention: each X token attends equally to all S tokens
        - Self-attention: each X token attends equally to all other X tokens
        
        Args:
            x_embedded: X embeddings (B, X_len, d_model) or tuple for SVFA
            s_embedded: S embeddings (B, S_len, d_model)
            x_mask: Missing mask for X
            s_mask: Missing mask for S
            
        Returns:
            Same outputs as normal forward pass
        """
        batch_size = s_embedded.shape[0]
        device = s_embedded.device
        
        # Handle SVFA mode
        if self.factorization == "svfa":
            x_struct, x_val = x_embedded
        else:
            x_struct = x_embedded
            x_val = x_embedded
        
        # Create uniform attention weights
        # Cross-attention: X queries (X_seq_len) attend to S keys (S_seq_len)
        uniform_cross = torch.ones(batch_size, self.X_seq_len, self.S_seq_len, device=device) / self.S_seq_len
        
        # Self-attention: X queries attend to X keys
        uniform_self = torch.ones(batch_size, self.X_seq_len, self.X_seq_len, device=device) / self.X_seq_len
        
        # Apply cross-attention with uniform weights (average over S)
        # cross_out = uniform_cross @ s_embedded -> (B, X_len, d_model)
        cross_out = torch.bmm(uniform_cross, s_embedded)
        
        # For bypass mode, we combine cross-attention output with X embeddings
        # This mimics the residual connection structure of the decoder
        if self.factorization == "svfa":
            # In SVFA: value embedding is what gets updated
            x_val = x_val + cross_out
        else:
            x_struct = x_struct + cross_out
            x_val = x_struct
        
        # Apply self-attention with uniform weights (average over X)
        # self_out = uniform_self @ x_val -> (B, X_len, d_model)
        self_out = torch.bmm(uniform_self, x_val)
        
        # Residual connection
        if self.factorization == "svfa":
            x_val = x_val + self_out
        else:
            x_val = x_val + self_out
        
        # Apply feed-forward from first decoder layer (reuse existing FF weights)
        # This ensures the MLP capacity is the same as in normal forward pass
        decoder_layer = self.decoder.layers[0]
        x_ff = decoder_layer.norm3(x_val, ~x_mask if x_mask is not None else None)
        x_ff = decoder_layer.dropout_ff(decoder_layer.activation(decoder_layer.linear1(x_ff)))
        x_ff = decoder_layer.dropout_ff(decoder_layer.linear2(x_ff))
        x_val = x_val + x_ff
        
        # Apply final normalization if present
        if self.decoder.norm_layer is not None:
            x_val = self.decoder.norm_layer(x_val, ~x_mask if x_mask is not None else None)
        
        # Prepare outputs in same format as normal forward
        if self.factorization == "svfa":
            dec_out = (x_struct, x_val)
        else:
            dec_out = x_val
        
        # Return uniform attention weights (as lists to match decoder output format)
        # Note: entropy is 0 for uniform attention (maximum entropy = log(seq_len))
        dec_cross_att = [uniform_cross]
        dec_self_att = [uniform_self]
        
        # Compute entropy for uniform attention (should be maximum = log(seq_len))
        cross_entropy = torch.log(torch.tensor(self.S_seq_len, dtype=torch.float32, device=device))
        self_entropy = torch.log(torch.tensor(self.X_seq_len, dtype=torch.float32, device=device))
        
        dec_cross_ent = [cross_entropy.expand(batch_size, self.X_seq_len)]
        dec_self_ent = [self_entropy.expand(batch_size, self.X_seq_len)]
        
        return dec_out, dec_cross_att, dec_self_att, dec_cross_ent, dec_self_ent
    
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
        key_seq_len: int,
        dag_parameterization: str = "independent",
        key_projection_type: str = "linear",
        orthogonal_scale: bool = True,
        orthogonal_init_scale: float = 1.0,
    ):
        """Create an attention layer with specified configuration."""
        
        assert attention_type in ["ScaledDotProduct", "LieAttention", "CausalCrossAttention", "PhiSoftMax", "ToeplitzLieAttention"]
        
        if attention_type == "ScaledDotProduct":
            attention_module = ScaledDotAttention
        elif attention_type == "LieAttention":
            attention_module = LieAttention
        elif attention_type == "CausalCrossAttention":
            attention_module = CausalCrossAttention
        elif attention_type == "PhiSoftMax":
            attention_module = PhiSoftMax
        elif attention_type == "ToeplitzLieAttention":
            attention_module = ToeplitzLieAttention
        
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
            key_seq_len=key_seq_len,
            dag_parameterization=dag_parameterization,
            key_projection_type=key_projection_type,
            orthogonal_scale=orthogonal_scale,
            orthogonal_init_scale=orthogonal_init_scale,
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
