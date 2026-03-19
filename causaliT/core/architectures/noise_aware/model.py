"""
NoiseAwareSingleCausalLayer: Noise-aware causal transformer for S → X learning.

This model extends the SingleCausalLayer with explicit noise modeling:
- Ambient noise (σ_A): Environmental variability in physical states
- Reading noise (σ_R): Sensor measurement uncertainty

Architecture:
1. Cross-attention: H_det = CrossAtt(X_struct, S, S)  [deterministic S → X]
2. Ambient noise:   H = H_det + σ_A * ε              [environmental variability]
3. Self-attention:  U = SelfAtt(X_struct, X_struct, H) [mixing noisy states]
4. Output head:     (μ, log_var) = head(U)           [+ reading noise σ_R]

Key Properties:
- Attention structure remains DETERMINISTIC (no noise in Q, K)
- Noise affects VALUES only (V in self-attention)
- Variance propagates through causal mixing: Var(X_i) = Σ_j α_ij² σ_A[j]² + σ_R[i]²
- Training uses Gaussian NLL for uncertainty-aware learning

Design Choices (marked for paper):
- Per-node noise: σ_A[j] and σ_R[i] are node-specific learnable parameters
- Noise before W_v: Ambient noise is injected in embedding space
- SVFA required: Structure-Value Factorized Attention for clean separation

References:
- docs/noise_aware_transformer_summary.md
- docs/NOISE_LEARNING.md
"""

from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from causaliT.core.modules import (
    LieAttention, ScaledDotAttention, CausalCrossAttention, PhiSoftMax, AttentionLayer, ToeplitzLieAttention, ToeplitzAttention,
    ModularEmbedding, OrthogonalMaskEmbedding,
    Normalization, UniformAttentionMask
)
from causaliT.core.modules.noise_layers import (
    AmbientNoiseLayer, ReadingNoiseHead, VariancePropagationTracker
)
from causaliT.core.architectures.stage_causal.decoder import ReversedDecoder
from causaliT.core.architectures.noise_aware.decoder import NoiseAwareReversedDecoderLayer


class NoiseAwareSingleCausalLayer(nn.Module):
    """
    Noise-aware causal transformer with ambient and reading noise.
    
    This model learns the causal relationship S → X while explicitly modeling:
    - Ambient noise: Environmental variability affecting physical states
    - Reading noise: Measurement uncertainty in sensor observations
    
    The model outputs a Gaussian distribution X ~ N(μ, τ²) instead of point
    predictions, enabling uncertainty quantification and NLL-based training.
    
    Architecture Flow:
    -----------------
    1. Embed S (orthogonal) and X (standard SVFA with struct + val)
    2. Cross-attention: X_val attends to S → H_det (deterministic hidden state)
    3. Ambient noise injection: H = H_det + σ_A * ε (noisy physical state)
    4. Self-attention: X_struct queries, X_struct keys, H values → U (mixed noisy states)
    5. Output head: (μ, log_var) = head(U) with reading noise σ_R
    
    SVFA Mode (required):
    --------------------
    - X embedding returns tuple (X_struct, X_val)
    - Cross-attention: Q from X_struct, K/V from S
    - Self-attention: Q, K from X_struct, V from noisy H
    - Only X_val is updated; X_struct passes through unchanged
    
    Args:
        model: Model name identifier
        ds_embed_S: S embedding configuration (orthogonal)
        ds_embed_X: X embedding configuration (standard)
        comps_embed_X: Embedding composition string
        dec_cross_attention_type: Type of cross-attention
        dec_cross_mask_type: Cross-attention mask type
        dec_self_attention_type: Type of self-attention
        dec_self_mask_type: Self-attention mask type
        n_heads: Number of attention heads
        dec_causal_mask: Whether to apply causal masking
        dropout_*: Various dropout rates
        dec_layers: Number of decoder layers
        activation: Activation function ('relu' or 'gelu')
        norm: Normalization type
        use_final_norm: Whether to use final normalization
        device: Device for model parameters
        out_dim: Output dimension per node
        d_ff: Feedforward dimension
        d_model: Model dimension
        d_qk: Query/key dimension
        S_seq_len: Length of source sequence
        X_seq_len: Length of intermediate sequence
        
        # Noise-aware specific
        init_sigma_A: Initial ambient noise standard deviation (default 0.01)
        init_sigma_R: Initial reading noise standard deviation (default 0.05)
        noise_per_dimension: If True, use per-dimension noise (default False)
        track_variance: If True, track variance propagation (default False)
        
        # DAG parameterization
        dag_parameterization_self: DAG parameterization for self-attention
        dag_parameterization_cross: DAG parameterization for cross-attention
    """
    
    def __init__(
        self,
        model: str,
        
        # S embedding configuration
        ds_embed_S: dict,
        
        # X embedding configuration (standard)
        ds_embed_X: dict,
        comps_embed_X: str,
        
        # Attention configuration for decoder
        dec_cross_attention_type: str,
        dec_cross_mask_type: str,
        dec_self_attention_type: str,
        dec_self_mask_type: str,
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
        device: str,
        
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
        
        # Noise-aware specific parameters
        init_sigma_A: float = 0.01,
        init_sigma_R: float = 0.05,
        noise_per_dimension: bool = False,
        track_variance: bool = False,
        
        # DAG parameterization
        dag_parameterization_self: str = "independent",
        dag_parameterization_cross: str = "independent",
        
        # ToeplitzLieAttention parameters (for controlling DAG decisiveness)
        # Lower gains and higher temperatures = more uncertain edge probabilities
        toeplitz_init_gain_gate: float = 2.0,   # Symmetric gate gain (was 5.0)
        toeplitz_init_gain_dir: float = 3.0,    # Direction gain (was 10.0)
        toeplitz_init_tau_gate: float = 0.5,    # Gate temperature
        toeplitz_init_tau_dir: float = 0.3,     # Direction temperature (was 0.2)
        toeplitz_max_gain: float = 20.0,        # Max gain during training (was 100.0)
        
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
        self.X_seq_len = X_seq_len
        self.S_seq_len = S_seq_len
        self.track_variance = track_variance
        
        # Noise-aware model REQUIRES SVFA factorization
        self.factorization = "svfa"
        self.dag_parameterization_self = dag_parameterization_self
        self.dag_parameterization_cross = dag_parameterization_cross
        
        # Store Toeplitz parameters for _attn method
        self._toeplitz_params = {
            "toeplitz_init_gain_gate": toeplitz_init_gain_gate,
            "toeplitz_init_gain_dir": toeplitz_init_gain_dir,
            "toeplitz_init_tau_gate": toeplitz_init_tau_gate,
            "toeplitz_init_tau_dir": toeplitz_init_tau_dir,
            "toeplitz_max_gain": toeplitz_max_gain,
        }
        
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
                var_id_offset=ds_embed_S.get("var_id_offset", 1),
                freeze=ds_embed_S.get("freeze", True),
                device=device
            )
        
        # Standard SVFA embedding for X (learnable, returns struct + val tuple)
        self.embedding_X = ModularEmbedding(
            ds_embed=ds_embed_X,
            comps=comps_embed_X,
            device=device
        )
        
        # =====================================================================
        # NOISE LAYERS
        # =====================================================================
        
        # Ambient noise layer (injected after cross-attention)
        self.ambient_noise = AmbientNoiseLayer(
            num_nodes=X_seq_len,
            d_model=d_model if noise_per_dimension else None,
            init_sigma=init_sigma_A,
            per_dimension=noise_per_dimension
        )
        
        # Variance propagation tracker (optional)
        if track_variance:
            self.variance_tracker = VariancePropagationTracker(num_nodes=X_seq_len)
        else:
            self.variance_tracker = None
        
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
            "dag_parameterization": dag_parameterization_cross,
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
            "dag_parameterization": dag_parameterization_self,
            "key_projection_type": key_projection_type_self,     # Linear for X keys
            "orthogonal_scale": orthogonal_scale,
            "orthogonal_init_scale": orthogonal_init_scale,
        }
        
        # =====================================================================
        # NOISE-AWARE DECODER
        # =====================================================================
        
        self.decoder = NoiseAwareReversedDecoder(
            decoder_layers=[
                NoiseAwareReversedDecoderLayer(
                    global_cross_attention=self._attn(**(attn_shared_kwargs | attn_dec_cross_kwargs)),
                    global_self_attention=self._attn(**(attn_shared_kwargs | attn_dec_self_kwargs)),
                    ambient_noise_layer=self.ambient_noise,
                    d_model_dec=d_model,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                ) for _ in range(dec_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )
        
        # =====================================================================
        # PROBABILISTIC OUTPUT HEAD
        # =====================================================================
        
        # Reading noise head (replaces standard forecaster)
        self.output_head = ReadingNoiseHead(
            d_model=d_model,
            num_nodes=X_seq_len,
            out_dim=out_dim,
            init_sigma_R=init_sigma_R,
            learn_variance=True
        )
    
    def forward(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
        inject_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple, Tuple]:
        """
        Forward pass through the noise-aware decoder.
        
        Args:
            source_tensor: Source nodes (S), shape (B, S_seq_len, features)
            intermediate_tensor_blanked: Intermediate variables (X) with values blanked,
                                         shape (B, X_seq_len, features)
            hard_masks: Optional dict of hard masks for attention. Keys:
                        - 'dec_cross': mask for decoder cross-attention (X_len, S_len)
                        - 'dec_self': mask for decoder self-attention (X_len, X_len)
            inject_noise: If True, inject ambient noise during training. Default True.
            
        Returns:
            mu: (B, X_seq_len, out_dim) predicted mean
            log_var: (B, X_seq_len, out_dim) log-variance
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
        
        # SVFA embedding for X (learnable, returns tuple)
        x_embedded = self.embedding_X(X=intermediate_tensor_blanked)
        x_input_pos = self.embedding_X.pass_var(X=intermediate_tensor_blanked)
        x_mask = self.embedding_X.get_mask(X=intermediate_tensor_blanked)
        
        # ===== NOISE-AWARE DECODER =====
        # x_embedded is tuple (X_struct, X_val)
        # s_embedded is single tensor (orthogonal basis)
        
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
            inject_noise=inject_noise,
        )
        
        # Extract value embedding from SVFA tuple for output
        _, x_val = dec_out  # (X_struct, X_val)
        
        # ===== PROBABILISTIC OUTPUT =====
        # Output head adds reading noise and returns distribution parameters
        mu, log_var = self.output_head(x_val)
        
        # ===== VARIANCE TRACKING (optional) =====
        if self.track_variance and self.variance_tracker is not None:
            # Track variance propagation through the last layer's self-attention
            if dec_self_att:
                self.variance_tracker.update(
                    attention_weights=dec_self_att[-1],
                    sigma_A_squared=self.ambient_noise.get_variance_contribution(),
                    sigma_R_squared=self.output_head.get_variance()
                )
        
        # Collect outputs
        attention_weights = (dec_cross_att, dec_self_att)
        masks = (s_mask, x_mask)
        entropies = (dec_cross_ent, dec_self_ent)
        
        return mu, log_var, attention_weights, masks, entropies
    
    def predict(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions (inference mode, no noise injection).
        
        Args:
            source_tensor: Source nodes (S)
            intermediate_tensor_blanked: Intermediate variables (X) with blanked values
            hard_masks: Optional hard masks
            
        Returns:
            mu: Predicted mean
            std: Predicted standard deviation
        """
        self.eval()
        with torch.no_grad():
            mu, log_var, _, _, _ = self.forward(
                source_tensor, 
                intermediate_tensor_blanked,
                hard_masks,
                inject_noise=False  # No noise at inference
            )
            std = torch.exp(0.5 * log_var)
        return mu, std
    
    def sample(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        num_samples: int = 100,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Sample from the predictive distribution (Monte Carlo).
        
        Args:
            source_tensor: Source nodes (S)
            intermediate_tensor_blanked: Intermediate variables (X) with blanked values
            num_samples: Number of samples to draw
            hard_masks: Optional hard masks
            
        Returns:
            samples: (num_samples, B, X_seq_len, out_dim) samples
        """
        mu, std = self.predict(source_tensor, intermediate_tensor_blanked, hard_masks)
        
        samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(mu)
            samples.append(mu + std * eps)
        
        return torch.stack(samples, dim=0)
    
    def get_noise_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current noise parameter values.
        
        Returns:
            Dict with 'sigma_A' and 'sigma_R' tensors
        """
        return {
            'sigma_A': self.ambient_noise.sigma_A.detach(),
            'sigma_R': self.output_head.sigma_R.detach(),
        }
    
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
        
        assert attention_type in ["ScaledDotProduct", "LieAttention", "CausalCrossAttention", "PhiSoftMax", "ToeplitzLieAttention", "ToeplitzAttention"]
        
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
        elif attention_type == "ToeplitzAttention":
            attention_module = ToeplitzAttention
        
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
            # Pass Toeplitz-specific parameters (used only when attention_type is ToeplitzLieAttention)
            **self._toeplitz_params
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
    
    def freeze_noise_parameters(self):
        """Freeze noise parameters (σ_A and σ_R)."""
        self.ambient_noise.log_sigma_A.requires_grad = False
        self.output_head.log_sigma_R.requires_grad = False
    
    def unfreeze_noise_parameters(self):
        """Unfreeze noise parameters."""
        self.ambient_noise.log_sigma_A.requires_grad = True
        self.output_head.log_sigma_R.requires_grad = True
    
    def freeze_output_head(self):
        """Freeze output head."""
        for param in self.output_head.parameters():
            param.requires_grad = False
    
    def unfreeze_output_head(self):
        """Unfreeze output head."""
        for param in self.output_head.parameters():
            param.requires_grad = True
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Return info about embedding and noise configuration."""
        return {
            "S_embedding": repr(self.embedding_S),
            "S_frozen": not any(p.requires_grad for p in self.embedding_S.parameters()),
            "d_model": self.d_model,
            "factorization": self.factorization,
            "sigma_A_mean": self.ambient_noise.sigma_A.mean().item(),
            "sigma_R_mean": self.output_head.sigma_R.mean().item(),
            "track_variance": self.track_variance,
        }


class NoiseAwareReversedDecoder(nn.Module):
    """
    Stack of NoiseAwareReversedDecoderLayer modules.
    
    This decoder processes input through multiple layers with noise injection,
    implementing the noise-aware causal mixing.
    """
    
    def __init__(
        self,
        decoder_layers: list,
        norm_layer: nn.Module,
        emb_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)
    
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
        cross_hard_mask: torch.Tensor = None,
        self_hard_mask: torch.Tensor = None,
        inject_noise: bool = True,
    ):
        """
        Forward pass through all decoder layers with noise injection.
        
        Args:
            X: Tuple (X_struct, X_val) - SVFA embeddings
            external_context: External context for cross-attention (S embedding)
            inject_noise: Whether to inject ambient noise
            ... (other args same as ReversedDecoder)
            
        Returns:
            (X_struct, X_val): Updated SVFA embeddings
            cross_att_list: Cross-attention weights per layer
            self_att_list: Self-attention weights per layer
            cross_ent_list: Cross-attention entropies per layer
            self_ent_list: Self-attention entropies per layer
        """
        not_mask = ~self_mask_miss_q if self_mask_miss_q is not None else None
        
        # Unpack SVFA tuple
        X_struct, X_val = X
        
        # Apply embedding dropout
        X_struct = self.emb_dropout(X_struct)
        X_val = self.emb_dropout(X_val)
        X = (X_struct, X_val)
        
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
                inject_noise=inject_noise,
            )
            
            cross_att_list.append(cross_att)
            self_att_list.append(self_att)
            cross_ent_list.append(cross_ent)
            self_ent_list.append(self_ent)
        
        # Apply final normalization
        if self.norm_layer is not None:
            X_struct, X_val = X
            X_val = self.norm_layer(X_val, not_mask)
            X = (X_struct, X_val)
        
        return X, cross_att_list, self_att_list, cross_ent_list, self_ent_list
