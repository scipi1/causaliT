"""
VarianceCausalLayer: variance-centric causal transformer (S -> X).

See docs/documentation/NOISE_AWARE_2.md for the full design description.

Forward pass:
    1. Embed S and X (SVFA: struct + val streams).
    2. Cross-attention (dual_value=True): H_det = CrossAtt(X, S); X_struct updated.
    3. Self-attention  (dual_value=True): U = SelfAtt(X, X, H_det); X_struct updated.
    4. mu    = head(U)
    5. log_var = log( (alpha**2) @ sigma_A**2 )   [analytical, from last layer alpha]
"""

from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn

from causaliT.core.modules import (
    ScaledDotSoftmax, CausalCrossAttention, SigmoidCrossAttention,
    AttentionLayer, ToeplitzAttention,
    ModularEmbedding, OrthogonalMaskEmbedding,
    Normalization, UniformAttentionMask,
    MLPHead,
)
from causaliT.core.modules.variance_layers import (
    IntrinsicNoiseLayer,
    AnalyticalVarianceHead,
)
from causaliT.core.architectures.variance_causal.decoder import (
    VarianceCausalDecoder,
    VarianceCausalDecoderLayer,
)


class VarianceCausalLayer(nn.Module):
    """Variance-centric causal transformer with analytical noise propagation."""

    def __init__(
        self,
        model: str,
        ds_embed_S: dict,
        ds_embed_X: dict,
        comps_embed_X: str,
        dec_cross_attention_type: str,
        dec_cross_mask_type: str,
        dec_self_attention_type: str,
        dec_self_mask_type: str,
        n_heads: int,
        dec_causal_mask: bool,
        dropout_emb: float,
        dropout_attn_out: float,
        dropout_ff: float,
        dec_cross_dropout_qkv: float,
        dec_cross_attention_dropout: float,
        dec_self_dropout_qkv: float,
        dec_self_attention_dropout: float,
        dec_layers: int,
        activation: str,
        norm: str,
        use_final_norm: bool,
        device: str,
        out_dim: int,
        d_ff: int,
        d_model: int,
        d_qk: int,
        S_seq_len: int,
        X_seq_len: int,
        comps_embed_S: str = None,
        init_sigma: float = 0.1,
        # Legacy kwargs accepted but ignored for backward config compat.
        init_sigma_A: float = None,
        init_sigma_R: float = None,
        variance_head_eps: float = 1e-6,
        output_mlp_layers: int = 1,
        output_mlp_hidden: int = None,
        output_mlp_activation: str = "relu",
        output_mlp_dropout: float = 0.0,
        share_structure_across_layers: bool = False,
        key_projection_type_cross: str = "linear",
        key_projection_type_self: str = "linear",
        orthogonal_scale: bool = True,
        orthogonal_init_scale: float = 1.0,
        toeplitz_init_gate_bias: float = -15.0,
        toeplitz_gate_bias_trainable: bool = True,
        init_tau: float = 3.0,
        shared_dag_across_heads: bool = True,
        batch_key_dropout: float = None,
        batch_key_dropout_p_final: float = None,
        batch_key_dropout_annealing_batches: int = None,
    ):
        super().__init__()

        self.model_name = model
        self.dec_causal_mask = dec_causal_mask
        self.d_model = d_model
        self.X_seq_len = X_seq_len
        self.S_seq_len = S_seq_len
        self.factorization = "svfa"  # SVFA required: Q/K from node identity only

        # =====================================================================
        # EMBEDDINGS
        # =====================================================================
        self.comps_embed_S = comps_embed_S

        if comps_embed_S is not None:
            self.embedding_S = ModularEmbedding(
                ds_embed=ds_embed_S, comps=comps_embed_S, device=device
            )
        else:
            self.embedding_S = OrthogonalMaskEmbedding(
                num_variables=ds_embed_S["num_variables"],
                d_model=d_model,
                value_input_dim=ds_embed_S.get("value_input_dim", 1),
                value_idx=ds_embed_S["value_idx"],
                var_idx=ds_embed_S["var_idx"],
                var_id_offset=ds_embed_S.get("var_id_offset", 1),
                freeze=ds_embed_S.get("freeze", True),
                device=device,
            )

        self.embedding_X = ModularEmbedding(
            ds_embed=ds_embed_X, comps=comps_embed_X, device=device
        )

        # =====================================================================
        # INTRINSIC NOISE PARAMETER  (no sampling — single σ per node)
        # =====================================================================
        _init_sigma = init_sigma_A if init_sigma_A is not None else init_sigma
        self.intrinsic_noise = IntrinsicNoiseLayer(
            num_nodes=X_seq_len,
            init_sigma=_init_sigma,
        )

        # =====================================================================
        # ANALYTICAL VARIANCE HEAD
        # =====================================================================
        self.variance_head = AnalyticalVarianceHead(eps=variance_head_eps)

        # =====================================================================
        # ATTENTION KWARGS
        # =====================================================================
        attn_shared = dict(
            n_heads=n_heads,
            d_queries_keys=d_qk,
            init_tau=init_tau,
            init_gate_bias=toeplitz_init_gate_bias,
            gate_bias_trainable=toeplitz_gate_bias_trainable,
            shared_dag_across_heads=shared_dag_across_heads,
            batch_key_dropout=batch_key_dropout,
            batch_key_dropout_p_final=batch_key_dropout_p_final,
            batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
        )

        attn_cross = dict(
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            attention_type=dec_cross_attention_type, mask_type=dec_cross_mask_type,
            dropout_qkv=dec_cross_dropout_qkv, attention_dropout=dec_cross_attention_dropout,
            register_entropy=True, layer_name="dec_cross_att",
            query_seq_len=X_seq_len, key_seq_len=S_seq_len,
            key_projection_type=key_projection_type_cross,
            orthogonal_scale=orthogonal_scale, orthogonal_init_scale=orthogonal_init_scale,
        )

        attn_self = dict(
            d_model_queries=d_model, d_model_keys=d_model, d_model_values=d_model,
            attention_type=dec_self_attention_type, mask_type=dec_self_mask_type,
            dropout_qkv=dec_self_dropout_qkv, attention_dropout=dec_self_attention_dropout,
            register_entropy=True, layer_name="dec_self_att",
            query_seq_len=X_seq_len, key_seq_len=X_seq_len,
            key_projection_type=key_projection_type_self,
            orthogonal_scale=orthogonal_scale, orthogonal_init_scale=orthogonal_init_scale,
        )

        # =====================================================================
        # DECODER LAYERS
        # =====================================================================
        def _make_layer(cross_att, self_att, shared_struct_norms=None):
            return VarianceCausalDecoderLayer(
                global_cross_attention=cross_att,
                global_self_attention=self_att,
                d_model_dec=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                dropout_attn_out=dropout_attn_out,
                activation=activation,
                norm=norm,
                shared_struct_norms=shared_struct_norms,
            )

        if share_structure_across_layers and dec_layers > 1:
            cross_att_0 = self._attn(**(attn_shared | attn_cross))
            self_att_0 = self._attn(**(attn_shared | attn_self))
            shared_cross = cross_att_0.get_shared_qk_inner()
            shared_self = self_att_0.get_shared_qk_inner()

            decoder_layers = []
            shared_struct_norms = None
            for i in range(dec_layers):
                if i == 0:
                    ca, sa = cross_att_0, self_att_0
                else:
                    ca = self._attn(**(attn_shared | attn_cross), shared_qk_inner=shared_cross)
                    sa = self._attn(**(attn_shared | attn_self), shared_qk_inner=shared_self)
                decoder_layers.append(_make_layer(ca, sa, shared_struct_norms))
                if i == 0:
                    shared_struct_norms = {
                        "norm1_struct": decoder_layers[0].norm1_struct,
                        "norm2_struct": decoder_layers[0].norm2_struct,
                    }
        else:
            decoder_layers = [
                _make_layer(
                    self._attn(**(attn_shared | attn_cross)),
                    self._attn(**(attn_shared | attn_self)),
                )
                for _ in range(dec_layers)
            ]

        self.decoder = VarianceCausalDecoder(
            decoder_layers=decoder_layers,
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        # =====================================================================
        # DETERMINISTIC MEAN HEAD
        # =====================================================================
        mlp_hidden = output_mlp_hidden if output_mlp_hidden is not None else d_ff
        self.output_head = MLPHead(
            d_model=d_model,
            out_dim=out_dim,
            n_layers=output_mlp_layers,
            d_hidden=mlp_hidden,
            activation=output_mlp_activation,
            dropout=output_mlp_dropout,
            bias=(output_mlp_layers > 1),
        )

    # ------------------------------------------------------------------
    def _attn(
        self,
        d_model_queries, d_model_keys, d_model_values,
        n_heads, d_queries_keys, attention_type, mask_type,
        dropout_qkv, attention_dropout, register_entropy, layer_name,
        query_seq_len, key_seq_len,
        key_projection_type="linear", orthogonal_scale=True,
        orthogonal_init_scale=1.0, shared_qk_inner=None,
        init_tau=3.0, init_gate_bias=-15.0, gate_bias_trainable=True,
        shared_dag_across_heads=True,
        batch_key_dropout=None, batch_key_dropout_p_final=None,
        batch_key_dropout_annealing_batches=None,
    ):
        """Build an AttentionLayer.  Always dual_value=True for dual-residual decoder."""
        assert attention_type in [
            "ScaledDotSoftmax", "CausalCrossAttention",
            "SigmoidCrossAttention", "ToeplitzAttention",
        ]
        attn_cls = {
            "ScaledDotSoftmax": ScaledDotSoftmax,
            "CausalCrossAttention": CausalCrossAttention,
            "SigmoidCrossAttention": SigmoidCrossAttention,
            "ToeplitzAttention": ToeplitzAttention,
        }[attention_type]

        mask_layer = UniformAttentionMask() if mask_type == "Uniform" else None

        return AttentionLayer(
            attention=attn_cls,
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
            key_projection_type=key_projection_type,
            orthogonal_scale=orthogonal_scale,
            orthogonal_init_scale=orthogonal_init_scale,
            shared_qk_inner=shared_qk_inner,
            init_tau=init_tau,
            init_gate_bias=init_gate_bias,
            gate_bias_trainable=gate_bias_trainable,
            shared_dag_across_heads=shared_dag_across_heads,
            dual_value=True,  # required by VarianceCausalDecoderLayer
            batch_key_dropout=batch_key_dropout,
            batch_key_dropout_p_final=batch_key_dropout_p_final,
            batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Tuple, Tuple]:
        """
        Fully deterministic forward pass.

        Args:
            source_tensor: (B, S_seq_len, features)
            intermediate_tensor_blanked: (B, X_seq_len, features) — values zeroed.
            hard_masks: optional dict with "dec_cross" / "dec_self" masks.

        Returns:
            mu:       (B, X_seq_len, out_dim)
            log_var:  (B, X_seq_len, 1)   — analytical log-variance
            attention_weights: (cross_att_list, self_att_list)
            masks:    (s_mask, x_mask)
            entropies: (cross_ent_list, self_ent_list)
        """
        dec_cross_hard = hard_masks.get("dec_cross") if hard_masks else None
        dec_self_hard  = hard_masks.get("dec_self")  if hard_masks else None

        # --- Embedding ---
        if self.comps_embed_S is not None:
            s_embedded = self.embedding_S(X=source_tensor)
            s_mask = self.embedding_S.get_mask(X=source_tensor)
        else:
            s_embedded = self.embedding_S(source_tensor)
            s_mask = self.embedding_S.get_mask(source_tensor)

        x_embedded   = self.embedding_X(X=intermediate_tensor_blanked)
        x_input_pos  = self.embedding_X.pass_var(X=intermediate_tensor_blanked)
        x_mask       = self.embedding_X.get_mask(X=intermediate_tensor_blanked)

        # --- Deterministic decoder ---
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
        _, x_val = dec_out  # (X_struct, X_val)

        # --- Mean ---
        mu = self.output_head(x_val)  # (B, L, out_dim)

        # --- Analytical variance from last layer's self-attention ---
        alpha_last = dec_self_att[-1]                          # (B, L, L)
        log_var = self.variance_head(
            alpha_last,
            self.intrinsic_noise.sigma,
        )  # (B, L, 1)

        return (
            mu,
            log_var,
            (dec_cross_att, dec_self_att),
            (s_mask, x_mask),
            (dec_cross_ent, dec_self_ent),
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, std); fully deterministic."""
        self.eval()
        with torch.no_grad():
            mu, log_var, _, _, _ = self.forward(
                source_tensor, intermediate_tensor_blanked, hard_masks
            )
        return mu, torch.exp(0.5 * log_var)

    def get_noise_parameters(self) -> Dict[str, torch.Tensor]:
        return {"sigma": self.intrinsic_noise.sigma.detach()}

    def get_sigma_model(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor_blanked: torch.Tensor,
        hard_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return (L, L) Wright covariance matrix from batch-mean alpha."""
        self.eval()
        with torch.no_grad():
            _, _, (_, dec_self_att), _, _ = self.forward(
                source_tensor, intermediate_tensor_blanked, hard_masks
            )
            return self.variance_head.compute_sigma_model(
                dec_self_att[-1], self.intrinsic_noise.sigma_A
            )

    # ------------------------------------------------------------------
    # Freezing utilities
    # ------------------------------------------------------------------

    def freeze_embedding_S(self):
        for p in self.embedding_S.parameters():
            p.requires_grad = False

    def unfreeze_embedding_S(self):
        self.embedding_S.unfreeze()

    def freeze_embedding_X(self):
        for p in self.embedding_X.parameters():
            p.requires_grad = False

    def unfreeze_embedding_X(self):
        for p in self.embedding_X.parameters():
            p.requires_grad = True

    def freeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_decoder(self):
        for p in self.decoder.parameters():
            p.requires_grad = True

    def freeze_noise_parameters(self):
        self.intrinsic_noise.log_sigma.requires_grad = False

    def unfreeze_noise_parameters(self):
        self.intrinsic_noise.log_sigma.requires_grad = True

    def freeze_output_head(self):
        for p in self.output_head.parameters():
            p.requires_grad = False

    def unfreeze_output_head(self):
        for p in self.output_head.parameters():
            p.requires_grad = True

    def get_embedding_info(self) -> Dict[str, Any]:
        return {
            "S_embedding": repr(self.embedding_S),
            "S_frozen": not any(
                p.requires_grad for p in self.embedding_S.parameters()
            ),
            "d_model": self.d_model,
            "factorization": self.factorization,
            "sigma_mean": self.intrinsic_noise.sigma.mean().item(),
        }
