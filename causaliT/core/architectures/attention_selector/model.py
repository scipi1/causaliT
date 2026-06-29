"""
AttentionSelectorLayer: Single cross-attention block for observational causal discovery.

Architecture
============

The model answers one focused question:
    "Can a single cross-attention operation learn to select the causal parents
     of each X variable from observational data, when given the actual values of
     all candidate parents (S and X) as keys?"

Forward pass
------------

1. **Embed queries** — X tokens with blanked values.
   The value column is set to 0, so only the variable-identity embedding
   carries information. The query encodes "I am X_i — who are my parents?"

2. **Embed keys/values** — [S_actual, X_actual] concatenated.
   Both S and X tokens are embedded with their ACTUAL values.
   The key encodes "I am variable j and my value is v_j".

3. **Single cross-attention** with hard mask:
   - S block  (columns 0 .. L_S-1):  all ones  → X_i may attend to any S_j
   - X block  (columns L_S .. L_S+L_X-1):  off-diagonal ones → X_i may attend
     to any X_j ≠ X_i (diagonal zeroed = no self-loops)

4. Residual + LayerNorm → FFN → LayerNorm → MLP head.

No self-attention block is included.  The single combined cross-attention IS
the causal structure.  The resulting attention matrix
    A  ∈  ℝ^{B × L_X × (L_S + L_X)}
contains the learned edge weights; splitting it gives
    A[:, :, :L_S]      → S→X adjacency
    A[:, :, L_S:]      → X→X adjacency  (diagonal = 0 by construction)

Comparison with existing architectures
---------------------------------------
SingleCausalLayer has:
  - cross-attention: Q from X_blanked, K/V from S_actual  (learns S→X)
  - self-attention:  Q/K/V from X_blanked_embedded         (learns X→X via embeddings only)

AttentionSelectorLayer has:
  - single combined cross-attention: Q from X_blanked, K/V from [S_actual, X_actual]
  - no self-attention
  - the X keys carry ACTUAL values, not just embeddings

This is a strictly easier test for the attention mechanism: it has direct access
to the true parent values, not just their structural embeddings.  If the mechanism
fails here (even with HSIC + sparsity), it cannot work in any downstream architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from causaliT.core.modules import (
    CausalCrossAttention,
    SigmoidCrossAttention,
    ScaledDotAttention,
    HardConcreteCrossAttention,
    AttentionLayer,
    ModularEmbedding,
    Normalization,
    MLPHead,
    OrthogonalMaskEmbedding,
)


class AttentionSelectorLayer(nn.Module):
    """
    AttentionSelectorLayer: single combined cross-attention causal discovery model.

    Args:
        model: Name string (stored for reference, e.g. "AttentionSelectorLayer").
        ds_embed_S: Embedding config dict for S (passed to ModularEmbedding).
        ds_embed_X: Embedding config dict for X (passed to ModularEmbedding).
        comps_embed_S: Composition mode for S embedding ("summation", etc.).
        comps_embed_X: Composition mode for X embedding ("summation", etc.).
        attention_type: One of "CausalCrossAttention", "SigmoidCrossAttention",
            "ScaledDotProduct". CausalCrossAttention (ReLU-Tanh) is recommended:
            it produces non-negative, non-normalized (can be zero) attention weights
            suitable for sparse causal structure recovery.
        n_heads: Number of attention heads (1 recommended for interpretability).
        dropout_emb: Dropout on embedding outputs.
        dropout_attn_out: Dropout on the attention output before residual add.
        dropout_ff: Dropout inside the FFN sublayer.
        dropout_qkv: Dropout on Q/K/V projections.
        attention_dropout: Dropout on the attention weights.
        activation: FFN activation ("relu" or "gelu").
        norm: Normalisation type ("layer", "batch", etc.).
        use_final_norm: Whether to apply LayerNorm after the FFN sublayer.
        device: Target device.
        out_dim: Output dimension per token (1 for scalar prediction).
        d_ff: Hidden dimension of the FFN sublayer.
        d_model: Transformer hidden dimension.
        d_qk: Q/K projection dimension per head.
        S_seq_len: Number of S variables.
        X_seq_len: Number of X variables.
        init_tau: Temperature for CausalCrossAttention / SigmoidCrossAttention (default 3.0).
        output_mlp_layers: Number of layers in the output MLP head (1 = linear).
        output_mlp_hidden: Hidden dimension of the MLP head (None → d_ff).
        output_mlp_activation: Activation in the MLP head.
        output_mlp_dropout: Dropout in the MLP head.
        shared_dag_across_heads: When True (default), a single (B,L,S) score is
            shared across all value heads (SVFA-style). When False, each head
            has its own independent DAG score.
        orthogonal_struct_embedding: When True, replace the structural stream
            (Q/K) embeddings for both S and X with ``OrthogonalMaskEmbedding``
            instances whose partitions tile the full d_model space without overlap:

                S occupies dims [0,           S_seq_len * k)
                X occupies dims [S_seq_len*k, (S_seq_len+X_seq_len)*k)

            where k = d_model // (S_seq_len + X_seq_len).  The value stream
            (V, residual, MLP head) continues to use the standard ModularEmbedding.
            Default False (original behaviour).
        batch_key_dropout: Initial probability for ``BatchConsistentKeyDropout``
            applied to the combined attention weights after the inner attention
            activation.  When set, entire key-position columns are zeroed
            consistently across the batch, preventing the model from relying on
            any single parent variable.  ``None`` (default) disables BKD.
        batch_key_dropout_p_final: Final dropout probability after annealing.
            When ``None`` (default), equals ``batch_key_dropout`` (no annealing).
        batch_key_dropout_annealing_batches: Number of optimiser steps over
            which to linearly anneal the dropout probability from
            ``batch_key_dropout`` to ``batch_key_dropout_p_final``.
            ``None`` (default) disables step-counter annealing.
    """

    def __init__(
        self,
        model: str,
        # Embedding configs
        ds_embed_S: dict,
        ds_embed_X: dict,
        comps_embed_S: str,
        comps_embed_X: str,
        # Attention
        attention_type: str,
        n_heads: int,
        # Dropout
        dropout_emb: float,
        dropout_attn_out: float,
        dropout_ff: float,
        dropout_qkv: float,
        attention_dropout: float,
        # Architecture
        activation: str,
        norm: str,
        use_final_norm: bool,
        device,
        # Dimensions
        out_dim: int,
        d_ff: int,
        d_model: int,
        d_qk: int,
        # Sequence lengths
        S_seq_len: int,
        X_seq_len: int,
        # Attention temperature
        init_tau: float = 3.0,
        # MLP output head
        output_mlp_layers: int = 1,
        output_mlp_hidden: Optional[int] = None,
        output_mlp_activation: str = "relu",
        output_mlp_dropout: float = 0.0,
        # Multi-head semantics
        shared_dag_across_heads: bool = True,
        # Orthogonal structural embeddings
        orthogonal_struct_embedding: bool = False,
        # Batch-consistent key dropout
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
    ):
        super().__init__()

        self.model_name = model
        self.d_model = d_model
        self.S_seq_len = S_seq_len
        self.X_seq_len = X_seq_len
        # Store embedding composition mode so callers (and diagnostics) can
        # inspect whether the model is operating in SVFA or standard mode.
        self.comps_embed_X = comps_embed_X
        self.orthogonal_struct_embedding = orthogonal_struct_embedding

        # ------------------------------------------------------------------
        # Embeddings
        # The SAME ModularEmbedding instance is used for both:
        #   - X queries (blanked value → only variable-ID embedding contributes)
        #   - X keys    (actual value → value + variable-ID embedding)
        # Q and K linear projections inside AttentionLayer handle the role
        # differentiation ("asking" vs "offering").
        # ------------------------------------------------------------------
        self.embedding_S = ModularEmbedding(
            ds_embed=ds_embed_S,
            comps=comps_embed_S,
            device=device,
        )
        self.embedding_X = ModularEmbedding(
            ds_embed=ds_embed_X,
            comps=comps_embed_X,
            device=device,
        )

        # ------------------------------------------------------------------
        # Combined cross-attention block
        # Q:   X_blanked        (B, L_X,        d_model)
        # K/V: [S, X_actual]    (B, L_S+L_X,    d_model)
        # ------------------------------------------------------------------
        att_cls_map = {
            "CausalCrossAttention": CausalCrossAttention,
            "SigmoidCrossAttention": SigmoidCrossAttention,
            "ScaledDotProduct": ScaledDotAttention,
            # Backward-compat alias: configs and checkpoint hparams saved before the
            # "ScaledDotProduct" rename still store attention_type="ScaledDotAttention".
            # load_from_checkpoint reconstructs the model from stored hparams, so
            # without this alias those checkpoints raise a ValueError at eval time.
            "ScaledDotAttention": ScaledDotAttention,
            # Hard Concrete L0 gates (Louizos et al., ICLR 2018).
            "HardConcreteCrossAttention": HardConcreteCrossAttention,
        }
        if attention_type not in att_cls_map:
            raise ValueError(
                f"attention_type='{attention_type}' is not supported for "
                f"AttentionSelectorLayer.  Choose from: {list(att_cls_map)}"
            )
        att_cls = att_cls_map[attention_type]

        self.attention = AttentionLayer(
            attention=att_cls,
            d_model_queries=d_model,
            d_model_keys=d_model,
            d_model_values=d_model,
            d_queries_keys=d_qk,
            n_heads=n_heads,
            mask_layer=None,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv,
            register_entropy=True,
            layer_name="selector_att",
            query_seq_len=X_seq_len,
            key_seq_len=S_seq_len + X_seq_len,
            init_tau=init_tau,
            shared_dag_across_heads=shared_dag_across_heads,
            # Batch-consistent key dropout.
            batch_key_dropout=batch_key_dropout,
            batch_key_dropout_p_final=batch_key_dropout_p_final,
            batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
        )

        # ------------------------------------------------------------------
        # Static combined hard mask: (L_X, L_S + L_X)
        #   S-block  [cols 0 .. L_S-1]:       all 1s (attend to any S)
        #   X-block  [cols L_S .. L_S+L_X-1]: off-diagonal 1s (no self-loop)
        # Registered as a buffer so it moves to the correct device automatically.
        # ------------------------------------------------------------------
        self.register_buffer(
            "combined_mask",
            self._build_combined_mask(S_seq_len, X_seq_len),
        )

        # ------------------------------------------------------------------
        # Post-attention sublayer: residual + norm + FFN + norm
        # ------------------------------------------------------------------
        self.dropout_emb = nn.Dropout(dropout_emb)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.use_final_norm = use_final_norm
        if use_final_norm:
            self.norm2 = Normalization(method=norm, d_model=d_model)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self._act_fn = F.gelu if activation == "gelu" else F.relu

        # ------------------------------------------------------------------
        # Orthogonal structural embeddings (optional override for Q/K stream)
        # ------------------------------------------------------------------
        # When orthogonal_struct_embedding=True the standard nn_embedding
        # structural role is replaced with two OrthogonalMaskEmbedding modules
        # whose dimension partitions tile d_model without overlap:
        #   S → dims [0,       S_seq_len * k)
        #   X → dims [S_seq_len*k, (S_seq_len+X_seq_len)*k)
        # where k = d_model // (S_seq_len + X_seq_len).
        # The value stream (V, residual, FFN, MLP head) is UNCHANGED.
        self.orth_embed_S: Optional[OrthogonalMaskEmbedding]
        self.orth_embed_X: Optional[OrthogonalMaskEmbedding]
        if orthogonal_struct_embedding:
            total_vars = S_seq_len + X_seq_len
            k = d_model // total_vars
            if k <= 0:
                raise ValueError(
                    f"d_model={d_model} is too small for orthogonal structural embeddings "
                    f"with S_seq_len={S_seq_len} + X_seq_len={X_seq_len} = {total_vars} variables "
                    f"(need d_model >= {total_vars})."
                )
            self.orth_embed_S = OrthogonalMaskEmbedding(
                num_variables=S_seq_len,
                d_model=d_model,
                mask_start_dim=0,
                dims_per_var=k,
                freeze=False,
                device=device,
            )
            self.orth_embed_X = OrthogonalMaskEmbedding(
                num_variables=X_seq_len,
                d_model=d_model,
                mask_start_dim=S_seq_len * k,
                dims_per_var=k,
                freeze=False,
                device=device,
            )
        else:
            self.orth_embed_S = None
            self.orth_embed_X = None

        # ------------------------------------------------------------------
        # Output MLP head
        # ------------------------------------------------------------------
        mlp_hidden = output_mlp_hidden if output_mlp_hidden is not None else d_ff
        self.forecaster = MLPHead(
            d_model=d_model,
            out_dim=out_dim,
            n_layers=output_mlp_layers,
            d_hidden=mlp_hidden,
            activation=output_mlp_activation,
            dropout=output_mlp_dropout,
            bias=(output_mlp_layers > 1),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_combined_mask(S_seq_len: int, X_seq_len: int) -> torch.Tensor:
        """Build the static (L_X, L_S + L_X) combined hard mask.

        S-block  (L_X × L_S):  all ones  — X_i may attend to any S_j.
        X-block  (L_X × L_X):  off-diagonal ones — X_i may attend to X_j ≠ X_i.
        The diagonal of the X-block is zeroed to prevent self-loops.
        """
        s_part = torch.ones(X_seq_len, S_seq_len)
        x_part = 1.0 - torch.eye(X_seq_len)   # 0 on diagonal, 1 elsewhere
        return torch.cat([s_part, x_part], dim=1)  # (L_X, L_S + L_X)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_tensor: torch.Tensor,
        intermediate_tensor: torch.Tensor,
        oracle: bool = False,
    ):
        """
        Forward pass.

        Args:
            source_tensor: S actual values, shape (B, L_S, features).
            intermediate_tensor: X actual values, shape (B, L_X, features).
                The forecaster blanks the value column externally before passing
                this tensor here; blanking is done in the forecaster so that
                actual X values are still available for constructing the key matrix.
                Internally, this method receives BOTH the blanked X (for queries)
                and is told the actual values (via the forecaster which splits them).

        Note:
            The value-blanking split is handled by the forecaster.  This method
            signature mirrors SingleCausalLayer: the forecaster passes
            `intermediate_tensor_blanked` as the first positional arg and the
            actual `intermediate_tensor` is held in the forecaster's `_step`.

            In practice, the forward call is:
                forward(S, X_blanked, X_actual)
            but the YAML/config layer sees only two tensor inputs (S, X).

        Returns:
            pred_x: (B, L_X, out_dim) predicted values.
            attention_weights: (B, L_X, L_S+L_X) attention matrix.
            entropy: Attention entropy tensor.
        """
        raise NotImplementedError(
            "Use forward_with_actual(source_tensor, x_blanked, x_actual) instead."
        )

    def forward_with_actual(
        self,
        source_tensor: torch.Tensor,
        x_blanked: torch.Tensor,
        x_actual: torch.Tensor,
        oracle: bool = False,
        oracle_combined_mask: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass with separate blanked (query) and actual (key) X.

        Args:
            source_tensor: S with actual values, shape (B, L_S, features).
            x_blanked: X with value column zeroed, shape (B, L_X, features).
                Used as the query input — only variable-identity embedding active.
            x_actual: X with real values, shape (B, L_X, features).
                Used as the key/value input — full embedding (identity + value).
            oracle: If True, bypass QK^T and use the hard mask directly as
                attention weights (inherited from CausalCrossAttention oracle mode).
            oracle_combined_mask: Optional (L_X, L_S+L_X) GT DAG combined mask.
                When oracle=True and this is provided, it is used as the hard_mask
                (i.e. the GT adjacency is the oracle attention).  When oracle=True
                but this is None, falls back to self.combined_mask (structural
                constraint — all-ones S block + off-diagonal X block), which gives
                uniform oracle attention and is useful only as a sanity baseline.
                When oracle=False this argument is ignored entirely.

        Returns:
            pred_x:           (B, L_X, out_dim)
            attention_weights: (B, L_X, L_S + L_X)  combined attention matrix
            entropy:           Attention entropy

        SVFA residual streams
        ---------------------
        When ``comps_embed_X="svfa"`` the embedding returns ``(struct, val)``
        tuples.  In that regime the architecture preserves **two separate
        residual streams**, exactly as in ``ReversedDecoderLayer``:

        * **Structure stream** (``x_struct = xq_struct``): carries the
          variable-identity signal used as the attention Query.  It passes
          through the block **unchanged** — no residual is added to it.
        * **Value stream** (``x_val``): starts from ``xq_val`` (the value
          embedding of the zero-blanked X) and accumulates the attention
          output, the FFN, and optional final norm via residual connections.
          The forecaster reads from this stream only.

        When ``comps_embed_X="summation"`` a single fused stream is used,
        preserving the original (pre-SVFA) behaviour.
        """
        # ---- Embed (SVFA-aware) ------------------------------------------
        # ModularEmbedding returns either:
        #   • a bare tensor (B, L, d)          when comps="summation"
        #   • a tuple (struct, val)             when comps="svfa"
        #     struct (B, L, d) — variable identity, used for Q and K
        #     val   (B, L, d) — actual value,     used for V
        def _emb_drop(raw):
            if isinstance(raw, tuple):
                struct, val = raw
                return self.dropout_emb(struct), self.dropout_emb(val)
            return self.dropout_emb(raw), None      # summation: no separate val

        s_struct,  s_val  = _emb_drop(self.embedding_S(X=source_tensor))
        xk_struct, xk_val = _emb_drop(self.embedding_X(X=x_actual))
        # xq_val is the initial *value* stream for x_blanked (embedding of the
        # zeroed value column).  In SVFA mode this is the residual target — it
        # must NOT be discarded.
        xq_struct, xq_val = _emb_drop(self.embedding_X(X=x_blanked))

        # ---- Orthogonal structural stream override -----------------------
        # When orthogonal_struct_embedding=True, replace the structural (Q/K)
        # embeddings with OrthogonalMaskEmbedding outputs.  The value stream
        # (s_val, xk_val, xq_val) is UNCHANGED — it still comes from the
        # standard ModularEmbedding above.
        if self.orthogonal_struct_embedding:
            assert self.orth_embed_S is not None and self.orth_embed_X is not None
            s_struct  = self.dropout_emb(self.orth_embed_S(source_tensor))
            xk_struct = self.dropout_emb(self.orth_embed_X(x_actual))
            xq_struct = self.dropout_emb(self.orth_embed_X(x_blanked))

        # Q/K always use the structural embedding; V uses value embedding if SVFA.
        sx_keys = torch.cat([s_struct, xk_struct], dim=1)   # (B, L_S+L_X, d)
        if s_val is not None and xk_val is not None:
            sx_vals = torch.cat([s_val, xk_val], dim=1)     # SVFA: values for V
        else:
            sx_vals = sx_keys                                # summation: K = V
        x_q_emb = xq_struct                                 # query tensor

        # ---- Single cross-attention --------------------------------------
        # Q: X_blanked struct  (B, L_X,       d)
        # K: [S, X_actual] struct (B, L_S+L_X, d)
        # V: [S, X_actual] val    (B, L_S+L_X, d)   (same as K in summation mode)
        #
        # hard_mask selection:
        #   oracle=True  + oracle_combined_mask provided → GT DAG combined mask
        #   oracle=True  + no oracle_combined_mask       → structural mask (fallback)
        #   oracle=False                                 → structural mask (learned att)
        if oracle and oracle_combined_mask is not None:
            hard_mask = oracle_combined_mask
        else:
            hard_mask = self.combined_mask

        attn_out, attention_weights, _aux = self.attention(
            query=x_q_emb,
            key=sx_keys,
            value=sx_vals,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=False,
            hard_mask=hard_mask,
            oracle=oracle,
        )
        # ---- Residual + Norm 1 -------------------------------------------
        # SVFA mode: the attention output (derived from the value V stream)
        # must go to the VALUE stream only.  The structure stream (x_struct =
        # xq_struct) passes through UNCHANGED — no residual is applied to it.
        # Standard (summation) mode: single fused stream (original behaviour).
        is_svfa = xq_val is not None
        if is_svfa:
            # Value stream: accumulate the attention output.
            # Structure stream: unchanged (xq_struct is held implicitly; it is
            # not modified here and is not passed to the FFN or forecaster,
            # preserving the separation of the two signals).
            x = self.norm1(xq_val + self.dropout_attn_out(attn_out))
        else:
            x = self.norm1(x_q_emb + self.dropout_attn_out(attn_out))

        # ---- FFN (value stream in SVFA, single stream in standard) -------
        x_ff = self.dropout_ff(self._act_fn(self.linear1(x)))
        x_ff = self.dropout_ff(self.linear2(x_ff))
        x = x + x_ff

        # ---- Norm 2 (optional) -------------------------------------------
        if self.use_final_norm:
            x = self.norm2(x)

        # ---- MLP head ---------------------------------------------------
        # In SVFA mode `x` is the value stream (set above); in standard mode
        # it is the single fused stream.  Either way, the forecaster reads from
        # the correct (reconstruction-targeted) stream.
        pred_x = self.forecaster(x)

        return pred_x, attention_weights, _aux

    # ------------------------------------------------------------------
    # Utility: split the combined attention matrix into S→X and X→X parts
    # ------------------------------------------------------------------

    def split_attention(
        self, attention_weights: torch.Tensor
    ):
        """
        Split the combined (B, L_X, L_S+L_X) attention matrix.

        Returns:
            att_sx: (B, L_X, L_S)  — S→X sub-matrix (learned S→X edges).
            att_xx: (B, L_X, L_X)  — X→X sub-matrix (learned X→X edges,
                                       diagonal = 0 by construction).
        """
        att_sx = attention_weights[:, :, : self.S_seq_len]
        att_xx = attention_weights[:, :, self.S_seq_len :]
        return att_sx, att_xx

    # ------------------------------------------------------------------
    # Freezing utilities (mirrors SingleCausalLayer)
    # ------------------------------------------------------------------

    def freeze_embedding_S(self):
        for p in self.embedding_S.parameters():
            p.requires_grad_(False)

    def unfreeze_embedding_S(self):
        for p in self.embedding_S.parameters():
            p.requires_grad_(True)

    def freeze_embedding_X(self):
        for p in self.embedding_X.parameters():
            p.requires_grad_(False)

    def unfreeze_embedding_X(self):
        for p in self.embedding_X.parameters():
            p.requires_grad_(True)

    def freeze_attention(self):
        for p in self.attention.parameters():
            p.requires_grad_(False)

    def unfreeze_attention(self):
        for p in self.attention.parameters():
            p.requires_grad_(True)

    def freeze_forecaster(self):
        for p in self.forecaster.parameters():
            p.requires_grad_(False)

    def unfreeze_forecaster(self):
        for p in self.forecaster.parameters():
            p.requires_grad_(True)
