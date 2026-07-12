"""
SelfSelectorLayer: whole-graph causal discovery via direction-aware self-attention.

Motivation
==========
``AttentionSelectorLayer`` assumes a KNOWN set of source nodes S: it hard-codes
the direction S -> X (S appears only as keys, never as queries) and lets a
combined cross-attention pick each X's parents.  In a standard causal-discovery
task this assumption does not hold — we do not know which variables are sources,
and the model must infer the WHOLE directed acyclic graph over all variables.

This layer removes the S/X distinction entirely and treats the data as a single
HOMOGENEOUS set of ``N = L_S + L_X`` nodes (Option 2).  Every node is
simultaneously:

* a **query** (candidate child, value-blanked) — "who are my parents?"
* a **key/value** (candidate parent, actual value) — "I am node j, value v_j".

A single :class:`GatedSelfAttention` produces the full directed ``(N, N)``
posterior.  It fuses the Hard-Concrete L0 selector of ``GatedCrossAttention``
with the directional Toeplitz parametrisation:

    raw    = <q_i, k_j> / sqrt(E)
    S_sym  = (raw + raw^T)/2   -> SYMMETRIC   HardConcrete existence gate  z
    A_anti = (raw - raw^T)/2   -> ANTISYMMETRIC coupled direction gate     d
    A_ij   = z_ij * d_ij * g_ij        (g = separate reconstruction gain)

so ``z`` is the sparse, thresholdable skeleton and ``d_ij + d_ji = 1`` per sample
(two-cycle suppression).  A node is *inferred* to be a source when its query-row
carries no surviving incoming edges.  The only structural prior is the
no-self-loop diagonal — the S -> X direction is never assumed.

Homogeneous embedding
=====================
Unlike ``AttentionSelectorLayer`` (separate ``embedding_S`` / ``embedding_X``),
this layer uses a SINGLE shared ``ModularEmbedding`` applied to the concatenated
``[S, X]`` tensor.  The forecaster is responsible for giving S and X a single
contiguous variable-id namespace (e.g. offsetting X ids by ``L_S``) so the shared
``nn_embedding`` table has one row per node.  The value stream (SVFA / summation)
behaves exactly as in the attention selector.

Evaluation
==========
* :meth:`split_attention` splits the ``(N, N)`` posterior into the four blocks
  ``S->X``, ``X->X``, ``X->S``, ``S->S`` for comparison against a ground-truth DAG.
* :meth:`source_scores` returns the per-node incoming-edge mass (row sum); a low
  value means the node is likely a source.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from causaliT.core.modules import (
    AttentionLayer,
    ModularEmbedding,
    Normalization,
    MLPHead,
    OrthogonalMaskEmbedding,
    FixedOrthonormalEmbedding,
)
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.core.modules.free_query_embedding import FreeQueryEmbedding


# Allowed values for ``struct_embedding_type`` (structural Q/K embedding scheme).
STRUCT_EMBEDDING_TYPES = (
    "standard_learnable",
    "orthogonal_learnable",
    "orthogonal_fixed",
)


class SelfSelectorLayer(nn.Module):
    """
    Homogeneous N-node self-attention causal discovery model.

    Args:
        model: Name string (stored for reference, e.g. "SelfSelectorLayer").
        ds_embed: Shared embedding config dict (passed to ModularEmbedding).
        comps_embed: Composition mode for the shared embedding
            ("summation" or "svfa").
        attention_type: Only "GatedSelfAttention" is supported.
        n_heads: Number of value heads (structure head is always single; requires
            shared_dag_across_heads=True).
        dropout_emb / dropout_attn_out / dropout_ff / dropout_qkv /
        attention_dropout: Dropout rates (mirror AttentionSelectorLayer).
        activation: FFN activation ("relu" or "gelu").
        norm: Normalisation type ("layer", "batch", ...).
        use_final_norm: Whether to apply a final norm after the FFN sublayer.
        device: Target device.
        out_dim: Output dimension per node (1 for scalar prediction).
        d_ff: FFN hidden dimension.
        d_model: Transformer hidden dimension.
        d_qk: Q/K projection dimension per head.
        S_seq_len / X_seq_len: Number of S / X variables.  N = S_seq_len+X_seq_len.
            Kept only so :meth:`split_attention` can carve the four GT blocks;
            the attention itself is fully homogeneous.
        init_tau: Hard-Concrete existence-gate temperature (beta).
        init_gamma / init_zeta: Hard-Concrete stretch bounds (gamma<0<1<zeta).
        dir_tau: Direction-gate (Binary-Concrete) temperature.
        gain_tau: Reconstruction-gain sigmoid temperature.
        output_mlp_layers / output_mlp_hidden / output_mlp_activation /
        output_mlp_dropout: Output MLP head configuration.
        shared_dag_across_heads: Must be True (GatedSelfAttention needs a single
            structural head).
        struct_embedding_type: Structural (Q/K) embedding scheme; one of
            STRUCT_EMBEDDING_TYPES.  Value stream is unchanged in all cases.
        orthogonal_fixed_frame_type / orthogonal_fixed_scale: Sub-options for
            struct_embedding_type="orthogonal_fixed".
        free_query_embedding: When True the QUERY structural stream gets its own
            free identity table (decoupling node-as-child from node-as-parent).
        key_projection_type / orthogonal_key_scale: Shared W_K projection type.
        batch_key_dropout / batch_key_dropout_p_final /
        batch_key_dropout_annealing_batches: Batch-consistent key dropout.
        optuna_protocol: Constant-score capacity override (gate-only).
        gain_stream_source: "separate" (own gain identity tables) or "shared"
            (reuse frozen structural embeddings; requires orthogonal_fixed).
    """

    def __init__(
        self,
        model: str,
        # Embedding config (single shared stream)
        ds_embed: dict,
        comps_embed: str,
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
        # Gate temperatures
        init_tau: float = 2.0 / 3.0,
        init_gamma: float = -0.1,
        init_zeta: float = 1.1,
        dir_tau: float = 2.0 / 3.0,
        gain_tau: float = 1.0,
        # MLP output head
        output_mlp_layers: int = 1,
        output_mlp_hidden: Optional[int] = None,
        output_mlp_activation: str = "relu",
        output_mlp_dropout: float = 0.0,
        # Multi-head semantics
        shared_dag_across_heads: bool = True,
        # Structural (Q/K) embedding scheme
        struct_embedding_type: str = "standard_learnable",
        orthogonal_fixed_frame_type: str = "random",
        orthogonal_fixed_scale: float = 1.0,
        # Decoupled query embedding
        free_query_embedding: bool = False,
        # Orthogonal (isometric) key projection
        key_projection_type: str = "linear",
        orthogonal_key_scale: bool = True,
        # Batch-consistent key dropout
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        # Constant-score capacity protocol (gate-only)
        optuna_protocol: Optional[float] = None,
        # Reconstruction-gain stream
        gain_stream_source: str = "separate",
    ):
        super().__init__()

        self.model_name = model
        self.d_model = d_model
        self.S_seq_len = S_seq_len
        self.X_seq_len = X_seq_len
        self.N = S_seq_len + X_seq_len
        self.comps_embed = comps_embed

        if attention_type != "GatedSelfAttention":
            raise ValueError(
                f"SelfSelectorLayer only supports attention_type="
                f"'GatedSelfAttention', got '{attention_type}'."
            )

        if not shared_dag_across_heads:
            raise ValueError(
                "SelfSelectorLayer requires shared_dag_across_heads=True "
                "(GatedSelfAttention uses a single structural head)."
            )

        # ------------------------------------------------------------------
        # Structural (Q/K) embedding scheme selection
        # ------------------------------------------------------------------
        if struct_embedding_type not in STRUCT_EMBEDDING_TYPES:
            raise ValueError(
                f"struct_embedding_type='{struct_embedding_type}' is invalid. "
                f"Must be one of: {list(STRUCT_EMBEDDING_TYPES)}."
            )
        self.struct_embedding_type = struct_embedding_type
        self.orthogonal_struct_embedding = (
            struct_embedding_type == "orthogonal_learnable"
        )
        self.orthogonal_fixed = (struct_embedding_type == "orthogonal_fixed")

        self.free_query_embedding = free_query_embedding
        self.gain_stream_source = gain_stream_source
        if gain_stream_source not in ("separate", "shared"):
            raise ValueError(
                f"gain_stream_source='{gain_stream_source}' is invalid. "
                f"Must be one of: 'separate', 'shared'."
            )
        if gain_stream_source == "shared" and not self.orthogonal_fixed:
            raise ValueError(
                "gain_stream_source='shared' requires "
                "struct_embedding_type='orthogonal_fixed' so the shared struct "
                "embeddings carry no structural gradient. "
                "Use gain_stream_source='separate' otherwise."
            )

        self.key_projection_type = key_projection_type
        if key_projection_type not in ("linear", "orthogonal"):
            raise ValueError(
                f"key_projection_type='{key_projection_type}' is invalid. "
                f"Must be one of: 'linear', 'orthogonal'."
            )
        if key_projection_type == "orthogonal" and d_qk < d_model:
            raise ValueError(
                f"key_projection_type='orthogonal' requires d_qk >= d_model to be "
                f"an isometry (W_K^T W_K = I), got d_qk={d_qk} < d_model={d_model}."
            )

        # ------------------------------------------------------------------
        # Shared embedding (single stream over all N nodes)
        # ------------------------------------------------------------------
        self.embedding = ModularEmbedding(
            ds_embed=ds_embed,
            comps=comps_embed,
            device=device,
        )

        # ------------------------------------------------------------------
        # Self-attention block (square N x N)
        # ------------------------------------------------------------------
        self.attention = AttentionLayer(
            attention=GatedSelfAttention,
            d_model_queries=d_model,
            d_model_keys=d_model,
            d_model_values=d_model,
            d_queries_keys=d_qk,
            n_heads=n_heads,
            mask_layer=None,
            attention_dropout=attention_dropout,
            dropout_qkv=dropout_qkv,
            register_entropy=True,
            layer_name="self_selector_att",
            query_seq_len=self.N,
            key_seq_len=self.N,
            init_tau=init_tau,
            shared_dag_across_heads=shared_dag_across_heads,
            key_projection_type=key_projection_type,
            orthogonal_scale=orthogonal_key_scale,
            batch_key_dropout=batch_key_dropout,
            batch_key_dropout_p_final=batch_key_dropout_p_final,
            batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
            optuna_protocol=optuna_protocol,
            init_gamma=init_gamma,
            init_zeta=init_zeta,
            dir_tau=dir_tau,
            gain_tau=gain_tau,
        )

        # ------------------------------------------------------------------
        # Static self-loop mask: (N, N) off-diagonal ones (no self-loops).
        # ------------------------------------------------------------------
        self.register_buffer(
            "self_loop_mask",
            1.0 - torch.eye(self.N),
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
        # Orthogonal structural embedding (optional Q/K stream override)
        # ------------------------------------------------------------------
        self.orth_embed: Optional[nn.Module]
        if self.orthogonal_struct_embedding:
            k = d_model // self.N
            if k <= 0:
                raise ValueError(
                    f"d_model={d_model} is too small for orthogonal structural "
                    f"embeddings with N={self.N} variables (need d_model >= {self.N})."
                )
            self.orth_embed = OrthogonalMaskEmbedding(
                num_variables=self.N,
                d_model=d_model,
                mask_start_dim=0,
                dims_per_var=k,
                freeze=False,
                device=device,
            )
        elif self.orthogonal_fixed:
            if self.N > d_model:
                raise ValueError(
                    f"struct_embedding_type='orthogonal_fixed' requires "
                    f"d_model >= N for mutually orthonormal rows, got "
                    f"d_model={d_model} < {self.N}."
                )
            frame_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            self.orth_embed = FixedOrthonormalEmbedding(
                num_variables=self.N,
                d_model=d_model,
                total_variables=self.N,
                row_offset=0,
                frame_type=orthogonal_fixed_frame_type,
                seed=frame_seed,
                scale=orthogonal_fixed_scale,
                device=device,
            )
        else:
            self.orth_embed = None

        # ------------------------------------------------------------------
        # Free query embedding (optional QUERY structural stream override)
        # ------------------------------------------------------------------
        self.query_embed: Optional[FreeQueryEmbedding]
        if free_query_embedding:
            self.query_embed = FreeQueryEmbedding(
                num_variables=self.N,
                d_model=d_model,
                device=device,
            )
        else:
            self.query_embed = None

        # ------------------------------------------------------------------
        # Gain-stream embeddings (separate mode).  Names start with "gain_"
        # so the gradient router classifies them as RECONSTRUCTION params.
        # ------------------------------------------------------------------
        self.gain_q_embed: Optional[FreeQueryEmbedding]
        self.gain_k_embed: Optional[FreeQueryEmbedding]
        if self.gain_stream_source == "separate":
            self.gain_q_embed = FreeQueryEmbedding(
                num_variables=self.N, d_model=d_model, device=device,
            )
            self.gain_k_embed = FreeQueryEmbedding(
                num_variables=self.N, d_model=d_model, device=device,
            )
        else:
            self.gain_q_embed = None
            self.gain_k_embed = None

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
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use forward_with_actual(all_blanked, all_actual) instead."
        )

    def forward_with_actual(
        self,
        all_blanked: torch.Tensor,
        all_actual: torch.Tensor,
        oracle: bool = False,
        oracle_mask: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass over the homogeneous N-node set.

        Args:
            all_blanked: (B, N, features) — all nodes with value column zeroed.
                Used as the QUERY input (variable-identity only).
            all_actual:  (B, N, features) — all nodes with real values.
                Used as the KEY/VALUE input.
            oracle: If True, use the GT adjacency (``oracle_mask``) as the
                structure gate; only the learned gain modulates magnitudes.
            oracle_mask: Optional (N, N) GT adjacency used when oracle=True.

        Returns:
            pred:  (B, N, out_dim)
            attn:  (B, N, N) directed structure posterior P(z>0)*d (masked)
            aux:   dict with "entropy" and "l0_penalty"
        """
        def _emb_drop(raw):
            if isinstance(raw, tuple):
                struct, val = raw
                return self.dropout_emb(struct), self.dropout_emb(val)
            return self.dropout_emb(raw), None

        k_struct, k_val = _emb_drop(self.embedding(X=all_actual))
        q_struct, q_val = _emb_drop(self.embedding(X=all_blanked))

        # ---- Orthogonal structural stream override -----------------------
        if self.orth_embed is not None:
            k_struct = self.dropout_emb(self.orth_embed(all_actual))
            q_struct = self.dropout_emb(self.orth_embed(all_blanked))

        # ---- Free query embedding override (QUERY stream only) -----------
        if self.free_query_embedding:
            assert self.query_embed is not None
            q_struct = self.dropout_emb(self.query_embed(all_blanked))

        # Q/K use the structural embedding; V uses the value embedding (SVFA)
        # or falls back to the structural embedding (summation mode).
        values = k_val if k_val is not None else k_struct

        # ---- Hard mask (allowed-edge topology) ---------------------------
        if oracle and oracle_mask is not None:
            hard_mask = oracle_mask
        else:
            hard_mask = self.self_loop_mask

        # ---- Reconstruction-gain stream ----------------------------------
        gain_query = None
        gain_key = None
        if self.gain_stream_source == "separate":
            assert self.gain_q_embed is not None and self.gain_k_embed is not None
            gain_query = self.dropout_emb(self.gain_q_embed(all_blanked))
            gain_key = self.dropout_emb(self.gain_k_embed(all_actual))

        attn_out, attn, aux = self.attention(
            query=q_struct,
            key=k_struct,
            value=values,
            mask_miss_k=None,
            mask_miss_q=None,
            pos=None,
            causal_mask=False,
            hard_mask=hard_mask,
            oracle=oracle,
            gain_query=gain_query,
            gain_key=gain_key,
        )

        # ---- Residual + Norm 1 -------------------------------------------
        is_svfa = q_val is not None
        if is_svfa:
            x = self.norm1(q_val + self.dropout_attn_out(attn_out))
        else:
            x = self.norm1(q_struct + self.dropout_attn_out(attn_out))

        # ---- FFN ---------------------------------------------------------
        x_ff = self.dropout_ff(self._act_fn(self.linear1(x)))
        x_ff = self.dropout_ff(self.linear2(x_ff))
        x = x + x_ff

        # ---- Norm 2 (optional) -------------------------------------------
        if self.use_final_norm:
            x = self.norm2(x)

        # ---- MLP head ----------------------------------------------------
        pred = self.forecaster(x)

        return pred, attn, aux

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def split_attention(self, attention_weights: torch.Tensor):
        """
        Split the ``(B, N, N)`` (or ``(N, N)``) directed posterior into four
        GT-comparable blocks, using the convention that entry ``[i, j]`` is the
        edge ``j -> i`` (node i attends to parent j).

        Nodes ``0 .. S-1`` are S; nodes ``S .. N-1`` are X.

        Returns dict with:
            "s_to_x": rows=X children, cols=S parents      (.., L_X, L_S)
            "x_to_x": rows=X children, cols=X parents       (.., L_X, L_X)
            "x_to_s": rows=S children, cols=X parents       (.., L_S, L_X)
            "s_to_s": rows=S children, cols=S parents       (.., L_S, L_S)
        """
        S = self.S_seq_len
        return {
            "s_to_x": attention_weights[..., S:, :S],
            "x_to_x": attention_weights[..., S:, S:],
            "x_to_s": attention_weights[..., :S, S:],
            "s_to_s": attention_weights[..., :S, :S],
        }

    def source_scores(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Per-node incoming-edge mass (sum over parents / columns).  A LOW value
        means the node has (almost) no parents and is therefore likely a
        SOURCE.  Accepts ``(B, N, N)`` or ``(N, N)`` and reduces over the last
        (parent) axis.
        """
        return attention_weights.sum(dim=-1)

    # ------------------------------------------------------------------
    # Gradient-routing parameter groups (identity-based)
    # ------------------------------------------------------------------

    def parameter_groups(self):
        """
        Partition trainable parameters into (structural, reconstruction) groups
        by MODULE REFERENCE — not by name — so the gradient router is robust to
        the ambiguous ``embed_modules_list.<i>`` naming of ``ModularEmbedding``
        and to the singular ``orth_embed`` / ``query_embed`` attribute names.

        Structural (theta_S) — everything that shapes the directed posterior via
        the Q/K gate score:
            * ``attention.query_projection`` / ``attention.key_projection``;
            * ``orth_embed`` (orthogonal structural key embedding, when enabled);
            * ``query_embed`` (free query identity table, when enabled);
            * the SVFA STRUCTURE embedding modules
              (``embedding.structure_modules_list``) — ONLY in ``comps_embed
              == "svfa"``.  In ``summation`` mode the single shared embedding
              feeds both Q/K and V, so it stays in the reconstruction group.

        Reconstruction (theta_R) — everything else: the reconstruction-gain
        stream (``gain_q_embed`` / ``gain_k_embed`` and the attention
        ``gain_*_proj``), ``value_projection`` / ``out_projection``, the value
        embedding modules, FFN, norms and the MLP head.

        Returns:
            (structural_params, reconstruction_params): two lists of
            ``nn.Parameter`` with ``requires_grad=True``.
        """
        structural_modules = [
            self.attention.query_projection,
            self.attention.key_projection,
        ]
        if self.orth_embed is not None:
            structural_modules.append(self.orth_embed)
        if self.query_embed is not None:
            structural_modules.append(self.query_embed)
        # SVFA structure stream feeds Q/K only -> structural.  In summation mode
        # the shared embedding also feeds V, so it belongs to reconstruction.
        if self.comps_embed == "svfa" and hasattr(
            self.embedding, "structure_modules_list"
        ):
            structural_modules.append(self.embedding.structure_modules_list)

        struct_ids = set()
        for module in structural_modules:
            for p in module.parameters():
                if p.requires_grad:
                    struct_ids.add(id(p))

        structural_params, reconstruction_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in struct_ids:
                structural_params.append(p)
            else:
                reconstruction_params.append(p)
        return structural_params, reconstruction_params

    # ------------------------------------------------------------------
    # Freezing utilities
    # ------------------------------------------------------------------

    def freeze_embedding(self):
        for p in self.embedding.parameters():
            p.requires_grad_(False)

    def unfreeze_embedding(self):
        for p in self.embedding.parameters():
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
