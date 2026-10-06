"""
AttentionSelectorLayer: Single cross-attention block for observational causal discovery.

Architecture
============

The model answers one focused question:
    "Can a single cross-attention operation learn to select the causal parents
     of each X variable from observational data, when given the actual values of
     all candidate parents (S and X) as keys?"

Three modes

**Split mode** (``homogeneous_nodes=False`` + ``self_attention_type`` set,
the default for the causal method) Ã¢â‚¬â€ the S -> X direction is ASSUMED: S nodes
appear only as keys, never as queries.  Two attention blocks:

    * ``attention``      Ã¢â‚¬â€ S -> X cross block  (``attention_type``, keys/values = S)
    * ``self_attention`` Ã¢â‚¬â€ X -> X self  block  (``self_attention_type``, keys/values = X)

Their outputs are summed into one value residual stream and their posteriors are
re-concatenated into the canonical ``(B, L_X, L_S + L_X)`` layout.

**Cross-only mode** (``self_attention_type=None``) Ã¢â‚¬â€ ONE combined block whose
keys/values are ``[S_actual ; X_actual]``, so a SINGLE softmax normalises over
the S and X parents JOINTLY (they compete on one simplex) and the X -> X columns
carry no direction gate.  This is the VANILLA-TRANSFORMER benchmark arm: paired
with ``attention_type="ScaledDotSoftmax"`` and ``comps_embed_X="summation"`` it
is a standard single-layer encoder block (softmax attention + residual + FFN),
with the only causal-discovery-specific ingredient being the zero diagonal that
forbids a node from copying itself.  It carries no sparsity/acyclicity score and
no value-structure injection, so the structure it reports is whatever plain
attention learns from the reconstruction loss alone.

**Homogeneous mode** (``homogeneous_nodes=True``) Ã¢â‚¬â€ the S/X prior is IGNORED.
The data stream is treated as a single homogeneous set of ``N = L_S + L_X``
nodes: every node is simultaneously a value-blanked query (candidate child) and
an actual-value key/value (candidate parent).  ONE square attention block Ã¢â‚¬â€
built from ``self_attention_type`` and stored in ``self.attention`` so all
downstream tooling (score sparsity, gradient routing, freezing, interference
probes) keeps working Ã¢â‚¬â€ produces the full directed ``(B, N, N)`` posterior, and
the regression head reconstructs ALL N variables (S included).  The cross
``attention_type`` is ignored entirely in this mode.  This subsumes
``SelfSelectorLayer`` while keeping every AttentionSelector feature (SVFA,
orthogonal/fixed structural embeddings, free queries, BKD, value-structure
injection, learnable query norms, ...).

Forward pass
------------

1. **Embed queries** Ã¢â‚¬â€ X tokens with blanked values.
   The value column is set to 0, so only the variable-identity embedding
   carries information. The query encodes "I am X_i Ã¢â‚¬â€ who are my parents?"

2. **Embed keys/values** Ã¢â‚¬â€ [S_actual, X_actual] concatenated.
   Both S and X tokens are embedded with their ACTUAL values.
   The key encodes "I am variable j and my value is v_j".

3. **Single cross-attention** with hard mask:
   - S block  (columns 0 .. L_S-1):  all ones  Ã¢â€ â€™ X_i may attend to any S_j
   - X block  (columns L_S .. L_S+L_X-1):  off-diagonal ones Ã¢â€ â€™ X_i may attend
     to any X_j Ã¢â€°Â  X_i (diagonal zeroed = no self-loops)

4. Residual + LayerNorm Ã¢â€ â€™ FFN Ã¢â€ â€™ LayerNorm Ã¢â€ â€™ MLP head.

In cross-only mode the single combined attention IS the causal structure.  The
resulting attention matrix
    A  Ã¢Ë†Ë†  Ã¢â€žÂ^{B Ãƒâ€” L_X Ãƒâ€” (L_S + L_X)}
contains the learned edge weights; splitting it gives
    A[:, :, :L_S]      Ã¢â€ â€™ SÃ¢â€ â€™X adjacency
    A[:, :, L_S:]      Ã¢â€ â€™ XÃ¢â€ â€™X adjacency  (diagonal = 0 by construction)

Comparison with existing architectures
---------------------------------------
SingleCausalLayer has:
  - cross-attention: Q from X_blanked, K/V from S_actual  (learns SÃ¢â€ â€™X)
  - self-attention:  Q/K/V from X_blanked_embedded         (learns XÃ¢â€ â€™X via embeddings only)

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
from typing import Dict, Optional

from causaliT.core.modules import (
    CausalCrossAttention,
    SigmoidCrossAttention,
    ScaledDotSoftmax,
    HardConcreteCrossAttention,
    GatedCrossAttention,
    GatedSelfAttention,
    CommutatorSelfAttention,
    AttentionLayer,
    ModularEmbedding,
    Normalization,
    MLPHead,
    OrthogonalMaskEmbedding,
    FixedOrthonormalEmbedding,
)
from causaliT.core.modules.topk_gate import TopKGate
from causaliT.core.modules.mlp_head import PerNodeMLPHead

# Imported directly from the submodule so the layer does not depend on the
# package ``__init__`` re-export being present.
from causaliT.core.modules.free_query_embedding import FreeQueryEmbedding
from causaliT.core.modules.value_identity_embedding import ValueIdentityEmbedding
from causaliT.utils.query_geometry import (
    assert_orthonormal_frame,
    correct_query,
    transitive_weights,
)
from causaliT.utils.query_norm import (
    DEFAULT_DIR_TAU,
    DEFAULT_GATE_GAMMA,
    DEFAULT_GATE_TAU,
    DEFAULT_GATE_ZETA,
)



# Allowed values for ``struct_embedding_type`` (structural Q/K embedding scheme).
STRUCT_EMBEDDING_TYPES = (
    "standard_learnable",
    "orthogonal_learnable",
    "orthogonal_fixed",
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
            "ScaledDotSoftmax". CausalCrossAttention (ReLU-Tanh) is recommended:
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
        init_tau: LEGACY shared temperature key.  Still the activation
            temperature of the non-gated attentions (CausalCrossAttention /
            SigmoidCrossAttention, default 3.0) and the shared fallback for the
            two Hard-Concrete existence gates.  Prefer the explicit split keys
            below (see docs/documentation/ATTENTION_TEMPERATURES.md).
        init_tau_cross: Hard-Concrete existence-gate temperature of the S->X
            cross block (GatedCrossAttention / HardConcreteCrossAttention).
            None -> ``init_tau`` -> the calculated default 0.5.
        init_tau_self: Hard-Concrete existence-gate temperature of the X->X
            self block (GatedSelfAttention / CommutatorSelfAttention; also the
            single square block in homogeneous mode).  None -> ``init_tau``
            -> 0.5.
        dir_tau_self: Antisymmetric direction-gate temperature of the self
            block.  None -> ``dir_tau`` (legacy name) -> 2/3 (Louizos et al.,
            ICLR 2018).
        output_mlp_layers: Number of layers in the output MLP head (1 = linear).
        output_mlp_hidden: Hidden dimension of the MLP head (None Ã¢â€ â€™ d_ff).
        output_mlp_activation: Activation in the MLP head.
        output_mlp_dropout: Dropout in the MLP head.
        shared_dag_across_heads: When True (default), a single (B,L,S) score is
            shared across all value heads (SVFA-style). When False, each head
            has its own independent DAG score.
        struct_embedding_type: Selects the structural (Q/K) embedding scheme for
            both S and X.  One of:

            * ``"standard_learnable"`` (default) Ã¢â‚¬â€ use the standard
              ``ModularEmbedding`` (learnable ``nn_embedding``) for the
              structural stream (original behaviour).
            * ``"orthogonal_learnable"`` Ã¢â‚¬â€ replace the structural stream (Q/K)
              embeddings for both S and X with ``OrthogonalMaskEmbedding``
              instances whose partitions tile the full d_model space without
              overlap:

                  S occupies dims [0,           S_seq_len * k)
                  X occupies dims [S_seq_len*k, (S_seq_len+X_seq_len)*k)

              where k = d_model // (S_seq_len + X_seq_len).  The value stream
              (V, residual, MLP head) continues to use the standard
              ``ModularEmbedding``.
            * ``"orthogonal_fixed"`` Ã¢â‚¬â€ replace the structural stream (Q/K)
              embeddings for both S and X with ``FixedOrthonormalEmbedding``
              instances.  Unlike ``"orthogonal_learnable"`` (disjoint binary
              blocks, so each variable lives on ``d_model // n_vars``
              axis-aligned dimensions with the remainder idle), this uses
              **dense frozen rows spanning the FULL d_model space** that are
              still mutually orthonormal (S and X share ONE frame via disjoint
              row slices, so all L_S + L_X rows are pairwise orthogonal).  It is
              value-independent (identity only), so the actual value must reach
              the output through the SVFA value (V) stream.  Requires
              ``d_model >= S_seq_len + X_seq_len``.

            In all cases the value stream (V, residual, FFN, MLP head) is
            unchanged.  Default ``"standard_learnable"``.
        orthogonal_fixed_frame_type: Frame construction for
            ``struct_embedding_type="orthogonal_fixed"``:
            ``"random"`` (default) Ã¢â‚¬â€ QR of a Gaussian matrix, seeded from the
            global training seed so it varies per run; ``"dct"`` Ã¢â‚¬â€ deterministic
            DCT-II basis rows (seed-independent).
        orthogonal_fixed_scale: Scalar norm applied to every fixed orthonormal
            row (default 1.0).  Rescales the rows without breaking orthogonality.
        free_query_embedding: When True, decouple the X **query** from the X
            **key** by giving the query its own free (unconstrained) learnable
            identity embedding (``FreeQueryEmbedding``).  Rationale: in the
            combined attention each X_i is used BOTH as a key (offered as a
            parent to other X_j) and as a query (selecting its own parents).
            With a single shared embedding, updating "X_i-as-child" also
            perturbs "X_i-as-parent", so ``X_i Ã¢â€ Â S`` and ``X_i Ã¢â€ Â X_j`` cannot be
            learned independently.  A separate query embedding removes this
            coupling.  Works with any ``struct_embedding_type``:
            the X KEY stream keeps whatever embedding is configured (orthogonal
            or standard), only the X QUERY structural stream is overridden.
            The value stream (V, residual, FFN, MLP head) is UNCHANGED.
            Default False (original behaviour).
        key_projection_type: Type of the shared W_K key projection inside the
            attention.  ``"linear"`` (default) is an unconstrained ``nn.Linear``.
            ``"orthogonal"`` constrains W_K to an isometry (``W_K^T W_K = I``)
            via ``OrthogonalLinear`` (Cayley parametrisation).  Because an
            isometry preserves inner products, orthogonal raw keys (e.g. from
            ``OrthogonalMaskEmbedding``) stay orthogonal AFTER projection:
            ``<k_i W_K, k_j W_K> = <k_i, k_j>``.  Requires ``d_qk >= d_model``.
            Most meaningful with ``struct_embedding_type="orthogonal_learnable"``
            and ``n_heads=1``.
        orthogonal_key_scale: When ``key_projection_type="orthogonal"``, whether
            the ``OrthogonalLinear`` includes a learnable scalar scale factor
            (default True).  Ignored for ``"linear"``.
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
        batch_key_dropout_min_keys: Minimum number of keys kept per forward
            (safeguard against a completely empty adjacency).  ``0``
            (default) disables the safeguard; ``1`` guarantees at least one
            key survives even at very high drop probabilities.
        batch_key_dropout_deterministic: If ``True``, keep a deterministic
            number of keys ``max(min_keys, round((1 - p) * S))`` drawn as a
            uniform random subset, instead of per-key Bernoulli sampling.
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
        # Attention temperatures (harmonized split keys; the legacy shared
        # ``init_tau`` key has been REMOVED).
        init_tau_cross: Optional[float] = None,
        init_tau_self: Optional[float] = None,
        # MLP output head
        output_mlp_layers: int = 1,
        output_mlp_hidden: Optional[int] = None,
        output_mlp_activation: str = "relu",
        output_mlp_dropout: float = 0.0,
        # Multi-head semantics
        shared_dag_across_heads: bool = True,
        # Structural (Q/K) embedding scheme.  One of STRUCT_EMBEDDING_TYPES:
        #   "orthogonal_fixed"     Ã¢â€ â€™ FixedOrthonormalEmbedding (dense frozen frame;
        #                          the DEFAULT for the SVFA stack).
        #   "orthogonal_learnable" Ã¢â€ â€™ OrthogonalMaskEmbedding (disjoint blocks)
        #   "standard_learnable"   Ã¢â€ â€™ ModularEmbedding (nn_embedding; baseline)
        struct_embedding_type: str = "orthogonal_fixed",
        # Sub-options for struct_embedding_type="orthogonal_fixed".
        orthogonal_fixed_frame_type: str = "random",
        orthogonal_fixed_scale: float = 1.0,
        # Decoupled key/query embedding for X
        free_query_embedding: bool = True,
        # Initialise the free X query embedding at the centroid of the (projected)
        # keys so every query starts from the SAME point and reads all candidate
        # parents uniformly (see init_query_at_key_centroid).  Requires
        # free_query_embedding=True.  The actual write happens lazily on the first
        # training batch (the forecaster calls the method), since value-modulated
        # key embeddings need real data.
        query_centroid_init: bool = True,
        # Known-edges prior: AFTER the initialisation (centroid or default),
        # overwrite the query of the listed nodes with the centroid of their
        # KNOWN parents'' (projected) keys, optionally keeping those rows frozen
        # for the whole run.  Format (GLOBAL 1-based dataset node IDs; S nodes
        # are 1..S_seq_len, X nodes are S_seq_len+1..S_seq_len+X_seq_len):
        #     {5: {"parents": [1, 2], "fixed": True}, ...}
        # Requires free_query_embedding=True.  In split mode only X nodes
        # (id > S_seq_len) are queries; S children need homogeneous_nodes.
        query_parents_prior: Optional[Dict[int, dict]] = None,
        # Source-nodes prior: set the query of the listed nodes to ZERO and
        # freeze it, so a source can never be a child.  With a zero query the
        # node's raw score row is exactly 0, so the direction of every
        # incident edge is decided ENTIRELY by the other endpoint's score:
        # A_anti[s, j] = -raw[j, s]/2 gives d[s <- j] = sigmoid(-raw[j, s]/2b)
        # and d[j -> s] = 1 - d[s <- j] -- whenever a child scores the source
        # positively, ALL direction mass flows source -> child.  Format (GLOBAL
        # 1-based dataset node IDs, same convention as query_parents_prior):
        #     {1: {"mask": True}, ...}
        # "mask" (DEFAULT True) additionally zeroes the source ROWS (incoming
        # edges) of the hard mask: a zero SCORE is not a zero POSTERIOR (the
        # sigmoid existence gate stays open at logit 0), so the mask makes
        # the no-incoming-edge constraint exact and removes the constant L0
        # contribution of the source rows.  Requires free_query_embedding=
        # True; mutually exclusive with query_parents_prior on the same node.
        # Applied by init_source_queries_zero AFTER the centroid/parents
        # initialisation; frozen rows are re-asserted after every optimizer
        # step (reassert_frozen_query_rows).  Incompatible with
        # centroid_commit (commit events would rewrite the frozen rows).
        # In split mode only X nodes (id > S_seq_len) are queries; S children
        # need homogeneous_nodes.
        query_source_prior: Optional[Dict[int, dict]] = None,
        # Forbidden-parents prior (orthogonal key exclusion): keep the query of
        # the listed nodes ORTHOGONAL to their forbidden parents' fixed
        # structural keys (e.g. parents whose observation comes in the future).
        # Unlike an adjacency mask Ã¢â‚¬â€ which lets the query waste alignment
        # budget on forbidden keys and only blanks the downstream score Ã¢â‚¬â€ this
        # removes the forbidden key directions from the query span directly:
        # on every forward the projected query is replaced by
        #     q <- q - K_f (K_f^T K_f)^{-1} K_f^T q,
        # where K_f stacks the forbidden parents' rows of the fixed orthonormal
        # frame (a detached buffer, so the chain rule automatically projects
        # the embedding-row gradient into the allowed subspace Ã¢â‚¬â€ GPM/INLP-style
        # null-space projection, NO penalty term).  Format (GLOBAL 1-based
        # dataset node IDs, same convention as query_parents_prior):
        #     {5: {"forbidden": [1, 2], "mask": True}, ...}
        # "mask" (DEFAULT True) additionally zeroes the forbidden parent
        # columns of the hard mask: a zero SCORE is not a zero POSTERIOR
        # (sigmoid gates stay open at logit 0), so the geometry constraint and
        # the mask blanking are complementary.  mask=False applies the pure
        # orthogonal projection only (ablation arm).  Requires
        # free_query_embedding=True, struct_embedding_type="orthogonal_fixed"
        # and remove_query_projection=remove_key_projection=True (so the
        # embedding space IS the QK scoring space and the projection is exact).
        # In split mode only X nodes (id > S_seq_len) are queries; S children
        # need homogeneous_nodes.
        query_forbidden_prior: Optional[Dict[int, dict]] = None,
        # Automatic SELF-key exclusion: project each childs OWN structural
        # key direction out of its query (the diagonal of every structural hard
        # mask is already zero, so self-alignment can never be used downstream
        # -- removing it from the query span stops the query from wasting
        # alignment budget on its own key).  Applied to ALL children (X nodes
        # in split/cross-only mode; every node in homogeneous mode),
        # independently of query_forbidden_prior, via the same fixed null-space
        # projector machinery (the self frame row is unioned with the forbidden
        # rows).  Same requirements: free_query_embedding=True,
        # struct_embedding_type="orthogonal_fixed" and
        # remove_query_projection=remove_key_projection=True.
        # NOTE: the library default is False for backwards compatibility; the
        # experiment template sets query_exclude_self=true (the sanctioned
        # orthogonal_fixed stack meets all requirements).
        query_exclude_self: bool = False,
        # Orthogonal (isometric) key projection: W_K^T W_K = I

        key_projection_type: str = "linear",
        orthogonal_key_scale: bool = True,
        # Batch-consistent key dropout
        batch_key_dropout: Optional[float] = None,
        batch_key_dropout_p_final: Optional[float] = None,
        batch_key_dropout_annealing_batches: Optional[int] = None,
        batch_key_dropout_min_keys: int = 0,
        batch_key_dropout_deterministic: bool = False,
        # Optuna capacity-search protocol (constant-score override). See
        # CausalCrossAttention: None disables; 0 = residual-only floor;
        # 1 = uniform mixing (pair with heavy batch_key_dropout).
        optuna_protocol: Optional[float] = None,
        # The reconstruction-gain stream (use_gain / gain_tau / gain_stream_source)
        # has been REMOVED from the gated attentions: A = structure gate.
        init_gamma: float = DEFAULT_GATE_GAMMA,
        init_zeta: float = DEFAULT_GATE_ZETA,
        # Additive logit offset on the SÃ¢â€ â€™X cross existence gate ONLY, to balance
        # its initialization against the directed XÃ¢â€ â€™X self edge (which the
        # undecided direction gate halves: P = p_existÃ‚Â·0.5).  The cross
        # posterior at the centroid init is
        # P = sigmoid(x Ã¢Ë†â€™ init_edge_offset Ã¢Ë†â€™ ÃŽÂº).  The config value is resolved
        # at data-load time by
        # causaliT.utils.query_norm.resolve_init_edge_offset: "auto" picks the
        # MATCHED offset ln(e^(xÃ¢Ë†â€™ÃŽÂº) + 2) so the cross gate starts at the
        # DIRECTED self posterior (directed-level balance); a float pins a
        # legacy ablation; 0.0 disables it (existence-level balance).  The
        # offset NEVER enters the query_fanin_scale (F) derivation Ã¢â‚¬â€ F targets
        # the offset-free existence posterior of BOTH gates.  Only
        # GatedCrossAttention consumes it; the self block is never offset.
        init_edge_offset: float = 0.0,
        # Direction-aware self-attention block.  MANDATORY (the legacy
        # cross-only variant, where the right-hand columns of one combined cross
        # block modelled XÃ¢â€ â€™X without direction, has been REMOVED).
        #   * homogeneous_nodes=False (split): SÃ¢â€ â€™X via the cross
        #     ``attention_type`` block (keys/values = S) + XÃ¢â€ â€™X via this
        #     direction-aware block (keys/values = X).  Outputs are SUMMED into
        #     one value residual stream; posteriors re-concatenated.
        #   * homogeneous_nodes=True: THIS is the only block Ã¢â‚¬â€ a square (N, N)
        #     attention over all nodes; ``attention_type`` is ignored.
        # Default "GatedSelfAttention" (the SVFA stack); None selects the
        # cross-only vanilla-transformer benchmark arm.
        self_attention_type: Optional[str] = "GatedSelfAttention",
        # Homogeneous N-node mode: ignore the S/X (source) prior entirely.  The
        # whole datastream [S ; X] becomes one set of N = L_S + L_X nodes, each
        # simultaneously a value-blanked QUERY (candidate child) and an
        # actual-value KEY/VALUE (candidate parent).  ONE square self-attention
        # block (``self_attention_type``, stored in ``self.attention``) yields
        # the full directed (B, N, N) posterior and the regression head
        # reconstructs ALL N variables (S included).  Requires
        # ``self_attention_type``; forbids ``shared_query`` / ``shared_key``
        # (there is a single block, so sharing is intrinsic).  Default False.
        homogeneous_nodes: bool = False,
        # ---- Geometric TRANSITIVE correction (grandparent suppression) ----
        # Opt-in.  Pass 1 probes the DETACHED posterior; every edge j -> i that
        # is already explained by a mediator (j -> k -> i) has its key component
        # removed from the query in pass 2, and the freed query-norm budget is
        # handed back to the surviving parents by ``normalize_query``.  See
        # causaliT/utils/query_geometry.py and
        # experiments/6_INVESTIGATIONS/HOMOGENEOUS/TRANSITIVE_CORRECTION.md.
        # Works in BOTH node topologies:
        #   * homogeneous: the single square (N, N) block probes its own
        #     posterior inside AttentionLayer.forward (the mediator axis exists
        #     inside the block);
        #   * split: the LAYER probes BOTH blocks (AttentionLayer.
        #     transitive_probe), fuses the (L_X, L_S+L_X) posterior, pads it to
        #     the square (N, N) graph with zero S-rows (the descendant_mask
        #     convention Ã¢â‚¬â€ every mediator is an X node there), computes the
        #     weights once and scatters the per-block slices; symmetrisation is
        #     limited to the square X-X self block (symmetric_span) and the
        #     ORTHONORMALITY check runs on the COMBINED [S ; X] key frame.
        # Requires an ORTHONORMAL key frame (orthogonal_fixed +
        # remove_key_projection) Ã¢â‚¬â€ enforced loudly on the first forward.
        transitive_correction: bool = False,
        transitive_alpha: float = 0.5,      # strength; 1.0 starts eating true edges
        transitive_delta: float = 0.5,      # target COSINE on the negative side
        transitive_tnorm: str = "min",      # Goedel; "prod" deflates and barely fires
        transitive_margin: bool = True,     # relu(m - Pi): silent at the all-on init
        transitive_symmetric: bool = True,  # existence part is symmetric
        # Direction-gate Binary-Concrete temperature of the self-attention
        # antisymmetric term.  ``dir_tau`` is the LEGACY name (fallback);
        # ``dir_tau_self`` wins when set; both None -> DEFAULT_DIR_TAU (2/3).
        dir_tau: Optional[float] = None,
        dir_tau_self: Optional[float] = None,
        # Additive direction-gate logit bias for the direction-aware block
        # (GatedSelfAttention) at CONSTRUCTION (default 0.0 = legacy coupled
        # gate).  Applies to the homogeneous single block AND the split-mode
        # self block.  Runtime/phase control goes through set_dir_bias (the
        # adaptive trainer phase controller, per-phase ``dir_bias`` keys).
        dir_bias: float = 0.0,
        # Transposed-score gradient routing in the Toeplitz split
        # (GatedSelfAttention only).  True (default) = legacy detached
        # transpose (no cross-row query gradient); False keeps the transpose
        # in the graph, so a row's loss can also PUSH the antagonist query
        # (balanced push/pull on the direction gate).
        detach_transpose: bool = True,
        # Centroid-collapse fix (GatedCrossAttention / GatedSelfAttention only):
        # L2-normalise the STRUCTURAL query before scoring and replace the
        # 1/sqrt(E) score scale with a fixed sqrt(query_fanin_scale).  This makes
        # the query DIRECTION (not its unbounded norm) drive parent selection,
        # removing the "lazy" centroid-alignment block shortcut.  Set
        # ``query_fanin_scale`` to the max in-degree you want cleanly
        # representable.  Only affects the structure gate, never the value /
        # reconstruction-gain streams.  Threaded into BOTH the SÃ¢â€ â€™X cross block
        # and (when split) the XÃ¢â€ â€™X self block.
        normalize_query: bool = False,
        query_fanin_scale: float = 1.0,
        # Learnable per-node query-norm multiplier (see causaliT.utils.query_norm).
        # When enabled, each child owns ``M_i = exp(log_scale_i)`` (init
        # ``query_norm_init_scale``) scaling its unit query so it can ADAPTIVELY
        # overspend the directional budget whenever the structural signal pays
        # for it; the structural loss charges ``relu(M_i - query_norm_target)^2``.
        # Threaded into BOTH the S->X cross block and (when split) the X->X self
        # block; under ``shared_query=True`` a single multiplier is TIED across
        # the two blocks.  Disabled (or init_scale=1.0) == plain unit-norm cap.
        query_norm_learnable: bool = False,
        query_norm_init_scale: float = 1.0,
        query_norm_target: float = 1.0,
        # Value-structure injection: combine a per-source-node identity code
        # with the (data-only) value stream before W_V, so the model can learn
        # a per-source-node functional.  One of:
        #   "none"           Ã¢â‚¬â€ disabled (default, original data-only value).
        #   "separate"       Ã¢â‚¬â€ dedicated reconstruction-routed identity tables
        #                      (val_id_embed_S / val_id_embed_X), CONCATENATED:
        #                      V_j = W_V([v_j ; e_j]) (W_V widened by d_model).
        #   "struct_detached"Ã¢â‚¬â€ reuse the structural identity embeddings, detached
        #                      before concat (zero new params, no gradient leak).
        #   "learned_sum"    Ã¢â‚¬â€ dedicated reconstruction-routed identity tables
        #                      (val_id_embed_S / val_id_embed_X; vanilla-style
        #                      nn.Embedding with max_norm=1), SUMMED onto the
        #                      value token before W_V: V_j = W_V(v_j + e_j).
        #                      This is the vanilla/benchmark value composition
        #                      (token = value map + learnable node embedding);
        #                      W_V is NOT widened.
        # Requires SVFA (comps_embed_X="svfa").
        value_structure_injection: str = "none",
        # Value-structure QUERY injection: make the value additionally depend on
        # the QUERY (child) node identity, so the shared W_V learns DIFFERENT
        # functionals of the same parent depending on which child it feeds (e.g.
        # X2 predicting X4 vs X5).  Same option set / SVFA requirement as
        # ``value_structure_injection``; combination is an additive query term.
        value_structure_query_injection: str = "none",
        # Shared structural query projection (W_q) across the SÃ¢â€ â€™X cross block and
        # the XÃ¢â€ â€™X self block (split mode only).  When True, the self block does
        # NOT build its own ``query_projection``; instead the cross block's W_q is
        # applied to the X structural embedding to produce the self block's query
        # (fed via ``query_external=True``).  This ties "how a child reads its
        # candidate parents" to a SINGLE projection regardless of whether the
        # parent is an S or an X node Ã¢â‚¬â€ the SÃ¢â€ â€™X and XÃ¢â€ â€™X selectors then share one
        # query geometry.  No effect unless ``self_attention_type`` is set.
        shared_query: bool = False,
        # Shared structural KEY projection (W_K) across the SÃ¢â€ â€™X cross block and
        # the XÃ¢â€ â€™X self block (split mode only).  When True, the self block does
        # NOT build its own ``key_projection``; instead the cross block's W_K is
        # applied to the X structural embedding to produce the self block's key
        # (fed via ``key_external=True``).  Motivation: with a fixed orthonormal
        # struct embedding, the cross W_K is an isometry, so S keys and X keys
        # projected through the SAME W_K stay mutually orthogonal Ã¢â‚¬â€ the shared
        # free query then aligns on genuinely orthonormal key axes for BOTH the
        # S and X subspaces, removing the cheap spurious XÃ¢â‚¬â€œX edges that flexible,
        # per-block non-orthonormal self keys made possible.  Edge DIRECTION is
        # still resolved by CommutatorSelfAttention's skew-query generator on the
        # shared query alone (direction_mode="skew_query"), which keeps the Lie
        # commutator valid even though query and key come from different
        # embeddings.  No effect unless ``self_attention_type`` is set.
        shared_key: bool = False,

        # Remove the structural query/key projections entirely (read Q/K straight
        # from the embeddings, dropping W_q / W_K) and/or freeze them at init.
        # Threaded into BOTH the S->X cross block and (when split) the X->X self
        # block; ignored on the self block when it borrows the cross projection
        # via shared_query / shared_key (the shared projection then lives in Ã¢â‚¬â€ and
        # is removed/frozen on Ã¢â‚¬â€ the cross block).  Motivation: the shared query
        # projection couples every query and aligns them to the keys, numbing the
        # per-node query embeddings (see the SELF_ATTENTION spurious-S3->X4
        # investigation).  ``remove_*`` requires d_qk == d_model so the raw
        # embedding width matches the score width.  ``freeze_*`` merely sets
        # requires_grad=False (the freeze persists across adaptive phase switches
        # because the gradient router keeps only requires_grad=True params).
        remove_query_projection: bool = True,
        remove_key_projection: bool = True,
        freeze_query_projection: bool = False,
        freeze_key_projection: bool = False,

        # CommutatorSelfAttention direction-gate parametrisation (only used when

        # self_attention_type="CommutatorSelfAttention").  "qk" (default) uses
        # the antisymmetric part of the raw XÃ¢â€ â€™X alignment Ã‚Â½(QKÃ¡Âµâ‚¬Ã¢Ë†â€™KQÃ¡Âµâ‚¬) as the
        # direction score Ã¢â‚¬â€ a valid so(N) commutator ONLY when Q and K share the
        # same embedding.  "skew_query" instead learns a genuine so(d) generator
        # ÃŽÂ© on the QUERY alone (A_anti_ij = q_iÃ¡Âµâ‚¬ ÃŽÂ© q_j), so edge direction stays
        # a valid Lie generator even when the free shared query and the fixed
        # orthonormal key come from DIFFERENT embeddings Ã¢â‚¬â€ the intended pairing
        # with shared_query=True.  ``commutator_direction_rank`` sets the rank of
        # ÃŽÂ© (defaults to full rank = d_qk).
        commutator_direction_mode: str = "qk",
        commutator_direction_rank: Optional[int] = None,

        # ---- Prior-softmax reconstruction gain (ATE_FIXCAP design) --------
        # Adds the GainSoftmax stage to the gated blocks (see
        # causaliT/core/modules/gain_softmax.py): the applied weight becomes
        # A = (1-lambda)*z + lambda*n*z*e^s/D Ã¢â‚¬â€ the gate stays the sole owner
        # of the support (z=0 -> A=0 exactly), the gain score s only
        # REDISTRIBUTES each row's gate mass within the support.  The score is
        # s = a_ij (zero-init static per-edge logit) + an optional
        # DATA-dependent term <q^v, k^v>/sqrt(d_g) built from the
        # value-identity tables and the value stream Ã¢â‚¬â€ NEVER the structural
        # embeddings (rule: the gain carries no structural signal).  lambda is
        # a buffer ramped by the trainer (``set_gain_lambda``); at lambda=0
        # (default) the model is bit-identical to the gate-only baseline.
        # Requires the gated blocks (split mode: GatedCrossAttention +
        # GatedSelfAttention; homogeneous: GatedSelfAttention) and, for
        # ``gain_data=True``, the "separate"/"learned_sum" value-identity
        # tables on both the source and the query side.
        use_gain_softmax: bool = False,
        gain_data: bool = True,
        gain_score_dim: Optional[int] = None,
        # Per-node output head (DAGMA-style): one independent decoder MLP per
        # variable.  When True, ``self.forecaster`` is a ``PerNodeMLPHead``
        # instead of the shared ``MLPHead``.  Requires the variable-ID column
        # at index 1 in the input tensors (production convention).
        per_node_output: bool = False,
        per_node_output_hidden: int = 32,
        # Raw-value adjacency passthrough (NOTEARS/DAGMA-style, Arm B): when
        # True, the value encoder / W_V / value residual stream are bypassed
        # and the gated attention posterior acts DIRECTLY on the raw parent
        # values: decoder MLP i receives Z_i = (A_i1*x_1, ..., A_iN*x_N) Ã¢â‚¬â€
        # the locally-connected first layer of NOTEARS-MLP / DAGMA-MLP.
        # Requires per_node_output=True.  The value embeddings still run
        # (their output feeds the now-unused attention V path) but receive no
        # reconstruction gradient.
        raw_value_adjacency: bool = False,
        # Adjacency-context injection into the per-node decoder MLP.  Modes:
        # False/None = off; True or "concat" = each per-node MLP input is
        # concatenated with the DETACHED applied-adjacency row (B, L_q,
        # L_S+L_X) Ã¢â‚¬â€ the gate weights actually used on this sample after
        # hard mask, gain, BKD, top-k blanking and dropout Ã¢â‚¬â€ so the
        # (nuisance) regressor knows which keys were selected, consistently
        # with stochastic key exclusion.  "film" = FiLM conditioning: a
        # shared conditioner MLP maps the context to per-channel scale/
        # shift of the decoder hidden activation (zero-init => identical
        # to the unconditioned decoder at step 0), without widening the
        # MLP input.  Requires per_node_output=True.
        per_node_adjacency_context: bool = False,
        # Number of linear layers in each per-node decoder MLP (>=2).  2
        # (default) is the legacy shallow head; deeper heads pair with FiLM
        # conditioning at EVERY hidden activation (multi-layer FiLM, Perez
        # et al. 2018) when per_node_adjacency_context == "film".
        per_node_output_layers: int = 2,
        # Eval-mode BKD (rung-aware validation): forwarded to the gated
        # attention blocks; when True, eval forward passes apply the current
        # BKD budget with a seeded generator so val metrics measure the same
        # key-budget regime as training (fixes the min-keys-ladder train/eval
        # mismatch where validation ran with the full dense gate).
        batch_key_dropout_eval: bool = False,
        # Source-side top-k budget (TopKGate; see
        # causaliT/core/modules/topk_gate.py).  ``topk_k=None`` (default)
        # disables it and keeps the dense behaviour bit-identical.  When set,
        # one gate is built per gated block and handed to the inner attention:
        # the square self block (and the homogeneous combined block) excludes
        # the diagonal; the bipartite S->X cross block does not.  Supported
        # for GatedCrossAttention / GatedSelfAttention only.
        topk_k: Optional[int] = None,
        topk_method: str = "noisy_hard_k",
        topk_slack_init: Optional[int] = None,
        topk_slack_final: int = 0,
        topk_annealing_batches: Optional[int] = None,
        topk_per_row_slack: bool = True,
        batch_key_dropout_eval_seed: int = 12345,
    ):







        super().__init__()


        self.model_name = model
        self.d_model = d_model
        self.S_seq_len = S_seq_len
        self.X_seq_len = X_seq_len
        # Store embedding composition mode so callers (and diagnostics) can
        # inspect whether the model is operating in SVFA or standard mode.
        self.comps_embed_X = comps_embed_X

        # Homogeneous N-node mode (no S/X prior); N = L_S + L_X.
        self.homogeneous_nodes = bool(homogeneous_nodes)

        # ---- Transitive correction (see the ctor docstring above) ---------
        self.transitive_correction = bool(transitive_correction)
        self.transitive_alpha = float(transitive_alpha)
        self.transitive_delta = float(transitive_delta)
        self.transitive_tnorm = str(transitive_tnorm)
        self.transitive_margin = bool(transitive_margin)
        self.transitive_symmetric = bool(transitive_symmetric)
        # Config bundle handed to the block on every forward (None = disabled).
        self._transitive_cfg = (
            {
                "alpha": self.transitive_alpha,
                "delta": self.transitive_delta,
                "tnorm": self.transitive_tnorm,
                "margin": self.transitive_margin,
                "symmetric": self.transitive_symmetric,
            }
            if self.transitive_correction
            else None
        )
        if (
            self.transitive_correction
            and not self.homogeneous_nodes
            and self_attention_type is None
        ):
            # CROSS-ONLY is the only unsupported topology: ONE bipartite
            # (L_X, L_S+L_X) block has no mediator axis (a node that is both
            # parent and child) and there is no X->X self block to supply it.
            # Homogeneous probes its own square posterior inside the block;
            # split is orchestrated by this layer's forward (two-pass over the
            # re-fused posterior Ã¢â‚¬â€ see forward_with_actual).
            raise ValueError(
                "transitive_correction=True with homogeneous_nodes=False "
                "requires self_attention_type (the split cross + self "
                "topology): the mediated paths span the two blocks.  The "
                "cross-only single block has no X->X self block and is not "
                "supported."
            )
        if (
            self.transitive_correction
            and not self.homogeneous_nodes
            and not shared_query
        ):
            # The fused trigger mixes the cross and self posteriors of the SAME
            # child, and the cross-freed norm budget must renormalise the query
            # the self block scores (the joint-cap reallocation; see
            # TRANSITIVE_CORRECTION.md F6).  Both require ONE shared query
            # geometry; shared_query=False gives the self block its own
            # Q = W_q(xk_struct), a different child representation.
            raise ValueError(
                "transitive_correction=True in split mode requires "
                "shared_query=True (ONE query geometry scored by both "
                "blocks).  Without it the fused (L_X, L_S+L_X) posterior mixes "
                "two different child representations and the cross-block "
                "correction cannot be chained into the self block's query."
            )
        # Split-mode orchestration state: the COMBINED [S ; X] key-frame Gram
        # is checked once (the frame is fixed by construction), and
        # ``last_transitive_W`` stashes the fused (L_X, L_S+L_X) weights for
        # diagnostics (the applied per-block slices live on the two
        # AttentionLayer wrappers' own ``last_transitive_W``).
        self._transitive_frame_checked = False
        self.last_transitive_W: Optional[torch.Tensor] = None
        self.N = S_seq_len + X_seq_len

        # ------------------------------------------------------------------
        # Structural (Q/K) embedding scheme selection.
        # A single key selects among three mutually-exclusive schemes; see the
        # class docstring for ``struct_embedding_type``.  Convenience booleans
        # ``orthogonal_struct_embedding`` / ``orthogonal_fixed`` are derived from
        # it for readability in the forward pass and diagnostics.
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
        self.query_centroid_init = bool(query_centroid_init)
        if self.query_centroid_init and not free_query_embedding:
            raise ValueError(
                "query_centroid_init=True requires free_query_embedding=True "
                "(there is no dedicated X query embedding table to initialise "
                "at the key centroid otherwise)."
            )
        # Known-edges query prior (applied by init_queries_from_parents on the
        # first training batch, AFTER the centroid/default init).  Normalise to
        # int keys and validate the GLOBAL 1-based dataset node IDs.
        self.query_parents_prior: Optional[Dict[int, dict]] = None
        if query_parents_prior:
            if not free_query_embedding:
                raise ValueError(
                    "query_parents_prior requires free_query_embedding=True "
                    "(there is no dedicated query embedding table to "
                    "initialise otherwise)."
                )
            _L = S_seq_len + X_seq_len
            _norm: Dict[int, dict] = {}
            for _child, _spec in query_parents_prior.items():
                _c = int(_child)
                _parents = [int(p) for p in _spec.get("parents", [])]
                _fixed = bool(_spec.get("fixed", False))
                if not 1 <= _c <= _L:
                    raise ValueError(
                        f"query_parents_prior: child id {_c} out of range "
                        f"1..{_L}."
                    )
                if _c <= S_seq_len and not self.homogeneous_nodes:
                    raise ValueError(
                        f"query_parents_prior: child id {_c} is an S node; in "
                        "split mode only X nodes (ids > S_seq_len) have a "
                        "query to initialise."
                    )
                if not _parents:
                    raise ValueError(
                        f"query_parents_prior: child {_c} lists no parents."
                    )
                for _p in _parents:
                    if not 1 <= _p <= _L:
                        raise ValueError(
                            f"query_parents_prior: parent id {_p} of child "
                            f"{_c} out of range 1..{_L}."
                        )
                    if _p == _c:
                        raise ValueError(
                            f"query_parents_prior: child {_c} cannot be its "
                            "own parent."
                        )
                _norm[_c] = {"parents": _parents, "fixed": _fixed}
            self.query_parents_prior = _norm
        # Source-nodes query prior (applied by init_source_queries_zero on the
        # first training batch, AFTER the centroid/parents initialisation).
        # Normalise to int keys and validate the GLOBAL 1-based dataset node
        # IDs.  A node may not be BOTH a known-edges child and a source.
        self.query_source_prior: Optional[Dict[int, dict]] = None
        if query_source_prior:
            if not free_query_embedding:
                raise ValueError(
                    "query_source_prior requires free_query_embedding=True "
                    "(there is no dedicated query embedding table to "
                    "initialise otherwise)."
                )
            _L = S_seq_len + X_seq_len
            _snorm: Dict[int, dict] = {}
            for _node, _spec in query_source_prior.items():
                _n = int(_node)
                _mask = bool(_spec.get("mask", True))
                if not 1 <= _n <= _L:
                    raise ValueError(
                        f"query_source_prior: node id {_n} out of range "
                        f"1..{_L}."
                    )
                if _n <= S_seq_len and not self.homogeneous_nodes:
                    raise ValueError(
                        f"query_source_prior: node id {_n} is an S node; in "
                        "split mode only X nodes (ids > S_seq_len) have a "
                        "query to zero."
                    )
                if self.query_parents_prior and _n in self.query_parents_prior:
                    raise ValueError(
                        f"query_source_prior: node {_n} is also listed in "
                        "query_parents_prior; a node cannot be BOTH a "
                        "known-edges child and a source."
                    )
                _snorm[_n] = {"mask": _mask}
            self.query_source_prior = _snorm

        # Forbidden-parents query prior (orthogonal key exclusion; applied by
        # _project_forbidden_query_rows on every forward).  Normalise to int
        # keys and validate the GLOBAL 1-based dataset node IDs.  The per-child
        # projector buffers are built further below, once the orthonormal
        # frames exist.
        self.query_forbidden_prior: Optional[Dict[int, dict]] = None
        if query_forbidden_prior:
            if not free_query_embedding:
                raise ValueError(
                    "query_forbidden_prior requires free_query_embedding=True "
                    "(there is no dedicated query embedding table to "
                    "constrain otherwise)."
                )
            if struct_embedding_type != "orthogonal_fixed":
                raise ValueError(
                    "query_forbidden_prior requires "
                    "struct_embedding_type='orthogonal_fixed' (a fixed "
                    "orthonormal key frame); got "
                    f"struct_embedding_type='{struct_embedding_type}'."
                )
            if not (remove_query_projection and remove_key_projection):
                raise ValueError(
                    "query_forbidden_prior requires "
                    "remove_query_projection=True and "
                    "remove_key_projection=True so the embedding space IS the "
                    "QK scoring space and the orthogonal exclusion is exact."
                )
            _L = S_seq_len + X_seq_len
            _fnorm: Dict[int, dict] = {}
            for _child, _spec in query_forbidden_prior.items():
                _c = int(_child)
                _forbidden = [int(p) for p in _spec.get("forbidden", [])]
                _mask = bool(_spec.get("mask", True))
                if not 1 <= _c <= _L:
                    raise ValueError(
                        f"query_forbidden_prior: child id {_c} out of range "
                        f"1..{_L}."
                    )
                if _c <= S_seq_len and not self.homogeneous_nodes:
                    raise ValueError(
                        f"query_forbidden_prior: child id {_c} is an S node; "
                        "in split mode only X nodes (ids > S_seq_len) have a "
                        "query to constrain."
                    )
                if not _forbidden:
                    raise ValueError(
                        f"query_forbidden_prior: child {_c} lists no "
                        "forbidden parents."
                    )
                for _p in _forbidden:
                    if not 1 <= _p <= _L:
                        raise ValueError(
                            f"query_forbidden_prior: forbidden parent id {_p} "
                            f"of child {_c} out of range 1..{_L}."
                        )
                    if _p == _c:
                        raise ValueError(
                            f"query_forbidden_prior: child {_c} cannot forbid "
                            "itself as a parent."
                        )
                _fnorm[_c] = {"forbidden": _forbidden, "mask": _mask}
            self.query_forbidden_prior = _fnorm
        # Automatic self-key exclusion: same prerequisites as the forbidden
        # prior (the projector machinery is shared).  The projector buffers
        # are built further below, once the orthonormal frames exist.
        self.query_exclude_self = bool(query_exclude_self)
        if self.query_exclude_self:
            if not free_query_embedding:
                raise ValueError(
                    "query_exclude_self=True requires "
                    "free_query_embedding=True (there is no dedicated query "
                    "embedding table to constrain otherwise)."
                )
            if struct_embedding_type != "orthogonal_fixed":
                raise ValueError(
                    "query_exclude_self=True requires "
                    "struct_embedding_type='orthogonal_fixed' (a fixed "
                    "orthonormal key frame); got "
                    f"struct_embedding_type='{struct_embedding_type}'."
                )
            if not (remove_query_projection and remove_key_projection):
                raise ValueError(
                    "query_exclude_self=True requires "
                    "remove_query_projection=True and "
                    "remove_key_projection=True so the embedding space IS the "
                    "QK scoring space and the orthogonal exclusion is exact."
                )

        # Whether the cross block is a gated attention (structure-gated).  Kept
        # for diagnostics; the reconstruction-gain stream has been removed.
        self.is_gated = (
            attention_type == "GatedCrossAttention" or self.homogeneous_nodes
        )

        # ------------------------------------------------------------------
        # Value-structure injection scheme selection.
        # Combine a per-SOURCE-node identity code with the (data-only) value
        # stream before W_V, so the shared W_V can specialise per source
        # variable (a per-node functional).  The identity is the KEY/source
        # identity, giving per-parent output functions while preserving SVFA's
        # source-shared value.  Combination is concatenation for "separate" /
        # "struct_detached" (V_j = W_V([v_j ; e_j]), W_V widened by vsi_dim) and
        # SUMMATION for "learned_sum" (V_j = W_V(v_j + e_j), the vanilla/
        # benchmark composition; W_V keeps its width).  ``vsi_dim`` and
        # ``vsi_mode`` are threaded into the AttentionLayer(s).
        # ------------------------------------------------------------------
        VALUE_STRUCTURE_INJECTION_TYPES = (
            "none", "separate", "struct_detached", "learned_sum",
        )
        if value_structure_injection not in VALUE_STRUCTURE_INJECTION_TYPES:
            raise ValueError(
                f"value_structure_injection='{value_structure_injection}' is "
                f"invalid. Must be one of: {list(VALUE_STRUCTURE_INJECTION_TYPES)}."
            )
        self.value_structure_injection = value_structure_injection
        self.inject_value_structure = value_structure_injection != "none"
        if self.inject_value_structure and comps_embed_X != "svfa":
            # In summation mode the value already carries the identity (K = V),
            # so injection is redundant and the value width would be ambiguous.
            raise ValueError(
                "value_structure_injection requires SVFA "
                "(comps_embed_X='svfa'); got comps_embed_X="
                f"'{comps_embed_X}'."
            )
        # Width of the injected identity code (0 disables the injection).
        vsi_dim = d_model if self.inject_value_structure else 0
        # Combination mode threaded into the AttentionLayer(s): "learned_sum"
        # SUMS the code onto the value token before W_V (V_j = W_V(v_j + e_j),
        # the vanilla/benchmark composition; W_V is NOT widened); every other
        # scheme CONCATENATES (V_j = W_V([v_j ; e_j]), widened W_V).
        vsi_mode = (
            "sum" if value_structure_injection == "learned_sum" else "concat"
        )

        # --- Value-structure QUERY injection (additive child-identity term) ---
        if value_structure_query_injection not in VALUE_STRUCTURE_INJECTION_TYPES:
            raise ValueError(
                f"value_structure_query_injection="
                f"'{value_structure_query_injection}' is invalid. Must be one "
                f"of: {list(VALUE_STRUCTURE_INJECTION_TYPES)}."
            )
        self.value_structure_query_injection = value_structure_query_injection
        self.inject_value_structure_query = value_structure_query_injection != "none"
        if self.inject_value_structure_query and comps_embed_X != "svfa":
            raise ValueError(
                "value_structure_query_injection requires SVFA "
                f"(comps_embed_X='svfa'); got comps_embed_X='{comps_embed_X}'."
            )
        vsq_dim = d_model if self.inject_value_structure_query else 0

        # ------------------------------------------------------------------
        # Prior-softmax reconstruction gain (validation; see the ctor docs).
        # The gain lives on the gated blocks only and reuses the
        # reconstruction-routed value-identity tables for its data term.
        # ------------------------------------------------------------------
        self.use_gain_softmax = bool(use_gain_softmax)
        self.gain_data = bool(gain_data)
        if self.use_gain_softmax:
            # NB: ``self.cross_only`` is not defined yet at this point; the
            # cross-only variant is exactly ``self_attention_type is None``.
            if self_attention_type is None:
                raise ValueError(
                    "use_gain_softmax=True requires a gated attention block "
                    "(GatedCrossAttention / GatedSelfAttention); the cross-only "
                    "vanilla block (self_attention_type=None) has no structure "
                    "gate to use as the softmax prior."
                )
            if not self.homogeneous_nodes and attention_type != "GatedCrossAttention":
                raise ValueError(
                    "use_gain_softmax=True in split mode requires "
                    f"attention_type='GatedCrossAttention', got {attention_type!r}."
                )
            if self_attention_type != "GatedSelfAttention":
                raise ValueError(
                    "use_gain_softmax=True requires "
                    "self_attention_type='GatedSelfAttention' "
                    f"(got {self_attention_type!r}); CommutatorSelfAttention "
                    "owns the legacy sigmoid gain stream instead."
                )
            if self.gain_data:
                if self.value_structure_injection not in ("separate", "learned_sum"):
                    raise ValueError(
                        "gain_data=True requires value_structure_injection in "
                        "('separate', 'learned_sum') so the reconstruction-routed "
                        "per-source identity tables (val_id_embed_*) exist; the "
                        "gain key is built from them and the value stream, NEVER "
                        "from the structural embeddings.  Got "
                        f"{self.value_structure_injection!r}."
                    )
                if self.value_structure_query_injection not in ("separate", "learned_sum"):
                    raise ValueError(
                        "gain_data=True requires value_structure_query_injection "
                        "in ('separate', 'learned_sum') so the per-child identity "
                        "table (val_q_id_embed_X) exists for the gain query.  "
                        f"Got {self.value_structure_query_injection!r}."
                    )



        # ------------------------------------------------------------------
        # Direction-aware XÃ¢â€ â€™X self-attention split.
        # When ``self_attention_type`` is set, the XÃ¢â€ â€™X interaction is modelled
        # by a dedicated ``GatedSelfAttention`` block (edge-direction aware)
        # instead of the right-hand columns of the single combined cross block.
        # ------------------------------------------------------------------
        SELF_ATTENTION_TYPES = ("GatedSelfAttention", "CommutatorSelfAttention")
        self.self_attention_type = self_attention_type
        if self_attention_type is not None and self_attention_type not in SELF_ATTENTION_TYPES:
            raise ValueError(
                f"self_attention_type='{self_attention_type}' is invalid. "
                f"Must be None (combined cross-only block) or one of: "
                f"{list(SELF_ATTENTION_TYPES)}."
            )
        if self_attention_type is None and self.homogeneous_nodes:
            raise ValueError(
                "homogeneous_nodes=True requires self_attention_type: the single "
                "square (N, N) block IS the self attention."
            )
        # ``self_attention_type=None`` restores the COMBINED CROSS-ONLY variant:
        # ONE attention block whose keys/values are [S_actual ; X_actual], so a
        # single softmax normalises over S and X parents jointly and the XÃ¢â€ â€™X
        # columns carry no explicit direction gate.  This is the configuration
        # used by the vanilla-transformer benchmark; the direction-aware split
        # (cross + self) remains the default for the causal method.
        self.cross_only = self_attention_type is None
        # In homogeneous mode the ONE square (N, N) block IS the self attention
        # (stored in ``self.attention``), so there is no separate split block.
        self.split_xx = not self.homogeneous_nodes and not self.cross_only

        # ------------------------------------------------------------------
        # Harmonized attention temperatures (see
        # docs/documentation/ATTENTION_TEMPERATURES.md).  The legacy shared
        # ``init_tau`` key has been REMOVED: the split keys fall back straight
        # to the calculated defaults.
        # ------------------------------------------------------------------
        self.init_tau_act = 3.0   # non-gated activation temperature (CausalCross/Sigmoid)
        self.init_tau_cross = (
            float(init_tau_cross) if init_tau_cross is not None
            else DEFAULT_GATE_TAU
        )
        self.init_tau_self = (
            float(init_tau_self) if init_tau_self is not None
            else DEFAULT_GATE_TAU
        )
        self.dir_tau = (
            float(dir_tau_self) if dir_tau_self is not None
            else (float(dir_tau) if dir_tau is not None else DEFAULT_DIR_TAU)
        )

        # Shared structural query projection (W_q) across the cross (SÃ¢â€ â€™X) and
        # self (XÃ¢â€ â€™X) blocks.  Only meaningful in split mode; ignored otherwise.
        self.shared_query = bool(shared_query)
        if self.shared_query and self.homogeneous_nodes:
            raise ValueError(
                "shared_query=True is invalid with homogeneous_nodes=True: "
                "there is a SINGLE attention block, so the query projection is "
                "shared by construction."
            )

        # Shared structural key projection (W_K) across the cross (SÃ¢â€ â€™X) and
        # self (XÃ¢â€ â€™X) blocks.  Only meaningful in split mode; ignored otherwise.
        self.shared_key = bool(shared_key)
        if self.shared_key and self.homogeneous_nodes:
            raise ValueError(
                "shared_key=True is invalid with homogeneous_nodes=True: "
                "there is a SINGLE attention block, so the key projection is "
                "shared by construction (all N keys pass through the same W_K, "
                "which preserves the orthogonality of the structural keys)."
            )

        # The sharing flags need TWO blocks: in cross-only mode there is ONE
        # block and nothing to share with, so the flags would be silently
        # ignored Ã¢â‚¬â€ refuse loudly instead of running a mis-configured arm.
        if self.shared_query and self.cross_only:
            raise ValueError(
                "shared_query=True requires self_attention_type (split mode): "
                "in cross-only mode there is a SINGLE attention block, so "
                "there is no second block to share the query projection with."
            )
        if self.shared_key and self.cross_only:
            raise ValueError(
                "shared_key=True requires self_attention_type (split mode): "
                "in cross-only mode there is a SINGLE attention block, so "
                "there is no second block to share the key projection with."
            )


        # Orthogonal (isometric) key projection.  When "orthogonal", the shared
        # W_K is constrained so that W_K^T W_K = I (Cayley parametrisation inside
        # OrthogonalLinear).  An isometry preserves inner products, so
        #   <k_i W_K, k_j W_K> = k_i (W_K^T W_K) k_j^T = <k_i, k_j> = 0
        # whenever the raw keys are orthogonal (e.g. OrthogonalMaskEmbedding).
        # This carries the embedding-level orthogonality through to the projected
        # keys.  NOTE: an isometry from R^d requires d_qk >= d_model.
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
        # Embeddings
        # The SAME ModularEmbedding instance is used for both:
        #   - X queries (blanked value Ã¢â€ â€™ only variable-ID embedding contributes)
        #   - X keys    (actual value Ã¢â€ â€™ value + variable-ID embedding)
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
            # Vanilla softmax attention (Vaswani et al., 2017).
            "ScaledDotSoftmax": ScaledDotSoftmax,
            # Hard Concrete L0 gates (Louizos et al., ICLR 2018).
            "HardConcreteCrossAttention": HardConcreteCrossAttention,
            # Disentangled structure-gate x reconstruction-gain (anti-run_0).
            "GatedCrossAttention": GatedCrossAttention,
        }

        self_att_cls_map = {
            "GatedSelfAttention": GatedSelfAttention,
            "CommutatorSelfAttention": CommutatorSelfAttention,
        }

        if self.homogeneous_nodes:
            # HOMOGENEOUS: ONE square (N, N) self-attention block over all
            # nodes.  The cross ``attention_type`` is IGNORED entirely; the
            # block is stored in ``self.attention`` (the canonical attribute) so
            # every downstream consumer (score sparsity, gradient routing,
            # freezing, interference probes, centroid init) keeps working.
            att_cls = self_att_cls_map[self_attention_type]
            query_seq_len = self.N
            cross_key_seq_len = self.N
        else:
            if attention_type not in att_cls_map:
                raise ValueError(
                    f"attention_type='{attention_type}' is not supported for "
                    f"AttentionSelectorLayer.  Choose from: {list(att_cls_map)}"
                )
            att_cls = att_cls_map[attention_type]
            query_seq_len = X_seq_len
            # SPLIT: the cross block attends to S ONLY (keys/values = S); the
            # XÃ¢â€ â€™X interaction is handled by the self-attention block below.
            # CROSS-ONLY: the single block attends to [S ; X] jointly.
            cross_key_seq_len = (
                S_seq_len if self.split_xx else S_seq_len + X_seq_len
            )

        # Temperature routed to the MAIN block: in homogeneous mode it IS the
        # self gate; otherwise the cross existence gate (Hard-Concrete) or the
        # legacy activation temperature (non-gated attentions).
        if self.homogeneous_nodes:
            main_tau = self.init_tau_self
        elif att_cls in (GatedCrossAttention, HardConcreteCrossAttention):
            main_tau = self.init_tau_cross
        else:
            main_tau = self.init_tau_act

        # ---- Source-side top-k budget (TopKGate) --------------------------
        # None (default) = disabled, dense behaviour bit-identical.  When
        # enabled, one TopKGate per gated block: the square self block (and
        # the homogeneous combined block, which IS square) excludes the
        # diagonal; the bipartite S->X cross block has no diagonal.
        topk_gate_main = None
        topk_gate_self = None
        if topk_k is not None:
            _topk_common = dict(
                k=topk_k,
                method=topk_method,
                slack_init=topk_slack_init,
                slack_final=topk_slack_final,
                annealing_batches=topk_annealing_batches,
                per_row_slack=topk_per_row_slack,
            )
            if self.homogeneous_nodes:
                topk_gate_main = TopKGate(exclude_diagonal=True, **_topk_common)
            else:
                topk_gate_main = TopKGate(exclude_diagonal=False, **_topk_common)
                if self.split_xx:
                    topk_gate_self = TopKGate(exclude_diagonal=True, **_topk_common)
        # Plain (non-registered) references for diagnostics: the gates are
        # OWNED by the inner attention modules (registered there, so device
        # moves and checkpoints work); registering them again at the layer
        # top level would duplicate the state-dict entries.
        self.__dict__["topk_gate_main"] = topk_gate_main
        self.__dict__["topk_gate_self"] = topk_gate_self


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
            query_seq_len=query_seq_len,
            key_seq_len=cross_key_seq_len,
            init_tau=main_tau,
            # Direction-gate parameters: consumed only when THIS block is the
            # square (N, N) self attention (homogeneous mode); the cross
            # attentions ignore them.
            dir_tau=self.dir_tau,
            dir_bias=dir_bias,
            detach_transpose=detach_transpose,

            direction_mode=commutator_direction_mode,
            direction_rank=commutator_direction_rank,
            shared_dag_across_heads=shared_dag_across_heads,
            # Isometric key projection (W_K^T W_K = I) when "orthogonal".
            key_projection_type=key_projection_type,
            orthogonal_scale=orthogonal_key_scale,
            # Batch-consistent key dropout.
            batch_key_dropout=batch_key_dropout,
            batch_key_dropout_p_final=batch_key_dropout_p_final,
            batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
            batch_key_dropout_min_keys=batch_key_dropout_min_keys,
            batch_key_dropout_deterministic=batch_key_dropout_deterministic,
            batch_key_dropout_eval=batch_key_dropout_eval,
            batch_key_dropout_eval_seed=batch_key_dropout_eval_seed,

            # Optuna capacity-search protocol (constant-score override; only
            # honoured by CausalCrossAttention, ignored by other attention types).
            optuna_protocol=optuna_protocol,
            # HardConcrete / GatedCrossAttention stretch params + gain temperature.
            init_gamma=init_gamma,
            init_zeta=init_zeta,
            # Init-balancing offset on the SÃ¢â€ â€™X cross existence gate ONLY (see
            # __init__ docstring; resolved by resolve_init_edge_offset, never
            # entering F); the self block below is never offset.
            init_edge_offset=init_edge_offset,

            # Centroid-collapse fix: unit-normalise the structural query and use
            # a fixed sqrt(query_fanin_scale) score scale (structure gate only).
            normalize_query=normalize_query,
            query_fanin_scale=query_fanin_scale,
            query_norm_learnable=query_norm_learnable,
            query_norm_init_scale=query_norm_init_scale,
            query_norm_target=query_norm_target,
            # Remove / freeze the structural query & key projections (W_q / W_K).

            # On the cross block these own the (optionally shared) projections,
            # so this is where the removal / freeze actually takes effect.
            remove_query_projection=remove_query_projection,
            remove_key_projection=remove_key_projection,
            freeze_query_projection=freeze_query_projection,
            freeze_key_projection=freeze_key_projection,
            # Value-structure injection: combine the source identity code with
            # the value stream ("concat" widens W_V; "sum" keeps its width).
            value_structure_dim=vsi_dim,
            value_structure_mode=vsi_mode,
            # Value-structure QUERY injection: add W_V^q(e_child) query term.
            value_structure_query_dim=vsq_dim,
            # Prior-softmax reconstruction gain (inert at gain_lambda=0).
            use_gain_softmax=use_gain_softmax,
            gain_data=gain_data,
            gain_score_dim=gain_score_dim,
            # Source-side top-k budget (None = disabled).
            topk_gate=topk_gate_main,
        )




        # ------------------------------------------------------------------
        # Direction-aware XÃ¢â€ â€™X self-attention block (split mode only).

        # A dedicated ``GatedSelfAttention`` models X_i Ã¢â€ â€™ X_j with an
        # antisymmetric direction gate (d_ij + d_ji = 1) that suppresses
        # two-cycles Ã¢â‚¬â€ the directionality the single combined cross block
        # cannot express.  Q and K SHARE the same X structural identity
        # embedding (a self-attention requirement for the symmetric/
        # antisymmetric Toeplitz split); values are the X value stream.
        # ------------------------------------------------------------------
        if self.split_xx:
            self_att_cls = self_att_cls_map[self_attention_type]
            self.self_attention = AttentionLayer(
                attention=self_att_cls,
                d_model_queries=d_model,
                d_model_keys=d_model,
                d_model_values=d_model,
                d_queries_keys=d_qk,
                n_heads=n_heads,
                mask_layer=None,
                attention_dropout=attention_dropout,
                dropout_qkv=dropout_qkv,
                register_entropy=True,
                layer_name="selector_self_att",
                query_seq_len=X_seq_len,
                key_seq_len=X_seq_len,
                init_tau=self.init_tau_self,
                shared_dag_across_heads=shared_dag_across_heads,
                # Shared structural query projection: when True the self block
                # does NOT own a W_q; the cross block's W_q is applied to the X
                # structural embedding and fed in as a pre-projected query.
                query_external=self.shared_query,
                # Shared structural key projection: when True the self block does
                # NOT own a W_K; the cross block's W_K is applied to the X
                # structural embedding and fed in as a pre-projected key.  Keeps
                # S and X keys on the same isometric axes (orthogonal frame).
                key_external=self.shared_key,
                key_projection_type=key_projection_type,


                orthogonal_scale=orthogonal_key_scale,
                batch_key_dropout=batch_key_dropout,
                batch_key_dropout_p_final=batch_key_dropout_p_final,
                batch_key_dropout_annealing_batches=batch_key_dropout_annealing_batches,
                batch_key_dropout_min_keys=batch_key_dropout_min_keys,
                batch_key_dropout_deterministic=batch_key_dropout_deterministic,
                batch_key_dropout_eval=batch_key_dropout_eval,
                batch_key_dropout_eval_seed=batch_key_dropout_eval_seed,

                optuna_protocol=optuna_protocol,
                init_gamma=init_gamma,
                init_zeta=init_zeta,
                dir_tau=self.dir_tau,
                dir_bias=dir_bias,
                detach_transpose=detach_transpose,
                # CommutatorSelfAttention direction-gate parametrisation
                # ("qk" or "skew_query"); ignored by GatedSelfAttention.
                direction_mode=commutator_direction_mode,
                direction_rank=commutator_direction_rank,
                # Centroid-collapse fix (structure gate only).
                normalize_query=normalize_query,
                query_fanin_scale=query_fanin_scale,
                query_norm_learnable=query_norm_learnable,
                query_norm_init_scale=query_norm_init_scale,
                query_norm_target=query_norm_target,
                # Remove / freeze the self block's OWN W_q / W_K.  No-op when it

                # borrows the cross projection (query_external / key_external),
                # since an external projection takes priority in AttentionLayer.
                remove_query_projection=remove_query_projection,
                remove_key_projection=remove_key_projection,
                freeze_query_projection=freeze_query_projection,
                freeze_key_projection=freeze_key_projection,
                # Value-structure injection: combine the source identity code
                # with the value stream ("concat" widens W_V; "sum" keeps it).
                value_structure_dim=vsi_dim,
                value_structure_mode=vsi_mode,
                # Value-structure QUERY injection: add W_V^q(e_child) query term.
                value_structure_query_dim=vsq_dim,
                # Prior-softmax reconstruction gain (inert at gain_lambda=0).
                use_gain_softmax=use_gain_softmax,
                gain_data=gain_data,
                gain_score_dim=gain_score_dim,
                # Source-side top-k budget (None = disabled).
                topk_gate=topk_gate_self,

            )

            # Tie the learnable per-node query-norm multiplier across the cross
            # (S->X) and self (X->X) blocks under ``shared_query=True``: both
            # blocks score the SAME children (query rows = X_seq_len), so a
            # single shared multiplier keeps "how hard a child spends its
            # directional budget" consistent regardless of whether the parent is
            # an S or an X node.  Assigning the SAME nn.Parameter object makes
            # ``named_parameters()`` dedup it (one structural param) and
            # ``collect_query_norm_penalty`` charge it exactly once (by id).
            if self.shared_query and query_norm_learnable:
                cross_ia = self.attention.inner_attention
                self_ia = self.self_attention.inner_attention
                if (
                    getattr(cross_ia, "query_norm_log_scale", None) is not None
                    and getattr(self_ia, "query_norm_log_scale", None) is not None
                ):
                    self_ia.query_norm_log_scale = cross_ia.query_norm_log_scale

        else:
            self.self_attention = None




        # ------------------------------------------------------------------
        # Static hard masks (allowed-edge topology).  Registered as buffers so
        # they follow the module to the right device.
        #   * homogeneous: ONE (N, N) off-diagonal mask (no self-loops); the
        #     SÃ¢â€ â€™X direction is NOT assumed.
        #   * split: the SÃ¢â€ â€™X cross block is fully connected (all X may attend
        #     any S) and the XÃ¢â€ â€™X self block is off-diagonal (no self-loops).
        # ------------------------------------------------------------------
        if self.homogeneous_nodes:
            self.register_buffer(
                "homogeneous_mask", 1.0 - torch.eye(self.N)
            )
        elif self.cross_only:
            # ONE (L_X, L_S + L_X) mask: the S block is fully connected and the
            # X block is off-diagonal (no self-loops).
            self.register_buffer(
                "combined_mask",
                torch.cat(
                    [torch.ones(X_seq_len, S_seq_len), 1.0 - torch.eye(X_seq_len)],
                    dim=1,
                ),
            )
        else:
            self.register_buffer(
                "cross_mask", torch.ones(X_seq_len, S_seq_len)
            )
            self.register_buffer(
                "self_mask", 1.0 - torch.eye(X_seq_len)
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
        # Dispatch on struct_embedding_type; the two orthogonal schemes populate
        # orth_embed_S / orth_embed_X:
        #
        #  (a) "orthogonal_learnable" Ã¢â€ â€™ OrthogonalMaskEmbedding: disjoint binary
        #      blocks tiling d_model (k = d_model // n_vars dims/var; the
        #      remainder d_model % n_vars is idle). Value-modulated.
        #
        #  (b) "orthogonal_fixed" Ã¢â€ â€™ FixedOrthonormalEmbedding: dense frozen rows
        #      spanning ALL d_model dims, mutually orthonormal across S and X via
        #      a shared frame + disjoint row slices. Value-independent (identity
        #      only); the value reaches the output through the SVFA V stream.
        #
        #  (c) "standard_learnable" Ã¢â€ â€™ no override (orth_embed_{S,X} = None); the
        #      standard ModularEmbedding above provides the structural stream.
        #
        # In the orthogonal cases the value stream (V, residual, FFN, MLP head)
        # is UNCHANGED.
        self.orth_embed_S: Optional[nn.Module]
        self.orth_embed_X: Optional[nn.Module]
        if self.orthogonal_struct_embedding:
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
        elif self.orthogonal_fixed:
            total_vars = S_seq_len + X_seq_len
            if total_vars > d_model:
                raise ValueError(
                    f"struct_embedding_type='orthogonal_fixed' requires "
                    f"d_model >= S_seq_len + X_seq_len for mutually orthonormal "
                    f"rows, got d_model={d_model} < {total_vars}."
                )
            # Derive the frame seed from the global RNG (set by the training
            # seed_everything) so the frame varies across runs/seeds while both
            # S and X instances receive the SAME seed Ã¢â€ â€™ identical shared frame Ã¢â€ â€™
            # S-rows are guaranteed orthogonal to X-rows.
            frame_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            self.orth_embed_S = FixedOrthonormalEmbedding(
                num_variables=S_seq_len,
                d_model=d_model,
                total_variables=total_vars,
                row_offset=0,
                frame_type=orthogonal_fixed_frame_type,
                seed=frame_seed,
                scale=orthogonal_fixed_scale,
                device=device,
            )
            self.orth_embed_X = FixedOrthonormalEmbedding(
                num_variables=X_seq_len,
                d_model=d_model,
                total_variables=total_vars,
                row_offset=S_seq_len,
                frame_type=orthogonal_fixed_frame_type,
                seed=frame_seed,
                scale=orthogonal_fixed_scale,
                device=device,
            )
        else:
            # "standard_learnable": no structural override.
            self.orth_embed_S = None
            self.orth_embed_X = None

        # ------------------------------------------------------------------
        # Forbidden-parents prior buffers (orthogonal key exclusion)
        # ------------------------------------------------------------------
        # Per-child fixed projector P_c = I - K_f (K_f^T K_f)^{-1} K_f^T, where
        # K_f stacks the forbidden parents' rows of the shared orthonormal
        # frame (FixedOrthonormalEmbedding.frame: S rows first, then X).  On
        # every forward the constrained query rows are replaced by q @ P_c
        # (see _project_forbidden_query_rows): the forbidden key directions
        # are removed from the query span DIRECTLY instead of masking the
        # downstream score, so no alignment budget is wasted on forbidden
        # parents.  The buffers are detached constants Ã¢â‚¬â€ the chain rule
        # through P_c projects the embedding-row gradient into the allowed
        # subspace (null-space projection; no penalty term).
        #
        # ``qfp_hard_mask`` additionally zeroes the forbidden parent columns
        # of the hard mask for children with mask=True (a zero SCORE is not a
        # zero POSTERIOR: sigmoid gates stay open at logit 0).  Layout matches
        # the block masks: (N, N) in homogeneous mode, else the combined
        # (L_X, L_S + L_X) layout (sliced per block at the call site).  None
        # when the prior is disabled or every spec has mask=False.
        self.register_buffer("qfp_hard_mask", None)
        # ``qsp_hard_mask`` zeroes the source ROWS (incoming edges) for
        # query_source_prior nodes with mask=True: a zero SCORE is not a zero
        # POSTERIOR (the sigmoid existence gate stays open at logit 0), so the
        # mask makes the no-incoming-edge constraint exact and removes the
        # constant L0 contribution of the source rows.  Layout matches the
        # block masks: (N, N) in homogeneous mode, else the combined
        # (L_X, L_S + L_X) layout (sliced per block at the call site).  None
        # when the prior is disabled or every spec has mask=False.
        self.register_buffer("qsp_hard_mask", None)
        if self.query_source_prior and any(
            _s["mask"] for _s in self.query_source_prior.values()
        ):
            if self.homogeneous_nodes:
                _sm = torch.ones(self.N, self.N)
                for _n, _spec in self.query_source_prior.items():
                    if _spec["mask"]:
                        _sm[_n - 1, :] = 0.0
            else:
                _sm = torch.ones(X_seq_len, S_seq_len + X_seq_len)
                for _n, _spec in self.query_source_prior.items():
                    if _spec["mask"]:
                        _sm[_n - S_seq_len - 1, :] = 0.0
            self.qsp_hard_mask = _sm

        # Constrained children: the forbidden-prior children UNION, when
        # query_exclude_self, EVERY child (its OWN frame row is added to K_f --
        # the structural masks already zero the diagonal, so self-alignment is
        # pure waste).  In split mode only X nodes (ids > S_seq_len) are
        # queries; in homogeneous mode all N nodes are.
        self._qfp_children: list = []
        if self.query_forbidden_prior or self.query_exclude_self:
            assert self.orth_embed_S is not None  # orthogonal_fixed (validated)
            assert self.orth_embed_X is not None
            frame = torch.cat(
                [self.orth_embed_S.frame, self.orth_embed_X.frame], dim=0
            ).detach()                                        # (L, d_model)
            _eye = torch.eye(frame.shape[1], dtype=frame.dtype)
            _children = set(self.query_forbidden_prior or {})
            if self.query_exclude_self:
                if self.homogeneous_nodes:
                    _children.update(range(1, self.N + 1))
                else:
                    _children.update(
                        range(S_seq_len + 1, S_seq_len + X_seq_len + 1)
                    )
            _needs_mask = False
            for _c in sorted(_children):
                _rows = []
                _spec = (self.query_forbidden_prior or {}).get(_c)
                if _spec is not None:
                    _rows.extend(p - 1 for p in _spec["forbidden"])
                    _needs_mask = _needs_mask or _spec["mask"]
                if self.query_exclude_self:
                    _rows.append(_c - 1)                      # the SELF key
                _Kf = frame[_rows]                            # (F, d)
                _P = _eye - _Kf.T @ torch.linalg.solve(
                    _Kf @ _Kf.T, _Kf
                )                                             # (d, d), symmetric
                self.register_buffer(
                    f"qfp_proj_{_c}", _P, persistent=True
                )
                self._qfp_children.append(_c)
            if _needs_mask:
                if self.homogeneous_nodes:
                    _m = torch.ones(self.N, self.N)
                    for _c, _spec in self.query_forbidden_prior.items():
                        if not _spec["mask"]:
                            continue
                        _m[_c - 1, [p - 1 for p in _spec["forbidden"]]] = 0.0
                else:
                    _m = torch.ones(X_seq_len, S_seq_len + X_seq_len)
                    for _c, _spec in self.query_forbidden_prior.items():
                        if not _spec["mask"]:
                            continue
                        _m[
                            _c - S_seq_len - 1,
                            [p - 1 for p in _spec["forbidden"]],
                        ] = 0.0
                self.qfp_hard_mask = _m


        # ------------------------------------------------------------------
        # Free query embedding (optional override for the X QUERY stream only)
        # ------------------------------------------------------------------
        # When free_query_embedding=True the X query gets its OWN learnable
        # identity embedding, decoupling "X_i as child" (query) from "X_i as
        # parent" (key).  The X KEY stream is untouched (it keeps the orthogonal
        # or standard embedding).  The value stream is also UNCHANGED.
        # In homogeneous mode S is ALSO a query (candidate child), so it gets its
        # own free query table too.
        self.query_embed_X: Optional[FreeQueryEmbedding]
        self.query_embed_S: Optional[FreeQueryEmbedding]
        if free_query_embedding:
            self.query_embed_X = FreeQueryEmbedding(
                num_variables=X_seq_len,
                d_model=d_model,
                device=device,
            )
            self.query_embed_S = (
                FreeQueryEmbedding(
                    num_variables=S_seq_len, d_model=d_model, device=device,
                )
                if self.homogeneous_nodes
                else None
            )
        else:
            self.query_embed_X = None
            self.query_embed_S = None

        # Arm the frozen-row gradient hooks at construction time (no data
        # needed): the hook zeroes gradients on prior-fixed rows from the very
        # first step.  The snapshot is REFRESHED with the prior values by
        # init_queries_from_parents on the first training batch.
        if self.query_parents_prior:
            _fixed_X = [
                c - S_seq_len for c, s in self.query_parents_prior.items()
                if s["fixed"] and c > S_seq_len
            ]
            _fixed_S = [
                c for c, s in self.query_parents_prior.items()
                if s["fixed"] and c <= S_seq_len
            ]
            if _fixed_X and self.query_embed_X is not None:
                self.query_embed_X.freeze_rows(_fixed_X)
            if _fixed_S and self.query_embed_S is not None:
                self.query_embed_S.freeze_rows(_fixed_S)

        # ------------------------------------------------------------------
        # Value-structure injection identity tables (value_structure_injection
        # in {"separate", "learned_sum"}).  Dedicated identity tables (one per
        # S / X source variable) whose names ("val_id_") contain NO structural
        # pattern, so the gradient router classifies them as RECONSTRUCTION
        # parameters.  "separate" uses the unconstrained FreeQueryEmbedding;
        # "learned_sum" uses the vanilla-style ValueIdentityEmbedding (plain
        # nn.Embedding with max_norm=1, matching the benchmark's variable
        # embedding) since its code is SUMMED directly onto the value token.
        # The "struct_detached" scheme reuses the (detached) structural identity
        # and needs no new parameters, so no tables are created there.
        # ------------------------------------------------------------------
        self.val_id_embed_S: Optional[nn.Module]
        self.val_id_embed_X: Optional[nn.Module]
        if self.value_structure_injection == "separate":
            self.val_id_embed_S = FreeQueryEmbedding(
                num_variables=S_seq_len, d_model=d_model, device=device,
            )
            self.val_id_embed_X = FreeQueryEmbedding(
                num_variables=X_seq_len, d_model=d_model, device=device,
            )
        elif self.value_structure_injection == "learned_sum":
            self.val_id_embed_S = ValueIdentityEmbedding(
                num_variables=S_seq_len, d_model=d_model, device=device,
            )
            self.val_id_embed_X = ValueIdentityEmbedding(
                num_variables=X_seq_len, d_model=d_model, device=device,
            )
        else:
            self.val_id_embed_S = None
            self.val_id_embed_X = None

        # Value-structure QUERY injection identity table ("separate" /
        # "learned_sum").  The query is always an X node (candidate child), so
        # a single per-X-node code is enough.  Name ("val_q_id_") keeps it in
        # the RECONSTRUCTION group; "struct_detached" reuses the detached X
        # identity (0 params).  The query combination is ADDITIVE in both
        # table-bearing schemes ((sum_j A_ij) * W_V^q(e_i)); "learned_sum" only
        # swaps the table class to the vanilla-style ValueIdentityEmbedding.
        self.val_q_id_embed_X: Optional[nn.Module]
        self.val_q_id_embed_S: Optional[nn.Module]
        if self.value_structure_query_injection == "separate":
            self.val_q_id_embed_X = FreeQueryEmbedding(
                num_variables=X_seq_len, d_model=d_model, device=device,
            )
            # Homogeneous mode: S nodes are queries (children) too.
            self.val_q_id_embed_S = (
                FreeQueryEmbedding(
                    num_variables=S_seq_len, d_model=d_model, device=device,
                )
                if self.homogeneous_nodes
                else None
            )
        elif self.value_structure_query_injection == "learned_sum":
            self.val_q_id_embed_X = ValueIdentityEmbedding(
                num_variables=X_seq_len, d_model=d_model, device=device,
            )
            self.val_q_id_embed_S = (
                ValueIdentityEmbedding(
                    num_variables=S_seq_len, d_model=d_model, device=device,
                )
                if self.homogeneous_nodes
                else None
            )
        else:
            self.val_q_id_embed_X = None
            self.val_q_id_embed_S = None


        # ------------------------------------------------------------------
        # Output MLP head

        # ------------------------------------------------------------------

        self.per_node_output = bool(per_node_output)
        # Adjacency-context mode: False/None -> off; True -> "concat"
        # (legacy); "concat" / "film" select the injection mechanism.
        _pac = per_node_adjacency_context
        if _pac is True:
            _pac = "concat"
        elif _pac in (False, None):
            _pac = None
        _pac = str(_pac).lower() if _pac is not None else None
        if _pac not in (None, "concat", "film"):
            raise ValueError(
                f"per_node_adjacency_context must be one of False / True / "
                f"'concat' / 'film', got {per_node_adjacency_context!r}"
            )
        self.per_node_adjacency_context = _pac is not None
        self._adjacency_context_mode = _pac
        if self.per_node_adjacency_context and not self.per_node_output:
            raise ValueError(
                "per_node_adjacency_context=True requires "
                "per_node_output=True (the adjacency row is concatenated "
                "onto the per-node MLP input)."
            )
        # Raw-value adjacency passthrough (Arm B): the attention posterior IS
        # the adjacency acting on raw values; the per-node decoder MLP input
        # is the weighted parent vector of dim L_S + L_X (see
        # forward_with_actual).  Requires per_node_output=True.
        self.raw_value_adjacency = bool(raw_value_adjacency)
        if self.raw_value_adjacency and not self.per_node_output:
            raise ValueError(
                "raw_value_adjacency=True requires per_node_output=True "
                "(the adjacency-weighted raw parent vector is decoded by the "
                "per-node MLP head)."
            )
        # Raw value column index per stream (fallback to the production
        # convention: value column 0, variable-ID column 1).
        self._raw_val_idx_S = (
            self.embedding_S.per_node_value_embed_list[0].val_idx
            if getattr(self.embedding_S, "per_node_value_embed_list", None)
            else 0
        )
        self._raw_val_idx_X = (
            self.embedding_X.per_node_value_embed_list[0].val_idx
            if getattr(self.embedding_X, "per_node_value_embed_list", None)
            else 0
        )

        if self.per_node_output:
            # DAGMA-style per-node decoder: one MLP per variable.  In
            # homogeneous mode the head must cover ALL N = L_S + L_X nodes
            # (S is also reconstructed); otherwise only the X nodes.
            n_out_nodes = self.N if self.homogeneous_nodes else X_seq_len
            # Optional adjacency context widens every per-node MLP input by
            # the applied-adjacency row width (L_S + L_X) in "concat" mode;
            # "film" mode keeps the base width and conditions via FiLM.
            _ctx_dim = (
                (S_seq_len + X_seq_len)
                if self._adjacency_context_mode == "concat"
                else 0
            )
            _film_dim = (
                (S_seq_len + X_seq_len)
                if self._adjacency_context_mode == "film"
                else 0
            )
            _base_dim = (
                (S_seq_len + X_seq_len) if self.raw_value_adjacency else d_model
            )
            self.forecaster = PerNodeMLPHead(
                d_model=_base_dim + _ctx_dim,
                out_dim=out_dim,
                num_variables=n_out_nodes,
                d_hidden=per_node_output_hidden,
                activation=output_mlp_activation,
                dropout=output_mlp_dropout,
                bias=True,
                film_context_dim=_film_dim,
                n_layers=per_node_output_layers,
            )
        else:
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
    # Forbidden-parents query projection (orthogonal key exclusion)
    # ------------------------------------------------------------------

    def _project_forbidden_query_rows(
        self, query: torch.Tensor, row_offset: int
    ) -> torch.Tensor:
        """Remove the forbidden-parent key span from the given query rows.

        ``query`` is ``(B, L, d)`` with row ``i`` holding the query of GLOBAL
        1-based node ``row_offset + i + 1`` (row_offset=0 for the homogeneous
        [S ; X] query stream, row_offset=S_seq_len for the X-only stream).
        For each CONSTRAINED child (``_qfp_children``: forbidden-prior
        children union, when ``query_exclude_self``, every child) the fixed projector
        ``P = I - K_f (K_f^T K_f)^{-1} K_f^T`` (buffer ``qfp_proj_<child>``,
        built from the orthonormal frame at construction) is applied:
        ``q <- q @ P``.  The constraint is EXACT at every step Ã¢â‚¬â€ unlike a hard
        mask, which only blanks the downstream score, the forbidden key
        directions can no longer attract the query Ã¢â‚¬â€ and the chain rule
        through the constant P automatically projects the embedding-row
        gradient into the allowed subspace (null-space projection, no penalty
        term).  Idempotent: P @ P == P.
        """
        for c in self._qfp_children:
            row = c - row_offset - 1
            if 0 <= row < query.shape[1]:
                P = getattr(self, f"qfp_proj_{c}")
                query[:, row, :] = query[:, row, :] @ P.to(query.dtype)
        return query

    # ------------------------------------------------------------------
    # Query centroid initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _query_embedding_for_target(query_proj, target: torch.Tensor) -> torch.Tensor:
        """Return an embedding ``e`` such that ``query_proj(e) Ã¢â€°Ë† target``.

        Used to place the projected query on a chosen point (the key centroid)
        by writing ONLY the query embedding Ã¢â‚¬â€ the query projection W_q itself is
        left untouched.  Cases:

        * ``None`` / ``nn.Identity`` (query projection removed) Ã¢â€ â€™ ``e = target``.
        * ``nn.Linear``  Ã¢â€ â€™ least-squares solve ``W e = target - b`` (exact when
          W is square/full-rank; minimum-residual otherwise).
        * anything else  Ã¢â€ â€™ fall back to ``target`` (assumes the projection
          preserves the embedding space).
        """
        if query_proj is None or isinstance(query_proj, nn.Identity):
            return target
        if isinstance(query_proj, nn.Linear):
            W = query_proj.weight.detach()                 # (out, in)
            rhs = target
            if query_proj.bias is not None:
                rhs = target - query_proj.bias.detach()
            sol = torch.linalg.lstsq(W, rhs.reshape(-1, 1)).solution
            return sol.reshape(-1)
        return target

    @torch.no_grad()
    def init_query_at_key_centroid(
        self,
        source_tensor: torch.Tensor,
        x_actual: torch.Tensor,
    ) -> None:
        """Initialise the free X query embedding at the centroid of the keys.

        Every real query row is overwritten with the SAME vector so all queries
        start from one point and read every candidate parent uniformly (for an
        orthonormal key frame the centroid yields identical ``<q, k_j>`` for all
        ``j``), giving HSIC a symmetric starting point to break toward the true
        parents *before* the directional budget saturates (see the SELF_ATTENTION
        spurious-S3->X4 investigation).

        The target is computed in the space the QK^T score lives in: the keys are
        first passed through the KEY projection (if any), so the centroid changes
        depending on whether the key projection is used.  Only the query
        embedding is written Ã¢â‚¬â€ the QUERY projection W_q is left untouched and
        inverted (least squares) when present, so the feature works both with and
        without the query/key projections.

        Requires ``free_query_embedding=True``.  Value-modulated key embeddings
        (e.g. ``orthogonal_learnable`` / ``standard_learnable``) need real data,
        which is why this is invoked lazily on the first training batch.
        """
        if self.query_embed_X is None:
            raise RuntimeError(
                "init_query_at_key_centroid requires free_query_embedding=True "
                "(no query_embed_X table to initialise)."
            )

        def _struct(raw):
            return raw[0] if isinstance(raw, tuple) else raw

        # ---- Key structural embeddings (mirror forward_with_actual) -------
        s_struct = _struct(self.embedding_S(X=source_tensor))
        xk_struct = _struct(self.embedding_X(X=x_actual))
        if self.orth_embed_S is not None:
            assert self.orth_embed_X is not None
            s_struct = self.orth_embed_S(source_tensor)
            xk_struct = self.orth_embed_X(x_actual)
        sx_keys = torch.cat([s_struct, xk_struct], dim=1)   # (B, L_S+L_X, d_model)

        # ---- Project keys into the scoring space (identity/none Ã¢â€ â€™ raw) ----
        key_proj = getattr(self.attention, "key_projection", None)
        if key_proj is None:
            k_proj = sx_keys
        else:
            k_proj = key_proj(sx_keys)

        # ---- Centroid over all key tokens (S and X) and the batch ---------
        centroid = k_proj.reshape(-1, k_proj.shape[-1]).mean(dim=0)   # (d_qk*H,)

        # ---- Invert the query projection to land the query on the centroid -
        query_proj = getattr(self.attention, "query_projection", None)
        e = self._query_embedding_for_target(query_proj, centroid)    # (d_model,)

        # ---- Write the SAME vector into every real query row (row 0 = pad) -
        # In homogeneous mode the S query table is written with the SAME vector,
        # so every node (S and X) starts from the identical point.
        tables = [self.query_embed_X]
        if self.query_embed_S is not None:
            tables.append(self.query_embed_S)
        for table in tables:
            w = table.embedding.weight
            e_w = e.to(dtype=w.dtype, device=w.device)
            w[1:].copy_(e_w.unsqueeze(0).expand(w.shape[0] - 1, -1))

    @torch.no_grad()
    def init_queries_from_parents(
        self,
        prior: Dict[int, dict],
        source_tensor: torch.Tensor,
        x_actual: torch.Tensor,
    ) -> int:
        """Overwrite the query of known-edges nodes with their parents'' keys.

        ``prior`` maps a GLOBAL 1-based dataset node id (S nodes are
        1..S_seq_len, X nodes S_seq_len+1..S_seq_len+X_seq_len) to
        ``{"parents": [...], "fixed": bool}``.  For each child the query
        embedding is placed, in the QK scoring space, at the MEAN of its
        parents'' projected keys (inverted through W_q like
        :meth:`init_query_at_key_centroid`).  Rows with ``fixed=True`` are
        then re-snapshotted at the prior value and stay frozen for the rest
        of training (see FreeQueryEmbedding.freeze_rows).

        Must be called AFTER the centroid/default initialisation so the prior
        overwrites it; nodes not listed keep their initialised query.  Returns
        the number of frozen rows.  Requires ``free_query_embedding=True``.
        """
        if self.query_embed_X is None:
            raise RuntimeError(
                "init_queries_from_parents requires free_query_embedding=True "
                "(no query_embed_X table to initialise)."
            )

        def _struct(raw):
            return raw[0] if isinstance(raw, tuple) else raw

        # ---- Key structural embeddings (mirror forward_with_actual) -------
        s_struct = _struct(self.embedding_S(X=source_tensor))
        xk_struct = _struct(self.embedding_X(X=x_actual))
        if self.orth_embed_S is not None:
            assert self.orth_embed_X is not None
            s_struct = self.orth_embed_S(source_tensor)
            xk_struct = self.orth_embed_X(x_actual)
        sx_keys = torch.cat([s_struct, xk_struct], dim=1)   # (B, L, d_model)

        # ---- Project keys into the scoring space (identity/none -> raw) ---
        key_proj = getattr(self.attention, "key_projection", None)
        k_proj = sx_keys if key_proj is None else key_proj(sx_keys)
        query_proj = getattr(self.attention, "query_projection", None)

        n_frozen = 0
        for child, spec in prior.items():
            c = int(child)
            cols = [p - 1 for p in spec["parents"]]          # global -> 0-based
            target = k_proj[:, cols, :].mean(dim=(0, 1))     # over batch+parents
            e = self._query_embedding_for_target(query_proj, target)
            # Compose with the forbidden-parents prior: keep the parent-
            # centroid target inside the allowed subspace (no-op when the
            # known parents are already orthogonal to the forbidden keys,
            # e.g. disjoint sets on an orthonormal frame).
            _P = getattr(self, f"qfp_proj_{c}", None)
            if _P is not None:
                e = _P @ e
            if c > self.S_seq_len:
                table, row = self.query_embed_X, c - self.S_seq_len
            else:
                assert self.query_embed_S is not None
                table, row = self.query_embed_S, c
            w = table.embedding.weight
            w[row].copy_(e.to(dtype=w.dtype, device=w.device))
            if spec.get("fixed", False):
                # Re-snapshot at the PRIOR value (the construction-time
                # snapshot captured the random init).
                table.freeze_rows([row])
                n_frozen += 1
        return n_frozen
    @torch.no_grad()
    def init_source_queries_zero(self, prior: Dict[int, dict]) -> int:
        """Zero and freeze the query rows of source nodes (never children).

        ``prior`` maps a GLOBAL 1-based dataset node id (S nodes are
        1..S_seq_len, X nodes S_seq_len+1..S_seq_len+X_seq_len) to
        ``{"mask": bool}``.  For each listed node the query embedding row is
        set to ZERO and frozen for the rest of training (see
        FreeQueryEmbedding.freeze_rows; re-asserted after every optimizer
        step via reassert_frozen_query_rows).

        With a zero query the node's raw score row is exactly 0, so the
        direction of every incident edge is decided ENTIRELY by the other
        endpoint's score: ``A_anti[s, j] = -raw[j, s]/2`` gives
        ``d[s <- j] = sigmoid(-raw[j, s] / (2*beta))`` and
        ``d[j -> s] = 1 - d[s <- j]`` -- all direction mass flows
        source -> child.  ``mask=True`` specs additionally zero the
        source rows of the hard mask (see the ``qsp_hard_mask`` buffer),
        making the no-incoming-edge constraint exact.

        Must be called AFTER the centroid/parents initialisation so the zero
        rows are not overwritten.  Returns the number of frozen rows.
        Requires ``free_query_embedding=True``.
        """
        if self.query_embed_X is None:
            raise RuntimeError(
                "init_source_queries_zero requires free_query_embedding=True "
                "(no query_embed_X table to initialise)."
            )

        n_frozen = 0
        for node in prior:
            n = int(node)
            if n > self.S_seq_len:
                table, row = self.query_embed_X, n - self.S_seq_len
            else:
                assert self.query_embed_S is not None
                table, row = self.query_embed_S, n
            w = table.embedding.weight
            w[row].zero_()
            table.freeze_rows([row])
            n_frozen += 1
        return n_frozen



    def reassert_frozen_query_rows(self) -> None:
        """Restore prior-frozen query rows to their snapshot values.

        Called by the forecaster after every structural optimizer step:
        decoupled weight decay / structural gradient noise act even on
        zero-gradient rows, so the hook alone is not sufficient.
        """
        for t in (self.query_embed_S, self.query_embed_X):
            if t is not None:
                t.reassert_frozen_rows()

    def enable_centroid_commit(self) -> None:
        """Enable the centroid-commit shadow on every free query table.

        After this call the embedding weights hold the COMMITTED centroids
        (constants in the graph, written only by commit events) and each
        table carries a persistent ``shadow`` buffer accumulating gradient
        evidence (see causaliT/training/centroid_commit.py).
        """
        for t in (self.query_embed_S, self.query_embed_X):
            if t is not None:
                t.enable_commit_shadow()

    def sync_commit_shadows(self) -> None:
        """Re-initialise all shadows at their committed weights (called after
        the lazy centroid init on the first training batch)."""
        for t in (self.query_embed_S, self.query_embed_X):
            if t is not None:
                t.sync_shadow_to_weight()

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
        s_blanked: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass with separate blanked (query) and actual (key) X.

        Args:
            source_tensor: S with actual values, shape (B, L_S, features).
            x_blanked: X with value column zeroed, shape (B, L_X, features).
                Used as the query input Ã¢â‚¬â€ only variable-identity embedding active.
            x_actual: X with real values, shape (B, L_X, features).
                Used as the key/value input Ã¢â‚¬â€ full embedding (identity + value).
            oracle: If True, bypass QK^T and use the hard mask directly as
                attention weights (inherited from CausalCrossAttention oracle mode).
            oracle_combined_mask: Optional (L_X, L_S+L_X) GT DAG combined mask.
                It is INDEPENDENT of ``oracle``: whenever it is provided it is
                intersected with the structural mask and passed as the block's
                ``hard_mask``, so forbidden keys are set to -inf BEFORE the
                softmax.  The three reachable regimes are:

                  mask=None, oracle=False -> learned QK^T, structural mask only.
                  mask=GT,   oracle=False -> learned QK^T renormalised over the
                      TRUE parents ("cheater": the support is given, the weights
                      are still learned).
                  mask=GT,   oracle=True  -> QK^T bypassed entirely; the GT
                      adjacency itself becomes the attention (uniform average
                      over the true parents).

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
          through the block **unchanged** Ã¢â‚¬â€ no residual is added to it.
        * **Value stream** (``x_val``): starts from ``xq_val`` (the value
          embedding of the zero-blanked X) and accumulates the attention
          output, the FFN, and optional final norm via residual connections.
          The forecaster reads from this stream only.

        When ``comps_embed_X="summation"`` a single fused stream is used,
        preserving the original (pre-SVFA) behaviour.
        """
        # ---- Embed (SVFA-aware) ------------------------------------------
        # ModularEmbedding returns either:
        #   Ã¢â‚¬Â¢ a bare tensor (B, L, d)          when comps="summation"
        #   Ã¢â‚¬Â¢ a tuple (struct, val)             when comps="svfa"
        #     struct (B, L, d) Ã¢â‚¬â€ variable identity, used for Q and K
        #     val   (B, L, d) Ã¢â‚¬â€ actual value,     used for V
        def _emb_drop(raw):
            if isinstance(raw, tuple):
                struct, val = raw
                return self.dropout_emb(struct), self.dropout_emb(val)
            return self.dropout_emb(raw), None      # summation: no separate val

        s_struct,  s_val  = _emb_drop(self.embedding_S(X=source_tensor))
        xk_struct, xk_val = _emb_drop(self.embedding_X(X=x_actual))
        # xq_val is the initial *value* stream for x_blanked (embedding of the
        # zeroed value column).  In SVFA mode this is the residual target Ã¢â‚¬â€ it
        # must NOT be discarded.
        xq_struct, xq_val = _emb_drop(self.embedding_X(X=x_blanked))

        # ---- Homogeneous mode: S is ALSO a query (candidate child) --------
        sq_struct = None
        sq_val = None
        if self.homogeneous_nodes:
            if s_blanked is None:
                raise ValueError(
                    "homogeneous_nodes=True requires s_blanked (S with the "
                    "value column zeroed) so S can act as a query."
                )
            sq_struct, sq_val = _emb_drop(self.embedding_S(X=s_blanked))

        # ---- Orthogonal structural stream override -----------------------
        # When struct_embedding_type selects an orthogonal scheme
        # ("orthogonal_learnable" or "orthogonal_fixed"), replace the structural
        # (Q/K) embeddings with the orthogonal outputs (OrthogonalMaskEmbedding
        # or FixedOrthonormalEmbedding respectively).  The value stream (s_val,
        # xk_val, xq_val) is UNCHANGED Ã¢â‚¬â€ it still comes from the standard
        # ModularEmbedding above.
        if self.orth_embed_S is not None:
            assert self.orth_embed_X is not None
            s_struct  = self.dropout_emb(self.orth_embed_S(source_tensor))
            xk_struct = self.dropout_emb(self.orth_embed_X(x_actual))
            xq_struct = self.dropout_emb(self.orth_embed_X(x_blanked))
            if self.homogeneous_nodes:
                sq_struct = self.dropout_emb(self.orth_embed_S(s_blanked))

        # ---- Free query embedding override (X QUERY stream only) ---------
        # When free_query_embedding=True, the X query uses its OWN learnable
        # identity embedding, decoupling it from the X key embedding (xk_struct
        # is left as set above Ã¢â‚¬â€ orthogonal or standard).  Only the STRUCTURAL
        # query is replaced; the value stream (xq_val) is untouched.
        if self.free_query_embedding:
            assert self.query_embed_X is not None
            xq_struct = self.dropout_emb(self.query_embed_X(x_blanked))
            if self.homogeneous_nodes:
                assert self.query_embed_S is not None
                sq_struct = self.dropout_emb(self.query_embed_S(s_blanked))

        # ---- Forbidden-parents prior: orthogonal key exclusion ------------
        # Keep each listed child's query ORTHOGONAL to its forbidden parents'
        # fixed structural keys: q <- q @ P_c (see _project_forbidden_query_rows
        # and __init__).  Applied AFTER dropout_emb so no forbidden component
        # leaks back in; normalize_query inside the blocks only rescales the
        # projected query, preserving orthogonality.  Every downstream consumer
        # (cross block, self block via shared_query, transitive probes, and the
        # structural residual stream) sees the constrained query.
        if self._qfp_children:
            xq_struct = self._project_forbidden_query_rows(
                xq_struct, row_offset=self.S_seq_len
            )
            if self.homogeneous_nodes and sq_struct is not None:
                sq_struct = self._project_forbidden_query_rows(
                    sq_struct, row_offset=0
                )

        # Q/K always use the structural embedding; V uses value embedding if SVFA.
        sx_keys = torch.cat([s_struct, xk_struct], dim=1)   # (B, L_S+L_X, d)
        if s_val is not None and xk_val is not None:
            sx_vals = torch.cat([s_val, xk_val], dim=1)     # SVFA: values for V
        else:
            sx_vals = sx_keys                                # summation: K = V
        x_q_emb = xq_struct                                 # query tensor

        # ---- Value-structure injection identity codes -------------------
        # Per-SOURCE-node identity combined with the value stream before W_V
        # (see __init__; concatenated or summed per ``vsi_mode`` inside the
        # AttentionLayer).  "separate" / "learned_sum" pull dedicated
        # reconstruction-routed tables; "struct_detached" reuses the (detached)
        # structural identity so no gradient leaks into the structure.  vsi_S /
        # vsi_X match the S / X value token order; vsi_SX is their
        # concatenation for the single-block path.  All None when injection is
        # disabled (value stream unchanged).
        vsi_S = None
        vsi_X = None
        vsi_SX = None
        if self.inject_value_structure:
            if self.value_structure_injection in ("separate", "learned_sum"):
                assert self.val_id_embed_S is not None
                assert self.val_id_embed_X is not None
                vsi_S = self.dropout_emb(self.val_id_embed_S(source_tensor))
                vsi_X = self.dropout_emb(self.val_id_embed_X(x_actual))
            else:  # "struct_detached": reuse the structural identity, detached.
                vsi_S = s_struct.detach()
                vsi_X = xk_struct.detach()
            vsi_SX = torch.cat([vsi_S, vsi_X], dim=1)       # (B, L_S+L_X, d)

        # ---- Value-structure QUERY injection identity code ---------------
        # Per-X-node (child) identity feeding the additive W_V^q term.  Same
        # code for the S->X, X->X and single-block paths (queries are X nodes).
        vsq_X = None
        if self.inject_value_structure_query:
            if self.value_structure_query_injection in ("separate", "learned_sum"):
                assert self.val_q_id_embed_X is not None
                vsq_X = self.dropout_emb(self.val_q_id_embed_X(x_blanked))
            else:  # "struct_detached": reuse the X identity, detached.
                vsq_X = xk_struct.detach()

        # Homogeneous mode: the query-identity code covers ALL N children.
        vsq_all = vsq_X
        if self.homogeneous_nodes and self.inject_value_structure_query:
            if self.value_structure_query_injection in ("separate", "learned_sum"):
                assert self.val_q_id_embed_S is not None
                vsq_S = self.dropout_emb(self.val_q_id_embed_S(s_blanked))
            else:  # "struct_detached"
                vsq_S = s_struct.detach()
            vsq_all = torch.cat([vsq_S, vsq_X], dim=1)      # (B, N, d)

        # ---- Prior-softmax gain tensors (value-identity + value stream) ---
        # The gain score's data term is built from the reconstruction-routed
        # value-identity tables and the value stream Ã¢â‚¬â€ NEVER the structural
        # embeddings (the gain carries no structural signal).  key = value +
        # source identity; query = child identity (the query value is blanked
        # by design).  Consumed by the AttentionLayer only when the inner
        # GainSoftmax module is active (gain_lambda > 0); None otherwise.
        gain_query = None
        gain_key_cross = None
        gain_key_self = None
        gain_key_all = None
        if self.use_gain_softmax and self.gain_data:
            if self.homogeneous_nodes:
                gain_query = vsq_all
                gain_key_all = torch.cat([s_val, xk_val], dim=1) + vsi_SX
            else:
                gain_query = vsq_X
                gain_key_cross = s_val + vsi_S               # (B, L_S, d)
                gain_key_self = xk_val + vsi_X               # (B, L_X, d)

        if self.homogeneous_nodes:
            # ==============================================================
            # HOMOGENEOUS MODE Ã¢â‚¬â€ ONE square (N, N) self-attention block.
            # Q: [S_blanked, X_blanked] struct   (B, N, d)
            # K: [S_actual,  X_actual ] struct   (B, N, d)
            # V: [S_actual,  X_actual ] val      (B, N, d)  (= K in summation)
            # The whole datastream is passed to the attention and the regression
            # reconstructs the S variables too Ã¢â‚¬â€ no SÃ¢â€ â€™X direction is assumed.
            # ==============================================================
            all_q_emb = torch.cat([sq_struct, x_q_emb], dim=1)   # (B, N, d)
            hard_mask = self.homogeneous_mask
            if self.qfp_hard_mask is not None:
                # Forbidden-parents prior (mask=True rows): zero the forbidden
                # parent columns on top of the orthogonal query projection.
                hard_mask = hard_mask * self.qfp_hard_mask
            if self.qsp_hard_mask is not None:
                # Source-nodes prior (mask=True rows): zero the source ROWS
                # (incoming edges) of the homogeneous mask.
                hard_mask = hard_mask * self.qsp_hard_mask
            if oracle_combined_mask is not None:
                # (N, N) GT DAG intersected with the architectural constraints.
                hard_mask = oracle_combined_mask * hard_mask
            attn_out, attention_weights, _aux = self.attention(
                query=all_q_emb,
                key=sx_keys,
                value=sx_vals,
                mask_miss_k=None,
                mask_miss_q=None,
                pos=None,
                causal_mask=False,
                hard_mask=hard_mask,
                oracle=oracle,
                value_structure=vsi_SX,
                value_structure_query=vsq_all,
                transitive_cfg=self._transitive_cfg,
                gain_query=gain_query,
                gain_key=gain_key_all,
            )
        elif self.cross_only:
            # ==============================================================
            # CROSS-ONLY MODE Ã¢â‚¬â€ ONE combined block (vanilla transformer).
            # Q: X_blanked struct                (B, L_X, d)
            # K: [S_actual ; X_actual] struct    (B, L_S+L_X, d)
            # V: [S_actual ; X_actual] val       (B, L_S+L_X, d)  (= K in summation)
            # A single softmax normalises over the S and X parents JOINTLY, so
            # the two parent families compete on one simplex Ã¢â‚¬â€ no direction gate
            # and no re-fusion of two separately normalised posteriors.
            # ==============================================================
            hard_mask = self.combined_mask
            if self.qfp_hard_mask is not None:
                # Forbidden-parents prior (mask=True rows): zero the forbidden
                # parent columns on top of the orthogonal query projection.
                hard_mask = hard_mask * self.qfp_hard_mask
            if self.qsp_hard_mask is not None:
                # Source-nodes prior (mask=True rows): zero the source ROWS
                # (incoming edges) of the combined mask.
                hard_mask = hard_mask * self.qsp_hard_mask
            if oracle_combined_mask is not None:
                # (L_X, L_S+L_X) GT DAG intersected with the architectural
                # constraints (zero diagonal on the X block).
                hard_mask = oracle_combined_mask * hard_mask
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
                value_structure=vsi_SX,
                value_structure_query=vsq_X,
            )
        else:

            # ==============================================================
            # SPLIT MODE Ã¢â‚¬â€ SÃ¢â€ â€™X cross block + direction-aware XÃ¢â€ â€™X self block.
            # ==============================================================
            # Slice the GT (L_X, L_S+L_X) combined mask into the two per-block
            # hard masks so each block receives its own GT adjacency.
            cross_hard = self.cross_mask
            self_hard = self.self_mask
            if self.qfp_hard_mask is not None:
                # Forbidden-parents prior (mask=True rows): slice the combined
                # (L_X, L_S+L_X) mask into the per-block hard masks.
                cross_hard = cross_hard * self.qfp_hard_mask[:, : self.S_seq_len]
                self_hard = self_hard * self.qfp_hard_mask[:, self.S_seq_len :]
            if self.qsp_hard_mask is not None:
                # Source-nodes prior (mask=True rows): zero the source ROWS
                # (incoming edges) Ã¢â‚¬â€ slice the combined (L_X, L_S+L_X) mask.
                cross_hard = cross_hard * self.qsp_hard_mask[:, : self.S_seq_len]
                self_hard = self_hard * self.qsp_hard_mask[:, self.S_seq_len :]
            if oracle_combined_mask is not None:
                cross_hard = oracle_combined_mask[:, : self.S_seq_len] * cross_hard
                self_hard = oracle_combined_mask[:, self.S_seq_len :] * self_hard

            # ---- Transitive correction, PASS 1 (split orchestration) -----
            # Neither block alone sees a square posterior: the mediator k of a
            # path j -> k -> i ranges over the X nodes (the S rows are empty by
            # construction), while the parents j span [S ; X].  So the LAYER
            # probes BOTH blocks' deterministic posteriors, fuses them into the
            # (L_X, L_S+L_X) graph, pads it to the square (N, N) with zero
            # S-rows (the descendant_mask convention), computes the shrink
            # weights ONCE, and scatters the per-block slices into the two
            # forward calls below.  The probes are dropout-free and consume no
            # RNG, so a disabled correction leaves the forward bit-identical.
            tc_cross = None
            tc_self = None
            if self._transitive_cfg is not None and not oracle:
                with torch.no_grad():
                    # Mirror the shared-projection choices of the scored pass
                    # below, but WITHOUT dropout (deterministic trigger).
                    probe_self_q = (
                        self.attention.query_projection(x_q_emb)
                        if self.shared_query
                        else xk_struct
                    )
                    probe_self_k = (
                        self.attention.key_projection(xk_struct)
                        if self.shared_key
                        else xk_struct
                    )
                    pi_cross, k_cross = self.attention.transitive_probe(
                        x_q_emb, s_struct, cross_hard
                    )                                        # (L_X, L_S)
                    pi_self, k_self = self.self_attention.transitive_probe(
                        probe_self_q, probe_self_k, self_hard
                    )                                        # (L_X, L_X)
                    if not self._transitive_frame_checked:
                        # The component removal is only exact on an orthonormal
                        # frame; check the COMBINED [S ; X] keys (per-block
                        # checks would miss mutually rotated frames).
                        assert_orthonormal_frame(
                            torch.cat([k_cross[0], k_self[0]], dim=0),
                            context="split-mode combined [S ; X] key frame",
                        )
                        self._transitive_frame_checked = True
                    # Pad to the square (N, N) graph: nothing points INTO an S
                    # node, so the top L_S rows are zero and every mediator is
                    # an X node Ã¢â‚¬â€ exactly the S -> X -> X grandparent case.
                    pi_sq = pi_cross.new_zeros(self.N, self.N)
                    pi_sq[self.S_seq_len :, :] = torch.cat(
                        [pi_cross, pi_self], dim=-1
                    )
                    W_sq = transitive_weights(
                        pi_sq,
                        alpha=self.transitive_alpha,
                        tnorm=self.transitive_tnorm,
                        margin=self.transitive_margin,
                        symmetric=self.transitive_symmetric,
                        # Only the square X-X self block is symmetrised; the
                        # bipartite S->X block has no reverse entry to balance.
                        symmetric_span=(
                            (self.S_seq_len, self.N)
                            if self.transitive_symmetric
                            else None
                        ),
                    )
                W = W_sq[self.S_seq_len :, :]                    # (L_X, N)
                tc_cross = {**self._transitive_cfg, "W": W[:, : self.S_seq_len]}
                tc_self = {**self._transitive_cfg, "W": W[:, self.S_seq_len :]}

            # ---- SÃ¢â€ â€™X cross block (keys/values = S only) -----------------
            out_sx, attn_sx, aux_sx = self.attention(
                query=x_q_emb,
                key=s_struct,
                value=(s_val if s_val is not None else s_struct),
                mask_miss_k=None,
                mask_miss_q=None,
                pos=None,
                causal_mask=False,
                hard_mask=cross_hard,
                oracle=oracle,
                value_structure=vsi_S,
                value_structure_query=vsq_X,
                transitive_cfg=tc_cross,
                gain_query=gain_query,
                gain_key=gain_key_cross,
            )



            # ---- XÃ¢â€ â€™X self block (keys/values = X only) ------------------
            # Key/value use the X structural identity ``xk_struct`` (a fixed
            # orthonormal frame under struct_embedding_type="orthogonal_fixed").
            # The QUERY, however, depends on ``shared_query`` (see below):
            #   * shared_query=True  Ã¢â€ â€™ the SHARED FREE query ``x_q_emb`` (the
            #     same query the SÃ¢â€ â€™X cross block aligns with), so a single free
            #     query aligns on BOTH the S and X subspaces.  Edge DIRECTION is
            #     then resolved by the CommutatorSelfAttention direction gate on
            #     that query alone (direction_mode="skew_query"), never by the
            #     fixed keys Ã¢â‚¬â€ this removes the spurious XÃ¢â‚¬â€œX coupling that the
            #     old symmetric Ã‚Â½(QKÃ¡Âµâ‚¬Ã¢Ë†â€™KQÃ¡Âµâ‚¬) split introduced with non-orthonormal
            #     shared Q/K.
            #   * shared_query=False Ã¢â€ â€™ the classic Toeplitz split with Q=K on
            #     ``xk_struct`` (original behaviour).
            self_value = xk_val if xk_val is not None else xk_struct

            # ---- Shared structural query projection ---------------------
            # When shared_query=True the self block owns NO W_q (built with
            # query_external=True); project the SHARED FREE query ``x_q_emb``
            # (the SAME query fed to the SÃ¢â€ â€™X cross block) with the CROSS block's
            # W_q, then feed it as a PRE-PROJECTED query.  This is the crux of
            # the design: ONE free query aligns on both the S and X subspaces,
            # while the fixed orthonormal keys ``xk_struct`` no longer participate
            # in a symmetric Ã‚Â½(QKÃ¡Âµâ‚¬Ã¢Ë†â€™KQÃ¡Âµâ‚¬) direction split (which, with flexible
            # non-orthonormal keys, produced spurious XÃ¢â‚¬â€œX edges).  Edge DIRECTION
            # is instead resolved by CommutatorSelfAttention's skew-query
            # generator qÃ¡Âµâ‚¬ÃŽÂ©q on this query alone.  We also apply the cross
            # block's dropout_qkv so the shared query is regularised identically
            # to the cross path.  When shared_query=False the self block projects
            # ``xk_struct`` with its own W_q internally (original behaviour).
            if self.shared_query:
                self_query = self.attention.dropout_qkv(
                    self.attention.query_projection(x_q_emb)
                )
                if tc_cross is not None:
                    # Chain the S-side correction into the SHARED query before
                    # the self block scores it: the budget freed on the S keys
                    # renormalises the SAME query the X-X block reads, so the
                    # freed mass can land on the X parents (joint-cap
                    # reallocation, TRANSITIVE_CORRECTION.md F6).  The self
                    # block then applies its own X-key slice (tc_self) on top.
                    # The weights are gated by the SAME (possibly oracle-
                    # intersected) cross mask the cross block applies; the
                    # probe keys are detached Ã¢â‚¬â€ the correction is a structural
                    # bias, and the sanctioned configuration (frozen
                    # orthonormal frame, no key projection) owns no key
                    # parameters to route gradients to anyway.
                    self_query = correct_query(
                        self_query,
                        k_cross,
                        tc_cross["W"] * cross_hard.to(tc_cross["W"].dtype),
                        delta=self.transitive_delta,
                    )
            else:
                self_query = xk_struct

            # ---- Shared structural key projection -----------------------
            # When shared_key=True the self block owns NO W_K (built with
            # key_external=True); project the X structural identity ``xk_struct``
            # with the CROSS block's W_K, then feed it as a PRE-PROJECTED key.
            # Because the cross W_K is applied to BOTH the S keys (in the SÃ¢â€ â€™X
            # block) and these X keys, S and X keys share the SAME projection Ã¢â‚¬â€
            # under a fixed orthonormal struct embedding W_K is an isometry, so
            # they remain mutually orthogonal.  The shared free query then aligns
            # on genuinely orthonormal key axes for both subspaces, removing the
            # cheap spurious XÃ¢â‚¬â€œX edges.  When shared_key=False the self block
            # projects ``xk_struct`` with its own W_K internally (original
            # behaviour).
            if self.shared_key:
                self_key = self.attention.dropout_qkv(
                    self.attention.key_projection(xk_struct)
                )
            else:
                self_key = xk_struct


            out_xx, attn_xx, aux_xx = self.self_attention(
                query=self_query,
                key=self_key,

                value=self_value,

                mask_miss_k=None,
                mask_miss_q=None,
                pos=None,
                causal_mask=False,
                hard_mask=self_hard,
                oracle=oracle,
                value_structure=vsi_X,
                value_structure_query=vsq_X,
                transitive_cfg=tc_self,
                gain_query=gain_query,
                gain_key=gain_key_self,
            )

            # Diagnostics: the fused (L_X, L_S+L_X) weights AS APPLIED (each
            # wrapper gated its own slice by its hard mask).
            if tc_cross is not None:
                self.last_transitive_W = torch.cat(
                    [
                        self.attention.last_transitive_W,
                        self.self_attention.last_transitive_W,
                    ],
                    dim=-1,
                )



            # ---- Unified re-fusion --------------------------------------
            # Both attention outputs land in the SAME value residual stream,
            # so X is reconstructed jointly from S and X.
            attn_out = out_sx + out_xx
            # Re-assemble the combined (B, L_X, L_S+L_X) posterior so all
            # downstream consumers (split_attention, DAG extraction, SHD) work
            # unchanged.
            attention_weights = torch.cat([attn_sx, attn_xx], dim=-1)

            # Combine aux: sum the two L0 penalties; add the entropies.
            def _combine(a, b):
                if a is None:
                    return b
                if b is None:
                    return a
                return a + b
            l0_sx = aux_sx.get("l0_penalty") if isinstance(aux_sx, dict) else None
            l0_xx = aux_xx.get("l0_penalty") if isinstance(aux_xx, dict) else None
            ent_sx = aux_sx.get("entropy") if isinstance(aux_sx, dict) else None
            ent_xx = aux_xx.get("entropy") if isinstance(aux_xx, dict) else None
            _aux = {
                "entropy": _combine(ent_sx, ent_xx),
                "l0_penalty": _combine(l0_sx, l0_xx),
            }


        # ---- Raw-value adjacency passthrough (NOTEARS/DAGMA-style) --------
        # When enabled, the gated attention posterior A IS the adjacency and
        # acts on the RAW parent values directly: the per-node decoder MLP i
        # receives the elementwise-weighted parent vector
        #     Z_i = (A_i1 * x_1, ..., A_iN * x_N)
        # i.e. exactly the locally-connected first layer of NOTEARS-MLP /
        # DAGMA-MLP.  No value encoder, no W_V, no residual/FFN value stream:
        # the reconstruction gradient w.r.t. A_ij is proportional to the raw
        # x_j, giving the structural parameters direct raw-value contrast.
        if self.raw_value_adjacency:
            s_raw = torch.nan_to_num(source_tensor[:, :, self._raw_val_idx_S])
            x_raw = torch.nan_to_num(x_actual[:, :, self._raw_val_idx_X])
            raw_all = torch.cat([s_raw, x_raw], dim=1)          # (B, L_S+L_X)
            z = attention_weights * raw_all.unsqueeze(1)        # (B, L_q, L_S+L_X)
            z = self.dropout_attn_out(z)
            if self.homogeneous_nodes:
                var_ids = torch.cat(
                    [s_blanked[:, :, 1], x_blanked[:, :, 1]], dim=1
                )
            else:
                var_ids = x_blanked[:, :, 1]
            # Adjacency context: the regressor input z is built from the
            # RETURNED posterior, so the context uses the same weights.
            _ctx = (
                attention_weights.detach()
                if self.per_node_adjacency_context
                else None
            )
            pred_x = self.forecaster(z, var_ids, context=_ctx)
            return pred_x, attention_weights, _aux


        # ---- Residual + Norm 1 -------------------------------------------
        # SVFA mode: the attention output (derived from the value V stream)
        # must go to the VALUE stream only.  The structure stream (x_struct =
        # xq_struct) passes through UNCHANGED Ã¢â‚¬â€ no residual is applied to it.
        # Standard (summation) mode: single fused stream (original behaviour).
        # In homogeneous mode the residual stream spans ALL N query nodes, so
        # the S rows are reconstructed alongside the X rows.
        is_svfa = xq_val is not None
        if is_svfa:
            # Value stream: accumulate the attention output.
            # Structure stream: unchanged (xq_struct is held implicitly; it is
            # not modified here and is not passed to the FFN or forecaster,
            # preserving the separation of the two signals).
            q_val_stream = (
                torch.cat([sq_val, xq_val], dim=1)
                if self.homogeneous_nodes
                else xq_val
            )
            x = self.norm1(q_val_stream + self.dropout_attn_out(attn_out))
        else:
            q_struct_stream = (
                torch.cat([sq_struct, x_q_emb], dim=1)
                if self.homogeneous_nodes
                else x_q_emb
            )
            x = self.norm1(q_struct_stream + self.dropout_attn_out(attn_out))

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
        if self.per_node_output:
            # Per-node decoder: route each token to its own MLP using the
            # variable-ID column.  In homogeneous mode the query stream spans
            # [S ; X], so the IDs are concatenated; otherwise only X IDs.
            if self.homogeneous_nodes:
                var_ids = torch.cat(
                    [s_blanked[:, :, 1], x_blanked[:, :, 1]], dim=1
                )
            else:
                var_ids = x_blanked[:, :, 1]
            _ctx = (
                self._adjacency_context(attention_weights)
                if self.per_node_adjacency_context
                else None
            )
            pred_x = self.forecaster(x, var_ids, context=_ctx)
        else:
            pred_x = self.forecaster(x)

        return pred_x, attention_weights, _aux

    # ------------------------------------------------------------------
    # Per-node adjacency context (detached applied weights)
    # ------------------------------------------------------------------

    def _adjacency_context(
        self, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Detached per-sample APPLIED adjacency rows (B, L_q, L_S+L_X).

        Standard (value-stream) path only: the regressor input is built
        from the TRUE applied weight A inside each gated block (hard mask,
        gain, BKD, top-k blanking, dropout already applied), so the
        context reads the stored ``last_applied_A`` of each block and
        re-concatenates the slices into the canonical column layout.
        Falls back to the returned posterior when the inner module does
        not expose an applied weight (vanilla ScaledDotSoftmax cross-only
        arm: there the returned softmax IS the applied weight).  Always
        detached Ã¢â‚¬â€ the context informs the (nuisance) regressor without
        leaking gradient into the structural parameters.
        """
        if self.homogeneous_nodes or self.cross_only:
            a = getattr(self.attention.inner_attention, "last_applied_A", None)
            ctx = a if a is not None else attention_weights
        else:
            a_cross = getattr(
                self.attention.inner_attention, "last_applied_A", None
            )
            a_self = getattr(
                self.self_attention.inner_attention, "last_applied_A", None
            )
            if a_cross is not None and a_self is not None:
                ctx = torch.cat([a_cross, a_self], dim=-1)
            else:
                ctx = attention_weights
        return ctx.detach()

    # ------------------------------------------------------------------
    # Utility: split the combined attention matrix into SÃ¢â€ â€™X and XÃ¢â€ â€™X parts
    # ------------------------------------------------------------------

    def split_attention(
        self, attention_weights: torch.Tensor
    ):
        """
        Split the attention matrix into the canonical SÃ¢â€ â€™X / XÃ¢â€ â€™X DAG blocks.

        Shape-aware, so the same call works in both modes:

        * split mode ``(B, L_X, L_S+L_X)`` Ã¢â€ â€™ columns are cut at ``L_S``.
        * homogeneous mode ``(B, N, N)``   Ã¢â€ â€™ the X child ROWS are selected
          first, then the S / X parent columns, i.e. ``A[:, L_S:, :L_S]`` and
          ``A[:, L_S:, L_S:]``.  The extra ``XÃ¢â€ â€™S`` / ``SÃ¢â€ â€™S`` blocks (which only
          exist in homogeneous mode) are available via
          :meth:`split_attention_blocks`.

        Returns:
            att_sx: (B, L_X, L_S)  Ã¢â‚¬â€ SÃ¢â€ â€™X sub-matrix (learned SÃ¢â€ â€™X edges).
            att_xx: (B, L_X, L_X)  Ã¢â‚¬â€ XÃ¢â€ â€™X sub-matrix (learned XÃ¢â€ â€™X edges,
                                       diagonal = 0 by construction).
        """
        if self.homogeneous_nodes:
            rows = attention_weights[:, self.S_seq_len :, :]
            return rows[:, :, : self.S_seq_len], rows[:, :, self.S_seq_len :]
        att_sx = attention_weights[:, :, : self.S_seq_len]
        att_xx = attention_weights[:, :, self.S_seq_len :]
        return att_sx, att_xx

    def split_attention_blocks(self, attention_weights: torch.Tensor) -> dict:
        """
        Split a ``(.., N, N)`` homogeneous posterior into all four blocks, using
        the convention that entry ``[i, j]`` is the edge ``j -> i``.

        Returns dict with ``"s_to_x"``, ``"x_to_x"``, ``"x_to_s"``, ``"s_to_s"``.
        In split mode only ``"s_to_x"`` / ``"x_to_x"`` exist (the S rows are not
        modelled), so the other two keys are ``None``.
        """
        S = self.S_seq_len
        if not self.homogeneous_nodes:
            att_sx, att_xx = self.split_attention(attention_weights)
            return {
                "s_to_x": att_sx,
                "x_to_x": att_xx,
                "x_to_s": None,
                "s_to_s": None,
            }
        return {
            "s_to_x": attention_weights[..., S:, :S],
            "x_to_x": attention_weights[..., S:, S:],
            "x_to_s": attention_weights[..., :S, S:],
            "s_to_s": attention_weights[..., :S, :S],
        }

    def source_scores(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Per-node incoming-edge mass (sum over the parent axis).  A LOW value
        means the node has (almost) no parents and is therefore likely a SOURCE.
        Only informative in homogeneous mode, where S nodes are queries too.
        """
        return attention_weights.sum(dim=-1)

    # ------------------------------------------------------------------
    # Unified score tensor for sparsity / NOTEARS regularisers
    # ------------------------------------------------------------------

    def get_score_tensor_for_sparsity(self) -> Optional[torch.Tensor]:
        """Return the combined ``(L_X, L_S + L_X)`` structure score tensor.

        This is the batch-mean, head-averaged edge posterior read by the
        forecaster's score-sparsity (L1) and NOTEARS acyclicity terms.

        * **Single mode**: the combined cross block already exposes the full
          ``(L_X, L_S + L_X)`` score tensor directly.
        * **Split mode**: concatenate the SÃ¢â€ â€™X cross gate posterior
          ``(L_X, L_S)`` with the XÃ¢â€ â€™X ``GatedSelfAttention`` DIRECTED posterior
          ``(L_X, L_X)`` along the last dim so the layout matches single mode
          (the NOTEARS term then acts on the direction-aware XÃ¢â€ â€™X posterior).

        Returns ``None`` if the underlying score tensors have not been populated
        yet (e.g. before the first forward pass).
        """
        cross = getattr(self.attention.inner_attention, "score_tensor_for_sparsity", None)
        if not self.split_xx:
            return cross
        assert self.self_attention is not None
        self_score = getattr(
            self.self_attention.inner_attention, "score_tensor_for_sparsity", None
        )
        if cross is None or self_score is None:
            return None
        return torch.cat([cross, self_score], dim=-1)   # (L_X, L_S + L_X)

    # ------------------------------------------------------------------
    # Prior-softmax gain schedule
    # ------------------------------------------------------------------

    def set_gain_lambda(self, value: float) -> int:
        """Set the prior-softmax gain interpolation weight ``lambda`` on every
        gated block that owns a ``GainSoftmax`` module.  Returns the number of
        blocks updated (0 when the gain is disabled).  Called by the adaptive
        trainer to ramp lambda 0 -> 1 in the gain (final) phase.
        """
        n = 0
        for layer in (self.attention, self.self_attention):
            inner = getattr(layer, "inner_attention", None)
            gs = getattr(inner, "gain_softmax", None)
            if gs is not None:
                gs.set_gain_lambda(value)
                n += 1
        return n

    def gain_lambda(self) -> float:
        """The current gain interpolation weight (0.0 when gain is disabled)."""
        for layer in (self.attention, self.self_attention):
            inner = getattr(layer, "inner_attention", None)
            gs = getattr(inner, "gain_softmax", None)
            if gs is not None:
                return float(gs.gain_lambda.item())
        return 0.0

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
        if self.self_attention is not None:
            for p in self.self_attention.parameters():
                p.requires_grad_(False)

    def unfreeze_attention(self):
        for p in self.attention.parameters():
            p.requires_grad_(True)
        if self.self_attention is not None:
            for p in self.self_attention.parameters():
                p.requires_grad_(True)


    def freeze_forecaster(self):
        for p in self.forecaster.parameters():
            p.requires_grad_(False)

    def unfreeze_forecaster(self):
        for p in self.forecaster.parameters():
            p.requires_grad_(True)
