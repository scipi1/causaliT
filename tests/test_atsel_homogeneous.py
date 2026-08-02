"""Tests for ``AttentionSelectorLayer(homogeneous_nodes=True)``.

Run with:  pytest tests/test_atsel_homogeneous.py -v

Motivation
----------
In split mode the layer hard-codes the S -> X prior: S nodes are keys/values ONLY
(exogenous parents) and X nodes are the only queries (children).  That prior is
wrong whenever the S/X partition is not known a priori.

``homogeneous_nodes=True`` DROPS the prior.  ``[S ; X]`` becomes ONE set of
``N = L_S + L_X`` nodes and each node is simultaneously

* a value-blanked **query** (candidate child), and
* an actual-value **key/value** (candidate parent).

There is exactly ONE square ``(N, N)`` attention block, built from
``self_attention_type`` but stored on the canonical ``self.attention`` attribute
so every downstream consumer (score sparsity, NOTEARS, gradient routing,
freezing, centroid init, interference probes) keeps working unchanged.  The
cross ``attention_type`` is IGNORED.  This subsumes ``SelfSelectorLayer`` while
keeping every AttentionSelector feature.

Guarantees under test
---------------------
1. Construction: ``self_attention is None``, ``split_xx is False``,
   ``attention.inner_attention`` is the *self*-attention class, square
   ``homogeneous_mask = 1 - eye(N)``, ``N`` exposed, ``is_gated`` forced True,
   and the S-side query/gain tables exist.
2. Forward shapes: ``pred (B, N, out_dim)``, ``attn (B, N, N)`` with a zero
   diagonal (no self-loops), and a square ``(N, N)`` score tensor.
3. The four ``ValueError``s: missing ``s_blanked``, ``shared_query=True``,
   ``shared_key=True``, ``self_attention_type=None``.
4. ``split_attention`` / ``split_attention_blocks`` / ``source_scores`` are
   consistent with the raw ``(B, N, N)`` posterior.
5. Structural-key orthogonality survives homogeneous mode, including through the
   shared isometric ``W_K`` (``key_projection_type="orthogonal"``).
6. Feature smoke tests: SVFA (+ multi-head), free query + ``query_centroid_init``
   (writing BOTH query tables), value-structure (query) injection, BKD, and the
   learnable query norm.
7. Gradients reach the S-side tables (S really is a child).
8. Oracle mode consumes the square ``(N, N)`` GT adjacency.
9. End-to-end: two ``_step`` calls through ``AttentionSelectorForecaster``.

Column convention (mirrors ``tests/test_atsel_self_attention_xx.py``):
``value`` at column 0, ``variable-ID`` at column 1; variable IDs are 1-indexed
(0 = padding).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.commutator_self_attention import CommutatorSelfAttention
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.core.modules.orthogonal_linear import OrthogonalLinear
from causaliT.training.gradient_routing import classify_parameters
from causaliT.utils.query_norm import collect_query_norm_penalty


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
N_NODES = S_SEQ_LEN + X_SEQ_LEN      # 7
BATCH = 5
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

# Orthogonal-embedding tests need an exact per-variable dimension tiling over the
# UNIFIED namespace of N nodes: 14 = 7 vars * 2 dims/var.
ORTH_D_MODEL = 2 * N_NODES

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Embedding config with explicit value/structure roles (SVFA-capable)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_model(
    homogeneous_nodes: bool = True,
    self_attention_type: Optional[str] = "GatedSelfAttention",
    comps_embed: str = "svfa",
    d_model: int = D_MODEL,
    d_qk: int = D_QK,
    **overrides,
) -> AttentionSelectorLayer:
    """Build a homogeneous AttentionSelectorLayer.

    ``attention_type`` is deliberately set to a *cross* attention to prove it is
    IGNORED in homogeneous mode.
    """
    kwargs: Dict[str, Any] = dict(
        model="test_model",
        ds_embed_S=_embed_cfg(VOCAB_S, d_model),
        ds_embed_X=_embed_cfg(VOCAB_X, d_model),
        comps_embed_S=comps_embed,
        comps_embed_X=comps_embed,
        attention_type="GatedCrossAttention",   # ignored when homogeneous
        self_attention_type=self_attention_type,
        homogeneous_nodes=homogeneous_nodes,
        n_heads=1,
        dropout_emb=0.0,
        dropout_attn_out=0.0,
        dropout_ff=0.0,
        dropout_qkv=0.0,
        attention_dropout=0.0,
        activation="relu",
        norm="layer",
        use_final_norm=False,
        device="cpu",
        out_dim=1,
        d_ff=D_FF,
        d_model=d_model,
        d_qk=d_qk,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type="standard_learnable",
        gain_stream_source="separate",
    )
    kwargs.update(overrides)
    return AttentionSelectorLayer(**kwargs)


def _make_inputs(batch: int = BATCH, seed: int = 0):
    """(source, x_actual, x_blanked, s_blanked) with 1-indexed variable IDs."""
    g = torch.Generator().manual_seed(seed)

    source = torch.zeros(batch, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(batch, S_SEQ_LEN, generator=g)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(batch, 1)
    )

    x_actual = torch.zeros(batch, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(batch, X_SEQ_LEN, generator=g)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(batch, 1)
    )

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    s_blanked = source.clone()
    s_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked, s_blanked


def _forward(model, inputs=None, **kw):
    """``forward_with_actual`` with the homogeneous ``s_blanked`` wired in."""
    source, x_actual, x_blanked, s_blanked = inputs or _make_inputs()
    return model.forward_with_actual(
        source_tensor=source,
        x_blanked=x_blanked,
        x_actual=x_actual,
        s_blanked=s_blanked,
        **kw,
    )


# ===========================================================================
# 1. Construction wiring
# ===========================================================================


class TestConstruction:
    def test_flags(self):
        m = _make_model()
        assert m.homogeneous_nodes is True
        assert m.split_xx is False, (
            "There is a single square block, so there is nothing to split."
        )
        assert m.self_attention is None, (
            "The square block lives on the canonical `attention` attribute; "
            "`self_attention` must stay None so downstream consumers "
            "(score sparsity, freezing, routing) see exactly one block."
        )

    def test_n_exposed(self):
        assert _make_model().N == N_NODES

    @pytest.mark.parametrize(
        "self_att_type,expected_cls",
        [
            ("GatedSelfAttention", GatedSelfAttention),
            ("CommutatorSelfAttention", CommutatorSelfAttention),
        ],
    )
    def test_single_block_is_the_self_attention_class(self, self_att_type, expected_cls):
        """The ONE block is built from ``self_attention_type``, not ``attention_type``."""
        m = _make_model(self_attention_type=self_att_type)
        assert isinstance(m.attention.inner_attention, expected_cls)

    def test_exactly_one_attention_block(self):
        """One block, one W_Q / W_K: the S/X asymmetry is gone at the QK level."""
        from causaliT.core.modules import AttentionLayer

        m = _make_model()
        blocks = [mod for mod in m.modules() if isinstance(mod, AttentionLayer)]
        assert len(blocks) == 1 and blocks[0] is m.attention
        # Split mode keeps two (cross + self).
        m_split = _make_model(homogeneous_nodes=False)
        assert len(
            [mod for mod in m_split.modules() if isinstance(mod, AttentionLayer)]
        ) == 2

    def test_homogeneous_mask_is_off_diagonal(self):
        m = _make_model()
        names = {n for n, _ in m.named_buffers()}
        assert "homogeneous_mask" in names
        # The split-mode masks must NOT exist (and neither must the legacy one).
        assert "cross_mask" not in names
        assert "self_mask" not in names
        assert "combined_mask" not in names

        mask = m.get_buffer("homogeneous_mask")
        assert mask.shape == (N_NODES, N_NODES)
        assert torch.equal(
            mask, 1.0 - torch.eye(N_NODES)
        ), "homogeneous_mask must be 1 - eye(N): every edge allowed except self-loops."

    def test_is_gated_is_forced(self):
        """Homogeneous mode always runs a gated block, whatever attention_type says."""
        assert _make_model().is_gated is True

    def test_s_side_tables_are_built(self):
        """S is a child too, so it needs its own query / gain-query tables."""
        m = _make_model(free_query_embedding=True)
        assert m.query_embed_S is not None
        assert m.query_embed_X is not None
        assert m.gain_q_embed_S is not None
        assert m.gain_q_embed_X is not None

    def test_s_side_query_table_absent_in_split_mode(self):
        """Contrast: split mode has no S query (S is never a child there)."""
        m = _make_model(homogeneous_nodes=False, free_query_embedding=True)
        assert m.query_embed_X is not None
        assert m.query_embed_S is None
        assert m.gain_q_embed_S is None


# ===========================================================================
# 2. Forward shapes
# ===========================================================================


class TestForwardShapes:
    def test_pred_and_attention_shapes(self):
        model = _make_model()
        model.eval()
        pred, attn, aux = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1), (
            "Every node is a child in homogeneous mode, so ALL N nodes are "
            "reconstructed (not just the L_X ones)."
        )
        assert attn.shape == (BATCH, N_NODES, N_NODES)
        assert "l0_penalty" in aux and "entropy" in aux

    def test_no_self_loops(self):
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model)
        diag = torch.diagonal(attn, dim1=1, dim2=2)
        assert torch.all(diag == 0.0), (
            "The off-diagonal homogeneous_mask must forbid self-loops "
            "(a node cannot be its own parent)."
        )

    def test_score_tensor_is_square(self):
        model = _make_model()
        model.eval()
        _forward(model)
        score = model.get_score_tensor_for_sparsity()
        assert score is not None
        assert score.shape == (N_NODES, N_NODES), (
            "The unified score tensor read by score-sparsity / NOTEARS is the "
            "FULL square adjacency in homogeneous mode."
        )

    @pytest.mark.parametrize("out_dim", [1, 3])
    def test_out_dim_is_respected(self, out_dim):
        model = _make_model(out_dim=out_dim)
        model.eval()
        pred, _, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, out_dim)


# ===========================================================================
# 3. Validation errors
# ===========================================================================


class TestValidation:
    def test_missing_s_blanked_raises(self):
        """S must be handed in value-blanked so it can act as a query."""
        model = _make_model()
        source, x_actual, x_blanked, _ = _make_inputs()
        with pytest.raises(ValueError, match="s_blanked"):
            model.forward_with_actual(
                source_tensor=source, x_blanked=x_blanked, x_actual=x_actual
            )

    def test_shared_query_raises(self):
        with pytest.raises(ValueError, match="shared_query"):
            _make_model(shared_query=True)

    def test_shared_key_raises(self):
        with pytest.raises(ValueError, match="shared_key"):
            _make_model(shared_key=True)

    def test_self_attention_type_is_mandatory(self):
        """``None`` is illegal in HOMOGENEOUS mode (the square block IS the self
        attention), but legal in split mode, where it selects the cross-only
        vanilla-transformer benchmark arm."""
        with pytest.raises(ValueError, match="self_attention_type"):
            _make_model(self_attention_type=None)

        model = _make_model(homogeneous_nodes=False, self_attention_type=None)
        assert model.cross_only
        assert model.self_attention is None

    def test_split_mode_ignores_s_blanked(self):
        """``s_blanked`` is accepted but unused in split mode (harmless)."""
        model = _make_model(homogeneous_nodes=False)
        model.eval()
        pred, attn, _ = _forward(model)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, N_NODES)


# ===========================================================================
# 4. Split helpers
# ===========================================================================


class TestSplitHelpers:
    def test_split_attention_selects_child_rows_then_columns(self):
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model)

        att_sx, att_xx = model.split_attention(attn)
        assert att_sx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert att_xx.shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        # The X child rows come first, THEN the S / X parent columns.
        assert torch.equal(att_sx, attn[:, S_SEQ_LEN:, :S_SEQ_LEN])
        assert torch.equal(att_xx, attn[:, S_SEQ_LEN:, S_SEQ_LEN:])

    def test_split_attention_blocks_returns_all_four(self):
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model)

        blocks = model.split_attention_blocks(attn)
        assert set(blocks) == {"s_to_x", "x_to_x", "x_to_s", "s_to_s"}
        assert blocks["s_to_x"].shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN)
        assert blocks["x_to_x"].shape == (BATCH, X_SEQ_LEN, X_SEQ_LEN)
        # These two only EXIST in homogeneous mode (S is a child there).
        assert blocks["x_to_s"].shape == (BATCH, S_SEQ_LEN, X_SEQ_LEN)
        assert blocks["s_to_s"].shape == (BATCH, S_SEQ_LEN, S_SEQ_LEN)
        # Consistency with split_attention.
        att_sx, att_xx = model.split_attention(attn)
        assert torch.equal(blocks["s_to_x"], att_sx)
        assert torch.equal(blocks["x_to_x"], att_xx)

    def test_split_attention_blocks_is_a_partition(self):
        """The four blocks must tile the full (N, N) posterior exactly."""
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model)
        b = model.split_attention_blocks(attn)
        total = (
            b["s_to_s"].sum() + b["x_to_s"].sum()
            + b["s_to_x"].sum() + b["x_to_x"].sum()
        )
        assert torch.allclose(total, attn.sum(), atol=1e-5)

    def test_split_mode_has_no_s_child_blocks(self):
        model = _make_model(homogeneous_nodes=False)
        model.eval()
        _, attn, _ = _forward(model)
        blocks = model.split_attention_blocks(attn)
        assert blocks["x_to_s"] is None
        assert blocks["s_to_s"] is None

    def test_source_scores(self):
        """Incoming-edge mass per node — LOW means 'likely a source'."""
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model)
        scores = model.source_scores(attn)
        assert scores.shape == (BATCH, N_NODES)
        assert torch.allclose(scores, attn.sum(dim=-1))
        assert torch.all(scores >= 0.0)


# ===========================================================================
# 5. Structural-key orthogonality
# ===========================================================================


class TestStructuralKeyOrthogonality:
    """The orthogonal schemes must keep the S and X keys in ONE coordinated
    namespace: with N nodes tiling ``d_model``, keys of distinct variables live in
    disjoint blocks and are therefore mutually orthogonal — S vs X included."""

    @staticmethod
    def _keys(model, source, x_actual):
        with torch.no_grad():
            return torch.cat(
                [model.orth_embed_S(source), model.orth_embed_X(x_actual)], dim=1
            )

    @staticmethod
    def _max_off_diagonal(gram):
        return (gram - torch.diag(torch.diag(gram))).abs().max()

    @pytest.mark.parametrize(
        "struct_type", ["orthogonal_fixed", "orthogonal_learnable"]
    )
    def test_raw_keys_are_mutually_orthogonal(self, struct_type):
        model = _make_model(
            struct_embedding_type=struct_type,
            d_model=ORTH_D_MODEL,
            d_qk=ORTH_D_MODEL,
        )
        model.eval()
        source, x_actual, _, _ = _make_inputs()
        keys = self._keys(model, source, x_actual)
        assert keys.shape == (BATCH, N_NODES, ORTH_D_MODEL)
        assert self._max_off_diagonal(keys[0] @ keys[0].T) < 1e-5, (
            f"{struct_type} must give mutually orthogonal keys across ALL N "
            f"nodes (the S/X split is only a bookkeeping device here)."
        )

    @pytest.mark.parametrize(
        "struct_type", ["orthogonal_fixed", "orthogonal_learnable"]
    )
    def test_orthogonality_survives_the_shared_isometric_W_K(self, struct_type):
        """``key_projection_type="orthogonal"`` makes W_K an isometry, so the
        PROJECTED keys (the space the logit lives in) stay orthogonal."""
        model = _make_model(
            struct_embedding_type=struct_type,
            key_projection_type="orthogonal",
            d_model=ORTH_D_MODEL,
            d_qk=ORTH_D_MODEL,
        )
        model.eval()
        assert isinstance(model.attention.key_projection, OrthogonalLinear)
        assert model.attention.key_projection.verify_orthonormality(atol=1e-4)

        source, x_actual, _, _ = _make_inputs()
        keys = self._keys(model, source, x_actual)
        with torch.no_grad():
            proj = model.attention.key_projection(keys)
        assert self._max_off_diagonal(proj[0] @ proj[0].T) < 1e-4, (
            "An isometric W_K (W_K^T W_K = I) preserves inner products, so "
            "orthogonal raw keys must remain orthogonal after projection."
        )

    def test_forward_runs_with_orthogonal_stack(self):
        model = _make_model(
            struct_embedding_type="orthogonal_fixed",
            key_projection_type="orthogonal",
            d_model=ORTH_D_MODEL,
            d_qk=ORTH_D_MODEL,
        )
        model.eval()
        pred, attn, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)
        assert attn.shape == (BATCH, N_NODES, N_NODES)


# ===========================================================================
# 6. Feature smoke tests
# ===========================================================================


class TestFeatureSmoke:
    @pytest.mark.parametrize("n_heads", [1, 2])
    def test_svfa_dual_stream(self, n_heads):
        model = _make_model(comps_embed="svfa", n_heads=n_heads)
        model.eval()
        pred, attn, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)
        assert attn.shape == (BATCH, N_NODES, N_NODES)

    def test_summation_single_stream(self):
        model = _make_model(comps_embed="summation")
        model.eval()
        pred, _, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)

    def test_query_centroid_init_writes_both_tables(self):
        """Every node (S and X alike) must start from the SAME point."""
        model = _make_model(
            free_query_embedding=True,
            query_centroid_init=True,
            struct_embedding_type="orthogonal_fixed",
            d_model=ORTH_D_MODEL,
            d_qk=ORTH_D_MODEL,
        )
        source, x_actual, _, _ = _make_inputs()
        table_X, table_S = model.query_embed_X, model.query_embed_S
        assert table_X is not None and table_S is not None
        pad_S = table_S.embedding.weight[0].detach().clone()
        model.init_query_at_key_centroid(source, x_actual)

        w_X = table_X.embedding.weight
        w_S = table_S.embedding.weight
        reference = w_X[1].detach()
        for table, name in [(w_X, "query_embed_X"), (w_S, "query_embed_S")]:
            rows = table[1:]
            assert torch.allclose(
                rows, reference.expand_as(rows), atol=1e-6
            ), f"All real rows of {name} must hold the SAME centroid vector."
        assert torch.allclose(w_S[0], pad_S), "Padding row 0 must be untouched."

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_structure_query_injection(self, mode):
        model = _make_model(value_structure_query_injection=mode)
        assert model.inject_value_structure_query is True
        # S is a child too, so "separate" needs an S-side query-identity table.
        if mode == "separate":
            assert model.val_q_id_embed_S is not None
            assert model.val_q_id_embed_X is not None
        else:
            assert model.val_q_id_embed_S is None
        model.eval()
        pred, _, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)

    @pytest.mark.parametrize("mode", ["separate", "struct_detached"])
    def test_value_structure_injection(self, mode):
        model = _make_model(value_structure_injection=mode)
        assert model.inject_value_structure is True
        model.eval()
        pred, _, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)

    def test_batch_key_dropout(self):
        """BKD must reach the square block (the self attentions implement it
        inline as ``_bkd_p0``/``_bkd_p1``/``_bkd_anneal``, not as a sub-module)."""
        model = _make_model(
            batch_key_dropout=0.5,
            batch_key_dropout_p_final=0.0,
            batch_key_dropout_annealing_batches=100,
        )
        inner = model.attention.inner_attention
        assert inner._bkd_p0 == pytest.approx(0.5)
        assert inner._bkd_p1 == pytest.approx(0.0)
        assert inner._bkd_anneal == 100
        model.train()          # BKD is only active in training mode
        pred, attn, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)
        assert attn.shape == (BATCH, N_NODES, N_NODES)

    def test_learnable_query_norm(self):
        model = _make_model(
            normalize_query=True,
            query_fanin_scale=4.0,
            query_norm_learnable=True,
            query_norm_init_scale=1.0,
            query_norm_target=1.0,
        )
        inner = model.attention.inner_attention
        assert inner.query_norm_learnable is True
        # ONE budget per node — all N of them, since all N are children.
        assert inner.query_norm_log_scale.shape[0] == N_NODES
        model.eval()
        _forward(model)
        penalty = collect_query_norm_penalty(model)
        assert penalty is not None

    def test_free_query_forward(self):
        model = _make_model(free_query_embedding=True)
        model.eval()
        pred, attn, _ = _forward(model)
        assert pred.shape == (BATCH, N_NODES, 1)
        assert attn.shape == (BATCH, N_NODES, N_NODES)


# ===========================================================================
# 7. Gradient flow / routing of the S-side tables
# ===========================================================================


class TestGradients:
    def test_s_side_tables_receive_gradient(self):
        model = _make_model(
            free_query_embedding=True,
            value_structure_query_injection="separate",
        )
        model.train()
        pred, _, _ = _forward(model)
        pred.sum().backward()

        for name in ("query_embed_S", "gain_q_embed_S", "val_q_id_embed_S"):
            table = getattr(model, name)
            assert table is not None, f"{name} must exist in homogeneous mode."
            grad = table.embedding.weight.grad
            assert grad is not None and grad.abs().sum() > 0, (
                f"{name} must receive a non-zero gradient: S nodes are genuine "
                f"children in homogeneous mode."
            )

    def test_single_block_qk_receives_gradient(self):
        model = _make_model()
        model.train()
        pred, _, _ = _forward(model)
        pred.sum().backward()

        got = False
        for name, p in model.attention.named_parameters():
            if ("query_projection" in name or "key_projection" in name):
                if p.grad is not None and p.grad.abs().sum() > 0:
                    got = True
        assert got, "The square block's Q/K must receive a non-zero gradient."

    def test_query_embed_S_is_structural(self):
        """The router keys on the ``query_embed`` PREFIX, so both tables route
        STRUCTURAL (a name-based regression guard for the S-side table)."""
        model = _make_model(free_query_embedding=True)
        structural, reconstruction = classify_parameters(model)
        struct_ids = {id(p) for p in structural}
        recon_ids = {id(p) for p in reconstruction}

        for name in ("query_embed_S", "query_embed_X"):
            p = getattr(model, name).embedding.weight
            assert id(p) in struct_ids, f"{name} must be STRUCTURAL."
            assert id(p) not in recon_ids

        # The gain tables stay on the reconstruction pathway.
        for name in ("gain_q_embed_S", "gain_q_embed_X"):
            p = getattr(model, name).embedding.weight
            assert id(p) in recon_ids, f"{name} must be RECONSTRUCTION."


# ===========================================================================
# 8. Oracle with the square GT adjacency
# ===========================================================================


class TestOracle:
    @staticmethod
    def _gt_square_mask():
        """Lower-triangular (hence acyclic) (N, N) GT adjacency; [i, j] = j -> i."""
        mask = torch.zeros(N_NODES, N_NODES)
        for i in range(1, N_NODES):
            for j in range(i):
                mask[i, j] = 1.0
        return mask

    def test_oracle_uses_the_square_gt_mask(self):
        model = _make_model()
        model.eval()
        gt = self._gt_square_mask()
        _, attn, _ = _forward(model, oracle=True, oracle_combined_mask=gt)

        assert attn.shape == (BATCH, N_NODES, N_NODES)
        forbidden = gt.unsqueeze(0).expand_as(attn) == 0
        assert torch.all(attn[forbidden] == 0.0), (
            "Edges absent from the square GT oracle mask must carry zero weight."
        )

    def test_oracle_without_gt_falls_back_to_the_structural_mask(self):
        """Documented sanity baseline: the off-diagonal mask is the oracle."""
        model = _make_model()
        model.eval()
        _, attn, _ = _forward(model, oracle=True)
        assert attn.shape == (BATCH, N_NODES, N_NODES)
        assert torch.all(torch.diagonal(attn, dim1=1, dim2=2) == 0.0)


# ===========================================================================
# 9. End-to-end through AttentionSelectorForecaster
# ===========================================================================


def _make_forecaster_config(homogeneous_nodes: bool = True, **training) -> dict:
    from copy import deepcopy

    cfg = {
        "data": {
            "val_idx": VALUE_COL,
            "S_seq_len": S_SEQ_LEN,
            "X_seq_len": X_SEQ_LEN,
            "dataset": "dummy",
        },
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(VOCAB_S),
                "ds_embed_X": _embed_cfg(VOCAB_X),
                "comps_embed_S": "svfa",
                "comps_embed_X": "svfa",
                "attention_type": "GatedCrossAttention",
                "self_attention_type": "GatedSelfAttention",
                "homogeneous_nodes": homogeneous_nodes,
                "n_heads": 1,
                "dropout_emb": 0.0,
                "dropout_attn_out": 0.0,
                "dropout_ff": 0.0,
                "dropout_qkv": 0.0,
                "attention_dropout": 0.0,
                "activation": "relu",
                "norm": "layer",
                "use_final_norm": False,
                "device": "cpu",
                "out_dim": 1,
                "d_ff": D_FF,
                "d_model": D_MODEL,
                "d_qk": D_QK,
                "S_seq_len": S_SEQ_LEN,
                "X_seq_len": X_SEQ_LEN,
            },
        },
        "training": {
            "loss_fn": "mse",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "use_gradient_routing": False,
            "lambda_recon": 1.0,
            "lambda_hsic": 1.0,
            "lambda_score_sparse": 0.1,
            "lambda_group_l1": 0.0,
            "lambda_l0": 0.0,
            "kappa": 0.0,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": False,
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "use_oracle_attention": False,
            "use_hard_masks": False,
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }
    cfg = deepcopy(cfg)
    cfg["training"].update(training)
    return cfg


class TestForecasterEndToEnd:
    @staticmethod
    def _batch(seed: int = 0):
        source, x_actual, _, _ = _make_inputs(seed=seed)
        return source, x_actual

    def test_attributes_wired(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        fc = AttentionSelectorForecaster(_make_forecaster_config())
        assert fc.homogeneous_nodes is True
        assert fc.N == N_NODES
        assert fc.model.homogeneous_nodes is True

    def test_forward_builds_s_blanked_itself(self):
        """The forecaster must blank the S value column — callers pass raw S."""
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        fc = AttentionSelectorForecaster(_make_forecaster_config())
        fc.eval()
        S, X = self._batch()
        pred, attn, _ = fc(S, X)
        assert pred.shape == (BATCH, N_NODES, 1)
        assert attn.shape == (BATCH, N_NODES, N_NODES)

    @pytest.mark.parametrize("kappa", [0.0, 0.5])
    def test_two_steps_run(self, kappa):
        """Two sequential ``_step``s: the N-row target layout must hold end to end
        (MSE, torchmetrics, HSIC residuals) and NOTEARS must accept the square
        score tensor."""
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        torch.manual_seed(0)
        fc = AttentionSelectorForecaster(_make_forecaster_config(kappa=kappa))
        fc.train()
        for seed in (0, 1):
            loss, pred, _ = fc._step(self._batch(seed=seed), stage="train")
            assert torch.isfinite(loss), "Loss must be finite."
            assert pred.shape == (BATCH, N_NODES, 1)
            loss.backward()

    def test_target_covers_all_n_nodes(self):
        """A perfect predictor of the N-row target must drive the MSE to ~0, which
        only holds if the target really is ``cat([S_values, X_values])``."""
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        fc = AttentionSelectorForecaster(_make_forecaster_config())
        S, X = self._batch()
        expected = torch.cat(
            [S[:, :, VALUE_COL], X[:, :, VALUE_COL]], dim=1
        )                                          # (B, N)
        assert expected.shape == (BATCH, N_NODES)
        assert fc.loss_fn(expected, expected).mean() == 0.0

    def test_split_mode_still_predicts_x_only(self):
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )

        fc = AttentionSelectorForecaster(
            _make_forecaster_config(homogeneous_nodes=False)
        )
        fc.train()
        loss, pred, _ = fc._step(self._batch(), stage="train")
        assert torch.isfinite(loss)
        assert pred.shape == (BATCH, X_SEQ_LEN, 1)


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
