"""Tests for AttentionSelectorLayer ``free_query_embedding`` (decoupled X Q/K).

Run with:  pytest tests/test_atsel_free_query.py -v

Motivation
----------
In the combined cross-attention each predicted node X_i is used BOTH as a
**key** (offered as a candidate parent to other X_j) and as a **query**
(selecting its own parents).  With a single shared embedding, a gradient that
updates "X_i-as-child" (query) also perturbs "X_i-as-parent" (key), so the
model cannot learn ``X_i <- S`` and ``X_i <- X_j`` independently.

``free_query_embedding=True`` gives the X query its own free
(``FreeQueryEmbedding``) identity lookup, decoupling the two roles.  These
tests verify:

1. Construction wiring (``query_embed_X`` present iff flag True).
2. Forward shapes are unchanged (svfa + summation, with/without orthogonal).
3. The query embedding actually drives the attention Query (perturbing it
   changes the attention weights).
4. **Key/query decoupling**: perturbing the X *key* embedding leaves the
   attention *query* free -- the query no longer depends on ``embedding_X``.
5. Gradient routing classifies ``query_embed_X`` as STRUCTURAL (feeds Q only),
   and the interference-block mapper assigns it its own block.

Column convention
-----------------
This test uses the PRODUCTION feature layout that ``OrthogonalMaskEmbedding``
and ``FreeQueryEmbedding`` assume: ``value`` at column 0, ``variable-ID`` at
column 1.  The embedding configs map ``idx: 0 -> value (linear)`` and
``idx: 1 -> variable (nn_embedding)`` to stay consistent with that layout.
Variable IDs are 1-indexed (0 = padding) and never exceed the sequence length
so the orthogonal masks (one partition per variable) can index them.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.free_query_embedding import FreeQueryEmbedding
from causaliT.training.gradient_routing import classify_parameters
from causaliT.training.interference_utils import (
    build_interference_blocks,
    _block_for_param,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
# Vocab sizes only bound the nn_embedding lookup; must exceed the max 1-indexed
# variable ID (= seq_len) so both summation and orthogonal paths are valid.
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0     # production: value at column 0
VAR_COL = 1       # production: variable-ID at column 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """SVFA-split embedding config (value=idx0, variable=idx1)."""
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


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Standard summation embedding config (value=idx0, variable=idx1)."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
        ],
    }


def _make_model(
    comps_embed_X: str = "svfa",
    free_query_embedding: bool = True,
    orthogonal_struct_embedding: bool = False,
) -> AttentionSelectorLayer:
    ds_embed_X = (
        _svfa_embed_cfg(VOCAB_X)
        if comps_embed_X == "svfa"
        else _summation_embed_cfg(VOCAB_X)
    )
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=ds_embed_X,
        comps_embed_S="summation",
        comps_embed_X=comps_embed_X,
        attention_type="ScaledDotProduct",
        # MANDATORY since the legacy cross-only variant was removed.
        self_attention_type="GatedSelfAttention",

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
        d_model=D_MODEL,
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type=(
            "orthogonal_learnable"
            if orthogonal_struct_embedding
            else "standard_learnable"
        ),
        free_query_embedding=free_query_embedding,
    )


def _make_inputs():
    """(source, x_actual, x_blanked) with value at col 0, variable-ID at col 1.

    Variable IDs are the fixed 1-indexed identities ``1..seq_len`` (0 = padding),
    guaranteeing they stay within every embedding's valid index range (including
    the per-variable orthogonal masks).
    """
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )

    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0  # blank the value column for queries
    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Construction wiring
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_flag_stored(self):
        assert _make_model(free_query_embedding=True).free_query_embedding is True
        assert _make_model(free_query_embedding=False).free_query_embedding is False

    def test_query_embed_present_only_when_enabled(self):
        m_on = _make_model(free_query_embedding=True)
        m_off = _make_model(free_query_embedding=False)
        assert isinstance(m_on.query_embed_X, FreeQueryEmbedding)
        assert m_off.query_embed_X is None

    def test_query_embed_table_shape(self):
        m = _make_model(free_query_embedding=True)
        assert m.query_embed_X is not None
        # num_variables + var_id_offset rows (padding index 0), d_model cols.
        assert m.query_embed_X.embedding.weight.shape == (X_SEQ_LEN + 1, D_MODEL)


# ---------------------------------------------------------------------------
# 2. Forward shapes across regimes
# ---------------------------------------------------------------------------


class TestForwardShapes:
    @pytest.mark.parametrize("comps", ["svfa", "summation"])
    @pytest.mark.parametrize("orth", [False, True])
    def test_shapes(self, comps, orth):
        model = _make_model(
            comps_embed_X=comps,
            free_query_embedding=True,
            orthogonal_struct_embedding=orth,
        )
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 3. The query embedding drives the attention Query
# ---------------------------------------------------------------------------


class TestQueryEmbeddingDrivesAttention:
    def test_perturbing_query_embedding_changes_attention(self):
        model_a = _make_model(free_query_embedding=True)
        model_b = _make_model(free_query_embedding=True)
        model_a.eval(); model_b.eval()
        model_b.load_state_dict(model_a.state_dict())
        assert model_b.query_embed_X is not None

        # Perturb ONLY the free query embedding table in model_b.
        with torch.no_grad():
            model_b.query_embed_X.embedding.weight += (
                torch.randn_like(model_b.query_embed_X.embedding.weight) * 2.0
            )

        source, x_actual, x_blanked = _make_inputs()
        with torch.no_grad():
            _, attn_a, _ = model_a.forward_with_actual(source, x_blanked, x_actual)
            _, attn_b, _ = model_b.forward_with_actual(source, x_blanked, x_actual)

        assert not torch.allclose(attn_a, attn_b), (
            "Perturbing query_embed_X must change the attention weights -- it is "
            "the sole source of the attention Query when free_query_embedding=True."
        )


# ---------------------------------------------------------------------------
# 4. Key / query decoupling
# ---------------------------------------------------------------------------


class TestKeyQueryDecoupling:
    def test_query_independent_of_key_embedding(self):
        """Changing the X KEY embedding must not affect the X QUERY."""
        model = _make_model(free_query_embedding=True)
        model.eval()
        assert model.query_embed_X is not None
        _, _, x_blanked = _make_inputs()

        q_before = model.query_embed_X(x_blanked).detach().clone()

        with torch.no_grad():
            for _, p in model.embedding_X.named_parameters():
                p += torch.randn_like(p) * 3.0

        q_after = model.query_embed_X(x_blanked).detach().clone()

        assert torch.allclose(q_before, q_after), (
            "The free query embedding must be independent of embedding_X: "
            "changing the X key embedding must not affect the X query."
        )

    def test_query_ignores_value_column(self):
        """FreeQueryEmbedding depends only on the variable-ID column."""
        model = _make_model(free_query_embedding=True)
        assert model.query_embed_X is not None
        _, x_actual, x_blanked = _make_inputs()
        q_blanked = model.query_embed_X(x_blanked)
        q_actual = model.query_embed_X(x_actual)  # differs only in value column
        assert torch.allclose(q_blanked, q_actual), (
            "The query embedding must ignore the value column (identity only)."
        )


# ---------------------------------------------------------------------------
# 5. Gradient-routing + interference-block classification
# ---------------------------------------------------------------------------


class TestParameterClassification:
    def test_query_embed_is_structural(self):
        model = _make_model(free_query_embedding=True)
        assert model.query_embed_X is not None
        structural, reconstruction = classify_parameters(model)
        struct_ids = {id(p) for p in structural}
        recon_ids = {id(p) for p in reconstruction}

        q_params = list(model.query_embed_X.parameters())
        assert len(q_params) > 0
        for p in q_params:
            assert id(p) in struct_ids, (
                "query_embed_X parameters must be classified as STRUCTURAL "
                "(they feed the attention Query only)."
            )
            assert id(p) not in recon_ids

    def test_interference_block_label(self):
        assert _block_for_param("query_embed_X.embedding.weight") == "query_embedding_X"

        model = _make_model(free_query_embedding=True)
        blocks = build_interference_blocks(model)
        assert "query_embedding_X" in blocks
        assert len(blocks["query_embedding_X"]) > 0


# ---------------------------------------------------------------------------
# 6. Backward pass smoke: gradients reach the query table
# ---------------------------------------------------------------------------


class TestBackward:
    def test_query_embed_receives_gradient(self):
        model = _make_model(free_query_embedding=True)
        model.train()
        assert model.query_embed_X is not None
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()

        w = model.query_embed_X.embedding.weight
        assert w.grad is not None and w.grad.abs().sum() > 0, (
            "query_embed_X must receive a non-zero gradient from the loss."
        )


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
