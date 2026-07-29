"""Tests for AttentionSelectorLayer ``query_centroid_init``.

Run with:  pytest tests/test_atsel_query_centroid_init.py -v

Motivation
----------
The SELF_ATTENTION spurious-``S3->X4`` investigation
(``experiments/6_INVESTIGATIONS/SELF_ATTENTION/.../investigate_S3_X4_spurious_barrier.ipynb``)
showed the shared query falls into an early spurious basin because different
nodes start pointing in arbitrary directions and one over-represented key (S3,
via X3) wins the directional budget before the true parents pay off.

``query_centroid_init=True`` initialises the free X **query** embedding at the
centroid of the (projected) keys, so EVERY query starts from the SAME point and
reads all candidate parents UNIFORMLY.  For an orthonormal key frame the centroid
yields identical ``<q, k_j>`` for all keys ``j`` -- a symmetric start that lets
HSIC break toward the true parents before the budget saturates.

Design guarantees under test
----------------------------
1. Construction wiring: the flag is stored and requires ``free_query_embedding``.
2. Only the query EMBEDDING is written -- every real row becomes the SAME vector
   (the padding row 0 is left untouched).
3. WITHOUT the query/key projections (orthonormal ``orthogonal_fixed`` frame) the
   resulting per-key alignments ``<q, k_j>`` are UNIFORM across all keys.
4. WITH the query/key projections present the query embedding is inverted through
   ``W_q`` so the PROJECTED query lands exactly on the centroid of the PROJECTED
   keys (the centroid moves with the key projection, as required).
5. Config-template wiring: the flag defaults to false and the
   ``experiment -> model.kwargs`` interpolation propagates it.

Column convention (mirrors ``test_atsel_free_query.py``): ``value`` at column 0,
``variable-ID`` at column 1; variable IDs are 1-indexed (0 = padding).
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
D_QK = 16          # == D_MODEL so remove_*_projection (needs d_qk == d_model) is legal
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
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
    free_query_embedding: bool = True,
    query_centroid_init: bool = False,
    struct_embedding_type: str = "orthogonal_fixed",
    remove_projections: bool = False,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
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
        struct_embedding_type=struct_embedding_type,
        free_query_embedding=free_query_embedding,
        query_centroid_init=query_centroid_init,
        remove_query_projection=remove_projections,
        remove_key_projection=remove_projections,
    )


def _make_inputs():
    """(source, x_actual) with value at col 0, 1-indexed variable IDs at col 1."""
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
    return source, x_actual


def _key_frame(model) -> torch.Tensor:
    """Concatenated (L_S + L_X, d_model) orthonormal key rows (orthogonal_fixed)."""
    assert model.orth_embed_S is not None and model.orth_embed_X is not None
    return torch.cat([model.orth_embed_S.frame, model.orth_embed_X.frame], dim=0)


# ---------------------------------------------------------------------------
# 1. Construction wiring
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_flag_stored(self):
        assert _make_model(query_centroid_init=True).query_centroid_init is True
        assert _make_model(query_centroid_init=False).query_centroid_init is False

    def test_requires_free_query_embedding(self):
        with pytest.raises(ValueError, match="query_centroid_init"):
            _make_model(free_query_embedding=False, query_centroid_init=True)

    def test_init_raises_without_query_table(self):
        """Calling the initialiser on a model with no free query table errors."""
        model = _make_model(free_query_embedding=False, query_centroid_init=False)
        assert model.query_embed_X is None
        source, x_actual = _make_inputs()
        with pytest.raises(RuntimeError, match="free_query_embedding"):
            model.init_query_at_key_centroid(source, x_actual)


# ---------------------------------------------------------------------------
# 2. Only the query embedding is written; every real row becomes identical
# ---------------------------------------------------------------------------


class TestWritesEmbeddingOnly:
    def test_all_real_rows_identical(self):
        model = _make_model(free_query_embedding=True)
        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)

        w = model.query_embed_X.embedding.weight
        rows = w[1:]  # skip padding row 0
        first = rows[0]
        for r in rows[1:]:
            assert torch.allclose(r, first, atol=1e-6), (
                "All real query rows must be initialised to the SAME centroid "
                "vector so every query starts from one point."
            )

    def test_padding_row_untouched(self):
        model = _make_model(free_query_embedding=True)
        w = model.query_embed_X.embedding.weight
        pad_before = w[0].detach().clone()
        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)
        assert torch.allclose(w[0], pad_before), (
            "Row 0 (padding_idx) must not be modified by the centroid init."
        )


# ---------------------------------------------------------------------------
# 3. No projection: uniform alignment across all keys (orthonormal frame)
# ---------------------------------------------------------------------------


class TestUniformAlignmentNoProjection:
    def test_scores_uniform(self):
        model = _make_model(
            free_query_embedding=True,
            struct_embedding_type="orthogonal_fixed",
            remove_projections=True,
        )
        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)

        q = model.query_embed_X.embedding.weight[1].detach()  # centroid (no proj)
        keys = _key_frame(model).detach()                     # (N, d_model)
        scores = keys @ q                                     # (N,)

        # Orthonormal frame => <centroid, k_i> = 1/N for every key i.
        n = keys.shape[0]
        assert torch.allclose(scores, torch.full_like(scores, 1.0 / n), atol=1e-5), (
            f"Per-key alignments must be uniform (~1/{n}); got {scores.tolist()}."
        )
        assert float(scores.std()) < 1e-5


# ---------------------------------------------------------------------------
# 4. With projection: projected query lands on the projected-key centroid
# ---------------------------------------------------------------------------


class TestProjectedCentroidWithProjection:
    def test_projected_query_matches_projected_centroid(self):
        model = _make_model(
            free_query_embedding=True,
            struct_embedding_type="orthogonal_fixed",
            remove_projections=False,
        )
        qproj = getattr(model.attention, "query_projection", None)
        kproj = getattr(model.attention, "key_projection", None)
        assert qproj is not None and kproj is not None, (
            "This test assumes the query/key projections are present."
        )

        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)

        keys = _key_frame(model).detach()                     # (N, d_model)
        with torch.no_grad():
            k_proj = kproj(keys)                              # (N, d_qk)
            centroid = k_proj.mean(dim=0)                     # (d_qk,)
            e = model.query_embed_X.embedding.weight[1]       # written embedding
            q_proj = qproj(e.unsqueeze(0)).squeeze(0)         # (d_qk,)

        assert torch.allclose(q_proj, centroid, atol=1e-4), (
            "The PROJECTED query must equal the centroid of the PROJECTED keys: "
            "the query embedding is inverted through W_q so the centroid moves "
            "correctly with the key projection."
        )

    def test_projected_scores_uniform_when_projection_orthonormal_like(self):
        """Sanity: with projections the written rows are still all identical."""
        model = _make_model(free_query_embedding=True, remove_projections=False)
        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)
        rows = model.query_embed_X.embedding.weight[1:]
        assert torch.allclose(rows, rows[0].expand_as(rows), atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Config-template wiring
# ---------------------------------------------------------------------------


class TestConfigTemplate:
    def test_default_false_and_interpolation(self):
        from omegaconf import OmegaConf

        tmpl = (
            project_root
            / "causaliT"
            / "config"
            / "templates"
            / "config_attention_selector.yaml"
        )
        cfg = OmegaConf.load(str(tmpl))

        # Default disabled.
        assert cfg.experiment.query_centroid_init is False
        assert cfg.model.kwargs.query_centroid_init is False

        # experiment -> model.kwargs interpolation propagates the toggle.
        cfg.experiment.query_centroid_init = True
        assert cfg.model.kwargs.query_centroid_init is True


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
