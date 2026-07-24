"""Tests for removing / freezing the structural query & key projections.

Run with:  pytest tests/test_atsel_remove_freeze_qk.py -v

Motivation
----------
The SELF_ATTENTION investigation (``investigate_S3_X4_spurious_barrier``) found
that the shared structural query projection ``W_q`` couples every node's query
together and does almost all the work of aligning queries to keys, leaving the
per-node query embeddings "numb" (they receive only a single-node gradient while
``W_q`` aggregates a consistent signal from all nodes).  Two levers address this
directly:

* ``remove_query_projection`` / ``remove_key_projection`` — drop ``W_q`` / ``W_K``
  entirely and read the structural query / key straight from the (identity)
  embedding.  Implemented as an ``nn.Identity`` in place of the ``nn.Linear``,
  which requires ``d_model == d_qk * n_heads`` so the pass-through has the right
  shape.
* ``freeze_query_projection`` / ``freeze_key_projection`` — keep ``W_q`` / ``W_K``
  but ``requires_grad_(False)`` at construction.  Because ``classify_parameters``
  skips ``requires_grad=False`` tensors, the freeze survives adaptive
  phase-switches (the ``PhaseController`` only re-enables params it classified).

These tests verify construction, dimension validation, forward/backward
behaviour, and the gradient-routing consequences of each lever.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.training.gradient_routing import classify_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
# remove_* requires d_model == d_qk * n_heads, so keep them equal here.
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 5
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
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
    self_attention_type="GatedSelfAttention",
    shared_query: bool = False,
    free_query_embedding: bool = True,
    remove_query_projection: bool = False,
    remove_key_projection: bool = False,
    freeze_query_projection: bool = False,
    freeze_key_projection: bool = False,
    d_qk: int = D_QK,
) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
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
        d_qk=d_qk,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type="standard_learnable",
        free_query_embedding=free_query_embedding,
        gain_stream_source="separate",
        self_attention_type=self_attention_type,
        shared_query=shared_query,
        remove_query_projection=remove_query_projection,
        remove_key_projection=remove_key_projection,
        freeze_query_projection=freeze_query_projection,
        freeze_key_projection=freeze_key_projection,
    )


def _make_inputs():
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
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


# ---------------------------------------------------------------------------
# 1. Removal — the projection becomes an nn.Identity with no parameters
# ---------------------------------------------------------------------------


class TestRemoval:
    def test_remove_query_projection_is_identity(self):
        m = _make_model(remove_query_projection=True)
        assert isinstance(m.attention.query_projection, nn.Identity)
        # No learnable query-projection parameters on the cross block.
        names = dict(m.attention.named_parameters())
        assert not any("query_projection" in n for n in names), (
            f"removed W_q must register no params: "
            f"{[n for n in names if 'query_projection' in n]}"
        )

    def test_remove_key_projection_is_identity(self):
        m = _make_model(remove_key_projection=True)
        assert isinstance(m.attention.key_projection, nn.Identity)
        names = dict(m.attention.named_parameters())
        assert not any("key_projection" in n for n in names), (
            f"removed W_K must register no params: "
            f"{[n for n in names if 'key_projection' in n]}"
        )

    def test_keep_projection_is_linear(self):
        m = _make_model()
        assert isinstance(m.attention.query_projection, nn.Linear)
        assert isinstance(m.attention.key_projection, nn.Linear)

    def test_remove_both_projections(self):
        m = _make_model(remove_query_projection=True, remove_key_projection=True)
        assert isinstance(m.attention.query_projection, nn.Identity)
        assert isinstance(m.attention.key_projection, nn.Identity)


# ---------------------------------------------------------------------------
# 2. Dimension validation — removal requires d_model == d_qk * n_heads
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    def test_remove_query_bad_dims_raises(self):
        with pytest.raises(ValueError):
            _make_model(remove_query_projection=True, d_qk=D_MODEL // 2)

    def test_remove_key_bad_dims_raises(self):
        with pytest.raises(ValueError):
            _make_model(remove_key_projection=True, d_qk=D_MODEL // 2)

    def test_remove_matched_dims_ok(self):
        # d_qk == d_model → pass-through has the right shape, no error.
        m = _make_model(
            remove_query_projection=True, remove_key_projection=True, d_qk=D_MODEL
        )
        assert isinstance(m.attention.query_projection, nn.Identity)


# ---------------------------------------------------------------------------
# 3. Freezing — requires_grad=False and excluded from gradient routing
# ---------------------------------------------------------------------------


class TestFreezing:
    def test_freeze_query_projection_requires_grad_false(self):
        m = _make_model(freeze_query_projection=True)
        assert isinstance(m.attention.query_projection, nn.Linear)  # still a W_q
        for p in m.attention.query_projection.parameters():
            assert p.requires_grad is False

    def test_freeze_key_projection_requires_grad_false(self):
        m = _make_model(freeze_key_projection=True)
        for p in m.attention.key_projection.parameters():
            assert p.requires_grad is False

    def test_unfrozen_projection_requires_grad_true(self):
        m = _make_model()
        assert any(p.requires_grad for p in m.attention.query_projection.parameters())
        assert any(p.requires_grad for p in m.attention.key_projection.parameters())

    def test_frozen_wq_excluded_from_structural_params(self):
        # classify_parameters skips requires_grad=False tensors, so the frozen
        # W_q is NOT routed to the structural group.  This is exactly what makes
        # the freeze persist across adaptive phase switches (the PhaseController
        # only re-enables params it classified as structural).
        m = _make_model(freeze_query_projection=True)
        structural, _ = classify_parameters(m)
        struct_ids = {id(p) for p in structural}
        for p in m.attention.query_projection.parameters():
            assert id(p) not in struct_ids

    def test_unfrozen_wq_is_structural(self):
        m = _make_model()
        structural, _ = classify_parameters(m)
        struct_ids = {id(p) for p in structural}
        assert any(
            id(p) in struct_ids for p in m.attention.query_projection.parameters()
        )


# ---------------------------------------------------------------------------
# 4. Forward / backward behaviour under removal
# ---------------------------------------------------------------------------


class TestForwardBackward:
    def test_forward_shapes_remove_query(self):
        model = _make_model(remove_query_projection=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, aux = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_forward_shapes_remove_both(self):
        model = _make_model(remove_query_projection=True, remove_key_projection=True)
        model.eval()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
        assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
        assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    def test_frozen_wq_receives_no_gradient(self):
        model = _make_model(freeze_query_projection=True)
        model.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()
        for p in model.attention.query_projection.parameters():
            # requires_grad=False → grad stays None.
            assert p.grad is None

    def test_removed_query_routes_gradient_to_embedding(self):
        # With W_q removed the query IS the structural embedding, so the query
        # gradient must flow into the (free) query embedding table instead.
        model = _make_model(remove_query_projection=True, free_query_embedding=True)
        model.train()
        source, x_actual, x_blanked = _make_inputs()
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
        pred_x.sum().backward()
        got = False
        for name, p in model.named_parameters():
            if "query_embed_X" in name and p.grad is not None:
                if p.grad.abs().sum() > 0:
                    got = True
        assert got, "removed W_q must route the query gradient to the embedding."


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
