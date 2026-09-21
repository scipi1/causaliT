"""Tests for AttentionSelectorLayer ``query_parents_prior``.

Run with:  pytest tests/test_atsel_query_parents_prior.py -v

The known-edges prior overwrites the (centroid/default) initialisation for the
listed nodes: each child's free query embedding is placed at the mean of its
KNOWN parents' (projected) keys, and rows marked ``fixed: true`` are frozen
for the rest of training (gradient hook + post-step re-assert).

Node IDs are GLOBAL and 1-based over the concatenated dataset layout: S nodes
are 1..L_S, X nodes are L_S+1..L_S+L_X.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer

D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1
L = S_SEQ_LEN + X_SEQ_LEN   # global ids: S=1..3, X=4..7

VALUE_COL = 0
VAR_COL = 1


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


def _make_model(prior=None, remove_projections: bool = True,
                free_query_embedding: bool = True) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="ScaledDotSoftmax",
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
        struct_embedding_type="orthogonal_fixed",
        free_query_embedding=free_query_embedding,
        query_centroid_init=False,
        remove_query_projection=remove_projections,
        remove_key_projection=remove_projections,
        query_parents_prior=prior,
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
    return source, x_actual


def _key_frame(model) -> torch.Tensor:
    return torch.cat([model.orth_embed_S.frame, model.orth_embed_X.frame], dim=0)


# ---------------------------------------------------------------------------
# 1. Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_prior_normalised_and_stored(self):
        m = _make_model(prior={"5": {"parents": ["1", 2], "fixed": 1}})
        assert m.query_parents_prior == {5: {"parents": [1, 2], "fixed": True}}

    def test_default_is_none(self):
        assert _make_model().query_parents_prior is None

    def test_requires_free_query_embedding(self):
        with pytest.raises(ValueError, match="query_parents_prior"):
            _make_model(prior={5: {"parents": [1], "fixed": False}},
                        free_query_embedding=False)

    def test_child_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_model(prior={L + 1: {"parents": [1], "fixed": False}})

    def test_s_child_rejected_in_split_mode(self):
        with pytest.raises(ValueError, match="S node"):
            _make_model(prior={2: {"parents": [1], "fixed": False}})

    def test_parent_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_model(prior={5: {"parents": [0], "fixed": False}})

    def test_self_parent_rejected(self):
        with pytest.raises(ValueError, match="own parent"):
            _make_model(prior={5: {"parents": [5], "fixed": False}})

    def test_empty_parents_rejected(self):
        with pytest.raises(ValueError, match="no parents"):
            _make_model(prior={5: {"parents": [], "fixed": False}})


# ---------------------------------------------------------------------------
# 2. Init: the child query lands on the mean of its parents' keys
# ---------------------------------------------------------------------------


class TestParentsInit:
    def test_child_aligns_with_parents_only(self):
        prior = {5: {"parents": [1, 2], "fixed": False}}
        model = _make_model(prior=prior)
        source, x_actual = _make_inputs()
        n_fixed = model.init_queries_from_parents(prior, source, x_actual)
        assert n_fixed == 0

        keys = _key_frame(model).detach()                    # (L, d_model)
        q = model.query_embed_X.embedding.weight[5 - S_SEQ_LEN].detach()
        scores = keys @ q                                    # (L,)
        # Orthonormal frame: <q, k_1> = <q, k_2> = 1/2, all others 0.
        assert torch.isclose(scores[0], scores[1], atol=1e-5)
        assert scores[0] > 0.0
        assert torch.allclose(
            scores[2:], torch.zeros_like(scores[2:]), atol=1e-5
        ), f"Non-parent keys must be orthogonal to the query; got {scores}"

    def test_prior_overwrites_centroid_init(self):
        prior = {6: {"parents": [4], "fixed": False}}
        model = _make_model(prior=prior)
        source, x_actual = _make_inputs()
        model.init_query_at_key_centroid(source, x_actual)   # init first
        model.init_queries_from_parents(prior, source, x_actual)  # then prior

        keys = _key_frame(model).detach()
        q = model.query_embed_X.embedding.weight[6 - S_SEQ_LEN].detach()
        scores = keys @ q
        # The prior (single parent 4 = X1) wins over the centroid start.
        assert torch.isclose(scores[3], torch.ones(()), atol=1e-5)
        assert torch.allclose(
            torch.cat([scores[:3], scores[4:]]),
            torch.zeros(L - 1),
            atol=1e-5,
        )
        # Unlisted nodes keep the centroid start (rows 1 and 4 are untouched;
        # row 3 = child 6 was overwritten by the prior).
        rows = model.query_embed_X.embedding.weight
        assert torch.allclose(rows[1], rows[4], atol=1e-6)

    def test_raises_without_query_table(self):
        model = _make_model(free_query_embedding=False)
        source, x_actual = _make_inputs()
        with pytest.raises(RuntimeError, match="free_query_embedding"):
            model.init_queries_from_parents(
                {5: {"parents": [1], "fixed": False}}, source, x_actual
            )


# ---------------------------------------------------------------------------
# 3. Freezing
# ---------------------------------------------------------------------------


class TestFreezing:
    def test_frozen_rows_get_zero_gradient_and_no_update(self):
        prior = {5: {"parents": [1, 2], "fixed": True},
                 6: {"parents": [4], "fixed": False}}
        model = _make_model(prior=prior)
        source, x_actual = _make_inputs()
        model.init_queries_from_parents(prior, source, x_actual)

        table = model.query_embed_X
        w = table.embedding.weight
        assert bool(table.frozen_rows[5 - S_SEQ_LEN])
        assert not bool(table.frozen_rows[6 - S_SEQ_LEN])

        before = w.detach().clone()
        w.sum().backward()
        assert torch.all(w.grad[table.frozen_rows] == 0.0)
        assert torch.all(w.grad[~table.frozen_rows] != 0.0)

        opt = torch.optim.SGD([w], lr=0.5)
        opt.step()
        assert torch.equal(w[table.frozen_rows], before[table.frozen_rows])
        assert not torch.allclose(w[~table.frozen_rows],
                                  before[~table.frozen_rows])

    def test_reassert_restores_after_drift(self):
        prior = {5: {"parents": [1, 2], "fixed": True}}
        model = _make_model(prior=prior)
        source, x_actual = _make_inputs()
        model.init_queries_from_parents(prior, source, x_actual)

        table = model.query_embed_X
        frozen_val = table.embedding.weight[5 - S_SEQ_LEN].detach().clone()
        with torch.no_grad():   # simulate weight-decay / noise drift
            table.embedding.weight[5 - S_SEQ_LEN] += 1.0
        model.reassert_frozen_query_rows()
        assert torch.equal(
            table.embedding.weight[5 - S_SEQ_LEN], frozen_val
        )

    def test_freeze_buffers_are_persistent(self):
        prior = {5: {"parents": [1, 2], "fixed": True}}
        model = _make_model(prior=prior)
        source, x_actual = _make_inputs()
        model.init_queries_from_parents(prior, source, x_actual)
        sd = model.state_dict()
        assert "query_embed_X.frozen_rows" in sd
        assert "query_embed_X.frozen_snapshot" in sd


# ---------------------------------------------------------------------------
# 4. Config-template wiring
# ---------------------------------------------------------------------------


class TestConfigTemplate:
    def test_default_null_and_interpolation(self):
        from omegaconf import OmegaConf

        tmpl = (
            project_root
            / "causaliT"
            / "config"
            / "templates"
            / "config_attention_selector.yaml"
        )
        cfg = OmegaConf.load(str(tmpl))

        assert cfg.experiment.query_parents_prior is None
        assert cfg.model.kwargs.query_parents_prior is None

        cfg.experiment.query_parents_prior = {5: {"parents": [1, 2], "fixed": True}}
        assert cfg.model.kwargs.query_parents_prior[5]["parents"] == [1, 2]
        assert cfg.model.kwargs.query_parents_prior[5]["fixed"] is True


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
