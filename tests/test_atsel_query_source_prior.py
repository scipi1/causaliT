"""Tests for AttentionSelectorLayer ``query_source_prior``.

Run with:  pytest tests/test_atsel_query_source_prior.py -v

The source-nodes prior sets the query of the listed nodes to ZERO and freezes
it, so a source can never be a child (the self-attention equivalent of the
cross-attention S/X split).  With a zero query the node's raw score row is
exactly 0, so the direction of every incident edge is decided ENTIRELY by
the other endpoint's score: ``A_anti[s, j] = -raw[j, s]/2`` gives
``d[s <- j] = sigmoid(-raw[j, s] / (2*beta))`` and ``d[j -> s] = 1 - d[s <- j]``
-- whenever a child scores the source positively, ALL direction mass flows
source -> child.  ``mask: true`` (default)
additionally zeroes the source ROWS (incoming edges) of the hard mask via
the ``qsp_hard_mask`` buffer: a zero SCORE is not a zero POSTERIOR (the
sigmoid existence gate stays open at logit 0).

Node IDs are GLOBAL and 1-based over the concatenated dataset layout: S nodes
are 1..L_S, X nodes are L_S+1..L_S+L_X.  S ids require homogeneous mode (in
split mode S nodes are not queries).
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.commutator_self_attention import CommutatorSelfAttention

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


def _make_model(prior=None, parents_prior=None, homogeneous: bool = False,
                free_query_embedding: bool = True) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_summation_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="GatedCrossAttention",
        self_attention_type="CommutatorSelfAttention",
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
        remove_query_projection=True,
        remove_key_projection=True,
        homogeneous_nodes=homogeneous,
        query_parents_prior=parents_prior,
        query_source_prior=prior,
    )



# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_id_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            _make_model(prior={L + 1: {"mask": True}})
        with pytest.raises(ValueError, match="out of range"):
            _make_model(prior={0: {"mask": True}})

    def test_s_node_rejected_in_split_mode(self):
        with pytest.raises(ValueError, match="S node"):
            _make_model(prior={1: {"mask": True}}, homogeneous=False)

    def test_s_node_allowed_in_homogeneous_mode(self):
        model = _make_model(prior={1: {"mask": True}}, homogeneous=True)
        assert model.query_source_prior == {1: {"mask": True}}

    def test_requires_free_query_embedding(self):
        with pytest.raises(ValueError, match="free_query_embedding"):
            _make_model(prior={5: {"mask": True}}, free_query_embedding=False)

    def test_conflict_with_parents_prior_rejected(self):
        parents = {5: {"parents": [1, 2], "fixed": False}}
        with pytest.raises(ValueError, match="cannot be BOTH"):
            _make_model(prior={5: {"mask": True}}, parents_prior=parents)

    def test_mask_defaults_to_true(self):
        model = _make_model(prior={5: {}})
        assert model.query_source_prior == {5: {"mask": True}}


# ---------------------------------------------------------------------------
# 2. init_source_queries_zero
# ---------------------------------------------------------------------------


class TestInit:
    def test_rows_zeroed_and_frozen_split(self):
        prior = {5: {"mask": True}, 7: {"mask": False}}
        model = _make_model(prior=prior)
        table = model.query_embed_X
        untouched = table.embedding.weight[6 - S_SEQ_LEN].detach().clone()
        n = model.init_source_queries_zero(prior)
        assert n == 2
        w = table.embedding.weight
        assert torch.all(w[5 - S_SEQ_LEN] == 0.0)
        assert torch.all(w[7 - S_SEQ_LEN] == 0.0)
        assert bool(table.frozen_rows[5 - S_SEQ_LEN])
        assert bool(table.frozen_rows[7 - S_SEQ_LEN])
        # Non-listed rows are untouched.
        assert torch.equal(w[6 - S_SEQ_LEN], untouched)

    def test_s_row_goes_to_query_embed_S_in_homogeneous(self):
        prior = {1: {"mask": True}, 5: {"mask": True}}
        model = _make_model(prior=prior, homogeneous=True)
        n = model.init_source_queries_zero(prior)
        assert n == 2
        assert torch.all(model.query_embed_S.embedding.weight[1] == 0.0)
        assert torch.all(model.query_embed_X.embedding.weight[5 - S_SEQ_LEN] == 0.0)

    def test_reassert_restores_zero_rows_after_drift(self):
        prior = {5: {"mask": True}}
        model = _make_model(prior=prior)
        model.init_source_queries_zero(prior)
        table = model.query_embed_X
        with torch.no_grad():   # simulate weight-decay / noise drift
            table.embedding.weight[5 - S_SEQ_LEN] += 1.0
        model.reassert_frozen_query_rows()
        assert torch.all(table.embedding.weight[5 - S_SEQ_LEN] == 0.0)

    def test_frozen_rows_get_zero_gradient(self):
        prior = {5: {"mask": True}}
        model = _make_model(prior=prior)
        model.init_source_queries_zero(prior)
        w = model.query_embed_X.embedding.weight
        w.sum().backward()
        assert torch.all(w.grad[model.query_embed_X.frozen_rows] == 0.0)
        assert torch.all(w.grad[~model.query_embed_X.frozen_rows] != 0.0)

    def test_requires_free_query_embedding_at_init(self):
        model = _make_model(free_query_embedding=False)
        with pytest.raises(RuntimeError, match="free_query_embedding"):
            model.init_source_queries_zero({5: {"mask": True}})


# ---------------------------------------------------------------------------
# 3. qsp_hard_mask buffer
# ---------------------------------------------------------------------------


class TestMaskBuffer:
    def test_none_when_prior_disabled(self):
        model = _make_model()
        assert model.qsp_hard_mask is None

    def test_none_when_all_mask_false(self):
        model = _make_model(prior={5: {"mask": False}})
        assert model.qsp_hard_mask is None

    def test_split_layout_rows_zeroed(self):
        prior = {5: {"mask": True}, 7: {"mask": False}}
        model = _make_model(prior=prior)
        m = model.qsp_hard_mask
        assert m.shape == (X_SEQ_LEN, L)
        assert torch.all(m[5 - S_SEQ_LEN - 1] == 0.0)
        assert torch.all(m[7 - S_SEQ_LEN - 1] == 1.0)

    def test_homogeneous_layout_rows_zeroed(self):
        prior = {1: {"mask": True}, 6: {"mask": True}}
        model = _make_model(prior=prior, homogeneous=True)
        m = model.qsp_hard_mask
        assert m.shape == (L, L)
        assert torch.all(m[0] == 0.0)
        assert torch.all(m[5] == 0.0)
        # Other rows untouched.
        assert torch.all(m[1] == 1.0)


# ---------------------------------------------------------------------------
# 4. Direction mechanics (CommutatorSelfAttention with a zeroed query row)
# ---------------------------------------------------------------------------


class TestDirectionMechanics:
    """The zero-query source behaves as predicted inside the self block."""

    N = 5
    E = 8
    SRC = 1   # 0-based row of the "source" node

    def _forward(self, hard_mask=None):
        torch.manual_seed(0)
        attn = CommutatorSelfAttention(
            use_gain=False, normalize_query=True, query_fanin_scale=4.0,
        )
        attn.eval()
        query = torch.randn(BATCH, self.N, self.E)
        query[:, self.SRC, :] = 0.0            # the frozen zero source query
        key = torch.randn(BATCH, self.N, self.E)
        value = torch.randn(BATCH, self.N, self.E)
        with torch.no_grad():
            out, p_directed, aux = attn(
                query=query, key=key, value=value,
                mask_miss_k=None, mask_miss_q=None, pos=None,
                causal_mask=False, hard_mask=hard_mask,
            )
        return attn, out, p_directed, aux

    def test_direction_mass_flows_source_to_child(self):
        attn, _, _, _ = self._forward()
        d = attn.last_direction                       # (N, N), eval-mode
        s = self.SRC
        # Zero raw row -> A_anti[s, j] = -raw[j, s]/2 = -A_anti[j, s], so the
        # incident direction stays exactly coupled: d[s, j] = 1 - d[j, s].
        off = [j for j in range(self.N) if j != s]
        assert torch.allclose(
            d[s, off], 1.0 - d[off, s], atol=1e-5,
        )
        # The zero query is EXACTLY undecided only on the (masked) diagonal.
        assert abs(d[s, s].item() - 0.5) < 1e-6
        # And the source row carries no learned competition: the existence
        # posterior on its row is the constant logit-0 gate value.
        pe = attn.last_p_edge_undirected[s, off]
        assert torch.allclose(pe, pe[0].expand_as(pe), atol=1e-6)

    def test_masked_source_row_has_zero_posterior(self):
        hm = torch.ones(self.N, self.N)
        hm[self.SRC, :] = 0.0                  # qsp_hard_mask row
        _, _, p_directed, _ = self._forward(hard_mask=hm)
        assert torch.all(p_directed[:, self.SRC, :] == 0.0)

    def test_unmasked_source_row_is_soft_only(self):
        # Without the mask the existence gate at logit 0 is NOT exactly zero
        # (the documented zero-score != zero-posterior caveat).
        _, _, p_directed, _ = self._forward(hard_mask=None)
        assert torch.any(p_directed[:, self.SRC, :] > 0.0)

    def test_masked_source_row_excluded_from_l0(self):
        hm = torch.ones(self.N, self.N)
        hm[self.SRC, :] = 0.0
        _, _, _, aux_masked = self._forward(hard_mask=hm)
        _, _, _, aux_plain = self._forward(hard_mask=None)
        assert aux_masked["l0_penalty"] < aux_plain["l0_penalty"]


# ---------------------------------------------------------------------------
# 5. Config-template wiring
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

        assert cfg.experiment.query_source_prior is None
        assert cfg.model.kwargs.query_source_prior is None

        cfg.experiment.query_source_prior = {1: {"mask": True}}
        assert cfg.model.kwargs.query_source_prior[1]["mask"] is True


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])


    def test_buffer_is_persistent(self):
        model = _make_model(prior={5: {"mask": True}})
        assert "qsp_hard_mask" in model.state_dict()

