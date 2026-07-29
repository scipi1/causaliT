"""Tests for the adaptive learnable per-node query-norm multiplier.

Run with:  pytest tests/test_atsel_learnable_query_norm.py -v

Motivation (SELF_ATTENTION spurious-S3->X4 study)
-------------------------------------------------
``normalize_query=True`` puts the structural query on the UNIT sphere, so with an
orthonormal key frame the total directional budget ``sum_j cos^2(q_hat, k_j)`` is
HARD-capped at 1.  A node sometimes needs to OVERSPEND that budget (host several
parents at once), and — after removing W_q/W_K — saturation bites LATE, past any
preset epoch window.  Instead of an epoch schedule, each child i owns a learnable
multiplier ``M_i = exp(log_scale_i)`` (init 1.0) scaling its unit query
``q_eff = (q/||q||) * M_i``.  The STRUCTURAL loss charges the over-spend penalty
``sum_i relu(M_i - target)^2`` so a node raises its own budget only when the
structural signal pays for it.

Coverage
--------
1. ``make_query_norm_log_scale`` — the per-node parameter initialises so
   ``M_i = exp(log_scale_i) = init_scale`` (1.0 by default, log_scale == 0).
2. ``apply_query_norm`` — unit direction * M, fixed sqrt(fanin) score scale.
3. ``overspend_penalty`` — 0 when ``M <= target``; ``> 0`` when ``M > target``,
   with a live gradient on ``log_scale``.
4. ``collect_query_norm_penalty`` / ``query_norm_stats`` — aggregate (deduped)
   over the learnable modules; None when the feature is off.
5. End-to-end wiring — ``AttentionSelectorLayer`` builds a learnable multiplier
   in the inner block, and under ``shared_query=True`` the cross & self blocks
   share ONE ``query_norm_log_scale`` Parameter (charged exactly once).
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.utils.query_norm import (
    apply_query_norm,
    collect_query_norm_penalty,
    make_query_norm_log_scale,
    overspend_penalty,
    query_norm_stats,
)
from causaliT.core.modules.gated_cross_attention import GatedCrossAttention
from causaliT.core.architectures.attention_selector import AttentionSelectorLayer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 2
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1
VALUE_COL = 0
VAR_COL = 1


# ===========================================================================
# Part 1 — make_query_norm_log_scale (per-node parameter init)
# ===========================================================================
class TestParameterInit:
    def test_default_init_is_M_equal_one(self):
        p = make_query_norm_log_scale(num_nodes=5)
        assert isinstance(p, torch.nn.Parameter)
        assert p.shape == (5,)
        # log_scale == 0  ->  M = exp(0) = 1.0 for every node
        assert torch.allclose(p, torch.zeros(5), atol=1e-7)
        assert torch.allclose(torch.exp(p), torch.ones(5), atol=1e-6)

    def test_init_scale_maps_to_log(self):
        p = make_query_norm_log_scale(num_nodes=3, init_scale=2.5)
        assert torch.allclose(torch.exp(p), torch.full((3,), 2.5), atol=1e-5)
        assert torch.allclose(p, torch.full((3,), math.log(2.5)), atol=1e-6)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            make_query_norm_log_scale(num_nodes=0)
        with pytest.raises(ValueError):
            make_query_norm_log_scale(num_nodes=3, init_scale=0.0)


# ===========================================================================
# Part 2 — apply_query_norm (unit direction * M, fixed sqrt(fanin) scale)
# ===========================================================================
class TestApplyQueryNorm:
    def test_unit_direction_scaled_by_M(self):
        L = 4
        q = torch.randn(2, L, D_QK) * 3.0        # arbitrary norms
        log_scale = torch.log(torch.tensor([1.0, 2.0, 0.5, 3.0]))
        q_s, scale_s = apply_query_norm(q, log_scale, query_fanin_scale=9.0)
        # each row norm == M_i
        expected_M = torch.tensor([1.0, 2.0, 0.5, 3.0])
        assert torch.allclose(q_s.norm(dim=-1), expected_M.expand(2, L), atol=1e-5)
        # direction preserved (unit-normalised q_s matches unit-normalised q)
        assert torch.allclose(
            F.normalize(q_s, dim=-1), F.normalize(q, dim=-1), atol=1e-5
        )
        # score scale is sqrt(fanin), independent of M
        assert scale_s == pytest.approx(3.0)

    def test_M_one_matches_plain_normalize(self):
        L = 5
        q = torch.randn(2, L, D_QK) * 7.0
        log_scale = torch.zeros(L)               # M = 1 everywhere
        q_s, scale_s = apply_query_norm(q, log_scale, query_fanin_scale=4.0)
        assert torch.allclose(
            q_s, F.normalize(q, p=2.0, dim=-1, eps=1e-8), atol=1e-6
        )
        assert scale_s == pytest.approx(2.0)


# ===========================================================================
# Part 3 — overspend_penalty (0 at M<=target, >0 with grad when M>target)
# ===========================================================================
class TestOverspendPenalty:
    def test_zero_at_or_below_target(self):
        # M = 1 == target -> no penalty
        log_scale = torch.zeros(4)
        assert overspend_penalty(log_scale, target=1.0).item() == pytest.approx(0.0)
        # M = 0.5 < target -> still free
        log_scale = torch.log(torch.full((4,), 0.5))
        assert overspend_penalty(log_scale, target=1.0).item() == pytest.approx(0.0)

    def test_positive_and_matches_closed_form_when_overspending(self):
        # M = 2 for all 3 nodes, target 1 -> sum relu(2-1)^2 = 3
        log_scale = torch.log(torch.full((3,), 2.0))
        pen = overspend_penalty(log_scale, target=1.0)
        assert pen.item() == pytest.approx(3.0, rel=1e-5)

    def test_gradient_present_when_overspending(self):
        log_scale = torch.nn.Parameter(torch.log(torch.tensor([2.0, 1.0, 3.0])))
        pen = overspend_penalty(log_scale, target=1.0)
        assert pen.item() > 0.0
        pen.backward()
        assert log_scale.grad is not None
        # nodes above target get a nonzero push; the at-target node (M=1) is flat
        assert log_scale.grad[0].item() > 0.0
        assert log_scale.grad[2].item() > 0.0
        assert log_scale.grad[1].item() == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Part 4 — collect_query_norm_penalty / query_norm_stats over a module
# ===========================================================================
class TestCollectAndStats:
    def _learnable_block(self, num_nodes=X_SEQ_LEN):
        return GatedCrossAttention(
            use_gain=False, attention_dropout=0.0,
            normalize_query=True, query_fanin_scale=4.0,
            query_norm_learnable=True,
            query_norm_init_scale=1.0,
            query_norm_target=1.0,
            query_norm_num_nodes=num_nodes,
        )

    def test_none_when_feature_off(self):
        att = GatedCrossAttention(
            normalize_query=True, query_fanin_scale=4.0,
            query_norm_learnable=False,
        )
        assert collect_query_norm_penalty(att) is None
        mean_M, max_M = query_norm_stats(att)
        assert mean_M is None and max_M is None

    def test_zero_at_init_then_positive(self):
        att = self._learnable_block()
        pen0 = collect_query_norm_penalty(att)
        assert pen0 is not None
        assert pen0.item() == pytest.approx(0.0)          # M == 1 at init
        mean_M, max_M = query_norm_stats(att)
        assert mean_M.item() == pytest.approx(1.0, rel=1e-5)
        assert max_M.item() == pytest.approx(1.0, rel=1e-5)

        # overspend one node -> penalty > 0 and gradient reaches log_scale
        with torch.no_grad():
            att.query_norm_log_scale[0] = math.log(2.5)
        pen1 = collect_query_norm_penalty(att)
        assert pen1.item() > 0.0
        pen1.backward()
        assert att.query_norm_log_scale.grad is not None
        assert att.query_norm_log_scale.grad[0].item() > 0.0
        mean_M, max_M = query_norm_stats(att)
        assert max_M.item() == pytest.approx(2.5, rel=1e-4)


# ===========================================================================
# Part 5 — End-to-end wiring (AttentionSelectorLayer)
# ===========================================================================
def _svfa_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {"idx": VALUE_COL, "embed": "linear", "label": "value", "role": "value",
             "kwargs": {"input_dim": 1, "embedding_dim": d_model}},
            {"idx": VAR_COL, "embed": "nn_embedding", "label": "variable", "role": "structure",
             "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model}},
        ],
    }


def _summation_embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {"idx": VALUE_COL, "embed": "linear", "label": "value",
             "kwargs": {"input_dim": 1, "embedding_dim": d_model}},
            {"idx": VAR_COL, "embed": "nn_embedding", "label": "variable",
             "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model}},
        ],
    }


def _make_model(
    self_attention_type="GatedSelfAttention", shared_query=False
) -> AttentionSelectorLayer:

    return AttentionSelectorLayer(
        model="test_learnable_qnorm",
        ds_embed_S=_summation_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",
        n_heads=1,
        dropout_emb=0.0, dropout_attn_out=0.0, dropout_ff=0.0,
        dropout_qkv=0.0, attention_dropout=0.0,
        activation="relu", norm="layer", use_final_norm=False, device="cpu",
        out_dim=1, d_ff=D_FF, d_model=D_MODEL, d_qk=D_QK,
        S_seq_len=S_SEQ_LEN, X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type="standard_learnable",
        gain_stream_source="separate",
        normalize_query=True,
        query_fanin_scale=4.0,
        query_norm_learnable=True,
        query_norm_init_scale=1.0,
        query_norm_target=1.0,
        self_attention_type=self_attention_type,
        shared_query=shared_query,
    )


class TestLayerWiring:
    def test_learnable_multiplier_reaches_inner_block(self):
        m = _make_model()
        cross = None
        for mod in m.modules():
            if isinstance(mod, GatedCrossAttention):
                cross = mod
                break
        assert cross is not None, "GatedCrossAttention block not found"
        assert cross.query_norm_learnable is True
        p = cross.query_norm_log_scale
        assert isinstance(p, torch.nn.Parameter)
        # one multiplier per X query row, initialised at M = 1
        assert p.shape == (X_SEQ_LEN,)
        assert torch.allclose(torch.exp(p.detach()), torch.ones(X_SEQ_LEN), atol=1e-6)
        # log_scale is classified STRUCTURAL by the gradient-routing name pattern
        names = [n for n, _ in m.named_parameters() if "query_norm_log_scale" in n]
        assert len(names) >= 1

    def test_shared_query_ties_one_multiplier(self):
        m = _make_model(self_attention_type="GatedSelfAttention", shared_query=True)
        cross_ia = m.attention.inner_attention
        self_ia = m.self_attention.inner_attention
        assert getattr(cross_ia, "query_norm_log_scale", None) is not None
        assert getattr(self_ia, "query_norm_log_scale", None) is not None
        # SAME Parameter object shared across the two blocks
        assert self_ia.query_norm_log_scale is cross_ia.query_norm_log_scale
        # over-spend penalty deduplicates the tied param -> charged exactly once
        with torch.no_grad():
            cross_ia.query_norm_log_scale.fill_(math.log(2.0))
        pen = collect_query_norm_penalty(m)
        assert pen is not None
        # X_SEQ_LEN nodes at M=2, target 1  ->  sum relu(1)^2 == X_SEQ_LEN (NOT 2x)
        assert pen.item() == pytest.approx(float(X_SEQ_LEN), rel=1e-4)

    def test_split_without_shared_query_has_two_multipliers(self):
        m = _make_model(self_attention_type="GatedSelfAttention", shared_query=False)
        cross_ia = m.attention.inner_attention
        self_ia = m.self_attention.inner_attention
        if getattr(self_ia, "query_norm_log_scale", None) is not None:
            # independent params -> not the same object
            assert self_ia.query_norm_log_scale is not cross_ia.query_norm_log_scale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
