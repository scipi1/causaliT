"""Tests for the elastic contraction of the structural query normalization.

Run with:  pytest tests/test_atsel_elastic_query_norm.py -v

Motivation (SELF_ATTENTION spurious-S3->X4 study)
-------------------------------------------------
``normalize_query=True`` puts the structural query on the UNIT sphere, so with an
orthonormal key frame the total directional budget ``sum_j cos^2(q_hat, k_j)`` is
HARD-capped at 1 from the very first optimiser step.  That cap is reached almost
immediately: every candidate edge lights up together, saturates the budget, and
is then shoved into a zero-sum competition that damps genuinely good multi-parent
pushes before they can pay off.

Elastic contraction relaxes the effective query norm to ``start_scale`` (>1, a
looser budget = free early exploration) and linearly contracts it back to
``end_scale`` (typically 1.0, the original hard cap) between ``start_epoch`` and
``end_epoch``.  The query DIRECTION still drives selection; only its scheduled
norm changes, so the directional budget is ``M(epoch)^2``.  With the feature
disabled (or ``start_scale == end_scale == 1.0``) the behaviour is EXACTLY the
original unit normalisation.

Coverage
--------
1. ``ElasticQueryNormConfig.multiplier`` — the linear epoch schedule (disabled,
   endpoints, linear interpolation, degenerate window).
2. ``elastic_normalize_query`` — unit direction * M(epoch), fixed sqrt(fanin)
   score scale, and exact equivalence to plain ``F.normalize`` when disabled.
3. ``set_elastic_query_epoch`` — the trainer epoch push updates every
   ``_elastic_epoch`` buffer and returns the module count.
4. ``GatedCrossAttention`` behaviour — a looser early budget (M>1) OPENS a
   positively-aligned edge more and SUPPRESSES an anti-aligned one; contracting
   the schedule back to M=1 reproduces the plain-normalisation posterior.
5. End-to-end wiring — ``AttentionSelectorLayer`` threads the elastic kwargs
   through ``AttentionLayer`` into the inner ``GatedCrossAttention`` block.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.utils.elastic_query import (
    ElasticQueryNormConfig,
    elastic_normalize_query,
    set_elastic_query_epoch,
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
# Part 1 — Schedule (ElasticQueryNormConfig.multiplier)
# ===========================================================================
class TestSchedule:
    def test_disabled_is_always_one(self):
        cfg = ElasticQueryNormConfig(
            enabled=False, start_epoch=0, end_epoch=10,
            start_scale=3.0, end_scale=1.0,
        )
        for e in (-5, 0, 5, 10, 100):
            assert cfg.multiplier(e) == 1.0

    def test_endpoints_and_clamping(self):
        cfg = ElasticQueryNormConfig(
            enabled=True, start_epoch=2, end_epoch=12,
            start_scale=4.0, end_scale=1.0,
        )
        assert cfg.multiplier(0) == 4.0        # before start -> start_scale
        assert cfg.multiplier(2) == 4.0        # at start
        assert cfg.multiplier(12) == 1.0       # at end
        assert cfg.multiplier(999) == 1.0      # after end -> end_scale

    def test_linear_interpolation_midpoint(self):
        cfg = ElasticQueryNormConfig(
            enabled=True, start_epoch=0, end_epoch=10,
            start_scale=3.0, end_scale=1.0,
        )
        # halfway: 3 + 0.5 * (1 - 3) = 2.0
        assert cfg.multiplier(5) == pytest.approx(2.0)
        # quarter: 3 + 0.25 * (1 - 3) = 2.5
        assert cfg.multiplier(2.5) == pytest.approx(2.5)

    def test_degenerate_window_returns_end_scale(self):
        cfg = ElasticQueryNormConfig(
            enabled=True, start_epoch=5, end_epoch=5,
            start_scale=3.0, end_scale=1.0,
        )
        assert cfg.multiplier(5) == 1.0
        assert cfg.multiplier(4) == 3.0        # still on the start side
        assert cfg.multiplier(6) == 1.0

    def test_monotone_contraction(self):
        cfg = ElasticQueryNormConfig(
            enabled=True, start_epoch=0, end_epoch=20,
            start_scale=5.0, end_scale=1.0,
        )
        vals = [cfg.multiplier(e) for e in range(0, 21)]
        # non-increasing from start_scale down to end_scale
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
        assert vals[0] == 5.0 and vals[-1] == 1.0


# ===========================================================================
# Part 2 — Helper (elastic_normalize_query)
# ===========================================================================
class TestNormalizeHelper:
    def test_disabled_matches_plain_normalize(self):
        cfg = ElasticQueryNormConfig(enabled=False)
        q = torch.randn(2, 5, D_QK) * 7.0      # arbitrary norms
        q_s, scale_s = elastic_normalize_query(q, query_fanin_scale=9.0, cfg=cfg, epoch=0)
        assert torch.allclose(q_s, F.normalize(q, p=2.0, dim=-1, eps=1e-8), atol=1e-6)
        assert scale_s == pytest.approx(math.sqrt(9.0))
        # unit norm rows
        assert torch.allclose(q_s.norm(dim=-1), torch.ones(2, 5), atol=1e-5)

    def test_multiplier_scales_unit_direction(self):
        cfg = ElasticQueryNormConfig(
            enabled=True, start_epoch=0, end_epoch=10,
            start_scale=3.0, end_scale=1.0,
        )
        q = torch.randn(4, D_QK) * 2.5
        q_start, scale_s = elastic_normalize_query(q, query_fanin_scale=4.0, cfg=cfg, epoch=0)
        q_end, _ = elastic_normalize_query(q, query_fanin_scale=4.0, cfg=cfg, epoch=10)
        # direction preserved, norm == M(epoch)
        assert torch.allclose(q_start.norm(dim=-1), torch.full((4,), 3.0), atol=1e-5)
        assert torch.allclose(q_end.norm(dim=-1), torch.ones(4), atol=1e-5)
        # score scale is always sqrt(fanin), independent of the multiplier
        assert scale_s == pytest.approx(2.0)
        # same unit direction at both epochs
        assert torch.allclose(
            F.normalize(q_start, dim=-1), F.normalize(q_end, dim=-1), atol=1e-5
        )


# ===========================================================================
# Part 3 — Trainer epoch push (set_elastic_query_epoch)
# ===========================================================================
class TestEpochPush:
    def test_push_updates_buffer_and_counts(self):
        att = GatedCrossAttention(
            normalize_query=True, query_fanin_scale=4.0,
            query_norm_elastic=True, query_norm_elastic_start_epoch=0,
            query_norm_elastic_end_epoch=10, query_norm_elastic_start_scale=3.0,
            query_norm_elastic_end_scale=1.0,
        )
        # default buffer == end_epoch (graceful un-pushed behaviour)
        assert int(att._elastic_epoch.item()) == 10
        n = set_elastic_query_epoch(att, 4)
        assert n == 1
        assert int(att._elastic_epoch.item()) == 4

    def test_push_on_container_hits_all_modules(self):
        container = torch.nn.ModuleList([
            GatedCrossAttention(normalize_query=True, query_norm_elastic=True),
            GatedCrossAttention(normalize_query=True, query_norm_elastic=True),
        ])
        n = set_elastic_query_epoch(container, 7)
        assert n == 2
        for m in container:
            assert int(m._elastic_epoch.item()) == 7


# ===========================================================================
# Part 4 — GatedCrossAttention behaviour under the schedule
# ===========================================================================
def _aligned_qk():
    """One query aligned to key row 0 and anti-aligned to key row 1 (unit keys)."""
    e0 = torch.zeros(D_QK); e0[0] = 1.0
    q = e0.clone().view(1, 1, D_QK) * 5.0          # norm is irrelevant (normalised)
    key = torch.stack([e0, -e0]).view(1, 2, D_QK)  # S=2 unit keys: +e0, -e0
    v = torch.randn(1, 2, D_MODEL)
    return q, key, v


class TestGatedCrossBehaviour:
    def test_looser_budget_opens_aligned_suppresses_antialigned(self):
        att = GatedCrossAttention(
            use_gain=False, attention_dropout=0.0,
            normalize_query=True, query_fanin_scale=4.0,
            query_norm_elastic=True, query_norm_elastic_start_epoch=0,
            query_norm_elastic_end_epoch=10, query_norm_elastic_start_scale=3.0,
            query_norm_elastic_end_scale=1.0,
        )
        att.eval()
        q, key, v = _aligned_qk()

        set_elastic_query_epoch(att, 0)          # M = 3 (loose)
        _, gate_loose, _ = att(q, key, v)
        set_elastic_query_epoch(att, 10)         # M = 1 (contracted / original cap)
        _, gate_tight, _ = att(q, key, v)

        # Aligned edge (col 0): a looser budget raises its posterior.
        assert gate_loose[0, 0, 0] > gate_tight[0, 0, 0]
        # Anti-aligned edge (col 1): a looser budget pushes it further OFF.
        assert gate_loose[0, 0, 1] < gate_tight[0, 0, 1]

    def test_contracted_equals_plain_normalization(self):
        """At M == end_scale == 1 the elastic module reproduces the plain
        normalize_query posterior exactly (no behavioural drift once contracted)."""
        common = dict(
            use_gain=False, attention_dropout=0.0,
            normalize_query=True, query_fanin_scale=4.0,
        )
        elastic = GatedCrossAttention(
            query_norm_elastic=True, query_norm_elastic_start_epoch=0,
            query_norm_elastic_end_epoch=10, query_norm_elastic_start_scale=3.0,
            query_norm_elastic_end_scale=1.0, **common,
        )
        plain = GatedCrossAttention(query_norm_elastic=False, **common)
        elastic.eval(); plain.eval()
        q, key, v = _aligned_qk()

        set_elastic_query_epoch(elastic, 10)     # fully contracted -> M = 1
        _, g_elastic, _ = elastic(q, key, v)
        _, g_plain, _ = plain(q, key, v)
        assert torch.allclose(g_elastic, g_plain, atol=1e-6)

    def test_disabled_is_epoch_invariant(self):
        """With the feature OFF the posterior must not depend on the pushed epoch."""
        att = GatedCrossAttention(
            use_gain=False, attention_dropout=0.0,
            normalize_query=True, query_fanin_scale=4.0,
            query_norm_elastic=False,
        )
        att.eval()
        q, key, v = _aligned_qk()
        set_elastic_query_epoch(att, 0)
        _, g0, _ = att(q, key, v)
        set_elastic_query_epoch(att, 500)
        _, g1, _ = att(q, key, v)
        assert torch.allclose(g0, g1, atol=1e-7)


# ===========================================================================
# Part 5 — End-to-end wiring (AttentionSelectorLayer -> GatedCrossAttention)
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


def _make_elastic_model() -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_elastic",
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
        query_norm_elastic=True,
        query_norm_elastic_start_epoch=0,
        query_norm_elastic_end_epoch=8,
        query_norm_elastic_start_scale=2.5,
        query_norm_elastic_end_scale=1.0,
    )


class TestLayerWiring:
    def test_elastic_config_reaches_inner_block(self):
        m = _make_elastic_model()
        cross = None
        for mod in m.modules():
            if isinstance(mod, GatedCrossAttention):
                cross = mod
                break
        assert cross is not None, "GatedCrossAttention block not found"
        cfg = cross.elastic_query_cfg
        assert cfg.enabled is True
        assert cfg.start_epoch == 0 and cfg.end_epoch == 8
        assert cfg.start_scale == 2.5 and cfg.end_scale == 1.0
        # multiplier is live and contracts to 1.0 by end_epoch
        assert cfg.multiplier(0) == pytest.approx(2.5)
        assert cfg.multiplier(8) == pytest.approx(1.0)

    def test_set_epoch_updates_layer_buffers(self):
        m = _make_elastic_model()
        n = set_elastic_query_epoch(m, 3)
        assert n >= 1
        for mod in m.modules():
            if isinstance(mod, GatedCrossAttention):
                assert int(mod._elastic_epoch.item()) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
