"""Verification: ``init_edge_offset`` balances the S->X cross existence gate
against the directed X->X self edge at initialization.

Run with:  pytest tests/test_atsel_init_edge_offset.py -v

Background
----------
The COND_INDEPENDENCE investigation
(``experiments/6_INVESTIGATIONS/COND_INDEPENDENCE/results/CI_shared_qk_qk_inj_7676651/
investigate_S2_X5_spurious_first.ipynb``) showed the two edge-existence gates
start at DIFFERENT probabilities:

* **S->X cross** (``GatedCrossAttention``): ``P(edge) = sigmoid(logα − offset)``.
  With ``logα ≈ 0`` at init and no offset, ``P ≈ sigmoid(0) = 0.5``.
* **X->X self** (``CommutatorSelfAttention``): ``P = p_exist · direction`` with
  ``p_exist ≈ 0.5`` and the antisymmetric direction gate undecided
  (``direction ≈ 0.5``), so the DIRECTED self edge starts at ``P ≈ 0.25``.

So a (frequently spurious) S->X edge gets a 2× head start.  ``init_edge_offset``
adds an additive logit offset ``c`` on the CROSS existence gate ONLY:

    P_cross(init) = sigmoid(logα − c) = sigmoid(−c),  want = 0.25  ⇒  c = ln 3.

These tests verify:

1. The offset is threaded to the cross block ONLY (self block unchanged).
2. With the SAME init, ``logit(P_cross^offset) = logit(P_cross^0) − c`` exactly.
3. ``c = ln 3`` lands the cross init edge prob at ≈ 0.25, matching the directed
   self edge (they start balanced, no 2× head start).
"""

import math
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
BATCH = 8
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1

LN3 = math.log(3.0)  # ≈ 1.0986; sigmoid(-ln3) = 0.25


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


def _make_model(init_edge_offset: float = 0.0, seed: int = 0) -> AttentionSelectorLayer:
    # Seed so two models built with the same seed share identical init params
    # (hence identical alignment logits logα for the SAME inputs).
    torch.manual_seed(seed)
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
        d_qk=D_QK,
        S_seq_len=S_SEQ_LEN,
        X_seq_len=X_SEQ_LEN,
        shared_dag_across_heads=True,
        # Match the CI/SELF_ATTENTION config where the gap was diagnosed.
        init_tau=0.5,
        init_gamma=-1.1,
        init_zeta=1.1,
        use_gain=False,
        normalize_query=True,
        query_fanin_scale=1.0,
        struct_embedding_type="standard_learnable",
        key_projection_type="linear",
        free_query_embedding=True,
        gain_stream_source="separate",
        self_attention_type="CommutatorSelfAttention",
        commutator_direction_mode="skew_query",
        shared_query=True,
        shared_key=True,
        # The parameter under test — applied to the cross block ONLY.
        init_edge_offset=init_edge_offset,
    )


def _make_inputs(seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN, generator=g)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )
    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN, generator=g)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).repeat(BATCH, 1)
    )
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_actual, x_blanked


def _cross_exist(model):
    """S->X existence posterior P(edge) from the cross block."""
    return model.attention.inner_attention.last_p_edge_on.detach()


def _self_directed(model):
    """Directed X->X posterior P = p_exist·direction from the self block."""
    return model.self_attention.inner_attention.last_p_edge_on.detach()


def _self_exist(model):
    """Undirected X->X existence p_exist from the self block."""
    return model.self_attention.inner_attention.last_p_edge_undirected.detach()


def _logit(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


# ---------------------------------------------------------------------------
# 1. The offset is threaded to the cross block ONLY.
# ---------------------------------------------------------------------------


class TestWiring:
    def test_offset_recorded_on_cross_not_self(self):
        m = _make_model(init_edge_offset=LN3)
        # Cross block carries the offset.
        assert m.attention.inner_attention.edge_offset == pytest.approx(LN3)
        # Self block keeps a zero offset (it is already halved by its direction
        # gate, so it must NOT be offset again).
        self_inner = m.self_attention.inner_attention
        assert isinstance(self_inner, CommutatorSelfAttention)
        assert getattr(self_inner, "edge_offset", 0.0) == pytest.approx(0.0)

    def test_default_offset_is_zero(self):
        m = _make_model()  # default 0.0 = original behaviour
        assert m.attention.inner_attention.edge_offset == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. With identical init, the offset shifts the cross logit by exactly -c.
# ---------------------------------------------------------------------------


class TestOffsetShiftsCrossLogit:
    def test_logit_shift_matches_offset(self):
        source, x_actual, x_blanked = _make_inputs()

        m0 = _make_model(init_edge_offset=0.0, seed=7).eval()
        mc = _make_model(init_edge_offset=LN3, seed=7).eval()  # same seed → same logα

        with torch.no_grad():
            m0.forward_with_actual(source, x_blanked, x_actual)
            mc.forward_with_actual(source, x_blanked, x_actual)

        p0 = _cross_exist(m0)
        pc = _cross_exist(mc)

        # logit(sigmoid(logα - c)) - logit(sigmoid(logα)) = -c, elementwise.
        delta = _logit(pc) - _logit(p0)
        assert torch.allclose(delta, torch.full_like(delta, -LN3), atol=1e-4), (
            f"cross logit must shift by exactly -ln3; got mean {delta.mean():.4f}"
        )

    def test_self_block_unchanged_by_offset(self):
        source, x_actual, x_blanked = _make_inputs()
        m0 = _make_model(init_edge_offset=0.0, seed=7).eval()
        mc = _make_model(init_edge_offset=LN3, seed=7).eval()
        with torch.no_grad():
            m0.forward_with_actual(source, x_blanked, x_actual)
            mc.forward_with_actual(source, x_blanked, x_actual)
        # The self (X->X) directed posterior must be identical — the offset only
        # touches the cross block.
        assert torch.allclose(_self_directed(m0), _self_directed(mc), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. c = ln3 balances cross init prob (0.5 → 0.25) with the directed self edge.
# ---------------------------------------------------------------------------


class TestBalancedInitialization:
    def test_cross_starts_at_half_without_offset(self):
        source, x_actual, x_blanked = _make_inputs()
        m = _make_model(init_edge_offset=0.0).eval()
        with torch.no_grad():
            m.forward_with_actual(source, x_blanked, x_actual)
        # Without offset the cross existence gate clusters around 0.5.
        assert _cross_exist(m).mean().item() == pytest.approx(0.5, abs=0.08)

    def test_offset_brings_cross_to_quarter(self):
        source, x_actual, x_blanked = _make_inputs()
        m = _make_model(init_edge_offset=LN3).eval()
        with torch.no_grad():
            m.forward_with_actual(source, x_blanked, x_actual)
        # With c = ln3 the cross existence gate clusters around 0.25.
        assert _cross_exist(m).mean().item() == pytest.approx(0.25, abs=0.08)

    def test_cross_and_self_balanced_with_offset(self):
        """The whole point: with the offset, the cross (S->X) init edge prob and
        the directed self (X->X) init edge prob start on EQUAL footing (both
        ≈0.25), removing the 2× head start of the spurious S->X edges."""
        source, x_actual, x_blanked = _make_inputs()

        m_no = _make_model(init_edge_offset=0.0).eval()
        m_off = _make_model(init_edge_offset=LN3).eval()
        with torch.no_grad():
            m_no.forward_with_actual(source, x_blanked, x_actual)
            m_off.forward_with_actual(source, x_blanked, x_actual)

        self_dir_mean = _self_directed(m_off).mean().item()
        cross_no_mean = _cross_exist(m_no).mean().item()
        cross_off_mean = _cross_exist(m_off).mean().item()

        # Sanity: the directed self edge is indeed ~0.25 at init.
        assert self_dir_mean == pytest.approx(0.25, abs=0.1)
        # The offset moves the cross gate MUCH closer to the self edge.
        gap_no = abs(cross_no_mean - self_dir_mean)
        gap_off = abs(cross_off_mean - self_dir_mean)
        assert gap_off < gap_no, (
            f"offset should shrink the cross↔self init gap: "
            f"no-offset gap={gap_no:.3f}, offset gap={gap_off:.3f}"
        )
        # Both blocks now start from a ~0.5 existence gate (one halved by the
        # offset, the other by the undecided direction gate), so they are
        # balanced up to the residual difference in their alignment logits logα
        # (which are not exactly 0 because the free query aligns partially with
        # the keys).  Use the same abs=0.1-class tolerance as the sanity checks
        # above rather than an over-tight bound.
        assert gap_off < 0.12



if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
