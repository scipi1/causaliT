"""Verification: ``init_edge_offset`` balances the S->X cross existence gate
against the directed X->X self edge at initialization.

Run with:  pytest tests/test_atsel_init_edge_offset.py -v

Background
----------
The split-mode self block (``GatedSelfAttention``) factors the directed X->X
posterior as ``p_exist * d`` with an undecided direction gate at init
(``d = 0.5``), so a directed self edge starts at HALF the existence posterior.
The direction-free cross gate (``GatedCrossAttention``) starts at the full
existence posterior, giving the (often spurious) S->X edges a head start
(COND_INDEPENDENCE investigation, ``investigate_S2_X5_spurious_first.ipynb``).

Current design (see docs/experimental_elaborations/QUERY_FANIN_SCALE_BUDGET.md,
Section 7.2):

* **F is T-free.**  ``query_fanin_scale`` is derived so the CENTROID init gives
  every candidate parent the EXISTENCE posterior ``query_centroid_max_p`` on
  both gates (``x = logit(p*) + kappa``); ``init_edge_offset`` never enters it.
* **The offset lives on the cross gate only** and is subtractive:
  ``P_cross = sigmoid(x - init_edge_offset - kappa)``.
* ``init_edge_offset: auto`` resolves AT DATA-LOAD TIME to the MATCHED offset
  ``T = ln(exp(x - kappa) + 2)`` - the value that lowers the cross posterior
  onto the DIRECTED self posterior (directed-level balance).  A float pins a
  legacy ablation; 0.0 disables the offset (existence-level balance: cross at
  p*, directed self at p*/2).

These tests verify:

1. The matched-offset formula and the config resolution (off / auto / pinned,
   guards, F independence).
2. The offset is threaded to the cross block ONLY (self block unchanged).
3. With the SAME init, ``logit(P_cross^offset) = logit(P_cross^0) - c``.
4. End-to-end (orthonormal frame + centroid init + resolved F): with ``auto``
   the cross init posterior equals the directed self init posterior.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.gated_self_attention import GatedSelfAttention
from causaliT.utils.query_norm import (
    is_auto_offset,
    matched_edge_offset,
    resolve_init_edge_offset,
    resolve_query_fanin_scale,
)


D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
N_KEYS = S_SEQ_LEN + X_SEQ_LEN
BATCH = 8
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1

LN3 = math.log(3.0)  # ≈ 1.0986; sigmoid(-ln3) = 0.25 (the pre-F-auto choice)

# The calculated operating point: p* = sigmoid(kappa_1), kappa = 0.
P_STAR = 0.8209
X_STAR = math.log(P_STAR / (1.0 - P_STAR))          # = kappa_1 ≈ 1.5223
T_MATCHED = math.log(math.exp(X_STAR) + 2.0)        # ≈ 1.8846
FANIN_STAR = N_KEYS * X_STAR ** 2                   # query_fanin_scale (F^2)


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


def _make_model(init_edge_offset: float = 0.0, seed: int = 0,
                query_fanin_scale: float = 1.0) -> AttentionSelectorLayer:
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
        init_tau=0.5,
        init_gamma=-1.1,
        init_zeta=1.1,
        normalize_query=True,
        query_fanin_scale=query_fanin_scale,
        struct_embedding_type="standard_learnable",
        key_projection_type="linear",
        free_query_embedding=True,
        self_attention_type="GatedSelfAttention",
        shared_query=True,
        shared_key=True,
        # The parameter under test — applied to the cross block ONLY.
        init_edge_offset=init_edge_offset,
    )


def _make_centroid_model(init_edge_offset: float, seed: int = 0) -> AttentionSelectorLayer:
    """The full centroid stack: fixed orthonormal frame, no Q/K projections,
    free query at the key centroid, resolved T-free F."""
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
        init_tau=0.5,
        init_gamma=-1.1,
        init_zeta=1.1,
        normalize_query=True,
        query_fanin_scale=FANIN_STAR,
        query_norm_learnable=True,
        struct_embedding_type="orthogonal_fixed",
        orthogonal_fixed_frame_type="dct",
        key_projection_type="linear",
        remove_query_projection=True,
        remove_key_projection=True,
        free_query_embedding=True,
        query_centroid_init=True,
        self_attention_type="GatedSelfAttention",
        shared_query=True,
        shared_key=True,
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


def _offdiag_mean(sq: torch.Tensor) -> float:
    """Mean of the OFF-DIAGONAL entries (the zeroed self-loop diagonal excluded)."""
    n = sq.shape[-1]
    return float((sq.sum() - sq.diagonal().sum()) / (n * (n - 1)))


def _logit(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


# ---------------------------------------------------------------------------
# 0. The matched-offset formula and the config resolution.
# ---------------------------------------------------------------------------


class TestMatchedOffsetFormula:
    def test_formula_matches_the_defining_equation(self):
        # sigma(x - T - kappa) == 0.5 * sigma(x - kappa), here at kappa = 0.
        for x in (0.5, 1.5223, 2.6209, 4.0):
            t = matched_edge_offset(x)
            lhs = 1.0 / (1.0 + math.exp(-(x - t)))
            rhs = 0.5 / (1.0 + math.exp(-x))
            assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_formula_carries_the_stretch(self):
        kap = 0.3
        x = 1.8
        t = matched_edge_offset(x, kap)
        lhs = 1.0 / (1.0 + math.exp(-(x - t - kap)))
        rhs = 0.5 / (1.0 + math.exp(-(x - kap)))
        assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_reference_value(self):
        # The design doc's matched offset at p* = 0.8209 (x = kappa_1 = 1.5223).
        assert matched_edge_offset(X_STAR) == pytest.approx(1.8846, abs=1e-3)

    def test_sentinel(self):
        assert is_auto_offset("auto") and is_auto_offset("AUTO")
        assert is_auto_offset("matched")
        assert not is_auto_offset(1.1) and not is_auto_offset(None)


def _balance_config(**experiment):
    base = {
        "attention_type": "GatedCrossAttention",
        "self_attention_type": "GatedSelfAttention",
        "homogeneous_nodes": False,
        "normalize_query": True,
        "query_norm_init_scale": 1.0,
        "query_fanin_scale": FANIN_STAR,
        "query_centroid_max_p": P_STAR,
        "init_tau": 0.5, "init_gamma": -1.1, "init_zeta": 1.1,
        "init_edge_offset": 0.0,
    }
    base.update(experiment)
    return OmegaConf.create({"experiment": base})


class TestResolveInitEdgeOffset:
    def test_absent_key_is_a_no_op(self):
        assert resolve_init_edge_offset(OmegaConf.create({"experiment": {}}),
                                        n_keys=N_KEYS) is None

    @pytest.mark.parametrize("off", [0.0, None])
    def test_off_writes_zero(self, off):
        cfg = _balance_config(init_edge_offset=off)
        info = resolve_init_edge_offset(cfg, n_keys=N_KEYS)
        assert info["mode"] == "off"
        assert cfg.experiment.init_edge_offset == 0.0
        # Existence-level balance: cross at p*, directed self at p*/2.
        assert info["p_cross"] == pytest.approx(P_STAR, abs=1e-3)
        assert info["p_self_existence"] == pytest.approx(P_STAR, abs=1e-3)
        assert info["p_self_directed"] == pytest.approx(P_STAR / 2, abs=1e-3)

    def test_auto_resolves_the_matched_offset(self):
        cfg = _balance_config(init_edge_offset="auto")
        info = resolve_init_edge_offset(cfg, n_keys=N_KEYS)
        assert info["mode"] == "auto"
        assert cfg.experiment.init_edge_offset == pytest.approx(T_MATCHED, abs=1e-3)
        # Directed-level balance: cross == directed self, both at p*/2.
        assert info["p_cross"] == pytest.approx(P_STAR / 2, abs=1e-3)
        assert info["p_self_directed"] == pytest.approx(P_STAR / 2, abs=1e-3)
        # ... and the deterministic cross gate therefore starts CLOSED (the
        # accepted trade-off of directed-level balance, trained around by the
        # stochastic gate).
        assert info["z_cross"] == 0.0

    def test_auto_works_with_a_pinned_fanin_scale(self):
        cfg = _balance_config(init_edge_offset="auto", query_fanin_scale=42.0)
        info = resolve_init_edge_offset(cfg, n_keys=N_KEYS)
        x = math.sqrt(42.0 / N_KEYS)
        assert info["init_edge_offset"] == pytest.approx(matched_edge_offset(x))
        assert cfg.experiment.query_fanin_scale == 42.0      # untouched

    def test_auto_needs_a_resolved_fanin_scale(self):
        cfg = _balance_config(init_edge_offset="auto", query_fanin_scale="auto")
        with pytest.raises(ValueError, match="query_fanin_scale"):
            resolve_init_edge_offset(cfg, n_keys=N_KEYS)

    def test_auto_needs_normalize_query(self):
        cfg = _balance_config(init_edge_offset="auto", normalize_query=False)
        with pytest.raises(ValueError, match="normalize_query"):
            resolve_init_edge_offset(cfg, n_keys=N_KEYS)

    def test_auto_is_inert_without_a_cross_self_pair(self, caplog):
        import logging
        cases = [
            # Homogeneous mode: one square block, no cross gate to offset.
            {"homogeneous_nodes": True},
            # Cross-only mode: no self block.
            {"self_attention_type": None},
            # A non-gated cross attention does not consume the offset.
            {"attention_type": "ScaledDotSoftmax"},
        ]
        for override in cases:
            cfg = _balance_config(init_edge_offset="auto", **override)
            with caplog.at_level(logging.WARNING, logger="causaliT.utils.query_norm"):
                info = resolve_init_edge_offset(cfg, n_keys=N_KEYS)
            assert cfg.experiment.init_edge_offset == 0.0
            assert info["mode"] == "off"
            assert "ignored" in caplog.text
            caplog.clear()

    def test_pinned_float_is_kept_verbatim(self):
        cfg = _balance_config(init_edge_offset=LN3)
        info = resolve_init_edge_offset(cfg, n_keys=N_KEYS)
        assert info["mode"] == "pinned"
        assert cfg.experiment.init_edge_offset == LN3

    def test_fanin_scale_never_reads_the_offset(self):
        # F is T-free: the SAME resolved value whatever the offset mode.
        f_off = resolve_query_fanin_scale(
            _balance_config(query_fanin_scale="auto", init_edge_offset=0.0),
            n_keys=N_KEYS)["query_fanin_scale"]
        f_pinned = resolve_query_fanin_scale(
            _balance_config(query_fanin_scale="auto", init_edge_offset=LN3),
            n_keys=N_KEYS)["query_fanin_scale"]
        f_auto = resolve_query_fanin_scale(
            _balance_config(query_fanin_scale="auto", init_edge_offset="auto"),
            n_keys=N_KEYS)["query_fanin_scale"]
        assert f_off == pytest.approx(FANIN_STAR, rel=1e-9)
        assert f_pinned == f_off and f_auto == f_off


# ---------------------------------------------------------------------------
# 1. The offset is threaded to the cross block ONLY.
# ---------------------------------------------------------------------------


class TestWiring:
    def test_offset_recorded_on_cross_not_self(self):
        m = _make_model(init_edge_offset=LN3)
        # Cross block carries the offset.
        assert m.attention.inner_attention.edge_offset == pytest.approx(LN3)
        # Self block keeps a zero offset (the balancing is entirely on the
        # cross side; the self block is never offset).
        self_inner = m.self_attention.inner_attention
        assert isinstance(self_inner, GatedSelfAttention)
        assert getattr(self_inner, "edge_offset", 0.0) == pytest.approx(0.0)

    def test_default_offset_is_zero(self):
        m = _make_model()  # default 0.0 = original behaviour
        assert m.attention.inner_attention.edge_offset == pytest.approx(0.0)

    def test_unresolved_sentinel_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="never resolved"):
            _make_model(init_edge_offset="auto")


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
# 3. End-to-end: the centroid stack with the resolved T-free F.
# ---------------------------------------------------------------------------


class TestBalancedInitialization:
    """Orthonormal frame + centroid init + F = N * x(p*)^2 (T-free)."""

    def _posteriors(self, offset):
        source, x_actual, x_blanked = _make_inputs()
        m = _make_centroid_model(init_edge_offset=offset).eval()
        with torch.no_grad():
            m.init_query_at_key_centroid(source, x_actual)
            m.forward_with_actual(source, x_blanked, x_actual)
        return m

    def test_existence_balance_without_offset(self):
        # Offset off: both existence gates at p*, directed self at p*/2.
        m = self._posteriors(0.0)
        assert _cross_exist(m).mean().item() == pytest.approx(P_STAR, abs=0.02)
        assert _self_exist(m).mean().item() == pytest.approx(P_STAR, abs=0.02)
        assert _offdiag_mean(_self_directed(m)) == pytest.approx(P_STAR / 2, abs=0.02)

    def test_directed_balance_with_the_matched_offset(self):
        # The whole point of "auto": cross and directed self start EQUAL.
        m = self._posteriors(T_MATCHED)
        cross = _cross_exist(m).mean().item()
        self_dir = _offdiag_mean(_self_directed(m))
        assert _self_exist(m).mean().item() == pytest.approx(P_STAR, abs=0.02)
        assert self_dir == pytest.approx(P_STAR / 2, abs=0.02)
        assert cross == pytest.approx(self_dir, abs=0.02)

    def test_resolve_then_build_round_trip(self):
        # The config path: auto resolves to the number the model then receives.
        cfg = _balance_config(init_edge_offset="auto")
        resolve_init_edge_offset(cfg, n_keys=N_KEYS)
        m = self._posteriors(float(cfg.experiment.init_edge_offset))
        assert _cross_exist(m).mean().item() == pytest.approx(
            _offdiag_mean(_self_directed(m)), abs=0.02)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
