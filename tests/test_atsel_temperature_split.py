"""Verification: the three harmonized attention temperatures are wired to the
right modules, with the legacy fallback chain intact.

Run with:  pytest tests/test_atsel_temperature_split.py -v

Background
----------
See docs/documentation/ATTENTION_TEMPERATURES.md.  Three temperatures:

1. ``init_tau_cross`` -> cross-block Hard-Concrete existence gate (``beta``).
2. ``init_tau_self``  -> self-block Hard-Concrete existence gate (``beta``);
   also the single square block in homogeneous mode.
3. ``dir_tau_self``   -> self-block antisymmetric direction gate (``dir_beta``).

Fallback chain when a split key is unset: legacy shared ``init_tau`` (resp.
``dir_tau``), then the calculated defaults (0.5 / 0.5 / 2/3 with the symmetric
stretch gamma=-1.1, zeta=1.1).  ``init_tau`` alone stays the activation
temperature of the NON-gated attentions (default 3.0).
"""

import logging
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.utils.query_norm import (
    DEFAULT_DIR_TAU,
    DEFAULT_GATE_GAMMA,
    DEFAULT_GATE_TAU,
    DEFAULT_GATE_ZETA,
    gate_tau_from_experiment,
    kappa,
    resolve_query_fanin_scale,
)


D_MODEL = 16
D_FF = 32
D_QK = 16
S_SEQ_LEN = 3
X_SEQ_LEN = 4
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1

VALUE_COL = 0
VAR_COL = 1


def _svfa_embed_cfg(vocab: int) -> dict:
    return {
        "setting": {"d_model": D_MODEL},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": D_MODEL},
            },
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": D_MODEL},
            },
        ],
    }


def _make_model(**kwargs) -> AttentionSelectorLayer:
    kwargs.setdefault("attention_type", "GatedCrossAttention")
    kwargs.setdefault("self_attention_type", "GatedSelfAttention")
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
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
        use_gain=False,
        struct_embedding_type="standard_learnable",
        key_projection_type="linear",
        free_query_embedding=True,
        **kwargs,
    )


def _betas(model: AttentionSelectorLayer):
    """(cross beta, self beta, self dir_beta) of a split-mode model."""
    cross = model.attention.inner_attention
    self_att = model.self_attention.inner_attention
    return cross.beta, self_att.beta, self_att.dir_beta


def test_explicit_split_keys_land_on_the_right_modules():
    m = _make_model(init_tau_cross=0.25, init_tau_self=0.7, dir_tau_self=0.9)
    b_cross, b_self, d_self = _betas(m)
    assert b_cross == 0.25
    assert b_self == 0.7
    assert d_self == pytest.approx(0.9)


def test_legacy_shared_keys_fallback():
    m = _make_model(init_tau=0.5, dir_tau=0.3)
    b_cross, b_self, d_self = _betas(m)
    assert b_cross == 0.5 and b_self == 0.5
    assert d_self == pytest.approx(0.3)


def test_bare_construction_uses_the_calculated_defaults():
    m = _make_model()
    b_cross, b_self, d_self = _betas(m)
    assert b_cross == DEFAULT_GATE_TAU == 0.5
    assert b_self == DEFAULT_GATE_TAU
    assert d_self == pytest.approx(DEFAULT_DIR_TAU)
    # The symmetric stretch the calculated temperature is derived with.
    cross = m.attention.inner_attention
    assert cross.gamma == DEFAULT_GATE_GAMMA == -1.1
    assert cross.zeta == DEFAULT_GATE_ZETA == 1.1
    assert kappa(cross.beta, cross.gamma, cross.zeta) == pytest.approx(0.0)


def test_non_gated_cross_attention_keeps_its_activation_temperature():
    m = _make_model(attention_type="CausalCrossAttention", self_attention_type=None)
    assert m.attention.inner_attention.tau == 3.0
    m = _make_model(
        attention_type="CausalCrossAttention", self_attention_type=None,
        init_tau=7.0,
    )
    assert m.attention.inner_attention.tau == 7.0


def test_homogeneous_block_gets_the_SELF_temperatures():
    m = _make_model(homogeneous_nodes=True, init_tau_self=0.6, dir_tau_self=0.42)
    block = m.attention.inner_attention
    assert block.beta == 0.6
    assert block.dir_beta == pytest.approx(0.42)
    # Homogeneous + legacy shared key falls back.
    m = _make_model(homogeneous_nodes=True, init_tau=0.31)
    assert m.attention.inner_attention.beta == 0.31


def test_gate_tau_from_experiment_resolution_order():
    # Split mode: the CROSS key wins (it carries init_edge_offset).
    assert gate_tau_from_experiment(
        {"init_tau_cross": 0.2, "init_tau_self": 0.8, "init_tau": 0.5},
        homogeneous=False,
    ) == 0.2
    # Homogeneous mode: the SELF key wins.
    assert gate_tau_from_experiment(
        {"init_tau_cross": 0.2, "init_tau_self": 0.8, "init_tau": 0.5},
        homogeneous=True,
    ) == 0.8
    # Legacy fallback, then the calculated default.
    assert gate_tau_from_experiment({"init_tau": 0.5}, homogeneous=False) == 0.5
    assert gate_tau_from_experiment({}, homogeneous=False) == DEFAULT_GATE_TAU


def test_gate_tau_from_experiment_warns_on_split_mismatch(caplog):
    with caplog.at_level(logging.WARNING, logger="causaliT.utils.query_norm"):
        gate_tau_from_experiment(
            {"init_tau_cross": 0.2, "init_tau_self": 0.8}, homogeneous=False
        )
    assert "init_tau_cross" in caplog.text


def test_fanin_derivation_uses_the_split_temperature():
    # The temperature enters the F derivation through kappa, which vanishes for
    # the symmetric stretch; use an asymmetric stretch so the tau choice shows.
    base = {
        "query_fanin_scale": "auto",
        "query_centroid_max_p": 0.9,
        "init_gamma": -0.1,     # asymmetric: kappa = tau * ln(0.1/1.1) != 0
        "init_zeta": 1.1,
        "init_edge_offset": 0.0,
    }
    n_keys = S_SEQ_LEN + X_SEQ_LEN

    cfg_legacy = {"experiment": {**base, "init_tau": 0.25}}
    out_legacy = resolve_query_fanin_scale(cfg_legacy, n_keys=n_keys)

    cfg_split = {"experiment": {**base, "init_tau_cross": 0.25}}
    out_split = resolve_query_fanin_scale(cfg_split, n_keys=n_keys)
    assert out_split["query_fanin_scale"] == pytest.approx(
        out_legacy["query_fanin_scale"]
    )

    # ... and it really is the CROSS temperature (a different value changes F).
    cfg_other = {"experiment": {**base, "init_tau_cross": 0.5}}
    out_other = resolve_query_fanin_scale(cfg_other, n_keys=n_keys)
    assert out_other["query_fanin_scale"] != pytest.approx(
        out_split["query_fanin_scale"]
    )
