"""Tests for the per-node adjacency-context injection (per_node_adjacency_context).

Run with:  pytest tests/test_atsel_adjacency_context.py -v

When ``per_node_adjacency_context=True`` (requires ``per_node_output=True``),
the PerNodeMLPHead input is concatenated with the DETACHED applied-adjacency
row (B, L_q, L_S+L_X) — the gate weights actually used on this sample after
hard mask, gain, BKD, top-k blanking and dropout.  The context tells the
(nuisance) per-node regressor which keys were selected, consistently with
stochastic key exclusion (BKD/top-k), without leaking gradient into the
structural parameters.

Column convention (production layout): value at column 0, variable-ID at
column 1.
"""

import pytest
import torch

from causaliT.core.architectures.attention_selector.model import (
    AttentionSelectorLayer,
)

S_SEQ_LEN = 3
X_SEQ_LEN = 4
BATCH = 4
D_MODEL = 16
D_QK = 16
D_FF = 32
VOCAB_S = S_SEQ_LEN + 1
VOCAB_X = X_SEQ_LEN + 1
PER_NODE_HIDDEN = 8

VALUE_COL = 0
VAR_COL = 1


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    """Summation embedding config in the production (value, variable) layout."""
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


def _make_inputs():
    source = torch.zeros(BATCH, S_SEQ_LEN, 2)
    source[:, :, VALUE_COL] = torch.randn(BATCH, S_SEQ_LEN)
    source[:, :, VAR_COL] = (
        torch.arange(1, S_SEQ_LEN + 1).float().unsqueeze(0).expand(BATCH, -1)
    )
    x_actual = torch.zeros(BATCH, X_SEQ_LEN, 2)
    x_actual[:, :, VALUE_COL] = torch.randn(BATCH, X_SEQ_LEN)
    x_actual[:, :, VAR_COL] = (
        torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).expand(BATCH, -1)
    )
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    return source, x_blanked, x_actual


def _make_model(**overrides) -> AttentionSelectorLayer:
    kwargs = dict(
        model="test_model",
        ds_embed_S=_embed_cfg(VOCAB_S),
        ds_embed_X=_embed_cfg(VOCAB_X),
        comps_embed_S="summation",
        comps_embed_X="summation",
        attention_type="GatedCrossAttention",
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
        per_node_output=True,
        per_node_output_hidden=PER_NODE_HIDDEN,
    )
    kwargs.update(overrides)
    return AttentionSelectorLayer(**kwargs)


# 1. Default off — head width unchanged, forward works
def test_default_off_head_width_and_forward():
    torch.manual_seed(0)
    model = _make_model()
    assert model.per_node_adjacency_context is False
    assert model.forecaster.d_model == D_MODEL
    model.eval()
    source, x_blanked, x_actual = _make_inputs()
    pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
    assert attn.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# 2. Guard — requires per_node_output=True
def test_requires_per_node_output():
    with pytest.raises(ValueError, match="per_node_adjacency_context"):
        _make_model(per_node_output=False, per_node_adjacency_context=True)


# 3. Head is widened by L_S + L_X when enabled
def test_head_widened_when_enabled():
    model = _make_model(per_node_adjacency_context=True)
    assert model.forecaster.d_model == D_MODEL + S_SEQ_LEN + X_SEQ_LEN


# 4. BKD consistency — dropped keys have exactly-zero context columns
def test_bkd_dropped_keys_zero_in_context():
    torch.manual_seed(0)
    model = _make_model(
        per_node_adjacency_context=True,
        batch_key_dropout=0.5,
    )
    model.train()  # BKD only applies in training mode
    source, x_blanked, x_actual = _make_inputs()
    _, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)

    cross_keep = model.attention.inner_attention.last_bkd_keep
    self_keep = model.self_attention.inner_attention.last_bkd_keep
    assert cross_keep is not None and self_keep is not None, "BKD not applied"

    ctx = model._adjacency_context(attn)
    assert ctx.shape == (BATCH, X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)

    dropped_cross = cross_keep == 0
    dropped_self = self_keep == 0
    if dropped_cross.any():
        assert (ctx[:, :, :S_SEQ_LEN][:, :, dropped_cross] == 0).all()
    if dropped_self.any():
        assert (ctx[:, :, S_SEQ_LEN:][:, :, dropped_self] == 0).all()


# 5. Context is detached — no gradient into structural parameters
def test_context_is_detached():
    torch.manual_seed(0)
    model = _make_model(per_node_adjacency_context=True)
    model.train()
    source, x_blanked, x_actual = _make_inputs()
    _, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
    ctx = model._adjacency_context(attn)
    assert ctx.requires_grad is False


# 6. Homogeneous mode — context spans all N = L_S + L_X nodes
def test_homogeneous_mode_forward():
    torch.manual_seed(0)
    model = _make_model(
        per_node_adjacency_context=True,
        homogeneous_nodes=True,
        batch_key_dropout=0.5,
    )
    model.train()
    source, x_blanked, x_actual = _make_inputs()
    s_blanked = source.clone()
    s_blanked[:, :, VALUE_COL] = 0.0
    pred, attn, _ = model.forward_with_actual(
        source, x_blanked, x_actual, s_blanked=s_blanked
    )
    n = S_SEQ_LEN + X_SEQ_LEN
    assert pred.shape == (BATCH, n, 1)
    ctx = model._adjacency_context(attn)
    assert ctx.shape == (BATCH, n, n)
    keep = model.attention.inner_attention.last_bkd_keep
    if keep is not None and (keep == 0).any():
        assert (ctx[:, :, keep == 0] == 0).all()


# 7. raw_value_adjacency arm — context uses the returned posterior (same
#    weights that built z), forward keeps shape
def test_raw_value_adjacency_arm():
    torch.manual_seed(0)
    model = _make_model(
        per_node_adjacency_context=True,
        raw_value_adjacency=True,
    )
    # Base width is already L_S+L_X; context adds another L_S+L_X.
    assert model.forecaster.d_model == 2 * (S_SEQ_LEN + X_SEQ_LEN)
    model.eval()
    source, x_blanked, x_actual = _make_inputs()
    pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# FiLM mode (per_node_adjacency_context="film")
# ---------------------------------------------------------------------------

# 8. True still maps to concat (backward compatible); "film" selects FiLM
def test_mode_parsing_backcompat():
    m_concat = _make_model(per_node_adjacency_context=True)
    assert m_concat._adjacency_context_mode == "concat"
    m_film = _make_model(per_node_adjacency_context="film")
    assert m_film._adjacency_context_mode == "film"
    m_off = _make_model(per_node_adjacency_context=False)
    assert m_off._adjacency_context_mode is None
    with pytest.raises(ValueError, match="per_node_adjacency_context"):
        _make_model(per_node_adjacency_context="bogus")


# 9. FiLM head: base width kept, conditioner built with zero-init last layer
def test_film_head_width_and_conditioner():
    model = _make_model(per_node_adjacency_context="film")
    head = model.forecaster
    assert head.d_model == D_MODEL  # not widened
    assert head.film is not None
    assert head.film_context_dim == S_SEQ_LEN + X_SEQ_LEN
    last = head.film[-1]
    assert (last.weight == 0).all() and (last.bias == 0).all()


# 10. Identity at init: zero-init FiLM == unconditioned decoder
def test_film_identity_at_init():
    torch.manual_seed(0)
    m_off = _make_model()
    torch.manual_seed(0)
    m_film = _make_model(per_node_adjacency_context="film")
    # FiLM adds only the conditioner params; the rest of the state dict must
    # match identically (same seed -> same init).
    sd_off = m_off.state_dict()
    sd_film = m_film.state_dict()
    for k in sd_off:
        assert torch.equal(sd_off[k], sd_film[k]), f"init mismatch at {k}"
    m_off.eval()
    m_film.eval()
    source, x_blanked, x_actual = _make_inputs()
    p_off, _, _ = m_off.forward_with_actual(source, x_blanked, x_actual)
    p_film, _, _ = m_film.forward_with_actual(source, x_blanked, x_actual)
    assert torch.allclose(p_off, p_film, atol=1e-6), (
        f"FiLM head not identity at init: max diff "
        f"{(p_off - p_film).abs().max().item()}"
    )


# 11. Forward shape + gradient reaches the conditioner; context stays detached
def test_film_forward_and_gradients():
    torch.manual_seed(0)
    model = _make_model(per_node_adjacency_context="film")
    model.train()
    source, x_blanked, x_actual = _make_inputs()
    pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)

    loss = pred_x.square().mean()
    loss.backward()
    # The conditioner's LAST (zero-init) layer receives gradient immediately
    # (its grad is delta x hidden, independent of its own zero weights);
    # the first layer's grad is exactly 0 at identity init by construction.
    g = model.forecaster.film[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "no gradient into conditioner"

    ctx = model._adjacency_context(attn)
    assert ctx.requires_grad is False


# 12. FiLM configured but context missing -> loud error
def test_film_requires_context():
    head = _make_model(per_node_adjacency_context="film").forecaster
    x = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
    var_ids = torch.arange(1, X_SEQ_LEN + 1).unsqueeze(0).expand(BATCH, -1)
    with pytest.raises(ValueError, match="film_context_dim"):
        head(x, var_ids, context=None)


# 13. FiLM with BKD: training forward completes, context zeros dropped keys
def test_film_with_bkd_forward():
    torch.manual_seed(0)
    model = _make_model(
        per_node_adjacency_context="film",
        batch_key_dropout=0.5,
    )
    model.train()
    source, x_blanked, x_actual = _make_inputs()
    pred_x, attn, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
    ctx = model._adjacency_context(attn)
    cross_keep = model.attention.inner_attention.last_bkd_keep
    if cross_keep is not None and (cross_keep == 0).any():
        assert (ctx[:, :, :S_SEQ_LEN][:, :, cross_keep == 0] == 0).all()

