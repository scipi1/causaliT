"""Tests for the per-node (DAGMA-style) value embedding and output head.

Run with:  pytest tests/test_atsel_per_node_value_emb.py -v

Motivation
----------
The FIT investigation (experiments/6_INVESTIGATIONS/FIT) tests whether
giving every node its own nonlinear value functional (one MLP per variable
for both encoding and decoding) improves reconstruction capacity compared
to a single shared MLP.  This is the DAGMA design: the only cross-node
mixing is the structural attention itself.

These tests verify:

1. ``ModularEmbedding`` accepts ``embed: "mlp_per_node"`` and returns the
   usual SVFA (emb_struct, emb_val) pair with unchanged shapes.
2. The per-node map is genuinely node-specific (different nodes produce
   different embeddings for the same input value).
3. A full ``AttentionSelectorLayer`` forward pass with per-node value
   embeddings AND per-node output head keeps output shape.
4. Gradient routing classifies the per-node value-embedding and per-node
   forecaster parameters as RECONSTRUCTION.
5. Homogeneous mode works (S nodes are also reconstructed by the per-node
   head).

Column convention: ``value`` at column 0, ``variable-ID`` at column 1
(1-indexed, 0 = padding), as in production.
"""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.architectures.attention_selector import AttentionSelectorLayer
from causaliT.core.modules.embedding import ModularEmbedding
from causaliT.core.modules.embedding_layers import mlp_per_node_emb
from causaliT.core.modules.mlp_head import PerNodeMLPHead
from causaliT.training.gradient_routing import classify_parameters


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
PER_NODE_HIDDEN = 8

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, value_embed: str = "mlp_per_node", d_model: int = D_MODEL, value_dropout: float = 0.0) -> dict:
    """SVFA-split embedding config; value stream uses ``value_embed``."""
    value_kwargs = {"input_dim": 1, "embedding_dim": d_model}
    if value_embed == "mlp_per_node":
        value_kwargs["hidden_dim"] = PER_NODE_HIDDEN
        value_kwargs["num_variables"] = vocab - 1  # exclude padding
        if value_dropout > 0.0:
            value_kwargs["dropout"] = value_dropout
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": value_embed,
                "label": "value",
                "role": "value",
                "kwargs": value_kwargs,
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


def _make_inputs():
    """Random (S, X) tensors in the production (value, variable-ID) layout."""
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
    return source, x_actual


def _make_model(value_embed: str = "mlp_per_node", per_node_output: bool = True, value_dropout: float = 0.0) -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S, value_embed, value_dropout=value_dropout),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X, value_embed, value_dropout=value_dropout),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
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
        struct_embedding_type="standard_learnable",
        free_query_embedding=False,
        query_centroid_init=False,
        per_node_output=per_node_output,
        per_node_output_hidden=PER_NODE_HIDDEN,
    )


# ---------------------------------------------------------------------------
# 1. ModularEmbedding accepts "mlp_per_node"
# ---------------------------------------------------------------------------


def test_modular_embedding_accepts_mlp_per_node():
    torch.manual_seed(0)
    emb = ModularEmbedding(_svfa_embed_cfg(VOCAB_X, "mlp_per_node"), comps="svfa", device="cpu")
    source, x_actual = _make_inputs()
    emb_struct, emb_val = emb(x_actual)
    assert emb_struct.shape == (BATCH, X_SEQ_LEN, D_MODEL)
    assert emb_val.shape == (BATCH, X_SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 2. The per-node map is genuinely node-specific
# ---------------------------------------------------------------------------


def test_mlp_per_node_is_node_specific():
    torch.manual_seed(0)
    emb = mlp_per_node_emb(
        input_dim=1,
        embedding_dim=D_MODEL,
        num_variables=X_SEQ_LEN,
        device="cpu",
        hidden_dim=PER_NODE_HIDDEN,
    )
    # Same scalar value for all nodes.
    values = torch.ones(1, X_SEQ_LEN)
    var_ids = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0)
    out = emb(values, var_ids)  # (1, X_SEQ_LEN, D_MODEL)
    # Different nodes must produce different embeddings (independent MLPs).
    for i in range(X_SEQ_LEN):
        for j in range(i + 1, X_SEQ_LEN):
            assert not torch.allclose(out[0, i], out[0, j]), (
                f"Nodes {i} and {j} produced identical embeddings"
            )


# ---------------------------------------------------------------------------
# 3. Full model forward with per-node value embedding + per-node output
# ---------------------------------------------------------------------------


def test_forward_shape_with_per_node_value_and_output():
    torch.manual_seed(0)
    model = _make_model("mlp_per_node", per_node_output=True)
    model.eval()
    source, x_actual = _make_inputs()
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0  # value-blanked queries, as the forecaster does
    with torch.no_grad():
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# 4. Gradient routing: per-node params are RECONSTRUCTION
# ---------------------------------------------------------------------------


def test_per_node_params_are_reconstruction():
    torch.manual_seed(0)
    model = _make_model("mlp_per_node", per_node_output=True)

    # Collect the per-node value-embedding parameters.
    val_params = []
    for emb_map in list(model.embedding_S.value_modules_list) + list(
        model.embedding_X.value_modules_list
    ):
        val_params += list(emb_map.parameters())
    assert len(val_params) > 0, "expected per-node value-embedding parameters"

    # Collect the per-node forecaster parameters.
    forecaster_params = list(model.forecaster.parameters())
    assert len(forecaster_params) > 0, "expected per-node forecaster parameters"

    structural, reconstruction = classify_parameters(model)
    struct_ids = {id(p) for p in structural}
    recon_ids = {id(p) for p in reconstruction}

    for p in val_params:
        assert id(p) in recon_ids, "per-node value-embedding params must be RECONSTRUCTION"
        assert id(p) not in struct_ids, "per-node value-embedding params must NOT be structural"

    for p in forecaster_params:
        assert id(p) in recon_ids, "per-node forecaster params must be RECONSTRUCTION"
        assert id(p) not in struct_ids, "per-node forecaster params must NOT be structural"


# ---------------------------------------------------------------------------
# 5. Homogeneous mode with per-node output
# ---------------------------------------------------------------------------


def test_homogeneous_mode_with_per_node_output():
    torch.manual_seed(0)
    model = AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S, "mlp_per_node"),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X, "mlp_per_node"),
        comps_embed_S="svfa",
        comps_embed_X="svfa",
        attention_type="GatedCrossAttention",  # ignored in homogeneous mode
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
        struct_embedding_type="standard_learnable",
        free_query_embedding=False,
        query_centroid_init=False,
        homogeneous_nodes=True,
        per_node_output=True,
        per_node_output_hidden=PER_NODE_HIDDEN,
    )
    model.eval()
    source, x_actual = _make_inputs()
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    s_blanked = source.clone()
    s_blanked[:, :, VALUE_COL] = 0.0
    with torch.no_grad():
        pred_x, attn, _ = model.forward_with_actual(
            source, x_blanked, x_actual, s_blanked=s_blanked
        )
    # Homogeneous mode reconstructs ALL N = L_S + L_X nodes.
    assert pred_x.shape == (BATCH, S_SEQ_LEN + X_SEQ_LEN, 1)
    assert attn.shape == (BATCH, S_SEQ_LEN + X_SEQ_LEN, S_SEQ_LEN + X_SEQ_LEN)


# ---------------------------------------------------------------------------
# 6. Dropout in the per-node value embedding: default is a no-op
# ---------------------------------------------------------------------------


def test_mlp_per_node_dropout_default_disabled():
    """Default dropout=0.0 must be a no-op (nn.Identity, backward compatible)."""
    torch.manual_seed(0)
    emb = mlp_per_node_emb(
        input_dim=1,
        embedding_dim=D_MODEL,
        num_variables=X_SEQ_LEN,
        device="cpu",
        hidden_dim=PER_NODE_HIDDEN,
    )
    assert isinstance(emb.dropout, torch.nn.Identity)
    # Layout per node: Linear -> activation -> Identity -> Linear.
    for mlp in emb.mlps:
        assert len(mlp) == 4
        assert isinstance(mlp[2], torch.nn.Identity)


# ---------------------------------------------------------------------------
# 7. Dropout placement and train/eval behaviour
# ---------------------------------------------------------------------------


def test_mlp_per_node_dropout_train_stochastic_eval_deterministic():
    """dropout>0 sits after the hidden activation; active in train, off in eval."""
    torch.manual_seed(0)
    emb_drop = mlp_per_node_emb(
        input_dim=1,
        embedding_dim=D_MODEL,
        num_variables=X_SEQ_LEN,
        device="cpu",
        hidden_dim=PER_NODE_HIDDEN,
        dropout=0.5,
    )
    # Placement: Linear -> activation -> Dropout -> Linear in every per-node MLP.
    for mlp in emb_drop.mlps:
        assert isinstance(mlp[2], torch.nn.Dropout)
        assert mlp[2].p == 0.5

    values = torch.randn(BATCH, X_SEQ_LEN)
    var_ids = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).expand(BATCH, -1)

    # Train mode: two forwards with different RNG seeds differ (dropout active).
    emb_drop.train()
    torch.manual_seed(0)
    out_train_1 = emb_drop(values, var_ids)
    torch.manual_seed(1)
    out_train_2 = emb_drop(values, var_ids)
    assert not torch.allclose(out_train_1, out_train_2)

    # Eval mode: dropout off -> deterministic and equal to a dropout-free copy
    # with identical weights (nn.Dropout owns no parameters, so the state_dict
    # is interchangeable with the default construction).
    emb_nodrop = mlp_per_node_emb(
        input_dim=1,
        embedding_dim=D_MODEL,
        num_variables=X_SEQ_LEN,
        device="cpu",
        hidden_dim=PER_NODE_HIDDEN,
        dropout=0.0,
    )
    emb_nodrop.load_state_dict(emb_drop.state_dict())
    emb_drop.eval()
    emb_nodrop.eval()
    with torch.no_grad():
        out_eval_drop = emb_drop(values, var_ids)
        out_eval_drop_2 = emb_drop(values, var_ids)
        out_eval_nodrop = emb_nodrop(values, var_ids)
    assert torch.allclose(out_eval_drop, out_eval_drop_2)
    assert torch.allclose(out_eval_drop, out_eval_nodrop)


# ---------------------------------------------------------------------------
# 8. Dropout kwarg flows through the ds_embed config into the full model
# ---------------------------------------------------------------------------


def test_forward_shape_with_per_node_value_dropout():
    """ModularEmbedding routes the dropout kwarg; train-mode forward keeps shape."""
    torch.manual_seed(0)
    cfg = _svfa_embed_cfg(VOCAB_X, "mlp_per_node", value_dropout=0.3)
    emb = ModularEmbedding(cfg, comps="svfa", device="cpu")
    per_node_emb = emb.per_node_value_embed_list[0].embedding
    assert isinstance(per_node_emb.dropout, torch.nn.Dropout)
    assert per_node_emb.dropout.p == 0.3

    # Full model forward in TRAIN mode (dropout active) keeps the output shape.
    model = _make_model("mlp_per_node", per_node_output=True, value_dropout=0.3)
    model.train()
    source, x_actual = _make_inputs()
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0
    pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)
