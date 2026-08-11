"""Tests for the nonlinear (MLP) value embedding option (``embed: "mlp"``).

Run with:  pytest tests/test_atsel_mlp_value_emb.py -v

Motivation
----------
The NONLINEARITIES investigation (experiments/6_INVESTIGATIONS/NONLINEARITIES)
tests whether a nonlinear VALUE embedding lets the gate keep edges whose
contribution is not representable by a linear map of the scalar value (e.g. the
sign-changing S5*X2 term in scm2_continuous).  ``mlp_emb`` replaces the shared
``linear_emb`` value map with a small SHARED one-hidden-layer MLP
(Linear -> activation -> Linear), selectable from the config via ``embed: mlp``.

These tests verify:

1. ``ModularEmbedding`` accepts ``embed: "mlp"`` and returns the usual SVFA
   (emb_struct, emb_val) pair with unchanged shapes.
2. The map is genuinely nonlinear (unlike ``linear_emb``).
3. A full ``AttentionSelectorLayer`` forward pass with MLP value embeddings
   (svfa for both S and X, as in the production arm config) keeps output shape.
4. Gradient routing classifies the MLP value-embedding parameters as
   RECONSTRUCTION (they must train in the reconstruct phase).

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
from causaliT.core.modules.embedding_layers import linear_emb, mlp_emb
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

VALUE_COL = 0
VAR_COL = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _svfa_embed_cfg(vocab: int, value_embed: str = "mlp", d_model: int = D_MODEL) -> dict:
    """SVFA-split embedding config; value stream uses ``value_embed``."""
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VALUE_COL,
                "embed": value_embed,
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


def _make_model(value_embed: str = "mlp") -> AttentionSelectorLayer:
    return AttentionSelectorLayer(
        model="test_model",
        ds_embed_S=_svfa_embed_cfg(VOCAB_S, value_embed),
        ds_embed_X=_svfa_embed_cfg(VOCAB_X, value_embed),
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
    )


# ---------------------------------------------------------------------------
# 1. ModularEmbedding accepts "mlp"
# ---------------------------------------------------------------------------


def test_modular_embedding_accepts_mlp():
    torch.manual_seed(0)
    emb = ModularEmbedding(_svfa_embed_cfg(VOCAB_X, "mlp"), comps="svfa", device="cpu")
    source, x_actual = _make_inputs()
    emb_struct, emb_val = emb(x_actual)
    assert emb_struct.shape == (BATCH, X_SEQ_LEN, D_MODEL)
    assert emb_val.shape == (BATCH, X_SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 2. The MLP map is genuinely nonlinear
# ---------------------------------------------------------------------------


def test_mlp_is_nonlinear():
    torch.manual_seed(0)
    x = torch.tensor([[0.0], [1.0], [2.0]])  # (L=3,) -> unsqueeze batch dim
    mlp = mlp_emb(input_dim=1, embedding_dim=D_MODEL, device="cpu")
    lin = linear_emb(input_dim=1, embedding_dim=D_MODEL, device="cpu")
    f_mlp = mlp(x.unsqueeze(0)).squeeze(0)  # (3, D_MODEL)
    f_lin = lin(x.unsqueeze(0)).squeeze(0)
    # Second differences vanish exactly for an affine map, not for the MLP.
    assert torch.allclose(f_lin[2] - f_lin[1], f_lin[1] - f_lin[0])
    assert not torch.allclose(f_mlp[2] - f_mlp[1], f_mlp[1] - f_mlp[0])


# ---------------------------------------------------------------------------
# 3. Full model forward with MLP value embeddings
# ---------------------------------------------------------------------------


def test_forward_shape_with_mlp_value_embedding():
    torch.manual_seed(0)
    model = _make_model("mlp")
    model.eval()
    source, x_actual = _make_inputs()
    x_blanked = x_actual.clone()
    x_blanked[:, :, VALUE_COL] = 0.0  # value-blanked queries, as the forecaster does
    with torch.no_grad():
        pred_x, _, _ = model.forward_with_actual(source, x_blanked, x_actual)
    assert pred_x.shape == (BATCH, X_SEQ_LEN, 1)


# ---------------------------------------------------------------------------
# 4. Gradient routing: MLP value params are RECONSTRUCTION
# ---------------------------------------------------------------------------


def test_mlp_value_params_are_reconstruction():
    torch.manual_seed(0)
    model = _make_model("mlp")

    # Collect the value-embedding parameters directly from the SVFA value lists.
    val_params = []
    for emb_map in list(model.embedding_S.value_modules_list) + list(
        model.embedding_X.value_modules_list
    ):
        val_params += list(emb_map.parameters())
    assert len(val_params) > 0, "expected MLP value-embedding parameters"

    structural, reconstruction = classify_parameters(model)
    struct_ids = {id(p) for p in structural}
    recon_ids = {id(p) for p in reconstruction}

    for p in val_params:
        assert id(p) in recon_ids, "MLP value-embedding params must be RECONSTRUCTION"
        assert id(p) not in struct_ids, "MLP value-embedding params must NOT be structural"
