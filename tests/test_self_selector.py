"""
Tests for the SelfSelectorLayer architecture and SelfSelectorForecaster.

Covers:
  * SelfSelectorLayer forward shapes (pred + full (N, N) posterior + aux);
  * no-self-loop diagonal on the directed posterior;
  * split_attention block shapes (s_to_x / x_to_x / x_to_s / s_to_s);
  * source_scores shape;
  * backward pass produces gradients (differentiable selector);
  * config validation guards (attention_type, shared_dag_across_heads, gain source);
  * SelfSelectorForecaster end-to-end _step (homogeneous [S, X], id offset,
    full-matrix HSIC + NOTEARS) with and without gradient routing.
"""

import copy
import pytest
import torch

from causaliT.core.architectures.self_selector import SelfSelectorLayer
from causaliT.training.forecasters import SelfSelectorForecaster


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

VAL_IDX = 0
VAR_IDX = 1
S_LEN, X_LEN = 2, 3
N = S_LEN + X_LEN
D_MODEL = 16


def _ds_embed(num_embeddings_shared: int):
    """Minimal ModularEmbedding config: linear value + nn_embedding id + mask."""
    return {
        "setting": {"d_model": D_MODEL, "sparse_grad": False},
        "modules": [
            {
                "idx": VAL_IDX, "embed": "linear", "label": "value", "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": D_MODEL},
            },
            {
                "idx": VAR_IDX, "embed": "nn_embedding", "label": "variable",
                "role": "structure",
                "kwargs": {
                    "num_embeddings": num_embeddings_shared,
                    "embedding_dim": D_MODEL, "padding_idx": 0,
                    "sparse": False, "max_norm": 1,
                },
            },
            {"idx": VAL_IDX, "embed": "mask", "label": "value_missing", "kwargs": {}},
        ],
    }


def _model_kwargs(comps_embed="summation", n_heads=1, **overrides):
    kwargs = dict(
        model="SelfSelectorLayer",
        ds_embed=_ds_embed(N + 1),
        comps_embed=comps_embed,
        attention_type="GatedSelfAttention",
        n_heads=n_heads,
        dropout_emb=0.0, dropout_attn_out=0.0, dropout_ff=0.0,
        dropout_qkv=0.0, attention_dropout=0.0,
        activation="gelu", norm="layer", use_final_norm=True, device="cpu",
        out_dim=1, d_ff=32, d_model=D_MODEL, d_qk=8,
        S_seq_len=S_LEN, X_seq_len=X_LEN,
        shared_dag_across_heads=True,
        struct_embedding_type="standard_learnable",
        free_query_embedding=False,
        gain_stream_source="separate",
    )
    kwargs.update(overrides)
    return kwargs


def _make_nodes(batch=6, seed=0):
    """Build (B, N, 2) tensors: column 0 = value, column 1 = 1-indexed var id."""
    g = torch.Generator().manual_seed(seed)
    vals = torch.randn(batch, N, 1, generator=g)
    ids = torch.arange(1, N + 1).float().view(1, N, 1).expand(batch, N, 1)
    all_actual = torch.cat([vals, ids], dim=-1)
    all_blanked = all_actual.clone()
    all_blanked[:, :, VAL_IDX] = 0.0
    return all_actual, all_blanked


# ---------------------------------------------------------------------------
# SelfSelectorLayer
# ---------------------------------------------------------------------------

def test_forward_shapes_and_diagonal():
    model = SelfSelectorLayer(**_model_kwargs()).eval()
    all_actual, all_blanked = _make_nodes()
    pred, attn, aux = model.forward_with_actual(all_blanked, all_actual)
    B = all_actual.shape[0]
    assert pred.shape == (B, N, 1)
    assert attn.shape == (B, N, N)
    diag = torch.diagonal(attn, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    assert "l0_penalty" in aux and "entropy" in aux


def test_split_attention_shapes():
    model = SelfSelectorLayer(**_model_kwargs()).eval()
    all_actual, all_blanked = _make_nodes()
    _, attn, _ = model.forward_with_actual(all_blanked, all_actual)
    blocks = model.split_attention(attn)
    B = all_actual.shape[0]
    assert blocks["s_to_x"].shape == (B, X_LEN, S_LEN)
    assert blocks["x_to_x"].shape == (B, X_LEN, X_LEN)
    assert blocks["x_to_s"].shape == (B, S_LEN, X_LEN)
    assert blocks["s_to_s"].shape == (B, S_LEN, S_LEN)


def test_source_scores_shape():
    model = SelfSelectorLayer(**_model_kwargs()).eval()
    all_actual, all_blanked = _make_nodes()
    _, attn, _ = model.forward_with_actual(all_blanked, all_actual)
    scores = model.source_scores(attn)
    assert scores.shape == (all_actual.shape[0], N)
    assert torch.all(scores >= 0)  # incoming-edge mass is non-negative


def test_backward_produces_gradients():
    model = SelfSelectorLayer(**_model_kwargs()).train()
    all_actual, all_blanked = _make_nodes()
    pred, attn, aux = model.forward_with_actual(all_blanked, all_actual)
    loss = pred.pow(2).mean() + aux["l0_penalty"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_svfa_multihead_forward():
    model = SelfSelectorLayer(**_model_kwargs(comps_embed="svfa", n_heads=2)).eval()
    all_actual, all_blanked = _make_nodes()
    pred, attn, _ = model.forward_with_actual(all_blanked, all_actual)
    assert pred.shape == (all_actual.shape[0], N, 1)
    assert attn.shape == (all_actual.shape[0], N, N)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

def test_rejects_non_gated_self_attention():
    with pytest.raises(ValueError):
        SelfSelectorLayer(**_model_kwargs(attention_type="ScaledDotSoftmax"))


def test_rejects_per_head_structure():
    with pytest.raises(ValueError):
        SelfSelectorLayer(**_model_kwargs(shared_dag_across_heads=False))


def test_shared_gain_requires_orthogonal_fixed():
    with pytest.raises(ValueError):
        SelfSelectorLayer(**_model_kwargs(
            gain_stream_source="shared",
            struct_embedding_type="standard_learnable",
        ))


# ---------------------------------------------------------------------------
# SelfSelectorForecaster
# ---------------------------------------------------------------------------

def _forecaster_config(use_gradient_routing=False):
    return {
        "model": {
            "model_object": "SelfSelectorLayer",
            "kwargs": _model_kwargs(),
        },
        "data": {
            "val_idx": VAL_IDX,
            "feature_indices": {"value": VAL_IDX, "variable": VAR_IDX},
            "S_seq_len": S_LEN,
            "X_seq_len": X_LEN,
            "dataset": "dummy",
        },
        "training": {
            "loss_fn": "mse",
            "lambda_recon": 1.0,
            "lambda_struct_recon": 0.0,
            "lambda_score_sparse": 0.0,
            "lambda_hsic": 0.5,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": False,
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "lambda_group_l1": 0.0,
            "lambda_l0": 0.1,
            "kappa": 0.1,
            "use_gradient_routing": use_gradient_routing,
            "use_oracle_attention": False,
            "use_hard_masks": False,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    }


def _batch(batch=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    # S and X each carry their OWN 1-indexed var ids (both start at 1); the
    # forecaster offsets X ids by S_seq_len so the shared table stays collision-free.
    s_vals = torch.randn(batch, S_LEN, 1, generator=g)
    s_ids = torch.arange(1, S_LEN + 1).float().view(1, S_LEN, 1).expand(batch, S_LEN, 1)
    S = torch.cat([s_vals, s_ids], dim=-1)
    x_vals = torch.randn(batch, X_LEN, 1, generator=g)
    x_ids = torch.arange(1, X_LEN + 1).float().view(1, X_LEN, 1).expand(batch, X_LEN, 1)
    X = torch.cat([x_vals, x_ids], dim=-1)
    return [S, X]


def test_forecaster_step_runs():
    model = SelfSelectorForecaster(_forecaster_config())
    batch = _batch()
    loss, pred, X = model._step(batch, stage="train")
    assert torch.isfinite(loss)
    assert pred.shape[0] == batch[0].shape[0]


def test_forecaster_id_offset_makes_ids_unique():
    model = SelfSelectorForecaster(_forecaster_config())
    S, X = _batch(batch=1)
    all_actual, _ = model._assemble_nodes(S, X)
    ids = all_actual[0, :, VAR_IDX].long().tolist()
    # S ids 1,2 and X ids offset by S_LEN -> 3,4,5 : all distinct, cover 1..N.
    assert sorted(ids) == list(range(1, N + 1))


def test_forecaster_gradient_routing_step():
    model = SelfSelectorForecaster(_forecaster_config(use_gradient_routing=True))
    assert model.automatic_optimization is False
    assert len(model._structural_params) > 0
    assert len(model._reconstruction_params) > 0
    # A plain _step must still be differentiable in routing mode.
    loss, _, _ = model._step(_batch(), stage="train")
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Gradient-routing parameter classification (identity-based parameter_groups)
# ---------------------------------------------------------------------------
#
# Regression guard: the self-selector uses SINGULAR attribute names
# (``orth_embed`` / ``query_embed``) and an ambiguously-named shared embedding
# (``embed_modules_list.<i>``), so the name-substring router misroutes the
# orthogonal + free-query structural embeddings and the SVFA structure table to
# the reconstruction group.  ``parameter_groups()`` partitions by MODULE
# REFERENCE and must classify every parameter correctly.

def _grouped_name_sets(model):
    """Return (structural_names, reconstruction_names) via parameter_groups()."""
    struct_params, recon_params = model.parameter_groups()
    struct_ids = {id(p) for p in struct_params}
    recon_ids = {id(p) for p in recon_params}
    struct_names, recon_names = set(), set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in struct_ids:
            struct_names.add(name)
        elif id(p) in recon_ids:
            recon_names.add(name)
    return struct_names, recon_names


def _assert_partition_complete(model, struct_names, recon_names):
    """Every trainable param lands in exactly one group."""
    all_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert struct_names.isdisjoint(recon_names)
    assert struct_names | recon_names == all_names


def test_parameter_groups_qk_and_gain_baseline():
    """Q/K projections -> structural; gain stream + V/out/FFN/head -> recon."""
    model = SelfSelectorLayer(**_model_kwargs(comps_embed="summation"))
    struct, recon = _grouped_name_sets(model)
    _assert_partition_complete(model, struct, recon)

    assert any("attention.query_projection" in n for n in struct)
    assert any("attention.key_projection" in n for n in struct)
    # Reconstruction stream
    for tok in ("gain_q_embed", "gain_k_embed", "gain_q_proj", "gain_k_proj",
                "value_projection", "out_projection", "forecaster", "linear1"):
        assert any(tok in n for n in recon), f"{tok} should be reconstruction"
    # In summation mode the shared embedding feeds Q/K AND V -> reconstruction.
    assert any("embedding.embed_modules_list" in n for n in recon)
    assert not any("embedding.embed_modules_list" in n for n in struct)


def test_parameter_groups_svfa_structure_table_is_structural():
    """In SVFA mode the structure embedding feeds Q/K only -> structural."""
    model = SelfSelectorLayer(**_model_kwargs(comps_embed="svfa"))
    struct_params, recon_params = model.parameter_groups()
    struct_ids = {id(p) for p in struct_params}
    recon_ids = {id(p) for p in recon_params}
    # Membership is checked by parameter IDENTITY: PyTorch's named_parameters
    # dedups the SVFA structure/value modules (they are re-registered under the
    # shared ``embed_modules_list`` path), so a name-substring check is unusable.
    struct_table_ids = {
        id(p) for p in model.embedding.structure_modules_list.parameters()
    }
    value_table_ids = {
        id(p) for p in model.embedding.value_modules_list.parameters()
    }
    assert struct_table_ids and struct_table_ids <= struct_ids
    assert value_table_ids and value_table_ids <= recon_ids



def test_parameter_groups_orthogonal_learnable_is_structural():
    model = SelfSelectorLayer(**_model_kwargs(
        struct_embedding_type="orthogonal_learnable",
    ))
    struct, recon = _grouped_name_sets(model)
    _assert_partition_complete(model, struct, recon)
    assert any("orth_embed" in n for n in struct)
    assert not any("orth_embed" in n for n in recon)
    # Gain tables must NOT be dragged in with the orthogonal embedding.
    assert any("gain_q_embed" in n for n in recon)


def test_parameter_groups_free_query_is_structural():
    model = SelfSelectorLayer(**_model_kwargs(free_query_embedding=True))
    struct, recon = _grouped_name_sets(model)
    _assert_partition_complete(model, struct, recon)
    assert any("query_embed" in n for n in struct)
    assert not any("query_embed" in n for n in recon)


def test_parameter_groups_orthogonal_fixed_and_shared_gain():
    """orthogonal_fixed frame is buffer-only; shared gain has no separate table."""
    model = SelfSelectorLayer(**_model_kwargs(
        comps_embed="svfa",
        struct_embedding_type="orthogonal_fixed",
        gain_stream_source="shared",
    ))
    struct, recon = _grouped_name_sets(model)
    _assert_partition_complete(model, struct, recon)
    # No separate gain identity tables in shared mode.
    assert not any("gain_q_embed" in n or "gain_k_embed" in n
                   for n in struct | recon)
    assert any("attention.query_projection" in n for n in struct)


def test_forecaster_routing_uses_identity_grouping():
    """Forecaster with orthogonal+free-query must route orth/query as structural."""
    kwargs = _model_kwargs(
        comps_embed="svfa",
        struct_embedding_type="orthogonal_learnable",
        free_query_embedding=True,
    )
    cfg = _forecaster_config(use_gradient_routing=True)
    cfg["model"]["kwargs"] = kwargs
    model = SelfSelectorForecaster(cfg)
    struct_ids = {id(p) for p in model._structural_params}
    recon_ids = {id(p) for p in model._reconstruction_params}
    for name, p in model.model.named_parameters():
        if not p.requires_grad:
            continue
        if "orth_embed" in name or "query_embed" in name:
            assert id(p) in struct_ids, f"{name} must be structural"
        if "gain_q_embed" in name or "gain_k_embed" in name:
            assert id(p) in recon_ids, f"{name} must be reconstruction"
