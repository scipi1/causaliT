"""Cheater arm: GT hard masks WITHOUT oracle attention.

The "cheater" benchmark gives the vanilla cross-only transformer the TRUE parent
SUPPORT while leaving the attention weights learned.  This pins down the three
reachable regimes of ``forward_with_actual``, which are driven by two
INDEPENDENT knobs (``oracle_combined_mask`` and ``oracle``):

    mask=None, oracle=False -> learned QK^T, structural mask only.
    mask=GT,   oracle=False -> learned QK^T renormalised over the true parents.
    mask=GT,   oracle=True  -> QK^T bypassed; GT adjacency IS the attention.

Note that the third regime is NOT available to the vanilla arm: the oracle path
is unimplemented for ``ScaledDotSoftmax``, so a full oracle needs an attention
type that supports it (e.g. ``CausalCrossAttention``).

Regression guard: the middle regime used to be unreachable because the model
only consulted the GT mask when ``oracle=True``, so the cheater arm silently
trained as a plain vanilla transformer.
"""

import os

import pandas as pd
import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)

D, VOCAB, LS, LX = 16, 8, 3, 3
VAL, VAR = 1, 0

# GT S->X mask (L_X, L_S): each X has exactly one S parent.
CROSS_GT = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
# GT X->X mask (L_X, L_X): strictly lower-triangular chain x0 -> x1 -> x2.
SELF_GT = [[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]


def _embed_cfg():
    return {
        "setting": {"d_model": D},
        "modules": [
            {"idx": VAR, "embed": "nn_embedding", "label": "variable",
             "role": "structure",
             "kwargs": {"num_embeddings": VOCAB, "embedding_dim": D}},
            {"idx": VAL, "embed": "linear", "label": "value", "role": "value",
             "kwargs": {"input_dim": 1, "embedding_dim": D}},
        ],
    }


def _config(
    use_hard_masks: bool,
    use_oracle: bool,
    attention_type: str = "ScaledDotSoftmax",
) -> dict:
    return {
        "data": {"val_idx": VAL, "S_seq_len": LS, "X_seq_len": LX,
                 "dataset": "dummy"},
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(),
                "ds_embed_X": _embed_cfg(),
                "comps_embed_S": "summation",
                "comps_embed_X": "summation",
                # Vanilla transformer arm: softmax attention, no self block.
                "attention_type": attention_type,
                "self_attention_type": None,
                "n_heads": 4,
                "dropout_emb": 0.0, "dropout_attn_out": 0.0, "dropout_ff": 0.0,
                "dropout_qkv": 0.0, "attention_dropout": 0.0,
                "activation": "gelu", "norm": "layer", "use_final_norm": True,
                "device": "cpu", "out_dim": 1, "d_ff": 32, "d_model": D,
                "d_qk": D, "S_seq_len": LS, "X_seq_len": LX,
                "struct_embedding_type": "standard_learnable",
                "value_structure_injection": "none",
                "value_structure_query_injection": "none",
            },
        },
        "training": {
            "loss_fn": "mse", "lr": 1e-3, "weight_decay": 0.0,
            "optimizer": "adamw", "use_gradient_routing": False,
            "lambda_recon": 1.0, "lambda_struct_recon": 0.0,
            "lambda_hsic": 0.0, "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0, "lambda_l0": 0.0, "kappa": 0.0,
            "lambda_query_norm": 0.0,
            "hsic_sigma": 1.0, "hsic_adaptive_bandwidth": True,
            "hsic_mode": "biased", "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "use_oracle_attention": use_oracle,
            "use_hard_masks": use_hard_masks,
            "hard_mask_files": {
                "dec_cross": "dec1_cross_att_mask.csv",
                "dec_self": "dec1_self_att_mask.csv",
            },
            "hard_masks_corruption_seed": None,
            "cross_control_shd": 0, "self_control_shd": 0,
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }


@pytest.fixture
def data_dir(tmp_path):
    """Write the GT DAG mask CSVs where the forecaster's loader expects them."""
    ds_dir = tmp_path / "dummy"
    ds_dir.mkdir()
    pd.DataFrame(CROSS_GT).to_csv(os.path.join(ds_dir, "dec1_cross_att_mask.csv"))
    pd.DataFrame(SELF_GT).to_csv(os.path.join(ds_dir, "dec1_self_att_mask.csv"))
    return str(tmp_path)


def _batch(batch_size: int = 8):
    g = torch.Generator().manual_seed(0)
    S = torch.zeros(batch_size, LS, 2)
    S[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LS), generator=g).float()
    S[:, :, VAL] = torch.randn(batch_size, LS, generator=g)
    X = torch.zeros(batch_size, LX, 2)
    X[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LX), generator=g).float()
    X[:, :, VAL] = torch.randn(batch_size, LX, generator=g)
    return S, X


def _attention(fc, S, X, oracle, mask):
    fc.eval()
    X_blank = X.clone()
    X_blank[:, :, VAL] = 0.0
    with torch.no_grad():
        _, attn, _ = fc.model.forward_with_actual(
            S, X_blank, X, oracle=oracle, oracle_combined_mask=mask
        )
    return attn


def test_cheater_zeroes_non_parents_without_oracle(data_dir):
    """mask=GT, oracle=False: non-parents get EXACTLY zero, rows still sum to 1."""
    fc = AttentionSelectorForecaster(_config(True, False), data_dir=data_dir)
    assert fc.model.cross_only, "cheater arm must be the cross-only vanilla block"
    assert not fc.use_oracle
    gt = fc.oracle_combined_mask
    assert gt is not None and gt.shape == (LX, LS + LX)

    S, X = _batch()
    attn = _attention(fc, S, X, oracle=False, mask=gt)

    forbidden = gt.unsqueeze(0).expand_as(attn) == 0
    assert torch.all(attn[forbidden] == 0.0), "non-parent received attention mass"
    # Softmax renormalises over the surviving (true) parents.
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    # The weights are LEARNED, not uniform: x1/x2 have 2 parents each, so a
    # uniform oracle would put exactly 0.5 on each.
    two_parent_rows = attn[:, 1:, :]
    nonzero = two_parent_rows[two_parent_rows > 0]
    assert not torch.allclose(nonzero, torch.full_like(nonzero, 0.5), atol=1e-3)


def test_no_mask_leaves_structural_mask_only(data_dir):
    """mask=None, oracle=False: only the zero diagonal constrains the attention."""
    fc = AttentionSelectorForecaster(_config(False, False), data_dir=data_dir)
    # With hard masks disabled the forecaster never builds the attribute.
    assert getattr(fc, "oracle_combined_mask", None) is None

    S, X = _batch()
    attn = _attention(fc, S, X, oracle=False, mask=None)

    # Every non-diagonal key keeps mass; only the X self-loops are forbidden.
    diag = torch.eye(LX).bool()
    assert torch.all(attn[:, :, LS:][:, diag] == 0.0)
    assert torch.all(attn[:, :, :LS] > 0.0)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)


def test_oracle_makes_gt_the_attention(data_dir):
    """mask=GT, oracle=True: QK^T bypassed, GT adjacency IS the attention.

    Uses CausalCrossAttention because the oracle path is not implemented for
    ScaledDotSoftmax (see test_vanilla_softmax_rejects_oracle).
    """
    cfg = _config(True, True, attention_type="CausalCrossAttention")
    fc = AttentionSelectorForecaster(cfg, data_dir=data_dir)
    assert fc.use_oracle
    gt = fc.oracle_combined_mask

    S, X = _batch()
    attn = _attention(fc, S, X, oracle=True, mask=gt)

    forbidden = gt.unsqueeze(0).expand_as(attn) == 0
    assert torch.all(attn[forbidden] == 0.0)
    # Surviving entries reproduce the GT adjacency exactly.
    allowed = ~forbidden
    assert torch.all(attn[allowed] > 0.0)
    assert torch.allclose(
        (attn > 0).float(), gt.unsqueeze(0).expand_as(attn), atol=1e-6
    )


def test_vanilla_softmax_rejects_oracle(data_dir):
    """ScaledDotSoftmax has no oracle path, so the vanilla arm cannot use one.

    This is why the cheater arm masks the SCORES (oracle=False) instead of
    replacing the attention outright.
    """
    fc = AttentionSelectorForecaster(_config(True, True), data_dir=data_dir)
    S, X = _batch()
    with pytest.raises(NotImplementedError, match="Oracle attention mode"):
        _attention(fc, S, X, oracle=True, mask=fc.oracle_combined_mask)


def test_cheater_training_step_backprops(data_dir):
    """The masked attention still yields a finite, differentiable loss."""
    fc = AttentionSelectorForecaster(_config(True, False), data_dir=data_dir)
    S, X = _batch()

    fc.train()
    out = fc.training_step((S, X), 0)
    loss = out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in fc.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_oracle_requires_hard_masks(data_dir):
    """use_oracle_attention=True without hard masks is a configuration error."""
    with pytest.raises(ValueError, match="use_hard_masks"):
        AttentionSelectorForecaster(_config(False, True), data_dir=data_dir)


def test_eval_load_without_data_dir_keeps_mask(data_dir):
    """Eval-time ``load_from_checkpoint`` (no data_dir) must keep the GT mask.

    Regression guard for the published cheater arm: training registers the
    ``oracle_combined_mask`` buffer (data_dir available), so the checkpoint
    carries it.  ``AttentionSelectorPredictor._load_model`` calls
    ``load_from_checkpoint`` WITHOUT ``data_dir``, so __init__ never registers
    the buffer and PL's strict load_state_dict used to raise
    ``RuntimeError: Unexpected key(s) in state_dict: "oracle_combined_mask"``
    -- killing eval_interventions / eval_attention_scores for every cheater
    run.  ``on_load_checkpoint`` now registers the buffer straight from the
    checkpoint tensor, so strict loading succeeds AND the mask stays applied
    (stripping the key instead would silently evaluate the cheater as a
    plain vanilla model).
    """
    cfg = _config(True, False)
    # Training-time model: data_dir available -> buffer registered + saved.
    trained = AttentionSelectorForecaster(cfg, data_dir=data_dir)
    gt_mask = trained.oracle_combined_mask.detach().clone()
    checkpoint = {"state_dict": dict(trained.state_dict())}
    assert "oracle_combined_mask" in checkpoint["state_dict"]

    # Eval-time model: no data_dir -> buffer missing before the fix.
    fresh = AttentionSelectorForecaster(cfg, data_dir=None)
    assert getattr(fresh, "oracle_combined_mask", None) is None
    assert not fresh._hard_masks_loaded

    # The PL hook runs before strict load_state_dict.
    fresh.on_load_checkpoint(checkpoint)
    assert hasattr(fresh, "oracle_combined_mask")
    assert fresh._hard_masks_loaded, "mask must stay active, else no cheating"

    # Strict load now succeeds and writes the checkpoint values.
    fresh.load_state_dict(checkpoint["state_dict"], strict=True)
    assert torch.equal(fresh.oracle_combined_mask, gt_mask)

    # Forward must apply the mask: non-parents receive exactly zero mass.
    # The predictor passes no explicit mask; forward picks the buffer itself.
    S, X = _batch()
    fresh.eval()
    with torch.no_grad():
        _, attn_fwd, _ = fresh.forward(S, X)
    forbidden = gt_mask.unsqueeze(0).expand(attn_fwd.shape[0], -1, -1) == 0
    assert torch.all(attn_fwd[forbidden] == 0.0), "eval forward lost the mask"


def test_on_load_checkpoint_fills_missing_oracle_mask(data_dir):
    """Symmetric case: model has the buffer, checkpoint predates it -> fill."""
    cfg = _config(True, False)
    current = AttentionSelectorForecaster(cfg, data_dir=data_dir)
    ckpt_sd = {
        k: v for k, v in current.state_dict().items()
        if k != "oracle_combined_mask"
    }
    checkpoint = {"state_dict": ckpt_sd}

    current.on_load_checkpoint(checkpoint)
    assert "oracle_combined_mask" in checkpoint["state_dict"]
    assert torch.equal(
        checkpoint["state_dict"]["oracle_combined_mask"],
        current.oracle_combined_mask,
    )
    # Strict load must not raise on the formerly-missing key.
    current.load_state_dict(checkpoint["state_dict"], strict=True)
