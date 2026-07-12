"""
End-to-end smoke test for the L0<->HSIC gradient-interference diagnostic on
the REAL HardConcreteCrossAttention AttentionSelector model.

Validates:
  1. build_interference_blocks() on the real model populates the expected
     structural blocks (query_projection, key_projection, embedding_*).
  2. l0_penalty from forward_with_actual is graph-attached (requires_grad),
     so autograd.grad on l0_reg produces real (non-None) gradients.
  3. compute_l0_hsic_interference() returns finite per-block cosines for the
     Q/K blocks (the shared structural pathway) — proving both objectives
     reach them.
  4. The probe is non-invasive (.grad stays None) AND a real backward on the
     total loss still works afterwards (graph retained).

Run:  python scripts/_smoke_interference.py
"""

import math

import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.training.interference_utils import (
    build_interference_blocks,
    compute_l0_hsic_interference,
)
from causaliT.utils.hsic_utils import hsic_cross_per_pair


# ---------------------------------------------------------------------------
# Config (concrete values, no OmegaConf interpolation)
# ---------------------------------------------------------------------------
D_MODEL = 16
D_FF = 32
D_QK = 8
N_HEADS = 2
S_LEN = 3
X_LEN = 4
NUM_EMB_S = 6
NUM_EMB_X = 8
VAL_IDX = 1
VAR_IDX = 0
BATCH = 64

LAMBDA_HSIC = 1.0
LAMBDA_L0 = 1.0e-3


def _embed_cfg(num_emb: int) -> dict:
    return {
        "setting": {"d_model": D_MODEL, "sparse_grad": False},
        "modules": [
            {
                "idx": VAL_IDX,
                "embed": "linear",
                "label": "value",
                "role": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": D_MODEL},
            },
            {
                "idx": VAR_IDX,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {
                    "num_embeddings": num_emb,
                    "embedding_dim": D_MODEL,
                    "padding_idx": 0,
                    "sparse": False,
                    "max_norm": 1,
                },
            },
            {
                "idx": VAL_IDX,
                "embed": "mask",
                "label": "value_missing",
                "kwargs": {},
            },
        ],
    }


def _build_config() -> dict:
    model_kwargs = {
        "model": "AttentionSelectorLayer",
        "ds_embed_S": _embed_cfg(NUM_EMB_S),
        "ds_embed_X": _embed_cfg(NUM_EMB_X),
        "comps_embed_S": "svfa",
        "comps_embed_X": "svfa",
        "attention_type": "HardConcreteCrossAttention",
        "n_heads": N_HEADS,
        "init_tau": 0.5,
        "shared_dag_across_heads": True,
        "dropout_emb": 0.0,
        "dropout_attn_out": 0.0,
        "dropout_ff": 0.0,
        "dropout_qkv": 0.0,
        "attention_dropout": 0.0,
        "activation": "gelu",
        "norm": "layer",
        "use_final_norm": True,
        "device": "cpu",
        "out_dim": 1,
        "d_ff": D_FF,
        "d_model": D_MODEL,
        "d_qk": D_QK,
        "S_seq_len": S_LEN,
        "X_seq_len": X_LEN,
        "output_mlp_layers": 1,
        "output_mlp_hidden": D_FF,
        "output_mlp_activation": "relu",
        "output_mlp_dropout": 0.0,
        "struct_embedding_type": "standard_learnable",
    }
    return {
        "model": {"model_object": "AttentionSelectorLayer", "kwargs": model_kwargs},
        "training": {
            "loss_fn": "mse",
            "lambda_recon": 1.0,
            "lambda_hsic": LAMBDA_HSIC,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": True,
            "hsic_kernel_source": "rbf",
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "lambda_group_l1": 0.0,
            "lambda_l0": LAMBDA_L0,
            "lambda_score_sparse": 0.0,
            "kappa": 0.0,
            "use_gradient_routing": False,  # simpler for the smoke probe
            "log_l0_hsic_interference": True,
            "interference_log_every_n_epochs": 1,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
        },
        "data": {
            "dataset": "smoke",
            "val_idx": VAL_IDX,
            "S_seq_len": S_LEN,
            "X_seq_len": X_LEN,
            "feature_indices": {"value": VAL_IDX, "variable": VAR_IDX},
        },
    }


def _make_batch():
    S = torch.zeros(BATCH, S_LEN, 2)
    S[:, :, VAR_IDX] = torch.randint(1, NUM_EMB_S, (BATCH, S_LEN)).float()
    S[:, :, VAL_IDX] = torch.randn(BATCH, S_LEN)

    X = torch.zeros(BATCH, X_LEN, 2)
    X[:, :, VAR_IDX] = torch.randint(1, NUM_EMB_X, (BATCH, X_LEN)).float()
    X[:, :, VAL_IDX] = torch.randn(BATCH, X_LEN)
    return S, X


def main():
    torch.manual_seed(0)
    cfg = _build_config()
    fc = AttentionSelectorForecaster(cfg)
    fc.train()

    model = fc.model

    # ---- Blocks on the REAL model --------------------------------------
    blocks = build_interference_blocks(model)
    print("=== Structural blocks discovered on real HardConcrete model ===")
    for name, plist in blocks.items():
        nparams = sum(p.numel() for p in plist)
        print(f"  {name:26s}: {len(plist)} tensors, {nparams} params")
    assert "query_projection" in blocks, "query_projection block missing!"
    assert "key_projection" in blocks, "key_projection block missing!"

    # ---- Forward + reg terms (mirrors _step, but no self.log) ----------
    S, X = _make_batch()
    x_val = X[:, :, VAL_IDX]
    x_blanked = X.clone()
    x_blanked[:, :, VAL_IDX] = 0.0

    pred_x, attn, aux = model.forward_with_actual(
        source_tensor=S, x_blanked=x_blanked, x_actual=X
    )
    l0_penalty = aux["l0_penalty"]
    print("\n=== l0_penalty ===")
    print(f"  value={l0_penalty.detach().item():.4f}  requires_grad={l0_penalty.requires_grad}")
    assert l0_penalty.requires_grad, "l0_penalty is NOT graph-attached!"

    x_target = torch.nan_to_num(x_val)
    residuals = x_target.squeeze() - pred_x.squeeze()
    s_values = S[:, :, VAL_IDX]
    combined_source = torch.cat([s_values, x_target.squeeze()], dim=1)

    hsic_value = hsic_cross_per_pair(
        combined_source, residuals,
        sigma=1.0, adaptive_bandwidth=True, mode="biased",
        nhsic_epsilon=0.01, source_kernel="rbf",
    )
    hsic_reg = LAMBDA_HSIC * hsic_value
    l0_reg = LAMBDA_L0 * l0_penalty

    # ---- Interference cosines ------------------------------------------
    cos = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)
    print("\n=== Per-block L0<->HSIC gradient cosine similarity ===")
    for name, c in cos.items():
        print(f"  train_interf_cos_{name:26s} = {c:+.4f}")

    # Q/K must have finite cosines (both objectives reach them).
    for key in ("query_projection", "key_projection", "overall"):
        assert key in cos, f"{key} missing from results"
        assert not math.isnan(cos[key]), f"{key} cosine is NaN (no shared gradient?)"

    # ---- Non-invasive check --------------------------------------------
    for p in model.parameters():
        assert p.grad is None, "Probe wrote into .grad (should be non-invasive)!"

    # ---- Retained graph allows a real backward -------------------------
    total = (pred_x.squeeze() - x_target.squeeze()).pow(2).mean() + hsic_reg + l0_reg
    total.backward()
    qk_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in blocks["query_projection"]
    )
    assert qk_has_grad, "query_projection got no gradient from the real backward!"

    print("\nALL SMOKE CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
