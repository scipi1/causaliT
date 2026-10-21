"""Log-transformed structural objective (training.hsic_log_transform).

Covers the fixed-lambda path of ``AttentionSelectorForecaster._step``:

1. Default config is unchanged (raw lambda * HSIC term).
2. With the transform on, ``hsic_reg == lambda * log(clamp(HSIC) + floor)``
   for both floor modes ("null" from hsic_null_stats, fixed float) and the
   epsilon fallback; the step backprops with finite gradients.
3. Config validation: mutually exclusive with hsic_constraint.enabled;
   bad floor / epsilon values raise.
"""

import math

import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)

D, VOCAB, LS, LX = 16, 8, 3, 3
VAL, VAR = 1, 0


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


def _config(**overrides):
    training = {
        "loss_fn": "mse", "lr": 1e-3, "weight_decay": 0.0,
        "optimizer": "adamw", "use_gradient_routing": False,
        "lambda_recon": 1.0, "lambda_struct_recon": 0.0,
        "lambda_hsic": 1.0, "lambda_score_sparse": 0.0,
        "lambda_group_l1": 0.0, "lambda_l0": 0.0, "kappa": 0.0,
        "lambda_query_norm": 0.0,
        "hsic_objective": "pairwise",
        "hsic_sigma": 1.0, "hsic_adaptive_bandwidth": True,
        "hsic_mode": "biased", "nhsic_epsilon": 0.01,
        "hsic_kernel_source": "rbf",
        "freeze_structural_params": False,
        "freeze_reconstruction_params": False,
    }
    training.update(overrides)
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
                "attention_type": "ScaledDotSoftmax",
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
        "training": training,
    }


def _batch(batch_size: int = 16):
    g = torch.Generator().manual_seed(0)
    S = torch.zeros(batch_size, LS, 2)
    S[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LS), generator=g).float()
    S[:, :, VAL] = torch.randn(batch_size, LS, generator=g)
    X = torch.zeros(batch_size, LX, 2)
    X[:, :, VAR] = torch.randint(1, VOCAB, (batch_size, LX), generator=g).float()
    X[:, :, VAL] = torch.randn(batch_size, LX, generator=g)
    return S, X


def _raw_hsic_reg(**overrides):
    """Reference hsic_reg from the UNTRANSFORMED path on identical weights."""
    torch.manual_seed(0)
    fc = AttentionSelectorForecaster(_config(**overrides))
    fc.train()
    fc.training_step(_batch(), 0)
    return float(fc._last_hsic_reg.detach())


class TestFloorSelection:
    def test_fixed_floor(self):
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor=0.25)
        )
        assert fc.hsic_log_floor_mode == "fixed"
        assert fc._hsic_log_floor() == pytest.approx(0.25)

    def test_null_floor_uses_null_stats(self):
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor="null")
        )
        fc.hsic_null_stats = {"mean": 0.5, "std": 0.1, "n": 100}
        assert fc._hsic_log_floor() == pytest.approx(0.5)

    def test_null_floor_falls_back_to_epsilon(self):
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor="null",
                    hsic_log_epsilon=1e-6)
        )
        fc.hsic_null_stats = None
        assert fc._hsic_log_floor() == pytest.approx(1e-6)
        assert fc._hsic_log_floor_warned

    def test_null_floor_rejects_bad_stats(self):
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor="null",
                    hsic_log_epsilon=1e-6)
        )
        fc.hsic_null_stats = {"mean": float("nan")}
        assert fc._hsic_log_floor() == pytest.approx(1e-6)


class TestLossTransform:
    def test_default_off_matches_raw_term(self):
        fc = AttentionSelectorForecaster(_config())
        assert not fc.hsic_log_transform
        fc.train()
        fc.training_step(_batch(), 0)
        # Raw path: hsic_reg == lambda_hsic * HSIC, non-negative.
        assert float(fc._last_hsic_reg) >= 0.0

    def test_log_value_matches_closed_form(self):
        lam = 1.0
        floor = 0.25
        raw = _raw_hsic_reg()  # = lam * HSIC on identical weights/batch
        torch.manual_seed(0)
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor=floor)
        )
        fc.train()
        fc.training_step(_batch(), 0)
        got = float(fc._last_hsic_reg.detach())
        expected = lam * math.log(raw / lam + floor)
        assert got == pytest.approx(expected, rel=1e-5)

    def test_null_floor_value_in_loss(self):
        raw = _raw_hsic_reg()
        null_mean = 0.5
        torch.manual_seed(0)
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor="null")
        )
        fc.hsic_null_stats = {"mean": null_mean, "std": 0.1, "n": 100}
        fc.train()
        fc.training_step(_batch(), 0)
        got = float(fc._last_hsic_reg.detach())
        assert got == pytest.approx(math.log(raw + null_mean), rel=1e-5)

    def test_log_step_backprops(self):
        torch.manual_seed(0)
        fc = AttentionSelectorForecaster(
            _config(hsic_log_transform=True, hsic_log_floor=0.25)
        )
        fc.train()
        out = fc.training_step(_batch(), 0)
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in fc.parameters()
                 if p.requires_grad and p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)


class TestValidation:
    def test_constraint_conflict_raises(self):
        cfg = _config(hsic_log_transform=True, lambda_hsic=0.0)
        cfg["training"]["hsic_constraint"] = {"enabled": True}
        with pytest.raises(ValueError, match="hsic_log_transform"):
            AttentionSelectorForecaster(cfg)

    def test_bad_floor_string_raises(self):
        with pytest.raises(ValueError, match="hsic_log_floor"):
            AttentionSelectorForecaster(
                _config(hsic_log_transform=True, hsic_log_floor="bogus")
            )

    def test_nonpositive_fixed_floor_raises(self):
        with pytest.raises(ValueError, match="hsic_log_floor"):
            AttentionSelectorForecaster(
                _config(hsic_log_transform=True, hsic_log_floor=0.0)
            )

    def test_nonpositive_epsilon_raises(self):
        with pytest.raises(ValueError, match="hsic_log_epsilon"):
            AttentionSelectorForecaster(
                _config(hsic_log_transform=True, hsic_log_epsilon=0.0)
            )
