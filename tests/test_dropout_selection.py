"""Tests for the pre-flight MLP-dropout selection (query-perturbation sensitivity).

Run with:  pytest tests/test_dropout_selection.py -v

Background
----------
``causaliT.utils.query_sensitivity.hsic_query_sensitivity`` measures how much
the train HSIC reacts to a small perturbation of the free query embeddings —
the probe that detects the "diluted" (flat-HSIC) regime.
``causaliT.training.dropout_selection.run_dropout_selection`` uses it as a
pre-flight stage of the adaptive trainer: one reconstruction-only warmup per
candidate dropout, argmax sensitivity wins, the main run is built with the
winning dropout and warm-started from the winner's weights.

Guarantees under test
---------------------
1. Sensitivity utility: finite, non-negative, deterministic given the seed,
   and the query weights are restored exactly after the call.
2. Guard: no free query embeddings -> ValueError.
3. ``_set_regressor_dropout`` (alias ``_set_mlp_dropout``): writes exactly the
   per-node value-embedding dropouts and the output head, leaves every other
   dropout untouched.
4. Selection loop (stubbed warmup + sensitivity): argmax picked, winner
   checkpoint + JSON report written, config override applied by the caller.
5. Selection guards: <2 candidates or free_query_embedding off -> (None, None).

Feature-index convention (mirrors tests/test_atsel_reg_safeguard.py):
    column 0 = variable ID, column 1 = value; ``val_idx = 1``.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.query_sensitivity import (
    hsic_query_sensitivity,
    scalar_hsic,
    _free_query_weights,
)
from causaliT.training import dropout_selection as ds


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_S = 8
VOCAB_X = 8
S_SEQ_LEN = 3
X_SEQ_LEN = 3
VALUE_COL = 1
VAR_COL = 0


def _embed_cfg(vocab: int, d_model: int = D_MODEL) -> dict:
    return {
        "setting": {"d_model": d_model},
        "modules": [
            {
                "idx": VAR_COL,
                "embed": "nn_embedding",
                "label": "variable",
                "kwargs": {"num_embeddings": vocab, "embedding_dim": d_model},
            },
            {
                "idx": VALUE_COL,
                "embed": "linear",
                "label": "value",
                "kwargs": {"input_dim": 1, "embedding_dim": d_model},
            },
        ],
    }


def _make_forecaster_config(free_query_embedding: bool = True) -> dict:
    """Minimal config dict accepted by AttentionSelectorForecaster.__init__.

    ``query_centroid_init`` must follow ``free_query_embedding``: the layer
    refuses centroid init when there is no free query table to initialise.
    """
    return {
        "data": {
            "val_idx": VALUE_COL,
            "S_seq_len": S_SEQ_LEN,
            "X_seq_len": X_SEQ_LEN,
            "dataset": "dummy",
        },
        "model": {
            "model_object": "AttentionSelectorLayer",
            "kwargs": {
                "model": "AttentionSelectorLayer",
                "ds_embed_S": _embed_cfg(VOCAB_S),
                "ds_embed_X": _embed_cfg(VOCAB_X),
                "comps_embed_S": "summation",
                "comps_embed_X": "summation",
                "attention_type": "CausalCrossAttention",
                "self_attention_type": "GatedSelfAttention",
                "free_query_embedding": free_query_embedding,
                "query_centroid_init": free_query_embedding,
                "n_heads": 1,
                "dropout_emb": 0.0,
                "dropout_attn_out": 0.0,
                "dropout_ff": 0.0,
                "dropout_qkv": 0.0,
                "attention_dropout": 0.0,
                "activation": "relu",
                "norm": "layer",
                "use_final_norm": False,
                "device": "cpu",
                "out_dim": 1,
                "d_ff": 32,
                "d_model": D_MODEL,
                "d_qk": D_MODEL,
                "S_seq_len": S_SEQ_LEN,
                "X_seq_len": X_SEQ_LEN,
                "remove_query_projection": False,
                "remove_key_projection": False,
            },
        },
        "training": {
            "loss_fn": "mse",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "use_gradient_routing": False,
            "lambda_recon": 1.0,
            "lambda_struct_recon": 0.0,
            "lambda_hsic": 1.0,
            "lambda_score_sparse": 0.0,
            "lambda_group_l1": 0.0,
            "lambda_l0": 0.0,
            "lambda_query_norm": 0.0,
            "kappa": 0.0,
            "hsic_sigma": 1.0,
            "hsic_adaptive_bandwidth": False,
            "hsic_mode": "biased",
            "nhsic_epsilon": 0.01,
            "hsic_kernel_source": "rbf",
            "use_oracle_attention": False,
            "use_hard_masks": False,
            "freeze_structural_params": False,
            "freeze_reconstruction_params": False,
        },
    }


def _make_batch(batch: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    S = torch.zeros(batch, S_SEQ_LEN, 2)
    S[:, :, VAR_COL] = torch.randint(1, VOCAB_S, (batch, S_SEQ_LEN), generator=g).float()
    S[:, :, VALUE_COL] = torch.randn(batch, S_SEQ_LEN, generator=g)
    X = torch.zeros(batch, X_SEQ_LEN, 2)
    X[:, :, VAR_COL] = torch.randint(1, VOCAB_X, (batch, X_SEQ_LEN), generator=g).float()
    X[:, :, VALUE_COL] = torch.randn(batch, X_SEQ_LEN, generator=g)
    return S, X


@pytest.fixture
def tmp_path():
    """Workspace-local temp dir (system temp root is not readable here)."""
    base = project_root / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Sensitivity utility
# ---------------------------------------------------------------------------

class TestQuerySensitivity:
    def test_finite_nonnegative_deterministic(self):
        torch.manual_seed(0)
        model = AttentionSelectorForecaster(_make_forecaster_config())
        batches = [_make_batch(seed=1), _make_batch(seed=2)]

        base1, sens1, deltas1 = hsic_query_sensitivity(
            model, batches, eps=1.0, n_pert=4, seed=42
        )
        base2, sens2, deltas2 = hsic_query_sensitivity(
            model, batches, eps=1.0, n_pert=4, seed=42
        )
        assert np.isfinite(base1)
        assert np.all(np.isfinite(sens1)) and np.all(sens1 >= 0.0)
        assert base1 == pytest.approx(base2)
        np.testing.assert_allclose(sens1, sens2)
        np.testing.assert_allclose(deltas1, deltas2)

    def test_weights_restored_after_call(self):
        torch.manual_seed(0)
        model = AttentionSelectorForecaster(_make_forecaster_config())
        before = [q.detach().clone() for q in _free_query_weights(model)]
        hsic_query_sensitivity(model, [_make_batch(seed=1)], eps=1.0, n_pert=3, seed=0)
        after = _free_query_weights(model)
        assert len(before) == len(after) and len(before) > 0
        for b, a in zip(before, after):
            assert torch.equal(b, a)

    def test_scalar_hsic_matches_step_layout(self):
        """scalar_hsic must equal the _step HSIC on the same batch (eval mode)."""
        torch.manual_seed(0)
        model = AttentionSelectorForecaster(_make_forecaster_config())
        batch = _make_batch(seed=3)
        model.eval()
        with torch.no_grad():
            model._step(batch, stage="val")
        ref = float(model._last_hsic_reg.detach())  # lambda_hsic == 1.0
        got = scalar_hsic(model, [batch])
        assert got == pytest.approx(ref, rel=1e-5)

    def test_guard_without_free_query_embedding(self):
        model = AttentionSelectorForecaster(
            _make_forecaster_config(free_query_embedding=False)
        )
        with pytest.raises(ValueError, match="free query"):
            hsic_query_sensitivity(model, [_make_batch(seed=1)], n_pert=1)


# ---------------------------------------------------------------------------
# 2. _set_mlp_dropout writes exactly the swept locations
# ---------------------------------------------------------------------------

class TestSetMlpDropout:
    def test_writes_swept_locations_only(self):
        config = {
            "model": {
                "kwargs": {
                    "ds_embed_S": {
                        "modules": [
                            {"embed": "mlp_per_node", "kwargs": {"dropout": 0.0}},
                            {"embed": "nn_embedding", "kwargs": {}},
                        ]
                    },
                    "ds_embed_X": {
                        "modules": [
                            {"embed": "mlp_per_node", "kwargs": {"dropout": 0.0}},
                        ]
                    },
                    "output_mlp_dropout": 0.0,
                    "dropout_emb": 0.0,
                    "attention_dropout": 0.0,
                }
            }
        }
        ds._set_mlp_dropout(config, 0.2)
        kw = config["model"]["kwargs"]
        assert kw["ds_embed_S"]["modules"][0]["kwargs"]["dropout"] == 0.2
        assert kw["ds_embed_X"]["modules"][0]["kwargs"]["dropout"] == 0.2
        assert kw["output_mlp_dropout"] == 0.2
        # Untouched: non-MLP module and all other dropout knobs.
        assert "dropout" not in kw["ds_embed_S"]["modules"][1]["kwargs"]
        assert kw["dropout_emb"] == 0.0
        assert kw["attention_dropout"] == 0.0


# ---------------------------------------------------------------------------
# 3. Selection loop (stubbed warmup + sensitivity)
# ---------------------------------------------------------------------------

class _FakeDM:
    """Minimal data module: a couple of (S, X) train batches, no phase API."""

    def train_dataloader(self):
        return [_make_batch(seed=11), _make_batch(seed=12)]


def _selection_config(tmp_path, candidates):
    cfg = _make_forecaster_config()
    cfg["adaptive_training"] = {
        "dropout_selection": {
            "enabled": True,
            "candidates": list(candidates),
            "warmup_epochs": 1,
            "n_pert": 2,
            "n_batches": 2,
            "eps": 1.0,
        }
    }
    return cfg


class TestSelectionLoop:
    def test_argmax_wins_and_artefacts_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ds, "_recon_warmup", lambda *a, **k: None)
        # Controlled sensitivities per candidate (iterated in order).
        sens_by_call = iter([1.0, 3.0, 2.0])  # candidates 0.0, 0.2, 0.4

        def _fake_sens(model, batches, eps, n_pert, seed):
            v = next(sens_by_call)
            return 0.0, np.array([v]), np.array([0.0])

        monkeypatch.setattr(ds, "hsic_query_sensitivity", _fake_sens)

        cfg = _selection_config(tmp_path, candidates=[0.0, 0.2, 0.4])
        best, ckpt = ds.run_dropout_selection(
            config=cfg, data_dir=None, dm=_FakeDM(),
            save_dir=str(tmp_path), cluster=True, seed=0,
        )
        assert best == pytest.approx(0.2)
        assert ckpt is not None and Path(ckpt).exists()
        # The winner checkpoint carries a loadable state_dict.
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "state_dict" in payload
        # Report JSON: per-candidate sensitivities + winner.
        report = json.loads((tmp_path / "dropout_selection.json").read_text())
        assert report["best_dropout"] == pytest.approx(0.2)
        assert report["results"]["0.2"]["sensitivity"] == pytest.approx(3.0)
        assert set(report["results"]) == {"0.0", "0.2", "0.4"}

    def test_fewer_than_two_candidates_skips(self, tmp_path):
        cfg = _selection_config(tmp_path, candidates=[0.2])
        best, ckpt = ds.run_dropout_selection(
            config=cfg, data_dir=None, dm=_FakeDM(),
            save_dir=str(tmp_path), cluster=True, seed=0,
        )
        assert best is None and ckpt is None

    def test_free_query_off_skips(self, tmp_path):
        cfg = _selection_config(tmp_path, candidates=[0.0, 0.2])
        cfg["model"]["kwargs"]["free_query_embedding"] = False
        best, ckpt = ds.run_dropout_selection(
            config=cfg, data_dir=None, dm=_FakeDM(),
            save_dir=str(tmp_path), cluster=True, seed=0,
        )
        assert best is None and ckpt is None


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
