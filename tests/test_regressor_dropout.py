"""Tests for the global regressor-dropout knob (expressivity control).

Run with:  pytest tests/test_regressor_dropout.py -v

Guarantees under test
---------------------
1. ``linear_per_node_emb``: optional dropout on the embedded value output;
   default 0.0 is a no-op (nn.Identity), train mode stochastic, eval mode
   deterministic, output shape unchanged.
2. ``PerNodeMLPHead`` FiLM conditioner: shares the head's ``dropout`` knob
   (Linear -> GELU -> Dropout -> Linear), identity-at-init preserved.
3. ``_set_regressor_dropout``: writes every dropout-capable regressor
   location (mlp_per_node AND linear_per_node value embeddings + output
   head), returns the written config paths, leaves structural knobs
   untouched.
"""

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.embedding_layers import linear_per_node_emb
from causaliT.core.modules.mlp_head import PerNodeMLPHead
from causaliT.training import dropout_selection as ds

D_MODEL = 16
X_SEQ_LEN = 4
BATCH = 3


def _make_linear_per_node(dropout=0.0):
    return linear_per_node_emb(
        input_dim=1,
        embedding_dim=D_MODEL,
        num_variables=X_SEQ_LEN,
        device="cpu",
        dropout=dropout,
    )


def _inputs():
    values = torch.randn(BATCH, X_SEQ_LEN)
    var_ids = torch.arange(1, X_SEQ_LEN + 1).float().unsqueeze(0).expand(BATCH, -1)
    return values, var_ids


class TestLinearPerNodeDropout:
    def test_default_disabled_is_noop(self):
        torch.manual_seed(0)
        emb = _make_linear_per_node()
        assert isinstance(emb.dropout, torch.nn.Identity)
        values, var_ids = _inputs()
        out = emb(values, var_ids)
        assert out.shape == (BATCH, X_SEQ_LEN, D_MODEL)

    def test_train_stochastic_eval_deterministic(self):
        torch.manual_seed(0)
        emb = _make_linear_per_node(dropout=0.5)
        torch.manual_seed(0)
        emb_ref = _make_linear_per_node(dropout=0.0)  # same weights
        assert isinstance(emb.dropout, torch.nn.Dropout)
        assert emb.dropout.p == 0.5
        values, var_ids = _inputs()

        emb.train()
        torch.manual_seed(0)
        out1 = emb(values, var_ids)
        torch.manual_seed(1)
        out2 = emb(values, var_ids)
        assert not torch.allclose(out1, out2)

        emb.eval()
        torch.manual_seed(0)
        out3 = emb(values, var_ids)
        torch.manual_seed(1)
        out4 = emb(values, var_ids)
        assert torch.allclose(out3, out4)
        # Eval output equals the undropped linear map.
        assert torch.allclose(out3, emb_ref(values, var_ids))


class TestFilmConditionerDropout:
    def _head(self, dropout):
        return PerNodeMLPHead(
            d_model=D_MODEL,
            out_dim=1,
            num_variables=X_SEQ_LEN,
            d_hidden=8,
            dropout=dropout,
            film_context_dim=X_SEQ_LEN,
        )

    def test_conditioner_shares_head_dropout(self):
        head = self._head(dropout=0.5)
        # Layout: Linear -> GELU -> Dropout -> Linear.
        assert len(head.film) == 4
        assert isinstance(head.film[2], torch.nn.Dropout)
        assert head.film[2].p == 0.5

    def test_conditioner_default_no_dropout(self):
        head = self._head(dropout=0.0)
        assert isinstance(head.film[2], torch.nn.Identity)

    def test_identity_at_init_preserved(self):
        torch.manual_seed(0)
        head = self._head(dropout=0.5)
        head.eval()
        x = torch.randn(BATCH, X_SEQ_LEN, D_MODEL)
        ctx = torch.randn(BATCH, X_SEQ_LEN, X_SEQ_LEN)
        var_ids = torch.arange(1, X_SEQ_LEN + 1).unsqueeze(0).expand(BATCH, -1)
        out_ctx = head(x, var_ids, context=ctx)
        # gamma = 1, beta = 0 at init -> identical to the unconditioned head.
        torch.manual_seed(0)
        head_plain = PerNodeMLPHead(
            d_model=D_MODEL, out_dim=1, num_variables=X_SEQ_LEN,
            d_hidden=8, dropout=0.5,
        )
        head_plain.eval()
        assert torch.allclose(out_ctx, head_plain(x, var_ids))


class TestSetRegressorDropout:
    def _config(self, value_embed):
        return {
            "model": {
                "kwargs": {
                    "ds_embed_S": {
                        "modules": [
                            {"embed": value_embed, "kwargs": {"dropout": 0.0}},
                            {"embed": "nn_embedding", "kwargs": {}},
                        ]
                    },
                    "ds_embed_X": {
                        "modules": [
                            {"embed": value_embed, "kwargs": {}},  # no dropout key
                        ]
                    },
                    "output_mlp_dropout": 0.0,
                    "dropout_emb": 0.0,
                    "attention_dropout": 0.0,
                }
            }
        }

    @pytest.mark.parametrize("value_embed", ["mlp_per_node", "linear_per_node"])
    def test_writes_all_regressor_locations(self, value_embed):
        config = self._config(value_embed)
        written = ds._set_regressor_dropout(config, 0.2)
        kw = config["model"]["kwargs"]
        assert kw["ds_embed_S"]["modules"][0]["kwargs"]["dropout"] == 0.2
        # Missing kwargs/dropout keys are created.
        assert kw["ds_embed_X"]["modules"][0]["kwargs"]["dropout"] == 0.2
        assert kw["output_mlp_dropout"] == 0.2
        # Untouched: non-regressor module and all other dropout knobs.
        assert "dropout" not in kw["ds_embed_S"]["modules"][1]["kwargs"]
        assert kw["dropout_emb"] == 0.0
        assert kw["attention_dropout"] == 0.0
        # Return value lists the written paths.
        assert any("ds_embed_S" in p for p in written)
        assert any("ds_embed_X" in p for p in written)
        assert "model.kwargs.output_mlp_dropout" in written

    def test_backward_compatible_alias(self):
        config = self._config("linear_per_node")
        ds._set_mlp_dropout(config, 0.3)
        assert config["model"]["kwargs"]["output_mlp_dropout"] == 0.3


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])

