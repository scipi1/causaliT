"""
The ATE arms share a tuned nonlinear output head; only the method arm embeds
values nonlinearly.

Run with:  pytest tests/test_ate_arms_mlp_config.py -v

Design pinned here (all paths relative to experiments/7_PUBLISH/ATE):

1. Every arm's Optuna search space tunes ``experiment.output_mlp_layers`` as
   an int inside [2, 4]: head depth is part of the capacity search and is
   priced into the parsimonious knee via the real parameter count.
2. Every arm's head hidden width is ``${experiment.d_ff}`` with
   ``d_ff_mult: 2.0``, i.e. exactly 2 * d_model.
3. ``svfa`` embeds the value stream with ``embed: mlp`` (1 hidden layer,
   hidden = 2 * d_model via ``${experiment.d_ff}``); ``vanilla`` / ``cheater``
   keep the linear value embedding (a vanilla transformer's token embedding
   is linear).
4. The svfa value-module config builds an ``mlp_emb`` whose hidden width is
   2 * d_model (offline construction check).
"""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.core.modules.embedding import ModularEmbedding
from causaliT.core.modules.embedding_layers import mlp_emb

ATE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "7_PUBLISH" / "ATE"
ARMS = ["svfa", "vanilla", "cheater"]
STREAMS = ["ds_embed_S", "ds_embed_X"]

D_MODEL = 16


def _load(arm, filename):
    """Load an arm file as plain dicts, keeping ``${...}`` refs unresolved.

    The wiring itself (not the derived value) is what these tests pin, so
    interpolations must stay raw strings.
    """
    path = ATE_DIR / arm / filename
    if not path.exists():
        pytest.skip(f"arm {arm} not present")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def _value_modules(cfg, stream):
    """Embedding modules of a stream with role 'value' (the 'mask' module has none)."""
    modules = cfg["model"]["kwargs"][stream]["modules"]
    return [m for m in modules if m.get("role") == "value"]


# ---------------------------------------------------------------------------
# 1. Head depth is tuned in [2, 4]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_search_space_tunes_output_mlp_layers(arm):
    settings = _load(arm, "optuna_settings.yaml")
    entry = (settings.get("search_space") or {}).get("experiment.output_mlp_layers")
    assert entry is not None, f"{arm}: output_mlp_layers missing from the search space"
    assert entry["type"] == "int"
    assert int(entry["low"]) >= 2 and int(entry["high"]) <= 4


# ---------------------------------------------------------------------------
# 2. Head hidden width = 2 * d_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ARMS)
def test_output_head_hidden_is_two_d_model(arm):
    cfg = _load(arm, "config_atsel.yaml")
    assert float(cfg["experiment"]["d_ff_mult"]) == 2.0
    assert cfg["model"]["kwargs"]["output_mlp_hidden"] == "${experiment.d_ff}"
    # The fallback depth stays inside the tuned range even before Optuna overrides it.
    assert 2 <= int(cfg["experiment"]["output_mlp_layers"]) <= 4


# ---------------------------------------------------------------------------
# 3. Value embedding: mlp for svfa, linear for the baselines
# ---------------------------------------------------------------------------

def test_svfa_value_embedding_is_mlp():
    cfg = _load("svfa", "config_atsel.yaml")
    for stream in STREAMS:
        values = _value_modules(cfg, stream)
        assert len(values) == 1, f"{stream}: exactly one value-role module"
        assert values[0]["embed"] == "mlp"
        assert values[0]["kwargs"]["hidden_dim"] == "${experiment.d_ff}"
        assert values[0]["kwargs"]["embedding_dim"] == "${model.embed_dim.val_emb_hidden}"


@pytest.mark.parametrize("arm", ["vanilla", "cheater"])
def test_baselines_keep_linear_value_embedding(arm):
    cfg = _load(arm, "config_atsel.yaml")
    for stream in STREAMS:
        values = _value_modules(cfg, stream)
        assert len(values) == 1, f"{stream}: exactly one value-role module"
        assert values[0]["embed"] == "linear"
        assert "hidden_dim" not in values[0]["kwargs"]


# ---------------------------------------------------------------------------
# 4. The mlp value module builds with hidden = 2 * d_model
# ---------------------------------------------------------------------------

def test_svfa_value_mlp_builds_with_two_d_model_hidden():
    cfg = {
        "setting": {"d_model": D_MODEL},
        "modules": [
            {
                "idx": 0,
                "embed": "mlp",
                "label": "value",
                "role": "value",
                "kwargs": {
                    "input_dim": 1,
                    "embedding_dim": D_MODEL,
                    "hidden_dim": 2 * D_MODEL,
                },
            },
            {
                "idx": 1,
                "embed": "nn_embedding",
                "label": "variable",
                "role": "structure",
                "kwargs": {"num_embeddings": 5, "embedding_dim": D_MODEL},
            },
        ],
    }
    emb = ModularEmbedding(cfg, comps="svfa", device="cpu")

    value_maps = [m.embedding for m in emb.value_modules_list]
    assert len(value_maps) == 1 and isinstance(value_maps[0], mlp_emb)
    layers = [m for m in value_maps[0].embedding if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 2, "the value MLP keeps exactly one hidden layer"
    assert layers[0].out_features == 2 * D_MODEL
    assert layers[1].in_features == 2 * D_MODEL
    assert layers[1].out_features == D_MODEL

    # Forward pass in the production (value, variable-ID) layout.
    x = torch.zeros(2, 3, 2)
    x[:, :, 0] = torch.randn(2, 3)
    x[:, :, 1] = torch.arange(1, 4).float().unsqueeze(0).expand(2, -1)
    emb_struct, emb_val = emb(x)
    assert emb_struct.shape == (2, 3, D_MODEL)
    assert emb_val.shape == (2, 3, D_MODEL)
