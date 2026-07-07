"""
Tests for the Optuna capacity-search constant-score protocol
(``optuna_protocol``) on ``CausalCrossAttention`` / ``AttentionLayer``.

The protocol REPLACES the learned QK^T attention weights with a constant on
every allowed edge, in BOTH train and eval, and BEFORE the hard mask so the
diagonal self-loop constraint still holds:

    optuna_protocol=0.0 → all allowed edges get weight 0 (residual-only floor)
    optuna_protocol=1.0 → all allowed edges get weight 1 (uniform mixing)
    optuna_protocol=None → normal learned ReLU(Tanh) behaviour (default)
"""

import os
import tempfile

import pytest
import torch


from causaliT.core.modules.attention import AttentionLayer, CausalCrossAttention
from causaliT.euler_optuna.euler_optuna.cli import (
    OPTUNA_PROTOCOL_EXTENSIONS,
    load_protocol_extension,
)



def _make_layer(optuna_protocol):
    return AttentionLayer(
        attention=CausalCrossAttention,
        d_model_queries=8,
        d_model_keys=8,
        d_model_values=8,
        d_queries_keys=8,
        n_heads=1,
        mask_layer=None,
        attention_dropout=0.0,
        dropout_qkv=0.0,
        optuna_protocol=optuna_protocol,
    )


def _inputs(B=4, L=3, S=5, d=8):
    torch.manual_seed(0)
    q = torch.randn(B, L, d)
    k = torch.randn(B, S, d)
    v = torch.randn(B, S, d)
    return q, k, v


def test_protocol_zero_gives_zero_attention():
    """optuna_protocol=0.0 → every attention weight is exactly 0 → V output 0."""
    layer = _make_layer(0.0)
    layer.eval()
    q, k, v = _inputs()

    out, attn, _aux = layer(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
    )

    assert torch.allclose(attn, torch.zeros_like(attn)), "all edges must be 0"
    assert torch.allclose(out, torch.zeros_like(out)), "value output must be 0"


def test_protocol_one_gives_uniform_attention_respecting_mask():
    """optuna_protocol=1.0 → allowed edges are 1, masked edges stay 0."""
    layer = _make_layer(1.0)
    layer.eval()
    q, k, v = _inputs()
    B, L, _ = q.shape
    S = k.shape[1]

    # Mask out the last key column for every query.
    hard_mask = torch.ones(L, S)
    hard_mask[:, -1] = 0.0

    out, attn, _aux = layer(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
        hard_mask=hard_mask,
    )

    # Allowed edges == 1, masked column == 0.
    assert torch.allclose(attn[..., :-1], torch.ones_like(attn[..., :-1]))
    assert torch.allclose(attn[..., -1], torch.zeros_like(attn[..., -1]))


def test_protocol_is_deterministic_in_train_mode():
    """The constant-score override applies in train mode too (not just eval)."""
    layer = _make_layer(1.0)
    layer.train()
    q, k, v = _inputs()

    _o1, a1, _ = layer(query=q, key=k, value=v, mask_miss_k=None,
                       mask_miss_q=None, pos=None, causal_mask=False)
    _o2, a2, _ = layer(query=q, key=k, value=v, mask_miss_k=None,
                       mask_miss_q=None, pos=None, causal_mask=False)

    assert torch.allclose(a1, torch.ones_like(a1))
    assert torch.allclose(a1, a2), "override must be deterministic across passes"


def test_protocol_none_is_normal_behaviour():
    """optuna_protocol=None → weights are the learned ReLU(Tanh), not constant."""
    layer = _make_layer(None)
    layer.eval()
    q, k, v = _inputs()

    _out, attn, _aux = layer(
        query=q, key=k, value=v,
        mask_miss_k=None, mask_miss_q=None, pos=None, causal_mask=False,
    )

    # Not all zeros and not all ones (learned, data-dependent).
    assert not torch.allclose(attn, torch.zeros_like(attn))
    assert not torch.allclose(attn, torch.ones_like(attn))


# =============================================================================
# Config-driven protocol resolution (load_protocol_extension)
# =============================================================================

def _settings_dir(body=None):
    """Create a fresh temp dir, optionally writing an optuna_settings.yaml."""
    d = tempfile.mkdtemp(prefix="causalit_protocol_test_")
    if body is not None:
        with open(os.path.join(d, "optuna_settings.yaml"), "w") as f:
            f.write(body)
    return d


def test_load_protocol_absent_key_defaults_to_none():
    """A settings file without a `protocol:` key resolves to None (base only)."""
    d = _settings_dir("n_trials: 10\ndirection: minimize\n")
    ext, name = load_protocol_extension(d)
    assert name == "none"
    assert ext is None


def test_load_protocol_missing_file_defaults_to_none():
    """No optuna*.yaml at all → default to none."""
    d = _settings_dir(None)
    ext, name = load_protocol_extension(d)
    assert name == "none"
    assert ext is None


def test_load_protocol_valid_name():
    """A valid `protocol:` name resolves to the matching extension dict."""
    d = _settings_dir('protocol: "regime_uniform_v1"\n')
    ext, name = load_protocol_extension(d)
    assert name == "regime_uniform_v1"
    assert ext is OPTUNA_PROTOCOL_EXTENSIONS["regime_uniform_v1"]
    assert ext["overrides"]["model.kwargs.optuna_protocol"] == 1.0


def test_load_protocol_invalid_name_raises():
    """An unknown `protocol:` name raises a clear ValueError."""
    d = _settings_dir('protocol: "does_not_exist"\n')
    with pytest.raises(ValueError, match="Unknown protocol"):
        load_protocol_extension(d)



