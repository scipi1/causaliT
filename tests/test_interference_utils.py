"""
Unit tests for causaliT.training.interference_utils.

Covers:
  * build_interference_blocks selects ONLY structural params (Q/K,
    structural embeddings, ...) and excludes reconstruction params.
  * compute_l0_hsic_interference returns the correct cosine similarity
    for hand-constructed gradients (orthogonal, aligned, anti-aligned).
  * A block that receives an L0 gradient of zero yields NaN (clear
    "no interference signal" marker) rather than a misleading 0.
  * The probe is non-invasive: it never populates ``.grad`` (so the real
    backward that follows in training is unaffected).
"""

import math

import torch
import torch.nn as nn

from causaliT.training.interference_utils import (
    build_interference_blocks,
    compute_l0_hsic_interference,
)


class _TinyModel(nn.Module):
    """Mimics the AttentionSelector structural/reconstruction naming."""

    def __init__(self):
        super().__init__()
        # Structural params (match STRUCTURAL_PATTERNS)
        self.query_projection = nn.Linear(2, 2, bias=False)
        self.key_projection = nn.Linear(2, 2, bias=False)
        # Structural embedding: name must contain "structure_modules"
        self.embedding_S = nn.Module()
        self.embedding_S.structure_modules = nn.Linear(2, 2, bias=False)
        # Reconstruction param (must be EXCLUDED from blocks)
        self.value_projection = nn.Linear(2, 2, bias=False)


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1)


def test_blocks_group_params_by_module():
    """All trainable params are grouped by readable module label.

    Q/K and structural embeddings get their own blocks; the reconstruction
    value projection gets its OWN 'value_projection' block (kept separate from
    the structural blocks so it never dilutes the Q/K cosine).  At compute time
    it will simply return NaN because it receives no L0 gradient.
    """
    model = _TinyModel()
    blocks = build_interference_blocks(model)

    assert "query_projection" in blocks
    assert "key_projection" in blocks
    assert "embedding_S" in blocks
    # value_projection is its own block, NOT mixed into a structural block.
    assert "value_projection" in blocks
    assert id(model.value_projection.weight) in {
        id(p) for p in blocks["value_projection"]
    }
    assert id(model.value_projection.weight) not in {
        id(p) for p in blocks["query_projection"]
    }


def test_cosine_orthogonal_aligned_antialigned():
    torch.manual_seed(0)
    model = _TinyModel()
    blocks = build_interference_blocks(model)

    Wq = model.query_projection.weight  # (2,2)

    # Fixed coefficient matrices; grad of (W*A).sum() wrt W is exactly A.
    A = torch.tensor([[1.0, 0.0], [0.0, 0.0]])   # -> grad direction [1,0,0,0]
    B = torch.tensor([[0.0, 1.0], [0.0, 0.0]])   # -> grad direction [0,1,0,0]

    # Orthogonal
    hsic_reg = (Wq * A).sum()
    l0_reg = (Wq * B).sum()
    res = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)
    assert math.isclose(res["query_projection"], 0.0, abs_tol=1e-6)

    # Aligned: same direction -> cosine +1
    hsic_reg = (Wq * A).sum()
    l0_reg = (Wq * A).sum()
    res = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)
    assert math.isclose(res["query_projection"], 1.0, abs_tol=1e-6)

    # Anti-aligned: opposite direction -> cosine -1 (max interference)
    hsic_reg = (Wq * A).sum()
    l0_reg = -(Wq * A).sum()
    res = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)
    assert math.isclose(res["query_projection"], -1.0, abs_tol=1e-6)


def test_block_with_zero_l0_gradient_is_nan():
    """embedding_S gets an HSIC gradient but no L0 gradient -> NaN cosine."""
    model = _TinyModel()
    blocks = build_interference_blocks(model)

    Wq = model.query_projection.weight
    We = model.embedding_S.structure_modules.weight

    # HSIC depends on Q and embedding; L0 depends ONLY on Q.
    hsic_reg = (Wq * 1.0).sum() + (We * 1.0).sum()
    l0_reg = (Wq * 1.0).sum()

    res = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)
    # embedding_S has zero L0 gradient -> NaN
    assert math.isnan(res["embedding_S"])
    # query_projection is aligned (+1)
    assert math.isclose(res["query_projection"], 1.0, abs_tol=1e-6)
    # overall is well-defined (Q contributes to both)
    assert not math.isnan(res["overall"])


def test_probe_is_non_invasive():
    """compute_l0_hsic_interference must not populate .grad."""
    model = _TinyModel()
    blocks = build_interference_blocks(model)
    Wq = model.query_projection.weight

    hsic_reg = (Wq * 1.0).sum()
    l0_reg = (Wq * 2.0).sum()

    # Ensure grads start clean
    for p in model.parameters():
        assert p.grad is None

    _ = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)

    # autograd.grad must NOT have written into .grad
    for p in model.parameters():
        assert p.grad is None


def test_retain_graph_allows_subsequent_backward():
    """After probing, a real backward on the same graph must still work."""
    model = _TinyModel()
    blocks = build_interference_blocks(model)
    Wq = model.query_projection.weight

    hsic_reg = (Wq * 1.0).sum()
    l0_reg = (Wq * 2.0).sum()
    total = hsic_reg + l0_reg

    _ = compute_l0_hsic_interference(model, hsic_reg, l0_reg, blocks)

    # Graph retained -> this backward should not raise.
    total.backward()
    assert Wq.grad is not None
