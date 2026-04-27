"""
Unit tests for `preserve_sparsity` mode in `causaliT.core.utils.corrupt_dag_masks`.

These tests pin down the contract documented in the function's docstring:

- preserve_sparsity=True keeps the corrupted mask's edge count exactly equal
  to the ground truth.
- Only even SHD values are achievable; odd requests are rounded DOWN.
- Realised SHD is `2 * n_swaps` and `n_swaps == shd_request // 2` whenever
  the request fits within `min(k_true, n_non_edges)` of the eligible pool.
- When the request exceeds that capacity, `fallback_used=True` is reported
  and the realised SHD reflects the cap (still even, still edge-preserving).
- Diagonal of square self-masks is never touched (no self-loops).
- The legacy mode (preserve_sparsity=False) is bit-identical to the
  pre-existing behavior — same output for the same seed.
"""

import numpy as np
import pytest
import torch

from causaliT.core.utils import corrupt_dag_masks


# -------------------------------------------------------------------
# Fixtures / helpers
# -------------------------------------------------------------------

def _gt_masks(X_len: int = 6, S_len: int = 4, seed: int = 0):
    """
    Build a small reproducible GT mask dict with one cross and one self mask.

    The self mask is strictly lower-triangular so it is acyclic by construction
    (matches the DAG-style ground truth used in the codebase).
    """
    rng = np.random.default_rng(seed)

    # Cross mask: ~50% density, fully eligible pool.
    cross = (rng.random((X_len, S_len)) > 0.5).astype(np.float32)

    # Self mask: strictly lower-triangular adjacency (rows = children, cols = parents).
    self_mask = np.zeros((X_len, X_len), dtype=np.float32)
    for i in range(1, X_len):
        for j in range(i):
            if rng.random() < 0.5:
                self_mask[i, j] = 1.0

    return {
        "dec_cross": torch.from_numpy(cross),
        "dec_self": torch.from_numpy(self_mask),
    }


# -------------------------------------------------------------------
# Edge-count invariance
# -------------------------------------------------------------------

def test_preserve_sparsity_invariant_edge_count():
    """corrupted.sum() must equal ground_truth.sum() under preserve_sparsity=True."""
    X_len, S_len = 6, 4
    masks = _gt_masks(X_len=X_len, S_len=S_len, seed=0)
    gt_counts = {name: int(m.sum().item()) for name, m in masks.items()}

    corrupted, info = corrupt_dag_masks(
        masks,
        seed=123,
        cross_shd=4,
        self_shd=2,
        X_len=X_len,
        preserve_sparsity=True,
    )

    for name, m in corrupted.items():
        assert int(m.sum().item()) == gt_counts[name], (
            f"{name}: edge count drifted under preserve_sparsity "
            f"(gt={gt_counts[name]}, corrupted={int(m.sum().item())})"
        )
        assert info[name]["preserve_sparsity"] is True
        assert info[name]["n_swaps"] is not None


# -------------------------------------------------------------------
# Even-SHD / round-down
# -------------------------------------------------------------------

@pytest.mark.parametrize("shd_req", [0, 1, 2, 3, 4, 5, 6])
def test_preserve_sparsity_realised_is_even_round_down(shd_req):
    """shd_realised == 2 * (shd_req // 2) whenever the swap fits in the pool."""
    X_len, S_len = 8, 6
    masks = _gt_masks(X_len=X_len, S_len=S_len, seed=1)

    _, info = corrupt_dag_masks(
        masks,
        seed=42,
        cross_shd=shd_req,
        self_shd=shd_req,
        X_len=X_len,
        preserve_sparsity=True,
    )

    expected_swaps = shd_req // 2
    expected_real = 2 * expected_swaps

    for name in ("dec_cross", "dec_self"):
        # Skip cases where capacity caps the swap (covered separately below).
        k_true = info[name]["num_true_edges"]
        pool = info[name]["eligible_pool_size"]
        cap = min(k_true, pool - k_true) if pool is not None else expected_swaps
        if expected_swaps > cap:
            continue
        assert info[name]["shd_realised"] == expected_real, (
            f"{name}: shd_realised={info[name]['shd_realised']} "
            f"expected {expected_real} for shd_req={shd_req}"
        )
        assert info[name]["n_swaps"] == expected_swaps


# -------------------------------------------------------------------
# Capacity cap / fallback
# -------------------------------------------------------------------

def test_preserve_sparsity_capacity_cap_triggers_fallback():
    """
    When shd_req > 2 * min(k_true, n_non_edges), corruption is capped and
    fallback_used=True is reported, while edge count remains preserved.
    """
    # Tiny saturated cross mask: 2x2 with both edges set ⇒ k_true=4, non_edges=0
    # so the capacity is min(4, 0) = 0 swaps. Any non-zero shd_req must fall back.
    saturated = torch.ones((2, 2), dtype=torch.float32)
    masks = {"dec_cross": saturated}

    corrupted, info = corrupt_dag_masks(
        masks,
        seed=7,
        cross_shd=2,
        self_shd=0,
        X_len=2,
        preserve_sparsity=True,
    )

    inf = info["dec_cross"]
    assert inf["fallback_used"] is True
    assert inf["n_swaps"] == 0
    assert inf["shd_realised"] == 0
    # Edge count still preserved.
    assert int(corrupted["dec_cross"].sum().item()) == int(saturated.sum().item())


# -------------------------------------------------------------------
# Self-mask diagonal stays clean
# -------------------------------------------------------------------

def test_preserve_sparsity_self_diagonal_untouched():
    """No self-loops introduced by corruption (off-diagonal pool only)."""
    X_len = 5
    masks = _gt_masks(X_len=X_len, S_len=3, seed=2)

    corrupted, _ = corrupt_dag_masks(
        masks,
        seed=99,
        cross_shd=0,
        self_shd=4,
        X_len=X_len,
        preserve_sparsity=True,
    )
    diag = torch.diag(corrupted["dec_self"])
    assert torch.all(diag == 0.0), f"self-loops introduced on diagonal: {diag}"


# -------------------------------------------------------------------
# Backward compatibility: preserve_sparsity=False is unchanged
# -------------------------------------------------------------------

def test_preserve_sparsity_false_is_bit_identical_to_legacy():
    """
    With preserve_sparsity=False (the default) the function must produce the
    exact same masks it did before this feature was added: SHD is enforced
    by free flips and edge count is generally NOT preserved.
    """
    X_len, S_len = 6, 4
    masks = _gt_masks(X_len=X_len, S_len=S_len, seed=3)

    out_a, info_a = corrupt_dag_masks(
        masks,
        seed=2024,
        cross_shd=3,
        self_shd=2,
        X_len=X_len,
        preserve_sparsity=False,
    )
    # Default (parameter omitted) should match preserve_sparsity=False exactly.
    out_b, info_b = corrupt_dag_masks(
        masks,
        seed=2024,
        cross_shd=3,
        self_shd=2,
        X_len=X_len,
    )
    for name in out_a:
        assert torch.equal(out_a[name], out_b[name])
        # Realised SHD equals request (no even-rounding in legacy mode).
        if info_a[name]["num_true_edges"] >= info_a[name]["shd_requested"]:
            assert info_a[name]["shd_realised"] == info_a[name]["shd_requested"]
        # Legacy info reports preserve_sparsity=False / n_swaps=None.
        assert info_a[name]["preserve_sparsity"] is False
        assert info_a[name]["n_swaps"] is None


# -------------------------------------------------------------------
# RNG sub-stream independence still holds
# -------------------------------------------------------------------

def test_preserve_sparsity_rng_substreams_independent():
    """Toggling cross_shd must not reshuffle the corrupted self mask."""
    X_len, S_len = 6, 4
    masks = _gt_masks(X_len=X_len, S_len=S_len, seed=4)

    out_low, _ = corrupt_dag_masks(
        masks, seed=11, cross_shd=2, self_shd=2,
        X_len=X_len, preserve_sparsity=True,
    )
    out_high, _ = corrupt_dag_masks(
        masks, seed=11, cross_shd=4, self_shd=2,  # only cross changed
        X_len=X_len, preserve_sparsity=True,
    )
    assert torch.equal(out_low["dec_self"], out_high["dec_self"]), (
        "Self mask should be untouched by changing cross_shd alone "
        "(independent RNG sub-streams violated)."
    )
