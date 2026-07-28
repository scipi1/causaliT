"""
Tests for the architecture-agnostic DAG query helper (eval_dag_query).

Covers the behaviour that motivated the harmonisation of eval_attention and
eval_attention_selector: the DAG must be assembled correctly whether the model
has cross-attention only, self-attention only, or both - and it must be derived
from the *shape* of the attention tensors, without a live model.
"""

import numpy as np
import pytest

from causaliT.evaluation.eval_funs.eval_dag_query import (
    CROSS,
    SELF,
    block_layer_index,
    block_mask_type,
    canonical_block_name,
    describe_topology,
    query_dag_blocks,
    reduce_attention,
    split_combined_attention,
    assemble_full_dag,
)

L_S, L_X = 3, 4


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("dec_cross", CROSS),
    ("dec_cross_L2", f"{CROSS}_L2"),
    ("dec_self", SELF),
    ("dec_self_L0", f"{SELF}_L0"),
    ("decoder", SELF),
    ("cross", CROSS),
    ("encoder", None),
    ("att_combined", None),
])
def test_canonical_block_name(key, expected):
    assert canonical_block_name(key) == expected


def test_block_mask_type_and_layer_index():
    assert block_mask_type("cross") == "dec_cross"
    assert block_mask_type("self_L3") == "dec_self"
    assert block_mask_type("nonsense") is None
    assert block_layer_index("cross_L3") == 3
    assert block_layer_index("cross") is None


# ---------------------------------------------------------------------------
# Attention reduction and shape-based classification
# ---------------------------------------------------------------------------

def test_reduce_attention_handles_batch_and_heads():
    assert reduce_attention(np.ones((L_X, L_S))).shape == (L_X, L_S)
    assert reduce_attention(np.ones((8, L_X, L_S))).shape == (L_X, L_S)
    assert reduce_attention(np.ones((8, 2, L_X, L_S))).shape == (L_X, L_S)
    assert reduce_attention(None) is None


def test_split_combined_attention_both_blocks():
    matrix = np.concatenate(
        [np.full((L_X, L_S), 0.2), np.full((L_X, L_X), 0.8)], axis=1
    )
    blocks = split_combined_attention(matrix, L_S=L_S, L_X=L_X)

    assert set(blocks) == {CROSS, SELF}
    assert blocks[CROSS].shape == (L_X, L_S)
    assert blocks[SELF].shape == (L_X, L_X)
    assert np.allclose(blocks[CROSS], 0.2)
    assert np.allclose(blocks[SELF], 0.8)


def test_split_combined_attention_cross_only():
    blocks = split_combined_attention(np.zeros((L_X, L_S)), L_S=L_S, L_X=L_X)
    assert set(blocks) == {CROSS}


def test_split_combined_attention_self_only():
    blocks = split_combined_attention(np.zeros((L_X, L_X)), L_S=L_S, L_X=L_X)
    assert set(blocks) == {SELF}


def test_split_combined_attention_unexpected_width_is_skipped():
    blocks = split_combined_attention(np.zeros((L_X, 99)), L_S=L_S, L_X=L_X)
    assert blocks == {}


def test_split_combined_attention_wrong_row_count_is_skipped():
    """Rows must be query tokens (X); an (L_S, L_S) encoder block is not a DAG block."""
    blocks = split_combined_attention(np.zeros((L_S, L_S)), L_S=L_S, L_X=L_X)
    assert blocks == {}


def test_split_combined_attention_without_dims_is_skipped():
    blocks = split_combined_attention(np.zeros((L_X, L_S)))
    assert blocks == {}


# ---------------------------------------------------------------------------
# query_dag_blocks
# ---------------------------------------------------------------------------

def test_query_dag_blocks_from_combined_tensor():
    att = np.random.rand(16, L_X, L_S + L_X)

    blocks = query_dag_blocks({"att_combined": att}, L_S=L_S, L_X=L_X)

    assert set(blocks) == {CROSS, SELF}
    assert np.allclose(blocks[CROSS], att.mean(axis=0)[:, :L_S])
    assert np.allclose(blocks[SELF], att.mean(axis=0)[:, L_S:])


def test_query_dag_blocks_from_multihead_tensor():
    att = np.random.rand(4, 2, L_X, L_S + L_X)

    blocks = query_dag_blocks(att, L_S=L_S, L_X=L_X)

    assert blocks[CROSS].shape == (L_X, L_S)
    assert blocks[SELF].shape == (L_X, L_X)


def test_query_dag_blocks_from_per_key_dict():
    blocks = query_dag_blocks(
        {
            "dec_cross": np.zeros((L_X, L_S)),
            "dec_self": np.zeros((L_X, L_X)),
            "encoder": np.zeros((L_S, L_S)),   # must be ignored
        },
        L_S=L_S,
        L_X=L_X,
    )
    assert set(blocks) == {CROSS, SELF}


def test_query_dag_blocks_per_layer_keys():
    blocks = query_dag_blocks(
        {
            "dec_cross_L0": np.zeros((L_X, L_S)),
            "dec_cross_L1": np.zeros((L_X, L_S)),
            "dec_self_L0": np.zeros((L_X, L_X)),
        },
        L_S=L_S,
        L_X=L_X,
    )
    assert set(blocks) == {f"{CROSS}_L0", f"{CROSS}_L1", f"{SELF}_L0"}


def test_query_dag_blocks_combined_per_layer_keys_get_suffix():
    """A combined block on a per-layer key splits into two suffixed blocks."""
    blocks = query_dag_blocks(
        {"att_combined_L1": np.zeros((L_X, L_S + L_X))}, L_S=L_S, L_X=L_X
    )
    assert set(blocks) == {f"{CROSS}_L1", f"{SELF}_L1"}


def test_query_dag_blocks_cross_attention_only_model():
    """A model without X->X attention yields only the cross block."""
    blocks = query_dag_blocks({"dec_cross": np.zeros((L_X, L_S))}, L_S=L_S, L_X=L_X)
    assert set(blocks) == {CROSS}


def test_query_dag_blocks_none_tensors_are_ignored():
    blocks = query_dag_blocks(
        {"dec_cross": np.zeros((L_X, L_S)), "dec_self": None}, L_S=L_S, L_X=L_X
    )
    assert set(blocks) == {CROSS}


def test_query_dag_blocks_ignores_mis_shaped_blocks():
    blocks = query_dag_blocks(
        {"dec_cross": np.zeros((L_X, L_S)), "dec_self": np.zeros((L_X, 99))},
        L_S=L_S,
        L_X=L_X,
    )
    assert set(blocks) == {CROSS}


# ---------------------------------------------------------------------------
# The L_S == L_X ambiguity is resolved by the key name
# ---------------------------------------------------------------------------

def test_square_blocks_are_disambiguated_by_key_name():
    n = 4
    blocks = query_dag_blocks(
        {
            "dec_cross": np.full((n, n), 0.1),
            "dec_self": np.full((n, n), 0.9),
        },
        L_S=n,
        L_X=n,
    )
    assert set(blocks) == {CROSS, SELF}
    assert np.allclose(blocks[CROSS], 0.1)
    assert np.allclose(blocks[SELF], 0.9)


def test_square_encoder_block_is_rejected_when_dims_are_equal():
    """`encoder` is (L_S, L_S); with L_S == L_X only the name can rule it out."""
    n = 4
    blocks = query_dag_blocks({"encoder": np.ones((n, n))}, L_S=n, L_X=n)
    assert blocks == {}


def test_square_unnamed_block_is_skipped_when_dims_are_equal():
    n = 4
    blocks = query_dag_blocks({"att_combined": np.ones((n, n))}, L_S=n, L_X=n)
    assert blocks == {}


def test_combined_block_still_splits_when_dims_are_equal():
    n = 4
    blocks = query_dag_blocks({"att_combined": np.ones((n, 2 * n))}, L_S=n, L_X=n)
    assert set(blocks) == {CROSS, SELF}


# ---------------------------------------------------------------------------
# describe_topology (logging only)
# ---------------------------------------------------------------------------

def test_describe_topology_reports_blocks_and_mec_skip():
    both = describe_topology(
        {CROSS: np.zeros((L_X, L_S)), SELF: np.zeros((L_X, L_X))}, L_S, L_X
    )
    assert CROSS in both and SELF in both and "MEC" not in both

    cross_only = describe_topology({CROSS: np.zeros((L_X, L_S))}, L_S, L_X)
    assert "MEC will be skipped" in cross_only

    empty = describe_topology({}, L_S, L_X)
    assert "no DAG block" in empty


# ---------------------------------------------------------------------------
# Full DAG assembly
# ---------------------------------------------------------------------------

def test_assemble_full_dag_places_blocks_correctly():
    cross = np.full((L_X, L_S), 0.3)
    self_adj = np.full((L_X, L_X), 0.7)

    full = assemble_full_dag({CROSS: cross, SELF: self_adj}, L_S=L_S, L_X=L_X)

    assert full is not None
    assert full.shape == (L_S + L_X, L_S + L_X)
    # Sources come first: X rows/S columns hold the cross block.
    assert np.allclose(full[L_S:, :L_S], cross)
    assert np.allclose(full[L_S:, L_S:], self_adj)
    # Nothing points into the sources.
    assert np.allclose(full[:L_S, :], 0.0)


def test_assemble_full_dag_requires_both_blocks():
    assert assemble_full_dag({CROSS: np.zeros((L_X, L_S))}) is None
    assert assemble_full_dag({SELF: np.zeros((L_X, L_X))}) is None
    assert assemble_full_dag({}) is None


def test_assemble_full_dag_layer_suffix():
    blocks = {
        f"{CROSS}_L0": np.zeros((L_X, L_S)),
        f"{SELF}_L0": np.zeros((L_X, L_X)),
    }
    assert assemble_full_dag(blocks, L_S=L_S, L_X=L_X, layer_suffix="_L0") is not None
    assert assemble_full_dag(blocks, L_S=L_S, L_X=L_X, layer_suffix="_L1") is None


def test_query_then_assemble_round_trip():
    """The pipeline used by eval_attention: query -> assemble."""
    att = np.random.rand(8, L_X, L_S + L_X)
    blocks = query_dag_blocks({"att_combined": att}, L_S=L_S, L_X=L_X)
    full = assemble_full_dag(blocks, L_S=L_S, L_X=L_X)

    assert full.shape == (L_S + L_X, L_S + L_X)
    assert np.allclose(full[L_S:, :L_S], att.mean(axis=0)[:, :L_S])
