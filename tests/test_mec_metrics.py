"""
Tests for Markov Equivalence Class (MEC) metrics.

These tests verify the correctness of:
- _dag_to_skeleton: Convert DAG to undirected skeleton
- _find_v_structures: Detect v-structures (colliders)
- _combine_attention_to_full_dag: Combine attention blocks into full DAG
- _compute_mec_distance: Soft MEC distance metric
- _check_mec_membership: Binary MEC membership check
"""

import numpy as np
import pytest

from causaliT.evaluation.eval_funs.helpers.eval_utils import (
    _dag_to_skeleton,
    _find_v_structures,
    _combine_attention_to_full_dag,
    _soft_skeleton_distance,
    _soft_v_structure_distance,
    _compute_mec_distance,
    _check_mec_membership,
    _compute_mec_threshold,
)


class TestDagToSkeleton:
    """Tests for _dag_to_skeleton function."""
    
    def test_simple_chain(self):
        """Test skeleton of chain graph: 0 → 1 → 2"""
        dag = np.array([
            [0, 0, 0],
            [1, 0, 0],  # 0 → 1
            [0, 1, 0],  # 1 → 2
        ])
        skeleton = _dag_to_skeleton(dag)
        
        assert frozenset({0, 1}) in skeleton
        assert frozenset({1, 2}) in skeleton
        assert frozenset({0, 2}) not in skeleton
        assert len(skeleton) == 2
    
    def test_v_structure(self):
        """Test skeleton of v-structure: 0 → 2 ← 1"""
        dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # 0 → 2, 1 → 2
        ])
        skeleton = _dag_to_skeleton(dag)
        
        assert frozenset({0, 2}) in skeleton
        assert frozenset({1, 2}) in skeleton
        assert frozenset({0, 1}) not in skeleton  # Parents not adjacent
        assert len(skeleton) == 2
    
    def test_empty_dag(self):
        """Test skeleton of DAG with no edges."""
        dag = np.zeros((3, 3))
        skeleton = _dag_to_skeleton(dag)
        
        assert len(skeleton) == 0
    
    def test_fully_connected(self):
        """Test skeleton of fully connected 3-node DAG."""
        dag = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ])
        skeleton = _dag_to_skeleton(dag)
        
        assert len(skeleton) == 3
        assert frozenset({0, 1}) in skeleton
        assert frozenset({0, 2}) in skeleton
        assert frozenset({1, 2}) in skeleton


class TestFindVStructures:
    """Tests for _find_v_structures function."""
    
    def test_simple_v_structure(self):
        """Test detection of v-structure: 0 → 2 ← 1 (parents NOT adjacent)."""
        dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # 0 → 2, 1 → 2
        ])
        v_structs = _find_v_structures(dag)
        
        assert (2, 0, 1) in v_structs
        assert len(v_structs) == 1
    
    def test_no_v_structure_when_parents_adjacent(self):
        """Test: no v-structure when parents are adjacent."""
        dag = np.array([
            [0, 0, 0],
            [1, 0, 0],  # 0 → 1 (parents are adjacent!)
            [1, 1, 0],  # 0 → 2, 1 → 2
        ])
        v_structs = _find_v_structures(dag)
        
        # No v-structure because 0 and 1 are adjacent
        assert len(v_structs) == 0
    
    def test_chain_no_v_structure(self):
        """Test: chain graph has no v-structures."""
        dag = np.array([
            [0, 0, 0],
            [1, 0, 0],  # 0 → 1
            [0, 1, 0],  # 1 → 2
        ])
        v_structs = _find_v_structures(dag)
        
        assert len(v_structs) == 0
    
    def test_multiple_v_structures(self):
        """Test detection of multiple v-structures."""
        # DAG: 0 → 2 ← 1, and also 0 → 3 ← 1
        dag = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],  # 0 → 2, 1 → 2
            [1, 1, 0, 0],  # 0 → 3, 1 → 3
        ])
        v_structs = _find_v_structures(dag)
        
        assert (2, 0, 1) in v_structs
        assert (3, 0, 1) in v_structs
        assert len(v_structs) == 2


class TestCombineAttentionToFullDag:
    """Tests for _combine_attention_to_full_dag function."""
    
    def test_simple_combination(self):
        """Test combining cross and self attention into full DAG."""
        # 2 sources, 2 intermediates
        # S1 → X1, S2 → X2
        cross_adj = np.array([
            [1, 0],  # X1 ← S1
            [0, 1],  # X2 ← S2
        ])
        # X1 → X2
        self_adj = np.array([
            [0, 0],
            [1, 0],  # X2 ← X1
        ])
        
        full_dag = _combine_attention_to_full_dag(cross_adj, self_adj, n_source=2, n_intermediate=2)
        
        assert full_dag.shape == (4, 4)
        # Check S → X edges (rows 2-3, cols 0-1)
        assert full_dag[2, 0] == 1  # X1 ← S1
        assert full_dag[3, 1] == 1  # X2 ← S2
        # Check X → X edges (rows 2-3, cols 2-3)
        assert full_dag[3, 2] == 1  # X2 ← X1
        # Check S → S edges (should be zero)
        assert full_dag[0, 1] == 0
        assert full_dag[1, 0] == 0


class TestSoftSkeletonDistance:
    """Tests for _soft_skeleton_distance function."""
    
    def test_perfect_match(self):
        """Test perfect skeleton match returns distance 0."""
        learned = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1.0, 1.0, 0],  # Strong edges 0→2, 1→2
        ])
        true_skeleton = {frozenset({0, 2}), frozenset({1, 2})}
        
        dist, details = _soft_skeleton_distance(learned, true_skeleton)
        
        assert details["recall"] == 1.0
        assert dist < 0.5  # Should be low distance
    
    def test_missing_edge(self):
        """Test missing edge reduces recall."""
        learned = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1.0, 0.0, 0],  # Only edge 0→2, missing 1→2
        ])
        true_skeleton = {frozenset({0, 2}), frozenset({1, 2})}
        
        dist, details = _soft_skeleton_distance(learned, true_skeleton)
        
        assert details["recall"] == 0.5  # One of two edges present


class TestComputeMecDistance:
    """Tests for _compute_mec_distance function."""
    
    def test_identical_dags(self):
        """Test identical DAGs have distance 0."""
        dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # V-structure: 0 → 2 ← 1
        ])
        
        dist, details = _compute_mec_distance(dag, dag)
        
        assert dist == 0.0
        assert details["skeleton_distance"] == 0.0
        assert details["v_structure_distance"] == 0.0
    
    def test_different_skeleton(self):
        """Test different skeleton increases distance."""
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # V-structure: 0 → 2 ← 1
        ])
        learned_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1.0, 0.0, 0],  # Only 0 → 2
        ])
        
        dist, details = _compute_mec_distance(learned_dag, true_dag)
        
        assert dist > 0.0
        assert details["skeleton_recall"] < 1.0


class TestCheckMecMembership:
    """Tests for _check_mec_membership function."""
    
    def test_identical_dag_in_mec(self):
        """Test identical DAG is in MEC."""
        dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1.0, 1.0, 0],  # V-structure
        ])
        
        in_mec, details = _check_mec_membership(dag, dag, threshold=0.5)
        
        assert in_mec is True
        assert details["skeleton_match"] is True
        assert details["v_structure_match"] is True
    
    def test_missing_edge_not_in_mec(self):
        """Test missing edge means not in MEC."""
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # V-structure: 0 → 2 ← 1
        ])
        learned_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0.9, 0.1, 0],  # Missing edge 1 → 2 (below threshold)
        ])
        
        in_mec, details = _check_mec_membership(learned_dag, true_dag, threshold=0.5)
        
        assert in_mec is False
        assert details["skeleton_match"] is False
        assert details["n_missing_edges"] == 1
    
    def test_extra_edge_not_in_mec(self):
        """Test extra edge means not in MEC."""
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],  # V-structure: 0 → 2 ← 1
        ])
        learned_dag = np.array([
            [0, 0.9, 0],  # Extra edge: 1 → 0
            [0, 0, 0],
            [1.0, 1.0, 0],
        ])
        
        in_mec, details = _check_mec_membership(learned_dag, true_dag, threshold=0.5)
        
        assert in_mec is False
        assert details["skeleton_match"] is False
        assert details["n_extra_edges"] == 1


class TestMecWithRealStructure:
    """Integration tests with realistic SCM structures."""
    
    def test_scm1_like_structure(self):
        """Test with SCM1-like structure: S1→X1, S2→X2, X1→X2."""
        # True DAG (4 nodes: S1, S2, X1, X2)
        # Indices: S1=0, S2=1, X1=2, X2=3
        true_dag = np.array([
            [0, 0, 0, 0],  # S1 has no parents
            [0, 0, 0, 0],  # S2 has no parents
            [1, 0, 0, 0],  # X1 ← S1
            [0, 1, 1, 0],  # X2 ← S2, X1
        ])
        
        # Check structure
        skeleton = _dag_to_skeleton(true_dag)
        v_structs = _find_v_structures(true_dag)
        
        # Expected: 3 edges (S1-X1, S2-X2, X1-X2)
        assert len(skeleton) == 3
        # Expected: 1 v-structure (S2 → X2 ← X1)
        # Because S2 and X1 are not adjacent
        assert (3, 1, 2) in v_structs
        
        # Perfect learned DAG should be in MEC
        in_mec, details = _check_mec_membership(true_dag.astype(float), true_dag)
        assert in_mec is True


class TestComputeMecThreshold:
    """Tests for _compute_mec_threshold function."""

    def test_perfect_scores_return_high_threshold(self):
        """
        When true edges have score 1.0 and non-edges have score 0.0, the MEC
        threshold should equal the minimum true-edge score (1.0 here).
        """
        # V-structure: 0 → 2 ← 1
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
        ])
        # Perfect scores: true edges = 1.0, non-edges = 0.0
        learned = true_dag.astype(float)

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        assert exists is True
        assert mec_thresh is not None
        assert mec_thresh == pytest.approx(1.0)

    def test_good_scores_threshold_below_min_edge(self):
        """
        When true edges have clear scores (0.85, 0.75) and non-edges are 0,
        the MEC threshold should be ≤ min(true-edge scores) = 0.75 and > 0.
        """
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
        ])
        learned = np.array([
            [0,    0,    0],
            [0,    0,    0],
            [0.85, 0.75, 0],
        ])

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        assert exists is True
        assert mec_thresh is not None
        assert 0.0 < mec_thresh <= 0.75

    def test_never_in_mec_returns_none(self):
        """
        When the learned scores are all equal (uniform), the binarised graph
        is either the full graph (all scores ≥ θ) or the empty graph
        (all scores < θ).  Neither matches the 2-edge fork skeleton, so
        no threshold achieves MEC membership.

        True DAG: fork  0 → 1,  0 → 2  (skeleton has exactly 2 edges;
        v-structures: none — each of 1 and 2 has only one parent).

        At every candidate threshold θ:
          θ ≤ 0.3       → full graph (3 edges: 0-1, 0-2, 1-2) ≠ 2-edge skeleton
          θ > 0.3+ε     → empty graph (0 edges) ≠ 2-edge skeleton
        Hence never in MEC.
        """
        # Fork: 0 → 1, 0 → 2
        true_dag = np.array([
            [0, 0, 0],
            [1, 0, 0],   # 0 → 1
            [1, 0, 0],   # 0 → 2
        ])
        # All scores uniform at 0.3 → no threshold recovers exactly 2 specific edges
        learned = np.full((3, 3), 0.3)

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        assert exists is False
        assert mec_thresh is None

    def test_threshold_respects_wrong_v_structure(self):
        """
        If the model perfectly recovers the skeleton but orients one edge
        incorrectly (creating a wrong v-structure), membership should fail.
        Conversely, at a higher threshold that drops the spurious edge, it
        might succeed.
        """
        # True DAG: 0 → 2 ← 1 (v-structure at 2; parents 0 and 1 NOT adjacent)
        true_dag = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
        ])
        # Model: correct edges (0→2, 1→2) but also adds spurious 0→1 edge
        # At threshold = 0.5: three edges, wrong v-structure (parents adjacent)
        # At threshold = 0.9: only 0→2 and 1→2 survive (spurious 0→1 dropped)
        learned = np.array([
            [0,   0.6, 0],   # spurious 0→1 edge at 0.6
            [0,   0,   0],
            [0.9, 0.9, 0],   # true edges at 0.9
        ])

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        # At threshold >= 0.9, only 0→2 and 1→2 survive → correct MEC
        # At threshold 0.6, spurious 0→1 is included → skeleton changes
        # Best threshold should be 0.9
        assert exists is True
        assert mec_thresh is not None
        assert mec_thresh == pytest.approx(0.9)

    def test_chain_graph_perfect_scores(self):
        """Chain graph: 0 → 1 → 2. True edges have high scores."""
        true_dag = np.array([
            [0, 0, 0],
            [1, 0, 0],  # 0 → 1
            [0, 1, 0],  # 1 → 2
        ])
        # True edges at 0.8, non-edges at 0.0
        learned = np.array([
            [0,   0,   0],
            [0.8, 0,   0],
            [0,   0.8, 0],
        ])

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        assert exists is True
        assert mec_thresh is not None
        assert 0.0 < mec_thresh <= 0.8

    def test_empty_true_dag(self):
        """
        True DAG with no edges: the empty binarised graph (any threshold ≥ max
        score) is trivially in the MEC (empty skeleton, no v-structures).
        """
        true_dag = np.zeros((3, 3))
        # Scores all near zero; binarised at any threshold > 0 gives empty graph
        learned = np.array([
            [0,    0,    0],
            [0.1,  0,    0],
            [0.05, 0.02, 0],
        ])

        mec_thresh, exists = _compute_mec_threshold(learned, true_dag)

        # At the highest threshold (0.1), all edges are dropped → in MEC
        assert exists is True
        assert mec_thresh is not None
        assert mec_thresh == pytest.approx(0.1)

    def test_return_type(self):
        """Verify return types are (float | None, bool)."""
        true_dag = np.array([[0, 0], [1, 0]])
        learned = np.array([[0.0, 0.0], [0.9, 0.0]])

        result = _compute_mec_threshold(learned, true_dag)

        assert isinstance(result, tuple)
        assert len(result) == 2
        mec_thresh, exists = result
        assert isinstance(exists, bool)
        if exists:
            assert isinstance(mec_thresh, float)
        else:
            assert mec_thresh is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
