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

from causaliT.evaluation.eval_funs.eval_utils import (
    _dag_to_skeleton,
    _find_v_structures,
    _combine_attention_to_full_dag,
    _soft_skeleton_distance,
    _soft_v_structure_distance,
    _compute_mec_distance,
    _check_mec_membership,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
