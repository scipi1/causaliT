"""Unit tests for centroid-commit query dynamics.

Covers the exact nearest-centroid projection (vs brute force over all 2^N
subsets), the straight-through shadow path in FreeQueryEmbedding, and the
commit event semantics (weight write, shadow re-centering, M reset).
"""

import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaliT.training.centroid_commit import (
    CentroidCommitController,
    best_subset,
    centroid_of,
)
from causaliT.core.modules.free_query_embedding import FreeQueryEmbedding

N, D = 6, 6


def _ortho_frame(n=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, d, generator=g).double())
    return q  # (n, d) orthonormal rows


class TestProjection:
    def test_matches_brute_force(self):
        K = _ortho_frame()
        g = torch.Generator().manual_seed(1)
        for trial in range(20):
            q = torch.randn(D, generator=g).double()
            best, best_score = (), -float("inf")
            for r in range(N + 1):
                for S in itertools.combinations(range(N), r):
                    if not S:
                        score = 0.0
                    else:
                        c = K[list(S)].mean(0)
                        score = float(q @ c / (q.norm() * c.norm()))
                    if score > best_score:
                        best, best_score = S, score
            got = best_subset(q, K)
            assert set(got.tolist()) == set(best), f"trial {trial}"

    def test_exclude_self_loop(self):
        K = _ortho_frame()
        got = best_subset(K[2].clone(), K, exclude=2)  # query IS key 2
        assert 2 not in got.tolist()
        assert got.numel() == 1

    def test_empty_subset_when_all_negative(self):
        K = _ortho_frame()
        got = best_subset(-K.mean(0), K)   # anti-centroid: nothing helps
        assert got.numel() == 0

    def test_centroid_of_empty_is_zero(self):
        K = _ortho_frame()
        assert torch.all(centroid_of(K, torch.empty(0, dtype=torch.long)) == 0)

    def test_full_centroid_selects_all_keys(self):
        K = _ortho_frame()
        got = best_subset(K.mean(0), K)    # the init point: "select all"
        assert got.numel() == N



class TestStraightThrough:
    def _table(self):
        torch.manual_seed(0)
        t = FreeQueryEmbedding(num_variables=3, d_model=D)
        t.enable_commit_shadow()
        return t

    def test_forward_value_is_committed_weight(self):
        t = self._table()
        X = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]])
        out = t(X)
        w = t.embedding.weight
        assert torch.allclose(out[0, 0], w[1].detach())
        assert torch.allclose(out[0, 2], w[3].detach())
        # moving the shadow does NOT change the forward value
        with torch.no_grad():
            t.shadow.add_(torch.randn_like(t.shadow))
        assert torch.allclose(t(X)[0, 0], w[1].detach())

    def test_gradient_lands_on_shadow_not_weight(self):
        t = self._table()
        X = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]])
        t(X).sum().backward()
        assert t.shadow.grad is not None
        assert t.embedding.weight.grad is None  # detached: no grad to weight

    def test_shadow_in_state_dict_and_sync(self):
        t = self._table()
        assert "shadow" in t.state_dict()
        with torch.no_grad():
            t.shadow.add_(1.0)
        t.sync_shadow_to_weight()
        assert torch.equal(t.shadow, t.embedding.weight)

    def test_apply_releafs_shadow(self):
        """Device/dtype moves (``.double()`` here; ``.cuda()`` on the cluster)
        replace buffers with op results — non-leaf, requires_grad preserved.
        The _apply override must re-leaf the shadow (cluster job 12294402:
        non-leaf shadow broke deepcopy in the bilevel probe and starved
        shadow.grad on GPU runs)."""
        t = self._table()
        assert t.shadow.is_leaf
        t = t.double()
        assert t.shadow.is_leaf, "shadow is non-leaf after .double()"
        assert t.shadow.requires_grad
        assert t.shadow.dtype == torch.float64
        # grads still accumulate on the moved shadow
        X = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]],
                         dtype=torch.float64)
        t(X).sum().backward()
        assert t.shadow.grad is not None


class TestController:
    def _setup(self, n=N, d=D):
        K = _ortho_frame(n, d)
        torch.manual_seed(0)
        qS = FreeQueryEmbedding(num_variables=1, d_model=d)   # node 0
        qX = FreeQueryEmbedding(num_variables=n - 1, d_model=d)  # nodes 1..n-1
        for t in (qS, qX):
            t.enable_commit_shadow()
            with torch.no_grad():
                c = K.mean(0)          # every query at the global centroid
                t.embedding.weight[1:] = c.float()
            t.sync_shadow_to_weight()
        m = torch.nn.Parameter(torch.full((n,), -1.0))  # M = e^-1 < 1
        ctl = CentroidCommitController([qS, qX], K, m, evidence_lr=50.0,
                                       evidence_leak=0.95)
        return K, qS, qX, m, ctl

    def test_consistent_evidence_commits_and_recenters(self):
        K, qS, qX, m, ctl = self._setup()
        # node 3 (qX row 3) gets a consistent gradient AWAY from key 1, so
        # evidence (-eta * g) points TOWARD key 1.
        n_steps = None
        for t in range(200):
            for tab in (qS, qX):
                tab.shadow.grad = torch.zeros_like(tab.shadow)
            qX.shadow.grad[3] = -0.01 * K[1].float() + 0.001 * torch.randn(D)
            n_new = ctl.step()
            if n_new:
                n_steps = t
                break
        assert n_new == 1
        # node 3 = global index 3 committed to the singleton {1}
        assert torch.allclose(qX.embedding.weight[3].double(), K[1])
        assert torch.allclose(qX.shadow[3].double(), K[1])  # shadow re-centred
        assert m.data[3].item() == 0.0                      # M reset to exp(0)=1
        assert m.data[0].item() == -1.0                     # others untouched
        assert ctl.commit_counts[3] == 1
        sizes = ctl.assignment_sizes()
        assert sizes[3] == 1

    def test_no_commit_under_pure_noise(self):
        K, qS, qX, m, ctl = self._setup()
        ctl.eta = 1.0   # weak evidence: random walk should not cross a cell
        torch.manual_seed(7)
        for _ in range(50):
            for tab in (qS, qX):
                tab.shadow.grad = 0.001 * torch.randn_like(tab.shadow)
            ctl.step()
        assert ctl.n_commits == 0

    def test_shadow_grads_cleared_and_grad_override(self):
        K, qS, qX, m, ctl = self._setup()
        for tab in (qS, qX):
            tab.shadow.grad = torch.randn_like(tab.shadow)
        override = [torch.zeros_like(qS.shadow), torch.zeros_like(qX.shadow)]
        override[1][2] = -10.0 * K[0].float()   # node 2 evidence toward key 0
        ctl.step(override)
        assert qS.shadow.grad is None and qX.shadow.grad is None
        assert torch.allclose(qX.embedding.weight[2].double(), K[0])
        assert ctl.commit_counts[2] == 1

    def test_invalid_reset_mode_raises(self):
        K, qS, qX, m, _ = self._setup()
        with pytest.raises(ValueError, match="reset_m_on_commit"):
            CentroidCommitController([qS, qX], K, m, reset_m_on_commit="bad")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_cuda_frame_and_shadow_step(self):
        """Regression: controller must work when the frame + shadows live on
        CUDA (the projection internally runs on whichever device q is on)."""
        K, qS, qX, m, ctl = self._setup()
        qS.cuda(); qX.cuda(); m = m.cuda()
        ctl.norm_param = m
        ctl.K = ctl.K.cuda()
        for tab in (qS, qX):
            tab.shadow.grad = 0.001 * torch.randn_like(tab.shadow)
        ctl.step()            # must not raise a device mismatch
        sizes = ctl.assignment_sizes()
        assert sizes.numel() == N

