"""Tests for the bilevel-gated centroid commits (Phase 2).

Covers: taboo-aware subset search (``best_subset_excluding``), the deferred
commit path of ``CentroidCommitController.step``, ``finalize`` accept/reject
semantics, taboo blocking/lifting dynamics, controller persistence, and the
forecaster-level gate wiring (``_run_bilevel_gate``).
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
    best_subset_excluding,
    centroid_of,
)
from causaliT.core.modules.free_query_embedding import FreeQueryEmbedding

N, D = 6, 6


def _ortho_frame(n=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, d, generator=g).double())
    return q


class TestBestSubsetExcluding:
    def test_matches_brute_force_with_forbidden(self):
        """best_subset_excluding == argmax over all admissible subsets."""
        K = _ortho_frame()
        g = torch.Generator().manual_seed(3)
        for trial in range(15):
            q = torch.randn(D, generator=g).double()
            forbidden = {frozenset(s) for r in range(N + 1)
                         for s in itertools.combinations(range(N), r)
                         if (hash((trial, s)) % 5) == 0}   # arbitrary subset
            best, best_score = (), 0.0                      # empty scores 0
            for r in range(N + 1):
                for S in itertools.combinations(range(N), r):
                    if not S or frozenset(S) in forbidden:
                        continue
                    c = K[list(S)].mean(0)
                    score = float(q @ c / (q.norm() * c.norm()))
                    if score > best_score:
                        best, best_score = S, score
            got = best_subset_excluding(q, K, forbidden=forbidden)
            assert set(got.tolist()) == set(best), f"trial {trial}"

    def test_no_forbidden_matches_best_subset(self):
        K = _ortho_frame()
        g = torch.Generator().manual_seed(4)
        for _ in range(10):
            q = torch.randn(D, generator=g).double()
            a = best_subset(q, K)
            b = best_subset_excluding(q, K, forbidden=set())
            assert set(a.tolist()) == set(b.tolist())

    def test_all_forbidden_returns_empty(self):
        K = _ortho_frame()
        q = K.mean(0).clone()
        forbidden = {frozenset(s) for r in range(1, N + 1)
                     for s in itertools.combinations(range(N), r)}
        got = best_subset_excluding(q, K, forbidden=forbidden)
        assert got.numel() == 0


class TestDeferAndFinalize:
    def _setup(self):
        K = _ortho_frame()
        torch.manual_seed(0)
        qS = FreeQueryEmbedding(num_variables=1, d_model=D)
        qX = FreeQueryEmbedding(num_variables=N - 1, d_model=D)
        for t in (qS, qX):
            t.enable_commit_shadow()
            with torch.no_grad():
                t.embedding.weight[1:] = K.mean(0).float()
            t.sync_shadow_to_weight()
        m = torch.nn.Parameter(torch.full((N,), -1.0))
        ctl = CentroidCommitController([qS, qX], K, m, evidence_lr=50.0,
                                       evidence_leak=0.95)
        return K, qS, qX, m, ctl

    def _drive(self, K, qS, qX, ctl, node=3, key=1, steps=200, defer=True):
        """Consistent gradient pushing ``node`` toward ``key``; stops at the
        first eligibility.  Returns (n_eligible, steps_taken)."""
        for t in range(steps):
            for tab in (qS, qX):
                tab.shadow.grad = torch.zeros_like(tab.shadow)
            qX.shadow.grad[node] = -0.01 * K[key].float()
            n = ctl.step(defer=defer)
            if n:
                return n, t
        return 0, steps

    def test_defer_stores_pending_without_committing(self):
        K, qS, qX, m, ctl = self._setup()
        w_before = qX.embedding.weight[3].detach().clone()
        n, _ = self._drive(K, qS, qX, ctl)
        assert n == 1 and len(ctl.pending) == 1
        assert ctl.pending[0]["gi"] == 3
        assert torch.equal(qX.embedding.weight[3], w_before)  # no commit
        assert ctl.n_commits == 0

    def test_finalize_accept_matches_legacy_commit(self):
        K, qS, qX, m, ctl = self._setup()
        n, _ = self._drive(K, qS, qX, ctl)
        entry = ctl.pending[0]
        ctl.finalize(entry, accepted=True)
        assert torch.allclose(qX.embedding.weight[3].double(), K[1])
        assert torch.allclose(qX.shadow[3].double(), K[1])   # re-centred
        assert m.data[3].item() == 0.0
        assert ctl.commit_counts[3] == 1

    def test_finalize_reject_taboos_and_keeps_shadow(self):
        K, qS, qX, m, ctl = self._setup()
        n, _ = self._drive(K, qS, qX, ctl)
        entry = ctl.pending[0]
        w_before = qX.embedding.weight[3].detach().clone()
        shadow_before = qX.shadow[3].detach().clone()
        assert not torch.allclose(shadow_before, w_before)  # evidence pending
        ctl.finalize(entry, accepted=False)
        assert frozenset({1}) in ctl.taboos[3]
        assert torch.equal(qX.embedding.weight[3], w_before)   # no write
        assert torch.equal(qX.shadow[3], shadow_before)        # no reset
        assert ctl.n_commits == 0
        assert ctl.last_rejects[0]["node"] == 3

    def test_taboo_blocks_same_subset_recandidacy(self):
        K, qS, qX, m, ctl = self._setup()
        self._drive(K, qS, qX, ctl)
        ctl.finalize(ctl.pending[0], accepted=False)
        # Keep pushing toward the SAME tabooed key: any new candidacy must
        # propose a DIFFERENT subset (or none).
        for t in range(200):
            for tab in (qS, qX):
                tab.shadow.grad = torch.zeros_like(tab.shadow)
            qX.shadow.grad[3] = -0.01 * K[1].float()
            n = ctl.step(defer=True)
            if n:
                for e in ctl.pending:
                    assert frozenset(e["cand"].tolist()) not in ctl.taboos[3]
                return
        # no re-candidacy at all is also compliant

    def test_taboo_lifts_when_shadow_moves_away(self):
        K, qS, qX, m, ctl = self._setup()
        # Committed at the global centroid; taboo {1}; shadow parked at key 2.
        ctl.taboos[3] = {frozenset({1})}
        with torch.no_grad():
            qX.shadow[3] = K[2].float()
        for tab in (qS, qX):
            tab.shadow.grad = torch.zeros_like(tab.shadow)
        ctl.step(defer=True)
        # raw projection {2} is neither committed nor {1}: taboo lifted.
        assert ctl.taboos[3] == set()

    def test_taboo_kept_while_shadow_points_at_it(self):
        K, qS, qX, m, ctl = self._setup()
        ctl.taboos[3] = {frozenset({1})}
        with torch.no_grad():
            qX.shadow[3] = K[1].float()
        for tab in (qS, qX):
            tab.shadow.grad = torch.zeros_like(tab.shadow)
        ctl.step(defer=True)
        assert ctl.taboos[3] == {frozenset({1})}

    def test_state_dict_roundtrip(self):
        K, qS, qX, m, ctl = self._setup()
        ctl.taboos[3] = {frozenset({1}), frozenset({0, 2})}
        ctl.n_commits = 5
        ctl.commit_counts[2] = 3
        sd = ctl.state_dict()
        _, _, _, _, ctl2 = self._setup()
        ctl2.load_state_dict(sd)
        assert ctl2.taboos[3] == {frozenset({1}), frozenset({0, 2})}
        assert ctl2.n_commits == 5
        assert int(ctl2.commit_counts[2]) == 3



class TestGateWiring:
    """Forecaster-level: _run_bilevel_gate resolves a pending candidacy."""

    def _gated_model(self):
        from test_bilevel_probe import _model, _qbatches
        m = _model()
        cfg = m.config["training"]
        cfg["centroid_commit"] = {
            "enabled": True, "evidence_lr": 10.0, "evidence_leak": 0.95,
            "shadow_source": "hsic", "reset_m_on_commit": "one",
            "bilevel_gate": {"enabled": True, "k_inner": 2,
                             "max_val_batches": 3},
        }
        # Homogeneous mode: both S and X carry query tables, so the controller
        # sees all N nodes and matches the combined key frame.
        cfg_full = m.config
        cfg_full["model"]["kwargs"]["homogeneous_nodes"] = True
        # Rebuild the forecaster so __init__ picks up the gate config.
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        torch.manual_seed(0)
        m2 = AttentionSelectorForecaster(cfg_full)
        m2.log = lambda *a, **k: None   # no Trainer attached in unit tests
        return m2, _qbatches

    def test_pending_candidacy_is_resolved(self):
        m, _qbatches = self._gated_model()
        assert m._gate_enabled and m._commit is not None
        m._val_probe_cache = _qbatches(3)
        entry = {"i": 0, "t": m._commit.tables[0], "r": 1, "gi": 0,
                 "cand": torch.tensor([0]), "cur_size": 3, "margin": 1.0}
        m._commit.pending = [entry]
        m._run_bilevel_gate()
        assert m._commit.pending == []
        # Resolved: either committed or tabooed, exactly one of the two.
        committed = int(m._commit.commit_counts[0]) == 1
        tabooed = len(m._commit.taboos[0]) == 1
        assert committed != tabooed

    def test_no_val_cache_drops_candidacy_without_taboo(self):
        m, _qbatches = self._gated_model()
        entry = {"i": 0, "t": m._commit.tables[0], "r": 1, "gi": 0,
                 "cand": torch.tensor([0]), "cur_size": 3, "margin": 1.0}
        m._commit.pending = [entry]
        m._run_bilevel_gate()
        assert m._commit.pending == []
        assert m._commit.taboos[0] == set()
        assert int(m._commit.commit_counts[0]) == 0

    def test_gate_requires_centroid_commit(self):
        from test_bilevel_probe import _model
        m = _model()
        cfg_full = m.config
        cfg_full["training"]["centroid_commit"] = {
            "enabled": False, "bilevel_gate": {"enabled": True},
        }
        from causaliT.training.forecasters.attention_selector_forecaster import (
            AttentionSelectorForecaster,
        )
        with pytest.raises(ValueError, match="bilevel_gate"):
            AttentionSelectorForecaster(cfg_full)

