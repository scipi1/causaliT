"""Unit tests for the node-wise (per-query) winner-take-all structural update.

Covers the mechanics of causaliT/training/nodewise_update.py in isolation:
row masking + snapshot/restore must make the update of non-selected nodes a
no-op under ANY optimizer (incl. AdamW decoupled weight decay, which acts on
zero-grad rows), the SNR selector must prefer consistent gradients over noise,
the significance gate must fire under pure noise, and topk = n_nodes must
reproduce the vanilla optimizer trajectory.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaliT.training.nodewise_update import NodewiseQuerySelector

D = 8
N_S, N_X = 2, 3          # 2 + 3 = 5 nodes
N = N_S + N_X


def _make_params():
    """Query tables with a padding row 0 each + a tied norm vector."""
    torch.manual_seed(0)
    qS = torch.nn.Parameter(torch.randn(1 + N_S, D))
    qX = torch.nn.Parameter(torch.randn(1 + N_X, D))
    m = torch.nn.Parameter(torch.zeros(N))
    return qS, qX, m


def _make_selector(qS, qX, m, topk=1):
    return NodewiseQuerySelector(query_params=[qS, qX], norm_param=m, topk=topk)


class TestInit:
    def test_node_map_skips_padding_rows(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        assert nw.n_nodes == N
        assert [r for _, r in nw.node_map] == [1, 2, 1, 2, 3]

    def test_norm_shape_mismatch_raises(self):
        qS, qX, m = _make_params()
        bad = torch.nn.Parameter(torch.zeros(N + 1))
        with pytest.raises(ValueError, match="rows"):
            _make_selector(qS, qX, bad)

    def test_topk_above_n_nodes_raises(self):
        qS, qX, m = _make_params()
        with pytest.raises(ValueError, match="topk"):
            _make_selector(qS, qX, m, topk=N + 1)



class TestMaskSnapshotRestore:
    @pytest.mark.parametrize("opt_cls,kwargs", [
        (torch.optim.AdamW, {"lr": 0.1, "weight_decay": 0.1}),
        (torch.optim.Adam, {"lr": 0.1, "weight_decay": 0.1}),
        (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.1}),
    ])
    def test_non_selected_rows_bit_identical(self, opt_cls, kwargs):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        opt = opt_cls([qS, qX, m], **kwargs)
        before = {id(p): p.data.clone() for p in (qS, qX, m)}

        selected = [2]     # node 2 = (qX, row 1)
        torch.manual_seed(1)
        for _ in range(50):  # warm up + run with momentum/decay active
            for p in (qS, qX, m):
                p.grad = torch.randn_like(p)
            nw.mask_grads(selected)
            snap = nw.snapshot(opt, selected)
            opt.step()
            NodewiseQuerySelector.restore(snap)

        for i, (p, r) in enumerate(nw.node_map):
            if i in selected:
                continue
            assert torch.equal(p.data[r], before[id(p)][r]), (
                f"non-selected row drifted under {opt_cls.__name__}"
            )
        # padding rows are not nodes: they must be restored too
        assert torch.equal(qS.data[0], before[id(qS)][0])
        assert torch.equal(qX.data[0], before[id(qX)][0])
        # the norm rows of non-selected nodes are frozen as well
        for i in range(N):
            if i not in selected:
                assert m.data[i] == before[id(m)][i]
        # the selected node DID move
        assert not torch.equal(qX.data[1], before[id(qX)][1])

    def test_selected_row_state_not_restored(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        opt = torch.optim.AdamW([qS, qX, m], lr=0.1)
        qS.grad = torch.randn_like(qS); qX.grad = torch.randn_like(qX)
        m.grad = torch.randn_like(m)
        selected = [0]
        nw.mask_grads(selected)
        snap = nw.snapshot(opt, selected)
        opt.step()
        st_before = opt.state[qS]["exp_avg"].clone()
        NodewiseQuerySelector.restore(snap)
        # selected row keeps the optimizer-written state...
        assert torch.equal(opt.state[qS]["exp_avg"][1], st_before[1])
        # ...non-selected rows keep zero-init state
        assert torch.all(opt.state[qS]["exp_avg"][2] == 0)


class TestSelection:
    def test_consistent_gradient_wins(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        torch.manual_seed(2)
        signal = torch.randn(D)
        wins = 0
        for t in range(200):
            for p in (qS, qX, m):
                p.grad = 10.0 * torch.randn_like(p)   # loud noise everywhere
            qX.grad[2] = signal + 0.1 * torch.randn(D)  # node 3: consistent
            sel = nw.select()
            if t >= 50 and sel is not None and len(sel):
                wins += int(sel[0] == 3)
        assert wins >= 140, f"node 3 won only {wins}/150 post-warmup steps"

    def test_gate_fires_under_pure_noise(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        torch.manual_seed(3)
        fired = 0
        for _ in range(200):
            for p in (qS, qX, m):
                p.grad = torch.randn_like(p)
            sel = nw.select()
            fired += int(sel == [])
        assert fired > 150, f"gate fired only {fired}/200 under pure noise"

    def test_no_grads_returns_none(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        assert nw.select() is None   # recon phase: grads are None

    def test_reset_clears_evidence(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        torch.manual_seed(4)
        for p in (qS, qX, m):
            p.grad = torch.randn_like(p)
        nw.select()
        assert nw.t == 1
        nw.reset_stats()
        assert nw.t == 0 and torch.all(nw.ema_mean == 0)

    def test_topk_all_matches_vanilla_adamw(self):
        # With topk = n_nodes and consistent gradients everywhere, every node
        # passes the gate and the trajectory must equal vanilla AdamW.
        qS, qX, m = _make_params()
        qS2, qX2, m2 = _make_params()
        nw = _make_selector(qS, qX, m, topk=N)
        opt_a = torch.optim.AdamW([qS, qX, m], lr=0.01, weight_decay=0.01)
        opt_b = torch.optim.AdamW([qS2, qX2, m2], lr=0.01, weight_decay=0.01)
        torch.manual_seed(5)
        grads = [torch.randn_like(p) for p in (qS, qX, m)]
        for _ in range(30):
            for p, g in zip((qS, qX, m), grads):
                p.grad = g.clone()
            for p, g in zip((qS2, qX2, m2), grads):
                p.grad = g.clone()
            sel = nw.select()
            assert sel is not None and len(sel) == N
            opt_a.step(); opt_b.step()
        # rows 1: (real nodes) match vanilla exactly; row 0 is the padding row,
        # which the selector deliberately freezes (vanilla AdamW would decay it)
        assert torch.allclose(qS.data[1:], qS2.data[1:])
        assert torch.allclose(qX.data[1:], qX2.data[1:])
        assert torch.allclose(m.data, m2.data)

    def test_norm_row_moves_with_its_query(self):
        qS, qX, m = _make_params()
        nw = _make_selector(qS, qX, m)
        opt = torch.optim.SGD([qS, qX, m], lr=0.5)
        for p in (qS, qX, m):
            p.grad = torch.randn_like(p)
        before = m.data.clone()
        nw.mask_grads([4])
        snap = nw.snapshot(opt, [4])
        opt.step()
        NodewiseQuerySelector.restore(snap)
        assert m.data[4] != before[4]
        assert torch.equal(m.data[:4], before[:4])

