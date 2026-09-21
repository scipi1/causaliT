"""Unit tests for HSICClassMetrics (pre-weighting hsic_class/* diagnostics)."""
import types

import numpy as np
import pytest
import torch

from causaliT.training.callbacks.model_callbacks import HSICClassMetrics


# Two disconnected edges 0 -> 1 and 2 -> 3 (true[i, j] = 1 means j -> i),
# so the parents/ancestors/descendants/others partition is fully populated.
TRUE_SELF = np.array(
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 1, 0]], dtype=float,
)
TRUE_CROSS = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float)  # (4, 2)

L_S = 2


def _cb():
    cb = HSICClassMetrics(config={"data": {"dataset": "dummy"}},
                          data_dir="/nonexistent", every_n_epochs=1)
    cb._true_masks = {"cross": TRUE_CROSS, "self": TRUE_SELF}
    return cb


def _module(mat, homogeneous=False, l_s=L_S):
    mod = types.SimpleNamespace()
    mod._last_hsic_pair_mat = torch.as_tensor(mat, dtype=torch.float32)
    mod.homogeneous_nodes = homogeneous
    mod.S_seq_len = l_s
    logged = {}
    mod.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    mod._logged = logged
    return mod


def _trainer(epoch=0, sanity=False, max_epochs=None):
    return types.SimpleNamespace(sanity_checking=sanity, current_epoch=epoch,
                                 max_epochs=max_epochs)


def _split_mat(rng):
    """(n_X=4, L_S + L_X = 6) pair matrix, distinct values per cell."""
    return rng.random((4, 6)) + 0.1


def _hom_mat(rng):
    """(N=6, N=6) homogeneous pair matrix."""
    return rng.random((6, 6)) + 0.1


class TestBlockExtraction:
    def test_split_mode_class_means(self):
        rng = np.random.default_rng(0)
        mat = _split_mat(rng)
        mod = _module(mat, homogeneous=False)
        _cb().on_train_epoch_end(_trainer(), mod)

        cross = mat[:, :L_S]
        selfb = mat[:, L_S:]
        fin = np.isfinite(cross)
        assert mod._logged["hsic_class/parents_cross"] == pytest.approx(
            cross[(TRUE_CROSS > 0.5) & fin].mean())
        assert mod._logged["hsic_class/others_cross"] == pytest.approx(
            cross[(TRUE_CROSS <= 0.5) & fin].mean())

        parents = TRUE_SELF > 0.5                     # (1,0), (3,2)
        dsc = parents.T                               # (0,1), (2,3)
        others = ~(parents | dsc) & ~np.eye(4, dtype=bool)
        for cls, m in (("parents", parents),
                       ("descendants", dsc), ("others", others)):
            assert mod._logged[f"hsic_class/{cls}_self"] == pytest.approx(
                selfb[m].mean()), cls
        # No chains in this DAG: the ancestors class is empty -> absent.
        assert "hsic_class/ancestors_self" not in mod._logged

    def test_homogeneous_mode_class_means(self):
        rng = np.random.default_rng(1)
        mat = _hom_mat(rng)
        mod = _module(mat, homogeneous=True)
        _cb().on_train_epoch_end(_trainer(), mod)
        cross = mat[L_S:, :L_S]
        selfb = mat[L_S:, L_S:]
        assert mod._logged["hsic_class/parents_cross"] == pytest.approx(
            cross[TRUE_CROSS > 0.5].mean())
        assert mod._logged["hsic_class/parents_self"] == pytest.approx(
            selfb[TRUE_SELF > 0.5].mean())

    def test_counts_logged_once(self):
        rng = np.random.default_rng(2)
        mod = _module(_split_mat(rng))
        cb = _cb()
        cb.on_train_epoch_end(_trainer(epoch=0), mod)
        assert "hsic_class/n_parents_self" in mod._logged
        n_logged = sum(1 for k in mod._logged if k.startswith("hsic_class/n_"))
        mod._logged.clear()
        cb.on_train_epoch_end(_trainer(epoch=1), mod)
        assert sum(1 for k in mod._logged
                   if k.startswith("hsic_class/n_")) == 0
        assert n_logged > 0

class TestNaNHandling:
    def test_nan_cells_excluded(self):
        rng = np.random.default_rng(3)
        mat = _split_mat(rng)
        mat[0, L_S + 0] = float("nan")   # poison one self cell
        mod = _module(mat)
        _cb().on_train_epoch_end(_trainer(), mod)
        selfb = mat[:, L_S:]
        parents = TRUE_SELF > 0.5
        sel = parents & np.isfinite(selfb)
        assert mod._logged["hsic_class/parents_self"] == pytest.approx(
            selfb[sel].mean())

    def test_all_nan_class_absent(self):
        rng = np.random.default_rng(4)
        mat = _split_mat(rng)
        parents = TRUE_SELF > 0.5
        selfb = mat[:, L_S:]
        selfb[parents] = float("nan")
        mat[:, L_S:] = selfb
        mod = _module(mat)
        _cb().on_train_epoch_end(_trainer(), mod)
        assert "hsic_class/parents_self" not in mod._logged
        assert "hsic_class/others_self" in mod._logged


class TestGuards:
    def test_missing_matrix_warns_once(self, caplog):
        mod = types.SimpleNamespace(_last_hsic_pair_mat=None,
                                    homogeneous_nodes=False, S_seq_len=L_S)
        mod.log = lambda *a, **k: None
        cb = _cb()
        with caplog.at_level("WARNING"):
            cb.on_train_epoch_end(_trainer(epoch=0), mod)
            cb.on_train_epoch_end(_trainer(epoch=1), mod)
        hits = [r for r in caplog.records if "_last_hsic_pair_mat" in r.message]
        assert len(hits) == 1

    def test_cadence_respected_and_final_epoch(self):
        rng = np.random.default_rng(5)
        cb = HSICClassMetrics(config={"data": {"dataset": "d"}},
                              data_dir="/x", every_n_epochs=50)
        cb._true_masks = {"cross": TRUE_CROSS, "self": TRUE_SELF}
        mod = _module(_split_mat(rng))
        cb.on_train_epoch_end(_trainer(epoch=7, max_epochs=10), mod)
        assert mod._logged == {}
        cb.on_train_epoch_end(_trainer(epoch=9, max_epochs=10), mod)
        assert mod._logged, "final epoch must log off-cadence"

    def test_early_stop_fallback_once(self):
        rng = np.random.default_rng(6)
        cb = HSICClassMetrics(config={"data": {"dataset": "d"}},
                              data_dir="/x", every_n_epochs=50)
        cb._true_masks = {"cross": TRUE_CROSS, "self": TRUE_SELF}
        mod = _module(_split_mat(rng))
        cb.on_train_end(_trainer(epoch=3, max_epochs=1000), mod)
        assert mod._logged
        n = len(mod._logged)
        cb.on_train_end(_trainer(epoch=3, max_epochs=1000), mod)
        assert len(mod._logged) == n

    def test_sanity_check_skipped(self):
        rng = np.random.default_rng(7)
        mod = _module(_split_mat(rng))
        _cb().on_train_epoch_end(_trainer(epoch=0, sanity=True), mod)
        assert mod._logged == {}
