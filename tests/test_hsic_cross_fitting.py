"""HSIC cross-fitting: MSE on fold A, independence statistic on fold B.

The point is that HSIC measured on the very rows the regressor just fitted is
optimistically biased -- the fit absorbs sample-specific noise, so the residual
looks more independent than it is.  Fold B is therefore a DISJOINT, PERMANENT
subset.

"Permanent" is the load-bearing word and the reason this is not a batch slice:
``train_dataloader`` uses ``shuffle=True``, so splitting by POSITION would
reassign every sample to a different fold each epoch and the separation would
dissolve.  ``test_membership_survives_shuffling`` pins that.
"""
import numpy as np
import pytest
import torch

from causaliT.training.dataloader import ProcessDataModule


def _dm(n=64, batch_size=8):
    dm = ProcessDataModule.__new__(ProcessDataModule)      # no file I/O
    dm.batch_size = batch_size
    dm.num_workers = 0
    dm.persistent_workers = False
    dm._xfit_ds_a = None
    dm._xfit_ds_b = None
    x = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).repeat(1, 3, 2)
    y = torch.zeros(n, 3, 2)
    from torch.utils.data.dataset import TensorDataset
    dm.train_ds = TensorDataset(x, y)
    return dm


def _ids(loader):
    """Sample identities (encoded in the tensor values) seen by a loader."""
    out = []
    for xb, _ in loader:
        out.extend(xb[:, 0, 0].tolist())
    return sorted(out)


class TestPartition:
    def test_folds_are_disjoint_and_cover_the_training_set(self):
        dm = _dm(n=64)
        n_a, n_b = dm.set_hsic_cross_fit(ratio=0.5, seed=0)
        assert n_a + n_b == 64

        a, b = dm.hsic_cross_fit_dataloaders()
        ids_a, ids_b = set(_ids(a)), set(_ids(b))
        assert ids_a.isdisjoint(ids_b), "folds must share no sample"
        assert ids_a | ids_b == set(range(64)), "folds must cover the set"

    def test_ratio_controls_the_split(self):
        dm = _dm(n=100)
        n_a, n_b = dm.set_hsic_cross_fit(ratio=0.8, seed=0)
        assert (n_a, n_b) == (80, 20)

    def test_membership_survives_shuffling(self):
        """THE property: shuffling reorders WITHIN a fold, never across."""
        dm = _dm(n=64)
        dm.set_hsic_cross_fit(ratio=0.5, seed=0)
        a, b = dm.hsic_cross_fit_dataloaders()
        # Three passes = three different shuffles of each loader.
        assert _ids(a) == _ids(a) == _ids(a)
        assert _ids(b) == _ids(b) == _ids(b)
        assert set(_ids(a)).isdisjoint(set(_ids(b)))

    def test_is_deterministic_given_the_seed(self):
        d1, d2 = _dm(n=64), _dm(n=64)
        d1.set_hsic_cross_fit(ratio=0.5, seed=7)
        d2.set_hsic_cross_fit(ratio=0.5, seed=7)
        assert _ids(d1.hsic_cross_fit_dataloaders()[0]) == \
               _ids(d2.hsic_cross_fit_dataloaders()[0])

    def test_different_seeds_give_different_partitions(self):
        d1, d2 = _dm(n=64), _dm(n=64)
        d1.set_hsic_cross_fit(ratio=0.5, seed=1)
        d2.set_hsic_cross_fit(ratio=0.5, seed=2)
        assert _ids(d1.hsic_cross_fit_dataloaders()[0]) != \
               _ids(d2.hsic_cross_fit_dataloaders()[0])


class TestTrainLoaderRedirect:
    def test_train_loader_serves_fold_a_when_enabled(self):
        dm = _dm(n=64)
        dm.set_hsic_cross_fit(ratio=0.5, seed=0)
        a, _ = dm.hsic_cross_fit_dataloaders()
        assert _ids(dm.train_dataloader()) == _ids(a), (
            "the Lightning train loader must serve fold A (the MSE fold)"
        )

    def test_train_loader_unchanged_when_disabled(self):
        dm = _dm(n=64)
        assert _ids(dm.train_dataloader()) == list(range(64))

    def test_dataloaders_none_when_disabled(self):
        assert _dm(n=64).hsic_cross_fit_dataloaders() is None


class TestGuards:
    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_ratio_raises(self, ratio):
        with pytest.raises(ValueError, match="ratio must be in"):
            _dm(n=64).set_hsic_cross_fit(ratio=ratio)

    def test_requires_setup(self):
        dm = _dm(n=64)
        dm.train_ds = None
        with pytest.raises(RuntimeError, match="requires train_ds"):
            dm.set_hsic_cross_fit()

    def test_extreme_ratio_keeps_both_folds_non_empty(self):
        dm = _dm(n=10)
        n_a, n_b = dm.set_hsic_cross_fit(ratio=0.001, seed=0)
        assert n_a >= 1 and n_b >= 1
