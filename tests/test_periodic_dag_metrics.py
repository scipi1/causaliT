"""Fast unit tests for ``PeriodicDAGMetrics`` (stub-based, ~3 s, no training).

The callback reads the batch-mean attention posterior stashed on the module by
``_step`` (``_last_att_mean``).  That is the ONLY valid soft adjacency here:
phi/dag_mask are deprecated and ``batch_env_mean``/``batch_att_mean`` is never
assigned repo-wide, so the old ``evaluate_dag_from_model`` path returned None at
every epoch and produced no dag/* columns at all.  ``test_reads_attention_not_phi``
pins that we no longer depend on it.
"""
import types

import numpy as np
import pytest
import torch

from causaliT.training.callbacks import PeriodicDAGMetrics


TRUE_CROSS = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)   # (L_X=3, L_S=2)
TRUE_SELF = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


def _cb(every_n_epochs=2):
    cb = PeriodicDAGMetrics(
        config={"data": {"dataset": "dummy"}}, data_dir="/nonexistent",
        every_n_epochs=every_n_epochs,
    )
    # Bypass disk: the mask loader is exercised by the eval_utils tests.
    cb._true_masks = {"cross": TRUE_CROSS, "self": TRUE_SELF}
    return cb


def _module(att_cross=None, att_self=None, stash=True):
    """Module stub exposing ``_last_att_mean`` + ``split_attention_blocks``."""
    logged = {}
    mod = types.SimpleNamespace()
    mod.log = lambda name, value, **kw: logged.__setitem__(name, float(value))
    mod._logged = logged
    mod._last_att_mean = torch.zeros(3, 5) if stash else None

    def split(att):   # mirrors the layer's shape-aware splitter
        return {
            "s_to_x": None if att_cross is None else torch.as_tensor(att_cross),
            "x_to_x": None if att_self is None else torch.as_tensor(att_self),
        }
    mod.split_attention_blocks = split
    return mod


def _trainer(epoch, sanity=False):
    return types.SimpleNamespace(sanity_checking=sanity, current_epoch=epoch)


class TestSourceAndCadence:
    def test_reads_attention_not_phi(self, monkeypatch):
        """Must NOT call the dead phi path (it returns None at every epoch)."""
        def boom(*a, **k):
            raise AssertionError("evaluate_dag_from_model must not be called")
        monkeypatch.setattr(
            "causaliT.training.causal_initialization.evaluate_dag_from_model", boom
        )
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged, "must log from the attention posterior"

    def test_hook_is_train_epoch_end(self):
        cb = _cb()
        assert hasattr(cb, "on_train_epoch_end")
        assert "on_validation_epoch_end" not in type(cb).__dict__

    def test_logs_on_cadence_epochs_only(self):
        cb = _cb(every_n_epochs=2)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        cb.on_train_epoch_end(_trainer(epoch=1), mod)
        assert mod._logged == {}, "off-cadence epoch must log nothing"

        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        cb.on_train_epoch_end(_trainer(epoch=2), mod)
        assert mod._logged, "on-cadence epoch must log"

    def test_skips_sanity_check(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb().on_train_epoch_end(_trainer(epoch=0, sanity=True), mod)
        assert mod._logged == {}


class TestMetricValues:
    def test_perfect_ranking_gives_auroc_one(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)

        assert mod._logged["dag/auroc_self"] == 1.0
        assert mod._logged["dag/auroc_cross"] == 1.0
        assert mod._logged["dag/contrast_self"] == 1.0
        assert mod._logged["dag/mass_on_edges_self"] == 1.0

    def test_uniform_attention_is_chance_and_zero_contrast(self):
        """THE key property: flat attention must NOT look like structure."""
        mod = _module(np.full_like(TRUE_CROSS, 0.3),
                      np.full_like(TRUE_SELF, 0.3))
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)

        assert mod._logged["dag/auroc_self"] == pytest.approx(0.5)
        assert mod._logged["dag/contrast_self"] == pytest.approx(0.0)

    def test_inverted_attention_is_negative_contrast(self):
        """Attending to non-parents: AUROC < 0.5 and contrast < 0."""
        mod = _module(1.0 - TRUE_CROSS, 1.0 - TRUE_SELF)
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)

        assert mod._logged["dag/auroc_self"] == 0.0
        assert mod._logged["dag/contrast_self"] < 0.0

    def test_auroc_is_invariant_to_rescaling(self):
        """Why AUROC replaced SHD: no threshold, no min-max sensitivity."""
        small = 0.001 * TRUE_SELF + 0.5
        mod = _module(TRUE_CROSS.copy(), small)
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged["dag/auroc_self"] == 1.0

    def test_mass_on_edges_drops_when_dense(self):
        dense = TRUE_SELF + 0.5   # every non-edge gets mass too
        mod = _module(TRUE_CROSS.copy(), dense)
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert 0.0 < mod._logged["dag/mass_on_edges_self"] < 1.0


class TestFailureIsLoudNotSilent:
    def test_missing_posterior_warns_once_and_logs_nothing(self, caplog):
        """The exact failure that produced an empty metrics.csv."""
        cb = _cb(every_n_epochs=1)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy(), stash=False)
        with caplog.at_level("WARNING"):
            cb.on_train_epoch_end(_trainer(epoch=0), mod)
            cb.on_train_epoch_end(_trainer(epoch=1), mod)

        assert mod._logged == {}
        hits = [r for r in caplog.records if "_last_att_mean" in r.message]
        assert len(hits) == 1, "must warn exactly once, not per epoch"

    def test_split_error_does_not_break_training(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())

        def boom(att):
            raise RuntimeError("split exploded")
        mod.split_attention_blocks = boom
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged == {}   # swallowed, training continues

    def test_shape_mismatch_is_skipped_and_warns(self, caplog):
        mod = _module(np.ones((5, 5)), np.ones((5, 5)))   # wrong shapes
        with caplog.at_level("WARNING"):
            _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged == {}
        assert any("shape mismatch" in r.message for r in caplog.records)

    def test_batched_block_is_accepted(self):
        """Splitter may return (1, L, L); the callback must squeeze it."""
        mod = _module(TRUE_CROSS[None, ...].copy(), TRUE_SELF[None, ...].copy())
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged["dag/auroc_self"] == 1.0
