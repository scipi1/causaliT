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


def _trainer(epoch, sanity=False, max_epochs=None):
    return types.SimpleNamespace(sanity_checking=sanity, current_epoch=epoch,
                                 max_epochs=max_epochs)


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


class TestAncestorDiagnostics:
    # TRUE_SELF is the chain 2 -> 1 -> 0, so the only (ancestor, not parent)
    # cell is [0, 2].
    def test_perfect_attention_has_zero_ancestor_mass(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged["dag/mass_on_ancestors_self"] == 0.0
        assert mod._logged["dag/parent_vs_ancestor_contrast_self"] == 1.0

    def test_ancestor_shortcut_is_detected(self):
        att = np.zeros_like(TRUE_SELF)
        att[0, 2] = 1.0   # all mass on the transitive ancestor
        mod = _module(TRUE_CROSS.copy(), att)
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged["dag/mass_on_ancestors_self"] == 1.0
        assert mod._logged["dag/parent_vs_ancestor_contrast_self"] == -1.0

    def test_uniform_attention_has_zero_parent_ancestor_contrast(self):
        mod = _module(TRUE_CROSS.copy(), np.full_like(TRUE_SELF, 0.3))
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert mod._logged["dag/parent_vs_ancestor_contrast_self"] == \
            pytest.approx(0.0)

    def test_no_ancestor_metrics_for_rectangular_cross_block(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb().on_train_epoch_end(_trainer(epoch=0), mod)
        assert "dag/mass_on_ancestors_cross" not in mod._logged
        assert "dag/parent_vs_ancestor_contrast_cross" not in mod._logged

    def test_no_ancestor_metrics_when_true_closure_adds_nothing(self):
        # Ancestor masks derive from the TRUE mask; a star DAG (no chains)
        # has no (ancestor, not parent) cells, so no metrics are logged.
        star = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
        cb = _cb()
        cb._true_masks = {"cross": TRUE_CROSS, "self": star}
        mod = _module(TRUE_CROSS.copy(), star)
        cb.on_train_epoch_end(_trainer(epoch=0), mod)
        assert "dag/mass_on_ancestors_self" not in mod._logged


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

class TestFullColumnSet:
    """Every run must emit the complete dag/* column set, including the
    parents/ancestors/descendants/others mass partition."""

    EXPECTED = [
        "dag/auroc_{b}", "dag/contrast_{b}", "dag/mass_on_edges_{b}",
        "dag/mass_on_parents_{b}", "dag/mass_on_others_{b}",
    ]

    def test_all_columns_logged_for_both_blocks(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        for b in ("cross", "self"):
            for key in self.EXPECTED:
                assert key.format(b=b) in mod._logged, key.format(b=b)
        for key in ("dag/mass_on_ancestors_self",
                    "dag/mass_on_descendants_self",
                    "dag/parent_vs_ancestor_contrast_self"):
            assert key in mod._logged, key

    def test_parents_equals_edges(self):
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        for b in ("cross", "self"):
            assert mod._logged[f"dag/mass_on_parents_{b}"] == \
                mod._logged[f"dag/mass_on_edges_{b}"]

    def test_mass_partition_sums_to_one(self):
        """parents + ancestors + descendants + others + diagonal = 1 (self)."""
        rng = np.random.default_rng(0)
        mod = _module(TRUE_CROSS.copy(), rng.random(TRUE_SELF.shape))
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        att = mod.split_attention_blocks(None)["x_to_x"].numpy()
        diag = float(np.diag(att).sum() / att.sum())
        total = sum(mod._logged[f"dag/mass_on_{k}_self"]
                    for k in ("parents", "ancestors", "descendants", "others"))
        assert total + diag == pytest.approx(1.0)

    def test_others_is_non_edge_mass_for_cross_block(self):
        rng = np.random.default_rng(1)
        att = rng.random(TRUE_CROSS.shape)
        mod = _module(att, TRUE_SELF.copy())
        _cb(every_n_epochs=1).on_train_epoch_end(_trainer(epoch=0), mod)
        on = att[TRUE_CROSS > 0.5].sum() / att.sum()
        assert mod._logged["dag/mass_on_others_cross"] == \
            pytest.approx(1.0 - float(on))


class TestGuaranteedCoverage:
    """Epoch 0 and the final epoch always log, regardless of cadence."""

    def test_final_epoch_logged_despite_cadence(self):
        cb = _cb(every_n_epochs=50)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        cb.on_train_epoch_end(_trainer(epoch=9, max_epochs=10), mod)
        assert mod._logged, "final epoch must log even off-cadence"

    def test_mid_run_off_cadence_logs_nothing(self):
        cb = _cb(every_n_epochs=50)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        cb.on_train_epoch_end(_trainer(epoch=7, max_epochs=10), mod)
        assert mod._logged == {}

    def test_on_train_end_fallback_after_early_stop(self):
        cb = _cb(every_n_epochs=50)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        cb.on_train_end(_trainer(epoch=3, max_epochs=1000), mod)
        assert mod._logged, "on_train_end must guarantee a final log"

    def test_on_train_end_does_not_double_log(self):
        cb = _cb(every_n_epochs=1)
        mod = _module(TRUE_CROSS.copy(), TRUE_SELF.copy())
        logged_rows = []
        mod.log = lambda name, value, **kw: logged_rows.append(name)
        cb.on_train_epoch_end(_trainer(epoch=0, max_epochs=10), mod)
        cb.on_train_end(_trainer(epoch=0, max_epochs=10), mod)
        names = set(logged_rows)
        assert len(logged_rows) == len(names), "must not double-log epoch"
