"""``training.hsic_aggregation`` selector resolution + legacy alias.

Constructs the REAL forecaster (no duplicated logic): guards the backward
compatibility of every existing config (HSIC_OPT..HSIC_OPT_3, the four joint
arms), all of which set the legacy boolean and none of which set the enum.
"""
import pytest
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from test_dropout_selection import _make_forecaster_config


def _resolve(**training):
    """Build a real forecaster with the given training overrides."""
    torch.manual_seed(0)
    cfg = _make_forecaster_config()
    cfg["data"]["val_idx"] = 0
    for embed in ("ds_embed_S", "ds_embed_X"):
        for mod in cfg["model"]["kwargs"][embed]["modules"]:
            mod["idx"] = 1 if mod["label"] == "variable" else 0
            mod["role"] = "structure" if mod["label"] == "variable" else "value"
    cfg["model"]["kwargs"]["comps_embed_S"] = "svfa"
    cfg["model"]["kwargs"]["comps_embed_X"] = "svfa"
    cfg["training"].pop("use_attention_weighted_hsic", None)
    cfg["training"].update(training)
    return AttentionSelectorForecaster(cfg)


class TestLegacyAlias:
    def test_absent_defaults_to_plain(self):
        o = _resolve()
        assert o.hsic_aggregation == "plain"
        assert not o.use_attention_weighted_hsic
        assert not o.hsic_weight_descendants_only

    def test_legacy_true_maps_to_attw(self):
        o = _resolve(use_attention_weighted_hsic=True)
        assert o.hsic_aggregation == "attw"
        assert o.use_attention_weighted_hsic
        assert not o.hsic_weight_descendants_only

    def test_legacy_false_maps_to_plain(self):
        o = _resolve(use_attention_weighted_hsic=False)
        assert o.hsic_aggregation == "plain"
        assert not o.use_attention_weighted_hsic


class TestEnum:
    @pytest.mark.parametrize("agg,weighted,desc_only", [
        ("plain", False, False),
        ("attw", True, False),
        ("attw_descendants", True, True),
    ])
    def test_each_variant(self, agg, weighted, desc_only):
        o = _resolve(hsic_aggregation=agg)
        assert o.hsic_aggregation == agg
        assert o.use_attention_weighted_hsic is weighted
        assert o.hsic_weight_descendants_only is desc_only

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="hsic_aggregation must be one of"):
            _resolve(hsic_aggregation="max")


class TestConflicts:
    def test_agreeing_settings_are_accepted(self):
        assert _resolve(hsic_aggregation="attw",
                        use_attention_weighted_hsic=True).hsic_aggregation == "attw"
        assert _resolve(hsic_aggregation="plain",
                        use_attention_weighted_hsic=False).hsic_aggregation == "plain"

    def test_hybrid_with_legacy_true_is_allowed(self):
        """attw_descendants IS attention-weighted; the alias does not contradict."""
        o = _resolve(hsic_aggregation="attw_descendants",
                     use_attention_weighted_hsic=True)
        assert o.hsic_weight_descendants_only

    @pytest.mark.parametrize("agg,legacy", [
        ("plain", True),
        ("attw", False),
        ("attw_descendants", False),
    ])
    def test_disagreeing_settings_raise(self, agg, legacy):
        with pytest.raises(ValueError, match="Conflicting"):
            _resolve(hsic_aggregation=agg, use_attention_weighted_hsic=legacy)



class TestLegacyAlias:
    def test_absent_defaults_to_plain(self):
        o = _resolve()
        assert o.hsic_aggregation == "plain"
        assert not o.use_attention_weighted_hsic
        assert not o.hsic_weight_descendants_only

    def test_legacy_true_maps_to_attw(self):
        o = _resolve(use_attention_weighted_hsic=True)
        assert o.hsic_aggregation == "attw"
        assert o.use_attention_weighted_hsic
        assert not o.hsic_weight_descendants_only

    def test_legacy_false_maps_to_plain(self):
        o = _resolve(use_attention_weighted_hsic=False)
        assert o.hsic_aggregation == "plain"
        assert not o.use_attention_weighted_hsic


class TestEnum:
    @pytest.mark.parametrize("agg,weighted,desc_only", [
        ("plain", False, False),
        ("attw", True, False),
        ("attw_descendants", True, True),
    ])
    def test_each_variant(self, agg, weighted, desc_only):
        o = _resolve(hsic_aggregation=agg)
        assert o.hsic_aggregation == agg
        assert o.use_attention_weighted_hsic is weighted
        assert o.hsic_weight_descendants_only is desc_only

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="hsic_aggregation must be one of"):
            _resolve(hsic_aggregation="max")


class TestConflicts:
    def test_agreeing_settings_are_accepted(self):
        assert _resolve(hsic_aggregation="attw",
                        use_attention_weighted_hsic=True).hsic_aggregation == "attw"
        assert _resolve(hsic_aggregation="plain",
                        use_attention_weighted_hsic=False).hsic_aggregation == "plain"

    def test_hybrid_with_legacy_true_is_allowed(self):
        """attw_descendants IS attention-weighted; the alias does not contradict."""
        o = _resolve(hsic_aggregation="attw_descendants",
                     use_attention_weighted_hsic=True)
        assert o.hsic_weight_descendants_only

    @pytest.mark.parametrize("agg,legacy", [
        ("plain", True),
        ("attw", False),
        ("attw_descendants", False),
    ])
    def test_disagreeing_settings_raise(self, agg, legacy):
        with pytest.raises(ValueError, match="Conflicting"):
            _resolve(hsic_aggregation=agg, use_attention_weighted_hsic=legacy)

class TestSoftmaxVariant:
    def test_enum_resolves(self):
        o = _resolve(hsic_aggregation="attw_softmax")
        assert o.hsic_aggregation == "attw_softmax"
        assert o.use_attention_weighted_hsic
        assert not o.hsic_weight_descendants_only
        assert o.hsic_softmax

    def test_other_variants_not_softmax(self):
        assert not _resolve(hsic_aggregation="attw").hsic_softmax
        assert not _resolve(hsic_aggregation="attw_descendants").hsic_softmax
        assert not _resolve(hsic_aggregation="plain").hsic_softmax

    def test_softmax_with_legacy_true_is_allowed(self):
        o = _resolve(hsic_aggregation="attw_softmax",
                     use_attention_weighted_hsic=True)
        assert o.hsic_softmax

    def test_softmax_with_legacy_false_raises(self):
        with pytest.raises(ValueError, match="Conflicting"):
            _resolve(hsic_aggregation="attw_softmax",
                     use_attention_weighted_hsic=False)
