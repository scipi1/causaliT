"""torchmetrics cross-version compatibility for the forecaster metrics.

``R2Score(num_outputs=...)`` was deprecated in torchmetrics 1.5.0 and REMOVED in
1.6.0, where the metric infers the output count from the input shape instead.
Passing the kwarg to >= 1.6 falls through to ``Metric.__init__`` and raises

    ValueError: Unexpected keyword arguments: `num_outputs`

which is a CRASH AT CONSTRUCTION -- the run dies before the first batch.  This
bit us on a cluster whose env carried a newer build than the ``==1.0.3`` pin in
requirements.txt.  ``AttentionSelectorForecaster`` therefore probes the
signature instead of the version string.

These tests pin that behaviour from BOTH sides: the installed version must work
(whatever it is), and the branch for the OTHER generation must work too -- the
one the local env cannot exercise is simulated with a stub.
"""

import inspect

import pytest
import torch
import torchmetrics as tm


# ---------------------------------------------------------------------------
# The guard itself: whatever is installed, the probe must agree with reality.
# ---------------------------------------------------------------------------

def test_probe_matches_installed_signature():
    """The signature probe must reflect the ACTUAL installed torchmetrics."""
    accepts = "num_outputs" in inspect.signature(tm.R2Score.__init__).parameters

    if accepts:
        # < 1.6: the kwarg is required to get per-output R2.
        metric = tm.R2Score(num_outputs=3, multioutput="uniform_average")
    else:
        # >= 1.6: the kwarg must NOT be passed; outputs come from the shape.
        with pytest.raises(ValueError, match="num_outputs"):
            tm.R2Score(num_outputs=3, multioutput="uniform_average")
        metric = tm.R2Score(multioutput="uniform_average")

    preds = torch.randn(16, 3)
    target = preds + 0.01 * torch.randn(16, 3)
    value = metric(preds, target)
    assert torch.isfinite(value), "macro R2 must be finite on a well-fit batch"


# ---------------------------------------------------------------------------
# The branch the installed version cannot exercise, via a stub.
# ---------------------------------------------------------------------------

class _R2Modern:
    """Stand-in for torchmetrics >= 1.6 R2Score (no ``num_outputs``)."""

    def __init__(self, adjusted: int = 0, multioutput: str = "uniform_average", **kwargs):
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        self.multioutput = multioutput


class _R2Legacy:
    """Stand-in for torchmetrics < 1.6 R2Score (``num_outputs`` required)."""

    def __init__(self, num_outputs: int = 1, adjusted: int = 0,
                 multioutput: str = "uniform_average", **kwargs):
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        self.num_outputs = num_outputs
        self.multioutput = multioutput


def _build_kwargs(r2_cls, x_seq_len: int) -> dict:
    """Mirror of the construction logic in AttentionSelectorForecaster."""
    kwargs = {"multioutput": "uniform_average"}
    if "num_outputs" in inspect.signature(r2_cls.__init__).parameters:
        kwargs["num_outputs"] = x_seq_len
    return kwargs


@pytest.mark.parametrize(
    "r2_cls, expect_num_outputs",
    [(_R2Legacy, True), (_R2Modern, False)],
    ids=["torchmetrics<1.6", "torchmetrics>=1.6"],
)
def test_construction_works_on_both_generations(r2_cls, expect_num_outputs):
    """The probe must build a working metric against either API."""
    kwargs = _build_kwargs(r2_cls, x_seq_len=5)
    assert ("num_outputs" in kwargs) is expect_num_outputs

    metric = r2_cls(**kwargs)          # must not raise
    assert metric.multioutput == "uniform_average"
    if expect_num_outputs:
        assert metric.num_outputs == 5


def test_forecaster_uses_the_probe_not_a_version_string():
    """Regression guard: the fix must be signature-based, not version-based.

    A ``torchmetrics.__version__`` comparison would silently mis-handle forks,
    release candidates and back-ports, so the source is checked to keep the
    probe honest.
    """
    from causaliT.training.forecasters import attention_selector_forecaster as mod

    src = inspect.getsource(mod)
    assert "inspect.signature(tm.R2Score.__init__)" in src, (
        "the R2Score compatibility branch must probe the signature"
    )
    assert "tm.__version__" not in src, (
        "do not gate the R2Score kwarg on a version string; probe the signature"
    )
