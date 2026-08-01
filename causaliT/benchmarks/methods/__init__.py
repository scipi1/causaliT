"""
Benchmark method wrappers.

One module per method, each exposing exactly two names:

    ``DEFAULT_PARAMS``  the paper's hyperparameters, frozen
    ``fit(X, **params) -> BenchmarkResult``

The wrappers are intentionally thin: they translate causaliT's parameters into
the reference implementation's call signature, time the fit, and return the raw
weighted adjacency in paper orientation (``W[i, j] != 0`` means ``i -> j``).
No thresholding, no re-orientation, no metric computation happens here - that
belongs to ``postprocess.py`` and ``runner.py``.

Third-party imports (``torch``, ``dagma``, ``causallearn``) live *inside* these
modules, so a missing optional dependency disables only its own method (see
``base.available_methods``).
"""
