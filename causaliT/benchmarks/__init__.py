"""
External structure-learning benchmarks for causaliT.

Five reference methods, all fitted with their papers' own hyperparameters:

===================  =====================================================
``notears_linear``   NOTEARS, linear SEM            (vendored paper code)
``notears_mlp``      NOTEARS, per-node MLP          (vendored paper code)
``dagma_linear``     DAGMA, linear SEM              (``dagma`` package)
``dagma_mlp``        DAGMA, per-node MLP            (``dagma`` package)
``pc``               PC, Fisher-z, PC-stable        (``causal-learn`` package)
===================  =====================================================

Module map::

    data.py         npz token tensors -> one (n_samples, N) design matrix,
                    columns ordered [S..., X...]
    base.py         BenchmarkResult, the method registry, paper defaults
    methods/        one thin wrapper per method (lazy third-party imports)
    postprocess.py  paper adjacency -> canonical cross/self DAG blocks
    runner.py       fit + write_dag_report + benchmark_run.json
    vendor/         verbatim third-party source (see each PROVENANCE.md)

The benchmarks share causaliT's evaluation path: ``runner`` calls the same
``write_dag_report`` as ``eval_attention_scores``, so results land in
``<experiment>/eval/eval_benchmark_<method>/files/dag_metrics.json`` with an
identical schema and are directly comparable.

Typical use::

    from causaliT.benchmarks import run_benchmarks
    results = run_benchmarks("experiments/.../my_run")

or from the command line::

    python -m causaliT.benchmarks.cli list
    python -m causaliT.benchmarks.cli run --experiment <path> --methods pc

Only ``base`` and ``runner`` are re-exported here; importing this package pulls
in neither torch nor the optional third-party solvers.
"""

from causaliT.benchmarks.base import (
    BenchmarkResult,
    available_methods,
    default_params,
    is_available,
    method_names,
    resolve_method,
)

__all__ = [
    "BenchmarkResult",
    "available_methods",
    "default_params",
    "is_available",
    "method_names",
    "resolve_method",
    "run_benchmark_method",
    "run_benchmarks",
    "summarize_benchmarks",
]


def __getattr__(name):
    """
    Lazily expose the runner entry points.

    ``runner`` imports the evaluation stack (matplotlib, omegaconf, scm_ds), so
    deferring it keeps ``import causaliT.benchmarks`` cheap for callers that only
    need the registry - e.g. the CLI's ``list`` command.
    """
    if name in ("run_benchmark_method", "run_benchmarks", "summarize_benchmarks"):
        from causaliT.benchmarks import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
