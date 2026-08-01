"""
Click CLI for the benchmarks (same style as ``causaliT/cli.py``).

    python -m causaliT.benchmarks.cli list
    python -m causaliT.benchmarks.cli run --experiment <path> --methods pc,dagma_linear

``list`` prints the registry with the availability of each optional dependency
and the paper hyperparameters that will be used - handy for checking an install
without touching data.

``run`` fits the methods on the dataset of an existing experiment folder and
writes the artefacts under ``<experiment>/eval/eval_benchmark_<method>/``.  Only
the config (dataset name, DAG dims) and the folder path are used - no checkpoint
is loaded - so a config-only folder is enough to benchmark a dataset.

Options left unset fall back to the ``benchmark`` section of the experiment
config, then to ``runner.DEFAULT_BENCHMARK_CONFIG``.  Comma-separated lists are
used for ``--methods`` and ``--seeds`` so the CLI composes cleanly inside the
shell scripts under ``scripts/``.
"""

# Standard library imports
import csv
import json
import sys
from typing import List, Optional

# Third-party imports
import click


def _split_list(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated option into a list (``None`` stays ``None``)."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


@click.group()
def cli():
    """Structure-learning benchmarks (NOTEARS, DAGMA, PC)."""
    pass


# LIST METHODS
@click.command(name="list")
@click.option("--as_json", is_flag=True, default=False, help="Machine-readable output")
def list_methods(as_json):
    """Show the registered methods, their availability and paper defaults."""
    from causaliT.benchmarks.base import (
        METHOD_DESCRIPTIONS,
        available_methods,
        default_params,
        method_names,
    )

    availability = available_methods()

    if as_json:
        payload = {
            name: {
                "available": availability[name],
                "description": METHOD_DESCRIPTIONS.get(name, ""),
                "default_params": default_params(name) if availability[name] else None,
            }
            for name in method_names()
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    click.echo("Available benchmark methods:\n")
    for name in method_names():
        mark = "OK " if availability[name] else "N/A"
        click.echo(f"  [{mark}] {name:16s} {METHOD_DESCRIPTIONS.get(name, '')}")
        if availability[name]:
            rendered = ", ".join(f"{k}={v}" for k, v in default_params(name).items())
            click.echo(f"        paper defaults: {rendered}")
        else:
            click.echo("        missing optional dependency (see requirements.txt)")


# RUN BENCHMARKS
@click.command()
@click.option("--experiment", required=True, help="Experiment folder with config*.yaml (also the output location)")
@click.option("--methods", default=None, help="Comma-separated methods (default: config benchmark.methods)")
@click.option("--seeds", default=None, help="Comma-separated seeds; one fit each, stored as pseudo-folds seed_<i>")
@click.option("--split", default=None, help="Data split to fit on (default: train)")
@click.option("--standardize/--no_standardize", default=None, help="Z-score the columns before fitting")
@click.option("--max_samples", type=int, default=None, help="Cap on the number of rows")
@click.option("--w_threshold", type=float, default=None, help="Magnitude threshold on |W| (papers use 0.3)")
@click.option("--score_mode", type=click.Choice(["binary", "scaled"]), default=None, help="Binary edges or magnitude-scaled confidences")
@click.option("--forbid_into_sources", is_flag=True, default=False, help="Background knowledge: drop edges into source variables")
@click.option("--csv_out", default=None, help="Also write the summary table to this CSV")
@click.option("--quiet", is_flag=True, default=False, help="Less progress output")
def run(experiment, methods, seeds, split, standardize, max_samples, w_threshold,
        score_mode, forbid_into_sources, csv_out, quiet):
    """Fit the configured benchmarks on an experiment's dataset."""
    from causaliT.benchmarks.runner import run_benchmarks, summarize_benchmarks

    seed_list = _split_list(seeds)
    overrides = {
        "seeds": [int(s) for s in seed_list] if seed_list else None,
        "split": split,
        "standardize": standardize,
        "max_samples": max_samples,
        "w_threshold": w_threshold,
        "score_mode": score_mode,
        "forbid_into_sources": True if forbid_into_sources else None,
    }

    results = run_benchmarks(
        experiment=experiment,
        methods=_split_list(methods),
        overrides=overrides,
        verbose=not quiet,
    )
    rows = summarize_benchmarks(results)

    click.echo("\n=== Benchmark summary ===")
    for row in rows:
        if "error" in row:
            click.echo(f"  {row['method']:16s} FAILED: {row['error']}")
            continue
        click.echo(
            f"  {row['method']:16s} "
            f"SHD cross={row.get('shd_cross_mean')}, self={row.get('shd_self_mean')}, "
            f"soft cross={row.get('soft_hamming_cross_mean')}, "
            f"time={row.get('seconds_mean')}"
        )

    if csv_out and rows:
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        click.echo(f"\nWrote {csv_out}")

    # Non-zero exit code if any method failed, so shell scripts can react.
    if any("error" in row for row in rows):
        sys.exit(1)


cli.add_command(list_methods)
cli.add_command(run)


if __name__ == "__main__":
    cli()
