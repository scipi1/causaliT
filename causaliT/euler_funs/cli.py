"""Run lifecycle: launch, submit, resume, status, cancel, finalize, export."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import uuid

import click

from . import slurm
from .plugins.base import load_plugin
from .scheduler import release_owners, submit_lane, task_states
from .spec import load_spec
from .state import lock, read_json, write_json

RUN_DIR = click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
                       required=True, help="Existing run directory on SCRATCH.")


def prepare_run(spec, spec_path):
    """Freeze inputs into a new run directory and let the plugin create tasks."""
    experiment = Path(spec["run"]["experiment"]).resolve()
    if not experiment.is_dir():
        raise click.ClickException(f"Experiment folder not found: {experiment}")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = (Path(spec["run"]["scratch_root"]) / f"{experiment.name}_{identifier}").resolve()
    (root / "tasks" / "logs").mkdir(parents=True)
    # Copy definitions only, never results, checkpoints or databases.
    for pattern in spec["run"]["inputs"]:
        for source in experiment.glob(pattern):
            destination = root / source.relative_to(experiment)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                shutil.copy2(source, destination)
    shutil.copy2(spec_path, root / "run_spec.yaml")
    manifest = dict(run_id=identifier, short_id=identifier[-8:], experiment=str(experiment),
                    spec=spec, tasks=[], lanes=0, jobs={}, cancelled=False)
    write_json(root / "run.json", manifest)  # plugins may read run.json in prepare
    tasks = load_plugin(spec["plugin"]["path"]).prepare(root, dict(spec["plugin"]["options"]))
    if not tasks:
        raise click.ClickException("Plugin generated no tasks")
    ids = [task["id"] for task in tasks]
    if len(set(ids)) != len(ids) or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", i) for i in ids):
        raise click.ClickException("Task ids must be unique and match [A-Za-z0-9_.-]+")
    manifest.update(tasks=tasks, lanes=min(int(spec["scheduling"]["lanes"]), len(tasks)))
    write_json(root / "run.json", manifest)
    for task_id in ids:
        write_json(root / "tasks" / task_id / "state.json", {"status": "PENDING", "owner": None})
    (root / "worker.sh").write_text(slurm.render_script(root, spec), encoding="utf-8", newline="\n")
    return root, manifest


def check_run(root, manifest, every=False):
    """Run the plugin pre-flight (CPU, seconds); raise ClickException on failure.

    Plugin hooks (optional): check_task(root, task) -> bool (False = skipped),
    check_selection(tasks) -> tasks to check by default (default: first task).
    """
    plugin = load_plugin(manifest["spec"]["plugin"]["path"])
    if not hasattr(plugin, "check_task"):
        click.echo("Check: plugin has no check_task; skipped")
        return
    tasks = manifest["tasks"]
    if not every:
        tasks = plugin.check_selection(tasks) if hasattr(plugin, "check_selection") else tasks[:1]
    for task in tasks:
        try:
            checked = plugin.check_task(root, task)
        except Exception as error:
            raise click.ClickException(
                f"Check failed for task {task['id']}: {type(error).__name__}: {error}\n"
                f"Run folder kept for inspection: {root}")
        click.echo(f"Check {task['id']}: {'OK' if checked is not False else 'skipped (custom train_fn)'}")


@click.group()
def cli():
    """Prepare frozen SCRATCH runs and manage their Slurm allocations."""


@cli.command()
@click.option("--spec", "spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Run specification YAML.")
@click.option("--submit/--dry-run", default=True, help="Submit lanes or only prepare.")
@click.option("--check/--no-check", default=True, help="Pre-flight check before submitting.")
def launch(spec_path, submit, check):
    """Create a NEW run from a spec; never reuses an old run."""
    try:
        spec = load_spec(spec_path)
    except ValueError as error:
        raise click.ClickException(str(error))
    root, manifest = prepare_run(spec, spec_path)
    click.echo(f"Run directory: {root}")
    click.echo(f"Tasks: {len(manifest['tasks'])}, lanes: {manifest['lanes']}")
    if check:
        check_run(root, manifest)
    if submit:
        for lane in range(manifest["lanes"]):
            click.echo(f"Lane {lane}: job {submit_lane(root, lane, 0)}")
    else:
        click.echo("Prepared only. Submit later with: submit --run-dir <run directory>")


@cli.command()
@RUN_DIR
@click.option("--all", "every", is_flag=True, help="Check every task, not a selection.")
def check(run_dir, every):
    """Pre-flight check of a prepared run (CPU, no training)."""
    check_run(run_dir.resolve(), read_json(run_dir / "run.json"), every)


@cli.command()
@RUN_DIR
def submit(run_dir):
    """Submit initial lanes of a prepared run; recorded jobs are reused."""
    manifest = read_json(run_dir / "run.json")
    for lane in range(manifest["lanes"]):
        click.echo(f"Lane {lane}: job {submit_lane(run_dir, lane, 0)}")


@cli.command()
@RUN_DIR
@click.option("--extra-chunks", type=click.IntRange(1), default=10, show_default=True)
@click.option("--lanes", type=click.IntRange(1), default=None, help="New lane count (default: unchanged).")
@click.option("--reset-failed", is_flag=True, help="Give FAILED tasks a new set of attempts.")
def resume(run_dir, extra_chunks, lanes, reset_failed):
    """Continue an inactive run in place; refuses while jobs are queued."""
    manifest = read_json(run_dir / "run.json")
    active = slurm.active_jobs(manifest)
    if active:
        raise click.ClickException(f"Run still has active jobs {active}; cancel them first")
    if reset_failed:
        for task_id, state in task_states(run_dir, manifest).items():
            if state["status"] == "FAILED":
                state.update(status="RETRY", failures=0)
                write_json(run_dir / "tasks" / task_id / "state.json", state)
    release_owners(run_dir, manifest)
    with lock(run_dir / "tasks" / "submission.lock"):
        manifest = read_json(run_dir / "run.json")
        next_chunk = 1 + max((int(key.split(":")[1]) for key in manifest["jobs"]), default=-1)
        manifest["spec"]["scheduling"]["max_chunks"] = next_chunk + extra_chunks
        manifest["cancelled"] = False
        if lanes:
            manifest["lanes"] = min(lanes, len(manifest["tasks"]))
        write_json(run_dir / "run.json", manifest)
    for lane in range(manifest["lanes"]):
        click.echo(f"Lane {lane}: job {submit_lane(run_dir, lane, next_chunk)}")


@cli.command()
@RUN_DIR
@click.option("--verbose", is_flag=True, help="List every task.")
def status(run_dir, verbose):
    """Summarize task states and active jobs."""
    manifest = read_json(run_dir / "run.json")
    states = task_states(run_dir, manifest)
    for task_id, state in states.items():
        if verbose or state["status"] == "FAILED":
            extra = f" owner={state.get('owner')} attempts={state.get('attempts', 0)}"
            click.echo(f"{task_id}: {state['status']}{extra}")
    click.echo(dict(Counter(state["status"] for state in states.values())))
    try:
        click.echo(f"Active jobs: {slurm.active_jobs(manifest)}")
    except (OSError, RuntimeError) as error:
        click.echo(f"Active jobs unknown: {error}")


@cli.command()
@RUN_DIR
def cancel(run_dir):
    """Block new successors, then cancel all jobs of this run."""
    with lock(run_dir / "tasks" / "submission.lock"):
        manifest = read_json(run_dir / "run.json")
        manifest["cancelled"] = True
        write_json(run_dir / "run.json", manifest)
    jobs = set(manifest["jobs"].values()) | set(slurm.active_jobs(manifest))
    slurm.scancel(sorted(jobs))
    click.echo(f"Cancelled {len(jobs)} job(s)")


@cli.command()
@RUN_DIR
@click.option("--export-dir", type=click.Path(path_type=Path), default=None,
              help="Copy the complete run to this new folder.")
def finalize(run_dir, export_dir):
    """Require completion, run plugin finalize, optionally export a copy."""
    manifest = read_json(run_dir / "run.json")
    incomplete = [i for i, s in task_states(run_dir, manifest).items() if s["status"] != "COMPLETE"]
    if incomplete:
        raise click.ClickException(f"Incomplete tasks: {incomplete}")
    plugin = load_plugin(manifest["spec"]["plugin"]["path"])
    if hasattr(plugin, "finalize"):
        plugin.finalize(run_dir, manifest)
    if export_dir is not None:
        if export_dir.exists():
            raise click.ClickException("Export destination must not exist")
        shutil.copytree(run_dir, export_dir)
        if hasattr(plugin, "export_rewrite"):
            plugin.export_rewrite(run_dir, export_dir)
        click.echo(f"Exported to {export_dir}; SCRATCH originals retained")
    click.echo("Finalization complete")


if __name__ == "__main__":
    cli()