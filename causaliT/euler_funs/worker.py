"""Execute one allocation of a lane: claim tasks until done, paused or stopped."""

import os
from pathlib import Path
import signal
import sys
import threading
import traceback

import click

from .plugins.base import TaskPaused, load_plugin
from .scheduler import claim_next, has_work, state_path, submit_lane
from .state import lock, read_json, write_json


def install_stop(stop_file):
    stop = threading.Event()
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: stop.set())

    def watch():
        # Covers signals delivered while Python still ignored USR1 at startup.
        while not stop.wait(0.5):
            if stop_file.exists():
                stop.set()
    if stop_file.exists():
        stop.set()
    threading.Thread(target=watch, daemon=True).start()
    return stop


def run_lane(run_dir, lane, chunk, stop, job_id):
    """Return the number of tasks that raised in this allocation."""
    run_dir = Path(run_dir).resolve()
    errors = 0
    with lock(run_dir / "tasks" / f"lane_{lane}.lock"):
        manifest = read_json(run_dir / "run.json")
        if manifest.get("cancelled") or not has_work(run_dir, manifest, lane):
            return 0
        # Import first: a broken environment must not spawn a successor chain.
        plugin = load_plugin(manifest["spec"]["plugin"]["path"])
        scheduling = manifest["spec"]["scheduling"]
        if scheduling["schedule"] == "chain" and chunk + 1 < int(scheduling["max_chunks"]):
            submit_lane(run_dir, lane, chunk + 1, job_id)
        attempted = set()
        while not stop.is_set() and not read_json(run_dir / "run.json").get("cancelled"):
            claimed = claim_next(run_dir, manifest, lane, job_id, attempted)
            if claimed is None:
                break
            task, state = claimed
            attempted.add(task["id"])
            state["chunk"] = chunk
            try:
                plugin.run(run_dir, task, stop)
            except TaskPaused:
                state["status"] = "PAUSED"
            except Exception:
                errors += 1
                state["failures"] = state.get("failures", 0) + 1
                state["error"] = traceback.format_exc()
                limit = int(scheduling["max_failures"])
                state["status"] = "FAILED" if state["failures"] >= limit else "RETRY"
                traceback.print_exc()
            else:
                state["status"] = "COMPLETE"
                state.pop("error", None)
            write_json(state_path(run_dir, task["id"]), state)
            if state["status"] == "PAUSED":
                break
    return errors


@click.command()
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--lane", type=int, required=True)
@click.option("--chunk", type=int, required=True)
@click.option("--stop-file", type=click.Path(path_type=Path), required=True)
def main(run_dir, lane, chunk, stop_file):
    stop = install_stop(stop_file)
    job_id = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    sys.exit(1 if run_lane(run_dir, lane, chunk, stop, job_id) else 0)


if __name__ == "__main__":
    main()
