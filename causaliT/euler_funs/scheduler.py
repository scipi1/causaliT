"""Task claiming and bounded lane continuation.

Task states (tasks/<id>/state.json):
  PENDING   never started; any lane may claim it
  RUNNING   owned by a lane and a job; if seen by a later job, that job crashed
  PAUSED    stopped cleanly at a checkpoint; resumed by its owner lane
  RETRY     raised an exception below max_failures; retried by its owner lane
  COMPLETE / FAILED   terminal
A started task keeps its owner lane, so only one job ever works on it: jobs of
one lane are serialized by afterany and by the lane lock. Owners are cleared
by `resume`, when no job of the run is active.
"""

from pathlib import Path

from . import slurm
from .state import lock, read_json, write_json

TERMINAL = ("COMPLETE", "FAILED")
RESUMABLE = ("RUNNING", "PAUSED", "RETRY")


def state_path(root, task_id):
    return Path(root) / "tasks" / task_id / "state.json"


def task_states(root, manifest):
    return {task["id"]: read_json(state_path(root, task["id"])) for task in manifest["tasks"]}


def claim_next(root, manifest, lane, job_id, skip):
    """Return (task, state) for this lane, or None. Owned tasks come first."""
    max_failures = int(manifest["spec"]["scheduling"]["max_failures"])
    with lock(Path(root) / "tasks" / "queue.lock"):
        states = task_states(root, manifest)
        candidates = sorted(
            (task for task in manifest["tasks"] if task["id"] not in skip),
            key=lambda task: states[task["id"]].get("owner") != lane)
        for task in candidates:
            state = states[task["id"]]
            owner = state.get("owner")
            if state["status"] in TERMINAL or owner not in (None, lane):
                continue
            if state["status"] == "RUNNING" and state.get("job_id") != job_id:
                # The previous job died without recording an outcome.
                state["failures"] = state.get("failures", 0) + 1
                state["error"] = f"Job {state.get('job_id')} ended while the task was running"
                if state["failures"] >= max_failures:
                    state["status"] = "FAILED"
                    write_json(state_path(root, task["id"]), state)
                    continue
            state.update(status="RUNNING", owner=lane, job_id=job_id)
            state["attempts"] = state.get("attempts", 0) + 1
            write_json(state_path(root, task["id"]), state)
            return task, state
    return None


def has_work(root, manifest, lane):
    """True if this lane could still claim something."""
    return any(state["status"] not in TERMINAL and state.get("owner") in (None, lane)
               for state in task_states(root, manifest).values())


def submit_lane(root, lane, chunk, predecessor=None):
    """Submit at most one recorded job per lane/chunk; idempotent."""
    root = Path(root).resolve()
    with lock(root / "tasks" / "submission.lock"):
        manifest = read_json(root / "run.json")
        if manifest.get("cancelled"):
            return None
        key = f"{lane}:{chunk}"
        if key not in manifest["jobs"]:
            manifest["jobs"][key] = slurm.sbatch(root, manifest, lane, chunk, predecessor)
            write_json(root / "run.json", manifest)
        return manifest["jobs"][key]


def release_owners(root, manifest):
    """Only when no job is active: let any lane pick up unfinished tasks."""
    with lock(Path(root) / "tasks" / "queue.lock"):
        for task_id, state in task_states(root, manifest).items():
            if state["status"] not in TERMINAL:
                if state["status"] == "RUNNING" and manifest.get("cancelled"):
                    state["status"] = "PAUSED"  # killed by cancel, not a crash
                state["owner"] = None
                write_json(state_path(root, task_id), state)
