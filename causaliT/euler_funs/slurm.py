"""Worker script rendering and thin wrappers around Slurm commands."""

from pathlib import Path
import shlex
import subprocess

# Derived from the package location, so the folder can be copied/renamed freely.
WORKER_MODULE = __name__.rsplit(".", 1)[0] + ".worker"


def render_script(root, spec):
    """One CPU/GPU worker script; USR1 is forwarded via signal and stop file."""
    root = Path(root).resolve()
    res = spec["resources"]
    lines = ["#!/bin/bash", "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={int(res['cpus_per_task'])}",
             f"#SBATCH --time={res['walltime']}",
             f"#SBATCH --mem-per-cpu={res['mem_per_cpu']}",
             f"#SBATCH --signal=B:USR1@{int(res['signal_margin'])}"]
    if res["accelerator"] == "gpu":
        gpu_type = res["gpu_type"] + ":" if res["gpu_type"] else ""
        lines.append(f"#SBATCH --gpus={gpu_type}{int(res['gpus'])}")
        if res["gpu_mem"]:
            lines.append(f"#SBATCH --gres=gpumem:{res['gpu_mem']}")
    if res["partition"]:
        lines.append(f"#SBATCH --partition={res['partition']}")
    lines += [f"#SBATCH {option}" for option in res["extra_sbatch"]]
    stop_file = shlex.quote(str(root / "tasks" / "stop_")) + '"${SLURM_JOB_ID}"'
    lines += [
        "set -euo pipefail",
        f"STOP_FILE={stop_file}",
        "child_pid=''",
        # The stop file covers a signal arriving before Python installs its handler.
        'forward_stop() { touch "$STOP_FILE"; if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi; }',
        "trap forward_stop USR1",
    ]
    lines += [f"module load {shlex.quote(module)}" for module in res["modules"]]
    lines += list(res["setup"])
    if res["venv"]:
        lines.append(f"source {shlex.quote(str(Path(res['venv']).as_posix()) + '/bin/activate')}")
        lines.append('if [[ -z "${VIRTUAL_ENV:-}" ]]; then echo "Python environment activation failed" >&2; exit 1; fi')
    lines += [f"export {key}={shlex.quote(str(value))}" for key, value in res["env"].items()]
    lines += [
        f'export PYTHONPATH={shlex.quote(spec["run"]["project_dir"])}"${{PYTHONPATH:+:$PYTHONPATH}}"',
        f"cd {shlex.quote(str(root))}",
        f'(trap "" USR1; exec python -m {WORKER_MODULE} --run-dir {shlex.quote(str(root))} '
        '--lane "$1" --chunk "$2" --stop-file "$STOP_FILE") &',
        "child_pid=$!",
        "set +e",
        'wait "$child_pid"; status=$?',
        # wait returns early when the trap fires; keep waiting for the child.
        'while kill -0 "$child_pid" 2>/dev/null; do wait "$child_pid"; status=$?; done',
    ]
    if res["venv"]:
        lines.append("deactivate")
    lines.append('exit "$status"')
    return "\n".join(lines) + "\n"


def job_name(manifest, lane):
    """Deterministic name used to reconcile jobs that were never recorded."""
    return f"{manifest['spec']['scheduling']['job_prefix']}_{manifest['short_id']}_L{lane}"


def sbatch(root, manifest, lane, chunk, predecessor=None):
    command = ["sbatch", "--parsable", f"--job-name={job_name(manifest, lane)}",
               f"--output={root}/tasks/logs/lane_{lane}_%j.out",
               f"--error={root}/tasks/logs/lane_{lane}_%j.err"]
    if predecessor:
        command.append(f"--dependency=afterany:{predecessor}")
    command += [str(Path(root) / "worker.sh"), str(lane), str(chunk)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout}")
    return job_id


def active_jobs(manifest):
    """Queued/running job IDs of this run, found by name (also unrecorded ones)."""
    prefix = job_name(manifest, "")
    result = subprocess.run(["squeue", "--me", "--noheader", "--format=%i %j"],
                            check=True, capture_output=True, text=True)
    jobs = [line.split(None, 1) for line in result.stdout.splitlines() if line.strip()]
    return [job for job, name in jobs if name.startswith(prefix)]


def scancel(job_ids):
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=True)
