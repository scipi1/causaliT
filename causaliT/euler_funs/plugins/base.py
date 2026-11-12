"""Plugin contract and loading.

A plugin is a module (or object) exposing:
  prepare(root, options) -> list[dict]
      Called once at launch on the frozen run directory. Each dict must
      contain a unique "id" matching [A-Za-z0-9_.-]+ and be JSON-serializable.
      Must not submit Slurm jobs.
  run(root, task, stop_event) -> None
      Execute or continue one task. Must be idempotent: it is called again
      after a pause or crash. Poll stop_event (threading.Event) at safe points;
      on a stop before completion, checkpoint and raise TaskPaused. Return
      normally only when the task is scientifically complete.
Optional:
  finalize(root, manifest)       after all tasks are COMPLETE
  export_rewrite(root, dest)     after copying a run to dest (fix paths)
"""

from importlib import import_module

BUILTIN = ("training", "sweep", "dagsweep")


class TaskPaused(Exception):
    """Task stopped at a checkpoint boundary, not at completion."""


def load_object(path):
    """Import "module:attribute" or "module"."""
    module_name, _, attribute = path.partition(":")
    obj = import_module(module_name)
    for part in filter(None, attribute.split(".")):
        obj = getattr(obj, part)
    return obj


def load_plugin(path):
    if path in BUILTIN:
        path = f"{__name__.rsplit('.', 1)[0]}.{path}"
    plugin = load_object(path)
    for attribute in ("prepare", "run"):
        if not callable(getattr(plugin, attribute, None)):
            raise TypeError(f"Plugin {path} must define {attribute}")
    return plugin
