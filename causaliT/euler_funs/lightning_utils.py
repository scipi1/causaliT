"""Optional PyTorch Lightning helpers for epoch-boundary pause/resume.

Typical use inside a project train_fn:
    pause = PauseAtEpochEnd(stop_event)
    trainer = Trainer(callbacks=[ResumeCheckpoint(ckpt_dir), pause, ...])
    trainer.fit(model, data, ckpt_path=latest_checkpoint(ckpt_dir))
    raise_if_paused(pause)
Project-specific state (e.g. adaptive weights) belongs in a project callback
implementing on_save_checkpoint / on_load_checkpoint.
"""

import os
from pathlib import Path

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .plugins.base import TaskPaused


class ResumeCheckpoint(ModelCheckpoint):
    """Publish last.ckpt each epoch atomically, keeping the previous file."""

    def __init__(self, directory):
        super().__init__(dirpath=directory, save_last=True, save_top_k=0, every_n_epochs=1,
                         save_on_train_epoch_end=True, enable_version_counter=False)

    def _save_checkpoint(self, trainer, filepath):
        temporary = filepath + ".pending"
        trainer.save_checkpoint(temporary, self.save_weights_only)
        if trainer.is_global_zero:
            if os.path.exists(filepath):
                os.replace(filepath, filepath + ".previous")
            os.replace(temporary, filepath)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath


class PauseAtEpochEnd(Callback):
    """Stop the trainer at the next epoch end once stop_event is set."""

    def __init__(self, stop_event=None):
        self.stop_event = stop_event
        self.paused = False

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stop_event is not None and self.stop_event.is_set():
            self.paused = True
            trainer.should_stop = True


def latest_checkpoint(directory):
    """Current last checkpoint, or the previous generation, or None."""
    for name in ("last.ckpt", "last.ckpt.previous"):
        candidate = Path(directory) / name
        if candidate.is_file():
            return str(candidate)
    return None


def raise_if_paused(pause_callback):
    if pause_callback.paused:
        raise TaskPaused()
