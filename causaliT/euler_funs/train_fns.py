"""causaliT train functions for the euler_funs plugins (training, sweep, dagsweep).

Signature required by plugins.training.call_train:
    train_fn(config, save_dir, data_dir, stop_event, folds, resources) -> results
Each function resumes from <save_dir>/k_<fold>/resume/last.ckpt when present,
stops at the next epoch end once stop_event is set and then raises TaskPaused.
A finished fold leaves resume/DONE (informative); re-running a finished fold
resumes at max_epochs, so it only recomputes the metrics.
"""

from pathlib import Path

from omegaconf import OmegaConf

from .lightning_utils import PauseAtEpochEnd, ResumeCheckpoint, latest_checkpoint, raise_if_paused

DONE = "DONE"


def _resume_dir(save_dir, fold):
    return Path(save_dir) / f"k_{fold}" / "resume"


def _resolve(config):
    """Same preprocessing as causaliT.cli: update_config on an OmegaConf object
    (it uses attribute access, so a plain dict fails)."""
    from causaliT.training.experiment_control import update_config
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    return update_config(config)


def _resume_from(config, base_dir):
    """Path of adaptive_training.resume_from (full-state start checkpoint,
    relative to base_dir = the run folder) or None. Raises if missing."""
    rel = OmegaConf.select(config, "adaptive_training.resume_from", default=None)
    if not rel:
        return None
    path = Path(rel) if Path(rel).is_absolute() else Path(base_dir) / rel
    if not path.is_file():
        raise FileNotFoundError(f"adaptive_training.resume_from not found: {path}")
    return str(path)


def _prepare(config, save_dir):
    config = _resolve(config)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(save_dir) / "config.yaml")
    return config


def check(config, data_dir, trainer="standard"):
    """Cheap pre-flight (CPU, no training): config preprocessing, dataset
    metadata, trainer requirements and one model build. Raises on problems."""
    import copy
    from causaliT.training.config_utils import populate_seq_lengths_from_dataset
    from causaliT.training.trainer import create_model_instance
    if trainer not in TRAINERS:
        raise ValueError(f"Unknown trainer {trainer!r}; expected one of {sorted(TRAINERS)}")
    config = _resolve(copy.deepcopy(config))
    if trainer == "adaptive":
        if not OmegaConf.select(config, "adaptive_training"):
            raise ValueError("adaptive trainer needs an adaptive_training block")
        if not OmegaConf.select(config, "training.use_gradient_routing", default=False):
            raise ValueError("adaptive trainer needs training.use_gradient_routing=true")
    dataset = OmegaConf.select(config, "data.dataset")
    metadata = Path(data_dir) / str(dataset) / "dataset_metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata} "
                                f"(data_dir={data_dir}, data.dataset={dataset})")
    config = populate_seq_lengths_from_dataset(config, str(data_dir))
    import torch
    if not torch.cuda.is_available() and OmegaConf.select(config, "model.kwargs.device"):
        # The launcher has no GPU: build on CPU (this copy only; jobs keep cuda).
        OmegaConf.update(config, "model.kwargs.device", "cpu")
    create_model_instance(config, str(data_dir))


def trainer_of(train_fn):
    """Trainer name of a train_fn path such as 'causaliT.euler_funs.train_fns:train'."""
    attribute = str(train_fn).partition(":")[2]
    for name, function in TRAINERS.items():
        if function.__name__ == attribute and str(train_fn).startswith(__name__):
            return name
    return None


def train(config, save_dir, data_dir, stop_event, folds=None, resources=None):
    """Standard k-fold trainer(); folds=None trains every fold."""
    from causaliT.training.trainer import trainer
    config = _prepare(config, save_dir)
    pauses = {}

    def fold_hooks(fold):
        directory = _resume_dir(save_dir, fold)
        pauses[fold] = PauseAtEpochEnd(stop_event)
        return [ResumeCheckpoint(str(directory)), pauses[fold]], latest_checkpoint(directory)

    def after_fold(fold):
        raise_if_paused(pauses[fold])
        (_resume_dir(save_dir, fold) / DONE).touch()

    warm = OmegaConf.select(config, 'training.warm_start_from', default=None)
    if warm:
        warm = Path(warm) if Path(warm).is_absolute() else Path(save_dir) / warm
        if not warm.is_file():
            raise FileNotFoundError(f'Warm-start checkpoint not found: {warm}')
    return trainer(config=config, data_dir=str(data_dir), save_dir=str(save_dir), cluster=True,
                   folds=folds, fold_hooks=fold_hooks, after_fold=after_fold,
                   warm_start_ckpt=str(warm) if warm else None)


def adaptive_train(config, save_dir, data_dir, stop_event, folds=None, resources=None):
    """adaptive_trainer() (single fold k_0); phase state lives in the checkpoint."""
    from causaliT.training.adaptive_trainer import adaptive_trainer
    config = _prepare(config, save_dir)
    directory = _resume_dir(save_dir, 0)
    pause = PauseAtEpochEnd(stop_event)
    # Own checkpoint wins (chain chunks); else the external full-state start.
    ckpt = latest_checkpoint(directory) or _resume_from(config, save_dir)
    results = adaptive_trainer(config=config, data_dir=str(data_dir), save_dir=str(save_dir),
                               cluster=True, extra_callbacks=[ResumeCheckpoint(str(directory)), pause],
                               resume_ckpt=ckpt)
    raise_if_paused(pause)
    (directory / DONE).touch()
    return results


TRAINERS = {"standard": train, "adaptive": adaptive_train}
