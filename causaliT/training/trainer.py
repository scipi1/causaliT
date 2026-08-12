# Standard library imports
import json
import logging
import os
import sys
import time
from os.path import dirname, abspath, join
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from sklearn.model_selection import KFold

# Local imports
from causaliT.training.callbacks import (
    get_checkpoint_callback, get_early_stopping_callback, MemoryLoggerCallback,
    GradientLogger, MetricsAggregator, PerRunManifest,
    BestReconstructionCheckpoint, BestCausalCheckpoint, DataIndexTracker,
    KFoldResultsTracker, GradientJacobianLogger
)
from causaliT.training.forecasters import (
    TransformerForecaster,
    StageCausalForecaster,
    SingleCausalForecaster,
    SingleCausalResForecaster,
    NoiseAwareCausalForecaster,
    NoiseAwareCausalResForecaster,
    VarianceCausalForecaster,
    AttentionSelectorForecaster,
)

from causaliT.training.dataloader import ProcessDataModule
from causaliT.training.stage_causal_dataloader import StageCausalDataModule
from causaliT.training.experiment_control import update_config
from causaliT.training.config_utils import populate_seq_lengths_from_dataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# =============================================================================
# PRIMITIVES
# =============================================================================

def train_single_fold(
    config: dict,
    model: pl.LightningModule,
    dm,
    fold: int,
    train_local_idx,
    val_local_idx,
    test_idx,
    train_val_idx,
    save_dir: str,
    trainable_params: int = 0,
    cluster: bool = False,
    resume_ckpt: str = None,
    warm_start_ckpt: str = None,
    experiment_tag: str = "NA",
    debug: bool = False,
    best: bool = False,
    extra_callbacks: Optional[List] = None,
    reload_dataloaders_every_n_epochs: int = 0,
) -> dict:
    """
    Execute training for a single fold and return metrics.

    This is the **core execution primitive** shared by:
    - ``trainer()``            — called for each k-fold, no extra callbacks
    - ``calibration.py``       — called with k=1, injects GradientNormTracker
    - ``causal_initialization.py`` — called with k=1, injects CausalInitProgressLogger

    The ``extra_callbacks`` parameter is the extension point: any caller can
    inject additional Lightning callbacks without modifying this function.
    All Lightning-level bookkeeping (loggers, checkpoints, pl.Trainer creation,
    fit/validate/test) lives exclusively here.

    Args:
        config:            Configuration dictionary.
        model:             Already-instantiated LightningModule (seed already set by caller).
        dm:                DataModule whose ``prepare_data()`` has been called.
        fold:              Fold index (0-based); determines the ``k_{fold}`` subfolder.
        train_local_idx:   Local indices into ``train_val_idx`` for this fold's training set.
        val_local_idx:     Local indices into ``train_val_idx`` for this fold's validation set.
        test_idx:          Global test indices (may be None if pre-split data).
        train_val_idx:     Global index mapping array (local → global).
        save_dir:          Parent directory; a ``k_{fold}`` subfolder is created here.
        trainable_params:  Pre-computed parameter count (pass 0 if not needed).
        cluster:           If True, disables progress bar and uses 1 GPU device.
        resume_ckpt:       Optional checkpoint path to **fully resume** training
                           (model weights + optimizer state + epoch counter).
                           Use for crash recovery or continuation of the same run.
        warm_start_ckpt:   Optional checkpoint path to load **model weights only**.
                           Optimizer state is reset, training starts at epoch 0.
                           Use when transitioning between pipeline stages (e.g.
                           causal init → main training) where the training config
                           has changed and stale optimizer state would be harmful.
                           Mutually exclusive with ``resume_ckpt``.
        experiment_tag:    Tag stored in the per-run manifest.
        debug:             Enables anomaly detection, memory logger, etc.
        best:              If True, return metrics from the best checkpoint rather
                           than the final epoch.
        extra_callbacks:   Additional Lightning callbacks injected by the caller.
                           These are appended **after** the standard callbacks so
                           they have access to all trainer state.
        reload_dataloaders_every_n_epochs:
                           Forwarded to ``pl.Trainer``.  Default 0 keeps the train
                           dataloader cached for the whole run (standard behaviour).
                           Set to 1 when a callback mutates the data module's train
                           split mid-fit (e.g. the adaptive cross-fit phase switch)
                           so Lightning re-queries ``dm.train_dataloader()`` at each
                           epoch boundary and picks up the new subset.

    Returns:
        dict: Metrics for this fold (val + test + timing).
    """
    logger_info = logging.getLogger("logger_info")

    save_dir_k = join(save_dir, f"k_{fold}")
    logs_dir = join(save_dir_k, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Convert local indices → global indices
    train_global_idx = train_val_idx[train_local_idx]
    val_global_idx = train_val_idx[val_local_idx]

    # ---- Loggers ----------------------------------------------------------------
    logger_csv = CSVLogger(save_dir=logs_dir, name="csv")

    # ---- Standard callbacks -----------------------------------------------------
    checkpoint_callback = get_checkpoint_callback(
        save_dir_k, config["training"]["save_ckpt_every_n_epochs"]
    )
    manifest_callback = PerRunManifest(config, path=save_dir_k, tag=experiment_tag)
    best_reconstruction_callback = BestReconstructionCheckpoint(
        save_dir_k, monitor="val_x_mae", mode="min"
    )
    best_causal_checkpoint = BestCausalCheckpoint(save_dir_k)
    data_index_tracker = DataIndexTracker(
        save_dir_k, fold, train_global_idx, val_global_idx, test_idx
    )

    callbacks_list = list(checkpoint_callback)
    callbacks_list += [manifest_callback, best_reconstruction_callback, best_causal_checkpoint, data_index_tracker]

    # Early stopping: config-driven (new) with legacy fallback
    es_callback = get_early_stopping_callback(config)
    if es_callback is not None:
        callbacks_list.append(es_callback)

    if debug:
        callbacks_list.append(MemoryLoggerCallback())

    if "debug_optimizer" in config["special"]["mode"]:
        callbacks_list.append(GradientLogger())
        callbacks_list.append(LearningRateMonitor(logging_interval="epoch"))

    if config["training"].get("log_jacobian", False):
        jacobian_every_n_epochs = config["training"].get("jacobian_every_n_epochs", 5)
        callbacks_list.append(
            GradientJacobianLogger(
                save_dir=save_dir_k,
                every_n_epochs=jacobian_every_n_epochs,
                enabled=True,
            )
        )

    # ---- Inject caller-supplied callbacks (gradient trackers, progress loggers…)
    if extra_callbacks:
        callbacks_list.extend(extra_callbacks)

    # ---- Data split for this fold -----------------------------------------------
    dm.update_idx(
        train_idx=train_local_idx,
        val_idx=val_local_idx,
        test_idx=test_idx,
    )

    # ---- Lightning Trainer ------------------------------------------------------
    # Defense-in-depth: reloading the dataloader every epoch respawns the worker
    # pool each epoch.  On Windows (spawn) each worker re-imports the package and
    # re-copies the in-memory dataset tensors, crashing the session at the first
    # epoch boundary; on Linux (fork) it is cheaper but leaks memory over long
    # runs.  Datasets here are in-memory TensorDatasets (no I/O to overlap), so
    # workers add pure overhead — force single-process loading whenever the train
    # dataloader is reloaded, so any caller of train_single_fold is safe.
    if reload_dataloaders_every_n_epochs:
        if getattr(dm, "num_workers", 0):
            dm.num_workers = 0
        if getattr(dm, "persistent_workers", False):
            dm.persistent_workers = False


    pl_trainer = pl.Trainer(
        callbacks=callbacks_list,

        logger=logger_csv,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=1 if cluster else "auto",
        max_epochs=config["training"]["max_epochs"],
        log_every_n_steps=1,
        deterministic=True,
        enable_progress_bar=not cluster,
        enable_model_summary=not cluster,
        detect_anomaly=debug,
        gradient_clip_val=config["training"].get("gradient_clip_val", None),
        gradient_clip_algorithm=config["training"].get("gradient_clip_algorithm", "norm"),
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )

    # ---- Warm-start: load model weights only (no optimizer / epoch restore) -----
    if warm_start_ckpt is not None:
        if resume_ckpt is not None:
            raise ValueError(
                "warm_start_ckpt and resume_ckpt are mutually exclusive. "
                "Use warm_start_ckpt for stage transitions (weights only), "
                "resume_ckpt for crash recovery (full state)."
            )
        logger_info.info(f"Warm-starting from: {warm_start_ckpt}")
        checkpoint = torch.load(warm_start_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger_info.warning(f"Warm-start missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logger_info.warning(f"Warm-start unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if not cluster:
            print(f"  ✓ Warm-started model weights from: {warm_start_ckpt}")

    # ---- Training ---------------------------------------------------------------
    start_time = time.time()
    pl_trainer.fit(model, dm, ckpt_path=resume_ckpt)
    training_time = time.time() - start_time
    num_epochs = pl_trainer.current_epoch + 1
    avg_time_per_epoch = training_time / num_epochs if num_epochs > 0 else 0

    # ---- Validation -------------------------------------------------------------
    if dm.val_ds is not None:
        pl_trainer.validate(model, dm)
        val_metrics = pl_trainer.callback_metrics.copy()
    else:
        val_metrics = {}

    # ---- Test -------------------------------------------------------------------
    pl_trainer.test(model, dm)
    test_metrics = pl_trainer.callback_metrics.copy()

    # ---- Collect metrics --------------------------------------------------------
    if best:
        best_metrics_file = join(save_dir_k, "best_metrics.json")
        if os.path.exists(best_metrics_file):
            with open(best_metrics_file, "r") as f:
                best_metrics_data = json.load(f)
            fold_metrics = {
                k: v
                for k, v in best_metrics_data.items()
                if k not in ["best_epoch", "best_checkpoint_path"]
            }
        else:
            fold_metrics = {**val_metrics, **test_metrics}
    else:
        fold_metrics = {**val_metrics, **test_metrics}

    fold_metrics["trainable_params"] = trainable_params
    fold_metrics["total_training_time"] = training_time
    fold_metrics["avg_time_per_epoch"] = avg_time_per_epoch

    logger_info.info(
        f"Fold {fold}: training_time={training_time:.1f}s, epochs={num_epochs}"
    )

    return fold_metrics


def _make_fold_splits(
    config: dict,
    dm,
    seed: int,
    data_dir: str = None,
) -> Tuple[list, Optional[np.ndarray], np.ndarray]:
    """
    Build k-fold train/val index splits and return test indices.

    Handles three cases:
    - Pre-split data  (train_file / test_file set in config)
    - Custom test-index file  (config["data"]["test_ds_ixd"] != None)
    - Automatic 80/20 test split

    Args:
        config:   Configuration dictionary.
        dm:       DataModule (``prepare_data()`` must have been called already,
                  or ``get_ds_len()`` must work without it).
        seed:     Random seed for KFold / permutation.
        data_dir: Root data directory.  Required only when
                  ``config["data"]["test_ds_ixd"]`` is set.

    Returns:
        fold_splits:   List of (train_local_idx, val_local_idx) tuples.
        test_idx:      Global test indices array, or None for pre-split data.
        train_val_idx: Global array mapping local → global indices.
    """
    use_presplit = (
        config["data"].get("train_file") is not None
        and config["data"].get("test_file") is not None
    )

    if use_presplit:
        dataset_size = dm.get_ds_len()
        train_val_idx = np.arange(dataset_size)
        test_idx = None
    else:
        dataset_size = dm.get_ds_len()
        indices = np.arange(dataset_size)
        test_ds_idx_filename = config["data"]["test_ds_ixd"]

        if test_ds_idx_filename is not None:
            if data_dir is None:
                raise ValueError(
                    "_make_fold_splits: data_dir must be provided when "
                    "config['data']['test_ds_ixd'] is set."
                )
            test_idx = np.load(
                join(data_dir, config["data"]["dataset"], test_ds_idx_filename)
            )
            mask = np.isin(indices, test_idx)
            train_val_idx = indices[~mask]
        else:
            test_size = int(0.2 * dataset_size)
            test_idx = indices[:test_size]
            train_val_idx = indices[test_size:]

    k_folds = config["training"]["k_fold"]

    if k_folds == 1:
        rng = np.random.default_rng(seed)
        shuffled_idx = rng.permutation(len(train_val_idx))
        split_point = int(0.8 * len(shuffled_idx))
        fold_splits = [(shuffled_idx[:split_point], shuffled_idx[split_point:])]
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_splits = list(kfold.split(train_val_idx))

    return fold_splits, test_idx, train_val_idx


def resolve_seeds(config: dict) -> Tuple[int, int]:
    """
    Resolve the (model_seed, data_seed) pair from a training config.

    Two independent sources of randomness are distinguished:

    * ``training.seed``      -> MODEL seed: weight initialization, dropout,
                                any torch/numpy RNG consumed during fitting.
    * ``training.data_seed`` -> DATA seed: train/val/test split and the
                                reconstruct/structure partition.

    ``data_seed`` is OPTIONAL and defaults to ``seed``, so every config written
    before this split behaves exactly as before.  Setting it explicitly (as the
    grouped DAG sweep does, where ``data_seed`` follows the DAG seed) keeps the
    data partition FIXED while the model seed varies - which is what makes
    per-edge stability across initializations measurable on one sampled DAG.

    Args:
        config: Configuration dictionary.

    Returns:
        ``(model_seed, data_seed)``.
    """
    model_seed = int(config["training"].get("seed", 42))
    data_seed = config["training"].get("data_seed", None)
    return model_seed, (model_seed if data_seed is None else int(data_seed))


def _count_trainable_params(config: dict, data_dir: str) -> int:
    """
    Instantiate a temporary model, count trainable parameters, then delete it.

    Args:
        config:   Configuration dictionary.
        data_dir: Data directory (needed for hard-mask loading in some models).

    Returns:
        Number of trainable parameters.
    """
    temp_model = create_model_instance(config, data_dir)
    n_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model
    return n_params


def _run_post_training_evaluations(
    config: dict,
    save_dir: str,
    data_dir: str,
) -> None:
    """
    Run configured post-training evaluation functions.

    Wrapped in try/except so that a failing evaluation never loses training
    results.  Evaluation strategy is controlled by
    ``config["evaluation"]["functions"]``:
    - If not set: runs all standard evaluations.
    - If set: runs only the listed functions.

    Args:
        config:   Configuration dictionary.
        save_dir: Experiment save directory.
        data_dir: Data directory.
    """
    try:
        from causaliT.evaluation.eval_funs import (
            run_all_evaluations,
            run_evaluations_from_config,
        )

        print("\n" + "=" * 60)
        print("Running post-training evaluations...")
        print("=" * 60)

        eval_functions = config.get("evaluation", {}).get("functions", None)

        if eval_functions is not None:
            print(f"Using config-specified evaluation functions: {eval_functions}")
            run_evaluations_from_config(
                experiment=save_dir,
                datadir_path=data_dir,
                show_plots=False,
                functions=eval_functions,
            )
        else:
            run_all_evaluations(
                experiment=save_dir,
                datadir_path=data_dir,
                show_plots=False,
            )

        print("\nPost-training evaluations completed!")

    except Exception as e:
        print(f"\nWarning: Post-training evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        print("Training results are saved. Run evaluations manually if needed.")


# =============================================================================
# COMPOSED FUNCTION: k-fold trainer
# =============================================================================

def trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str = "NA",
    resume_ckpt: str = None,
    warm_start_ckpt: str = None,
    plot_pred_check: bool = False,
    debug: bool = False,
    best: bool = False,
) -> pd.DataFrame:
    """
    Training function with k-fold cross-validation.

    Orchestrates the full training pipeline:
    1. Populates sequence lengths from dataset metadata.
    2. Creates data module and computes fold splits.
    3. Calls ``train_single_fold`` for each fold.
    4. Runs post-training evaluations.

    The ``train_single_fold`` primitive carries all Lightning-level logic,
    so this function is purely about orchestration.

    Args:
        config:          Configuration dictionary.
        data_dir:        Path to data directory.
        save_dir:        Path to save outputs (checkpoints, logs, results).
        cluster:         Whether running on a compute cluster.
        experiment_tag:  Tag for experiment tracking.
        resume_ckpt:     Optional checkpoint path to **fully resume** training
                         (model weights + optimizer state + epoch counter).
                         Use for crash recovery.
        warm_start_ckpt: Optional checkpoint path to load **model weights only**
                         (fresh optimizer, epoch 0). Use for stage transitions
                         (e.g. causal init → main training).
                         Mutually exclusive with ``resume_ckpt``.
        plot_pred_check: Whether to plot prediction checks (currently unused).
        debug:           Enable debug mode (anomaly detection, memory logging).
        best:            If True, use metrics from best checkpoint; otherwise
                         use final-epoch metrics.

    Returns:
        pd.DataFrame: Metrics for each fold (one row per fold).
    """
    logger_info = logging.getLogger("logger_info")

    # Model seed drives weight init; data seed drives the split (see resolve_seeds).
    # They coincide unless training.data_seed is set explicitly.
    seed, data_seed = resolve_seeds(config)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    if data_seed != seed:
        logger_info.info(f"seeds: model={seed}, data={data_seed}")

    # Populate sequence lengths from dataset metadata
    config = populate_seq_lengths_from_dataset(config, data_dir)

    # Build data module and index splits (data_seed: fixed across model seeds)
    dm = get_dataloader(config, data_dir, cluster, data_seed)
    dm.prepare_data()
    fold_splits, test_idx, train_val_idx = _make_fold_splits(config, dm, data_seed, data_dir=data_dir)

    k_folds = len(fold_splits)
    print(
        f"k={k_folds}: "
        + ("simple 80/20 split" if k_folds == 1 else f"{k_folds}-fold cross-validation")
    )
    logger_info.info(f"k_fold={k_folds}")

    # Count trainable parameters once (identical for all folds)
    trainable_params = _count_trainable_params(config, data_dir)

    metrics_dict = {}
    kfold_tracker = KFoldResultsTracker(save_dir, k_folds)

    for fold, (train_local_idx, val_local_idx) in enumerate(fold_splits):
        # Reset seed before each fold so model initialization is identical
        seed_everything(seed)
        model = create_model_instance(config, data_dir)

        print(f"\nFold {fold + 1}/{k_folds}")
        logger_info.info(f"Fold {fold + 1}/{k_folds}")

        fold_metrics = train_single_fold(
            config=config,
            model=model,
            dm=dm,
            fold=fold,
            train_local_idx=train_local_idx,
            val_local_idx=val_local_idx,
            test_idx=test_idx,
            train_val_idx=train_val_idx,
            save_dir=save_dir,
            trainable_params=trainable_params,
            cluster=cluster,
            resume_ckpt=resume_ckpt,
            warm_start_ckpt=warm_start_ckpt,
            experiment_tag=experiment_tag,
            debug=debug,
            best=best,
        )

        metrics_dict[fold] = fold_metrics

        best_ckpt_path = fold_metrics.pop("_best_checkpoint_path", None)
        kfold_tracker.add_fold_result(fold, fold_metrics, best_ckpt_path)

    # Convert to DataFrame
    df_metric = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df_metric = df_metric.applymap(
        lambda x: x.item() if isinstance(x, torch.Tensor) else x
    )

    _run_post_training_evaluations(config, save_dir, data_dir)

    return df_metric


# =============================================================================
# REGISTRY HELPERS  (unchanged public API)
# =============================================================================

def get_model_class(config: dict):
    """
    Return the model class (not instance) from the MODEL_REGISTRY.

    Args:
        config: Configuration dictionary.

    Returns:
        LightningModule subclass.
    """
    model_obj = config["model"]["model_object"]
    available_models = [
        "proT", "StageCausaliT", "SingleCausalLayer",
        "SingleCausalLayerRes",
        "NoiseAwareSingleCausalLayer", "NoiseAwareSingleCausalLayerRes",
        "VarianceCausalLayer",
        "AttentionSelectorLayer",
        "LSTM", "GRU", "TCN", "MLP",
    ]

    assert model_obj in available_models, (
        f"{model_obj} unavailable! Choose between {available_models}"
    )
    MODEL_REGISTRY = {
        "proT": TransformerForecaster,
        "StageCausaliT": StageCausalForecaster,
        "SingleCausalLayer": SingleCausalForecaster,
        "SingleCausalLayerRes": SingleCausalResForecaster,
        "NoiseAwareSingleCausalLayer": NoiseAwareCausalForecaster,
        "NoiseAwareSingleCausalLayerRes": NoiseAwareCausalResForecaster,
        "VarianceCausalLayer": VarianceCausalForecaster,
        "AttentionSelectorLayer": AttentionSelectorForecaster,
    }
    return MODEL_REGISTRY[model_obj]



def create_model_instance(config: dict, data_dir: str = None) -> pl.LightningModule:
    """
    Instantiate the model specified in the configuration.

    Args:
        config:   Configuration dictionary.
        data_dir: Data directory (required for models that load hard masks).

    Returns:
        Instantiated LightningModule.
    """
    model_obj = config["model"]["model_object"]

    if model_obj == "StageCausaliT":
        return StageCausalForecaster(config, data_dir=data_dir)
    elif model_obj == "SingleCausalLayer":
        return SingleCausalForecaster(config, data_dir=data_dir)
    elif model_obj == "SingleCausalLayerRes":
        return SingleCausalResForecaster(config, data_dir=data_dir)
    elif model_obj == "NoiseAwareSingleCausalLayer":
        return NoiseAwareCausalForecaster(config, data_dir=data_dir)
    elif model_obj == "NoiseAwareSingleCausalLayerRes":
        return NoiseAwareCausalResForecaster(config, data_dir=data_dir)
    elif model_obj == "VarianceCausalLayer":
        return VarianceCausalForecaster(config, data_dir=data_dir)
    elif model_obj == "AttentionSelectorLayer":
        return AttentionSelectorForecaster(config, data_dir=data_dir)
    elif model_obj == "proT":

        return TransformerForecaster(config)
    else:
        return get_model_class(config)(config)


def resolve_num_workers(config: dict, cluster: bool) -> int:
    """
    Decide how many DataLoader worker processes to use.

    Rules, in order:

    1. ``training.num_workers`` in the config wins (escape hatch, incl. 0).
    2. Windows -> 0.  Workers are spawned there (no fork), so each one re-imports
       torch and re-materializes the dataset; with a double-digit worker count
       this reliably dies with "DataLoader worker exited unexpectedly", which is
       what used to abort every Optuna trial of a dagsweep run.
    3. Cluster -> 1 (one CPU is requested per task).
    4. Otherwise -> min(8, cpu_count // 2), never more than the machine can feed.

    The previous hardcoded ``20`` off-cluster was both unsafe on Windows and
    wasteful for the small datasets used by the sizing trials.
    """
    declared = config.get("training", {}).get("num_workers", None)
    if declared is not None:
        return max(0, int(declared))
    if sys.platform.startswith("win"):
        return 0
    if cluster:
        return 1
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def get_dataloader(config: dict, data_dir: str, cluster: bool, seed: int):
    """
    Instantiate the appropriate DataModule for the model type.

    Args:
        config:   Configuration dictionary.
        data_dir: Path to data directory.
        cluster:  Whether running on cluster (affects num_workers).
        seed:     Random seed.

    Returns:
        DataModule instance (ProcessDataModule or StageCausalDataModule).
    """
    model_obj = config["model"]["model_object"]

    DATALOADER_REGISTRY = {
        "proT": ProcessDataModule,
        "StageCausaliT": StageCausalDataModule,
        "SingleCausalLayer": StageCausalDataModule,
        "SingleCausalLayerRes": StageCausalDataModule,
        "NoiseAwareSingleCausalLayer": StageCausalDataModule,
        "NoiseAwareSingleCausalLayerRes": StageCausalDataModule,
        "VarianceCausalLayer": StageCausalDataModule,
        "AttentionSelectorLayer": StageCausalDataModule,
        "LSTM": ProcessDataModule,

        "GRU": ProcessDataModule,
        "TCN": ProcessDataModule,
        "MLP": ProcessDataModule,
    }
    DataModuleClass = DATALOADER_REGISTRY.get(model_obj, ProcessDataModule)

    if model_obj in ["StageCausaliT", "SingleCausalLayer", "SingleCausalLayerRes", "NoiseAwareSingleCausalLayer", "NoiseAwareSingleCausalLayerRes", "VarianceCausalLayer", "AttentionSelectorLayer"]:

        return DataModuleClass(
            data_dir=join(data_dir, config["data"]["dataset"]),
            input_file=config["data"]["filename_input"],
            batch_size=config["training"]["batch_size"],
            num_workers=resolve_num_workers(config, cluster),
            data_format="float32",
            max_data_size=config["data"]["max_data_size"],
            seed=seed,
            train_file=config["data"].get("train_file", None),
            test_file=config["data"].get("test_file", None),
        )
    else:
        return DataModuleClass(
            data_dir=join(data_dir, config["data"]["dataset"]),
            input_file=config["data"]["filename_input"],
            target_file=config["data"]["filename_target"],
            batch_size=config["training"]["batch_size"],
            num_workers=resolve_num_workers(config, cluster),
            data_format="float32",
            max_data_size=config["data"]["max_data_size"],
            seed=seed,
            train_file=config["data"].get("train_file", None),
            test_file=config["data"].get("test_file", None),
        )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import re
    from causaliT.paths import ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR

    exp_dir = EXPERIMENTS_DIR / "SoftMax_scm4"
    data_dir = str(DATA_DIR)

    pattern_config = re.compile(r"config_.*\.yaml")
    config_matching_files = [
        file for file in os.listdir(exp_dir) if pattern_config.match(file)
    ]
    if len(config_matching_files) == 1:
        config = OmegaConf.load(join(exp_dir, config_matching_files[0]))
    else:
        raise ValueError(
            f"None or more than one config file found in {exp_dir}"
        )

    config_updated = update_config(config)

    trainer(
        config=config_updated,
        data_dir=data_dir,
        save_dir=str(exp_dir),
        experiment_tag="test",
        cluster=False,
        resume_ckpt=None,
        debug=True,
    )
