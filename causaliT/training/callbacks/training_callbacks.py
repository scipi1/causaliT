# Standard library imports
import datetime
import json
import logging
import os
import sys
import time
from os.path import join, dirname, abspath
from pathlib import Path

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Local imports
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))


class PerRunManifest(pl.Callback):
    def __init__(self, config, path, tag=""):
        self.config = config
        self.tag    = tag
        self.path   = path
        self.manifest_f = Path(ROOT_DIR) / "logs" / "manifest.ldjson"
        self.record = None
    
    def _gather_common(self):
        return {
            "timestamp" : datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "model"     : self.config["model"]["model_object"],
            "dataset"   : self.config["data"]["dataset"],    
            "tag"       : self.tag,
            "path"      : self.path
        }
            
    def _append(self, fields: dict):
        if self.record is None:
            self.record = {**self._gather_common(), **fields}
        else:
            self.record.update(fields)
        self.manifest_f.parent.mkdir(parents=True, exist_ok=True)
        
    def _write_manifest(self):
        with open(self.manifest_f, "a") as f:
            f.write(json.dumps(self.record, default=str) + "\n")
    
    def _elapsed(self):
        return time.time() - getattr(self, "_fit_start_time", time.time())
    
    def on_fit_start(self,trainer, pl_module):
        self._fit_start_time = time.time()
    
    def on_fit_end(self, trainer, pl_module):
        m = trainer.logged_metrics
        epochs_run = trainer.current_epoch
        self._append({
            "val_loss"      : float(m.get("val_loss", float("nan"))),
            "val_x_mae"       : float(m.get("val_x_mae",  float("nan"))),
            "val_r2"        : float(m.get("val_r2",   float("nan"))),
            "val_rmse"      : float(m.get("val_rmse", float("nan"))),
            "train_seconds" : round(self._elapsed(), 2),
            "epochs"        : epochs_run,
        })

    def on_test_end(self, trainer, pl_module):
        m = trainer.logged_metrics
        self._append({
            "test_loss" : float(m.get("test_loss", float("nan"))),
            "test_mae"  : float(m.get("test_mae", float("nan"))),
            "test_r2"   : float(m.get("test_r2", float("nan"))),
            "test_rmse" : float(m.get("test_rmse", float("nan")))
        })
        self._write_manifest()


# DEPRECATED: Module-level early stopping instance.
# Use get_early_stopping_callback(config) for config-driven early stopping.
early_stopping_callbacks = EarlyStopping(
    monitor="val_x_mae",
    min_delta=1E-5,
    patience=50,
    verbose=True, 
    mode="min"
)


def get_early_stopping_callback(config: dict):
    """
    Create an EarlyStopping callback from the training config.

    Reads ``config["training"]["early_stopping"]`` and returns an
    ``EarlyStopping`` instance, or ``None`` if early stopping is disabled.

    Config example::

        training:
          early_stopping:
            enabled: true
            monitor: "val_x_mae"
            patience: 50
            min_delta: 1.0e-5
            mode: "min"

    Falls back to the legacy ``special.mode: ["early_stopping"]`` check
    for backward compatibility.

    Args:
        config: Full experiment configuration dict.

    Returns:
        ``EarlyStopping`` callback or ``None``.
    """
    # New config-driven approach
    es_config = config.get("training", {}).get("early_stopping", {})
    if es_config.get("enabled", False):
        return EarlyStopping(
            monitor=es_config.get("monitor", "val_x_mae"),
            min_delta=float(es_config.get("min_delta", 1e-5)),
            patience=int(es_config.get("patience", 50)),
            verbose=True,
            mode=es_config.get("mode", "min"),
        )

    # Legacy fallback: special.mode list
    special_modes = config.get("special", {}).get("mode", [])
    if "early_stopping" in (special_modes or []):
        return early_stopping_callbacks

    return None


def get_checkpoint_callback(experiment_dir: str, save_ckpt_every_n_epochs: int):
    checkpoint_dir = join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    periodic_ckpt = ModelCheckpoint(
        dirpath     = checkpoint_dir,
        filename    = "{epoch}-{train_loss:.2f}",
        every_n_epochs = save_ckpt_every_n_epochs,
        save_top_k = -1,
        monitor    = "val_loss",
        mode       = "min",
        save_last  = True,
    )

    class SaveInitial(Callback):
        """Dump weights before the first optimization step."""
        @rank_zero_only
        def on_fit_start(self, trainer, pl_module):
            trainer.save_checkpoint(join(checkpoint_dir, "epoch0-initial.ckpt"))

    return [SaveInitial(), periodic_ckpt]


class MemoryLoggerCallback(Callback):
    
    def log_memory(self, stage):
        """Logs CPU & GPU memory usage."""
        allocated_gpu = torch.cuda.memory_allocated() / 1e9
        reserved_gpu = torch.cuda.memory_reserved() / 1e9
        logger_memory = logging.getLogger("logger_memory")
        logger_memory.info(
            f"[{stage}] GPU Allocated: {allocated_gpu:.2f} GB | GPU Reserved: {reserved_gpu:.2f} GB | "
        )

    def on_train_start(self, trainer, pl_module):
        self.log_memory("TRAIN START")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_memory(f"EPOCH {trainer.current_epoch} START")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_memory(f"BATCH {batch_idx} START")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_memory(f"BATCH {batch_idx} END")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_memory(f"EPOCH {trainer.current_epoch} END")

    def on_train_end(self, trainer, pl_module):
        self.log_memory("TRAIN END")
        logger_memory = logging.getLogger("logger_memory")
        logger_memory.info(f"Max allocated GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        logger_memory.info(f"Max reserved GPU: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


# Backward-compatible alias
BestCheckpointCallback = None  # Removed — use BestReconstructionCheckpoint


class BestReconstructionCheckpoint(Callback):
    """
    Save the checkpoint with the best reconstruction quality (lowest val_x_mae).

    Saves ``best_reconstruction_checkpoint.ckpt`` whenever the monitored
    metric improves, and writes ``best_reconstruction_metrics.json`` at
    test end.

    This is the counterpart of ``BestCausalCheckpoint`` (which tracks HSIC).
    For baseline models (no HSIC), this is the primary checkpoint for ATE
    evaluation.  For causal models, use ``BestCausalCheckpoint`` instead.
    """
    
    def __init__(self, save_dir: str, monitor: str = "val_x_mae", mode: str = "min"):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        self.best_metrics = {}
        self.best_epoch = 0
        self.best_checkpoint_path = None
        
        self.checkpoint_dir = join(save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _is_better(self, current_value):
        """Check if current metric value is better than the best so far."""
        if self.mode == 'min':
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check if current epoch has the best validation metric and save if so."""
        current_metrics = trainer.logged_metrics
        
        if self.monitor in current_metrics:
            current_value = float(current_metrics[self.monitor])
            
            if self._is_better(current_value):
                self.best_metric_value = current_value
                self.best_epoch = trainer.current_epoch
                
                self.best_metrics = {
                    key: float(value) if isinstance(value, torch.Tensor) else value
                    for key, value in current_metrics.items()
                }
                
                self.best_checkpoint_path = join(
                    self.checkpoint_dir, "best_reconstruction_checkpoint.ckpt"
                )
                trainer.save_checkpoint(self.best_checkpoint_path)
    
    def on_test_end(self, trainer, pl_module):
        """Save the final best metrics including test metrics after testing is complete."""
        if self.best_metrics:
            current_metrics = trainer.logged_metrics
            test_metrics = {k: float(v) for k, v in current_metrics.items() if k.startswith('test_')}
            
            final_best_metrics = {**self.best_metrics, **test_metrics}
            
            best_metrics_path = join(self.save_dir, "best_reconstruction_metrics.json")
            metrics_to_save = {
                **final_best_metrics,
                "best_epoch": self.best_epoch,
                "best_checkpoint_path": self.best_checkpoint_path
            }
            
            with open(best_metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)


class BestCausalCheckpoint(Callback):
    """
    Save the checkpoint with the lowest HSIC (best causal structure) during training.

    Monitors HSIC metrics in priority order: val_hsic_reg → val_hsic_cross → val_hsic.
    Saves ``best_causal_checkpoint.ckpt`` whenever the monitored HSIC improves,
    and writes ``best_causal_metrics.json`` at test end.

    This is the counterpart of ``BestCheckpointCallback`` (which tracks val_x_mae).
    The best-causal checkpoint is useful for downstream tasks like ATE estimation,
    where the model with the most accurate causal structure is preferred over the
    model with the lowest prediction error.
    """

    # HSIC metrics to try, in order of preference
    HSIC_METRICS_PRIORITY = ["val_hsic_reg", "val_hsic_cross", "val_hsic"]

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.best_hsic_value = float('inf')
        self.best_metrics = {}
        self.best_epoch = 0
        self.best_checkpoint_path = None
        self.monitor_key = None  # resolved on first validation

        self.checkpoint_dir = join(save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _resolve_monitor_key(self, metric_names):
        """Find the first available HSIC metric from the logged metrics."""
        for key in self.HSIC_METRICS_PRIORITY:
            if key in metric_names:
                return key
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save checkpoint if current epoch has the lowest HSIC so far."""
        current_metrics = trainer.logged_metrics

        # Lazily resolve which HSIC metric is available
        if self.monitor_key is None:
            self.monitor_key = self._resolve_monitor_key(current_metrics.keys())
        if self.monitor_key is None:
            return  # no HSIC metric logged — nothing to do

        if self.monitor_key not in current_metrics:
            return

        current_value = float(current_metrics[self.monitor_key])
        if current_value < self.best_hsic_value:
            self.best_hsic_value = current_value
            self.best_epoch = trainer.current_epoch

            self.best_metrics = {
                key: float(value) if isinstance(value, torch.Tensor) else value
                for key, value in current_metrics.items()
            }

            self.best_checkpoint_path = join(
                self.checkpoint_dir, "best_causal_checkpoint.ckpt"
            )
            trainer.save_checkpoint(self.best_checkpoint_path)

    def on_test_end(self, trainer, pl_module):
        """Save best-causal metrics JSON after testing is complete."""
        if self.best_metrics:
            current_metrics = trainer.logged_metrics
            test_metrics = {
                k: float(v) for k, v in current_metrics.items() if k.startswith('test_')
            }

            final_best_metrics = {**self.best_metrics, **test_metrics}

            best_causal_path = join(self.save_dir, "best_causal_metrics.json")
            metrics_to_save = {
                **final_best_metrics,
                "best_causal_epoch": self.best_epoch,
                "monitor_key": self.monitor_key,
                "best_causal_checkpoint_path": self.best_checkpoint_path,
            }

            with open(best_causal_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)


class DataIndexTracker(Callback):
    """Callback to save train/validation/test data indices for each fold."""
    
    def __init__(self, save_dir: str, fold_num: int, train_idx, val_idx, test_idx):
        super().__init__()
        self.save_dir = save_dir
        self.fold_num = fold_num
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    def on_fit_start(self, trainer, pl_module):
        """Save data indices at the start of training."""
        train_indices_path = join(self.save_dir, f"fold_{self.fold_num}_train_indices.npy")
        np.save(train_indices_path, self.train_idx)
        
        val_indices_path = join(self.save_dir, f"fold_{self.fold_num}_val_indices.npy")
        np.save(val_indices_path, self.val_idx)
        
        test_indices_path = join(self.save_dir, f"fold_{self.fold_num}_test_indices.npy")
        np.save(test_indices_path, self.test_idx)


class KFoldResultsTracker:
    """
    Class to track and aggregate results across all k-folds.
    
    Best fold selection strategy (for causal inference):
    - Primary: Minimum val_hsic_reg (independence between residuals and parents)
    - Fallback: val_hsic_cross, val_hsic, then val_x_mae if no HSIC available
    
    Rationale: Within a seed, CV fold results are clustered together. Minimum 
    validation HSIC rarely corresponds to worst SHD (structural Hamming distance),
    making it a reliable proxy for causal structure quality when ground truth is unavailable.
    """
    
    # HSIC metrics to try, in order of preference
    HSIC_METRICS_PRIORITY = ["val_hsic_reg", "val_hsic_cross", "val_hsic"]
    
    def __init__(self, save_dir: str, k_folds: int):
        self.save_dir = save_dir
        self.k_folds = k_folds
        self.fold_results = {}
        self.summary_file = join(save_dir, "kfold_summary.json")
    
    def add_fold_result(self, fold_num: int, metrics: dict, best_checkpoint_path: str = None):
        """Add results for a specific fold."""
        # Convert any tensor values to Python floats for JSON serialization
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                clean_metrics[key] = float(value.cpu().item()) if value.numel() == 1 else value.cpu().tolist()
            elif isinstance(value, (np.ndarray, np.floating, np.integer)):
                clean_metrics[key] = float(value) if np.isscalar(value) or value.ndim == 0 else value.tolist()
            else:
                clean_metrics[key] = value
        
        self.fold_results[fold_num] = {
            "metrics": clean_metrics,
            "best_checkpoint_path": best_checkpoint_path,
            "fold_dir": join(self.save_dir, f"k_{fold_num}")
        }
        
        self._update_summary()
    
    def _find_hsic_metric(self, metric_names: list) -> str:
        """
        Find the best available HSIC metric from the logged metrics.
        
        Returns:
            str: Name of the HSIC metric to use, or None if not found
        """
        for hsic_metric in self.HSIC_METRICS_PRIORITY:
            if hsic_metric in metric_names:
                return hsic_metric
        return None
    
    def _select_best_fold(self, metric_names: list) -> dict:
        """
        Select the best fold using causal-first selection strategy.
        
        Strategy:
        1. Try to select by minimum HSIC (causal quality proxy)
        2. Fall back to minimum val_x_mae (prediction quality)
        
        Returns:
            dict: Best fold info with selection_criterion field
        """
        # Try HSIC-based selection first (causal inference default)
        hsic_metric = self._find_hsic_metric(metric_names)
        
        if hsic_metric is not None:
            # Filter folds with valid (non-NaN, non-None) HSIC values
            valid_folds = [
                fold for fold in self.fold_results.keys()
                if self.fold_results[fold]["metrics"].get(hsic_metric) is not None
                and isinstance(self.fold_results[fold]["metrics"].get(hsic_metric), (int, float))
                and not np.isnan(self.fold_results[fold]["metrics"].get(hsic_metric))
            ]
            
            if valid_folds:
                best_fold = min(valid_folds,
                               key=lambda x: self.fold_results[x]["metrics"][hsic_metric])
                return {
                    "fold_number": best_fold,
                    "selection_criterion": hsic_metric,
                    "selection_value": self.fold_results[best_fold]["metrics"][hsic_metric],
                    "metrics": self.fold_results[best_fold]["metrics"],
                    "checkpoint_path": self.fold_results[best_fold]["best_checkpoint_path"]
                }
        
        # Fallback to val_x_mae (prediction-based selection)
        if "val_x_mae" in metric_names:
            valid_folds = [
                fold for fold in self.fold_results.keys()
                if self.fold_results[fold]["metrics"].get("val_x_mae") is not None
                and isinstance(self.fold_results[fold]["metrics"].get("val_x_mae"), (int, float))
                and not np.isnan(self.fold_results[fold]["metrics"].get("val_x_mae"))
            ]
            
            if valid_folds:
                best_fold = min(valid_folds,
                               key=lambda x: self.fold_results[x]["metrics"]["val_x_mae"])
                return {
                    "fold_number": best_fold,
                    "selection_criterion": "val_x_mae",
                    "selection_value": self.fold_results[best_fold]["metrics"]["val_x_mae"],
                    "metrics": self.fold_results[best_fold]["metrics"],
                    "checkpoint_path": self.fold_results[best_fold]["best_checkpoint_path"]
                }
        
        # Last resort: return first fold
        first_fold = min(self.fold_results.keys())
        return {
            "fold_number": first_fold,
            "selection_criterion": "first_available",
            "selection_value": None,
            "metrics": self.fold_results[first_fold]["metrics"],
            "checkpoint_path": self.fold_results[first_fold]["best_checkpoint_path"]
        }
    
    def _update_summary(self):
        """Update the k-fold summary file."""
        if not self.fold_results:
            return
        
        metric_names = list(next(iter(self.fold_results.values()))["metrics"].keys())
        summary = {
            "total_folds": self.k_folds,
            "completed_folds": len(self.fold_results),
            "fold_results": self.fold_results,
            "statistics": {}
        }
        
        for metric_name in metric_names:
            values = [self.fold_results[fold]["metrics"][metric_name] 
                     for fold in self.fold_results.keys()]
            
            if values and all(isinstance(v, (int, float)) for v in values):
                summary["statistics"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Select best fold using causal-first strategy
        summary["best_fold"] = self._select_best_fold(metric_names)
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_summary(self):
        """Get the current summary of all folds."""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {}
