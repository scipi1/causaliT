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
            "val_mae"       : float(m.get("val_mae",  float("nan"))),
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


early_stopping_callbacks = EarlyStopping(
    monitor="val_mae",
    min_delta=1E-5,
    patience=50,
    verbose=True, 
    mode="min"
)         


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


class BestCheckpointCallback(Callback):
    """Callback to save the best checkpoint based on val_mae and store associated metrics."""
    
    def __init__(self, save_dir: str, monitor: str = "val_mae", mode: str = "min"):
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
                
                self.best_checkpoint_path = join(self.checkpoint_dir, "best_checkpoint.ckpt")
                trainer.save_checkpoint(self.best_checkpoint_path)
    
    def on_test_end(self, trainer, pl_module):
        """Save the final best metrics including test metrics after testing is complete."""
        if self.best_metrics:
            current_metrics = trainer.logged_metrics
            test_metrics = {k: float(v) for k, v in current_metrics.items() if k.startswith('test_')}
            
            final_best_metrics = {**self.best_metrics, **test_metrics}
            
            best_metrics_path = join(self.save_dir, "best_metrics.json")
            metrics_to_save = {
                **final_best_metrics,
                "best_epoch": self.best_epoch,
                "best_checkpoint_path": self.best_checkpoint_path
            }
            
            with open(best_metrics_path, 'w') as f:
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
    """Class to track and aggregate results across all k-folds."""
    
    def __init__(self, save_dir: str, k_folds: int):
        self.save_dir = save_dir
        self.k_folds = k_folds
        self.fold_results = {}
        self.summary_file = join(save_dir, "kfold_summary.json")
    
    def add_fold_result(self, fold_num: int, metrics: dict, best_checkpoint_path: str = None):
        """Add results for a specific fold."""
        self.fold_results[fold_num] = {
            "metrics": metrics,
            "best_checkpoint_path": best_checkpoint_path,
            "fold_dir": join(self.save_dir, f"k_{fold_num}")
        }
        
        self._update_summary()
    
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
        
        if "val_mae" in metric_names:
            best_fold = min(self.fold_results.keys(), 
                           key=lambda x: self.fold_results[x]["metrics"]["val_mae"])
            summary["best_fold"] = {
                "fold_number": best_fold,
                "val_mae": self.fold_results[best_fold]["metrics"]["val_mae"],
                "metrics": self.fold_results[best_fold]["metrics"],
                "checkpoint_path": self.fold_results[best_fold]["best_checkpoint_path"]
            }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_summary(self):
        """Get the current summary of all folds."""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {}
