"""
Base predictor: checkpoint loading, SX data module and prediction loop.

Subclasses implement ``_load_model``, ``_forward`` and
``_process_forward_output`` (must return ``pred_x`` and optionally
``attention_weights`` / ``pred_s``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from os.path import join

from causaliT.training.dataloader_sx import SXDataModule


@dataclass
class PredictionResult:
    """
    Container for prediction outputs.
    
    Attributes:
        inputs: Input data array (B x L x F)
        outputs: Model predictions (B x L) or (B x L x F')
        targets: Target data array (B x L x F)
        attention_weights: Optional dict with 'encoder', 'decoder', 'cross' keys (None for baseline)
        metadata: Optional dict with model info, dataset info, etc.
    """
    inputs: np.ndarray
    outputs: np.ndarray
    targets: np.ndarray
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy access"""
        return {
            'inputs': self.inputs,
            'outputs': self.outputs,
            'targets': self.targets,
            'attention_weights': self.attention_weights,
            'metadata': self.metadata
        }


class BasePredictor(ABC):
    """
    Abstract base class for model predictors.
    
    Provides common functionality for:
    - Loading models from checkpoints
    - Creating data modules
    - Running predictions
    - Processing outputs
    
    Subclasses must implement:
    - _load_model(): Load the correct model type
    - _process_forward_output(): Extract relevant outputs from model
    """
    
    def __init__(self, config: dict, checkpoint_path: Path, datadir_path: Path = None):
        """
        Initialize predictor.
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            datadir_path: Path to data directory (optional)
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.datadir_path = datadir_path
        
        # Load model
        self.model = self._load_model()
        self.device = self._get_device()
        
        # Set model to eval mode
        self.model.to(self.device)
        self.model.eval()
        
    @abstractmethod
    def _load_model(self) -> pl.LightningModule:
        """
        Load the appropriate model from checkpoint.
        
        Returns:
            Loaded PyTorch Lightning model
        """
        pass
    
    @abstractmethod
    def _process_forward_output(self, output: Any) -> Dict[str, Any]:
        """
        Process the output from model.forward().
        
        Args:
            output: Raw output from model forward pass
            
        Returns:
            Dict with 'pred_x' and optionally 'attention_weights'
        """
        pass
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_data_module(
        self,
        external_dataset: dict = None,
        cluster: bool = False
    ) -> SXDataModule:
        """
        ``SXDataModule`` of the config dataset, or of ``external_dataset``
        (dict with ``dataset`` and optional ``filename_input``).
        """
        cfg_d, cfg_t = self.config["data"], self.config["training"]
        common = dict(batch_size=cfg_t["batch_size"], num_workers=1 if cluster else 20,
                      data_format="float32", seed=cfg_t["seed"], use_val_split=False)
        if external_dataset is not None:
            return SXDataModule(
                data_dir=join(self.datadir_path, external_dataset["dataset"]),
                input_file=external_dataset.get("filename_input", "ds.npz"),
                train_file=None, test_file=None, **common)

        dm = SXDataModule(
            data_dir=join(self.datadir_path, cfg_d["dataset"]),
            input_file=cfg_d["filename_input"],
            max_data_size=cfg_d.get("max_data_size", None),
            train_file=cfg_d.get("train_file", None),
            test_file=cfg_d.get("test_file", None), **common)
        # Explicit test indices (only for non-pre-split data).
        if (cfg_d.get("test_ds_idx") is not None
                and cfg_d.get("train_file") is None and cfg_d.get("test_file") is None):
            test_idx = np.load(join(self.datadir_path, cfg_d["dataset"], cfg_d["test_ds_idx"]))
            dm.update_idx(train_idx=None, val_idx=None, test_idx=test_idx)
        return dm

    def predict(
        self,
        dm: SXDataModule,
        dataset_label: str = "test",
        debug_flag: bool = False,
        input_conditioning_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ) -> PredictionResult:
        """
        Predict on ``dataset_label`` (train | test | all).

        ``input_conditioning_fn`` transforms S before the forward pass (e.g.
        interventions); ``kwargs`` go to ``_forward``.  Returns inputs = S,
        outputs = pred_x, targets = X, plus concatenated attention weights.
        """
        assert dataset_label in ["train", "test", "all"], f"Invalid dataset label: {dataset_label}"
        dm.prepare_data()
        dm.setup(stage=None)
        loader = {"train": dm.train_dataloader, "test": dm.test_dataloader,
                  "all": dm.all_dataloader}[dataset_label]()

        s_list, x_list, pred_list = [], [], []
        attention = defaultdict(list)
        for batch in tqdm(loader, desc=f"Predicting [{dataset_label}]"):
            batch = [item.to(self.device) for item in batch]
            S, X = batch[0], batch[1]                # Y (if present) is ignored
            if input_conditioning_fn is not None:
                S = input_conditioning_fn(S)
            with torch.no_grad():
                processed = self._process_forward_output(self._forward(S, X, **kwargs))
            s_list.append(S.cpu())
            x_list.append(X.cpu())
            pred_list.append(processed["pred_x"].cpu())
            for key, att in (processed.get("attention_weights") or {}).items():
                if att is not None:
                    attention[key].append(att.cpu())
            if debug_flag:
                break

        s_arr = torch.cat(s_list).numpy().squeeze()
        attention_weights = {k: torch.cat(v).numpy().squeeze()
                             for k, v in attention.items() if v} or None
        return PredictionResult(
            inputs=s_arr,
            outputs=torch.cat(pred_list).numpy().squeeze(),
            targets=torch.cat(x_list).numpy().squeeze(),
            attention_weights=attention_weights,
            metadata={
                "model_type": self.config["model"]["model_object"],
                "dataset_label": dataset_label,
                "batch_size": self.config["training"]["batch_size"],
                "num_samples": len(s_arr),
            },
        )

    @abstractmethod
    def _forward(self, S: torch.Tensor, X: torch.Tensor, **kwargs) -> Any:
        """Raw model output for one batch (X is blanked inside the model)."""
