"""
Base predictor class providing common prediction functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from os.path import join

from proT.training.dataloader import ProcessDataModule


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
            Dictionary with 'forecast' and optionally 'attention_weights'
        """
        pass
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_data_module(
        self,
        external_dataset: dict = None,
        cluster: bool = False
    ) -> ProcessDataModule:
        """
        Create data module based on config or external dataset.
        
        Handles both pre-split data (train_file/test_file) and normal data loading,
        matching the behavior in trainer.py.
        
        Args:
            external_dataset: Optional dict with 'dataset', 'filename_input', 'filename_target'
            cluster: Whether running on cluster
            
        Returns:
            ProcessDataModule instance
        """
        seed = self.config["training"]["seed"]
        
        if external_dataset is not None:
            # External dataset: predict from fresh data
            # Feature implemented for the industrial partner
            dm = ProcessDataModule(
                data_dir=join(self.datadir_path, external_dataset["dataset"]),
                input_file=external_dataset["filename_input"],
                target_file=external_dataset["filename_target"],
                batch_size=self.config["training"]["batch_size"],
                num_workers=1 if cluster else 20,
                data_format="float32",
                seed=seed,
                train_file=None,  # External datasets don't use pre-split mode
                test_file=None,
                use_val_split=False,  # For prediction, we typically don't need val split
            )
        else:
            # Normal dataset: use all config parameters including pre-split support
            dm = ProcessDataModule(
                data_dir=join(self.datadir_path, self.config["data"]["dataset"]),
                input_file=self.config["data"]["filename_input"],
                target_file=self.config["data"]["filename_target"],
                batch_size=self.config["training"]["batch_size"],
                num_workers=1 if cluster else 20,
                data_format="float32",
                max_data_size=self.config["data"].get("max_data_size", None),
                seed=seed,
                train_file=self.config["data"].get("train_file", None),
                test_file=self.config["data"].get("test_file", None),
                use_val_split=False,  # For prediction, we typically don't need val split
            )
        
        # Handle test dataset indices (only for non-pre-split data)
        # Pre-split data already has test set defined in test_file
        if not external_dataset and self.config["data"].get("test_ds_ixd") is not None:
            # Only load and use test indices if not using pre-split data
            if self.config["data"].get("train_file") is None and self.config["data"].get("test_file") is None:
                test_idx = np.load(join(self.datadir_path, self.config["data"]["dataset"], 
                                       self.config["data"]["test_ds_ixd"]))
                dm.update_idx(train_idx=None, val_idx=None, test_idx=test_idx)
        
        return dm
    
    def predict(
        self,
        dm: ProcessDataModule,
        dataset_label: str = "test",
        debug_flag: bool = False,
        input_conditioning_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ) -> PredictionResult:
        """
        Run standard prediction on specified dataset.
        
        Args:
            dm: Data module
            dataset_label: One of ["train", "test", "all"]
            debug_flag: If True, predict only one batch
            input_conditioning_fn: Optional function to condition inputs before forward pass.
                                  Function signature: fn(X: torch.Tensor) -> torch.Tensor
            **kwargs: Additional arguments passed to forward
            
        Returns:
            PredictionResult object containing all outputs
        """
        assert dataset_label in ["train", "test", "all"], \
            f"Invalid dataset label: {dataset_label}"
        
        # Prepare data
        dm.prepare_data()
        dm.setup(stage=None)
        
        # Select dataset
        if dataset_label == "train":
            dataset = dm.train_dataloader()
            print("Train dataset selected.")
        elif dataset_label == "test":
            dataset = dm.test_dataloader()
            print("Test dataset selected.")
        elif dataset_label == "all":
            dataset = dm.all_dataloader()
            print("All data selected.")
        
        # Initialize lists for collecting outputs
        input_list = []
        output_list = []
        target_list = []
        attention_dict = {'encoder': [], 'decoder': [], 'cross': []}
        
        # Loop over prediction batches
        print("Predicting...")
        for batch in tqdm(dataset):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) for item in batch]
            else:
                batch = batch.to(self.device)
            
            X, trg = batch
            
            # Apply input conditioning if provided
            if input_conditioning_fn is not None:
                X = input_conditioning_fn(X)
            
            # Forward pass
            with torch.no_grad():
                output = self._forward(X, trg, **kwargs)
            
            # Process output
            processed = self._process_forward_output(output)
            
            # Append batch predictions
            input_list.append(X.cpu())
            output_list.append(processed['forecast'].cpu())
            target_list.append(trg.cpu())
            
            # Collect attention weights if available
            if processed.get('attention_weights') is not None:
                for key in ['encoder', 'decoder', 'cross']:
                    if key in processed['attention_weights']:
                        attention_dict[key].append(processed['attention_weights'][key].cpu())
            
            if debug_flag:
                print("Debug mode: stopping after one batch...")
                break
        
        # Concatenate all batches
        input_tensor = torch.cat(input_list, dim=0)
        output_tensor = torch.cat(output_list, dim=0)
        target_tensor = torch.cat(target_list, dim=0)
        
        # Convert to numpy
        input_array = input_tensor.numpy().squeeze()
        output_array = output_tensor.numpy().squeeze()
        target_array = target_tensor.numpy().squeeze()
        
        # Process attention weights
        attention_weights = None
        if any(attention_dict.values()):
            attention_weights = {}
            for key, val_list in attention_dict.items():
                if val_list:
                    attention_tensor = torch.cat(val_list, dim=0)
                    attention_weights[key] = attention_tensor.numpy().squeeze()
        
        # Create metadata
        metadata = {
            'model_type': self.config["model"]["model_object"],
            'dataset_label': dataset_label,
            'batch_size': self.config["training"]["batch_size"],
            'num_samples': len(input_array),
        }
        
        return PredictionResult(
            inputs=input_array,
            outputs=output_array,
            targets=target_array,
            attention_weights=attention_weights,
            metadata=metadata
        )
    
    @abstractmethod
    def _forward(self, X: torch.Tensor, trg: torch.Tensor, **kwargs) -> Any:
        """
        Perform forward pass through the model.
        
        Args:
            X: Input tensor
            trg: Target tensor
            **kwargs: Additional arguments
            
        Returns:
            Raw model output (format depends on model type)
        """
        pass
