"""
Predictor for StageCausaliT dual-decoder architecture.
"""

from typing import Any, Dict, Optional, Callable
import numpy as np
import torch
from pathlib import Path
from os.path import join
from tqdm import tqdm

from .base_predictor import BasePredictor, PredictionResult
from causaliT.training.forecasters.stage_causal_forecaster import StageCausalForecaster
from causaliT.training.stage_causal_dataloader import StageCausalDataModule
from causaliT.core.utils import load_dag_masks


class StageCausalPredictor(BasePredictor):
    """
    Predictor for StageCausaliT dual-decoder forecasting models.
    
    Handles:
    - StageCausalForecaster with dual decoders
    - Three-input data format (S, X, Y)
    - Dual outputs (pred_x, pred_y)
    - Automatic loading of hard masks if model was trained with them
    
    Returns predictions along with attention weights from both decoders:
    - Decoder 1: S → X (dec1_cross, dec1_self)
    - Decoder 2: X → Y (dec2_cross, dec2_self)
    
    Note:
        The blanking of X and Y values is handled internally by StageCausalForecaster.forward().
        It blanks X[:, :, val_idx] for decoder 1 and Y[:, :, val_idx] for decoder 2.
    """
    
    def _load_model(self) -> StageCausalForecaster:
        """
        Load StageCausalForecaster model from checkpoint.
        
        If the model was trained with hard masks (config has use_hard_masks=True),
        this method also loads the masks from the data directory to ensure
        they are available during inference.
        
        Returns:
            Loaded StageCausalForecaster model with masks loaded if applicable
        """
        model = StageCausalForecaster.load_from_checkpoint(self.checkpoint_path)
        
        # Verify model loaded correctly
        if model is None:
            raise RuntimeError("Model failed to load from checkpoint.")
        
        if not any(param.requires_grad for param in model.parameters()):
            raise RuntimeError("Model parameters seem uninitialized. Check the checkpoint path.")
        
        # If model was trained with hard masks, ensure they are loaded
        if model.use_hard_masks and not model._hard_masks_loaded:
            self._load_hard_masks_for_model(model)
        
        return model
    
    def _load_hard_masks_for_model(self, model: StageCausalForecaster):
        """
        Load hard masks from data directory and register them to the model.
        
        This is called when loading from checkpoint if the model was trained
        with hard masks but they weren't saved/restored properly from the checkpoint.
        
        Args:
            model: StageCausalForecaster model to load masks into
        """
        # Get mask filenames from config
        mask_files = self.config["training"].get("hard_mask_files", None)
        
        if mask_files is None:
            print("Warning: Model was trained with use_hard_masks=True but no hard_mask_files in config.")
            return
        
        if self.datadir_path is None:
            print("Warning: Cannot load hard masks - no datadir_path provided.")
            return
        
        # Construct full data path
        dataset_name = self.config["data"]["dataset"]
        dataset_dir = join(self.datadir_path, dataset_name)
        
        # Load masks
        masks = load_dag_masks(dataset_dir, mask_files, device='cpu')
        
        if masks is not None:
            model._hard_masks = masks
            model._hard_masks_loaded = True
            
            # Register masks as buffers (ensures correct device handling)
            for name, mask in masks.items():
                model.register_buffer(f'hard_mask_{name}', mask)
            
            print(f"✓ Hard masks loaded from data directory for inference.")
        else:
            print("Warning: Failed to load hard masks from data directory.")
    
    def create_data_module(
        self,
        external_dataset: dict = None,
        cluster: bool = False
    ) -> StageCausalDataModule:
        """
        Create StageCausalDataModule for the three-input data format (S, X, Y).
        
        Args:
            external_dataset: Optional dict with 'dataset', 'input_file' keys
            cluster: Whether running on cluster
            
        Returns:
            StageCausalDataModule instance
        """
        seed = self.config["training"]["seed"]
        
        if external_dataset is not None:
            # External dataset case
            dm = StageCausalDataModule(
                data_dir=join(self.datadir_path, external_dataset["dataset"]),
                input_file=external_dataset.get("filename_input", "ds.npz"),
                batch_size=self.config["training"]["batch_size"],
                num_workers=1 if cluster else 20,
                data_format="float32",
                seed=seed,
                train_file=None,
                test_file=None,
                use_val_split=False,
            )
        else:
            # Normal dataset: use config parameters
            # Note: StageCausaliT uses 'filename_input' for the .npz file containing s, x, y keys
            dm = StageCausalDataModule(
                data_dir=join(self.datadir_path, self.config["data"]["dataset"]),
                input_file=self.config["data"]["filename_input"],
                batch_size=self.config["training"]["batch_size"],
                num_workers=1 if cluster else 20,
                data_format="float32",
                max_data_size=self.config["data"].get("max_data_size", None),
                seed=seed,
                train_file=self.config["data"].get("train_file", None),
                test_file=self.config["data"].get("test_file", None),
                use_val_split=False,
            )
        
        # Handle test dataset indices (only for non-pre-split data)
        if not external_dataset and self.config["data"].get("test_ds_idx") is not None:
            if self.config["data"].get("train_file") is None and self.config["data"].get("test_file") is None:
                test_idx = np.load(join(self.datadir_path, self.config["data"]["dataset"], 
                                       self.config["data"]["test_ds_idx"]))
                dm.update_idx(train_idx=None, val_idx=None, test_idx=test_idx)
        
        return dm
    
    def _forward(self, S: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, 
                 disable_hard_masks: bool = False, disable_in_context_masks: bool = False, **kwargs) -> Any:
        """
        Perform forward pass through StageCausalForecaster model.
        
        Note:
            The StageCausalForecaster.forward() method handles blanking internally:
            - X values at val_idx are blanked (set to 0.0) for decoder 1
            - Y values at val_idx are blanked (set to 0.0) for decoder 2
            This ensures the model cannot "cheat" by seeing the target values.
        
        Args:
            S: Source tensor (B x L_s x F)
            X: Intermediate tensor (B x L_x x F) - will be blanked internally
            Y: Target tensor (B x L_y x F) - will be blanked internally
            disable_hard_masks: If True, disables hard masks even if model was trained with them.
                               Useful for ablation studies during inference. Default False = use
                               masks if model was trained with them (model.use_hard_masks).
            disable_in_context_masks: If True, disables in-context masks even if model was trained
                                      with them. Default False = use masks if model was trained
                                      with them (model.use_in_context_masks).
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (pred_x, pred_y, attention_weights, masks, entropies)
        """
        # StageCausalForecaster.forward() handles the blanking of X and Y values internally:
        # - x_blanked = X.clone(); x_blanked[:, :, val_idx] = 0.0
        # - y_blanked = Y.clone(); y_blanked[:, :, val_idx] = 0.0
        # Teacher forcing is automatically disabled during eval mode (self.model.training = False)
        output = self.model.forward(
            data_source=S,
            data_intermediate=X,
            data_target=Y,
            disable_hard_masks=disable_hard_masks,
            disable_in_context_masks=disable_in_context_masks,
            **kwargs
        )
        return output
    
    def _process_forward_output(self, output: Any) -> Dict[str, Any]:
        """
        Process StageCausalForecaster model output.
        
        Args:
            output: Tuple from model.forward():
                (pred_x, pred_y, attention_weights, masks, entropies)
                where attention_weights = (dec1_cross, dec1_self, dec2_cross, dec2_self)
        
        Returns:
            Dictionary with:
                - 'pred_x': X reconstruction predictions
                - 'pred_y': Y prediction (main forecast)
                - 'attention_weights': Dict with 'dec1_cross', 'dec1_self', 'dec2_cross', 'dec2_self'
        """
        pred_x, pred_y, attention_weights_tuple, masks, entropies = output
        
        # Unpack attention weights from both decoders
        dec1_cross_att, dec1_self_att, dec2_cross_att, dec2_self_att = attention_weights_tuple
        
        # Extract attention weights (take first layer for simplicity)
        attention_weights = {
            'dec1_cross': dec1_cross_att[0] if dec1_cross_att else None,
            'dec1_self': dec1_self_att[0] if dec1_self_att else None,
            'dec2_cross': dec2_cross_att[0] if dec2_cross_att else None,
            'dec2_self': dec2_self_att[0] if dec2_self_att else None,
        }
        
        return {
            'pred_x': pred_x,
            'pred_y': pred_y,
            'attention_weights': attention_weights
        }
    
    def predict(
        self,
        dm: StageCausalDataModule,
        dataset_label: str = "test",
        debug_flag: bool = False,
        input_conditioning_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        disable_hard_masks: bool = False,
        disable_in_context_masks: bool = False,
        **kwargs
    ) -> PredictionResult:
        """
        Run prediction on specified dataset for StageCausaliT model.
        
        This overrides the base predict() method to handle the three-input
        data format (S, X, Y) used by StageCausalForecaster.
        
        Note:
            The blanking of X and Y values is handled internally by the model's
            forward method. X[:, :, val_idx] and Y[:, :, val_idx] are set to 0.0
            before being passed to the decoders.
        
        Args:
            dm: StageCausalDataModule instance
            dataset_label: One of ["train", "test", "all"]
            debug_flag: If True, predict only one batch
            input_conditioning_fn: Optional function to condition S inputs before forward pass
            disable_hard_masks: If True, disables hard masks even if model was trained with them.
                               Useful for ablation studies during inference. Default False = use
                               masks if model was trained with them (model.use_hard_masks).
            disable_in_context_masks: If True, disables in-context masks even if model was trained
                                      with them. Default False = use masks if model was trained
                                      with them (model.use_in_context_masks).
            **kwargs: Additional arguments passed to forward
            
        Returns:
            PredictionResult object containing:
                - inputs: S (source data)
                - outputs: pred_y (Y predictions)
                - targets: Y (Y actual)
                - attention_weights: Dict with dec1_cross, dec1_self, dec2_cross, dec2_self
                - metadata: Contains pred_x, targets_x (X actual), and other info
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
        s_list = []      # Source inputs (S)
        x_list = []      # Intermediate targets (X)
        y_list = []      # Target outputs (Y)
        pred_x_list = [] # X predictions
        pred_y_list = [] # Y predictions
        
        attention_dict = {
            'dec1_cross': [],
            'dec1_self': [],
            'dec2_cross': [],
            'dec2_self': []
        }
        
        # Loop over prediction batches
        print("Predicting...")
        for batch in tqdm(dataset):
            # Move batch to device - expect (S, X, Y) tuple
            if isinstance(batch, (list, tuple)):
                batch = [item.to(self.device) for item in batch]
            else:
                raise ValueError("Expected batch to be a tuple of (S, X, Y)")
            
            S, X, Y = batch
            
            # Apply input conditioning to S if provided
            if input_conditioning_fn is not None:
                S = input_conditioning_fn(S)
            
            # Forward pass - model handles blanking of X and Y internally
            with torch.no_grad():
                output = self._forward(S, X, Y, disable_hard_masks=disable_hard_masks, 
                                       disable_in_context_masks=disable_in_context_masks, **kwargs)
            
            # Process output
            processed = self._process_forward_output(output)
            
            # Append batch data
            s_list.append(S.cpu())
            x_list.append(X.cpu())
            y_list.append(Y.cpu())
            pred_x_list.append(processed['pred_x'].cpu())
            pred_y_list.append(processed['pred_y'].cpu())
            
            # Collect attention weights if available
            if processed.get('attention_weights') is not None:
                for key in ['dec1_cross', 'dec1_self', 'dec2_cross', 'dec2_self']:
                    if key in processed['attention_weights'] and processed['attention_weights'][key] is not None:
                        attention_dict[key].append(processed['attention_weights'][key].cpu())
            
            if debug_flag:
                print("Debug mode: stopping after one batch...")
                break
        
        # Concatenate all batches
        s_tensor = torch.cat(s_list, dim=0)
        x_tensor = torch.cat(x_list, dim=0)
        y_tensor = torch.cat(y_list, dim=0)
        pred_x_tensor = torch.cat(pred_x_list, dim=0)
        pred_y_tensor = torch.cat(pred_y_list, dim=0)
        
        # Convert to numpy
        s_array = s_tensor.numpy().squeeze()
        x_array = x_tensor.numpy().squeeze()
        y_array = y_tensor.numpy().squeeze()
        pred_x_array = pred_x_tensor.numpy().squeeze()
        pred_y_array = pred_y_tensor.numpy().squeeze()
        
        # Process attention weights
        attention_weights = None
        if any(attention_dict.values()):
            attention_weights = {}
            for key, val_list in attention_dict.items():
                if val_list:
                    attention_tensor = torch.cat(val_list, dim=0)
                    attention_weights[key] = attention_tensor.numpy().squeeze()
        
        # Create metadata with X reconstruction info
        metadata = {
            'model_type': self.config["model"]["model_object"],
            'dataset_label': dataset_label,
            'batch_size': self.config["training"]["batch_size"],
            'num_samples': len(s_array),
            'pred_x': pred_x_array,      # X reconstruction predictions
            'targets_x': x_array,         # X actual (intermediate target)
        }
        
        return PredictionResult(
            inputs=s_array,           # S (source data)
            outputs=pred_y_array,     # pred_y (Y predictions)
            targets=y_array,          # Y (Y actual)
            attention_weights=attention_weights,
            metadata=metadata
        )
