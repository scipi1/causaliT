import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataset import TensorDataset
import pytorch_lightning as pl

# from lightning.pytorch.core import LightningDataModule

from os.path import join



class ProcessDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and managing process data.
    
    Handles loading data from numpy files, creating train/val/test splits,
    and providing DataLoaders for training. Supports both automatic splitting
    and manual index-based splitting for k-fold cross-validation.
    """
    def __init__(
        self,
        data_dir: str, 
        input_file: str,
        target_file: str, 
        batch_size: int, 
        num_workers: int,
        data_format: str,
        max_data_size: int=None,
        seed:int=42,
        train_file: str=None,
        test_file: str=None,
        use_val_split: bool=True,
        ) -> None:
        
        super().__init__()
        
        self.data_dir = data_dir
        self.input_file = input_file
        self.target_file = target_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_format = data_format
        self.max_data_size = max_data_size
        self.seed = seed
        self.train_file = train_file
        self.test_file = test_file
        self.use_val_split = use_val_split
        # Store data as tensors (converted in prepare_data)
        self.X_tensor = None
        self.Y_tensor = None
        self.X_train_tensor = None
        self.Y_train_tensor = None
        self.X_test_tensor = None
        self.Y_test_tensor = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.ds_length = None
        
        
    def prepare_data(self) -> None:
        """
        Load data from numpy files and convert to PyTorch tensors.
        
        Supports two modes:
        - Pre-split data: Load separate train/test files
        - Normal data: Load single dataset file for later splitting
        """
        # Check if pre-split data is provided
        if self.train_file is not None or self.test_file is not None:
            print("Loading pre-split data.")
            
            # Reset any indices to prevent further splitting
            self.train_idx = None
            self.val_idx = None
            self.test_idx = None
            
            if self.train_file is not None and self.test_file is not None:
                
                # TRAIN
                train_loaded = np.load(join(self.data_dir, self.train_file), allow_pickle=True, mmap_mode='r')
                X_train_np = train_loaded['x']
                Y_train_np = train_loaded['y']
                print("Train input shape: ", X_train_np.shape)
                print("Train target shape: ", Y_train_np.shape)
                
                if self.max_data_size is not None:
                    X_train_np = X_train_np[:self.max_data_size]
                    Y_train_np = Y_train_np[:self.max_data_size]
                
                # Convert to tensors immediately
                self.X_train_tensor = torch.Tensor(X_train_np.astype(self.data_format))
                self.Y_train_tensor = torch.Tensor(Y_train_np.astype(self.data_format))
                
                # Create datasets from pre-split tensors
                self.train_ds = TensorDataset(self.X_train_tensor, self.Y_train_tensor)
                self.val_ds = self.train_ds           
                
                # TEST
                test_loaded = np.load(join(self.data_dir, self.test_file), allow_pickle=True, mmap_mode='r')
                X_test_np = test_loaded['x']
                Y_test_np = test_loaded['y']
                print("Test input shape: ", X_test_np.shape)
                print("Test target shape: ", Y_test_np.shape)
                
                if self.max_data_size is not None:
                    X_test_np = X_test_np[:self.max_data_size]
                    Y_test_np = Y_test_np[:self.max_data_size]
                
                # Convert to tensors immediately
                self.X_test_tensor = torch.Tensor(X_test_np.astype(self.data_format))
                self.Y_test_tensor = torch.Tensor(Y_test_np.astype(self.data_format))
                
                self.test_ds = TensorDataset(self.X_test_tensor, self.Y_test_tensor)
                
                # ALL DS
                self.X_all = torch.cat([self.X_train_tensor, self.X_test_tensor], dim=0)
                self.Y_all = torch.cat([self.Y_train_tensor, self.Y_test_tensor], dim=0)
                self.all_ds = TensorDataset(self.X_all, self.Y_all)
                
            self.ds_length = len(self.X_train_tensor)
            
            return
        
        # Normal data loading (not pre-split)
        else:
            
            if self.input_file == self.target_file:
                print("Dataset in one numpy file.")
                loaded = np.load(join(self.data_dir, self.input_file), allow_pickle=True, mmap_mode='r')
                X_np: np.ndarray = loaded['x']
                Y_np: np.ndarray = loaded['y']

            else:
                X_np: np.ndarray = np.load(join(self.data_dir, self.input_file), allow_pickle=True, mmap_mode='r')
                Y_np: np.ndarray = np.load(join(self.data_dir, self.target_file), allow_pickle=True, mmap_mode='r')

            print("Input shape: ", X_np.shape)
            print("Target shape: ", Y_np.shape)

            if self.max_data_size is not None:
                X_np = X_np[:self.max_data_size]
                Y_np = Y_np[:self.max_data_size]

            # Convert to tensors immediately
            self.X_tensor = torch.Tensor(X_np.astype(self.data_format))
            self.Y_tensor = torch.Tensor(Y_np.astype(self.data_format))

            self.all_ds = TensorDataset(self.X_tensor, self.Y_tensor)

            # Store dataset length for normal data loading
            self.ds_length = len(self.X_tensor)
            
            return
    
    
    
    
    def get_ds_len(self)->int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        # If ds_length is already set, return it
        if self.ds_length is not None:
            return self.ds_length
        # Otherwise, prepare data and return length
        
        if self.X_tensor is None or (self.X_train_tensor is None and self.X_test_tensor is None):
            self.prepare_data()
        
        if self.ds_length is not None:
            return self.ds_length
        elif self.X_tensor is not None:
            return len(self.X_tensor)
        else:
            raise ValueError("Data is not loaded correctly.")
    
    
        
    def auto_split_ds(self)->None:
        """
        Automatically split dataset into train/val/test sets.
        
        Split ratios depend on use_val_split parameter:
        - If True: 60% train, 20% val, 20% test
        - If False: 80% train, 20% test (for k-fold CV)
        """
        if self.use_val_split:
            # Split into train/val/test (0.6/0.2/0.2)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.all_ds, [0.6,0.2,0.2], generator=torch.Generator().manual_seed(self.seed))
        else:
            # Split into train/test only (0.8/0.2) - for cross-validation
            self.train_ds, self.test_ds = random_split(
                self.all_ds, [0.8,0.2], generator=torch.Generator().manual_seed(self.seed))
            self.val_ds = None
        return
    
    def idx_split(self):
        """
        Create datasets from provided indices.
        
        Used for k-fold cross-validation where indices are provided externally.
        """
        X_tensor = self.X_train_tensor if self.X_train_tensor is not None else self.X_tensor
        Y_tensor = self.Y_train_tensor if self.Y_train_tensor is not None else self.Y_tensor
        
        if self.test_idx is not None:
            self.test_ds = TensorDataset(X_tensor[self.test_idx], Y_tensor[self.test_idx])
        
        if self.val_idx is not None:
            self.val_ds = TensorDataset(X_tensor[self.val_idx], Y_tensor[self.val_idx])
        
        if self.train_idx is not None:
            self.train_ds = TensorDataset(X_tensor[self.train_idx], Y_tensor[self.train_idx])
        
        
    def split_ds(self)->None:
        """
        Split dataset into train/val/test sets.
        
        Uses either automatic splitting or index-based splitting depending on
        whether indices have been provided via update_idx().
        """
        # Normal data: create datasets from X_tensor and Y_tensor
        if (self.X_tensor is None and self.Y_tensor is None) and (self.X_train_tensor is None and self.Y_train_tensor is None):
            raise ValueError("Tensors not loaded. Call setup() or prepare_data() first.")
        
        # If no indices provided, perform automatic splitting
        if self.train_idx is None and self.val_idx is None and self.test_idx is None:
            self.auto_split_ds()
        
        # Create datasets from indices
        else:
            self.idx_split()
    
    
    def update_idx(
        self,
        train_idx: list=None, 
        val_idx: list=None,
        test_idx: list=None) -> None:
        """
        Update dataset indices for train/val/test splits.
        
        Used primarily for k-fold cross-validation to set custom splits.
        
        Args:
            train_idx: Indices for training set
            val_idx: Indices for validation set
            test_idx: Indices for test set
        """
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        # Proceed only if data has been loaded (either normal or pre-split)
        if self.X_tensor is not None or self.X_train_tensor is not None:
            self.split_ds()
        else:
            print("Warning: update_idx() called before setup(). Datasets will be created when setup() is called.")
            return
        
        
    
    
    def setup(self, stage) -> None:
        """
        Setup method called by PyTorch Lightning.
        Loads data (if not already loaded) and creates datasets.
        """
        self.prepare_data()
        self.split_ds()
    
    
    def train_dataloader(self,):
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = True,
        )
    
    def val_dataloader(self,):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = False,
        )
    
    def test_dataloader(self,):
        return DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = False,
        )
    
    def pred_test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size = 1,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = False,
        )
    
    def all_dataloader(self):
        return DataLoader(
            self.all_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = False,
        )
