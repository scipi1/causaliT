
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataset import TensorDataset
import pytorch_lightning as pl
from os.path import join


class ProcessDataModule(pl.LightningDataModule):
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
        self.X = None
        self.Y = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        
    def prepare_data(self) -> None:
        
        if self.input_file == self.target_file:
            print("Dataset in one numpy file.")
            loaded = np.load(join(self.data_dir, self.input_file), allow_pickle=True, mmap_mode='r')
            self.X: np.ndarray = loaded['x']
            self.Y: np.ndarray = loaded['y']
        
        else:
            if self.X is None:
                self.X: np.ndarray = np.load(join(self.data_dir, self.input_file), allow_pickle=True, mmap_mode='r')
            if self.Y is None:
                self.Y: np.ndarray = np.load(join(self.data_dir, self.target_file), allow_pickle=True, mmap_mode='r')
            
        print("Input shape: ", self.X.shape)
        print("Target shape: ", self.Y.shape)
            
        if self.max_data_size is not None:
            self.X = self.X[:self.max_data_size]
            self.Y = self.Y[:self.max_data_size]
    
    def get_ds_len(self)->int:
        
        if self.X is None:
            self.prepare_data()
        
        if self.X is not None:
            return len(self.X)
        else:
            raise ValueError("Data is not loaded correctly.")
    
    
    def update_idx(
        self,
        train_idx: list=None, 
        val_idx: list=None,
        test_idx: list=None) -> None:
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    
    def setup(self,stage) -> None:
        
        if self.X is None or self.Y is None:
            self.prepare_data()
            
        if self.X is None or self.Y is None:
            raise ValueError("Data not loaded correctly. Please check your input files.")
        
        X = torch.Tensor(self.X.astype(self.data_format))
        Y = torch.Tensor(self.Y.astype(self.data_format))
        
        self.all_ds = TensorDataset(X,Y)
        
        if self.train_idx is None and self.val_idx is None and self.test_idx is None:
            self.ds = TensorDataset(X,Y)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.ds,[0.6,0.2,0.2],generator=torch.Generator().manual_seed(self.seed))
            
        if self.test_idx is not None:
            self.test_ds = TensorDataset(X[self.test_idx], Y[self.test_idx])
            
        if self.val_idx is not None:
            self.val_ds = TensorDataset(X[self.val_idx], Y[self.val_idx])
            
        if self.train_idx is not None:
            self.train_ds = TensorDataset(X[self.train_idx], Y[self.train_idx])
    
    
    def train_dataloader(self,):
        return DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            persistent_workers=True,
            shuffle = True,
        )
    
    def val_dataloader(self,):
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
