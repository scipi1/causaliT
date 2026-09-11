"""
StageCausalDataModule: DataLoader for StageCausaliT architecture.

Handles loading two or three data streams (S, X, [Y]) from a single .npz file
and provides DataLoaders compatible with the dual-decoder architecture.

Supports datasets without Y (target variables) for S→X only training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataset import TensorDataset
import pytorch_lightning as pl
from os.path import join


class StageCausalDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for StageCausaliT.
    
    Loads two or three data streams from a single .npz file:
    - S: Source nodes (required)
    - X: Intermediate variables (required)
    - Y: Target variables (optional)
    
    Expected file format:
        np.savez('data.npz', s=S_array, x=X_array, y=Y_array)  # with targets
        np.savez('data.npz', s=S_array, x=X_array)              # without targets
    
    Supports:
    - Automatic train/val/test splitting
    - K-fold cross-validation with manual indices
    - Pre-split train/test files
    - Data size limiting for debugging
    - Datasets without Y (target variables)
    """
    def __init__(
        self,
        data_dir: str,
        input_file: str,  # Single .npz file containing s, x, [y]
        batch_size: int,
        num_workers: int,
        data_format: str,
        max_data_size: int = None,
        seed: int = 42,
        train_file: str = None,
        test_file: str = None,
        use_val_split: bool = True,
    ) -> None:
        
        super().__init__()
        
        self.data_dir = data_dir
        self.input_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_format = data_format
        self.max_data_size = max_data_size
        self.seed = seed
        self.train_file = train_file
        self.test_file = test_file
        self.use_val_split = use_val_split
        # Persistent workers are only valid when num_workers > 0.  They are also
        # disabled by callers that reload the dataloader every epoch (e.g. the
        # adaptive cross-fit trainer) to avoid worker/memory accumulation.
        self.persistent_workers = num_workers > 0
        # Cross-fit stage splits owned by the datamodule (see set_stage_splits).
        self._stage_splits = None
        # HSIC cross-fitting folds (see set_hsic_cross_fit); None = disabled.
        self._xfit_ds_a = None
        self._xfit_ds_b = None
        self._stage_val_idx = None
        self._stage_test_idx = None
        
        # Flag to track if Y (targets) exist
        self.has_targets = None

        
        # Store data as tensors
        self.S_tensor = None
        self.X_tensor = None
        self.Y_tensor = None
        self.S_train_tensor = None
        self.X_train_tensor = None
        self.Y_train_tensor = None
        self.S_test_tensor = None
        self.X_test_tensor = None
        self.Y_test_tensor = None
        
        # Dataset indices
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        
        # Datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.all_ds = None
        self.ds_length = None
    
    def _create_dataset(self, S_tensor, X_tensor, Y_tensor=None):
        """
        Create a TensorDataset with 2 or 3 tensors depending on whether Y exists.
        
        Returns:
            TensorDataset: (S, X) or (S, X, Y)
        """
        if Y_tensor is not None:
            return TensorDataset(S_tensor, X_tensor, Y_tensor)
        else:
            return TensorDataset(S_tensor, X_tensor)
    
    def prepare_data(self) -> None:
        """
        Load data from .npz file and convert to PyTorch tensors.
        
        Supports two modes:
        - Pre-split data: Load separate train/test files
        - Normal data: Load single dataset file for later splitting
        
        Also supports datasets without Y (target variables).
        """
        # Check if pre-split data is provided
        if self.train_file is not None or self.test_file is not None:
            print("Loading pre-split data (S, X, [Y] format).")
            
            # Reset indices to prevent further splitting
            self.train_idx = None
            self.val_idx = None
            self.test_idx = None
            
            if self.train_file is not None and self.test_file is not None:
                
                # TRAIN
                train_loaded = np.load(join(self.data_dir, self.train_file), allow_pickle=True, mmap_mode='r')
                S_train_np = train_loaded['s']
                X_train_np = train_loaded['x']
                
                # Check if Y exists in train data
                if 'y' in train_loaded.files:
                    Y_train_np = train_loaded['y']
                    self.has_targets = True
                    print(f"Train shapes - S: {S_train_np.shape}, X: {X_train_np.shape}, Y: {Y_train_np.shape}")
                    
                    # Validate dimensions
                    assert S_train_np.shape[0] == X_train_np.shape[0] == Y_train_np.shape[0], \
                        f"Batch size mismatch in train data: S={S_train_np.shape[0]}, X={X_train_np.shape[0]}, Y={Y_train_np.shape[0]}"
                else:
                    Y_train_np = None
                    self.has_targets = False
                    print(f"Train shapes - S: {S_train_np.shape}, X: {X_train_np.shape} (no Y)")
                    
                    # Validate dimensions
                    assert S_train_np.shape[0] == X_train_np.shape[0], \
                        f"Batch size mismatch in train data: S={S_train_np.shape[0]}, X={X_train_np.shape[0]}"
                
                if self.max_data_size is not None:
                    S_train_np = S_train_np[:self.max_data_size]
                    X_train_np = X_train_np[:self.max_data_size]
                    if Y_train_np is not None:
                        Y_train_np = Y_train_np[:self.max_data_size]
                
                # Convert to tensors
                self.S_train_tensor = torch.Tensor(S_train_np.astype(self.data_format))
                self.X_train_tensor = torch.Tensor(X_train_np.astype(self.data_format))
                if Y_train_np is not None:
                    self.Y_train_tensor = torch.Tensor(Y_train_np.astype(self.data_format))
                else:
                    self.Y_train_tensor = None
                
                # Create datasets
                self.train_ds = self._create_dataset(self.S_train_tensor, self.X_train_tensor, self.Y_train_tensor)
                self.val_ds = self.train_ds
                
                # TEST
                test_loaded = np.load(join(self.data_dir, self.test_file), allow_pickle=True, mmap_mode='r')
                S_test_np = test_loaded['s']
                X_test_np = test_loaded['x']
                
                # Check if Y exists in test data (should match train)
                if 'y' in test_loaded.files:
                    Y_test_np = test_loaded['y']
                    print(f"Test shapes - S: {S_test_np.shape}, X: {X_test_np.shape}, Y: {Y_test_np.shape}")
                    
                    # Validate dimensions
                    assert S_test_np.shape[0] == X_test_np.shape[0] == Y_test_np.shape[0], \
                        f"Batch size mismatch in test data: S={S_test_np.shape[0]}, X={X_test_np.shape[0]}, Y={Y_test_np.shape[0]}"
                else:
                    Y_test_np = None
                    print(f"Test shapes - S: {S_test_np.shape}, X: {X_test_np.shape} (no Y)")
                    
                    # Validate dimensions
                    assert S_test_np.shape[0] == X_test_np.shape[0], \
                        f"Batch size mismatch in test data: S={S_test_np.shape[0]}, X={X_test_np.shape[0]}"
                
                if self.max_data_size is not None:
                    S_test_np = S_test_np[:self.max_data_size]
                    X_test_np = X_test_np[:self.max_data_size]
                    if Y_test_np is not None:
                        Y_test_np = Y_test_np[:self.max_data_size]
                
                # Convert to tensors
                self.S_test_tensor = torch.Tensor(S_test_np.astype(self.data_format))
                self.X_test_tensor = torch.Tensor(X_test_np.astype(self.data_format))
                if Y_test_np is not None:
                    self.Y_test_tensor = torch.Tensor(Y_test_np.astype(self.data_format))
                else:
                    self.Y_test_tensor = None
                
                self.test_ds = self._create_dataset(self.S_test_tensor, self.X_test_tensor, self.Y_test_tensor)
                
                # Concatenate for all_ds
                self.S_all = torch.cat([self.S_train_tensor, self.S_test_tensor], dim=0)
                self.X_all = torch.cat([self.X_train_tensor, self.X_test_tensor], dim=0)
                if self.has_targets:
                    self.Y_all = torch.cat([self.Y_train_tensor, self.Y_test_tensor], dim=0)
                    self.all_ds = TensorDataset(self.S_all, self.X_all, self.Y_all)
                else:
                    self.Y_all = None
                    self.all_ds = TensorDataset(self.S_all, self.X_all)
            
            self.ds_length = len(self.S_train_tensor)
            return
        
        # Normal data loading (not pre-split)
        else:
            print("Loading single data file (S, X, [Y] format).")
            loaded = np.load(join(self.data_dir, self.input_file), allow_pickle=True, mmap_mode='r')
            
            S_np: np.ndarray = loaded['s']
            X_np: np.ndarray = loaded['x']
            
            # Check if Y exists
            if 'y' in loaded.files:
                Y_np: np.ndarray = loaded['y']
                self.has_targets = True
                print(f"Data shapes - S: {S_np.shape}, X: {X_np.shape}, Y: {Y_np.shape}")
                
                # Validate dimensions
                assert S_np.shape[0] == X_np.shape[0] == Y_np.shape[0], \
                    f"Batch size mismatch: S={S_np.shape[0]}, X={X_np.shape[0]}, Y={Y_np.shape[0]}"
            else:
                Y_np = None
                self.has_targets = False
                print(f"Data shapes - S: {S_np.shape}, X: {X_np.shape} (no Y)")
                
                # Validate dimensions
                assert S_np.shape[0] == X_np.shape[0], \
                    f"Batch size mismatch: S={S_np.shape[0]}, X={X_np.shape[0]}"
            
            if self.max_data_size is not None:
                S_np = S_np[:self.max_data_size]
                X_np = X_np[:self.max_data_size]
                if Y_np is not None:
                    Y_np = Y_np[:self.max_data_size]
            
            # Convert to tensors
            self.S_tensor = torch.Tensor(S_np.astype(self.data_format))
            self.X_tensor = torch.Tensor(X_np.astype(self.data_format))
            if Y_np is not None:
                self.Y_tensor = torch.Tensor(Y_np.astype(self.data_format))
            else:
                self.Y_tensor = None
            
            self.all_ds = self._create_dataset(self.S_tensor, self.X_tensor, self.Y_tensor)
            
            # Store dataset length
            self.ds_length = len(self.S_tensor)
            return
    
    def get_ds_len(self) -> int:
        """Get the length of the dataset."""
        if self.ds_length is not None:
            return self.ds_length
        
        if self.S_tensor is None and self.S_train_tensor is None:
            self.prepare_data()
        
        if self.ds_length is not None:
            return self.ds_length
        elif self.S_tensor is not None:
            return len(self.S_tensor)
        else:
            raise ValueError("Data is not loaded correctly.")
    
    def auto_split_ds(self) -> None:
        """
        Automatically split dataset into train/val/test sets.
        
        Split ratios:
        - If use_val_split=True: 60% train, 20% val, 20% test
        - If use_val_split=False: 80% train, 20% test (for k-fold CV)
        """
        if self.use_val_split:
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.all_ds, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self.seed))
        else:
            self.train_ds, self.test_ds = random_split(
                self.all_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(self.seed))
            self.val_ds = None
    
    def idx_split(self):
        """Create datasets from provided indices (for k-fold CV)."""
        S_tensor = self.S_train_tensor if self.S_train_tensor is not None else self.S_tensor
        X_tensor = self.X_train_tensor if self.X_train_tensor is not None else self.X_tensor
        Y_tensor = self.Y_train_tensor if self.Y_train_tensor is not None else self.Y_tensor
        
        if self.test_idx is not None:
            if Y_tensor is not None:
                self.test_ds = TensorDataset(
                    S_tensor[self.test_idx],
                    X_tensor[self.test_idx],
                    Y_tensor[self.test_idx]
                )
            else:
                self.test_ds = TensorDataset(
                    S_tensor[self.test_idx],
                    X_tensor[self.test_idx]
                )
        
        if self.val_idx is not None:
            if Y_tensor is not None:
                self.val_ds = TensorDataset(
                    S_tensor[self.val_idx],
                    X_tensor[self.val_idx],
                    Y_tensor[self.val_idx]
                )
            else:
                self.val_ds = TensorDataset(
                    S_tensor[self.val_idx],
                    X_tensor[self.val_idx]
                )
        
        if self.train_idx is not None:
            if Y_tensor is not None:
                self.train_ds = TensorDataset(
                    S_tensor[self.train_idx],
                    X_tensor[self.train_idx],
                    Y_tensor[self.train_idx]
                )
            else:
                self.train_ds = TensorDataset(
                    S_tensor[self.train_idx],
                    X_tensor[self.train_idx]
                )
    
    def split_ds(self) -> None:
        """Split dataset into train/val/test sets."""
        if (self.S_tensor is None and self.S_train_tensor is None):
            raise ValueError("Tensors not loaded. Call setup() or prepare_data() first.")
        
        # Automatic splitting if no indices provided
        if self.train_idx is None and self.val_idx is None and self.test_idx is None:
            self.auto_split_ds()
        else:
            self.idx_split()
    
    def update_idx(
        self,
        train_idx: list = None,
        val_idx: list = None,
        test_idx: list = None
    ) -> None:
        """Update dataset indices for train/val/test splits (for k-fold CV)."""
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        if self.S_tensor is not None or self.S_train_tensor is not None:
            self.split_ds()
        else:
            print("Warning: update_idx() called before setup(). Datasets will be created when setup() is called.")
    
    def set_stage_splits(self, stage_splits: dict = None,
                         val_idx=None, test_idx=None) -> None:
        """
        Register per-phase training-index subsets for cross-fit / staged training.

        The datamodule owns the split mapping so the phase orchestrator only needs
        to request a phase by name (see ``set_active_phase``).  Validation/test
        indices are stored once and reused across phases so metrics stay
        comparable.
        """
        self._stage_splits = dict(stage_splits) if stage_splits else None
        self._stage_val_idx = val_idx
        self._stage_test_idx = test_idx
    
    def set_active_phase(self, phase: str):
        """
        Point ``train_ds`` at ``phase``'s registered cross-fit subset.

        Returns the subset size (int) or ``None`` when no split is registered for
        the phase (cross-fitting disabled).
        """
        if not self._stage_splits:
            return None
        subset = self._stage_splits.get(phase)
        if subset is None:
            return None
        self.update_idx(
            train_idx=subset,
            val_idx=self._stage_val_idx,
            test_idx=self._stage_test_idx,
        )
        return int(len(subset))
    
    def set_hsic_cross_fit(self, ratio: float = 0.5, seed: int = 0) -> tuple:
        """
        Partition the CURRENT training set into two disjoint, permanent folds.

        Fold A carries the reconstruction (MSE) loss, fold B the independence
        (HSIC) statistic, so the structural signal is measured on samples the
        regressor did not fit -- HSIC on the rows the fit just absorbed is
        optimistically biased.

        The partition is by SAMPLE IDENTITY (a fixed permutation of the training
        subset), NOT by position within a batch: ``train_dataloader`` shuffles
        every epoch, so a positional split would reassign samples each epoch and
        the separation would dissolve.  Membership here is decided once and
        survives shuffling.

        Call AFTER the train/val/test indices are final (i.e. after
        ``update_idx`` / ``setup``).  Returns ``(n_a, n_b)``.
        """
        if self.train_ds is None:
            raise RuntimeError(
                "set_hsic_cross_fit() requires train_ds; call setup() first."
            )
        if not 0.0 < float(ratio) < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}.")

        n = len(self.train_ds)
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=g)
        n_a = int(round(float(ratio) * n))
        n_a = max(1, min(n - 1, n_a))          # both folds non-empty
        idx_a, idx_b = perm[:n_a], perm[n_a:]

        # ``train_ds`` may be a TensorDataset or a Subset (auto_split_ds uses
        # random_split).  Subset indexing works for both and keeps the folds
        # views rather than copies.
        self._xfit_ds_a = Subset(self.train_ds, idx_a.tolist())
        self._xfit_ds_b = Subset(self.train_ds, idx_b.tolist())
        return int(len(idx_a)), int(len(idx_b))

    def hsic_cross_fit_dataloaders(self):
        """``(loader_a, loader_b)`` over the two folds, or ``None`` if unset."""
        if getattr(self, "_xfit_ds_a", None) is None:
            return None

        def _mk(ds):
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                shuffle=True,
            )
        return _mk(self._xfit_ds_a), _mk(self._xfit_ds_b)

    def setup(self, stage) -> None:
        """Setup method called by PyTorch Lightning."""
        self.prepare_data()
        self.split_ds()
    
    def train_dataloader(self):
        # HSIC cross-fitting: fold A is the Lightning train loader (it drives
        # the epoch length and the MSE); fold B is pulled alongside it by the
        # forecaster via ``hsic_cross_fit_dataloaders``.
        if getattr(self, "_xfit_ds_a", None) is not None:
            return DataLoader(
                self._xfit_ds_a,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                shuffle=True,
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
    
    def pred_test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
    
    def all_dataloader(self):
        return DataLoader(
            self.all_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )


