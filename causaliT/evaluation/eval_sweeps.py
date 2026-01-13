import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from os import makedirs, listdir
from os.path import join, isdir, abspath, dirname, exists
from typing import List, Callable, Union, Dict
import pandas as pd
import shutil
# import papermill as pm
from typing import List, Callable
import pandas as pd
import os
import warnings
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from io import StringIO, BytesIO
import torch
import re

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"







# condition to identify the bottom level________________________________________________________________________________________________

def has_logs_subfolder(directory: str, s3: bool = False, bucket: str = None) -> bool:
    """
    Check if the given directory contains at least one logs subfolder
    """
    target_folder = "logs"

    if s3:
        s3_client = get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=directory, Delimiter='/')

        for page in pages:
            for prefix in page.get("CommonPrefixes", []):
                if prefix["Prefix"].rstrip('/').endswith(target_folder):
                    return True
        return False

    else:
        directory_path = Path(directory)
        return any(subdir.name == target_folder for subdir in directory_path.iterdir() if subdir.is_dir())


def has_kfold_summary(directory: str, s3: bool = False, bucket: str = None) -> bool:
    """
    Check if the given directory contains kfold_summary.json file.
    This indicates we've reached a trained model directory (bottom level).
    
    Args:
        directory: Directory path to check
        s3: Whether to check on S3
        bucket: S3 bucket name (required if s3=True)
        
    Returns:
        bool: True if kfold_summary.json exists in directory
    """
    target_file = "kfold_summary.json"

    if s3:
        s3_client = get_s3_client()
        # Ensure directory ends with /
        prefix = directory.rstrip('/') + '/'
        file_key = prefix + target_file
        
        try:
            s3_client.head_object(Bucket=bucket, Key=file_key)
            return True
        except:
            return False
    else:
        directory_path = Path(directory)
        return (directory_path / target_file).exists()


def is_gradients_folder(directory: str, s3: bool = False, bucket: str = None) -> bool:
    """
    Check if the given directory is a gradients folder (bottom level for gradient processing).
    This checks if the directory name is 'gradients' and contains .npz files.
    
    Args:
        directory: Directory path to check
        s3: Whether to check on S3
        bucket: S3 bucket name (required if s3=True)
        
    Returns:
        bool: True if this is a gradients folder
    """
    if s3:
        # Get the folder name from the path
        folder_name = directory.rstrip('/').split('/')[-1]
        if folder_name != "gradients":
            return False
        
        # Check if it contains .npz files
        s3_client = get_s3_client()
        prefix = directory.rstrip('/') + '/'
        
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.npz'):
                        return True
            return False
        except:
            return False
    else:
        directory_path = Path(directory)
        # Check if the folder name is 'gradients'
        if directory_path.name != "gradients":
            return False
        
        # Check if it contains .npz files
        if not directory_path.exists() or not directory_path.is_dir():
            return False
        
        # Look for any .npz files
        npz_files = list(directory_path.glob("*.npz"))
        return len(npz_files) > 0


# helper functions________________________________________________________________________________________________

def array_to_long_df(arr: np.ndarray) -> pd.DataFrame:
    """
    Convert a B x M1 x M2 numpy array to a long DataFrame.

    Parameters:
        arr (np.ndarray): Input array of shape (B, M1, M2)

    Returns:
        pd.DataFrame: DataFrame with columns 'sample', 'i', 'j', 'value'
    """
    B, M1, M2 = arr.shape
    # Reshape and create indices
    df = pd.DataFrame({
        'sample': np.repeat(np.arange(B), M1 * M2),
        'i': np.tile(np.repeat(np.arange(M1), M2), B),
        'j': np.tile(np.arange(M2), B * M1),
        'value': arr.reshape(-1)
    })
    return df


# bottom actions________________________________________________________________________________________________

def process_gradients_bottom_action(filepath: str, level_folders: List[str], s3: bool = False, bucket: str = None) -> pd.DataFrame:
    """
    Bottom action that processes gradient files by:
    - Finding all .npz files in the gradients folder
    - Extracting epoch number from filenames
    - Loading jacobian data from each file
    - Converting to long dataframe format
    - Returning concatenated results
    
    This function is designed to be used as a bottom_action callable with get_df_recursive.
    Level information (experiment name, fold number) is automatically added by get_df_recursive.
    
    Args:
        filepath: Current directory path (parent of gradients folder)
        level_folders: List of subdirectories at this level (should contain 'gradients')
        s3: Whether files are on S3
        bucket: S3 bucket name (required if s3=True)
        
    Returns:
        pd.DataFrame: Gradient data with columns:
            - sample: sample index within batch
            - i: first dimension index
            - j: second dimension index
            - value: gradient value
            - epoch: epoch number
            - batch: batch identifier
            
    Example:
        >>> df = get_df_recursive(
        ...     filepath="experiments/experiment_name",
        ...     bottom_action=process_gradients_bottom_action,
        ...     is_bottom=is_gradients_folder
        ... )
        >>> # Result will have level_0 (experiment), level_1 (fold), etc. columns added automatically
    """
    df_list = []
    
    for folder in level_folders:
        if s3:
            # S3 paths
            folder_path = f"{filepath.rstrip('/')}/{folder}"
            print(f"Warning: S3 support for process_gradients_bottom_action not fully implemented")
            continue
        else:
            # Local paths
            folder_path = join(filepath, folder)
            
            if not exists(folder_path) or not isdir(folder_path):
                print(f"Warning: {folder_path} does not exist or is not a directory, skipping...")
                continue
            
            # Find all .npz files matching the pattern jacobian_epoch_*.npz
            npz_files = list(Path(folder_path).glob("jacobian_epoch_*.npz"))
            
            if not npz_files:
                print(f"Warning: No jacobian_epoch_*.npz files found in {folder_path}, skipping...")
                continue
            
            print(f"Processing gradients in {folder}...")
            print(f"  Found {len(npz_files)} npz files")
            
            for npz_file in npz_files:
                try:
                    # Extract epoch number from filename
                    # Pattern: jacobian_epoch_0000.npz -> epoch 0
                    match = re.search(r'jacobian_epoch_(\d+)\.npz', npz_file.name)
                    if not match:
                        print(f"  Warning: Could not extract epoch from {npz_file.name}, skipping...")
                        continue
                    
                    epoch = int(match.group(1))
                    
                    # Load the npz file
                    data = np.load(str(npz_file))
                    
                    # Process each batch in the file
                    batch_keys = [key for key in data.keys() if key.startswith('batch_')]
                    
                    if not batch_keys:
                        print(f"  Warning: No batch keys found in {npz_file.name}, skipping...")
                        continue
                    
                    for batch_key in batch_keys:
                        # Extract jacobian data with shape (B, M1, ?, M2, ?)
                        # Apply slicing [:,:,0,:,0] to get (B, M1, M2)
                        jacob_data = data[batch_key]
                        
                        # Handle different dimensionalities
                        if jacob_data.ndim == 5:
                            # Shape: (B, M1, ?, M2, ?) -> extract (B, M1, M2)
                            jacob = jacob_data[:, :, 0, :, 0]
                        elif jacob_data.ndim == 3:
                            # Already (B, M1, M2)
                            jacob = jacob_data
                        else:
                            print(f"  Warning: Unexpected shape {jacob_data.shape} for {batch_key} in {npz_file.name}, skipping...")
                            continue
                        
                        # Convert to long dataframe
                        df_batch = array_to_long_df(jacob)
                        df_batch['epoch'] = epoch
                        df_batch['batch'] = batch_key
                        
                        df_list.append(df_batch)
                    
                    print(f"  Processed {npz_file.name} (epoch {epoch}, {len(batch_keys)} batches)")
                    
                except Exception as e:
                    print(f"  Error processing {npz_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Concatenate all results
    if df_list:
        result = pd.concat(df_list, ignore_index=True)
        print(f"Total gradient records: {len(result)}")
        return result
    else:
        print("Warning: No gradient files were successfully processed")
        return pd.DataFrame()


def eval_models_bottom_action(filepath: str, level_folders: List[str], s3: bool = False, bucket: str = None, datadir_path: str = None) -> pd.DataFrame:
    """
    Bottom action that evaluates trained models by:
    - Loading config and kfold_summary.json from each folder
    - Finding the best checkpoint based on kfold results
    - Running predictions on test set
    - Computing metrics
    - Returning results as a tidy DataFrame
    
    This function is designed to be used as a bottom_action callable with get_df_recursive.
    It keeps metadata extraction minimal (only folder name) - level information 
    (level_0, level_1, ...) is automatically added by get_df_recursive.
    
    Args:
        filepath: Current directory path
        level_folders: List of subdirectories at this level
        s3: Whether files are on S3
        bucket: S3 bucket name (required if s3=True)
        datadir_path: Path to data directory. If None, uses "../data/input"
        
    Returns:
        pd.DataFrame: Metrics results with columns:
            - index: sample index
            - feature: feature name (if multivariate)
            - R2, MSE, MAE, RMSE: metric values
            - model_folder: folder name containing the model
            
    Example:
        >>> df = get_df_recursive(
        ...     filepath="experiments/ds_size",
        ...     bottom_action=eval_models_bottom_action,
        ...     is_bottom=has_kfold_summary
        ... )
        >>> # Result will have level_0, level_1, etc. columns added automatically
    """
    from causaliT.evaluation.predict import predict_test_from_ckpt
    from causaliT.evaluation.metrics import compute_prediction_metrics
    
    # Default data directory path
    if datadir_path is None:
        datadir_path = "../data/input"
    
    df_list = []
    
    for folder in level_folders:
        if s3:
            # S3 paths
            folder_path = f"{filepath.rstrip('/')}/{folder}"
            kfold_summary_key = f"{folder_path}/kfold_summary.json"
            config_key = f"{folder_path}/config.yaml"
            
            # Note: S3 implementation would need additional work to download files
            # and handle checkpoint loading from S3
            print(f"Warning: S3 support for eval_models_bottom_action not fully implemented")
            continue
            
        else:
            # Local paths
            folder_path = join(filepath, folder)
            kfold_summary_path = join(folder_path, "kfold_summary.json")
            config_path = join(folder_path, "config.yaml")
            
            # Validate required files exist
            if not exists(config_path):
                print(f"Warning: No config.yaml in {folder_path}, skipping...")
                continue
                
            if not exists(kfold_summary_path):
                print(f"Warning: No kfold_summary.json in {folder_path}, skipping...")
                continue
            
            try:
                # Load config and kfold summary
                config = OmegaConf.load(config_path)
                kfold_summary = OmegaConf.load(kfold_summary_path)
                best_fold_number = kfold_summary.best_fold.fold_number
                
                # Build checkpoint path
                checkpoint_path = join(
                    folder_path,
                    f'k_{best_fold_number}',
                    'checkpoints',
                    'best_checkpoint.ckpt'
                )
                
                if not exists(checkpoint_path):
                    print(f"Warning: Checkpoint not found: {checkpoint_path}, skipping...")
                    continue
                
                print(f"Processing {folder}...")
                print(f"  Config: {config_path}")
                print(f"  Checkpoint: {checkpoint_path}")
                
                # Run predictions
                results = predict_test_from_ckpt(
                    config=config,
                    datadir_path=datadir_path,
                    checkpoint_path=checkpoint_path,
                    dataset_label="test",
                    cluster=False
                )
                
                # Compute metrics
                # Try to get val_idx from config
                val_idx = None
                if "data" in config and "val_idx" in config["data"]:
                    val_idx = config["data"]["val_idx"]
                
                metrics_df = compute_prediction_metrics(
                    results,
                    target_feature_idx=val_idx
                )
                
                # Add folder name column for identification
                metrics_df["model_folder"] = folder
                
                df_list.append(metrics_df)
                print(f"  Successfully processed {folder}")
                
            except Exception as e:
                print(f"Error processing {folder}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Concatenate all results
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        print("Warning: No models were successfully processed")
        return pd.DataFrame()  # Empty dataframe if no models processed


# main recursive function________________________________________________________________________________________________

def get_df_recursive(filepath: str, bottom_action: Callable, is_bottom: Callable, s3: bool=False, bucket: str=None, lev: int=0)->pd.DataFrame:
    """
    Loops recursively inside folders, keeping track of the various levels, until the bottom is reached
    At the bottom, performs the bottom_action.
    
    N.B. The condition for the bottom is hard-coded

    Args:
        filepath (str): level path, if user input, starting level
        bottom_action (Callable): function to perform at the bottom level
        s3 (bool): AWS s3 flag
        lev (int, optional): Current level, leave default value. Defaults to 0.

    Returns:
        pd.DataFrame: multi-level dataframe
    """
    
    # files on s3 bucket
    if s3:
        
        s3_client = get_s3_client()
        
        # List all "directories" one level under the current prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=filepath, Delimiter='/')
        
        level_folders = []
        for page in pages:
            for prefix in page.get("CommonPrefixes", []):
                level_folders.append(prefix["Prefix"].rstrip('/').split('/')[-1])
        
        if not level_folders:
            return pd.DataFrame()  # empty folder
        
        # init dataframe
        df = None
        
        # Check each folder individually to handle mixed bottom/non-bottom scenarios
        for case in level_folders:
            subpath = f"{filepath.rstrip('/')}/{case}/"
            
            # Check if THIS specific folder is a bottom folder
            if is_bottom(subpath, s3=s3, bucket=bucket):
                print(f"reached bottom level: {case}")
                # Process this single bottom folder
                df_temp = bottom_action(filepath, [case], s3=s3, bucket=bucket)
                
                if df_temp is not None and not df_temp.empty:
                    df_temp[f"level_{lev}"] = case
                    df = df_temp if df is None else pd.concat([df, df_temp])
            else:
                # Not a bottom folder - try to recurse into it
                try:
                    df_temp = get_df_recursive(subpath, bottom_action, is_bottom, s3=s3, bucket=bucket, lev=lev+1)
                    
                    if df_temp is not None and not df_temp.empty:
                        df_temp[f"level_{lev}"] = case
                        df = df_temp if df is None else pd.concat([df, df_temp])
                except Exception as e:
                    print(f"Warning: Could not process {subpath}: {e}")
                    continue  # Skip this folder and continue with others
    
    # files on local machine
    else:
        
        # get all folders of current level
        level_folders = [d for d in listdir(filepath) if isdir(join(filepath,d))]
        
        if not level_folders:
            return pd.DataFrame()  # Empty directory
        
        # init dataframe
        df = None
        
        # Check each folder individually to handle mixed bottom/non-bottom scenarios
        for case in level_folders:
            filepath_ = join(filepath, case)
            
            # Check if THIS specific folder is a bottom folder
            if is_bottom(filepath_):
                print(f"reached bottom level: {case}")
                # Process this single bottom folder
                df_temp = bottom_action(filepath, [case], s3=s3, bucket=bucket)
                
                if df_temp is not None and not df_temp.empty:
                    df_temp[f"level_{lev}"] = case
                    df = df_temp if df is None else pd.concat([df, df_temp])
            else:
                # Not a bottom folder - try to recurse into it
                try:
                    df_temp = get_df_recursive(filepath_, bottom_action, is_bottom, s3=s3, bucket=bucket, lev=lev+1)
                    
                    if df_temp is not None and not df_temp.empty:
                        df_temp[f"level_{lev}"] = case
                        df = df_temp if df is None else pd.concat([df, df_temp])
                except Exception as e:
                    print(f"Warning: Could not process {filepath_}: {e}")
                    continue  # Skip this folder and continue with others

    return df if df is not None else pd.DataFrame()




def eval_sweeps(filepath: str, outpath: str = None, s3: bool = False, datadir_path: str = None):
    """
    Evaluates sweep experiment by recursively traversing directories, running model
    predictions and computing metrics at the bottom level.
    
    Args:
        filepath: Root directory containing experiment sweep
        outpath: Output directory to save results CSV. If None, only returns DataFrame without saving
        s3: Whether files are on S3 (default: False)
        datadir_path: Path to data directory (default: "../data/input")
        
    Returns:
        pd.DataFrame: Complete results with metrics and level information
        
    Example:
        >>> # Save results to file
        >>> df = eval_sweeps(
        ...     filepath="experiments/ds_size",
        ...     outpath="results",
        ...     datadir_path="../data/input"
        ... )
        >>> 
        >>> # Only return DataFrame without saving
        >>> df = eval_sweeps(
        ...     filepath="experiments/ds_size",
        ...     datadir_path="../data/input"
        ... )
    """
    # Create a wrapper for bottom_action that includes datadir_path
    def bottom_action_with_datadir(filepath, level_folders, s3=False, bucket=None):
        return eval_models_bottom_action(filepath, level_folders, s3, bucket, datadir_path)
    
    df = get_df_recursive(
        filepath=filepath, 
        bottom_action=bottom_action_with_datadir, 
        is_bottom=has_kfold_summary, 
        s3=s3, 
        bucket="scipi1-public"
    )
    
    # Save results if outpath is provided
    if outpath is not None:
        makedirs(outpath, exist_ok=True)
        output_path = join(outpath, "eval_sweeps.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def eval_gradients(filepath: str, outpath: str = None, s3: bool = False, bucket: str = None):
    """
    Evaluates gradients from experiments by recursively traversing directories and
    processing all .npz gradient files at the bottom level (gradients folders).
    
    This function walks through experiment directories with structure:
        experiments/{experiment_name}/k_{fold}/gradients/jacobian_epoch_*.npz
    
    And returns a consolidated DataFrame with:
    - Gradient data (sample, i, j, value)
    - Epoch and batch information
    - Level information (experiment name, fold number, etc.)
    
    Args:
        filepath: Root directory containing experiments (e.g., "../experiments")
        outpath: Output directory to save results CSV. If None, only returns DataFrame without saving
        s3: Whether files are on S3 (default: False)
        bucket: S3 bucket name (required if s3=True)
        
    Returns:
        pd.DataFrame: Complete gradient data with columns:
            - sample: sample index within batch
            - i: first dimension index
            - j: second dimension index  
            - value: gradient value
            - epoch: epoch number
            - batch: batch identifier
            - level_0: experiment name
            - level_1: fold identifier (e.g., k_0, k_1)
            - (additional level columns if more nesting exists)
        
    Example:
        >>> # Process all experiments and save results
        >>> df = eval_gradients(
        ...     filepath="../experiments",
        ...     outpath="results"
        ... )
        >>> 
        >>> # Only return DataFrame without saving
        >>> df = eval_gradients(
        ...     filepath="../experiments/my_experiment"
        ... )
        >>> 
        >>> # Access gradient data with level information
        >>> # df will have columns: sample, i, j, value, epoch, batch, level_0, level_1, ...
        >>> print(df.groupby(['level_0', 'level_1', 'epoch'])['value'].mean())
    """
    df = get_df_recursive(
        filepath=filepath, 
        bottom_action=process_gradients_bottom_action, 
        is_bottom=is_gradients_folder, 
        s3=s3, 
        bucket=bucket
    )
    
    # Save results if outpath is provided
    if outpath is not None:
        makedirs(outpath, exist_ok=True)
        output_path = join(outpath, "eval_gradients.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return df



# helpers________________________________________________________________________________________________
def get_s3_client(public_only: bool = True):
    if public_only:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    else:
        return boto3.client("s3")
