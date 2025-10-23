# Standard library imports
import json
import logging
import os
import sys
from os.path import dirname, abspath, join

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
from proT.training.callbacks import (
    early_stopping_callbacks, get_checkpoint_callback, MemoryLoggerCallback, 
    GradientLogger, MetricsAggregator, PerRunManifest, 
    BestCheckpointCallback, DataIndexTracker, 
    KFoldResultsTracker
)
from proT.training.forecasters import TransformerForecaster
from proT.training.dataloader import ProcessDataModule
from proT.training.experiment_control import update_config
#from proT.predict import mk_quick_pred_plot

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str="NA",
    resume_ckpt: str=None,
    plot_pred_check: bool=False,
    debug: bool=False,
    best: bool=False,
    )->pd.DataFrame:
    """
    Training function with k-fold cross-validation

    Args:
        config (dict): configuration file
        data_dir (str): data directory
        save_dir (str): saving directory
        cluster (bool): cluster used?
        experiment_tag (str, optional): experiment tag for logging. Defaults to "NA".
        resume_ckpt (str, optional): checkpoint to resume training. Defaults to None.
        plot_pred_check (bool, optional): whether to plot prediction check. Defaults to False.
        debug (bool, optional): turn on debug options. Defaults to False.
        best (bool, optional): if True, use metrics from best checkpoint (based on val_mae), 
                              if False, use final epoch metrics. Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame containing metrics for each fold
    """
    
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    # set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    # get model object from configuration (version 4.6.0 or higher)
    model_object = get_model_object(config)
    
    
    dm = ProcessDataModule(
        data_dir = join(data_dir,config["data"]["dataset"]),
        input_file =  config["data"]["filename_input"],
        target_file = config["data"]["filename_target"],
        batch_size =  config["training"]["batch_size"],
        num_workers = 1 if cluster else 20,
        data_format = "float32",
        max_data_size = config["data"]["max_data_size"],
        seed = seed,
    )
    
    
    dataset_size = dm.get_ds_len()
    indices = np.arange(dataset_size)
    
    test_ds_idx_filename = config["data"]["test_ds_ixd"]
    
    if test_ds_idx_filename is not None:
        test_idx = np.load(join(data_dir,config["data"]["dataset"],test_ds_idx_filename))
        mask = np.isin(indices, test_idx)
        train_val_idx = indices[~mask]
    
    else:
        test_size = int(0.2 * dataset_size)  # Reserve 20% for testing
        test_idx = indices[:test_size]       # Fixed test indices
        train_val_idx = indices[test_size:]  # Remaining for train/val cross-validation
    
    
    
    
    # k-fold cross-validation
    k_folds = config["training"]["k_fold"]
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    metrics_dict = {}
    
    # Initialize k-fold results tracker
    kfold_tracker = KFoldResultsTracker(save_dir, k_folds)
    
    for fold, (train_local_idx, val_local_idx) in enumerate(kfold.split(train_val_idx)):
        
        # re-initialize the model at any fold
        model = model_object(config)
        
        print(f"Fold {fold + 1}/{k_folds}")
        logger_info.info(f"Fold {fold + 1}/{k_folds}")
        
        save_dir_k = join(save_dir, f"k_{fold}") # make subfolder for the given fold
        logs_dir = join(save_dir_k, "logs")      # save dir for Tensorboard/CSV logs and checkpoints
        os.makedirs(logs_dir, exist_ok=True)

        # Convert local indices to global indices
        train_global_idx = train_val_idx[train_local_idx]
        val_global_idx = train_val_idx[val_local_idx]

        # define loggers and callbacks
        #logger = TensorBoardLogger(save_dir=logs_dir, name="tensorboard")
        logger_csv = CSVLogger(save_dir=logs_dir, name="csv")
        checkpoint_callback = get_checkpoint_callback(save_dir_k,config["training"]["save_ckpt_every_n_epochs"])        
        manifest_callback  = PerRunManifest(config, path=save_dir_k, tag=experiment_tag)
        
        # Add new callbacks for best checkpoint tracking and data index saving
        best_checkpoint_callback = BestCheckpointCallback(save_dir_k, monitor="val_mae", mode="min")
        data_index_tracker = DataIndexTracker(save_dir_k, fold, train_global_idx, val_global_idx, test_idx)
        
        callbacks_list = [cb for cb in checkpoint_callback]
        callbacks_list.append(manifest_callback)
        callbacks_list.append(best_checkpoint_callback)
        callbacks_list.append(data_index_tracker)
        
        # Add attention entropy logging callback
        #entropy_enabled = True
        #callbacks_list.append(AttentionEntropyLogger(enabled=entropy_enabled))
        
        if "early_stopping" in config["special"]["mode"]:
            callbacks_list.append(early_stopping_callbacks)
        
        if debug:
            callbacks_list.append(MemoryLoggerCallback())
        
        if "debug_optimizer" in config["special"]["mode"]:
            callbacks_list.append(GradientLogger())
            callbacks_list.append(LearningRateMonitor(logging_interval="epoch"))
            # callbacks_list.append(MetricsAggregator())
        
        # update ds
        dm.update_idx(train_idx=train_local_idx, val_idx=val_local_idx, test_idx=test_idx)

        trainer = pl.Trainer(
            callbacks=callbacks_list,
            logger=logger_csv, #[logger, logger_csv],
            accelerator="gpu" if torch.cuda.is_available() else "auto",
            devices=1 if cluster else "auto",
            #overfit_batches=1 if debug else 0,
            max_epochs=config["training"]["max_epochs"],
            log_every_n_steps= 1,
            deterministic=True,
            enable_progress_bar=False if cluster else True,  # Disables the progress bar
            enable_model_summary=False if cluster else True,
            detect_anomaly=True if debug else False,
        )
        # * other stuff we can do
        # trainer.tune() to find optimal hyperparameters

        # training
        trainer.fit(
            model,
            dm,
            ckpt_path=resume_ckpt, # resume training from checkpoint
        )

        # validation
        trainer.validate(model, dm)
        val_metrics = trainer.callback_metrics.copy()
        
        # test
        trainer.test(model, dm)
        test_metrics = trainer.callback_metrics.copy()
        
        # Determine which metrics to use based on 'best' parameter
        if best:
            # Use metrics from the best checkpoint
            best_metrics_file = join(save_dir_k, "best_metrics.json")
            if os.path.exists(best_metrics_file):
                with open(best_metrics_file, 'r') as f:
                    best_metrics_data = json.load(f)
                
                # Extract metrics (excluding metadata like best_epoch, best_checkpoint_path)
                fold_metrics = {k: v for k, v in best_metrics_data.items() 
                               if k not in ['best_epoch', 'best_checkpoint_path']}
                best_checkpoint_path = best_metrics_data.get('best_checkpoint_path')
            else:
                # Fallback to current metrics if best not available
                fold_metrics = {**val_metrics, **test_metrics}
                best_checkpoint_path = None
        else:
            # Use current behavior - final epoch metrics
            fold_metrics = {**val_metrics, **test_metrics}
            best_checkpoint_path = None
        
        # Update metrics dictionary
        metrics_dict[fold] = fold_metrics
        
        # Add fold result to tracker
        kfold_tracker.add_fold_result(fold, fold_metrics, best_checkpoint_path)
    
    # Convert the dictionary to a pandas DataFrame
    df_metric = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df_metric = df_metric.applymap(lambda x: x.item() if isinstance(x, torch.Tensor) else x) # Convert tensor values to floats
    
    if plot_pred_check:
        mk_quick_pred_plot(model=model, dm=dm, val_idx=config["data"]["val_idx"], save_dir=save_dir)

    return df_metric



def get_model_object(config: dict)->pl.LightningModule:
    
    model_obj = config["model"]["model_object"]
    available_models = ["proT","LSTM","GRU", "TCN", "MLP"]
    
    assert model_obj in available_models, AssertionError(f"{model_obj} unavailable! Choose between {available_models}")

    MODEL_REGISTRY = {
    "proT": TransformerForecaster,
    # "GRU" : RNNForecaster,
    # "LSTM": RNNForecaster,
    # "TCN": RNNForecaster,
    # "MLP": RNNForecaster,
    }
    return MODEL_REGISTRY[model_obj]






if __name__ == "__main__":
    
    """
    Run a quick test
    """
    import re
    from proT.paths import ROOT_DIR, DATA_DIR, EXPERIMENTS_DIR
    
    exp_dir = EXPERIMENTS_DIR / "example"
    data_dir = str(DATA_DIR)
    
    # look for config file
    pattern_config = re.compile(r'config_.*\.yaml')
    config_matching_files = [file for file in os.listdir(exp_dir) if pattern_config.match(file)]
    if len(config_matching_files) == 1:
        config = OmegaConf.load(join(exp_dir,config_matching_files[0]))
    else:
        raise ValueError(f"None or more than one config file found in {exp_dir}")
    
    
    config_updated = update_config(config)
        
    save_dir = exp_dir
        
    trainer(
        config = config_updated,
        data_dir = data_dir, 
        save_dir = save_dir,
        experiment_tag = "test", 
        cluster = False, 
        resume_ckpt = None,
        plot_pred_check = True,
        debug = True)
