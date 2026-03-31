# Standard library imports
import logging
import sys
from os import makedirs
from os.path import abspath, join, exists, dirname
from pathlib import Path

# Third-party imports
import click
from omegaconf import OmegaConf

# Local imports
sys.path.append(dirname(dirname(abspath(__file__))))
from causaliT.training.experiment_control import find_yml_files
from causaliT.core.utils import mk_fname
from causaliT.training.trainer import trainer
from causaliT.training.staged_trainer import staged_trainer, check_staged_training_config


@click.group()
def cli():
    pass

# TRAINING
@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--exp_tag", default="NA", help="Tag for model manifest")
@click.option("--scratch_path", default=None, help="SCRATCH path") # for the cluster
@click.option("--resume_checkpoint", default=None, help="Resume training from checkpoint")
@click.option("--plot_pred_check", default=True, help="Set to True for a quick prediction plot after training")
def train(exp_id, debug, cluster, exp_tag, scratch_path, resume_checkpoint, plot_pred_check):
    
    # Get folders
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    print(exp_id)
    print(scratch_path)
    
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments/", exp_id)
    else:
        exp_dir = join(scratch_path)
        
    # Data directory: use ROOT_DIR/data (not data/input) to match actual structure
    data_dir = join(ROOT_DIR, "data")
    
    
    # Create logs directory if it doesn't exist
    logs_dir = join(ROOT_DIR, "logs")
    makedirs(logs_dir, exist_ok=True)
    
    # Create loggers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger_info = logging.getLogger("logger_info")
    info_handler = logging.FileHandler(join(logs_dir,  mk_fname(filename="log", label="train", suffix="log")))
    logger_info.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger_info.addHandler(info_handler)
    
    if debug:
        # memory logger
        logger_memory = logging.getLogger("logger_memory")
        memory_handler = logging.FileHandler(join(logs_dir, mk_fname(filename="log", label="memory", suffix="log")))
        logger_memory.setLevel(logging.INFO)
        memory_handler.setFormatter(formatter)
        logger_memory.addHandler(memory_handler)
    
    # Load config file (ignoring sweep config)
    config, _ = find_yml_files(dir=exp_dir)
    
    # Run training once with the loaded config
    trainer(
        config=config, 
        data_dir=data_dir, 
        save_dir=exp_dir, 
        cluster=cluster,
        experiment_tag=exp_tag,
        resume_ckpt=resume_checkpoint,
        plot_pred_check=plot_pred_check,
        debug=debug)


cli.add_command(train)


# STAGED TRAINING
@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--exp_tag", default="NA", help="Tag for model manifest")
@click.option("--scratch_path", default=None, help="SCRATCH path for cluster")
@click.option("--resume_checkpoint", default=None, help="Resume from checkpoint (skips calibration/causal_init)")
@click.option("--plot_pred_check", default=True, help="Plot prediction check after training")
def calitrain(exp_id, debug, cluster, exp_tag, scratch_path, resume_checkpoint, plot_pred_check):
    """
    Run staged training pipeline: calibration → causal_init → training.
    
    The staged training addresses the 'flat HSIC landscape' problem by:
    1. Stage 0 (Calibration): Find λ_group for gradient balance
    2. Stage 1 (Causal Init): Pre-train with HSIC-dominated loss
    3. Stage 2 (Main Training): Standard training with annealing
    
    Enable stages in config:
        staged_training:
          use_calibration: true
          use_causal_init: true
    
    Example:
        python -m causaliT.cli staged_train --exp_id single/scm6/my_exp
    """
    
    # Get folders
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    print(f"Staged Training: {exp_id}")
    print(f"Scratch path: {scratch_path}")
    
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments/", exp_id)
    else:
        exp_dir = join(scratch_path)
        
    # Data directory
    data_dir = join(ROOT_DIR, "data")
    
    # Create logs directory
    logs_dir = join(ROOT_DIR, "logs")
    makedirs(logs_dir, exist_ok=True)
    
    # Create loggers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger_info = logging.getLogger("logger_info")
    info_handler = logging.FileHandler(join(logs_dir, mk_fname(filename="log", label="staged_train", suffix="log")))
    logger_info.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger_info.addHandler(info_handler)
    
    if debug:
        logger_memory = logging.getLogger("logger_memory")
        memory_handler = logging.FileHandler(join(logs_dir, mk_fname(filename="log", label="memory", suffix="log")))
        logger_memory.setLevel(logging.INFO)
        memory_handler.setFormatter(formatter)
        logger_memory.addHandler(memory_handler)
    
    # Load config file
    config, _ = find_yml_files(dir=exp_dir)
    
    # Validate staged training config
    validation = check_staged_training_config(config)
    if validation["warnings"]:
        print("\n⚠️  Staged Training Warnings:")
        for w in validation["warnings"]:
            print(f"  - {w}")
    if validation["info"]:
        print("\nℹ️  Staged Training Info:")
        for i in validation["info"]:
            print(f"  - {i}")
    
    # Run staged training pipeline
    staged_trainer(
        config=config, 
        data_dir=data_dir, 
        save_dir=exp_dir, 
        cluster=cluster,
        experiment_tag=exp_tag,
        resume_ckpt=resume_checkpoint,
        plot_pred_check=plot_pred_check,
        debug=debug)


cli.add_command(calitrain)


if __name__ == "__main__":
    cli()
