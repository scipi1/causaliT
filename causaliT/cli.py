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

if __name__ == "__main__":
    cli()
