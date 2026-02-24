"""
CausaliT Parameter Sweep CLI

This module provides a command-line interface for running parameter sweeps
for the causaliT project. It supports both independent and combination sweeps,
with sequential or parallel execution modes.

Wired to use:
- Training function: causaliT.training.trainer.trainer
- Config preprocessing: causaliT.training.experiment_control.update_config
"""

# Standard library imports
import logging
import sys
from os import makedirs
from os.path import abspath, join, exists, dirname
from pathlib import Path

# Third-party imports
import click
from omegaconf import OmegaConf

# =============================================================================
# Project root directory (causaliT repository root)
# =============================================================================
# euler_sweep/euler_sweep/cli.py -> euler_sweep/euler_sweep -> euler_sweep -> causaliT -> ROOT
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

# Import the sweep framework
from causaliT.euler_sweep.euler_sweep.sweeper import run_sequential_sweep, run_parallel_sweep

# Import causaliT training components
from causaliT.training.trainer import trainer
from causaliT.training.experiment_control import update_config

# =============================================================================
# CausaliT Training Function Wrapper
# =============================================================================
def train_function_for_sweep(
    config: OmegaConf,
    save_dir: Path,
    data_dir: Path,
    cluster: bool,
    **kwargs
):
    """
    Training function wrapper for causaliT parameter sweeps.
    
    This wrapper:
    1. Applies update_config() preprocessing to handle config placeholders
    2. Calls the causaliT trainer with proper arguments
    
    Args:
        config: Configuration object (OmegaConf) with all hyperparameters
        save_dir: Directory to save outputs (checkpoints, logs, results)
        data_dir: Directory containing training data
        cluster: Whether running on a cluster (affects num_workers, etc.)
        **kwargs: Additional arguments passed to trainer
        
    Returns:
        pd.DataFrame: DataFrame containing metrics for each fold from trainer
    """
    # Apply config preprocessing (handles d_model calculations, etc.)
    config_updated = update_config(config)
    
    # Call causaliT trainer
    return trainer(
        config=config_updated,
        save_dir=str(save_dir),
        data_dir=str(data_dir),
        cluster=cluster,
        **kwargs
    )


# =============================================================================
# CLI Commands
# =============================================================================
@click.group()
def cli():
    """Parameter Sweep CLI - Run systematic parameter explorations."""
    pass


# =============================================================================
# SWEEP COMMAND
# =============================================================================
@click.command()
@click.option(
    "--exp_id",
    required=True,
    help="Experiment ID (folder name containing config.yaml and sweep.yaml)"
)
@click.option(
    "--sweep_mode",
    required=True,
    type=click.Choice(['independent', 'combination']),
    help="Sweep mode: 'independent' (one param at a time) or 'combination' (all combinations)"
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run in parallel using SLURM job arrays (cluster only)"
)
@click.option(
    "--cluster",
    default=False,
    is_flag=True,
    help="Running on cluster (affects paths and resource usage)"
)
@click.option(
    "--scratch_path",
    default=None,
    help="Scratch path for cluster execution (e.g., $SCRATCH/my_exp)"
)
# SLURM parameters (only used with --parallel)
@click.option(
    "--max_concurrent_jobs",
    default=6,
    type=int,
    help="Maximum concurrent SLURM jobs (default: 6)"
)
@click.option(
    "--walltime",
    default="4:00:00",
    help="SLURM walltime limit (default: 4:00:00)"
)
@click.option(
    "--gpu_mem",
    default="11g",
    help="GPU memory requirement (default: 11g)"
)
@click.option(
    "--mem_per_cpu",
    default="10g",
    help="CPU memory requirement (default: 10g)"
)
@click.option(
    "--submit_jobs",
    default=True,
    is_flag=True,
    help="Actually submit jobs (False for dry run)"
)
def sweep(exp_id, sweep_mode, parallel, cluster, scratch_path,
          max_concurrent_jobs, walltime, gpu_mem, mem_per_cpu, submit_jobs):
    """
    Run parameter sweeps with various execution modes.
    
    This command runs systematic parameter explorations defined in sweeper/sweep.yaml.
    
    Sweep Modes:
      - independent: Vary one parameter at a time (baseline comparison)
      - combination: Explore all combinations (Cartesian product)
    
    Execution Modes:
      - Sequential (default): Run combinations one after another
      - Parallel (--parallel): Use SLURM job arrays for cluster parallelization
    
    Examples:
      
      # Sequential independent sweep
      python cli.py sweep --exp_id my_exp --sweep_mode independent
      
      # Sequential combination sweep
      python cli.py sweep --exp_id my_exp --sweep_mode combination
      
      # Parallel combination sweep on cluster
      python cli.py sweep --exp_id my_exp --sweep_mode combination \\
          --parallel --cluster --scratch_path $SCRATCH/my_exp \\
          --max_concurrent_jobs 10
      
      # Dry run (generate scripts without submitting)
      python cli.py sweep --exp_id my_exp --sweep_mode combination \\
          --parallel --submit_jobs False
    
    Directory Structure:
      
      Independent sweep creates:
        experiments/my_exp/
        └── sweeps/
            ├── sweep_param1/
            │   ├── sweep_param1_value1/
            │   └── sweep_param1_value2/
            └── sweep_param2/
                └── ...
      
      Combination sweep creates:
        experiments/my_exp/
        └── combinations/
            ├── combo_param1_val1_param2_val1/
            ├── combo_param1_val1_param2_val2/
            └── ...
    """
    print(f"Starting parameter sweep: exp_id={exp_id}, mode={sweep_mode}, parallel={parallel}")
    
    # =============================================================================
    # Validate execution mode
    # =============================================================================
    if parallel and not cluster:
        raise ValueError(
            "Parallel execution (--parallel) requires cluster mode (--cluster).\n"
            "Parallel sweeps use SLURM job arrays which are only available on clusters.\n"
            "For local execution, use sequential mode (omit --parallel flag)."
        )
    
    # =============================================================================
    # Set up directories for causaliT project
    # =============================================================================
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments", exp_id)
        home_exp_dir = exp_dir
    else:
        exp_dir = scratch_path
        home_exp_dir = join(ROOT_DIR, "experiments", exp_id)
    
    # Data directory
    data_dir = join(ROOT_DIR, "data")
    
    # Check if experiment directory exists
    check_dir = home_exp_dir if scratch_path is not None else exp_dir
    if not exists(check_dir):
        raise ValueError(f"Experiment directory does not exist: {check_dir}")
    
    # Check for required config files (supports config*.yaml pattern)
    import glob
    config_pattern = join(check_dir, "config*.yaml")
    config_files = glob.glob(config_pattern)
    sweeper_dir = join(check_dir, "sweeper")
    sweep_path = join(sweeper_dir, "sweep.yaml")
    
    if not config_files:
        raise ValueError(
            f"Config file not found in: {check_dir}\n"
            "Create a config.yaml (or config_*.yaml) file in your experiment directory."
        )
    
    if not exists(sweeper_dir):
        raise ValueError(
            f"Sweeper directory not found: {sweeper_dir}\n"
            "Create a 'sweeper' subdirectory in your experiment folder.\n"
            f"Expected structure: {check_dir}/sweeper/sweep.yaml"
        )
    
    if not exists(sweep_path):
        raise ValueError(
            f"Sweep file not found: {sweep_path}\n"
            "Create a sweep.yaml file in the sweeper subdirectory.\n"
            f"Expected location: {check_dir}/sweeper/sweep.yaml"
        )
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Config: {config_files[0]}")
    print(f"Sweep: {sweep_path}")
    
    # =============================================================================
    # CausaliT training function
    # =============================================================================
    train_fn = train_function_for_sweep
    
    # =============================================================================
    # Execute sweep based on mode
    # =============================================================================
    if not parallel:
        # Sequential sweep
        print(f"\nRunning sequential {sweep_mode} sweep...")
        print("This will run combinations one after another.\n")
        
        run_sequential_sweep(
            exp_dir=exp_dir,
            sweep_mode=sweep_mode,
            train_fn=train_fn,
            data_dir=data_dir,
            cluster=cluster,
            experiment_id=exp_id  # Pass exp_id for unique folder naming
        )
        
        print("\n" + "="*60)
        print("Sequential sweep completed!")
        print("="*60)
        
        if sweep_mode == "independent":
            print(f"Results: {exp_dir}/sweeps/")
        else:
            print(f"Results: {exp_dir}/combinations/")
        print("="*60 + "\n")
    
    else:
        # Parallel sweep using SLURM job arrays
        print(f"\nPreparing parallel {sweep_mode} sweep...")
        print(f"Max concurrent jobs: {max_concurrent_jobs}")
        print(f"Walltime: {walltime}")
        print(f"GPU memory: {gpu_mem}")
        print(f"CPU memory: {mem_per_cpu}\n")
        
        # Prepare SLURM parameters
        slurm_params = {
            'max_concurrent_jobs': max_concurrent_jobs,
            'walltime': walltime,
            'gpu_mem': gpu_mem,
            'mem_per_cpu': mem_per_cpu
        }
        
        # For parallel execution, specify the training function by module and name
        # so it can be imported by worker jobs on cluster nodes
        train_fn_module = "causaliT.euler_sweep.euler_sweep.cli"
        train_fn_name = "train_function_for_sweep"
        
        run_parallel_sweep(
            exp_dir=exp_dir,
            home_exp_dir=home_exp_dir,
            sweep_mode=sweep_mode,
            train_fn_module=train_fn_module,
            train_fn_name=train_fn_name,
            experiment_id=exp_id,
            data_dir=data_dir,
            scratch_path=scratch_path,
            slurm_params=slurm_params,
            cluster=cluster,
            submit_jobs=submit_jobs
        )


# =============================================================================
# Register commands with CLI
# =============================================================================
cli.add_command(sweep)


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    cli()
