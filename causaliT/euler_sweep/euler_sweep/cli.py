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
from causaliT.training.staged_trainer import staged_trainer
from causaliT.training.adaptive_trainer import adaptive_trainer
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


def staged_train_function_for_sweep(
    config: OmegaConf,
    save_dir: Path,
    data_dir: Path,
    cluster: bool,
    **kwargs
):
    """
    Staged training function wrapper for calibrated parameter sweeps.

    Like ``train_function_for_sweep`` but delegates to
    ``staged_trainer`` which runs:
    1. (Optional) causal initialization
    2. Main training with warm-start from init checkpoint

    Use this for experiments that require score-sparsity CV or
    causal initialization before the main training loop.

    Args:
        config: Configuration object (OmegaConf) with all hyperparameters
        save_dir: Directory to save outputs (checkpoints, logs, results)
        data_dir: Directory containing training data
        cluster: Whether running on a cluster (affects num_workers, etc.)
        **kwargs: Additional arguments passed to staged_trainer

    Returns:
        pd.DataFrame: DataFrame containing metrics for each fold
    """
    config_updated = update_config(config)

    return staged_trainer(
        config=config_updated,
        save_dir=str(save_dir),
        data_dir=str(data_dir),
        cluster=cluster,
        **kwargs
    )


def adaptive_train_function_for_sweep(
    config: OmegaConf,
    save_dir: Path,
    data_dir: Path,
    cluster: bool,
    **kwargs
):
    """
    Adaptive alternating training function wrapper for parameter sweeps.

    Like ``staged_train_function_for_sweep`` but delegates to
    ``adaptive_trainer`` which runs the metric-driven reconstruct/structure
    schedule defined in ``config['adaptive_training']`` (a single in-memory
    ``pl.Trainer.fit()`` that switches phases automatically on ``val_x_mae``).

    Requires ``training.use_gradient_routing=True``.  Structure-phase loss
    weights (e.g. ``lambda_l0``) live under ``adaptive_training.structure``;
    sweep them via interpolation from ``experiment.*`` scalars so the generic
    flat ``category.param`` grid can reach the nested value.

    Args:
        config: Configuration object (OmegaConf) with all hyperparameters
        save_dir: Directory to save outputs (checkpoints, logs, results)
        data_dir: Directory containing training data
        cluster: Whether running on a cluster (affects num_workers, etc.)
        **kwargs: Additional arguments passed to adaptive_trainer

    Returns:
        pd.DataFrame: Adaptive-run summary metrics
    """
    config_updated = update_config(config)

    return adaptive_trainer(
        config=config_updated,
        save_dir=str(save_dir),
        data_dir=str(data_dir),
        cluster=cluster,
        **kwargs
    )


def benchmark_function_for_sweep(
    config: OmegaConf,
    save_dir: Path,
    data_dir: Path,
    cluster: bool,
    **kwargs
):
    """
    Benchmark "training" function wrapper for parameter / DAG sweeps.

    Drop-in replacement for ``train_function_for_sweep`` that fits the external
    structure learners (NOTEARS, DAGMA, PC) instead of a causaliT model.  It
    keeps the sweep signature so a DAG sweep can run baselines on exactly the
    same generated datasets by setting ``training.trainer: benchmark``.

    No model is trained: the methods read ``ds.npz`` directly and each writes
    ``eval/eval_benchmark_<method>/`` inside ``save_dir`` with the standard
    ``dag_metrics.json`` produced by ``write_dag_report``.  The methods and their
    settings come from the ``benchmark`` section of the config (see
    ``causaliT.benchmarks.runner.DEFAULT_BENCHMARK_CONFIG``).

    Args:
        config: Staged run config; ``data.dataset`` selects the dataset and the
            optional ``benchmark`` section selects methods/seeds/threshold.
        save_dir: Run folder that receives the eval subfolders.
        data_dir: Directory containing the dataset folder (group-local in a DAG
            sweep), passed explicitly so no data-root resolution is needed.
        cluster: Ignored; benchmarks are single-process CPU fits.
        **kwargs: Forwarded as overrides to ``run_benchmarks``.

    Returns:
        pd.DataFrame: One row per method with the headline DAG metrics.  The same
        table is written to ``save_dir/benchmark_summary.csv``, because the sweep
        discards trainer return values - without the file a run folder could only
        be summarised by walking every eval subfolder.
    """

    import pandas as pd

    from causaliT.benchmarks.runner import run_benchmarks, summarize_benchmarks

    results = run_benchmarks(
        experiment=str(save_dir),
        datadir_path=None if data_dir is None else str(data_dir),
        overrides=kwargs or None,
    )
    summary = pd.DataFrame(summarize_benchmarks(results))
    summary.to_csv(Path(save_dir) / "benchmark_summary.csv", index=False)
    return summary



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
# CALISWEEP COMMAND — mirrors `sweep` but uses staged_trainer
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
def calisweep(exp_id, sweep_mode, parallel, cluster, scratch_path,
              max_concurrent_jobs, walltime, gpu_mem, mem_per_cpu, submit_jobs):
    """
    Run parameter sweeps using the staged trainer (causal init → main training).

    Identical to the ``sweep`` command but uses ``staged_trainer`` instead of
    ``trainer``.  The staged trainer runs:
    1. (Optional) causal initialization stage
    2. Main training with warm-start from the init checkpoint

    Use this for experiments whose config includes staged training settings
    (e.g. ``use_score_sparsity_cv``, ``causal_initialization``).

    Sweep Modes:
      - independent: Vary one parameter at a time (baseline comparison)
      - combination: Explore all combinations (Cartesian product)

    Execution Modes:
      - Sequential (default): Run combinations one after another
      - Parallel (--parallel): Use SLURM job arrays for cluster parallelization

    Examples::

      # Sequential combination sweep with staged trainer
      python cli.py calisweep --exp_id my_exp --sweep_mode combination

      # Parallel combination sweep on cluster
      python cli.py calisweep --exp_id my_exp --sweep_mode combination \\
          --parallel --cluster --scratch_path $SCRATCH/my_exp \\
          --max_concurrent_jobs 10
    """
    print(f"Starting staged parameter sweep: exp_id={exp_id}, mode={sweep_mode}, parallel={parallel}")

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
    print(f"Training function: staged_train_function_for_sweep (staged_trainer)")

    # =============================================================================
    # CausaliT training function — staged variant
    # =============================================================================
    train_fn = staged_train_function_for_sweep

    # =============================================================================
    # Execute sweep based on mode
    # =============================================================================
    if not parallel:
        # Sequential sweep
        print(f"\nRunning sequential {sweep_mode} sweep (staged trainer)...")
        print("This will run combinations one after another.\n")

        run_sequential_sweep(
            exp_dir=exp_dir,
            sweep_mode=sweep_mode,
            train_fn=train_fn,
            data_dir=data_dir,
            cluster=cluster,
            experiment_id=exp_id
        )

        print("\n" + "=" * 60)
        print("Sequential staged sweep completed!")
        print("=" * 60)

        if sweep_mode == "independent":
            print(f"Results: {exp_dir}/sweeps/")
        else:
            print(f"Results: {exp_dir}/combinations/")
        print("=" * 60 + "\n")

    else:
        # Parallel sweep using SLURM job arrays
        print(f"\nPreparing parallel {sweep_mode} sweep (staged trainer)...")
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
        train_fn_name = "staged_train_function_for_sweep"

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
# ADAPTIVESWEEP COMMAND — mirrors `calisweep` but uses adaptive_trainer
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
def adaptivesweep(exp_id, sweep_mode, parallel, cluster, scratch_path,
                  max_concurrent_jobs, walltime, gpu_mem, mem_per_cpu, submit_jobs):
    """
    Run parameter sweeps using the adaptive alternating trainer.

    Identical to the ``calisweep`` command but uses ``adaptive_trainer`` instead
    of ``staged_trainer``.  The adaptive trainer runs the metric-driven
    reconstruct/structure schedule defined in ``config['adaptive_training']``
    (single in-memory ``pl.Trainer.fit()`` switching phases on ``val_x_mae``).

    Structure-phase loss weights live under ``adaptive_training.structure``
    (e.g. ``lambda_l0``).  Because the generic sweeper only overrides flat
    ``category.param`` keys, sweep these via interpolation: add a scalar under
    ``experiment`` (e.g. ``experiment.lambda_l0_structure``), point
    ``adaptive_training.structure.lambda_l0`` at ``${experiment.lambda_l0_structure}``,
    and list ``experiment.lambda_l0_structure`` in ``sweeper/sweep.yaml``.

    Sweep Modes:
      - independent: Vary one parameter at a time (baseline comparison)
      - combination: Explore all combinations (Cartesian product)

    Examples::

      # Sequential 2D combination sweep with adaptive trainer
      python cli.py adaptivesweep --exp_id my_exp --sweep_mode combination

      # Parallel combination sweep on cluster
      python cli.py adaptivesweep --exp_id my_exp --sweep_mode combination \\
          --parallel --cluster --scratch_path $SCRATCH/my_exp \\
          --max_concurrent_jobs 10
    """
    print(f"Starting adaptive alternating parameter sweep: exp_id={exp_id}, mode={sweep_mode}, parallel={parallel}")

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
    print(f"Training function: adaptive_train_function_for_sweep (adaptive_trainer)")

    # =============================================================================
    # CausaliT training function — adaptive alternating variant
    # =============================================================================
    train_fn = adaptive_train_function_for_sweep

    # =============================================================================
    # Execute sweep based on mode
    # =============================================================================
    if not parallel:
        # Sequential sweep
        print(f"\nRunning sequential {sweep_mode} sweep (adaptive trainer)...")
        print("This will run combinations one after another.\n")

        run_sequential_sweep(
            exp_dir=exp_dir,
            sweep_mode=sweep_mode,
            train_fn=train_fn,
            data_dir=data_dir,
            cluster=cluster,
            experiment_id=exp_id
        )

        print("\n" + "=" * 60)
        print("Sequential adaptive alternating sweep completed!")
        print("=" * 60)

        if sweep_mode == "independent":
            print(f"Results: {exp_dir}/sweeps/")
        else:
            print(f"Results: {exp_dir}/combinations/")
        print("=" * 60 + "\n")

    else:
        # Parallel sweep using SLURM job arrays
        print(f"\nPreparing parallel {sweep_mode} sweep (adaptive trainer)...")
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
        train_fn_name = "adaptive_train_function_for_sweep"

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
# =============================================================================
# DAG SWEEP: optimize once per DAG size, then train every seed
# =============================================================================

@click.command()
@click.option(
    "--exp_id",
    required=True,
    help="Experiment folder (relative to experiments/) holding config*.yaml + dagsweep*.yaml",
)
@click.option(
    "--cluster",
    is_flag=True,
    default=False,
    help="Cluster mode: SUBMIT the sweep as a chained SLURM job graph (add "
         "--sequential to run linearly on a cluster node instead)",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="Force the in-process (linear) sweep even with --cluster",
)
@click.option(
    "--keep_data",
    is_flag=True,
    default=False,
    help="Keep every generated ds.npz instead of pruning it after each run (debug; costly)",
)

@click.option(
    "--skip_optuna",
    is_flag=True,
    default=False,
    help="Do not tune; reuse the best_trial.yaml already present in each group",
)
@click.option(
    "--force_optuna",
    is_flag=True,
    default=False,
    help="Re-run the study even when a group already has a best_trial.yaml",
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="Print the plan (and, with --cluster, write the SLURM scripts) without "
         "generating, training or submitting anything",
)
@click.option(
    "--scratch_path",
    default=None,
    help="Run the sweep in this folder instead of experiments/<exp_id> "
         "(cluster only; the spec files are copied there)",
)
# SLURM parameters (only used with --cluster)
@click.option(
    "--max_concurrent_jobs",
    default=6,
    type=int,
    help="Maximum concurrent array tasks, for BOTH the trial and the train array",
)
@click.option(
    "--walltime",
    default="4:00:00",
    help="SLURM walltime, applied to every job of the chain (default: 4:00:00)",
)
@click.option(
    "--gpu_mem",
    default="11g",
    help="GPU memory requirement of the trial/train arrays (default: 11g)",
)
@click.option(
    "--mem_per_cpu",
    default="10g",
    help="CPU memory requirement (default: 10g)",
)
@click.option(
    "--venv_path",
    default="$HOME/myenv",
    help="Python environment the SLURM jobs activate (default: $HOME/myenv)",
)
def dagsweep(exp_id, cluster, sequential, keep_data, skip_optuna, force_optuna,
             dry_run, scratch_path, max_concurrent_jobs, walltime, gpu_mem,
             mem_per_cpu, venv_path):

    """
    Run a grouped DAG sweep: one Optuna study per DAG size, shared by all seeds.

    Hyper-parameters are tuned once per group (e.g. per ``n_nodes``) on a
    dedicated optimisation DAG, then reused for every evaluation seed of that
    group. This turns the naive ``sizes x seeds x trials`` explosion into
    ``sizes studies + sizes x seeds`` runs.

    The evaluation phase uses TWO decoupled seeds (``dagsweep.yaml``):
    ``dag_seeds`` draws the graph and pins the data split
    (``training.data_seed``), while ``model_seeds`` varies only the weight
    initialisation (``training.seed``).  Several model seeds on one DAG give the
    per-edge stability of the learned structure; omit ``model_seeds`` to keep the
    legacy one-run-per-DAG behaviour (model seed = DAG seed).

    Execution modes:

    \b
      (default)      linear, in-process: every trial then every run, one after
                     the other.  Use it locally and for smoke tests.
      --cluster      PARALLEL: submits a chained SLURM job graph
                     prep -> trials[array] -> select -> train[array] -> cleanup.
                     Trials and runs are parallelised inside their phase; the
                     `select` barrier guarantees every run uses the tuned model.
      --cluster --sequential
                     linear execution ON a cluster node (single job, no arrays).

    Progress of a submitted sweep: ``cli dagsweep-status --exp_id ...``.

    \b
    Examples:
        # local, linear
        python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 7_SCALING/atsel_nodes

        # cluster, parallel (10 concurrent GPU tasks per phase)
        python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 7_SCALING/atsel_nodes --cluster --max_concurrent_jobs 10 --walltime 24:00:00

        # inspect the job scripts without submitting
        python -m causaliT.euler_sweep.euler_sweep.cli dagsweep --exp_id 7_SCALING/atsel_nodes --cluster --dry_run

    """
    from causaliT.euler_sweep.euler_sweep.opt_train_sweep import run_dag_sweep

    home_exp_dir = join(ROOT_DIR, "experiments", exp_id)
    if not exists(home_exp_dir):
        raise click.ClickException(f"Experiment folder not found: {home_exp_dir}")

    parallel = cluster and not sequential

    if scratch_path is not None and not parallel:
        raise click.ClickException(
            "--scratch_path is only supported for the parallel cluster mode "
            "(--cluster without --sequential)."
        )

    print("=" * 60)
    print("DAG SWEEP" + ("  [PARALLEL / SLURM]" if parallel else "  [SEQUENTIAL]"))
    print("=" * 60)
    print(f"Experiment: {exp_id}")
    print(f"Folder:     {home_exp_dir}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Parallel: plan + submit the job chain, then return (nothing trains here)
    # -------------------------------------------------------------------------
    if parallel:
        from causaliT.euler_sweep.euler_sweep.dagsweep_parallel import (
            stage_experiment_to_scratch,
            submit_parallel_dag_sweep,
        )

        if scratch_path is not None:
            # Only the spec files are copied; datasets, results and sweep state
            # are written straight into scratch, so $HOME stays small.
            exp_dir = stage_experiment_to_scratch(home_exp_dir, scratch_path)
            print(f"Staged spec files to scratch: {exp_dir}")
        else:
            exp_dir = home_exp_dir

        submit_parallel_dag_sweep(
            exp_dir=exp_dir,
            home_exp_dir=home_exp_dir,
            slurm_params={
                "max_concurrent_jobs": max_concurrent_jobs,
                "walltime": walltime,
                "gpu_mem": gpu_mem,
                "mem_per_cpu": mem_per_cpu,
                "venv_path": venv_path,
            },
            keep_data=keep_data,
            skip_optuna=skip_optuna,
            force_optuna=force_optuna,
            submit_jobs=not dry_run,
        )
        return

    # -------------------------------------------------------------------------
    # Sequential (default): run everything in this process
    # -------------------------------------------------------------------------
    results = run_dag_sweep(
        exp_dir=home_exp_dir,
        cluster=cluster,
        keep_data=keep_data,
        skip_optuna=skip_optuna,
        force_optuna=force_optuna,
        dry_run=dry_run,
    )

    if not dry_run:
        for group_name, group in results.get("groups", {}).items():
            statuses = [r["status"] for r in group["seeds"].values()]
            ok = sum(1 for s in statuses if s == "ok")
            n_dags = len({r.get("dag_seed") for r in group["seeds"].values()})
            print(f"  {group_name}: {ok}/{len(statuses)} run(s) ok "
                  f"over {n_dags} DAG(s), "
                  f"{len(group['best_params'])} tuned param(s)")


@click.command(name="dagsweep-status")
@click.option(
    "--exp_id",
    required=True,
    help="Experiment folder (relative to experiments/) of a submitted DAG sweep",
)
@click.option(
    "--scratch_path",
    default=None,
    help="Read the sweep state from this folder instead (same value used at submit)",
)
def dagsweep_status(exp_id, scratch_path):
    """
    Report a parallel DAG sweep: PLANNED vs REACHED trials and runs.

    Rebuilt from the per-item files in ``<exp_dir>/dagsweep/progress/``, so it is
    accurate even when SLURM killed jobs at the walltime: each group shows its
    completed/failed trials, whether it ended up tuned, and the state of every
    run (with the error of the failed ones).
    """
    from causaliT.euler_sweep.euler_sweep.dagsweep_parallel import (
        format_progress,
        rebuild_progress,
    )

    exp_dir = scratch_path or join(ROOT_DIR, "experiments", exp_id)
    try:
        print(format_progress(rebuild_progress(exp_dir)))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))



@click.command(name="dagsweep-regen")
@click.option(
    "--dataset_dir",
    required=True,
    help="Path to a dataset folder containing dag_recipe.json",
)
def dagsweep_regen(dataset_dir):
    """
    Re-materialize a pruned sampled DAG from its ``dag_recipe.json``.

    DAG sweeps delete the heavy ``ds.npz`` after each run to keep disk usage
    flat. Because the recipe pins the seed and every generator argument, the
    exact same dataset can be rebuilt on demand - e.g. to re-run an evaluation
    months later.
    """
    from causaliT.euler_sweep.euler_sweep.dag_provider import regenerate_from_recipe

    path = regenerate_from_recipe(dataset_dir)
    print(f"Re-materialized: {path}")


@click.command(name="calibrate-batch-budget")
@click.option("--safety", default=None, type=float,
              help="Fraction of device memory allowed for activations (default 0.35)")
@click.option("--multiplicity", default=None, type=int,
              help="Live activation tensors of the dominant shape (default 12)")
@click.option("--dtype_bytes", default=4, type=int,
              help="Bytes per activation element (4 = fp32, 2 = fp16/bf16)")
@click.option("--no_cache", is_flag=True, default=False,
              help="Only print the measured budget; do not write the cache")
def calibrate_batch_budget(safety, multiplicity, dtype_bytes, no_cache):
    """
    Measure this machine's activation budget C for the size-derived batch size.

    ``dagsweep``'s ``size_derived`` rule ``activation_budget`` solves
    ``B = C / (N * H * (N + d))`` for the batch size, so C is the one
    device-specific number in the whole scaling sweep.  Running this once per
    machine writes it to ``~/.causalit/activation_budget.json`` (or
    ``$CAUSALIT_CACHE_DIR``), keyed by GPU name, so the same ``dagsweep.yaml``
    yields a device-appropriate batch on a laptop and on a cluster node.

    After an OOM, re-run with a smaller ``--safety`` (e.g. 0.2): the batch then
    shrinks at every DAG size at once, which keeps sizes comparable.

    \b
    Example:
        python -m causaliT.euler_sweep.euler_sweep.cli calibrate-batch-budget
    """
    from causaliT.euler_sweep.euler_sweep.batch_budget import (
        DEFAULT_MULTIPLICITY,
        DEFAULT_SAFETY,
        calibrate_activation_budget,
    )
    from causaliT.euler_sweep.euler_sweep.search_space import activation_batch_size

    report = calibrate_activation_budget(
        dtype_bytes=dtype_bytes,
        multiplicity=DEFAULT_MULTIPLICITY if multiplicity is None else multiplicity,
        safety=DEFAULT_SAFETY if safety is None else safety,
        write_cache=not no_cache,
    )

    print("=" * 60)
    print("ACTIVATION BUDGET CALIBRATION")
    print("=" * 60)
    for key in ("device", "total_bytes", "dtype_bytes", "multiplicity", "safety"):
        print(f"{key:>14}: {report[key]}")
    print(f"{'C':>14}: {report['C']:.4g}")
    if "cache_path" in report:
        print(f"{'cached in':>14}: {report['cache_path']}")

    # Show the resulting batch sizes so the number is interpretable.
    print("-" * 60)
    print("Derived batch size (d_ref = 2 * n_keys, n_heads = 4):")
    for n_keys in (10, 50, 100, 200, 400, 800):
        batch = activation_batch_size(n_keys, 2 * n_keys, 4, budget=report["C"])
        print(f"  n_keys={n_keys:>4} -> batch_size={batch}")
    print("=" * 60)


cli.add_command(sweep)
cli.add_command(calisweep)
cli.add_command(adaptivesweep)
cli.add_command(dagsweep)
cli.add_command(dagsweep_status)
cli.add_command(dagsweep_regen)

cli.add_command(calibrate_batch_budget)



# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    cli()
