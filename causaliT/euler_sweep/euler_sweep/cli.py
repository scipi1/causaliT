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
# CALIBRATED SWEEP COMMAND
# =============================================================================
@click.command()
@click.option(
    "--exp_id",
    required=True,
    help="Experiment ID (folder containing config.yaml and sweeper/sweep.yaml)"
)
@click.option(
    "--cluster",
    default=False,
    is_flag=True,
    help="Running on cluster (affects paths and resource usage)"
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run in parallel using SLURM job arrays (cluster only)"
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
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for calibration (default: 42)"
)
@click.option(
    "--skip_calibration",
    default=False,
    is_flag=True,
    help="Skip calibration (use existing pre_sweep_calibration.json if found)"
)
@click.option(
    "--analysis_only",
    default=False,
    is_flag=True,
    help="Only run post-sweep analysis (skip training sweep)"
)
@click.option(
    "--selection_rule",
    default="1se",
    type=click.Choice(["1se", "min_hsic"]),
    help="Lambda selection rule for post-sweep analysis (default: 1se)"
)
def calisweep(exp_id, cluster, parallel, scratch_path,
              max_concurrent_jobs, walltime, gpu_mem, mem_per_cpu, submit_jobs,
              seed, skip_calibration, analysis_only, selection_rule):
    """
    Run a score-sparsity sweep with automatic group-L1 calibration.

    This command implements the full lambda_score selection pipeline:

    1. CALIBRATION (pre-sweep): Finds optimal lambda_group s.t. ||grad_Recon|| ~ ||grad_HSIC||.
       This is dataset-specific and must run before the score-sparsity grid.
       Output: lambda_group*, lambda_hsic_cross*, lambda_hsic_self*

    2. SCORE-SPARSITY SWEEP: Sweeps lambda_cross_score_sparse over the grid
       defined in sweeper/sweep.yaml.  Every combination inherits the calibrated
       lambda_group from step 1.

    3. ANALYSIS: Produces the LASSO-path plots and selects lambda* using the
       specified rule (1se or min_hsic).

    Execution Modes:
      - Sequential (default): Run combinations one after another
      - Parallel (--parallel): Use SLURM job arrays for cluster parallelization.
        Calibration runs on the submit node, then each sweep combination is
        dispatched as a SLURM array task.  Post-sweep analysis is skipped
        automatically; run with --analysis_only after all jobs complete.

    sweep.yaml format for this command::

        training:
          lambda_cross_score_sparse: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
          # optionally also sweep seed for robustness:
          # seed: [0, 1, 2, 3, 4]

    After running, inspect:
      - sweeper/score_sparsity_path.png
      - sweeper/variable_importance_path.png
      - sweeper/score_sparsity_analysis.json  (contains selected lambda)

    Example::

      # Sequential (local or single-node)
      python cli.py calisweep --exp_id my_score_sparsity_exp
      python cli.py calisweep --exp_id my_score_sparsity_exp --skip_calibration

      # Parallel on cluster (SLURM job arrays)
      python cli.py calisweep --exp_id my_score_sparsity_exp \\
          --parallel --cluster --scratch_path $SCRATCH/my_score_sparsity_exp \\
          --max_concurrent_jobs 10

      # Post-sweep analysis (after parallel jobs finish)
      python cli.py calisweep --exp_id my_score_sparsity_exp --analysis_only
    """
    print(f"Starting calibrated score-sparsity sweep: exp_id={exp_id}")

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
    # Set up directories (mirrors sweep command logic)
    # =============================================================================
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments", exp_id)
        home_exp_dir = exp_dir
    else:
        exp_dir = scratch_path
        home_exp_dir = join(ROOT_DIR, "experiments", exp_id)

    data_dir = join(ROOT_DIR, "data")

    check_dir = home_exp_dir if scratch_path is not None else exp_dir
    if not exists(check_dir):
        raise ValueError(f"Experiment directory does not exist: {check_dir}")

    # ── BUILD PRE-SWEEP HOOK ─────────────────────────────────────────────────
    from causaliT.euler_sweep.euler_sweep.pre_sweep_actions import (
        make_calibration_pre_sweep,
        make_noop_pre_sweep,
        load_pre_sweep_calibration,
    )

    if skip_calibration:
        # Try to load existing calibration
        existing = load_pre_sweep_calibration(check_dir)
        if existing is not None:
            print("Using existing calibration from pre_sweep_calibration.json")
            pre_fn = lambda config, data_dir, save_dir: existing
        else:
            print("No existing calibration found. Running noop pre-sweep.")
            pre_fn = make_noop_pre_sweep()
    else:
        pre_fn = make_calibration_pre_sweep(seed=seed)

    # ── RUN SWEEP (skip if analysis_only) ───────────────────────────────────
    if not analysis_only:
        if not parallel:
            # ── SEQUENTIAL execution ────────────────────────────────────────
            print(f"\nRunning sequential score-sparsity sweep (combination mode)...")
            run_sequential_sweep(
                exp_dir=exp_dir,
                sweep_mode="combination",
                train_fn=train_function_for_sweep,
                data_dir=data_dir,
                cluster=cluster,
                experiment_id=exp_id,
                pre_sweep_fn=pre_fn,
            )
            print("\nScore-sparsity sweep complete.")

        else:
            # ── PARALLEL execution (SLURM job arrays) ──────────────────────
            # Step 1: Run calibration on the submit node BEFORE dispatching
            # the parallel sweep.  The calibration overrides are baked into
            # the base config that run_parallel_sweep serialises to JSON,
            # so every SLURM worker inherits the calibrated values.
            from causaliT.euler_sweep.euler_sweep.sweeper import find_config_files

            config, _sweep_config = find_config_files(check_dir)

            print("\nRunning pre-sweep calibration on submit node...")
            overrides = pre_fn(
                config=OmegaConf.to_container(config, resolve=False),
                data_dir=data_dir,
                save_dir=check_dir,
            )
            if overrides:
                config = OmegaConf.merge(config, OmegaConf.create(overrides))
                print(f"Pre-sweep overrides applied: {list(overrides.keys())}")

            # Save the calibrated config back so run_parallel_sweep picks it up
            # when it calls find_config_files(config_dir).
            # We write to a temporary calibrated config in the sweeper dir and
            # use it as the base config for the parallel sweep.
            import glob
            config_pattern = join(check_dir, "config*.yaml")
            config_files = glob.glob(config_pattern)
            config_files.sort()
            calibrated_config_path = config_files[0]
            # Save calibrated config (overwrite the original so workers use it)
            OmegaConf.save(config, calibrated_config_path)
            print(f"Calibrated config saved to: {calibrated_config_path}")

            # Step 2: Dispatch parallel sweep
            print(f"\nPreparing parallel score-sparsity sweep...")
            print(f"Max concurrent jobs: {max_concurrent_jobs}")
            print(f"Walltime: {walltime}")
            print(f"GPU memory: {gpu_mem}")
            print(f"CPU memory: {mem_per_cpu}\n")

            slurm_params = {
                'max_concurrent_jobs': max_concurrent_jobs,
                'walltime': walltime,
                'gpu_mem': gpu_mem,
                'mem_per_cpu': mem_per_cpu,
            }

            train_fn_module = "causaliT.euler_sweep.euler_sweep.cli"
            train_fn_name = "train_function_for_sweep"

            run_parallel_sweep(
                exp_dir=exp_dir,
                home_exp_dir=home_exp_dir,
                sweep_mode="combination",
                train_fn_module=train_fn_module,
                train_fn_name=train_fn_name,
                experiment_id=exp_id,
                data_dir=data_dir,
                scratch_path=scratch_path,
                slurm_params=slurm_params,
                cluster=cluster,
                submit_jobs=submit_jobs,
            )

            # Analysis is skipped for parallel — jobs haven't finished yet
            print("\n" + "=" * 60)
            print("PARALLEL CALIBRATED SWEEP DISPATCHED")
            print("=" * 60)
            print("SLURM jobs are running.  Post-sweep analysis is NOT run")
            print("automatically because the jobs have not completed yet.")
            print("\nOnce all jobs finish, run analysis with:")
            print(f"  python cli.py calisweep --exp_id {exp_id} --analysis_only")
            print("=" * 60 + "\n")
            return  # Exit early — no analysis

    # ── POST-SWEEP ANALYSIS ──────────────────────────────────────────────────
    # Reached for: sequential sweep completion, or --analysis_only
    print(f"\nRunning score-sparsity analysis...")
    from causaliT.evaluation.eval_score_sparsity import run_score_sparsity_analysis

    # For analysis, always use home_exp_dir (results may have been copied back)
    analysis_dir = home_exp_dir if scratch_path is not None else exp_dir

    try:
        analysis_result = run_score_sparsity_analysis(
            sweep_dir=analysis_dir,
            rule=selection_rule,
            show_plots=False,
            save_dir=join(analysis_dir, "sweeper"),
        )
        lambda_star = analysis_result["lambda_cross_score_selected"]

        print("\n" + "=" * 60)
        print("CALIBRATED SWEEP COMPLETE")
        print("=" * 60)
        print(f"  Selected lambda_cross_score = {lambda_star:.4f}")
        print(f"  Selection rule: {selection_rule}")
        print(f"  Analysis:  {analysis_dir}/sweeper/score_sparsity_analysis.json")
        print(f"  LASSO plot: {analysis_dir}/sweeper/score_sparsity_path.png")
        print(f"  Var plot:   {analysis_dir}/sweeper/variable_importance_path.png")
        print("=" * 60)
        print(
            f"\nNext step: set training.lambda_cross_score_sparse = {lambda_star:.4f} "
            "in your experiment config and run causal_initialization."
        )
    except Exception as e:
        print(f"\nWarning: Analysis failed: {e}")
        print("The sweep results are saved. Run analysis manually if needed.")


# =============================================================================
# Register commands with CLI
# =============================================================================
cli.add_command(sweep)
cli.add_command(calisweep)


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    cli()
