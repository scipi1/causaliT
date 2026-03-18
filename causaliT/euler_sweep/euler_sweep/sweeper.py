"""
Generic Parameter Sweep Framework

This module provides a flexible, reusable framework for running parameter sweeps
in machine learning experiments. It supports both independent and combination sweeps,
with sequential or parallel execution modes.

Key Features:
- Independent sweep: Vary one parameter at a time
- Combination sweep: Explore all combinations of parameters (Cartesian product)
- Sequential execution: Run combinations one after another
- Parallel execution: Use SLURM job arrays for cluster parallelization
- SCRATCH support: Efficient use of cluster scratch storage

Architecture:
1. COMBINATION GENERATION: Generate parameter combinations to explore
2. SWEEP EXECUTION: Run combinations sequentially or in parallel
3. CORE EXECUTION: Execute training for individual combinations

Usage:
    See euler_sweep/cli.py for integration examples.
    
To customize for your project:
1. Pass your training function with signature: train_fn(config, save_dir, data_dir, **kwargs)
2. Define sweep parameters in sweep.yaml
3. Call run_sequential_sweep() or run_parallel_sweep()
"""

# Standard library imports
import itertools
import json
import logging
import subprocess
import warnings
from os import makedirs, scandir
from os.path import exists, join, dirname
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Optional, Any

# Third-party imports
from omegaconf import OmegaConf

# Local imports (deferred to avoid circular imports - see functions that need it)
# from causaliT.training.config_utils import populate_seq_lengths_from_dataset

# ════════════════════════════════════════════════════════════════════════════
# 1. COMBINATION GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_independent_combinations(config: OmegaConf, sweep_config: OmegaConf, experiment_id: str = None) -> List[Dict]:
    """
    Generate combinations for independent parameter sweep.
    
    In independent sweep mode, each parameter is varied one at a time while
    keeping all other parameters at their default values.
    
    Example:
        config: {param1: 10, param2: 20}
        sweep: {param1: [10, 20, 30], param2: [20, 40]}
        
        Generates:
        - param1=10, param2=20 (baseline)
        - param1=20, param2=20
        - param1=30, param2=20
        - param1=10, param2=40
    
    Args:
        config: Base configuration (OmegaConf)
        sweep_config: Sweep definition (OmegaConf)
        experiment_id: Experiment identifier to prefix folder names (optional)
        
    Returns:
        List of combination dictionaries, each containing:
            - 'params': dict of parameter changes {param_name: value}
            - 'categories': dict mapping param names to config categories
            - 'name': unique folder name for this combination (prefixed with experiment_id if provided)
            - 'description': human-readable description
    """
    combinations = []
    
    # Build name prefix from experiment_id if provided
    name_prefix = f"{experiment_id}_" if experiment_id else ""
    
    for category in sweep_config:
        for param_name in sweep_config[category]:
            # Verify parameter exists in config
            if param_name not in config[category]:
                raise AssertionError(
                    f"Parameter '{param_name}' not found in config category '{category}'"
                )
            
            for param_value in sweep_config[category][param_name]:
                combination = {
                    'params': {param_name: param_value},
                    'categories': {param_name: category},
                    'name': f"{name_prefix}sweep_{param_name}_{param_value}",
                    'description': f"{category}.{param_name}={param_value}"
                }
                combinations.append(combination)
    
    return combinations


def generate_all_combinations(config: OmegaConf, sweep_config: OmegaConf, experiment_id: str = None) -> List[Dict]:
    """
    Generate all possible parameter combinations (Cartesian product).
    
    In combination sweep mode, all possible combinations of all parameter values
    are explored. This grows exponentially with the number of parameters.
    
    Example:
        sweep: {param1: [10, 20], param2: [5, 10]}
        
        Generates 4 combinations:
        - param1=10, param2=5
        - param1=10, param2=10
        - param1=20, param2=5
        - param1=20, param2=10
    
    Args:
        config: Base configuration (OmegaConf)
        sweep_config: Sweep definition (OmegaConf)
        experiment_id: Experiment identifier to prefix folder names (optional)
        
    Returns:
        List of combination dictionaries with same structure as generate_independent_combinations
    """
    # Extract all parameters and their values
    param_values = {}
    param_categories = {}
    
    for category in sweep_config:
        for param_name in sweep_config[category]:
            # Verify parameter exists in config
            if param_name not in config[category]:
                raise AssertionError(
                    f"Parameter '{param_name}' not found in config category '{category}'"
                )
            param_values[param_name] = sweep_config[category][param_name]
            param_categories[param_name] = category
    
    # Generate Cartesian product of all parameter values
    param_names = list(param_values.keys())
    value_combinations = list(itertools.product(*(param_values[param] for param in param_names)))
    
    # Build name prefix from experiment_id if provided
    name_prefix = f"{experiment_id}_" if experiment_id else ""
    
    # Create combination dictionaries
    combinations = []
    for value_combo in value_combinations:
        # Build parameter dict for this combination
        params = {param_names[i]: value_combo[i] for i in range(len(param_names))}
        
        # Create descriptive name (with experiment_id prefix if provided)
        name_parts = [f"{param}_{value}" for param, value in params.items()]
        combo_name = f"{name_prefix}combo_" + "_".join(name_parts)
        
        # Create description
        desc_parts = [
            f"{param_categories[param]}.{param}={value}"
            for param, value in params.items()
        ]
        description = ", ".join(desc_parts)
        
        combination = {
            'params': params,
            'categories': param_categories,
            'name': combo_name,
            'description': description
        }
        combinations.append(combination)
    
    return combinations


# ════════════════════════════════════════════════════════════════════════════
# 2. CORE EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def run_single_combination(
    config: OmegaConf,
    save_dir: Path,
    train_fn: Callable,
    data_dir: Optional[Path] = None,
    cluster: bool = False,
    **kwargs
) -> Any:
    """
    Execute training for a single parameter combination.
    
    This is the core execution function called by both sequential and parallel
    sweep modes. It applies the parameter combination to the config and calls
    the user's training function.
    
    Args:
        config: Configuration with parameters already applied
        save_dir: Directory to save results
        train_fn: User's training function with signature:
                  train_fn(config, save_dir, data_dir, cluster, **kwargs) -> results
        data_dir: Path to data directory (optional)
        cluster: Whether running on cluster
        **kwargs: Additional arguments passed to train_fn
        
    Returns:
        Training results from train_fn
        
    Note:
        The config should already have the sweep parameters applied before
        calling this function. The config will be saved to save_dir/config.yaml
        after training completes.
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config BEFORE training starts
    # This ensures config.yaml exists even if training fails or is still running,
    # allowing evaluation functions to find it during training
    config_path = save_dir / "config.yaml"
    OmegaConf.save(config, config_path)
    
    # Call user's training function
    results = train_fn(
        config=config,
        save_dir=save_dir,
        data_dir=data_dir,
        cluster=cluster,
        **kwargs
    )
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# 3. SEQUENTIAL SWEEP EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def run_sequential_sweep(
    exp_dir: str,
    sweep_mode: str,
    train_fn: Callable,
    data_dir: Optional[str] = None,
    cluster: bool = False,
    experiment_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Run parameter sweep sequentially (one combination after another).
    
    This function loads the config and sweep files, generates all combinations,
    and runs them sequentially. Suitable for:
    - Local execution
    - Small number of combinations
    - When parallel resources are not available
    
    Args:
        exp_dir: Experiment directory containing config.yaml and sweep.yaml
        sweep_mode: "independent" or "combination"
        train_fn: User's training function
        data_dir: Path to data directory (optional)
        cluster: Whether running on cluster
        experiment_id: Experiment identifier for folder naming (optional, derived from exp_dir if not provided)
        **kwargs: Additional arguments passed to train_fn
        
    Raises:
        ValueError: If sweep_mode is not "independent" or "combination"
        FileNotFoundError: If config files are not found
        
    Directory Structure Created:
        Independent mode:
            exp_dir/
            └── sweeper/
                └── runs/
                    └── sweeps/
                        ├── sweep_param1/
                        │   ├── {exp_id}_sweep_param1_value1/
                        │   │   └── config.yaml
                        │   └── {exp_id}_sweep_param1_value2/
                        │       └── config.yaml
                        └── sweep_param2/
                            └── ...
        
        Combination mode:
            exp_dir/
            └── sweeper/
                └── runs/
                    └── combinations/
                        ├── {exp_id}_combo_param1_val1_param2_val1/
                        │   └── config.yaml
                        ├── {exp_id}_combo_param1_val1_param2_val2/
                        │   └── config.yaml
                        └── ...
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Load configuration files
    config, sweep_config = find_config_files(exp_dir)
    
    if sweep_config is None:
        logger.warning("No sweep configuration found. Nothing to sweep.")
        return
    
    # Extract experiment_id from exp_dir if not provided
    if experiment_id is None:
        experiment_id = Path(exp_dir).name
    
    # Generate combinations based on mode
    # Results are saved in sweeper/runs/ folder
    runs_dir = join(exp_dir, "sweeper", "runs")
    if sweep_mode == "independent":
        combinations = generate_independent_combinations(config, sweep_config, experiment_id=experiment_id)
        base_dir = join(runs_dir, "sweeps")
    elif sweep_mode == "combination":
        combinations = generate_all_combinations(config, sweep_config, experiment_id=experiment_id)
        base_dir = join(runs_dir, "combinations")
    else:
        raise ValueError(
            f"Unknown sweep_mode: {sweep_mode}. Use 'independent' or 'combination'."
        )
    
    logger.info(f"Starting {sweep_mode} sweep with {len(combinations)} combinations")
    
    # Run each combination sequentially
    for idx, combo in enumerate(combinations):
        logger.info(f"[{idx+1}/{len(combinations)}] Running: {combo['description']}")
        
        # Create modified config for this combination
        # IMPORTANT: Don't resolve interpolations yet! We need to update parameters first
        # so that any references (e.g., model.kwargs.param: ${experiment.param}) will
        # pick up the updated values when resolved later.
        config_copy = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        for param_name, param_value in combo['params'].items():
            category = combo['categories'][param_name]
            config_copy[category][param_name] = param_value
        
        # Populate sequence lengths from dataset metadata BEFORE resolving interpolations
        # This ensures ${data.S_seq_len}, ${data.X_seq_len} etc. have actual values
        # when interpolation references like ds_embed_S.num_variables: ${data.S_seq_len}
        # are resolved. Without this, these would resolve to null.
        if data_dir is not None:
            from causaliT.training.config_utils import populate_seq_lengths_from_dataset
            logger.info(f"Populating dataset metadata from: {data_dir}")
            config_copy = populate_seq_lengths_from_dataset(config_copy, data_dir)
        
        # Now resolve interpolations - references will use the updated parameter values
        config_copy = OmegaConf.create(OmegaConf.to_container(config_copy, resolve=True))
        
        # Determine save directory
        if sweep_mode == "independent":
            # For independent sweeps: sweeps/sweep_paramname/sweep_paramname_value/
            param_name = list(combo['params'].keys())[0]  # Only one param in independent
            sweep_param_dir = join(base_dir, f"sweep_{param_name}")
            save_dir = join(sweep_param_dir, combo['name'])
        else:
            # For combination sweeps: combinations/combo_...
            save_dir = join(base_dir, combo['name'])
        
        # Run training for this combination
        try:
            run_single_combination(
                config=config_copy,
                save_dir=Path(save_dir),
                train_fn=train_fn,
                data_dir=data_dir,
                cluster=cluster,
                **kwargs
            )
            logger.info(f"[{idx+1}/{len(combinations)}] Completed: {combo['description']}")
        except Exception as e:
            logger.error(f"[{idx+1}/{len(combinations)}] Failed: {combo['description']}")
            logger.error(f"Error: {str(e)}")
            # Continue with next combination even if one fails
            continue
    
    logger.info(f"Sequential {sweep_mode} sweep completed!")


# ════════════════════════════════════════════════════════════════════════════
# 4. PARALLEL SWEEP EXECUTION (SLURM)
# ════════════════════════════════════════════════════════════════════════════

def run_parallel_sweep(
    exp_dir: str,
    home_exp_dir: str,
    sweep_mode: str,
    train_fn_module: str,
    train_fn_name: str,
    experiment_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    scratch_path: Optional[str] = None,
    slurm_params: Optional[Dict] = None,
    cluster: bool = True,
    submit_jobs: bool = True,
    **kwargs
) -> None:
    """
    Run parameter sweep in parallel using SLURM job arrays.
    
    This function generates all combinations, saves metadata to a JSON file,
    creates a SLURM job array script, and submits it to the cluster. Each
    array job runs one parameter combination.
    
    Critical: For cluster usage with SCRATCH
    -----------------------------------------
    - Config files (config.yaml, sweep.yaml) are read from home_exp_dir (HOME)
    - Combinations metadata is saved in exp_dir (could be SCRATCH)
    - Each job runs in SCRATCH for fast I/O
    - The SLURM script handles copying data between HOME and SCRATCH
    
    Args:
        exp_dir: Experiment directory (could be SCRATCH path)
        home_exp_dir: Home experiment directory (for config files)
        sweep_mode: "independent" or "combination"
        train_fn_module: Module path to training function (e.g., "my_project.trainer")
        train_fn_name: Function name (e.g., "train_model")
        experiment_id: Experiment ID for SLURM job naming (optional, derived from home_exp_dir if not provided)
        data_dir: Path to data directory (optional)
        scratch_path: Scratch path if using scratch storage
        slurm_params: Dictionary of SLURM parameters:
            - max_concurrent_jobs: Maximum parallel jobs (default: 6)
            - walltime: Time limit (default: "5-00:00:00")
            - gpu_mem: GPU memory (default: "11g")
            - mem_per_cpu: CPU memory (default: "10g")
        cluster: Whether running on cluster
        submit_jobs: Whether to actually submit jobs (False for dry run)
        **kwargs: Additional arguments for training function
        
    Raises:
        ValueError: If sweep_mode is invalid
        FileNotFoundError: If config files not found
        
    Directory Structure Created:
        exp_dir/
        └── sweeper/
            ├── combinations_data.json    # Metadata for all combinations
            ├── run_sweep_array.sh        # Generated SLURM script
            ├── job_id.txt               # Submitted job ID
            ├── slurm_logs/              # SLURM output logs
            │   ├── sweep_<jobid>_0.out
            │   ├── sweep_<jobid>_0.err
            │   └── ...
            └── runs/
                └── combinations/        # Results (one dir per combination)
                    ├── combo_param1_val1_param2_val1/
                    └── ...
    
    Note:
        The training function is specified by module and name because it needs
        to be importable by the SLURM array jobs running on cluster nodes.
    """
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Default SLURM parameters
    # TODO define those from config
    if slurm_params is None:
        slurm_params = {}
    slurm_params.setdefault('max_concurrent_jobs', 6)
    slurm_params.setdefault('walltime', '4:00:00')
    slurm_params.setdefault('gpu_mem', '11g')
    slurm_params.setdefault('mem_per_cpu', '10g')
    
    # Load configuration files from home directory
    config_dir = home_exp_dir if scratch_path is not None else exp_dir
    config, sweep_config = find_config_files(config_dir)
    
    if sweep_config is None:
        logger.error("No sweep configuration found. Cannot run parallel sweep.")
        return
    
    # Use provided experiment_id or extract from home_exp_dir (not exp_dir which may be scratch path)
    if experiment_id is None:
        experiment_id = Path(home_exp_dir).name
    
    # Generate combinations (with experiment_id prefix for unique folder names)
    if sweep_mode == "independent":
        combinations = generate_independent_combinations(config, sweep_config, experiment_id=experiment_id)
    elif sweep_mode == "combination":
        combinations = generate_all_combinations(config, sweep_config, experiment_id=experiment_id)
    else:
        raise ValueError(
            f"Unknown sweep_mode: {sweep_mode}. Use 'independent' or 'combination'."
        )
    
    logger.info(f"Generated {len(combinations)} combinations for parallel execution")
    
    # Create combinations directory
    combinations_dir = join(exp_dir, "sweeper", "runs", "combinations")
    if not exists(combinations_dir):
        makedirs(combinations_dir)
    
    # Prepare combinations metadata
    # IMPORTANT: Save base_config WITHOUT resolving interpolations so that
    # the worker can apply parameter changes before resolution.
    # This ensures references like ${experiment.param} pick up updated values.
    combinations_data = {
        'base_config': OmegaConf.to_container(config, resolve=False),
        'combinations': combinations,
        'sweep_mode': sweep_mode,
        'data_dir': data_dir,
        'cluster': cluster,
        'train_fn_module': train_fn_module,
        'train_fn_name': train_fn_name,
        'additional_kwargs': kwargs
    }
    
    # Save combinations metadata
    combinations_file = join(exp_dir, "sweeper", "combinations_data.json")
    with open(combinations_file, 'w') as f:
        json.dump(combinations_data, f, indent=2)
    
    logger.info(f"Saved combinations metadata to: {combinations_file}")
    
    if submit_jobs:
        # Use provided experiment_id or extract from home_exp_dir (not exp_dir which may be scratch path)
        if experiment_id is None:
            experiment_id = Path(home_exp_dir).name
        
        # Generate SLURM job array script
        script_path = generate_slurm_job_array_script(
            exp_dir=exp_dir,
            home_exp_dir=home_exp_dir,
            experiment_id=experiment_id,
            combinations_file=combinations_file,
            slurm_params=slurm_params,
            scratch_path=scratch_path
        )
        
        logger.info(f"Generated SLURM script: {script_path}")
        
        # Submit job array to SLURM
        try:
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                cwd=exp_dir
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                job_id = result.stdout.strip().split()[-1]
                logger.info(f"Submitted job array with ID: {job_id}")
                
                # Save job ID for reference
                with open(join(exp_dir, "sweeper", "job_id.txt"), 'w') as f:
                    f.write(job_id)
                
                print("\n" + "="*60)
                print("Parallel Sweep Submitted Successfully!")
                print("="*60)
                print(f"Job ID: {job_id}")
                print(f"Total combinations: {len(combinations)}")
                print(f"Max concurrent jobs: {slurm_params['max_concurrent_jobs']}")
                print(f"Walltime per job: {slurm_params['walltime']}")
                print(f"\nMonitor progress with: squeue -u $USER")
                print(f"Check logs in: {exp_dir}/sweeper/slurm_logs/")
                print("="*60 + "\n")
            else:
                logger.error(f"Failed to submit job: {result.stderr}")
                print(f"Error submitting job: {result.stderr}")
        
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            print(f"Error submitting job: {str(e)}")
    
    else:
        print(f"\nDry run completed. Generated {len(combinations)} combinations.")
        print(f"Script ready: {join(exp_dir, 'sweeper', 'run_sweep_array.sh')}")
        print(f"To submit: sbatch {join(exp_dir, 'sweeper', 'run_sweep_array.sh')}")


# ════════════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def find_config_files(exp_dir: str) -> Tuple[Optional[OmegaConf], Optional[OmegaConf]]:
    """
    Find and load config file and sweeper/sweep.yaml from experiment directory.
    
    Supports config files with names starting with "config" (e.g., config.yaml, 
    config_proT_ishigami_v5_2.yaml).
    
    Structure:
        exp_dir/
        ├── config*.yaml         # Main configuration (any file starting with "config")
        └── sweeper/
            └── sweep.yaml       # Sweep definition
    
    Args:
        exp_dir: Experiment directory to search
        
    Returns:
        Tuple of (config, sweep_config) as OmegaConf objects
        
    Raises:
        FileNotFoundError: If no config file or sweeper/ directory is not found
        Warning: If sweeper/sweep.yaml is not found (returned as None)
    """
    # Load main config from root - find any file starting with "config"
    import glob
    config_pattern = join(exp_dir, "config*.yaml")
    config_files = glob.glob(config_pattern)
    
    if not config_files:
        raise FileNotFoundError(
            f"Config file not found in: {exp_dir}\n"
            "Expected a file starting with 'config' (e.g., config.yaml, config_proT_v5.yaml) in experiment root directory."
        )
    
    # If multiple config files found, use the first one (sorted alphabetically)
    config_files.sort()
    config_path = config_files[0]
    
    if len(config_files) > 1:
        warnings.warn(
            f"Multiple config files found in {exp_dir}: {[Path(f).name for f in config_files]}\n"
            f"Using: {Path(config_path).name}"
        )
    
    config = OmegaConf.load(config_path)
    
    # Load sweep config from sweeper subdirectory
    sweeper_dir = join(exp_dir, "sweeper")
    sweep_path = join(sweeper_dir, "sweep.yaml")
    
    if not exists(sweeper_dir):
        warnings.warn(
            f"Sweeper directory not found: {sweeper_dir}\n"
            f"Expected structure: {exp_dir}/sweeper/sweep.yaml"
        )
        return config, None
    
    if not exists(sweep_path):
        warnings.warn(
            f"Sweep configuration not found: {sweep_path}\n"
            "Expected sweep.yaml in sweeper subdirectory."
        )
        return config, None
    
    sweep_config = OmegaConf.load(sweep_path)
    
    return config, sweep_config


def generate_slurm_job_array_script(
    exp_dir: str,
    home_exp_dir: str,
    experiment_id: str,
    combinations_file: str,
    slurm_params: Dict,
    scratch_path: Optional[str] = None
) -> str:
    """
    Generate SLURM job array script for parallel sweep execution.
    
    Creates a bash script that:
    1. Sets up SLURM parameters (array, time, memory, GPU)
    2. Loads required modules
    3. Activates Python environment
    4. Runs the sweep worker script for this array task
    
    Args:
        exp_dir: Experiment directory (could be SCRATCH)
        home_exp_dir: Home experiment directory
        experiment_id: Experiment ID for naming
        combinations_file: Path to combinations metadata JSON
        slurm_params: SLURM parameters dictionary
        scratch_path: Scratch path if using scratch storage
        
    Returns:
        Path to generated SLURM script
        
    Note:
        The script needs to be customized with your cluster's module names
        and Python environment path. See TODO markers in generated script.
    """
    # Load combinations to determine array size
    with open(combinations_file, 'r') as f:
        combinations_data = json.load(f)
    
    total_jobs = len(combinations_data['combinations'])
    max_concurrent = slurm_params['max_concurrent_jobs']
    
    # Create slurm_logs directory
    slurm_logs_dir = join(exp_dir, "sweeper", "slurm_logs")
    if not exists(slurm_logs_dir):
        makedirs(slurm_logs_dir)
    
    # Generate script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name=sweep_{experiment_id}
#SBATCH --output={exp_dir}/sweeper/slurm_logs/sweep_%A_%a.out
#SBATCH --error={exp_dir}/sweeper/slurm_logs/sweep_%A_%a.err
#SBATCH --array=0-{total_jobs-1}%{max_concurrent}
#SBATCH --ntasks=1
#SBATCH --time={slurm_params['walltime']}
#SBATCH --gpus=1
#SBATCH --mem-per-cpu={slurm_params['mem_per_cpu']}
#SBATCH --gres=gpumem:{slurm_params['gpu_mem']}

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# ───────────────────────────────────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────────────────────────────────
HOME_EXP="{home_exp_dir}"
EXP_DIR="{exp_dir}"

echo "[$(date)] Home exp folder : $HOME_EXP"
echo "[$(date)] Experiment dir  : $EXP_DIR"

# ───────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ───────────────────────────────────────────────────────────────────────────
# TODO: Customize these module loads for your cluster
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# TODO: Update this path to your virtual environment
VENV_PATH="$HOME/myenv"
source "$VENV_PATH/bin/activate"

if [[ -z "${{VIRTUAL_ENV:-}}" ]]; then
    echo "[$(date)] Failed to activate Python environment!" >&2
    exit 1
fi
echo "[$(date)] Python env: $VIRTUAL_ENV"

# ───────────────────────────────────────────────────────────────────────────
# RUN COMBINATION
# ───────────────────────────────────────────────────────────────────────────
cd "$EXP_DIR"

echo "[$(date)] Running combination $SLURM_ARRAY_TASK_ID..."

# Run the sweep worker for this array task
python -m causaliT.euler_sweep.euler_sweep.sweep_worker \\
    --exp_dir "$EXP_DIR" \\
    --combinations_file "{combinations_file}" \\
    --task_id $SLURM_ARRAY_TASK_ID

# ───────────────────────────────────────────────────────────────────────────
# WRAP-UP
# ───────────────────────────────────────────────────────────────────────────
deactivate
echo "[$(date)] Python environment deactivated"
echo "[$(date)] Job finished – results in $EXP_DIR/sweeper/runs/combinations/"
"""
    
    # Write script file
    script_path = join(exp_dir, "sweeper", "run_sweep_array.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path
