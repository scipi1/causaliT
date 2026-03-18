"""
Sweep Worker Script for Parallel Execution

This script is called by SLURM array jobs to execute individual parameter
combinations in parallel. It loads the combination metadata, imports the
training function, and runs the training for one specific combination.

Usage (called by SLURM job array):
    python -m causaliT.euler_sweep.euler_sweep.sweep_worker \\
        --exp_dir /path/to/experiment \\
        --combinations_file /path/to/combinations_data.json \\
        --task_id $SLURM_ARRAY_TASK_ID
"""

import json
import sys
import importlib
from pathlib import Path

import click
from omegaconf import OmegaConf

# Set up path for causaliT imports
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

# Import sweep functions
from causaliT.euler_sweep.euler_sweep.sweeper import run_single_combination
from causaliT.training.config_utils import populate_seq_lengths_from_dataset


@click.command()
@click.option(
    "--exp_dir",
    required=True,
    help="Experiment directory"
)
@click.option(
    "--combinations_file",
    required=True,
    help="Path to combinations metadata JSON file"
)
@click.option(
    "--task_id",
    type=int,
    required=True,
    help="SLURM array task ID (combination index)"
)
def main(exp_dir, combinations_file, task_id):
    """Execute training for a single parameter combination."""
    
    print(f"[Worker] Starting task {task_id}")
    print(f"[Worker] Experiment directory: {exp_dir}")
    print(f"[Worker] Combinations file: {combinations_file}")
    
    # Load combinations metadata
    try:
        with open(combinations_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"[Worker ERROR] Failed to load combinations file: {e}")
        sys.exit(1)
    
    # Get the specific combination for this task
    if task_id >= len(metadata['combinations']):
        print(f"[Worker ERROR] Task ID {task_id} out of range (max: {len(metadata['combinations'])-1})")
        sys.exit(1)
    
    combination = metadata['combinations'][task_id]
    print(f"[Worker] Running combination: {combination['description']}")
    
    # Load base config (unresolved - contains interpolation references like ${experiment.param})
    base_config = OmegaConf.create(metadata['base_config'])
    
    # Apply parameter changes for this combination BEFORE resolving interpolations
    # This ensures that references like model.kwargs.param: ${experiment.param}
    # will pick up the updated sweep values when resolved later.
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=False))
    for param_name, param_value in combination['params'].items():
        category = combination['categories'][param_name]
        config[category][param_name] = param_value
    
    # Populate sequence lengths from dataset metadata BEFORE resolving interpolations
    # This ensures ${data.S_seq_len}, ${data.X_seq_len} etc. have actual values
    # when interpolation references like ds_embed_S.num_variables: ${data.S_seq_len}
    # are resolved. Without this, these would resolve to null.
    data_dir = metadata.get('data_dir')
    if data_dir is not None:
        print(f"[Worker] Populating dataset metadata from: {data_dir}")
        config = populate_seq_lengths_from_dataset(config, data_dir)
    
    # Now resolve interpolations - references will use the updated parameter values
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    
    # Import training function dynamically
    try:
        train_fn_module = metadata['train_fn_module']
        train_fn_name = metadata['train_fn_name']
        
        print(f"[Worker] Importing training function: {train_fn_module}.{train_fn_name}")
        
        module = importlib.import_module(train_fn_module)
        train_fn = getattr(module, train_fn_name)
    except Exception as e:
        print(f"[Worker ERROR] Failed to import training function: {e}")
        sys.exit(1)
    
    # Determine save directory (using sweeper/runs/combinations/)
    save_dir = Path(exp_dir) / "sweeper" / "runs" / "combinations" / combination['name']
    
    # Get additional parameters
    data_dir = metadata.get('data_dir')
    cluster = metadata.get('cluster', True)
    additional_kwargs = metadata.get('additional_kwargs', {})
    
    # Run training for this combination
    try:
        print(f"[Worker] Starting training...")
        run_single_combination(
            config=config,
            save_dir=save_dir,
            train_fn=train_fn,
            data_dir=data_dir,
            cluster=cluster,
            **additional_kwargs
        )
        print(f"[Worker] Training completed successfully!")
    except Exception as e:
        print(f"[Worker ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
