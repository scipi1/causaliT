"""
Boilerplate Trainer CLI - Simple Sweep Example

This is a minimal example showing how to integrate the sweep framework with your trainer.
This example focuses on LOCAL SEQUENTIAL sweeps only (no cluster, no parallel).
"""

import sys
from os.path import abspath, dirname, join
from pathlib import Path

import click

# Add parent directories to path to import sweep framework
example_dir = dirname(abspath(__file__))
euler_sweep_dir = dirname(dirname(example_dir))
sys.path.insert(0, euler_sweep_dir)

# Import sweep framework
from euler_sweep.sweeper import run_sequential_sweep

# Import our trainer
from trainer import simple_trainer


@click.command()
@click.option(
    "--exp_id",
    required=True,
    help="Experiment ID (folder name in base directory)"
)
@click.option(
    "--sweep_mode",
    required=True,
    type=click.Choice(['independent', 'combination']),
    help="Sweep mode: independent or combination"
)
@click.option(
    "--base_dir",
    type=click.Choice(['experiments', 'examples']),
    default='experiments',
    help="Base directory to look for experiment (default: experiments)"
)
def sweep(exp_id, sweep_mode, base_dir):
    """
    Run parameter sweeps with the boilerplate trainer.
    
    This is a simple example focusing on LOCAL SEQUENTIAL execution.
    
    Examples:
        # Run the boiler_trainer example directly (from examples/ folder)
        python cli.py sweep --exp_id boiler_trainer --sweep_mode independent --base_dir examples
        
        # Run the boiler_trainer example with combination mode
        python cli.py sweep --exp_id boiler_trainer --sweep_mode combination --base_dir examples
        
        # Create your own experiment in experiments/ folder
        python cli.py sweep --exp_id my_experiment --sweep_mode independent
    """
    print(f"🚀 Starting {sweep_mode} sweep for experiment: {exp_id}")
    print(f"📂 Base directory: {base_dir}")
    print("=" * 60)
    
    # Setup paths using the selected base directory
    base_directory = join(euler_sweep_dir, base_dir)
    exp_dir = join(base_directory, exp_id)
    data_dir = join(euler_sweep_dir, "data")  # Not used for synthetic data
    
    # Check if experiment exists
    if not Path(exp_dir).exists():
        print(f"❌ Error: Experiment directory not found: {exp_dir}")
        print(f"\nTo create it:")
        print(f"  mkdir -p {exp_dir}/sweeper")
        if base_dir == "experiments":
            print(f"  cp examples/boiler_trainer/config.yaml {exp_dir}/")
            print(f"  cp examples/boiler_trainer/sweeper/sweep.yaml {exp_dir}/sweeper/")
        else:
            print(f"  # Or use existing example with: --base_dir examples --exp_id boiler_trainer")
        sys.exit(1)
    
    print(f"📁 Experiment directory: {exp_dir}")
    print(f"📝 Config: {exp_dir}/config.yaml")
    print(f"🔄 Sweep config: {exp_dir}/sweeper/sweep.yaml")
    print("=" * 60)
    print()
    
    # Run the sequential sweep
    run_sequential_sweep(
        exp_dir=exp_dir,
        sweep_mode=sweep_mode,
        train_fn=simple_trainer,
        data_dir=data_dir,
        cluster=False  # Local execution
    )
    
    print()
    print("=" * 60)
    print("✅ Sweep completed!")
    print("=" * 60)
    
    if sweep_mode == "independent":
        print(f"📊 Results location: {exp_dir}/sweeper/runs/sweeps/")
    else:
        print(f"📊 Results location: {exp_dir}/sweeper/runs/combinations/")
    
    print("\nEach combination directory contains:")
    print("  - config.yaml (parameters used)")
    print("  - results.txt (training output)")
    print("=" * 60)


if __name__ == "__main__":
    sweep()
