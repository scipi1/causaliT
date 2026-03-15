"""
Command-line interface for CausaliT evaluation functions.

This module provides a CLI for running evaluations and updating the experiment manifest.

Usage:
    # Run evaluations on experiments
    python -m notebooks.eval_funs.eval_fun_cli evaluate -e path/to/exp1 -e path/to/exp2
    python -m notebooks.eval_funs.eval_fun_cli evaluate -f path/to/experiments/euler
    python -m notebooks.eval_funs.eval_fun_cli evaluate -e path/to/exp --functions eval_train_metrics
    
    # Update manifest only (for experiments that already have evaluations)
    python -m notebooks.eval_funs.eval_fun_cli manifest -e path/to/exp1 -e path/to/exp2
    python -m notebooks.eval_funs.eval_fun_cli manifest -f path/to/experiments/euler
    
    # Combined: run evaluations then update manifest
    python -m notebooks.eval_funs.eval_fun_cli evaluate -e path/to/exp --update-manifest

Available evaluation functions:
    - eval_train_metrics: Training curves and loss analysis
    - eval_attention_scores: DAG recovery metrics  
    - eval_embed: Embedding evolution analysis
    - eval_interventions: Causal intervention tests
    - eval_embedding_dag_correlation: Embedding-DAG correlation
    - eval_dyconex_predictions: Dyconex-specific prediction evaluation
    - eval_ans: Attention Necessity Score (ANS) for sweep experiments
"""

import sys
import traceback
from os import listdir
from os.path import join, exists, isdir
from typing import List, Tuple

import click

# Import from sibling modules
from .eval_utils import root_path
from .eval_funs_wraps import run_all_evaluations, run_evaluations_from_config
from .update_manifest import batch_update_manifest, MANIFEST_PATH


# =============================================================================
# Available evaluation functions
# =============================================================================

AVAILABLE_FUNCTIONS = [
    "eval_train_metrics",
    "eval_attention_scores",
    "eval_embed",
    "eval_interventions",
    "eval_embedding_dag_correlation",
    "eval_dyconex_predictions",
    "eval_ans",  # ANS evaluation for sweep experiments
    "fix_kfold_summary",
    "enrich_kfold_summary",
]


def discover_experiments(experiments: Tuple[str], folders: Tuple[str]) -> List[str]:
    """Discover all experiment paths from explicit paths and folders."""
    all_experiments = list(experiments)
    
    for folder in folders:
        if exists(folder) and isdir(folder):
            for subdir in listdir(folder):
                subdir_path = join(folder, subdir)
                if isdir(subdir_path):
                    contents = listdir(subdir_path)
                    has_config = any(f.startswith("config") and f.endswith(".yaml") for f in contents)
                    has_kfold = any(f.startswith("k_") for f in contents)
                    if has_config or has_kfold:
                        all_experiments.append(subdir_path)
        else:
            click.echo(f"Warning: Folder not found: {folder}")
    
    return all_experiments


@click.group()
def cli():
    """CausaliT Evaluation CLI - Run evaluations and update experiment manifest."""
    pass


@cli.command()
@click.option("-e", "--experiment", "experiments", multiple=True, help="Experiment path(s) to evaluate")
@click.option("-f", "--folder", "folders", multiple=True, help="Folder(s) containing experiments to scan")
@click.option("--functions", multiple=True, type=click.Choice(AVAILABLE_FUNCTIONS), help="Specific functions to run (default: all)")
@click.option("--no-show", is_flag=True, help="Don't display plots (save to files only)")
@click.option("--update-manifest", is_flag=True, help="Update manifest after evaluations")
def evaluate(experiments, folders, functions, no_show, update_manifest):
    """Run evaluation functions on experiments."""
    all_experiments = discover_experiments(experiments, folders)
    
    if not all_experiments:
        click.echo("No experiments found. Use -e or -f to specify paths.")
        return
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Running evaluations for {len(all_experiments)} experiment(s)")
    click.echo(f"Functions: {list(functions) if functions else 'ALL'}")
    click.echo('='*60)
    
    success, failed = 0, 0
    for exp in all_experiments:
        click.echo(f"\n--- Evaluating: {exp} ---")
        try:
            if functions:
                run_evaluations_from_config(exp, show_plots=not no_show, functions=list(functions))
            else:
                run_all_evaluations(exp, show_plots=not no_show)
            success += 1
        except Exception as e:
            click.echo(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Summary: {success} success, {failed} failed")
    
    if update_manifest:
        click.echo("\n--- Updating manifest ---")
        batch_update_manifest(experiments=all_experiments)
        click.echo(f"Manifest saved to: {MANIFEST_PATH}")


@cli.command()
@click.option("-e", "--experiment", "experiments", multiple=True, help="Experiment path(s)")
@click.option("-f", "--folder", "folders", multiple=True, help="Folder(s) containing experiments")
def manifest(experiments, folders):
    """Update the experiments manifest CSV (without running evaluations)."""
    all_experiments = discover_experiments(experiments, folders)
    
    if not all_experiments:
        click.echo("No experiments found. Use -e or -f to specify paths.")
        return
    
    click.echo(f"\nUpdating manifest for {len(all_experiments)} experiment(s)")
    result = batch_update_manifest(experiments=all_experiments)
    click.echo(f"\nManifest saved to: {MANIFEST_PATH}")
    click.echo(f"Total experiments: {len(result)}")


if __name__ == "__main__":
    cli()
