"""
Evaluation wrapper functions for CausaliT experiments.

This module provides wrapper functions that run multiple evaluations:
- run_evaluations_from_config: Run config-specified evaluations
- run_all_evaluations: Run all standard evaluations
"""

import traceback
from typing import List

# Import evaluation functions from sibling modules
from .eval_training import eval_train_metrics
from .eval_attention import eval_attention_scores
from .eval_embeddings import eval_embed, eval_embedding_dag_correlation
from .eval_interventions import eval_interventions
from .update_manifest import fix_kfold_summary, enrich_kfold_summary


# =============================================================================
# Post-Training Evaluation Wrapper
# =============================================================================

def run_evaluations_from_config(
    experiment: str,
    datadir_path: str = None,
    show_plots: bool = False,
    functions: List[str] = None,
) -> dict:
    """
    Run evaluation functions specified in the config file.
    
    This dispatcher function allows choosing which evaluation functions to run
    based on the config file's evaluation.functions list.
    
    Args:
        experiment: Path to the experiment folder
        datadir_path: Path to data directory. If None, uses default.
        show_plots: If True, display plots. If False, only save to files.
        functions: List of function names to run. Available functions:
            - "eval_train_metrics": Training curves and loss analysis
            - "eval_attention_scores": DAG recovery metrics
            - "eval_embed": Embedding evolution analysis
            - "eval_interventions": Causal intervention tests
            - "eval_embedding_dag_correlation": Embedding-DAG correlation
            - "eval_dyconex_predictions": Dyconex-specific prediction evaluation
            
    Returns:
        dict: Summary of evaluation results
        
    Example:
        >>> results = run_evaluations_from_config(
        ...     experiment="../experiments/stage/dyconex_exp",
        ...     functions=["eval_train_metrics", "eval_dyconex_predictions"]
        ... )
    """
    import traceback
    
    print(f"\n{'='*60}")
    print(f"Running config-specified evaluations")
    print(f"Experiment: {experiment}")
    print(f"Functions: {functions}")
    print('='*60)
    
    results = {
        "experiment": experiment,
        "evaluations": {},
    }
    
    if functions is None:
        print("No functions specified, running default evaluations...")
        return run_all_evaluations(experiment, datadir_path, show_plots)
    
    # Function registry - maps function names to callables
    # Import dyconex functions only when needed
    FUNCTION_REGISTRY = {
        "eval_train_metrics": lambda exp: eval_train_metrics(exp, show_plots=show_plots),
        "eval_attention_scores": lambda exp: eval_attention_scores(exp, show_plots=show_plots),
        "eval_embed": lambda exp: eval_embed(exp, show_plots=show_plots),
        "eval_interventions": lambda exp: eval_interventions(exp, show_plots=show_plots),
        "eval_embedding_dag_correlation": lambda exp: eval_embedding_dag_correlation(exp, show_plots=show_plots),
        "fix_kfold_summary": lambda exp: fix_kfold_summary(exp),
        "enrich_kfold_summary": lambda exp: enrich_kfold_summary(exp),
    }
    
    # Dyconex-specific functions (lazy import)
    def _get_dyconex_predictions(exp):
        from .eval_dyconex import eval_dyconex_predictions
        return eval_dyconex_predictions(exp, datadir_path=datadir_path, show_plots=show_plots)
    
    FUNCTION_REGISTRY["eval_dyconex_predictions"] = _get_dyconex_predictions
    
    # Run specified functions
    for idx, func_name in enumerate(functions, start=1):
        print(f"\n--- Step {idx}: Running {func_name} ---")
        
        if func_name not in FUNCTION_REGISTRY:
            print(f"  ✗ Unknown function: {func_name}")
            results["evaluations"][func_name] = f"failed: Unknown function"
            continue
        
        try:
            FUNCTION_REGISTRY[func_name](experiment)
            results["evaluations"][func_name] = "success"
            print(f"  ✓ {func_name} completed successfully")
        except Exception as e:
            print(f"  ✗ {func_name} failed: {e}")
            traceback.print_exc()
            results["evaluations"][func_name] = f"failed: {e}"
    
    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print('='*60)
    success_count = sum(1 for v in results["evaluations"].values() if v == "success")
    total_count = len(results["evaluations"])
    print(f"  Completed: {success_count}/{total_count}")
    for name, status in results["evaluations"].items():
        status_icon = "✓" if status == "success" else "✗"
        print(f"    {status_icon} {name}: {status}")
    
    return results


def run_all_evaluations(
    experiment: str,
    datadir_path: str = None,
    show_plots: bool = False,
) -> dict:
    """
    Run all evaluation functions on an experiment after training.
    
    This function is called automatically by trainer.py after training completes
    on the cluster. It runs all standard evaluations with error handling to ensure
    that failures in one evaluation don't prevent others from running.
    
    Args:
        experiment: Path to the experiment folder containing k_* subdirectories
        datadir_path: Path to data directory. If None, uses default "../data/" relative to project root
        show_plots: If True, display plots. If False (default), only save to files.
                   Should be False for cluster execution (headless environment).
        
    Returns:
        dict: Summary of evaluation results with keys:
            - experiment: Path to experiment
            - evaluations: Dict mapping function names to "success" or error message
            
    Example:
        >>> from notebooks.eval_funs import run_all_evaluations
        >>> results = run_all_evaluations("../experiments/single/euler/my_experiment")
        >>> print(results)
        {'experiment': '...', 'evaluations': {'eval_train_metrics': 'success', ...}}
    """
    import traceback
    
    print(f"\n{'='*60}")
    print(f"Running post-training evaluations")
    print(f"Experiment: {experiment}")
    print('='*60)
    
    results = {
        "experiment": experiment,
        "evaluations": {},
    }
    
    # Step 1: Fix and enrich kfold_summary.json
    print(f"\n--- Step 1: Fixing kfold_summary.json ---")
    try:
        fix_kfold_summary(experiment)
        results["evaluations"]["fix_kfold_summary"] = "success"
    except Exception as e:
        print(f"Warning: fix_kfold_summary failed: {e}")
        results["evaluations"]["fix_kfold_summary"] = f"failed: {e}"
    
    print(f"\n--- Step 2: Enriching kfold_summary.json ---")
    try:
        enrich_kfold_summary(experiment)
        results["evaluations"]["enrich_kfold_summary"] = "success"
    except Exception as e:
        print(f"Warning: enrich_kfold_summary failed: {e}")
        results["evaluations"]["enrich_kfold_summary"] = f"failed: {e}"
    
    # Step 3: Run all evaluation functions (hard-coded list)
    # Order matters: some evaluations depend on others
    eval_functions = [
        ("eval_train_metrics", lambda exp: eval_train_metrics(exp, show_plots=show_plots)),
        ("eval_attention_scores", lambda exp: eval_attention_scores(exp, show_plots=show_plots)),
        ("eval_embed", lambda exp: eval_embed(exp, show_plots=show_plots)),
        ("eval_interventions", lambda exp: eval_interventions(exp, show_plots=show_plots)),
        # eval_embedding_dag_correlation requires eval_embed and eval_attention_scores to run first
        ("eval_embedding_dag_correlation", lambda exp: eval_embedding_dag_correlation(exp, show_plots=show_plots)),
    ]
    
    for idx, (name, func) in enumerate(eval_functions, start=3):
        print(f"\n--- Step {idx}: Running {name} ---")
        try:
            func(experiment)
            results["evaluations"][name] = "success"
            print(f"  ✓ {name} completed successfully")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            traceback.print_exc()
            results["evaluations"][name] = f"failed: {e}"
    
    # Summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print('='*60)
    success_count = sum(1 for v in results["evaluations"].values() if v == "success")
    total_count = len(results["evaluations"])
    print(f"  Completed: {success_count}/{total_count}")
    for name, status in results["evaluations"].items():
        status_icon = "✓" if status == "success" else "✗"
        print(f"    {status_icon} {name}: {status}")
    
    return results