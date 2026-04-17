"""
Score Sparsity Evaluation: LASSO-path analysis for lambda_score selection.

This module analyses the outputs of a ``calibrated_sweep`` run where
``lambda_cross_score_sparse`` (and optionally ``lambda_self_score_sparse``)
were swept over a grid.  For each lambda value it computes:

- Mean HSIC (cross and self) — the causal signal we want to minimise
- Mean attention score density — proxy for variable importance
- Per-variable attention score norms — the "variable importance" axis

The resulting LASSO-path plot shows HSIC vs variable importance for each
lambda value, letting you pick the lambda that:
  * Minimises HSIC (maximises causal structure recovery)
  * Maintains sufficient variable importance (non-degenerate attention)

Recommended selection rule
---------------------------
Select the SMALLEST lambda_cross_score such that the HSIC does not increase
significantly compared to lambda=0.  This is the sparsest model that still
captures the full causal signal — analogous to the "1-SE rule" in LASSO.

Usage::

    from causaliT.evaluation.eval_score_sparsity import (
        collect_score_sparsity_results,
        plot_score_sparsity_path,
        select_lambda_score,
    )

    results = collect_score_sparsity_results(sweep_dir)
    fig = plot_score_sparsity_path(results)
    lambda_star = select_lambda_score(results)
    print(f"Selected lambda_cross_score = {lambda_star:.4f}")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_score_sparsity_results(sweep_dir: str) -> pd.DataFrame:
    """
    Collect per-run metrics from a score-sparsity sweep.

    Walks ``sweep_dir/sweeper/runs/`` and reads ``calibration_metrics.json``
    (if present) and the standard Lightning ``csv_logger`` metrics CSV for
    each combination run.

    Args:
        sweep_dir: Root directory of the sweep (contains ``sweeper/``).

    Returns:
        DataFrame with one row per lambda value:
        - ``lambda_cross_score``
        - ``lambda_self_score``
        - ``hsic_cross``   — mean HSIC cross over the run
        - ``hsic_self``    — mean HSIC self over the run
        - ``val_mae``      — final validation MAE
        - ``att_density``  — mean attention score density (fraction non-zero)
        - ``run_dir``      — path to the run directory
    """
    sweep_dir = Path(sweep_dir)
    runs_root = sweep_dir / "sweeper" / "runs"

    rows = []
    for run_dir in sorted(runs_root.rglob("config.yaml")):
        run_path = run_dir.parent

        # Load config to get lambda values
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(run_dir)
            lam_cross = float(cfg.get("training", {}).get("lambda_cross_score_sparse", 0.0))
            lam_self = float(cfg.get("training", {}).get("lambda_self_score_sparse", 0.0))
        except Exception:
            lam_cross = float("nan")
            lam_self = float("nan")

        # Find Lightning CSV logger metrics
        hsic_cross = float("nan")
        hsic_self = float("nan")
        val_mae = float("nan")
        att_density = float("nan")

        for fold_dir in sorted(run_path.glob("k_*")):
            csv_dir = fold_dir / "logs" / "csv" / "version_0"
            metrics_csv = csv_dir / "metrics.csv"
            if metrics_csv.exists():
                try:
                    df = pd.read_csv(metrics_csv)
                    if "train_hsic_cross" in df.columns:
                        hsic_cross = float(df["train_hsic_cross"].dropna().mean())
                    if "train_hsic_self" in df.columns:
                        hsic_self = float(df["train_hsic_self"].dropna().mean())
                    if "val_mae" in df.columns:
                        val_mae = float(df["val_mae"].dropna().iloc[-1])
                    if "train_att_density" in df.columns:
                        att_density = float(df["train_att_density"].dropna().mean())
                except Exception as e:
                    logger.warning(f"Could not read metrics from {metrics_csv}: {e}")
            break  # use only k_0 for now (extend to average over folds if needed)

        rows.append(
            {
                "lambda_cross_score": lam_cross,
                "lambda_self_score": lam_self,
                "hsic_cross": hsic_cross,
                "hsic_self": hsic_self,
                "val_mae": val_mae,
                "att_density": att_density,
                "run_dir": str(run_path),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No sweep runs found under {runs_root}. "
            "Run a score-sparsity sweep first."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("lambda_cross_score").reset_index(drop=True)
    return df


def collect_attention_score_norms(run_dir: str) -> Optional[np.ndarray]:
    """
    Extract per-variable attention score norms from a single run directory.

    Looks for ``attention_scores.npy`` saved by ``GradientJacobianLogger``
    or similar evaluation outputs.

    Args:
        run_dir: Path to a single run's output directory.

    Returns:
        Array of shape ``(n_sources,)`` with per-source attention norms,
        or ``None`` if not found.
    """
    run_path = Path(run_dir)
    for candidate in [
        run_path / "k_0" / "attention_scores.npy",
        run_path / "k_0" / "att_scores.npy",
    ]:
        if candidate.exists():
            return np.load(str(candidate))
    return None


# =============================================================================
# LAMBDA SELECTION
# =============================================================================

def select_lambda_score(
    results: pd.DataFrame,
    metric: str = "hsic_cross",
    rule: str = "1se",
    tolerance: float = 0.05,
) -> float:
    """
    Select the optimal lambda_cross_score from sweep results.

    Rules:
    - ``"min_hsic"`` : select the lambda that achieves the lowest HSIC.
      This is the most aggressive sparsification.
    - ``"1se"``      : select the SMALLEST lambda such that HSIC is within
      ``tolerance`` (fractional) of the minimum HSIC.  Analogous to the
      "1-standard-error rule" in LASSO — prefers simpler (sparser) models.

    Args:
        results:   DataFrame from ``collect_score_sparsity_results``.
        metric:    Column to optimise (default: ``"hsic_cross"``).
        rule:      Selection rule (``"min_hsic"`` or ``"1se"``).
        tolerance: Fractional tolerance for the ``"1se"`` rule.

    Returns:
        The selected ``lambda_cross_score`` value.
    """
    valid = results.dropna(subset=[metric]).copy()
    if valid.empty:
        raise ValueError(f"No valid rows for metric '{metric}' in results.")

    valid = valid.sort_values("lambda_cross_score")
    hsic_values = valid[metric].values
    lambdas = valid["lambda_cross_score"].values

    min_hsic = hsic_values.min()

    if rule == "min_hsic":
        idx = int(np.argmin(hsic_values))
        return float(lambdas[idx])

    elif rule == "1se":
        # Find smallest lambda where HSIC <= min_hsic * (1 + tolerance)
        threshold = min_hsic * (1.0 + tolerance)
        candidates = lambdas[hsic_values <= threshold]
        if len(candidates) == 0:
            # Fallback: return lambda that achieves minimum
            idx = int(np.argmin(hsic_values))
            return float(lambdas[idx])
        return float(candidates[0])  # sorted ascending, so this is the smallest

    else:
        raise ValueError(f"Unknown rule '{rule}'. Use 'min_hsic' or '1se'.")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_score_sparsity_path(
    results: pd.DataFrame,
    save_path: Optional[str] = None,
    selected_lambda: Optional[float] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot the LASSO-path style figure: HSIC vs lambda_score.

    Shows two panels:
    - Left:  HSIC (cross and self) vs log(lambda_score)
    - Right: Validation MAE vs log(lambda_score)

    A vertical dashed line is drawn at ``selected_lambda`` if provided.

    Args:
        results:         DataFrame from ``collect_score_sparsity_results``.
        save_path:       If provided, save figure to this path.
        selected_lambda: Lambda value to highlight.

    Returns:
        Matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Score Sparsity Path (LASSO-style)", fontsize=14, fontweight="bold")

    lambdas = results["lambda_cross_score"].values
    log_lambdas = np.log10(np.where(lambdas == 0, 1e-6, lambdas))

    x_label = "log10(lambda_cross_score)"

    # ── Left: HSIC ───────────────────────────────────────────────────────────
    ax = axes[0]
    if "hsic_cross" in results.columns and results["hsic_cross"].notna().any():
        ax.plot(log_lambdas, results["hsic_cross"], "o-", label="HSIC cross (S->X)", color="steelblue")
    if "hsic_self" in results.columns and results["hsic_self"].notna().any():
        ax.plot(log_lambdas, results["hsic_self"], "s--", label="HSIC self (X->X)", color="coral")
    if selected_lambda is not None:
        ax.axvline(np.log10(max(selected_lambda, 1e-6)), color="green", linestyle=":", lw=2,
                   label=f"Selected lambda={selected_lambda:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("HSIC")
    ax.set_title("HSIC vs Score Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: Validation MAE ─────────────────────────────────────────────────
    ax = axes[1]
    if "val_mae" in results.columns and results["val_mae"].notna().any():
        ax.plot(log_lambdas, results["val_mae"], "^-", label="val MAE", color="darkorange")
    if selected_lambda is not None:
        ax.axvline(np.log10(max(selected_lambda, 1e-6)), color="green", linestyle=":", lw=2,
                   label=f"Selected lambda={selected_lambda:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation MAE")
    ax.set_title("Validation MAE vs Score Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Score sparsity path plot saved to {save_path}")

    return fig


def plot_variable_importance_path(
    sweep_dir: str,
    results: pd.DataFrame,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot per-variable attention norms vs lambda_score (LASSO coefficient path).

    Each line is one source variable.  As lambda_score increases, attention
    scores shrink toward zero — variables whose lines reach zero first are
    the least causally important.

    Args:
        sweep_dir: Sweep root directory.
        results:   DataFrame from ``collect_score_sparsity_results``.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.")

    all_norms = []
    lambdas = []

    for _, row in results.iterrows():
        norms = collect_attention_score_norms(row["run_dir"])
        if norms is not None:
            all_norms.append(norms)
            lambdas.append(row["lambda_cross_score"])

    if not all_norms:
        logger.warning("No attention score data found. Skipping variable importance path.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attention score data found", ha="center", va="center")
        return fig

    n_vars = all_norms[0].shape[0]
    norms_arr = np.stack(all_norms)  # (n_lambdas, n_vars)
    log_lambdas = np.log10(np.where(np.array(lambdas) == 0, 1e-6, lambdas))

    fig, ax = plt.subplots(figsize=(10, 6))
    for v in range(n_vars):
        ax.plot(log_lambdas, norms_arr[:, v], label=f"var {v}")

    ax.set_xlabel("log10(lambda_cross_score)")
    ax.set_ylabel("Attention score norm")
    ax.set_title("Variable Importance Path (LASSO-style)\nEach line = one source variable")
    ax.legend(loc="upper right", ncol=min(4, n_vars), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# FULL ANALYSIS PIPELINE
# =============================================================================

def run_score_sparsity_analysis(
    sweep_dir: str,
    rule: str = "1se",
    tolerance: float = 0.05,
    show_plots: bool = False,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Full analysis pipeline for a score-sparsity sweep.

    Steps:
    1. Collect per-run metrics
    2. Select optimal lambda using the specified rule
    3. Plot LASSO-path figures
    4. Save results summary

    Args:
        sweep_dir:  Root directory of the score-sparsity sweep.
        rule:       Lambda selection rule (``"1se"`` or ``"min_hsic"``).
        tolerance:  Fractional tolerance for the ``"1se"`` rule.
        show_plots: Whether to display plots interactively.
        save_dir:   Directory to save plots and summary (defaults to sweep_dir).

    Returns:
        Dict with keys:
        - ``lambda_cross_score_selected``
        - ``results`` (DataFrame)
        - ``summary_path``
    """
    save_dir = Path(save_dir or sweep_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print("SCORE SPARSITY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Sweep directory: {sweep_dir}")
    print(f"  Selection rule:  {rule}")

    results = collect_score_sparsity_results(sweep_dir)
    print(f"\n  Found {len(results)} lambda values:")
    for _, row in results.iterrows():
        print(
            f"    lambda={row['lambda_cross_score']:.4f}  "
            f"HSIC_cross={row['hsic_cross']:.4f}  "
            f"val_MAE={row['val_mae']:.4f}"
        )

    lambda_star = select_lambda_score(results, rule=rule, tolerance=tolerance)
    print(f"\n  Selected lambda_cross_score = {lambda_star:.4f} (rule='{rule}')")

    # Plot LASSO paths
    sparsity_path_fig = plot_score_sparsity_path(
        results,
        save_path=str(save_dir / "score_sparsity_path.png"),
        selected_lambda=lambda_star,
    )

    var_importance_fig = plot_variable_importance_path(
        sweep_dir,
        results,
        save_path=str(save_dir / "variable_importance_path.png"),
    )

    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()

    # Save summary
    summary = {
        "lambda_cross_score_selected": lambda_star,
        "selection_rule": rule,
        "tolerance": tolerance,
        "n_lambda_values": len(results),
        "results": results.to_dict(orient="records"),
    }
    summary_path = save_dir / "score_sparsity_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x,
        )

    print(f"\n  Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    return {
        "lambda_cross_score_selected": lambda_star,
        "results": results,
        "summary_path": str(summary_path),
    }
