"""
Config Operations: Pure config transform utilities.

All functions here are **pure**: they accept a config dict and return a new
config dict (deep copy).  No training, no I/O — only config manipulation.

This makes them trivially testable and composable.  They are the "glue" that
connects the outputs of one pipeline stage to the inputs of the next:

    calibrate_group_l1(config) → CalibrationResult
        ↓  apply_calibration_to_config(config, result)
    updated_config
        ↓  euler_sweep(updated_config, sweep.yaml)   [λ_score search]
    sweep results → selected λ_score
        ↓  apply_score_sparsity_to_config(config, λ_cross, λ_self)
    updated_config
        ↓  staged_trainer / trainer

Usage example::

    from causaliT.training.config_operations import (
        apply_calibration_to_config,
        apply_score_sparsity_to_config,
        configure_main_training_from_staged,
    )

    config = apply_calibration_to_config(config, cal_result)
    config = apply_score_sparsity_to_config(config, 0.01, 0.001)
    config = configure_main_training_from_staged(config)
"""

import copy
from typing import Optional


# =============================================================================
# CALIBRATION → CONFIG
# =============================================================================

def apply_calibration_to_config(config: dict, cal_result) -> dict:
    """
    Bake calibration results into the config.

    Writes the following keys so that every downstream stage (sweep, causal
    init, main training) automatically picks up the calibrated values:

    - ``training.lambda_group_l1``                ← λ_group*
    - ``training.lambda_hsic_cross``              ← suggested cross HSIC λ
    - ``training.lambda_hsic_self``               ← suggested self  HSIC λ
    - ``staged_training.lambda_group_l1``         ← same, for staged pipeline
    - ``staged_training.calibration_checkpoint``  ← path to Phase-2 checkpoint
    - ``staged_training.lambda_hsic_cross_suggested``
    - ``staged_training.lambda_hsic_self_suggested``

    Args:
        config:     Configuration dict (not modified in-place).
        cal_result: ``CalibrationResult`` namedtuple from
                    ``calibration.calibrate_group_l1``.

    Returns:
        A deep copy of ``config`` with calibrated values applied.
    """
    config = copy.deepcopy(config)

    # Write into training section (used by trainer / causal_init)
    config["training"]["lambda_group_l1"] = float(cal_result.lambda_group_optimal)
    config["training"]["lambda_hsic_cross"] = float(cal_result.lambda_hsic_cross_suggested)
    config["training"]["lambda_hsic_self"] = float(cal_result.lambda_hsic_self_suggested)

    # Write into staged_training section (used by staged_trainer summary)
    if "staged_training" not in config:
        config["staged_training"] = {}
    config["staged_training"]["lambda_group_l1"] = float(cal_result.lambda_group_optimal)
    config["staged_training"]["calibration_checkpoint"] = cal_result.checkpoint_path
    config["staged_training"]["lambda_hsic_cross_suggested"] = float(
        cal_result.lambda_hsic_cross_suggested
    )
    config["staged_training"]["lambda_hsic_self_suggested"] = float(
        cal_result.lambda_hsic_self_suggested
    )

    return config


# =============================================================================
# SCORE SPARSITY → CONFIG
# =============================================================================

def apply_score_sparsity_to_config(
    config: dict,
    lambda_cross_score: float,
    lambda_self_score: float = 0.0,
) -> dict:
    """
    Set score sparsity lambdas in the config.

    These values add an L1 penalty to the attention score matrix, enforcing
    sparse attention (i.e. anti-parallel embeddings for Toeplitz attention).

    Downstream consumers:
    - ``trainer`` / ``train_single_fold`` → read from ``training.*``
    - ``causal_initialization.run_causal_initialization`` → reads and passes
      through these values so causal init runs inside the selected sparse
      landscape.

    Args:
        config:              Configuration dict (not modified in-place).
        lambda_cross_score:  L1 penalty on cross-attention score matrix.
        lambda_self_score:   L1 penalty on self-attention score matrix.

    Returns:
        A deep copy of ``config`` with score sparsity lambdas applied.
    """
    config = copy.deepcopy(config)
    config["training"]["lambda_cross_score_sparse"] = float(lambda_cross_score)
    config["training"]["lambda_self_score_sparse"] = float(lambda_self_score)
    return config


# =============================================================================
# MAIN TRAINING OVERRIDES (post causal-init)
# =============================================================================

def configure_main_training_from_staged(config: dict) -> dict:
    """
    Apply config overrides for the main training stage after a staged pipeline.

    Propagates calibrated HSIC values from the staged pipeline to main
    training.  The user's ``training.use_hsic_annealing`` setting is
    **respected**:

    - If ``use_hsic_annealing == True`` (default for backward compat):
      Sets up linear annealing from calibrated values to end values.
    - If ``use_hsic_annealing == False``:
      Sets calibrated values as **constant** ``lambda_hsic_cross`` and
      ``lambda_hsic_self`` (no annealing).  Use this with adaptive
      bandwidth (``hsic_adaptive_bandwidth: true``) where the HSIC
      signal is already scale-invariant.

    Changes applied **only when** ``staged_training.use_causal_init == True``:

    When annealing is enabled:
    - ``training.use_hsic_annealing``       = True
    - ``training.hsic_lambda_cross_start``  ← calibrated λ_hsic_cross
    - ``training.hsic_lambda_self_start``   ← calibrated λ_hsic_self
    - ``training.hsic_lambda_cross_end``    ← kept from config (default 0.0)
    - ``training.hsic_lambda_self_end``     ← kept from config (default 0.0)
    - ``training.hsic_anneal_epochs``       ← kept from config or set to 50%
      of ``max_epochs``

    When annealing is disabled:
    - ``training.lambda_hsic_cross``        ← calibrated λ_hsic_cross
    - ``training.lambda_hsic_self``         ← calibrated λ_hsic_self

    Also propagates unconditionally (when the staged value is not None):
    - ``staged_training.lambda_group_l1``      → ``training.lambda_group_l1``
    - ``staged_training.lambda_score_suggested``→ ``training.lambda_cross_score_sparse``
      and ``training.lambda_self_score_sparse``

    Args:
        config: Configuration dict (not modified in-place).

    Returns:
        A deep copy of ``config`` with main-training overrides applied.
    """
    config = copy.deepcopy(config)
    staged = config.get("staged_training", {})
    training = config["training"]

    # Propagate group L1 from staged_training if present
    lambda_group = staged.get("lambda_group_l1", None)
    if lambda_group is not None:
        training["lambda_group_l1"] = float(lambda_group)

    # Propagate score sparsity lambda from CV (Stage 2) if present
    lambda_score = staged.get("lambda_score_suggested", None)
    if lambda_score is not None:
        training["lambda_cross_score_sparse"] = float(lambda_score)
        training["lambda_self_score_sparse"] = float(lambda_score)

    use_causal_init = staged.get("use_causal_init", False)
    if use_causal_init:
        # Resolve calibrated HSIC lambda values
        lambda_hsic_cross = (
            staged.get("lambda_hsic_cross_suggested")
            or training.get("lambda_hsic_cross", training.get("lambda_hsic", 0.1))
        )
        lambda_hsic_self = (
            staged.get("lambda_hsic_self_suggested")
            or training.get("lambda_hsic_self", 0.0)
        )

        # Respect the user's annealing preference
        use_annealing = training.get("use_hsic_annealing", True)

        if use_annealing:
            # Anneal from calibrated values to end values (original behavior)
            training["use_hsic_annealing"] = True
            training["hsic_lambda_cross_start"] = float(lambda_hsic_cross)
            training["hsic_lambda_self_start"] = float(lambda_hsic_self)
            training["hsic_lambda_cross_end"] = float(
                training.get("hsic_lambda_cross_end", 0.0)
            )
            training["hsic_lambda_self_end"] = float(
                training.get("hsic_lambda_self_end", 0.0)
            )

            if training.get("hsic_anneal_epochs") is None:
                training["hsic_anneal_epochs"] = int(training["max_epochs"] * 0.5)
        else:
            # Constant HSIC: set calibrated values as fixed lambdas
            training["use_hsic_annealing"] = False
            training["lambda_hsic_cross"] = float(lambda_hsic_cross)
            training["lambda_hsic_self"] = float(lambda_hsic_self)

    return config


# =============================================================================
# SEED OVERRIDE (for multi-seed sweeps)
# =============================================================================

def apply_seed_to_config(config: dict, seed: int) -> dict:
    """
    Override the random seed in the config.

    Useful when assembling a multi-seed combination sweep via euler_sweep:

    .. code-block:: yaml

        # sweep.yaml
        training:
          seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          lambda_cross_score_sparse: [0.0, 0.001, 0.01, 0.1]

    Args:
        config: Configuration dict (not modified in-place).
        seed:   Seed value to set.

    Returns:
        A deep copy of ``config`` with ``training.seed`` = seed.
    """
    config = copy.deepcopy(config)
    config["training"]["seed"] = int(seed)
    return config
