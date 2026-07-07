"""
causaliT CLI for Optuna Hyperparameter Optimisation

Wires the generic euler_optuna framework to causaliT's trainer and causal
transformer models.  Only causal transformer models are supported.

SUPPORTED MODELS
----------------
- proT
- SingleCausalLayer / SingleCausalLayerRes
- NoiseAwareSingleCausalLayer / NoiseAwareSingleCausalLayerRes
- AttentionSelectorLayer

Note: StageCausaliT is intentionally excluded from the current Optuna scope
because it uses a different metric naming convention (``val_mae_X`` instead
of ``val_x_mae``).  Add it in a future iteration.

OPTIMISED PARAMETERS (capacity-focused)
----------------------------------------
All models share the same ``experiment.*`` config path convention:

  experiment.d_model_set  — embedding / model dimension
  experiment.n_heads      — number of attention heads (sampled from {1, 2, 4})
  experiment.dec_layers   — decoder depth  (Single* / NoiseAware*)
  experiment.e_layers     — encoder depth  (proT)
  experiment.d_layers     — decoder depth  (proT)
  training.lr             — learning rate
  experiment.dropout      — dropout rate

``d_ff`` and ``d_qk`` are intentionally NOT sampled: they are derived
automatically from ``d_model_set`` via the ``d_ff_mult`` / ``d_qk_mult``
multipliers inside ``update_config()``.

``n_heads`` is sampled from the categorical set {1, 2, 4} — NOT from a
continuous integer range.  This guarantees n_heads always divides d_model_set
(which is a multiple of 16), avoiding the silent error where n_heads=3
creates a non-integer attention head dimension.

RECONSTRUCTION PROTOCOL
------------------------
Every trial applies ``OPTUNA_RECONSTRUCTION_PROTOCOL`` overrides before
training.  These overrides zero out all structural regularisation, disable
gradient routing, and set k_fold=1 so that every model is evaluated on an
identical fair footing: pure reconstruction ability.  Two JSON files track
what was forced:

  exp_dir/optuna/optuna_protocol.json     — study-level copy (written on create)
  exp_dir/optuna/run_N/optuna_protocol.json — per-trial copy (written on train)
  exp_dir/optuna/run_N/config.yaml          — ACTUAL training config (with both
                                              sampled params AND protocol applied)

USAGE
-----
# Create study
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
    --exp_id my_exp --study_name capacity_study --mode create

# Run optimisation (sequential)
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
    --exp_id my_exp --study_name capacity_study --mode resume

# Run optimisation (parallel SLURM)
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
    --exp_id my_exp --study_name capacity_study --mode resume \\
    --parallel --cluster --n_trials 50

# View best result
python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
    --exp_id my_exp --study_name capacity_study --mode summary
"""

# Standard library
import json
import os
import re
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

# Third-party
import click
from omegaconf import OmegaConf

# ── Project root ──────────────────────────────────────────────────────────────
# cli.py is at  causaliT/causaliT/euler_optuna/euler_optuna/cli.py
# ROOT_DIR is   causaliT/
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

# ── causaliT imports ──────────────────────────────────────────────────────────
from causaliT.training.trainer import trainer
from causaliT.training.experiment_control import update_config
from causaliT.euler_optuna.euler_optuna.optuna_opt import OptunaStudy
from causaliT.euler_optuna.euler_optuna.optuna_parallel import run_parallel_optuna


# =============================================================================
# RECONSTRUCTION PROTOCOL
# =============================================================================

#: Canonical reconstruction protocol applied to every Optuna trial.
#:
#: Rationale for each override:
#:
#:   k_fold=1          — Single 80/20 split makes each trial ~5x faster than
#:                       5-fold CV.  Capacity parameters (d_model, n_layers)
#:                       have low variance across splits, so the accuracy gain
#:                       from k-fold does not justify the cost.
#:
#:   use_gradient_routing=False — Ensures ALL model parameters receive
#:                       reconstruction gradients.  With routing ON, Q/K and
#:                       structure embeddings get no gradient from reconstruction
#:                       loss (only from HSIC which is zero here), starving the
#:                       structural stream and artificially inflating the
#:                       measured capacity requirement.
#:
#:   lambda_* = 0.0    — Zero out every structural regularisation term.
#:                       HSIC, group L1, score sparsity and noise prior all
#:                       compete with or distort the reconstruction loss.
#:                       Disabling them ensures the only optimisation signal
#:                       is pure reconstruction quality (val_x_mae).
#:
#:   early_stopping    — Stops wasteful trials early and ensures that
#:                       best=True (best-checkpoint metrics) picks the epoch
#:                       with the lowest val_x_mae rather than the final epoch.
#:
#:   staged_training.* — Staged protocols (calibration, causal init, score
#:                       sparsity CV) are designed for the full causal training
#:                       pipeline, not for capacity discovery.  They would
#:                       multiply trial cost 3-10x with no benefit here.
OPTUNA_RECONSTRUCTION_PROTOCOL = {
    "protocol_name": "optuna_reconstruction_v1",
    "description": (
        "Forced overrides applied to every Optuna trial to ensure fair "
        "reconstruction capacity comparison across all models. "
        "All structural regularization is disabled so the only signal "
        "is pure reconstruction quality (val_x_mae). "
        "Do NOT edit this dict; change OPTUNA_RECONSTRUCTION_PROTOCOL in cli.py."
    ),
    "rationale": {
        "k_fold": (
            "Single 80/20 split: ~5x faster trials. Capacity parameters "
            "(d_model, n_layers) have low variance across splits."
        ),
        "use_gradient_routing": (
            "OFF: all parameters receive reconstruction gradients, giving "
            "the true architectural capacity upper bound."
        ),
        "lambda_*": (
            "All structural regularization zeroed: prevents HSIC / L1 / "
            "score-sparsity from confounding the reconstruction quality signal."
        ),
        "early_stopping": (
            "Stops wasteful trials early; best=True uses best-epoch metrics "
            "rather than final-epoch metrics."
        ),
        "staged_training": (
            "All staged protocols disabled: calibration / causal_init / "
            "score_sparsity_cv are for causal training, not capacity finding."
        ),
    },
    "overrides": {
        "training.k_fold": 1,
        "training.use_gradient_routing": False,
        "training.lambda_hsic_cross": 0.0,
        "training.lambda_hsic_self": 0.0,
        "training.lambda_group_l1": 0.0,
        "training.lambda_self_score_sparse": 0.0,
        "training.lambda_cross_score_sparse": 0.0,
        "training.lambda_noise_prior": 0.0,
        "training.lambda_sparse": 0.0,
        "training.lambda_sparse_cross": 0.0,
        # AttentionSelectorLayer uses unified lambda keys (not _cross/_self variants).
        # These are silently skipped for models that don't have them (try/except in
        # apply_optuna_reconstruction_protocol), so adding them here is backward-safe.
        "training.lambda_hsic": 0.0,
        "training.lambda_score_sparse": 0.0,
        "training.early_stopping.enabled": True,
        "training.early_stopping.monitor": "val_x_mae",
        "training.early_stopping.patience": 30,
        "training.early_stopping.min_delta": 1e-5,
        "training.early_stopping.mode": "min",
        "staged_training.use_calibration": False,
        "staged_training.use_causal_init": False,
        "staged_training.use_score_sparsity_cv": False,
    },
}


#: -----------------------------------------------------------------------------
#: PROTOCOL EXTENSIONS (constant-score capacity search)
#: -----------------------------------------------------------------------------
#: The base ``OPTUNA_RECONSTRUCTION_PROTOCOL`` completely severs the structural
#: (query/key) stream by zeroing every structural lambda AND disabling gradient
#: routing.  With the newer architectures (free query embedding, orthogonal
#: embeddings, orthogonal key projection, HardConcrete attention) the learned
#: QK^T attention weights still fluctuate freely during a search trial, so the
#: measured reconstruction floor is polluted by an *untrained, noisy* selection
#: matrix.  The result is the very poor first-phase fit observed in the ARCH
#: studies.
#:
#: These extensions REPLACE the learned attention weights with a CONSTANT on
#: every allowed edge (via ``model.kwargs.optuna_protocol``; see
#: ``CausalCrossAttention``).  This makes the value/reconstruction capacity
#: measurable independently of an untrained structure — mimicking the
#: frozen-structure warmup phase — so ``d_model_set`` can be tuned honestly.
#:
#:   floor_residual_v1  → optuna_protocol=0.0
#:       All allowed edges receive weight 0 → the value stream contributes
#:       nothing; only the query residual / FFN / MLP-head reconstruct X.
#:       Measures the *residual-only* floor (a strict lower bound on capacity).
#:
#:   regime_uniform_v1  → optuna_protocol=1.0 + heavy batch_key_dropout=0.5
#:       All allowed edges receive weight 1 (uniform mixing of every candidate
#:       parent).  Because uniform mixing is selection-INSENSITIVE, a heavy
#:       batch-consistent key dropout (p=0.5, non-annealed) forces the value
#:       stream to reconstruct from random *subsets* of parents, which is the
#:       regime the downstream sparse structure will operate in.  This is the
#:       recommended profile for choosing the embedding dimension.
#:
#: NOTE: ``optuna_protocol`` is honoured by ``CausalCrossAttention`` (the
#: AttentionSelectorLayer default) and by ``GatedCrossAttention``.  For
#: ``GatedCrossAttention`` the override is GATE-ONLY: it freezes the
#: Hard-Concrete STRUCTURE gate ``z`` at the constant (0/1) while the
#: bounded-sigmoid RECONSTRUCTION gain ``g`` stays learnable — so the capacity
#: search measures the true gain+value reconstruction pathway.  For all other
#: attention types the flag is silently ignored, so these extensions have no
#: effect there and the base protocol behaviour is retained.

OPTUNA_PROTOCOL_EXTENSIONS = {
    "none": None,
    "floor_residual_v1": {
        "protocol_name": "optuna_floor_residual_v1",
        "description": (
            "Constant-score capacity search: the learned QK^T attention is "
            "replaced by 0 on every allowed edge, severing the value stream so "
            "only the query residual/FFN/MLP-head reconstruct X. Measures the "
            "residual-only reconstruction floor (strict lower bound)."
        ),
        "overrides": {
            "model.kwargs.optuna_protocol": 0.0,
        },
    },
    "regime_uniform_v1": {
        "protocol_name": "optuna_regime_uniform_v1",
        "description": (
            "Constant-score capacity search: the learned QK^T attention is "
            "replaced by 1 on every allowed edge (uniform mixing of all "
            "candidate parents), paired with heavy non-annealed batch-consistent "
            "key dropout (p=0.5) so the value stream must reconstruct X from "
            "random subsets of parents. This is the recommended profile for "
            "tuning the embedding dimension because it measures value-stream "
            "capacity under the sparse-selection regime the model will face "
            "downstream, without relying on an untrained structure."
        ),
        "overrides": {
            "model.kwargs.optuna_protocol": 1.0,
            "model.kwargs.batch_key_dropout": 0.5,
            "model.kwargs.batch_key_dropout_p_final": 0.5,
            "model.kwargs.batch_key_dropout_annealing_batches": None,
        },
    },
}

#: Module-level variable — set from the ``protocol:`` key of the experiment's
#: ``optuna*.yaml`` (via ``load_protocol_extension``) before optimisation
#: starts.  Holds the selected extension dict (or None).  Both the sequential
#: driver (``paramsopt``) and each parallel SLURM worker set this in their own
#: process, so the constant-score override applies identically in every mode.
ACTIVE_PROTOCOL_EXTENSION = None


def load_protocol_extension(home_exp_dir):
    """
    Resolve the capacity-search protocol extension for an experiment.

    Reads the ``protocol:`` key from the single ``optuna*.yaml`` settings file
    in ``home_exp_dir``.  This is the single source of truth so that BOTH the
    sequential driver and every parallel SLURM worker (which each load the
    settings fresh from disk) apply the same extension automatically — no CLI
    flag required.

    Args:
        home_exp_dir: Directory containing ``optuna*.yaml`` (and ``config*.yaml``).

    Returns:
        Tuple ``(extension_dict_or_None, protocol_name)``.

    Raises:
        ValueError: If ``protocol`` names an unknown extension.
    """
    name = "none"
    optuna_files = [
        f for f in os.listdir(home_exp_dir) if re.match(r"optuna.*\.yaml", f)
    ]
    if len(optuna_files) == 1:
        settings = OmegaConf.load(join(home_exp_dir, optuna_files[0]))
        name = settings.get("protocol", "none") or "none"

    if name not in OPTUNA_PROTOCOL_EXTENSIONS:
        raise ValueError(
            f"Unknown protocol '{name}' in optuna settings. "
            f"Valid options: {list(OPTUNA_PROTOCOL_EXTENSIONS.keys())}"
        )
    return OPTUNA_PROTOCOL_EXTENSIONS[name], name



def apply_optuna_reconstruction_protocol(config, save_dir=None, protocol_extension=None):
    """
    Apply the standard reconstruction protocol overrides to a trial config.


    Ensures every trial runs under identical fair conditions:
      - k_fold=1 (single 80/20 split, ~5x faster)
      - All structural regularisation lambdas set to 0.0
      - Gradient routing disabled (pure reconstruction gradients)
      - Early stopping enabled (patience=30, monitor=val_x_mae)
      - Staged training protocols disabled

    If ``save_dir`` is provided the protocol dict is written to
    ``save_dir/optuna_protocol.json`` for per-trial auditability.

    When ``protocol_extension`` is provided (one of the
    ``OPTUNA_PROTOCOL_EXTENSIONS`` dicts, or None), its overrides are applied
    AFTER the base protocol so they take precedence (e.g. the constant-score
    ``model.kwargs.optuna_protocol`` flag and heavy batch_key_dropout).

    Args:
        config:             OmegaConf config (already has sampled params applied).
        save_dir:           Optional path to write ``optuna_protocol.json``.
        protocol_extension: Optional extension dict from
                            ``OPTUNA_PROTOCOL_EXTENSIONS`` (or None).

    Returns:
        Modified OmegaConf config (struct mode disabled, overrides applied).
    """
    OmegaConf.set_struct(config, False)

    for dotted_key, value in OPTUNA_RECONSTRUCTION_PROTOCOL["overrides"].items():
        try:
            OmegaConf.update(config, dotted_key, value, merge=True)
        except Exception:
            # Some keys (e.g. lambda_sparse) may not exist in all model
            # configs.  This is expected and safe to ignore — the key is
            # simply not relevant for that model.
            pass

    # Apply the constant-score extension overrides (if any) on top of the base
    # protocol so they win where keys overlap.
    if protocol_extension is not None:
        for dotted_key, value in protocol_extension["overrides"].items():
            try:
                OmegaConf.update(config, dotted_key, value, merge=True)
            except Exception:
                pass

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = save_dir / "optuna_protocol.json"
        payload = dict(OPTUNA_RECONSTRUCTION_PROTOCOL)
        if protocol_extension is not None:
            payload = {**payload, "protocol_extension": protocol_extension}
        with open(protocol_path, "w") as f:
            json.dump(payload, f, indent=2)

    return config



# =============================================================================
# SAMPLING BOUNDS
# =============================================================================

BASELINE_SAMPLING_BOUNDS = {
    # ── Capacity ─────────────────────────────────────────────────────────────
    "d_model_set":   {"low": 16, "high": 128, "step": 16},
    # n_heads is NOT sampled as an integer range — see N_HEADS_CHOICES below.
    # Sampling from {1, 2, 4} guarantees divisibility with any d_model_set
    # that is a multiple of 16.  n_heads=3 would break d_model=16 silently.
    "n_layers":      {"low": 1,  "high": 4,   "step": 1},   # generic depth

    # ── Regularisation ───────────────────────────────────────────────────────
    "dropout": {"low": 0.0, "high": 0.3, "step": 0.1},

    # ── Optimiser ────────────────────────────────────────────────────────────
    "lr": {"low": 1e-4, "high": 1e-3, "log": True},
}

#: Valid n_heads values — must always divide d_model_set (multiples of 16).
#: {1, 2, 4} covers the useful range; 3 and non-power-of-2 values are excluded
#: because they cannot divide 16.
N_HEADS_CHOICES = [1, 2, 4]

SAMPLING_PROFILES = {
    "baseline": BASELINE_SAMPLING_BOUNDS,
}

# Module-level variable — updated by CLI flag before optimisation starts.
SAMPLING_BOUNDS = BASELINE_SAMPLING_BOUNDS


# =============================================================================
# MODEL-SPECIFIC SAMPLING FUNCTIONS
# =============================================================================

def proT_sample_params(trial):
    """
    Capacity sampling for ``proT`` (TransformerForecaster).

    Config keys used (all under ``experiment.*``):
      - d_model_set  : embedding / model dimension
      - e_layers     : encoder depth
      - d_layers     : decoder depth
      - n_heads      : attention heads (categorical from {1, 2, 4})
      - dropout      : dropout rate
      - training.lr  : learning rate
    """
    return {
        "experiment.d_model_set": trial.suggest_int("d_model_set",   **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.e_layers":    trial.suggest_int("e_layers",      **SAMPLING_BOUNDS["n_layers"]),
        "experiment.d_layers":    trial.suggest_int("d_layers",      **SAMPLING_BOUNDS["n_layers"]),
        "experiment.n_heads":     trial.suggest_categorical("n_heads", N_HEADS_CHOICES),
        "experiment.dropout":     trial.suggest_float("dropout",     **SAMPLING_BOUNDS["dropout"]),
        "training.lr":            trial.suggest_float("lr",          **SAMPLING_BOUNDS["lr"]),
    }


def StageCausaliT_sample_params(trial):
    """
    Capacity sampling for ``StageCausaliT``.

    Config keys used (all under ``experiment.*``):
      - d_model_set  : shared embedding dimension
      - d1_layers    : decoder-1 (structure pathway) depth
      - d2_layers    : decoder-2 (reconstruction pathway) depth
      - n_heads      : attention heads (categorical from {1, 2, 4})
      - dropout      : dropout rate
      - training.lr  : learning rate

    Note: StageCausaliT is currently excluded from the Optuna capacity study
    scope (it logs ``val_mae_X`` instead of ``val_x_mae``).  This function is
    kept for future use.
    """
    return {
        "experiment.d_model_set": trial.suggest_int("d_model_set",   **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.d1_layers":   trial.suggest_int("d1_layers",     **SAMPLING_BOUNDS["n_layers"]),
        "experiment.d2_layers":   trial.suggest_int("d2_layers",     **SAMPLING_BOUNDS["n_layers"]),
        "experiment.n_heads":     trial.suggest_categorical("n_heads", N_HEADS_CHOICES),
        "experiment.dropout":     trial.suggest_float("dropout",     **SAMPLING_BOUNDS["dropout"]),
        "training.lr":            trial.suggest_float("lr",          **SAMPLING_BOUNDS["lr"]),
    }


def SingleCausal_sample_params(trial):
    """
    Capacity sampling for ``SingleCausalLayer``, ``SingleCausalLayerRes``,
    ``NoiseAwareSingleCausalLayer``, and ``NoiseAwareSingleCausalLayerRes``.

    Config keys used (all under ``experiment.*``):
      - d_model_set  : embedding dimension
      - dec_layers   : decoder depth
      - n_heads      : attention heads (categorical from {1, 2, 4})
      - dropout      : dropout rate
      - training.lr  : learning rate
    """
    return {
        "experiment.d_model_set": trial.suggest_int("d_model_set",   **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.dec_layers":  trial.suggest_int("dec_layers",    **SAMPLING_BOUNDS["n_layers"]),
        "experiment.n_heads":     trial.suggest_categorical("n_heads", N_HEADS_CHOICES),
        "experiment.dropout":     trial.suggest_float("dropout",     **SAMPLING_BOUNDS["dropout"]),
        "training.lr":            trial.suggest_float("lr",          **SAMPLING_BOUNDS["lr"]),
    }


def AttentionSelector_sample_params(trial):
    """
    Capacity sampling for ``AttentionSelectorLayer``.

    This architecture uses a *single* combined cross-attention block (no
    stacked decoder layers), so ``dec_layers`` is intentionally absent from
    the search space.

    Config keys sampled:
      - d_model_set  : embedding / model dimension
      - n_heads      : attention heads (categorical from {1, 2, 4})
      - training.lr  : learning rate

    Note: ``dropout`` is intentionally NOT sampled for this architecture — it is
    fixed at 0.0 in the config.  The constant-score capacity protocol already
    regularises the value stream via heavy batch_key_dropout, and pure
    reconstruction capacity is best measured without additional dropout noise.

    Note on n_heads:
      When comps_embed is "svfa" and n_heads=1, the SVFA path degrades
      gracefully to summation-equivalent behaviour (one value head = full
      d_model vector).  n_heads ∈ {1, 2, 4} is safe for any d_model_set
      that is a multiple of 16 (guaranteed by the d_model_set bounds).
    """
    return {
        "experiment.d_model_set": trial.suggest_int("d_model_set",   **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.n_heads":     trial.suggest_categorical("n_heads", N_HEADS_CHOICES),
        "training.lr":            trial.suggest_float("lr",          **SAMPLING_BOUNDS["lr"]),
    }



# =============================================================================
# DISPATCHER
# =============================================================================

#: Map from model_object string → sampling function
_SAMPLING_DISPATCH = {
    "proT":                           proT_sample_params,
    "StageCausaliT":                  StageCausaliT_sample_params,
    "SingleCausalLayer":              SingleCausal_sample_params,
    "SingleCausalLayerRes":           SingleCausal_sample_params,
    "NoiseAwareSingleCausalLayer":    SingleCausal_sample_params,
    "NoiseAwareSingleCausalLayerRes": SingleCausal_sample_params,
    "AttentionSelectorLayer":         AttentionSelector_sample_params,
}


def sample_params_for_optuna(trial, config):
    """
    Dispatch to the model-specific sampling function.

    Args:
        trial:  Optuna trial object.
        config: OmegaConf configuration (used to read ``model.model_object``).

    Returns:
        Dict of dotted config keys → sampled values.

    Raises:
        ValueError: If the model is not supported.
    """
    model_obj = config["model"]["model_object"]
    if model_obj not in _SAMPLING_DISPATCH:
        supported = list(_SAMPLING_DISPATCH.keys())
        raise ValueError(
            f"No sampling function defined for model '{model_obj}'. "
            f"Supported models: {supported}"
        )
    return _SAMPLING_DISPATCH[model_obj](trial)


# =============================================================================
# TRAINING WRAPPER
# =============================================================================

def train_function_for_optuna(
    config,
    save_dir: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    **kwargs,
):
    """
    Wrapper that connects the Optuna framework to causaliT's ``trainer``.

    Steps:
    1. Apply ``update_config()`` to resolve derived fields (d_ff, d_qk, …).
    2. Apply ``OPTUNA_RECONSTRUCTION_PROTOCOL`` overrides for fair conditions.
    3. Re-save ``config.yaml`` in the trial folder so it reflects what was
       actually trained (sampled params + protocol overrides both applied).
    4. Write ``optuna_protocol.json`` to the trial folder for auditability.
    5. Call ``trainer()`` with ``best=True`` so that metrics come from the
       best checkpoint (lowest val_x_mae epoch), not the final epoch.

    Args:
        config:          OmegaConf config (already has sampled params applied).
        save_dir:        Directory for checkpoints / logs for this trial.
        data_dir:        Root data directory.
        experiment_tag:  Tag for experiment manifests.
        cluster:         Whether running on a cluster.
        **kwargs:        Forwarded to ``trainer`` (e.g. resume_ckpt, debug).

    Returns:
        pd.DataFrame — one row per fold, columns are metric names.
    """
    # 1. Resolve d_ff, d_qk, d_model_enc/dec from multipliers
    config_updated = update_config(config)

    # 2. Apply protocol: zero structural lambdas, disable routing, k_fold=1,
    #    inject early stopping, disable staged training.
    #    Also writes optuna_protocol.json to save_dir.
    config_updated = apply_optuna_reconstruction_protocol(
        config_updated, save_dir=save_dir,
        protocol_extension=ACTIVE_PROTOCOL_EXTENSION,
    )


    # 3. Re-save the trial config so the on-disk config.yaml matches what
    #    trainer() actually received (sampled params + protocol overrides).
    OmegaConf.save(config_updated, Path(save_dir) / "config.yaml")

    # 4. Train and return the per-fold metrics DataFrame.
    #    best=True: use metrics from the best-reconstruction checkpoint
    #    (lowest val_x_mae epoch) rather than the final epoch.
    return trainer(
        config=config_updated,
        data_dir=str(data_dir),
        save_dir=str(save_dir),
        cluster=cluster,
        experiment_tag=experiment_tag,
        resume_ckpt=kwargs.get("resume_ckpt", None),
        debug=kwargs.get("debug", False),
        best=True,
    )


# =============================================================================
# METRICS EXTRACTION
# =============================================================================

def get_metrics_for_optuna(train_results):
    """
    Extract aggregated metrics from the k-fold DataFrame returned by ``trainer``.

    The function is adaptive: it computes mean and std for every numeric column
    present in the DataFrame, so it works regardless of which specific metrics
    the model logs (e.g. ``val_x_mae``, ``val_loss``, custom causal metrics).

    Args:
        train_results: pd.DataFrame returned by ``trainer`` (one row per fold).

    Returns:
        Dict of ``{col}_mean`` and ``{col}_std`` for every numeric column.
    """
    df = train_results

    result = {}
    for col in df.columns:
        if df[col].dtype.kind in "fc":   # float or complex (numeric)
            result[f"{col}_mean"] = float(df[col].mean())
            result[f"{col}_std"]  = float(df[col].std())

    if not result:
        raise ValueError(
            "No numeric columns found in trainer output. "
            "Check that the trainer returns a DataFrame with metric columns."
        )

    return result


# =============================================================================
# CLI
# =============================================================================

@click.group()
def cli():
    """causaliT Optuna Optimisation CLI."""
    pass


@click.command()
@click.option("--exp_id",      required=True,
              help="Experiment folder (inside experiments/) containing config*.yaml")
@click.option("--cluster",     default=False, is_flag=True,
              help="Running on a cluster?")
@click.option("--study_name",  default="capacity_study",
              help="Name for the Optuna study (default: capacity_study)")
@click.option("--exp_tag",     default="NA",
              help="Tag stored in experiment manifests")
@click.option("--mode",        required=True,
              type=click.Choice(["create", "resume", "summary"]),
              help="create: new study | resume: continue | summary: show best trial")
@click.option("--scratch_path", default=None,
              help="SCRATCH path (for cluster); leave empty for local execution")
@click.option("--study_path",   default=None,
              help="Override path for the study database (default: exp_dir/optuna/)")
@click.option("--optimization_metric", default="val_x_mae_mean",
              help="Metric to minimise/maximise (default: val_x_mae_mean)")
@click.option("--optimization_direction", default="minimize",
              type=click.Choice(["minimize", "maximize"]),
              help="Optimisation direction (default: minimize)")
@click.option("--sampling_profile", default="baseline",
              type=click.Choice(["baseline"]),
              help="Sampling bounds profile (default: baseline)")

# ── Parallel execution (only with --mode resume --parallel) ──────────────────

@click.option("--parallel",          default=False, is_flag=True,
              help="Use SLURM job arrays for parallel execution (requires --cluster)")
@click.option("--n_trials",          type=int, default=50,
              help="Total trials for parallel mode (default: 50)")
@click.option("--max_concurrent_jobs", type=int, default=6,
              help="Max simultaneous SLURM jobs (default: 6)")
@click.option("--walltime",          default="5-00:00:00",
              help="SLURM walltime per trial (default: 5-00:00:00)")
@click.option("--gpu_type",          default="rtx_4090",
              help="GPU type for SLURM (default: rtx_4090)")
@click.option("--mem_per_cpu",       default="23g",
              help="Memory per CPU for SLURM (default: 23g)")
def paramsopt(
    exp_id, cluster, study_name, exp_tag, mode, scratch_path, study_path,
    optimization_metric, optimization_direction, sampling_profile,
    parallel, n_trials, max_concurrent_jobs, walltime, gpu_type, mem_per_cpu,
):


    """
    Hyperparameter optimisation for causaliT causal transformer models.

    \b
    Modes:
      create  — initialise a new Optuna study
      resume  — run optimisation trials (sequential or parallel with --parallel)
      summary — display and save the best trial result

    \b
    Protocol:
      Every trial is trained under the OPTUNA_RECONSTRUCTION_PROTOCOL: all
      structural regularisation is zeroed, gradient routing is disabled,
      k_fold=1, and early stopping is enabled.  See optuna_protocol.json
      in exp_dir/optuna/ for the full specification.

    \b
    Examples:
      # Sequential (local)
      python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
          --exp_id 3_OPT_STUDY/my_exp --study_name cap --mode create
      python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
          --exp_id 3_OPT_STUDY/my_exp --study_name cap --mode resume

      # Parallel (cluster)
      python -m causaliT.euler_optuna.euler_optuna.cli paramsopt \\
          --exp_id 3_OPT_STUDY/my_exp --study_name cap --mode resume \\
          --parallel --cluster --n_trials 50 --max_concurrent_jobs 8
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if parallel and not cluster:
        raise click.UsageError("--parallel requires --cluster.")
    if parallel and mode != "resume":
        raise click.UsageError("--parallel is only valid with --mode resume.")

    # ── Sampling bounds ───────────────────────────────────────────────────────
    global SAMPLING_BOUNDS
    SAMPLING_BOUNDS = SAMPLING_PROFILES[sampling_profile]

    print(f"causaliT Optuna | exp={exp_id} | study={study_name} | mode={mode}")
    print(f"Sampling profile : {sampling_profile}")
    print(f"Metric           : {optimization_metric} ({optimization_direction})")


    # ── Directories ───────────────────────────────────────────────────────────
    if scratch_path is None:
        exp_dir      = str(ROOT_DIR / "experiments" / exp_id)
        home_exp_dir = exp_dir
    else:
        exp_dir      = scratch_path
        home_exp_dir = str(ROOT_DIR / "experiments" / exp_id)

    data_dir = str(ROOT_DIR / "data")

    if not exists(home_exp_dir):
        raise ValueError(f"Experiment directory not found: {home_exp_dir}")

    print(f"Experiment dir   : {exp_dir}")
    print(f"Data dir         : {data_dir}")

    # ── Constant-score protocol extension (config-driven) ─────────────────────
    # Read from the `protocol:` key of the experiment's optuna*.yaml so the
    # SAME extension is applied in sequential AND parallel modes.  In parallel
    # mode the SLURM worker independently re-reads this file, so no CLI flag is
    # needed and there is no risk of the driver/worker desyncing.
    global ACTIVE_PROTOCOL_EXTENSION
    ACTIVE_PROTOCOL_EXTENSION, protocol_name = load_protocol_extension(home_exp_dir)
    print(f"Protocol ext.    : {protocol_name}")


    # ── Load base config ──────────────────────────────────────────────────────
    pattern = re.compile(r"config.*\.yaml")
    config_files = [f for f in os.listdir(home_exp_dir) if pattern.match(f)]
    if len(config_files) != 1:
        raise ValueError(
            f"Expected exactly 1 config*.yaml in {home_exp_dir}, "
            f"found {len(config_files)}: {config_files}"
        )
    base_config = OmegaConf.load(join(home_exp_dir, config_files[0]))
    model_obj = base_config["model"]["model_object"]
    print(f"Model            : {model_obj}")

    # Validate model is supported
    if model_obj not in _SAMPLING_DISPATCH:
        raise ValueError(
            f"Model '{model_obj}' is not supported by euler_optuna. "
            f"Supported: {list(_SAMPLING_DISPATCH.keys())}"
        )

    # ── Sampling function (binds config for dispatcher) ───────────────────────
    def sample_params_fn(trial):
        return sample_params_for_optuna(trial, base_config)

    # ── Resolve study directory (mirrors OptunaStudy logic) ───────────────────
    study_dir = Path(study_path) if study_path else Path(exp_dir) / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)

    # ── Execute ───────────────────────────────────────────────────────────────
    if mode == "create":
        # Save study-level protocol JSON before creating the study so it is
        # present even if study creation subsequently fails.
        protocol_path = study_dir / "optuna_protocol.json"
        with open(protocol_path, "w") as f:
            json.dump(OPTUNA_RECONSTRUCTION_PROTOCOL, f, indent=2)
        print(f"Protocol saved   : {protocol_path}")

        optuna_study = OptunaStudy(
            exp_dir=exp_dir,
            data_dir=data_dir,
            cluster=cluster,
            study_name=study_name,
            manifest_tag=exp_tag,
            study_path=study_path,
            sample_params_fn=sample_params_fn,
            train_fn=train_function_for_optuna,
            get_metrics_fn=get_metrics_for_optuna,
            optimization_metric=optimization_metric,
            optimization_direction=optimization_direction,
        )
        print("\n" + "=" * 60)
        print("Creating new Optuna study ...")
        print("=" * 60)
        optuna_study.create()
        print(
            f"\nNext step:\n"
            f"  python -m causaliT.euler_optuna.euler_optuna.cli paramsopt "
            f"--exp_id {exp_id} --study_name {study_name} --mode resume"
        )

    elif mode == "resume":
        if parallel:
            # ── Parallel (SLURM) ──────────────────────────────────────────────
            print("\n" + "=" * 60)
            print(f"Parallel execution | {n_trials} trials | {max_concurrent_jobs} concurrent")
            print("=" * 60)
            run_parallel_optuna(
                exp_dir=exp_dir,
                home_exp_dir=home_exp_dir,
                experiment_id=os.path.basename(home_exp_dir),
                study_name=study_name,
                n_trials=n_trials,
                data_dir=data_dir,
                worker_module="causaliT.euler_optuna.euler_optuna.optuna_worker",
                scratch_path=scratch_path,
                slurm_params={
                    "max_concurrent_jobs": max_concurrent_jobs,
                    "walltime": walltime,
                    "gpu_type": gpu_type,
                    "mem_per_cpu": mem_per_cpu,
                },
                cluster=cluster,
                optimization_metric=optimization_metric,
                optimization_direction=optimization_direction,
                study_path=study_path,
            )
        else:
            # ── Sequential ────────────────────────────────────────────────────
            optuna_study = OptunaStudy(
                exp_dir=exp_dir,
                data_dir=data_dir,
                cluster=cluster,
                study_name=study_name,
                manifest_tag=exp_tag,
                study_path=study_path,
                sample_params_fn=sample_params_fn,
                train_fn=train_function_for_optuna,
                get_metrics_fn=get_metrics_for_optuna,
                optimization_metric=optimization_metric,
                optimization_direction=optimization_direction,
            )
            print("\n" + "=" * 60)
            print("Resuming Optuna study (sequential) ...")
            print("=" * 60)
            try:
                optuna_study.resume()
                print("\nOptimisation complete!")
                print(
                    f"  Summary: python -m causaliT.euler_optuna.euler_optuna.cli paramsopt "
                    f"--exp_id {exp_id} --study_name {study_name} --mode summary"
                )
            except Exception as e:
                print(f"\nError during optimisation: {e}")
                raise

    elif mode == "summary":
        optuna_study = OptunaStudy(
            exp_dir=exp_dir,
            data_dir=data_dir,
            cluster=cluster,
            study_name=study_name,
            manifest_tag=exp_tag,
            study_path=study_path,
            sample_params_fn=sample_params_fn,
            train_fn=train_function_for_optuna,
            get_metrics_fn=get_metrics_for_optuna,
            optimization_metric=optimization_metric,
            optimization_direction=optimization_direction,
        )
        print("\n" + "=" * 60)
        print("Generating study summary ...")
        print("=" * 60)
        try:
            optuna_study.summary()
        except Exception as e:
            print(f"\nError generating summary: {e}")
            raise


# ── Register + entry point ────────────────────────────────────────────────────
cli.add_command(paramsopt)

if __name__ == "__main__":
    cli()
