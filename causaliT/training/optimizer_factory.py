"""
Optimizer Factory: Create optimizers and schedulers from config dictionaries.

Supports separate optimizer/scheduler configurations for structural and
reconstruction parameter groups in gradient-routed training.

Config Fields (under training:):
    # Reconstruction optimizer (default)
    optimizer: "adamw"              # adamw, adam, sgd, adagrad, rmsprop
    lr: 0.001
    weight_decay: 0.01
    optimizer_kwargs: {}            # extra kwargs passed to the optimizer

    # Structure optimizer (only used when use_gradient_routing: true)
    structural_optimizer: null      # null = same as optimizer
    structural_lr: null             # null = same as lr
    structural_weight_decay: null   # null = same as weight_decay
    structural_optimizer_kwargs: {} # e.g. {momentum: 0.9, nesterov: true}

    # Scheduler for structural optimizer
    structural_scheduler: null      # null, cosine_warm_restarts, step, cosine
    structural_scheduler_kwargs: {} # e.g. {T_0: 100, T_mult: 1}

    # Scheduler for reconstruction optimizer
    use_scheduler: false
    scheduler: null                 # null, reduce_on_plateau, cosine, step
    scheduler_kwargs: {}

    # Gradient noise for structural params
    structural_gradient_noise: 0.0
    structural_gradient_noise_decay: 1.0
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

# =============================================================================
# OPTIMIZER FACTORY
# =============================================================================

SUPPORTED_OPTIMIZERS = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop,
}


def make_optimizer(
    params: List[torch.nn.Parameter],
    optimizer_type: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> optim.Optimizer:
    """
    Create an optimizer from a string type and keyword arguments.

    Args:
        params: Parameters to optimize.
        optimizer_type: One of 'adamw', 'adam', 'sgd', 'adagrad', 'rmsprop'.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        extra_kwargs: Additional keyword arguments passed directly to the
                      optimizer constructor (e.g. momentum, nesterov, betas).

    Returns:
        Configured optimizer instance.
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_type}'. "
            f"Choose from: {list(SUPPORTED_OPTIMIZERS.keys())}"
        )

    cls = SUPPORTED_OPTIMIZERS[optimizer_type]
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.info(f"Creating {optimizer_type} optimizer: lr={lr}, wd={weight_decay}, extra={extra_kwargs or {}}")
    return cls(params, **kwargs)


# =============================================================================
# SCHEDULER FACTORY
# =============================================================================

def make_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: Optional[str] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    max_epochs: int = 1000,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create an LR scheduler from a string type and keyword arguments.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of None, 'cosine_warm_restarts', 'step', 'cosine',
                        'reduce_on_plateau'.
        scheduler_kwargs: Extra kwargs for the scheduler constructor.
        max_epochs: Total training epochs (used for cosine schedule).

    Returns:
        Scheduler instance or None.
    """
    if scheduler_type is None:
        return None

    scheduler_type = scheduler_type.lower()
    kwargs = dict(scheduler_kwargs or {})

    if scheduler_type == "cosine_warm_restarts":
        T_0 = kwargs.pop("T_0", 100)
        T_mult = kwargs.pop("T_mult", 1)
        eta_min = kwargs.pop("eta_min", 0.0)
        logger.info(f"CosineAnnealingWarmRestarts: T_0={T_0}, T_mult={T_mult}")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, **kwargs
        )

    elif scheduler_type == "step":
        step_size = kwargs.pop("step_size", 100)
        gamma = kwargs.pop("gamma", 0.5)
        logger.info(f"StepLR: step_size={step_size}, gamma={gamma}")
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, **kwargs
        )

    elif scheduler_type == "cosine":
        T_max = kwargs.pop("T_max", max_epochs)
        eta_min = kwargs.pop("eta_min", 0.0)
        logger.info(f"CosineAnnealingLR: T_max={T_max}, eta_min={eta_min}")
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, **kwargs
        )

    elif scheduler_type == "reduce_on_plateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.5)
        patience = kwargs.pop("patience", 10)
        logger.info(f"ReduceLROnPlateau: factor={factor}, patience={patience}")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, **kwargs
        )

    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_type}'. "
            f"Choose from: cosine_warm_restarts, step, cosine, reduce_on_plateau"
        )


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_recon_optimizer_config(training_config: dict) -> dict:
    """Extract reconstruction optimizer config from the training section."""
    return {
        "optimizer_type": training_config.get("optimizer", "adamw"),
        "lr": training_config.get("lr", 1e-3),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "extra_kwargs": training_config.get("optimizer_kwargs", {}),
    }


def get_structural_optimizer_config(training_config: dict) -> dict:
    """
    Extract structural optimizer config from the training section.
    Falls back to reconstruction optimizer settings when structural-specific
    values are null/missing.
    """
    recon = get_recon_optimizer_config(training_config)
    return {
        "optimizer_type": training_config.get("structural_optimizer") or recon["optimizer_type"],
        "lr": training_config.get("structural_lr") or recon["lr"],
        "weight_decay": (
            training_config.get("structural_weight_decay")
            if training_config.get("structural_weight_decay") is not None
            else recon["weight_decay"]
        ),
        "extra_kwargs": training_config.get("structural_optimizer_kwargs", {}),
    }


def get_structural_scheduler_config(training_config: dict) -> dict:
    """Extract structural scheduler config from the training section."""
    return {
        "scheduler_type": training_config.get("structural_scheduler"),
        "scheduler_kwargs": training_config.get("structural_scheduler_kwargs", {}),
    }


def get_gradient_noise_config(training_config: dict) -> dict:
    """Extract gradient noise config for structural parameters."""
    return {
        "noise_std": training_config.get("structural_gradient_noise", 0.0),
        "noise_decay": training_config.get("structural_gradient_noise_decay", 1.0),
    }
