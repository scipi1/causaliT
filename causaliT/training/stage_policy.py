"""
Stage policy of the order-factorized selector (single source of truth).

Schema (``adaptive_training``)::

    recon:
      gate_opt: gated | softmax | softmax_prenorm | none   # what G_R IS
      gate_sampling: hard | soft            # ST Bernoulli | Hard-Concrete
      order_sampling: none | hard | soft | oriented
      oriented: {p_anticausal: 0.5, noise_scale: 1.0}   # order_sampling=oriented
    structure:
      gate_opt: gated | directional_free | none  # G_S architecture
      gate_sampling: soft | hard
      order_sampling: soft | hard | hard_st | none

``gate_opt`` decides which parameters exist and is global.  The recon
sampling keys can be overridden per reconstruction phase block
(``warmup`` / ``reconstruct`` / ``final_reconstruct``) with the same key
names.  Removed keys raise with the replacement.
"""

from typing import Any, Dict, Optional

RECON_GATE_OPTS = ("gated", "softmax", "softmax_prenorm", "none")
STRUCT_GATE_OPTS = ("gated", "directional_free", "none")
GATE_SAMPLING = ("hard", "soft")
RECON_ORDER_SAMPLING = ("none", "hard", "soft", "oriented")
STRUCT_ORDER_SAMPLING = ("soft", "hard", "hard_st", "none")
RECON_PHASES = ("warmup", "reconstruct", "final_reconstruct")

RECON_DEFAULTS: Dict[str, Any] = {
    "gate_opt": "gated",
    "gate_sampling": "hard",
    "order_sampling": "none",
    "oriented": {"p_anticausal": 0.5, "noise_scale": 1.0},
}
STRUCT_DEFAULTS: Dict[str, Any] = {
    "gate_opt": "none",
    "gate_sampling": "soft",
    "order_sampling": "soft",
}
_RECON_SAMPLING_KEYS = ("gate_sampling", "order_sampling", "oriented")

# Removed keys -> replacement (no backward compatibility).
_REMOVED = {
    "recon_gates": "recon.gate_sampling",
    "use_order_in_recon": "recon.order_sampling",
    "recon_order_sampling": "recon.order_sampling: oriented + recon.oriented",
    "struct_gates": "structure.gate_sampling",
}


def _plain(x: Any) -> Dict[str, Any]:
    """Shallow plain dict of a mapping / OmegaConf node (None -> {})."""
    if x is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(x, DictConfig):
            return dict(OmegaConf.to_container(x, resolve=True))
    except ImportError:  # pragma: no cover
        pass
    return dict(x)


def _check_removed(block: Dict[str, Any], where: str) -> None:
    for old, new in _REMOVED.items():
        if old in block:
            raise ValueError(f"{where}.{old} is not a key: use adaptive_training.{new}.")


def _check(value: Any, allowed: tuple, where: str) -> str:
    if value not in allowed:
        raise ValueError(f"{where} must be one of {allowed}, got {value!r}.")
    return str(value)


def _check_oriented(o: Dict[str, Any], where: str) -> Dict[str, float]:
    unknown = set(o) - {"p_anticausal", "noise_scale"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}.")
    p, s = float(o["p_anticausal"]), float(o["noise_scale"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{where}.p_anticausal must be in [0, 1], got {p!r}.")
    if s < 0.0:
        raise ValueError(f"{where}.noise_scale must be >= 0, got {s!r}.")
    return {"p_anticausal": p, "noise_scale": s}



def parse_stage_policy(ad: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate the ``adaptive_training`` stage policy.

    Returns ``{"recon": {...}, "structure": {...}, "phases": {phase: {...}}}``;
    ``phases`` holds the effective recon sampling of every reconstruction
    phase (``final_reconstruct`` inherits ``reconstruct``).
    """
    ad = _plain(ad)
    _check_removed(ad, "adaptive_training")
    recon_in = _plain(ad.get("recon"))
    struct_in = _plain(ad.get("structure"))
    _check_removed(recon_in, "adaptive_training.recon")
    _check_removed(struct_in, "adaptive_training.structure")
    unknown = set(recon_in) - set(RECON_DEFAULTS)
    if unknown:
        raise ValueError(f"adaptive_training.recon: unknown keys {sorted(unknown)}.")

    recon = {**RECON_DEFAULTS, **recon_in}
    recon["gate_opt"] = _check(recon["gate_opt"], RECON_GATE_OPTS,
                               "adaptive_training.recon.gate_opt")
    recon["gate_sampling"] = _check(recon["gate_sampling"], GATE_SAMPLING,
                                    "adaptive_training.recon.gate_sampling")
    recon["order_sampling"] = _check(recon["order_sampling"], RECON_ORDER_SAMPLING,
                                     "adaptive_training.recon.order_sampling")
    recon["oriented"] = _check_oriented(
        {**RECON_DEFAULTS["oriented"], **_plain(recon_in.get("oriented"))},
        "adaptive_training.recon.oriented")

    # ``structure`` is also the structure phase block: read the policy keys only.
    structure = {k: struct_in.get(k, v) for k, v in STRUCT_DEFAULTS.items()}
    where = "adaptive_training.structure"
    structure["gate_opt"] = _check(structure["gate_opt"], STRUCT_GATE_OPTS, f"{where}.gate_opt")
    structure["gate_sampling"] = _check(structure["gate_sampling"], GATE_SAMPLING,
                                        f"{where}.gate_sampling")
    structure["order_sampling"] = _check(structure["order_sampling"], STRUCT_ORDER_SAMPLING,
                                         f"{where}.order_sampling")
    # Both gate_opt 'none' is allowed: order-only ablation.

    blocks = {p: _plain(ad.get(p)) for p in RECON_PHASES}
    blocks["final_reconstruct"] = {**blocks["reconstruct"], **blocks["final_reconstruct"]}
    phases: Dict[str, Dict[str, Any]] = {}
    for phase in RECON_PHASES:
        blk, where = blocks[phase], f"adaptive_training.{phase}"
        _check_removed(blk, where)
        if "gate_opt" in blk:
            raise ValueError(f"{where}.gate_opt: gate_opt is global (adaptive_training.recon).")
        phases[phase] = {
            "gate_sampling": _check(blk.get("gate_sampling", recon["gate_sampling"]),
                                    GATE_SAMPLING, f"{where}.gate_sampling"),
            "order_sampling": _check(blk.get("order_sampling", recon["order_sampling"]),
                                     RECON_ORDER_SAMPLING, f"{where}.order_sampling"),
            "oriented": _check_oriented({**recon["oriented"], **_plain(blk.get("oriented"))},
                                        f"{where}.oriented"),
        }
    return {"recon": recon, "structure": structure, "phases": phases}


def resolve_gate_opts(config: Dict[str, Any]) -> Dict[str, str]:
    """Model kwargs ``{recon_gate_opt, struct_gate_opt}`` from a full config."""
    pol = parse_stage_policy(_plain(config).get("adaptive_training"))
    return {"recon_gate_opt": pol["recon"]["gate_opt"],
            "struct_gate_opt": pol["structure"]["gate_opt"]}
