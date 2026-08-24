"""Restructure the BKD dose-response as an adaptivesweep: one config + one sweep.yaml.

The sweeper applies parameters as ``config[category][param] = value`` (one level
deep), so the nested ``model.kwargs.batch_key_dropout`` is exposed as an
``experiment.``-level knob interpolated into ``model.kwargs`` — the pattern the
config already uses everywhere, and exactly what the sweeper's "update params
BEFORE resolving interpolations" flow is designed for.

Produces::

    bkd_sweep/
    ├── config.yaml          # base config (no_NT + the experiment-level BKD knob)
    └── sweeper/
        └── sweep.yaml       # experiment.batch_key_dropout: [0.0, 0.05, 0.1, 0.2, 0.3]

Launch (adaptive trainer, one run per BKD level):
    python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep \
        --exp_id experiments/6_INVESTIGATIONS/LARGER_DAGS/bkd_sweep \
        --sweep_mode independent [--parallel --cluster ...]

Run:  python scripts/setup_larger_dags_bkd_sweep.py
"""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("experiments/6_INVESTIGATIONS/LARGER_DAGS")
SWEEP_DIR = ROOT / "bkd_sweep"
DATA_ROOT = str(Path("data").resolve())   # the shared datasets dir

BKD_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]


def main():
    # ---- Base config: no_NT + the experiment-level BKD knob ---------------------
    cfg = OmegaConf.load(ROOT / "no_NT" / "config.yaml")
    OmegaConf.set_struct(cfg, False)

    # The swept knob (a plain number; 0.0 = a no-op BKD module, i.e. no_NT).
    OmegaConf.update(cfg, "experiment.batch_key_dropout", 0.0, merge=True)
    # Interpolated into model.kwargs: the sweeper updates the knob BEFORE
    # resolving interpolations, so every combination picks up its level.
    OmegaConf.update(cfg, "model.kwargs.batch_key_dropout",
                     "${experiment.batch_key_dropout}", merge=True)
    OmegaConf.update(cfg, "model.kwargs.batch_key_dropout_p_final",
                     "${experiment.batch_key_dropout}", merge=True)
    OmegaConf.update(cfg, "model.kwargs.batch_key_dropout_annealing_batches",
                     None, merge=True)
    OmegaConf.update(cfg, "data.data_root", DATA_ROOT, merge=True)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, SWEEP_DIR / "config.yaml")
    print(f"wrote {SWEEP_DIR / 'config.yaml'}")

    # ---- Sweep definition ---------------------------------------------------------
    sweep = OmegaConf.create({"experiment": {"batch_key_dropout": BKD_LEVELS}})
    (SWEEP_DIR / "sweeper").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(sweep, SWEEP_DIR / "sweeper" / "sweep.yaml")
    print(f"wrote {SWEEP_DIR / 'sweeper' / 'sweep.yaml'}")

    # ---- Verify: the combinations generate correctly -------------------------------
    from causaliT.euler_sweep.euler_sweep.sweeper import (
        find_config_files,
        generate_independent_combinations,
    )
    config, sweep_config = find_config_files(str(SWEEP_DIR))
    combos = generate_independent_combinations(config, sweep_config,
                                               experiment_id="bkd_sweep")
    print(f"\n{len(combos)} combinations:")
    for c in combos:
        print(f"  {c['description']}  ->  {c['name']}")


if __name__ == "__main__":
    main()
