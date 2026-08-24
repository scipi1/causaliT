"""Generate the configs for the two new LARGER_DAGS experiments.

Both are built from the proven arm configs (same dataset
``random_n20_k4_er_nonlinear_gaussian_s1``, now in ``data/``), changing only the
relevant knobs:

1. ``no_NT_bkd_02_nodesc_budget`` — the no_NT_bkd_02_nodesc arm with the NEW
   budgeted descendant mask (``hsic_descendant_mode: budget``): the threshold
   becomes a cap on how many pairs are removed, so the mask triggers every
   step instead of collapsing.  This is the arm that actually tests the
   descendant exclusion (the threshold-mode nodesc arm never fired).

2. ``bkd_sweep/no_NT_bkd_<level>`` — a batch-key-dropout dose-response on top of
   no_NT, to compute the HSIC dilution per dropout level.  Levels
   {0.0, 0.05, 0.1, 0.2, 0.3}; 0.0 (``null`` = no BKD module) reproduces no_NT
   and 0.1 reproduces no_NT_bkd_02, so the grid is self-contained.

Run:  python scripts/setup_larger_dags_new_arms.py
"""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("experiments/6_INVESTIGATIONS/LARGER_DAGS")
DATA_ROOT = str(Path("data").resolve())   # the shared datasets dir


def load(name):
    return OmegaConf.load(ROOT / name / "config.yaml")


def save(cfg, name):
    out = ROOT / name / "config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out)
    print(f"wrote {out}")


def set_data_root(cfg):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "data.data_root", DATA_ROOT, merge=True)
    return cfg


def main():
    # ---- 1. Budgeted descendants (on no_NT_bkd_02_nodesc) ----------------------
    cfg = load("no_NT_bkd_02_nodesc")
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "training.hsic_descendant_mode", "budget", merge=True)
    OmegaConf.update(cfg, "training.hsic_descendant_budget_frac", 0.25, merge=True)
    OmegaConf.update(cfg, "training.hsic_descendant_per_row", True, merge=True)
    OmegaConf.update(cfg, "training.hsic_descendant_tnorm", "min", merge=True)
    # The cap bounds the self-confirmation risk, so the mask can be always on.
    OmegaConf.update(cfg, "training.hsic_descendant_warmup_epochs", 0, merge=True)
    set_data_root(cfg)
    save(cfg, "no_NT_bkd_02_nodesc_budget")

    # ---- 2. BKD dose-response sweep (on no_NT) ----------------------------------
    for level in [0.0, 0.05, 0.1, 0.2, 0.3]:
        cfg = load("no_NT")
        OmegaConf.set_struct(cfg, False)
        # 0.0 -> null: no BKD module at all (exactly the no_NT arm).
        value = None if level == 0.0 else float(level)
        OmegaConf.update(cfg, "model.kwargs.batch_key_dropout", value, merge=True)
        OmegaConf.update(cfg, "model.kwargs.batch_key_dropout_p_final", value,
                         merge=True)
        OmegaConf.update(cfg, "model.kwargs.batch_key_dropout_annealing_batches",
                         None, merge=True)
        set_data_root(cfg)
        tag = f"{level:.2f}".replace(".", "p")
        save(cfg, f"bkd_sweep/no_NT_bkd_{tag}")

    print("\nDone. Launch with the sweep machinery; the dataset is in data/.")


if __name__ == "__main__":
    main()
