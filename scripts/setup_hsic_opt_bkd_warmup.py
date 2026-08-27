"""Initialize the HSIC_OPT BKD-warmup arms (phase-managed batch-key dropout).

Hypothesis under test (see experiments/1_FOUNDATIONS/2_HSIC/critical/
test_D1_hsic/test_bkd_warmup_mechanism.py): a dense reconstruction warmup lets
the readout absorb the whole dependence signal, so HSIC starts the structure
phase pinned at its 1/n noise floor with ~zero directed gradient.  Heavy BKD
during reconstruct phases forces the value paths to specialize on sparse key
subsets, keeping the residual dependence (and hence the HSIC gradient on the
structure) well above the floor.

Arms (all with descendant exclusion OFF — hsic_exclude_descendants: false):

    bkd_warmup_06_nodescoff             BKD warmup + standard structure opt
    bkd_warmup_06_nodescoff_nodewise    BKD warmup + nodewise update rule
    no_bkd_warmup_nodescoff             control: constant 0.05 BKD (base), plain
    no_bkd_warmup_nodescoff_nodewise    control: constant 0.05 BKD, nodewise

BKD arms: the phase controller (adaptive_training.reconstruct.batch_key_drop*)
overrides the build-time schedule: p0=0.6 annealed to 0.0 over the whole run
(GLOBAL batch clock — the counter advances in structure phases too, where BKD
is off), active in reconstruct phases only.  Model kwargs keep
batch_key_dropout: 0.05 so the GatedCross/SelfAttention BKD machinery exists;
the controller drives the schedule (PhaseController._apply_bkd_cfg).

Annealing budget: 5000 samples -> ~4000 train -> recon/struct subsets of ~2000
at batch_size 1024 = ~2 batches/epoch/phase; the clock advances in both phases
(~4 batches/epoch), so ANNEAL_BATCHES=2000 lands p~0 around epoch ~500 of the
1000-epoch budget.  The annealed p is logged per epoch as `bkd_p` — verify
post-hoc.

Controls: unmanaged BKD (no phase block keys) -> the constant 0.05 build-time
value applies in all phases, exactly as the existing completed runs.

Launch (one run per arm):
    python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep \
        --exp_id experiments/6_INVESTIGATIONS/HSIC_OPT/<arm> \
        --sweep_mode independent [--parallel --cluster ...]

Run:  python scripts/setup_hsic_opt_bkd_warmup.py
"""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("experiments/6_INVESTIGATIONS/HSIC_OPT")
BASE = ROOT / "noNT_safeguard_dropoutsel_prior_init_05_nodesc_nodewise" / "config.yaml"

BKD_P0 = 0.6            # drop probability at run start (reconstruct phases)
BKD_P1 = 0.0            # anneal target
ANNEAL_BATCHES = 2000   # global batch clock; p~0 around epoch ~500 of 1000

ARMS = {
    "bkd_warmup_06_nodescoff":            dict(bkd=True,  nodewise=False),
    "bkd_warmup_06_nodescoff_nodewise":   dict(bkd=True,  nodewise=True),
    "no_bkd_warmup_nodescoff":            dict(bkd=False, nodewise=False),
    "no_bkd_warmup_nodescoff_nodewise":   dict(bkd=False, nodewise=True),
}



def build_arm(name: str, bkd: bool, nodewise: bool):
    cfg = OmegaConf.load(BASE)
    OmegaConf.set_struct(cfg, False)

    # --- Descendant exclusion OFF (all arms) --------------------------------
    OmegaConf.update(cfg, "training.hsic_exclude_descendants", False, merge=True)

    # --- No cross-fit split swapping (all arms) ------------------------------
    OmegaConf.update(cfg, "adaptive_training.swap_splits", False, merge=True)

    # --- Structure update rule ----------------------------------------------
    OmegaConf.update(cfg, "training.nodewise_update.enabled", nodewise, merge=True)

    # --- BKD warmup curriculum ----------------------------------------------
    if bkd:
        OmegaConf.update(cfg, "adaptive_training.reconstruct.batch_key_dropout",
                         BKD_P0, merge=True)
        OmegaConf.update(cfg, "adaptive_training.reconstruct.batch_key_dropout_final",
                         BKD_P1, merge=True)
        OmegaConf.update(
            cfg, "adaptive_training.reconstruct.batch_key_dropout_annealing_batches",
            ANNEAL_BATCHES, merge=True)
    # Controls: no phase-block keys -> PhaseController does not manage BKD and
    # the build-time constant 0.05 applies in every phase (existing behaviour).

    # --- Header ---------------------------------------------------------------
    bkd_line = (
        f"  bkd_warmup=True: phase-managed p {BKD_P0}->0.0 over "
        f"{ANNEAL_BATCHES} global batches, reconstruct phases only."
        if bkd else
        "  bkd_warmup=False: constant 0.05 build-time BKD in all phases."
    )
    header = "\n".join(
        "# " + line if line else "#"
        for line in [
            "=" * 75,
            name,
            "=" * 75,
            "BKD-warmup HSIC optimization arm (scripts/setup_hsic_opt_bkd_warmup.py).",
            bkd_line,
            f"  nodewise={nodewise}, descendant exclusion OFF.",
            "Base: noNT_safeguard_dropoutsel_prior_init_05_nodesc_nodewise/config.yaml",
            "=" * 75,
            "",
        ]
    ) + "\n"
    return cfg, header


def main():
    for name, spec in ARMS.items():
        cfg, header = build_arm(name, **spec)
        arm_dir = ROOT / name
        if arm_dir.exists():
            raise FileExistsError(f"{arm_dir} already exists — refusing to overwrite")
        arm_dir.mkdir(parents=True)
        (arm_dir / "config.yaml").write_text(header + OmegaConf.to_yaml(cfg),
                                             encoding="utf-8")
        print(f"wrote {arm_dir / 'config.yaml'}")

    # ---- Self-check: reload each config and assert the discriminating keys ----
    for name, spec in ARMS.items():
        cfg = OmegaConf.load(ROOT / name / "config.yaml")
        assert cfg.training.hsic_exclude_descendants is False, name
        assert cfg.adaptive_training.swap_splits is False, name
        assert cfg.training.nodewise_update.enabled is spec["nodewise"], name
        recon = cfg.adaptive_training.reconstruct
        struct = cfg.adaptive_training.structure
        if spec["bkd"]:
            assert recon.batch_key_dropout == BKD_P0, name
            assert recon.batch_key_dropout_final == BKD_P1, name
            assert recon.batch_key_dropout_annealing_batches == ANNEAL_BATCHES, name
        else:
            assert "batch_key_dropout" not in recon, name
        assert "batch_key_dropout" not in struct, name  # BKD off in structure
        # BKD machinery must exist at build time (controller drives the rest).
        assert cfg.model.kwargs.batch_key_dropout is not None, name
    print("\nself-check passed for all 4 arms")

    print("\nLaunch commands:")
    for name in ARMS:
        print(f"  python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep "
              f"--exp_id experiments/6_INVESTIGATIONS/HSIC_OPT/{name} "
              f"--sweep_mode independent --parallel --cluster")


if __name__ == "__main__":
    main()