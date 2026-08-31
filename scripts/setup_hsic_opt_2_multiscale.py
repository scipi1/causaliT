"""Initialize the HSIC_OPT_2 single- vs multi-bandwidth RBF arms.

Both arms inherit the d_model=20 residual-changing BKD configuration from the
HSIC_OPT long d20 run, with L0 / NOTEARS / PCGrad removed.  The only intended
contrast is the HSIC RBF bandwidth path:

* base:  legacy single adaptive median bandwidth;
* msrbf: multiscale RBF with bandwidth multipliers [0.5, 1.0, 2.0].

Run:  python scripts/setup_hsic_opt_2_multiscale.py
"""

from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path("experiments/6_INVESTIGATIONS/HSIC_OPT_2")
BASE = Path(
    "experiments/6_INVESTIGATIONS/HSIC_OPT/results/"
    "bkd_warmup_06_global_nonorm_hsicbkd_NT_L0_long_d20_11994114/config.yaml"
)
MULTIPLIERS = [0.5, 1.0, 2.0]

ARMS = {
    "bkd_warmup_06_global_nonorm_hsicbkd_d20_base": None,
    "bkd_warmup_06_global_nonorm_hsicbkd_d20_msrbf": MULTIPLIERS,
}


def build_arm(name: str, multipliers):
    cfg = OmegaConf.load(BASE)
    OmegaConf.set_struct(cfg, False)

    # Keep the objective clean: no L0, NOTEARS, gradient surgery, nodewise
    # winner-take-all, or centroid commits in either arm.
    OmegaConf.update(cfg, "training.lambda_l0", 0.0, merge=True)
    OmegaConf.update(cfg, "training.kappa", 0.0, merge=True)
    OmegaConf.update(cfg, "training.gradient_surgery", False, merge=True)
    OmegaConf.update(cfg, "training.nodewise_update.enabled", False, merge=True)
    OmegaConf.update(cfg, "training.centroid_commit.enabled", False, merge=True)
    OmegaConf.update(
        cfg, "training.hsic_bandwidth_multipliers", multipliers, merge=True
    )

    kernel_line = (
        "  HSIC kernel: single adaptive median-bandwidth RBF."
        if multipliers is None
        else f"  HSIC kernel: multiscale RBF, multipliers={multipliers}."
    )
    header = "\n".join(
        "# " + line if line else "#"
        for line in [
            "=" * 75,
            name,
            "=" * 75,
            "HSIC_OPT_2 kernel-signal arm "
            "(scripts/setup_hsic_opt_2_multiscale.py).",
            kernel_line,
            "  Base architecture/schedule: residual-changing BKD warmup, d_model=20,",
            "  frozen orthonormal keys, free normalized queries, HSIC-BKD masking.",
            "  L0=0, NOTEARS kappa=0, PCGrad off, nodewise off, centroid-commit off.",
            f"Base: {BASE.as_posix()}",
            "=" * 75,
            "",
        ]
    ) + "\n"
    return cfg, header


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, multipliers in ARMS.items():
        cfg, header = build_arm(name, multipliers)
        arm_dir = ROOT / name
        if arm_dir.exists():
            raise FileExistsError(f"{arm_dir} already exists - refusing to overwrite")
        arm_dir.mkdir(parents=True)
        (arm_dir / "config.yaml").write_text(
            header + OmegaConf.to_yaml(cfg), encoding="utf-8"
        )
        print(f"wrote {arm_dir / 'config.yaml'}")

    # Reload and pin the intended contrast/invariants.
    for name, multipliers in ARMS.items():
        cfg = OmegaConf.load(ROOT / name / "config.yaml")
        assert cfg.experiment.d_model_set == 20, name
        assert cfg.training.lambda_l0 == 0.0, name
        assert cfg.training.kappa == 0.0, name
        assert cfg.training.gradient_surgery is False, name
        assert cfg.training.nodewise_update.enabled is False, name
        assert cfg.training.centroid_commit.enabled is False, name
        assert cfg.training.hsic_bkd_exclude_dropped is True, name
        assert cfg.training.hsic_exclude_descendants is False, name
        got = cfg.training.hsic_bandwidth_multipliers
        assert (list(got) if got is not None else None) == multipliers, name
        recon = cfg.adaptive_training.reconstruct
        struct = cfg.adaptive_training.structure
        assert recon.batch_key_dropout == 0.6, name
        assert recon.batch_key_dropout_final == 0.05, name
        assert recon.batch_key_dropout_annealing_batches == 3000, name
        assert struct.batch_key_dropout == 0.6, name
        assert struct.batch_key_dropout_final == 0.05, name
        assert struct.batch_key_dropout_annealing_batches == 3000, name
    print("\nself-check passed for both HSIC_OPT_2 arms")

    print("\nLaunch commands:")
    for name in ARMS:
        print(
            "  python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep "
            f"--exp_id experiments/6_INVESTIGATIONS/HSIC_OPT_2/{name} "
            "--sweep_mode independent --parallel --cluster"
        )


if __name__ == "__main__":
    main()
