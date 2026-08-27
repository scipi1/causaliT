"""Initialize the HSIC_OPT BKD-warmup V2 arms (phase-managed batch-key dropout).

Follow-up to scripts/setup_hsic_opt_bkd_warmup.py after the v1 runs
(no_bkd_warmup_nodescoff_11754070 / bkd_warmup_06_nodescoff_11754091) showed:

  1. The warmup reconstruction never gets good enough to lift HSIC off the
     reconstruction-noise floor: the per-node output heads carry
     output_mlp_dropout=0.2 on top of the BKD key noise.  The output head
     learns a GENERIC map from any selected-subset hidden vector to the
     output; BKD already supplies all the subset stochasticity, so the
     functional dropout on the head is redundant and only caps the
     achievable fit.  -> output_mlp_dropout: 0.0 in ALL v2 arms.

  2. The v1 anneal (0.6 -> 0.0 over 2000 GLOBAL batches) spanned the whole
     1000-epoch budget: p was still ~0.5 at the end of the warmup and the
     schedule was mostly spent in structure phases (where BKD was inactive
     but the clock kept ticking).  Moreover structure phases ran with the
     mask cleared (p=0) while the regressor was trained under heavy BKD -
     a train/structure distribution shift.  -> v2 anneals 0.6 -> 0.05 over
     200 batches (~100 epochs at ~2 steps/epoch), i.e. entirely INSIDE the
     first warmup, and HOLDS p=0.05 through the structure phases (structure
     opt-in), matching the constant 0.05 of the control arm.  The only
     contrast between the arms is the warmup curriculum itself.

  3. The v1 controller ended the warmup at the first plateau wobble
     (patience 1, warmup_min_epochs 99).  -> warmup_min_epochs: 150 in ALL
     v2 arms so the plateau check can only fire after ~50 clean post-anneal
     epochs (anneal completes ~epoch 100).

The annealing step counter is a GLOBAL clock that is never reset; since the
warmup is the first phase, 200 batches complete inside it.  On cycles 2+ the
re-applied reconstruct schedule re-anchors on the global counter, so p stays
parked at 0.05 for the rest of the run (BatchConsistentKeyDropout.
set_schedule / PhaseController._apply_bkd_cfg).

Arms (descendant exclusion OFF, nodewise OFF, output_mlp_dropout 0.0):

    no_bkd_warmup_nodescoff_v2    control: constant 0.05 BKD, all phases
    bkd_warmup_06_nodescoff_v2    BKD warmup 0.6 -> 0.05 over 200 batches,
                                  then 0.05 held in structure phases too

Launch (one run per arm):
    python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep \
        --exp_id experiments/6_INVESTIGATIONS/HSIC_OPT/<arm> \
        --sweep_mode independent [--parallel --cluster ...]

Run:  python scripts/setup_hsic_opt_bkd_warmup_v2.py
"""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("experiments/6_INVESTIGATIONS/HSIC_OPT")
BASE = ROOT / "noNT_safeguard_dropoutsel_prior_init_05_nodesc_nodewise" / "config.yaml"

OUT_MLP_DROPOUT = 0.0     # per-node output heads: no functional dropout
WARMUP_MIN_EPOCHS = 150   # anneal ends ~epoch 100; >= 50 clean epochs after

BKD_P0 = 0.6            # drop probability at run start (first warmup)
BKD_P1 = 0.05           # anneal target = the control arm's constant level
ANNEAL_BATCHES = 200    # ~2 steps/epoch -> anneal done around epoch ~100,
                        # inside the first warmup (global clock == warmup
                        # steps during the first phase)

ARMS = {
    "no_bkd_warmup_nodescoff_v2":  dict(bkd=False),
    "bkd_warmup_06_nodescoff_v2":  dict(bkd=True),
}



def build_arm(name: str, bkd: bool):
    cfg = OmegaConf.load(BASE)
    OmegaConf.set_struct(cfg, False)

    # --- Descendant exclusion OFF (all arms) --------------------------------
    OmegaConf.update(cfg, "training.hsic_exclude_descendants", False, merge=True)

    # --- No cross-fit split swapping (all arms) ------------------------------
    OmegaConf.update(cfg, "adaptive_training.swap_splits", False, merge=True)

    # --- Structure update rule: plain (nodewise off) -------------------------
    OmegaConf.update(cfg, "training.nodewise_update.enabled", False, merge=True)

    # --- Functional dropout OFF on the (per-node) output head (all arms) -----
    # With per_node_output=True the forecaster is a PerNodeMLPHead fed with
    # dropout=output_mlp_dropout (attention_selector/model.py:1349-1355), so
    # this reaches the per-node regressors.  All other dropout knobs (emb,
    # qkv, attn_out, ff) stay at experiment.dropout=0.2.
    OmegaConf.update(cfg, "model.kwargs.output_mlp_dropout",
                     OUT_MLP_DROPOUT, merge=True)

    # --- Longer minimum warmup (all arms) ------------------------------------
    OmegaConf.update(cfg, "adaptive_training.reconstruct.warmup_min_epochs",
                     WARMUP_MIN_EPOCHS, merge=True)

    # --- BKD warmup curriculum ------------------------------------------------
    if bkd:
        OmegaConf.update(cfg, "adaptive_training.reconstruct.batch_key_dropout",
                         BKD_P0, merge=True)
        OmegaConf.update(cfg, "adaptive_training.reconstruct.batch_key_dropout_final",
                         BKD_P1, merge=True)
        OmegaConf.update(
            cfg, "adaptive_training.reconstruct.batch_key_dropout_annealing_batches",
            ANNEAL_BATCHES, merge=True)
        # Structure opt-in: hold the annealed floor so the structure phases
        # see the same masking statistics as the end of the warmup (and as
        # the control arm) — no dense-aggregator distribution shift.
        OmegaConf.update(cfg, "adaptive_training.structure.batch_key_dropout",
                         BKD_P1, merge=True)
    # Control: no phase-block keys -> PhaseController does not manage BKD and
    # the build-time constant 0.05 applies in every phase (v1 behaviour).

    # --- Header ---------------------------------------------------------------
    bkd_line = (
        f"  bkd_warmup=True: phase-managed p {BKD_P0}->{BKD_P1} over "
        f"{ANNEAL_BATCHES} batches (~epoch 100, inside the first warmup); "
        f"p={BKD_P1} held in structure phases."
        if bkd else
        "  bkd_warmup=False: constant 0.05 build-time BKD in all phases."
    )
    header = "\n".join(
        "# " + line if line else "#"
        for line in [
            "=" * 75,
            name,
            "=" * 75,
            "BKD-warmup HSIC optimization arm, V2 "
            "(scripts/setup_hsic_opt_bkd_warmup_v2.py).",
            bkd_line,
            "  output_mlp_dropout=0.0 (per-node heads; BKD supplies the noise),",
            f"  warmup_min_epochs={WARMUP_MIN_EPOCHS}, nodewise=False, "
            "descendant exclusion OFF.",
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
        assert cfg.training.nodewise_update.enabled is False, name
        assert cfg.model.kwargs.output_mlp_dropout == OUT_MLP_DROPOUT, name
        assert cfg.experiment.dropout == 0.2, name  # other knobs untouched
        recon = cfg.adaptive_training.reconstruct
        struct = cfg.adaptive_training.structure
        assert recon.warmup_min_epochs == WARMUP_MIN_EPOCHS, name
        if spec["bkd"]:
            assert recon.batch_key_dropout == BKD_P0, name
            assert recon.batch_key_dropout_final == BKD_P1, name
            assert recon.batch_key_dropout_annealing_batches == ANNEAL_BATCHES, name
            assert struct.batch_key_dropout == BKD_P1, name  # held in structure
        else:
            assert "batch_key_dropout" not in recon, name
            assert "batch_key_dropout" not in struct, name
        # BKD machinery must exist at build time (controller drives the rest).
        assert cfg.model.kwargs.batch_key_dropout is not None, name
    print("\nself-check passed for all v2 arms")

    print("\nLaunch commands:")
    for name in ARMS:
        print(f"  python -m causaliT.euler_sweep.euler_sweep.cli adaptivesweep "
              f"--exp_id experiments/6_INVESTIGATIONS/HSIC_OPT/{name} "
              f"--sweep_mode independent --parallel --cluster")


if __name__ == "__main__":
    main()
