"""
Quick smoke test: ANM staged training with AttentionSelectorLayer (scm1c).
Runs 2 epochs per stage (both stages), with eval and heavy evaluation disabled.
Exits with code 0 on success, 1 on error.

Must be run as a module entry point (if __name__ == '__main__') for Windows
multiprocessing compatibility with PyTorch DataLoader workers.
"""
import os
import sys
import multiprocessing


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    from omegaconf import OmegaConf
    from causaliT.training.anm_staged_trainer import anm_alternating_trainer

    EXP_DIR = os.path.join(
        ROOT,
        "experiments/5_EXPLORATORY/PARTIAL_ANM/"
        "H3_ANM_ATSEL_joint_vs_separate/H3_1_joint_vs_sep_scm1c",
    )
    DATA_DIR = os.path.join(ROOT, "data")
    SAVE_DIR = os.path.join(EXP_DIR, "_smoke_test")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading config …")
    cfg = OmegaConf.load(os.path.join(EXP_DIR, "config_atsel.yaml"))

    # --- Override for fast smoke test ---
    cfg.experiment.max_epochs = 2
    cfg.training.max_epochs = 2
    cfg.training.save_ckpt_every_n_epochs = 2

    for stage in cfg.anm_training.stages:
        stage["max_epochs"] = 2
        stage["eval_every_n_epochs"] = 0
        stage["eval_dag"] = False
        if "evaluation" in stage:
            del stage["evaluation"]

    print("=== Starting smoke test (2 epochs x 2 stages, scm1c, freeze_struct=True/freeze_recon=True) ===")
    try:
        df = anm_alternating_trainer(
            config=cfg,
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
            cluster=False,
            experiment_tag="smoke_test",
            debug=False,
            best=False,
        )
        print("\n=== Smoke test PASSED ===")
        print(df.to_string())
        sys.exit(0)
    except Exception:
        import traceback
        print("\n=== Smoke test FAILED ===")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
