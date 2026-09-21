"""Short local smoke of the two TOP_K centroid-commit arms (in-memory overrides).

Runs the REAL adaptive trainer end-to-end (warmup -> alternating 2/1-epoch
reconstruct/structure cycles) for the first-order-shadow arm and the DARTS
unrolled-shadow arm: gradient routing + PCGrad + centroid commit + top-k +
cross-fit HSIC.  The config files on disk are untouched.
"""
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
ROOT = Path(r"c:\Users\ScipioneFrancesco\Documents\Projects\causaliT")
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402

from causaliT.training.config_utils import populate_seq_lengths_from_dataset  # noqa: E402
from causaliT.training.experiment_control import update_config  # noqa: E402
from causaliT.training.adaptive_trainer import adaptive_trainer  # noqa: E402

ARMS = [
    "topk_noisy_ccommit_pcgrad_l0_notears",
]

for arm in ARMS:
    cfg = OmegaConf.load(
        ROOT / f"experiments/6_INVESTIGATIONS/TOP_K/{arm}/config.yaml")
    cfg = populate_seq_lengths_from_dataset(cfg, str(ROOT / "data"))
    cfg = update_config(cfg)

    # Short-run overrides (in-memory only)
    cfg.experiment.max_epochs = 14
    cfg.training.max_epochs = 14
    cfg.adaptive_training.total_epoch_budget = 14
    cfg.adaptive_training.warmup.max_epochs = 4
    cfg.adaptive_training.warmup.batch_key_dropout_annealing_batches = 12
    cfg.adaptive_training.eval_dag = False
    cfg.adaptive_training.run_final_evaluations = False
    cfg.training.dag_metrics_every_n_epochs = 3
    cfg.training.save_ckpt_every_n_epochs = 5

    save_dir = ROOT / f"experiments/6_INVESTIGATIONS/TOP_K/results/_smoke_{arm}"
    print(f"\n================ SMOKE {arm} ================")
    df = adaptive_trainer(config=cfg, data_dir=str(ROOT / "data"),
                          save_dir=str(save_dir), cluster=False,
                          experiment_tag=f"smoke_{arm}")
    print("SMOKE phases:")
    print(df[["phase_index", "phase", "end_reason", "global_epoch_end",
              "phase_epochs"]].to_string(index=False))

print("\nALL SMOKE RUNS COMPLETED")
