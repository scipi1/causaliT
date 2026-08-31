"""Short local validation run of the warmcos arm (in-memory overrides)."""
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
ROOT = Path(r"c:\Users\ScipioneFrancesco\Documents\Projects\causaliT")
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from causaliT.training.config_utils import populate_seq_lengths_from_dataset
from causaliT.training.experiment_control import update_config

cfg = OmegaConf.load(ROOT / "experiments/6_INVESTIGATIONS/HSIC_OPT_2/bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw_warmcos/config.yaml")
cfg = populate_seq_lengths_from_dataset(cfg, str(ROOT / "data"))
cfg = update_config(cfg)

# Short-run overrides (in-memory only; the file is untouched)
cfg.model.kwargs.device = "cpu"
cfg.experiment.max_epochs = 22
cfg.adaptive_training.total_epoch_budget = 22
cfg.adaptive_training.warmup.max_epochs = 8
cfg.adaptive_training.warmup.batch_key_dropout_annealing_batches = 24
cfg.adaptive_training.warmup.bkd_cycles = 2
cfg.adaptive_training.reconstruct.min_epochs = 2
cfg.adaptive_training.reconstruct.max_epochs = 6
cfg.adaptive_training.reconstruct.warmup_min_epochs = 2
cfg.adaptive_training.structure.min_epochs = 100
cfg.adaptive_training.structure.max_epochs = 100
cfg.adaptive_training.eval_dag = False
cfg.adaptive_training.run_final_evaluations = False

save_dir = ROOT / "experiments/6_INVESTIGATIONS/HSIC_OPT_2/results/_warmcos_smoke"
from causaliT.training.adaptive_trainer import adaptive_trainer

df = adaptive_trainer(config=cfg, data_dir=str(ROOT / "data"), save_dir=str(save_dir),
                      cluster=False, experiment_tag="warmcos_smoke")
print("SMOKE phases:")
print(df[["phase_index", "phase", "end_reason", "global_epoch_end", "phase_epochs"]].to_string(index=False))
