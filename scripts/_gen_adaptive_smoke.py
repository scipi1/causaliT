from pathlib import Path

src_path = Path("experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/lagrangian_l0_e5_nhsic_gt/config.yaml")
out_dir = Path("experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_smoke")
out_dir.mkdir(parents=True, exist_ok=True)
src = src_path.read_text(encoding="utf-8")

# Header title
src = src.replace(
    "# HSIC_CONSTRAINT / lagrangian_l0_e5_nhsic_gt",
    "# HSIC_CONSTRAINT / adaptive_nhsic_ladder_smoke")
src = src.replace(
    "python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/lagrangian_l0_e5_nhsic_gt",
    "python -m causaliT.cli train --exp_id 6_INVESTIGATIONS/HSIC_CONSTRAINT/adaptive_nhsic_ladder_smoke")

# Dataset -> small additive nonlinear SCM (5S + 5X, 8 edges)
src = src.replace("dataset: random_n20_k4_er_nonlinear_gaussian_s1",
                  "dataset: scm2_continuous")
src = src.replace("train_file: null", "train_file: ds_train.npz")
src = src.replace("test_file: null", "test_file: ds_test.npz")

# Node counts
src = src.replace("n_nodes: 20", "n_nodes: 10")
src = src.replace("n_source: 2", "n_source: 5")
src = src.replace("n_input: 18", "n_input: 5")

# Budget
src = src.replace("max_epochs: 10000", "max_epochs: 800")

# Model must own BKD modules (non-null init); the ladder overrides per cycle.
src = src.replace("batch_key_dropout: 0.05", "batch_key_dropout: 0.8")
src = src.replace("batch_key_dropout_p_final: 0.05", "batch_key_dropout_p_final: 0.8")

# Tolerance: mid bracket for the smoke (recalibrate per-dataset later).
src = src.replace("tolerance: 0.11", "tolerance: 0.075")

adaptive = '''
adaptive_training:
  total_epoch_budget: 800
  start_phase: warmup
  max_cycles: null
  starting_checkpoint: null
  reset_optimizer_state_on_switch: false
  monitor: val_x_mae
  eval_dag: true
  run_final_evaluations: true
  data_split_ratio: 0.5
  swap_splits: true
  # BKD ladder: constant p within a warmup/reconstruct + structure cycle,
  # discrete decrease per cycle.  Warmup = rung 0.
  bkd_ladder: [0.8, 0.6, 0.4, 0.2, 0.0]
  warmup:
    enabled: true
    max_epochs: 150
  reconstruct:
    max_epochs: 50
    min_epochs: 10
    warmup_min_epochs: 0
    plateau_patience: 1
    plateau_min_delta: 1.0e-4
  structure:
    max_epochs: 100
    min_epochs: 10
    drop_pct: 0.20
    drop_patience: 1
    lambda_l0: 1.0e-5
    hsic_monitor: val_hsic
    hsic_patience: 1
    hsic_min_delta: 1.0e-4
'''
src = src.rstrip() + "\n" + adaptive

(out_dir / "config.yaml").write_text(src, encoding="utf-8")
print("wrote", out_dir / "config.yaml")

import yaml
c = yaml.safe_load(src)
assert c["data"]["dataset"] == "scm2_continuous"
assert c["training"]["hsic_constraint"]["enabled"] is True
assert c["training"]["hsic_mode"] == "normalized"
assert c["training"]["lambda_hsic"] == 0.0
assert c["adaptive_training"]["bkd_ladder"] == [0.8, 0.6, 0.4, 0.2, 0.0]
assert c["adaptive_training"]["start_phase"] == "warmup"
assert "lambda_hsic" not in c["adaptive_training"]["structure"]
print("config validated")
