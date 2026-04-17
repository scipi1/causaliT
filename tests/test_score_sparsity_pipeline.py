"""
Test suite for the score sparsity / calibrated-sweep pipeline.

Covers:
1. calibrate_group_l1             (Stage 0)
2. run_causal_initialization       (Stage 1)
3. run_sequential_sweep + pre_sweep_fn  (calibrated_sweep end-to-end)

All tests use the NoiseAwareSingleCausalLayer template
(causaliT/config/templates/config_noise_aware.yaml) with scm1 as dataset.

Speed overrides applied throughout:
- calibration_epochs = 3
- calibration_max_iterations = 3
- causal_init_epochs = 3
- training.max_epochs = 1
- training.k_fold = 1
- training.batch_size = 500

Run with:
    pytest tests/test_score_sparsity_pipeline.py -v
    pytest tests/test_score_sparsity_pipeline.py -v -k "test_calibration"
"""

import copy
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"
TEMPLATE_PATH = project_root / "causaliT" / "config" / "templates" / "config_noise_aware.yaml"

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def data_dir() -> str:
    return str(DATA_DIR)


@pytest.fixture(scope="session")
def base_config() -> dict:
    """
    Load the noise-aware template and apply minimal overrides for fast tests.

    The config is returned as a plain dict (OmegaConf resolved) so it can be
    freely mutated per test without shared-state issues.
    """
    cfg = OmegaConf.load(TEMPLATE_PATH)

    # ── Dataset ──────────────────────────────────────────────────────────────
    cfg.data.dataset = "scm1"

    # ── Fast training ─────────────────────────────────────────────────────────
    cfg.training.max_epochs = 1
    cfg.training.k_fold = 1
    cfg.training.batch_size = 500
    cfg.training.save_ckpt_every_n_epochs = 1
    cfg.training.seed = 42

    # ── Enable HSIC (calibration needs a non-zero HSIC signal) ───────────────
    cfg.training.lambda_hsic_cross = 0.1
    cfg.training.lambda_hsic_self = 0.0
    cfg.training.normalize_hsic_by_loss = True

    # ── Staged training: fast calibration + causal init ───────────────────────
    cfg.staged_training.use_calibration = True
    cfg.staged_training.use_causal_init = True
    cfg.staged_training.calibration_epochs = 3
    cfg.staged_training.calibration_max_iterations = 3
    cfg.staged_training.calibration_lambda_group_range = [1.0e-4, 1.0e-1]
    cfg.staged_training.calibration_balance_threshold = 2.0
