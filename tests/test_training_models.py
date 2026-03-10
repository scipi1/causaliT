"""
Test suite for verifying training works across all model configurations.

Run with: pytest tests/test_training_models.py -v
Run single model: pytest tests/test_training_models.py -v -k "single_Lie_CC_scm6"
Custom directory: pytest tests/test_training_models.py --models-dir=experiments/single/scm6

Standalone usage:
    python tests/test_training_models.py --list
    python tests/test_training_models.py --model single_Lie_CC_scm6
    python tests/test_training_models.py --models-dir experiments/single/scm6 --list

Environment variable:
    CAUSALT_MODELS_DIR=experiments/single/scm6 pytest tests/test_training_models.py -v

This suite validates:
1. All models in a directory can successfully train
2. Training completes without errors
3. Expected outputs are produced
4. Original config files are NOT modified
"""

import pytest
import os
import sys
import shutil
import hashlib
import tempfile
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from causaliT.training.trainer import trainer
from causaliT.training.experiment_control import find_yml_files, update_config


# ============================================================================
# Configuration
# ============================================================================

# Default debug training parameters (override for fast testing)
DEBUG_TRAINING_OVERRIDES = {
    "max_epochs": 1,
    "k_fold": 3,  # Minimum for meaningful cross-validation (sklearn allows 2, but 3 is safer)
    "batch_size": 500,
    "save_ckpt_every_n_epochs": 1,
}

# Default paths (can be overridden via env var, pytest option, or CLI argument)
DEFAULT_MODELS_DIR = Path(os.environ.get(
    "CAUSALT_MODELS_DIR", 
    str(project_root / "experiments")
))
DEFAULT_DATA_DIR = Path(os.environ.get(
    "CAUSALT_DATA_DIR",
    str(project_root / "data")
))

# Global variable to store pytest option value (set by conftest.py)
_pytest_models_dir = None

def get_models_directory() -> Path:
    """Get the models directory from various sources (in priority order)."""
    # 1. Pytest option (set via conftest.py hook)
    if _pytest_models_dir is not None:
        return Path(_pytest_models_dir)
    # 2. Environment variable
    if "CAUSALT_MODELS_DIR" in os.environ:
        return Path(os.environ["CAUSALT_MODELS_DIR"])
    # 3. Default
    return DEFAULT_MODELS_DIR


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def models_directory() -> Path:
    """Directory containing model configurations to test."""
    return get_models_directory()


@pytest.fixture(scope="module")
def data_directory() -> Path:
    """Directory containing training data."""
    return DEFAULT_DATA_DIR


def discover_model_configs(models_dir: Path, recursive: bool = False) -> List[Tuple[str, Path]]:
    """
    Discover all model configuration directories in the given directory.
    
    Args:
        models_dir: Base directory containing model subdirectories
        recursive: If True, search recursively through all subdirectories
        
    Returns:
        List of tuples: (model_name, config_dir_path)
    """
    configs = []
    
    if not models_dir.exists():
        return configs
    
    # Directories to skip
    skip_dirs = {'euler', 'sweeps', 'combinations', 'report', 'tests', 'stage'}
    
    if recursive:
        # Recursive search: find all directories containing config files
        for config_file in models_dir.rglob("config*.yaml"):
            config_dir = config_file.parent
            # Skip if in a skip directory
            if any(skip_dir in config_dir.parts for skip_dir in skip_dirs):
                continue
            if config_dir.name.startswith('__'):
                continue
            # Use relative path from models_dir as identifier
            rel_path = config_dir.relative_to(models_dir)
            model_name = str(rel_path).replace(os.sep, '/')
            configs.append((model_name, config_dir))
    else:
        # Non-recursive: only look one level deep
        for item in models_dir.iterdir():
            if item.is_dir():
                # Skip non-model directories (like 'euler', '__pycache__', etc.)
                if item.name.startswith('__') or item.name in skip_dirs:
                    continue
                
                # Check if directory contains a config file
                config_files = list(item.glob("config*.yaml"))
                if config_files:
                    configs.append((item.name, item))
    
    return sorted(configs, key=lambda x: x[0])


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file for integrity checking."""
    if not filepath.exists():
        return ""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_config_file_path(config_dir: Path) -> Optional[Path]:
    """Find the config file in a directory."""
    config_files = list(config_dir.glob("config*.yaml"))
    if config_files:
        return config_files[0]
    return None


# ============================================================================
# Parametrized Test Discovery
# ============================================================================

def pytest_generate_tests(metafunc):
    """
    Dynamically generate test cases for each model configuration found.
    This allows each model to appear as a separate test case in pytest output.
    """
    if "model_config" in metafunc.fixturenames:
        models_dir = get_models_directory()
        configs = discover_model_configs(models_dir)
        if configs:
            metafunc.parametrize(
                "model_config",
                configs,
                ids=[name for name, _ in configs]
            )
        else:
            # No configs found, create a skip marker
            metafunc.parametrize(
                "model_config",
                [pytest.param(None, marks=pytest.mark.skip(reason="No model configs found"))]
            )


# ============================================================================
# Test Functions
# ============================================================================

class TestTrainingModels:
    """Tests for verifying training works across all model configurations."""
    
    def test_training_single_model(
        self, 
        model_config: Tuple[str, Path], 
        data_directory: Path,
        tmp_path: Path
    ):
        """
        Test that training completes successfully for a single model configuration.
        
        This test:
        1. Loads the config without modifying the original file
        2. Overrides training parameters for fast debugging (1 epoch, 3 folds)
        3. Runs the trainer
        4. Verifies expected outputs are created
        5. Ensures original config was not modified
        
        Args:
            model_config: Tuple of (model_name, config_directory)
            data_directory: Path to data directory
            tmp_path: Pytest's temporary directory (auto-cleaned)
        """
        model_name, config_dir = model_config
        
        # Get path to original config file
        original_config_path = get_config_file_path(config_dir)
        assert original_config_path is not None, f"No config file found in {config_dir}"
        
        # Store hash of original file for integrity check
        original_hash = get_file_hash(original_config_path)
        
        # Load config using the standard method (creates a copy)
        try:
            config, sweep_config = find_yml_files(str(config_dir))
        except Exception as e:
            pytest.fail(f"Failed to load config for {model_name}: {e}")
        
        # Convert to container and back to ensure we have an independent copy
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config_dict)
        
        # Apply debug training overrides
        config = apply_debug_overrides(config, DEBUG_TRAINING_OVERRIDES)
        
        # Update config (handles any dynamic references)
        config = update_config(config)
        
        # Create save directory in temp location
        save_dir = tmp_path / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config file to save directory (needed for post-training evaluations)
        config_dest = save_dir / original_config_path.name
        shutil.copy(original_config_path, config_dest)
        
        # Run training
        try:
            df_metrics = trainer(
                config=config,
                data_dir=str(data_directory),
                save_dir=str(save_dir),
                cluster=False,
                experiment_tag="test",
                resume_ckpt=None,
                plot_pred_check=False,  # Skip prediction plots for speed
                debug=True,  # Enable debug mode
            )
        except Exception as e:
            pytest.fail(f"Training failed for {model_name}: {e}")
        
        # Verify training completed successfully
        assert df_metrics is not None, f"Trainer returned None for {model_name}"
        assert len(df_metrics) > 0, f"No metrics returned for {model_name}"
        
        # Verify expected fold directories were created
        expected_folds = DEBUG_TRAINING_OVERRIDES["k_fold"]
        for fold in range(expected_folds):
            fold_dir = save_dir / f"k_{fold}"
            assert fold_dir.exists(), f"Fold directory k_{fold} not created for {model_name}"
        
        # CRITICAL: Verify original config was NOT modified
        final_hash = get_file_hash(original_config_path)
        assert original_hash == final_hash, (
            f"Original config file was modified during training for {model_name}! "
            f"Hash changed from {original_hash} to {final_hash}"
        )
        
        print(f"✓ Training completed successfully for {model_name}")
        print(f"  Metrics: {df_metrics.columns.tolist()}")


class TestTrainingAllModels:
    """Test that runs training for ALL models in directory sequentially."""
    
    def test_training_all_models_in_directory(
        self,
        models_directory: Path,
        data_directory: Path,
        tmp_path: Path
    ):
        """
        Test training for all models in the specified directory.
        
        This is an alternative to parametrized tests - runs all models
        in a single test, useful for CI/CD where you want a single pass/fail.
        """
        configs = discover_model_configs(models_directory)
        
        if not configs:
            pytest.skip(f"No model configurations found in {models_directory}")
        
        failed_models = []
        successful_models = []
        
        for model_name, config_dir in configs:
            original_config_path = get_config_file_path(config_dir)
            if original_config_path is None:
                failed_models.append((model_name, "No config file found"))
                continue
            
            original_hash = get_file_hash(original_config_path)
            
            try:
                # Load and prepare config
                config, _ = find_yml_files(str(config_dir))
                config_dict = OmegaConf.to_container(config, resolve=True)
                config = OmegaConf.create(config_dict)
                config = apply_debug_overrides(config, DEBUG_TRAINING_OVERRIDES)
                config = update_config(config)
                
                # Setup save directory
                save_dir = tmp_path / model_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy config file to save directory (needed for post-training evaluations)
                config_dest = save_dir / original_config_path.name
                shutil.copy(original_config_path, config_dest)
                
                # Run training
                df_metrics = trainer(
                    config=config,
                    data_dir=str(data_directory),
                    save_dir=str(save_dir),
                    cluster=False,
                    experiment_tag="test",
                    resume_ckpt=None,
                    plot_pred_check=False,
                    debug=True,
                )
                
                # Verify integrity
                final_hash = get_file_hash(original_config_path)
                if original_hash != final_hash:
                    failed_models.append((model_name, "Config file was modified"))
                    continue
                
                if df_metrics is None or len(df_metrics) == 0:
                    failed_models.append((model_name, "No metrics returned"))
                    continue
                
                successful_models.append(model_name)
                print(f"✓ {model_name}")
                
            except Exception as e:
                failed_models.append((model_name, str(e)))
                print(f"✗ {model_name}: {e}")
        
        # Report results
        print(f"\n{'='*60}")
        print(f"Training Test Results:")
        print(f"  Successful: {len(successful_models)}/{len(configs)}")
        print(f"  Failed: {len(failed_models)}/{len(configs)}")
        
        if failed_models:
            print(f"\nFailed models:")
            for name, error in failed_models:
                print(f"  - {name}: {error}")
        
        assert len(failed_models) == 0, f"{len(failed_models)} models failed training"


# ============================================================================
# Helper Functions
# ============================================================================

def apply_debug_overrides(config: OmegaConf, overrides: dict) -> OmegaConf:
    """
    Apply debug/fast-training overrides to a configuration.
    
    Args:
        config: Original configuration (OmegaConf object)
        overrides: Dictionary of training parameters to override
        
    Returns:
        Modified configuration (original is not modified)
    """
    # Create a mutable copy if structured
    if OmegaConf.is_readonly(config):
        config = OmegaConf.create(OmegaConf.to_container(config))
    
    # Apply overrides to training section
    if "training" in config:
        for key, value in overrides.items():
            if key in config.training:
                config.training[key] = value
    
    return config


# ============================================================================
# Standalone Test Runner
# ============================================================================

def run_single_model_test(
    model_name: str,
    models_dir: Path = DEFAULT_MODELS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    cleanup: bool = True
) -> bool:
    """
    Run training test for a single model (for manual testing/debugging).
    
    Args:
        model_name: Name of the model directory
        models_dir: Base models directory
        data_dir: Data directory
        cleanup: Whether to cleanup temp files after test
        
    Returns:
        True if test passed, False otherwise
    """
    config_dir = models_dir / model_name
    
    if not config_dir.exists():
        print(f"Model directory not found: {config_dir}")
        return False
    
    # Use tempfile for cleanup
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        try:
            original_config_path = get_config_file_path(config_dir)
            original_hash = get_file_hash(original_config_path)
            
            config, _ = find_yml_files(str(config_dir))
            config_dict = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config_dict)
            config = apply_debug_overrides(config, DEBUG_TRAINING_OVERRIDES)
            config = update_config(config)
            
            save_dir = tmp_path / model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy config file to save directory (needed for post-training evaluations)
            config_dest = save_dir / original_config_path.name
            shutil.copy(original_config_path, config_dest)
            
            df_metrics = trainer(
                config=config,
                data_dir=str(data_dir),
                save_dir=str(save_dir),
                cluster=False,
                experiment_tag="test",
                resume_ckpt=None,
                plot_pred_check=False,
                debug=True,
            )
            
            final_hash = get_file_hash(original_config_path)
            
            if original_hash != final_hash:
                print(f"ERROR: Config file was modified!")
                return False
            
            if df_metrics is None or len(df_metrics) == 0:
                print(f"ERROR: No metrics returned")
                return False
            
            print(f"SUCCESS: Training completed for {model_name}")
            print(f"Metrics columns: {df_metrics.columns.tolist()}")
            return True
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training for models")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Specific model to test (e.g., 'single_Lie_CC_scm6')"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model configurations (default: experiments)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing training data (default: data/)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search recursively for model configurations in subdirectories"
    )
    
    args = parser.parse_args()
    
    # Resolve models directory
    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not models_dir.is_absolute():
            models_dir = project_root / models_dir
    else:
        models_dir = get_models_directory()
    
    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
    else:
        data_dir = DEFAULT_DATA_DIR
    
    if args.list:
        configs = discover_model_configs(models_dir, recursive=args.recursive)
        print(f"Available models in {models_dir}{' (recursive)' if args.recursive else ''}:")
        for name, path in configs:
            print(f"  - {name}")
    elif args.model:
        success = run_single_model_test(args.model, models_dir=models_dir, data_dir=data_dir)
        sys.exit(0 if success else 1)
    else:
        # Run pytest with models-dir option if specified
        pytest_args = [__file__, "-v"]
        if args.models_dir:
            pytest_args.extend(["--models-dir", str(models_dir)])
        pytest.main(pytest_args)
