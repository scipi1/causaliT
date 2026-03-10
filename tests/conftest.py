"""
Pytest configuration for causaliT tests.

This file adds custom pytest options for configuring test directories.
"""

import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--models-dir",
        action="store",
        default=None,
        help="Directory containing model configurations to test (default: experiments)"
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default=None,
        help="Directory containing training data (default: data/)"
    )


def pytest_configure(config):
    """
    Configure pytest with custom options.
    
    This hook runs before tests are collected, allowing us to set up
    global variables that will be used by test_training_models.py
    """
    models_dir = config.getoption("--models-dir")
    data_dir = config.getoption("--data-dir")
    
    # Import and update the module-level variable in test_training_models
    if models_dir is not None:
        import tests.test_training_models as ttm
        
        # Resolve relative path
        models_path = Path(models_dir)
        if not models_path.is_absolute():
            models_path = ttm.project_root / models_path
        
        ttm._pytest_models_dir = str(models_path)
    
    if data_dir is not None:
        import tests.test_training_models as ttm
        
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = ttm.project_root / data_path
        
        # Update the data directory (need to add support for this)
        import os
        os.environ["CAUSALT_DATA_DIR"] = str(data_path)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent
