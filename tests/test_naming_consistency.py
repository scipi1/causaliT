"""
Test suite for validating consistency between experiment folder names and config parameters.

Run with: pytest tests/test_naming_consistency.py -v
Run single directory: pytest tests/test_naming_consistency.py -v -k "single_Lie_CC_scm6"

Naming Convention:
    forecaster_SelfAttentionClass_CrossAttentionClass_dataset_PhiParametrization_embeddingsComposition_hard

Example names:
    - single_Lie_CC_scm6
    - single_PhiSM_PhiSM_scm6_antisym
    - single_SM_SM_scm6_SVFA
    - single_Toeplitz_CC_scm6_gated

This test validates that the experiment folder name matches the config file parameters.
It does NOT modify any config files - only reports inconsistencies.
"""

import pytest
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf


# ============================================================================
# Naming Rules Configuration
# ============================================================================

NAMING_RULES = {
    "forecaster": {
        "config_key": "model.model_object",
        "position": 0,  # First part of the name
        "options": {
            "single": "SingleCausalLayer",
            "stage": "StageCausaliT",
        }
    },
    "SelfAttentionClass": {
        "config_key": "model.kwargs.dec_self_attention_type",
        "position": 1,  # Second part
        "options": {
            "Lie": "LieAttention",
            "PhiSM": "PhiSoftMax",
            "SM": "ScaledDotProduct",
            "Toeplitz": "ToeplitzLieAttention",
        }
    },
    "CrossAttentionClass": {
        "config_key": "model.kwargs.dec_cross_attention_type",
        "position": 2,  # Third part
        "options": {
            "CC": "CausalCrossAttention",
            "PhiSM": "PhiSoftMax",
            "SM": "ScaledDotProduct",
        }
    },
    "dataset": {
        "config_key": "data.dataset",
        "position": 3,  # Fourth part
        "options": {
            "scm6": "scm6",
            "scm7": "scm7",
            "scm8": "scm8",
        }
    },
    # Optional parts (may not be present in name)
    "PhiParametrization": {
        "config_key": "model.kwargs.dag_parameterization_self",  # Only self-attention!
        "default": None,  # If not in name, config should be None (validated!)
        "valid_values": ["independent", "gated", "antisymmetric"],  # Valid non-None values
        "options": {
            "antisym": "antisymmetric",
            "gated": "gated",
            "indep": "independent",
        }
    },
    "embeddingsComposition": {
        "config_key": "model.kwargs.comps_embed_X",
        "default": "summation",  # Default value if not in name
        "options": {
            "SVFA": "svfa",
        }
    },
    "hard": {
        "config_key": "training.use_hard_masks",
        "default": False,  # Default value if not in name
        "options": {
            "hard": True,
        }
    },
}

# All possible optional suffixes (order matters for parsing)
OPTIONAL_SUFFIXES = ["antisym", "gated", "indep", "SVFA", "hard"]


# ============================================================================
# Standard Values Configuration
# ============================================================================
# These are values that should be consistent across experiments for fair comparison.
# The test will WARN (not fail) if these values don't match the expected standards.

STANDARD_VALUES = {
    
    # Experimental parameters - architecture
    "experiment.d_model_set": 12,
    "experiment.dec_layers": 1,
    "experiment.n_heads": 1,
    "experiment.d_ff": 24,
    "experiment.d_qk": 6,
    "experiment.dropout": 0.0,
    
    # Experimental parameters - training
    "experiment.lr": 0.001,          # Learning rate
    "experiment.batch_size": 64,
    "experiment.max_epochs": 100,
    
    # Logging flags - all should be True for proper comparison and analysis
    "training.log_entropy": True,
    "training.log_acyclicity": True,
    "training.log_sparsity": True,
    "training.log_hsic": True,
    "training.log_decisiveness": True,
    
    # Regularization lambdas - baseline values (HSIC off for architecture comparison)
    "training.gamma": 0.0,              # Entropy regularization weight
    "training.kappa": 0.0,              # Acyclicity regularization (NOTEARS)
    "training.lambda_sparse": 0.0,      # Sparsity for self-attention
    "training.lambda_sparse_cross": 0.0,  # Sparsity for cross-attention
    "training.lambda_hsic": 0.0,        # HSIC OFF for baseline architecture comparison
    "training.hsic_sigma": 1.0,         # RBF kernel bandwidth for HSIC
    "training.lambda_decisive": 0.0,    # Decisiveness (binary entropy) loss
    "training.lambda_decisive_cross": 0.,  # Decisiveness for cross-attention
    "training.lambda_tau": 0.0,        # Temperature penalty
    "training.target_tau": 0.0,         # Target Gumbel-Softmax temperature
    
    # Cross-attention consistency - for isolating self-attention effects
    "model.kwargs.dec_cross_attention_type": "ScaledDotProduct",
}


@dataclass
class StandardValueWarning:
    """Represents a non-standard value that differs from expected."""
    config_key: str
    expected_value: Any
    actual_value: Any
    
    def __str__(self):
        return (f"  ⚠ {self.config_key}: "
                f"expected '{self.expected_value}' but found '{self.actual_value}'")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ParsedName:
    """Parsed experiment name components."""
    forecaster: str
    self_attention: str
    cross_attention: str
    dataset: str
    phi_param: Optional[str] = None  # antisym, gated, indep, or None
    embeddings_composition: Optional[str] = None  # SVFA or None
    hard: bool = False


@dataclass
class Inconsistency:
    """Represents an inconsistency between name and config."""
    component: str
    expected_from_name: Any
    actual_in_config: Any
    config_key: str
    
    def __str__(self):
        return (f"  - {self.component}: "
                f"name says '{self.expected_from_name}' → "
                f"expected config[{self.config_key}] = '{NAMING_RULES[self.component]['options'].get(self.expected_from_name, self.expected_from_name)}' "
                f"but found '{self.actual_in_config}'")


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_experiment_name(name: str) -> ParsedName:
    """
    Parse an experiment folder name into its components.
    
    Format: forecaster_SelfAttention_CrossAttention_dataset[_optional...]
    
    Args:
        name: Experiment folder name (e.g., "single_Lie_CC_scm6_antisym")
        
    Returns:
        ParsedName with all components extracted
    """
    parts = name.split("_")
    
    if len(parts) < 4:
        raise ValueError(f"Name '{name}' has fewer than 4 required parts")
    
    # Required parts (always in positions 0-3)
    forecaster = parts[0]
    self_attention = parts[1]
    cross_attention = parts[2]
    dataset = parts[3]
    
    # Optional parts (positions 4+)
    phi_param = None
    embeddings_comp = None
    hard = False
    
    for part in parts[4:]:
        if part in ["antisym", "gated", "indep"]:
            phi_param = part
        elif part == "SVFA":
            embeddings_comp = "SVFA"
        elif part == "hard":
            hard = True
        elif part.startswith("sweep"):
            # Ignore sweep suffixes (e.g., sweep_kl)
            continue
        else:
            # Unknown suffix - might be part of dataset name like "scm6" → "scm6_v2"
            # or could be an error. For now, we'll ignore it.
            pass
    
    return ParsedName(
        forecaster=forecaster,
        self_attention=self_attention,
        cross_attention=cross_attention,
        dataset=dataset,
        phi_param=phi_param,
        embeddings_composition=embeddings_comp,
        hard=hard
    )


def get_config_value(config: OmegaConf, key_path: str) -> Any:
    """
    Get a value from config using dot notation.
    
    Args:
        config: OmegaConf configuration object
        key_path: Dot-separated path (e.g., "model.kwargs.dec_self_attention_type")
        
    Returns:
        Value at the path, or None if not found
    """
    try:
        keys = key_path.split(".")
        value = config
        for key in keys:
            if hasattr(value, key) or (isinstance(value, dict) and key in value):
                value = value[key] if isinstance(value, dict) else getattr(value, key)
            else:
                return None
        return value
    except Exception:
        return None


# ============================================================================
# Validation Functions
# ============================================================================

def validate_experiment(name: str, config: OmegaConf) -> List[Inconsistency]:
    """
    Validate that experiment name matches config parameters.
    
    Args:
        name: Experiment folder name
        config: Loaded config file
        
    Returns:
        List of Inconsistency objects (empty if all matches)
    """
    try:
        parsed = parse_experiment_name(name)
    except ValueError as e:
        return [Inconsistency("name_parsing", "valid name", str(e), "N/A")]
    
    inconsistencies = []
    
    # Check forecaster
    rule = NAMING_RULES["forecaster"]
    expected_value = rule["options"].get(parsed.forecaster)
    if expected_value:
        actual_value = get_config_value(config, rule["config_key"])
        if actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "forecaster", parsed.forecaster, actual_value, rule["config_key"]
            ))
    
    # Check SelfAttentionClass
    rule = NAMING_RULES["SelfAttentionClass"]
    expected_value = rule["options"].get(parsed.self_attention)
    if expected_value:
        actual_value = get_config_value(config, rule["config_key"])
        if actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "SelfAttentionClass", parsed.self_attention, actual_value, rule["config_key"]
            ))
    
    # Check CrossAttentionClass
    rule = NAMING_RULES["CrossAttentionClass"]
    expected_value = rule["options"].get(parsed.cross_attention)
    if expected_value:
        actual_value = get_config_value(config, rule["config_key"])
        if actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "CrossAttentionClass", parsed.cross_attention, actual_value, rule["config_key"]
            ))
    
    # Check dataset
    rule = NAMING_RULES["dataset"]
    expected_value = rule["options"].get(parsed.dataset, parsed.dataset)  # Allow exact match
    actual_value = get_config_value(config, rule["config_key"])
    if actual_value != expected_value:
        inconsistencies.append(Inconsistency(
            "dataset", parsed.dataset, actual_value, rule["config_key"]
        ))
    
    # Check PhiParametrization
    rule = NAMING_RULES["PhiParametrization"]
    actual_value = get_config_value(config, rule["config_key"])
    
    if parsed.phi_param is not None:
        # Suffix present: config should match the specified value
        expected_value = rule["options"].get(parsed.phi_param)
        if expected_value and actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "PhiParametrization", parsed.phi_param, actual_value, rule["config_key"]
            ))
    else:
        # No suffix: config should be None (not in valid_values)
        # If the self-attention type doesn't support learnable phi, config should be None
        valid_phi_values = rule.get("valid_values", [])
        if actual_value is not None and actual_value in valid_phi_values:
            # Config has a valid phi parametrization but name doesn't indicate it
            inconsistencies.append(Inconsistency(
                "PhiParametrization", 
                "None (no suffix in name)", 
                actual_value, 
                rule["config_key"]
            ))
    
    # Check embeddingsComposition
    rule = NAMING_RULES["embeddingsComposition"]
    if parsed.embeddings_composition is not None:
        expected_value = rule["options"].get(parsed.embeddings_composition)
    else:
        expected_value = rule["default"]
    
    if expected_value is not None:
        actual_value = get_config_value(config, rule["config_key"])
        if actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "embeddingsComposition", 
                parsed.embeddings_composition or "default", 
                actual_value, 
                rule["config_key"]
            ))
    
    # Check hard mask
    rule = NAMING_RULES["hard"]
    if parsed.hard:
        expected_value = rule["options"]["hard"]
    else:
        expected_value = rule["default"]
    
    actual_value = get_config_value(config, rule["config_key"])
    if actual_value != expected_value:
        inconsistencies.append(Inconsistency(
            "hard", parsed.hard, actual_value, rule["config_key"]
        ))
    
    return inconsistencies


# ============================================================================
# Standard Values Validation
# ============================================================================

def check_standard_values(config: OmegaConf) -> List[StandardValueWarning]:
    """
    Check if config values match the expected standard values.
    
    This is for ensuring consistent experimental conditions across experiments.
    Returns warnings (not errors) for non-standard values.
    
    Args:
        config: Loaded config file
        
    Returns:
        List of StandardValueWarning objects (empty if all match standards)
    """
    warnings = []
    
    for key_path, expected_value in STANDARD_VALUES.items():
        actual_value = get_config_value(config, key_path)
        
        # Skip if key doesn't exist (might be optional)
        if actual_value is None:
            continue
        
        if actual_value != expected_value:
            warnings.append(StandardValueWarning(
                config_key=key_path,
                expected_value=expected_value,
                actual_value=actual_value
            ))
    
    return warnings


# ============================================================================
# Fixtures and Test Discovery
# ============================================================================

DEFAULT_MODELS_DIR = project_root / "experiments" / "single" / "scm6"


def discover_experiments(models_dir: Path) -> List[Tuple[str, Path]]:
    """
    Discover all experiment directories containing config files.
    
    Returns:
        List of (experiment_name, config_path) tuples
    """
    experiments = []
    
    if not models_dir.exists():
        return experiments
    
    for item in models_dir.iterdir():
        if item.is_dir():
            # Skip non-experiment directories
            if item.name.startswith('__') or item.name in ['euler', 'sweeps', 'combinations']:
                continue
            
            # Find config file
            config_files = list(item.glob("config*.yaml"))
            if config_files:
                experiments.append((item.name, config_files[0]))
    
    return sorted(experiments, key=lambda x: x[0])


def pytest_generate_tests(metafunc):
    """Dynamically generate test cases for each experiment."""
    if "experiment_config" in metafunc.fixturenames:
        experiments = discover_experiments(DEFAULT_MODELS_DIR)
        if experiments:
            metafunc.parametrize(
                "experiment_config",
                experiments,
                ids=[name for name, _ in experiments]
            )
        else:
            metafunc.parametrize(
                "experiment_config",
                [pytest.param(None, marks=pytest.mark.skip(reason="No experiments found"))]
            )


# ============================================================================
# Test Functions
# ============================================================================

class TestNamingConsistency:
    """Tests for validating experiment naming consistency with config."""
    
    def test_naming_matches_config(self, experiment_config: Tuple[str, Path]):
        """
        Test that experiment folder name matches config parameters.
        
        This test does NOT modify config files - it only reports inconsistencies.
        """
        name, config_path = experiment_config
        
        # Load config (read-only)
        try:
            config = OmegaConf.load(config_path)
        except Exception as e:
            pytest.fail(f"Failed to load config for {name}: {e}")
        
        # Validate naming consistency
        inconsistencies = validate_experiment(name, config)
        
        if inconsistencies:
            msg = f"\n\nNaming inconsistencies in '{name}':\n"
            for inc in inconsistencies:
                msg += str(inc) + "\n"
            pytest.fail(msg)
        
        print(f"✓ {name}: naming is consistent with config")


class TestAllExperimentsConsistency:
    """Run consistency check on all experiments and report summary."""
    
    def test_all_experiments_naming(self, tmp_path):
        """
        Validate naming consistency for all experiments in directory.
        
        Reports all inconsistencies in a summary format.
        """
        experiments = discover_experiments(DEFAULT_MODELS_DIR)
        
        if not experiments:
            pytest.skip(f"No experiments found in {DEFAULT_MODELS_DIR}")
        
        all_inconsistencies = {}
        consistent_count = 0
        
        for name, config_path in experiments:
            try:
                config = OmegaConf.load(config_path)
                inconsistencies = validate_experiment(name, config)
                
                if inconsistencies:
                    all_inconsistencies[name] = inconsistencies
                else:
                    consistent_count += 1
                    
            except Exception as e:
                all_inconsistencies[name] = [
                    Inconsistency("config_load", "valid config", str(e), "N/A")
                ]
        
        # Report summary
        print(f"\n{'='*70}")
        print(f"Naming Consistency Report")
        print(f"{'='*70}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Consistent: {consistent_count}")
        print(f"Inconsistent: {len(all_inconsistencies)}")
        
        if all_inconsistencies:
            print(f"\n{'='*70}")
            print("INCONSISTENCIES FOUND:")
            print(f"{'='*70}")
            for name, issues in all_inconsistencies.items():
                print(f"\n{name}:")
                for issue in issues:
                    print(str(issue))
        
        assert len(all_inconsistencies) == 0, (
            f"{len(all_inconsistencies)} experiments have naming inconsistencies"
        )


class TestStandardValues:
    """Check if experiments use standard values for fair comparison."""
    
    def test_all_experiments_standard_values(self):
        """
        Check all experiments for non-standard values.
        
        This test WARNS but does not fail for non-standard values.
        It helps identify experiments that may not be directly comparable.
        """
        experiments = discover_experiments(DEFAULT_MODELS_DIR)
        
        if not experiments:
            pytest.skip(f"No experiments found in {DEFAULT_MODELS_DIR}")
        
        all_warnings = {}
        standard_count = 0
        
        for name, config_path in experiments:
            try:
                config = OmegaConf.load(config_path)
                warnings = check_standard_values(config)
                
                if warnings:
                    all_warnings[name] = warnings
                else:
                    standard_count += 1
                    
            except Exception as e:
                all_warnings[name] = [
                    StandardValueWarning("config_load", "valid config", str(e))
                ]
        
        # Report summary
        print(f"\n{'='*70}")
        print(f"Standard Values Report")
        print(f"{'='*70}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Using standard values: {standard_count}")
        print(f"Non-standard values: {len(all_warnings)}")
        
        if all_warnings:
            print(f"\n{'='*70}")
            print("NON-STANDARD VALUES (warnings):")
            print(f"{'='*70}")
            for name, warnings in all_warnings.items():
                print(f"\n{name}:")
                for warning in warnings:
                    print(str(warning))
        
        # FAIL if any experiment has non-standard values
        # This ensures consistent experimental conditions for fair comparison
        assert len(all_warnings) == 0, (
            f"{len(all_warnings)} experiments have non-standard values. "
            f"Update configs to match STANDARD_VALUES or adjust STANDARD_VALUES in test file."
        )
        print("\n✓ All experiments use standard values!")


# ============================================================================
# Standalone Runner
# ============================================================================

def check_all_experiments(models_dir: Path = DEFAULT_MODELS_DIR) -> Dict[str, List[Inconsistency]]:
    """
    Check all experiments for naming consistency (standalone function).
    
    Returns:
        Dictionary of {experiment_name: [Inconsistency, ...]}
    """
    experiments = discover_experiments(models_dir)
    results = {}
    
    for name, config_path in experiments:
        try:
            config = OmegaConf.load(config_path)
            inconsistencies = validate_experiment(name, config)
            if inconsistencies:
                results[name] = inconsistencies
        except Exception as e:
            results[name] = [Inconsistency("config_load", "valid", str(e), "N/A")]
    
    return results


def main():
    """Run standalone consistency check with detailed output."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check experiment naming consistency")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Check a specific experiment"
    )
    
    args = parser.parse_args()
    
    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not models_dir.is_absolute():
            models_dir = project_root / models_dir
    else:
        models_dir = DEFAULT_MODELS_DIR
    
    print(f"Checking experiments in: {models_dir}")
    print(f"{'='*70}\n")
    
    if args.experiment:
        # Check single experiment
        config_path = models_dir / args.experiment
        config_files = list(config_path.glob("config*.yaml"))
        if not config_files:
            print(f"ERROR: No config file found in {config_path}")
            return 1
        
        config = OmegaConf.load(config_files[0])
        inconsistencies = validate_experiment(args.experiment, config)
        
        if inconsistencies:
            print(f"INCONSISTENCIES in '{args.experiment}':")
            for inc in inconsistencies:
                print(str(inc))
            return 1
        else:
            print(f"✓ '{args.experiment}' is consistent")
            return 0
    
    else:
        # Check all experiments
        inconsistencies = check_all_experiments(models_dir)
        experiments = discover_experiments(models_dir)
        
        consistent = len(experiments) - len(inconsistencies)
        
        print(f"Results:")
        print(f"  Total: {len(experiments)}")
        print(f"  Consistent: {consistent}")
        print(f"  Inconsistent: {len(inconsistencies)}")
        
        if inconsistencies:
            print(f"\n{'='*70}")
            print("INCONSISTENCIES:")
            print(f"{'='*70}")
            for name, issues in inconsistencies.items():
                print(f"\n{name}:")
                for issue in issues:
                    print(str(issue))
            return 1
        else:
            print("\n✓ All experiments are consistent!")
            return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
