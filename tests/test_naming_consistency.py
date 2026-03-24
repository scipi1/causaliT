"""
Test suite for validating consistency between experiment folder names and config parameters.

Run with: pytest tests/test_naming_consistency.py -v
Run single directory: pytest tests/test_naming_consistency.py -v -k "single_Lie_CC_scm6"

=============================================================================
NAMING CONVENTION
=============================================================================
Format: <architecture>_<SelfAttn>_<CrossAttn>_<dataset>_[modifiers]_<ID>

Components:
  architecture : single, na_single, stage
  SelfAttn     : SM (ScaledDotProduct), Lie (LieAttention), 
                 PhiSM (PhiSoftMax), Toeplitz (ToeplitzLieAttention)
  CrossAttn    : SM (ScaledDotProduct), CC (CausalCrossAttention), PhiSM (PhiSoftMax)
  dataset      : scm1, scm2, scm3, scm6, scm7, etc.
  modifiers    : (optional) SVFA, antisym, gated, indep, hard, orthS
  ID           : Euler job ID or random hash

Architecture Mapping:
  single    → SingleCausalLayer
  na_single → NoiseAwareSingleCausalLayer  
  stage     → StageCausaliT

Modifier Meanings:
  SVFA    : comps_embed_X = "svfa" (explicit for single/stage, implicit for na_single)
  antisym : dag_parameterization_self = "antisymmetric"
  gated   : dag_parameterization_self = "gated"
  indep   : dag_parameterization_self = "independent"
  hard    : use_hard_masks = true
  orthS   : orthogonal frozen S embedding (default is learnable)

EXCLUDED from naming checks:
  - Experiments starting with "ANS_" (Attention Necessity Score)
  - Experiments starting with "sweep_" (parameter sweeps)
  - Sweep combination folders

Example names:
  - single_SM_SM_scm6_60473063
  - single_Lie_SM_scm6_SVFA_59250879
  - single_Toeplitz_SM_scm6_antisym_59250699
  - na_single_Toeplitz_SM_scm1_60740750
  - stage_SM_SM_scm6_12345678

This test validates that the experiment folder name matches the config file parameters.
It does NOT modify any config files - only reports inconsistencies.
=============================================================================
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
    "architecture": {
        "config_key": "model.model_object",
        "position": 0,  # First part of the name
        "options": {
            "single": "SingleCausalLayer",
            "na_single": "NoiseAwareSingleCausalLayer",
            "stage": "StageCausaliT",
        }
    },
    "SelfAttentionClass": {
        "config_key": "model.kwargs.dec_self_attention_type",
        "position": 1,  # Second part (or after architecture)
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
        # Dataset aliases for expanded names
        "aliases": {
            "scm1_linear_gaussian": "scm1",
            "scm2_nonlinear_gaussian": "scm2",
            "scm3_nonlinear_nongaussian": "scm3",
        }
    },
    # Optional modifiers (may not be present in name)
    "PhiParametrization": {
        "config_key": "model.kwargs.dag_parameterization_self",
        "default": None,  # If not in name, config should be None
        "valid_values": ["independent", "gated", "antisymmetric"],
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
    "orthS": {
        # Orthogonal frozen S embedding - check for freeze: true in ds_embed_S
        "config_key": "model.kwargs.ds_embed_S.freeze",
        "default": None,  # Default is learnable (no freeze key)
        "options": {
            "orthS": True,
        }
    },
}

# All possible optional suffixes (order matters for parsing)
OPTIONAL_SUFFIXES = ["antisym", "gated", "indep", "SVFA", "hard", "orthS"]

# Prefixes to skip from validation
SKIP_PREFIXES = ["ANS_", "sweep_", "combo_"]


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
    "experiment.lr": 0.001,
    "experiment.batch_size": 64,
    "experiment.max_epochs": 100,
    
    # Logging flags - all should be True for proper comparison and analysis
    "training.log_entropy": True,
    "training.log_acyclicity": True,
    "training.log_sparsity": True,
    "training.log_hsic": True,
    "training.log_decisiveness": True,
    
    # Regularization lambdas - baseline values (all off for fair comparison)
    "training.kappa": 0.0,
    "training.lambda_sparse": 0.0,
    "training.lambda_sparse_cross": 0.0,
    "training.lambda_hsic": 0.0,
    "training.lambda_decisive": 0.0,
    "training.lambda_decisive_cross": 0.0,
    "training.lambda_tau": 0.0,
    
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
    architecture: str
    self_attention: str
    cross_attention: str
    dataset: str
    phi_param: Optional[str] = None  # antisym, gated, indep, or None
    embeddings_composition: Optional[str] = None  # SVFA or None
    hard: bool = False
    orthS: bool = False


@dataclass
class Inconsistency:
    """Represents an inconsistency between name and config."""
    component: str
    expected_from_name: Any
    actual_in_config: Any
    config_key: str
    
    def __str__(self):
        rule = NAMING_RULES.get(self.component, {})
        options = rule.get("options", {})
        expected_config_value = options.get(self.expected_from_name, self.expected_from_name)
        return (f"  - {self.component}: "
                f"name says '{self.expected_from_name}' → "
                f"expected config[{self.config_key}] = '{expected_config_value}' "
                f"but found '{self.actual_in_config}'")


# ============================================================================
# Parsing Functions
# ============================================================================

def should_skip_experiment(name: str) -> bool:
    """
    Check if an experiment should be skipped from validation.
    
    Skips:
    - ANS experiments (ANS_*)
    - Sweep experiments (sweep_*)
    - Sweep combinations (combo_*)
    """
    for prefix in SKIP_PREFIXES:
        if name.startswith(prefix):
            return True
    # Also skip if 'combo' appears anywhere (sweep combinations)
    if "combo" in name.lower():
        return True
    return False


def parse_experiment_name(name: str) -> ParsedName:
    """
    Parse an experiment folder name into its components.
    
    Format: architecture_SelfAttention_CrossAttention_dataset[_modifiers][_ID]
    
    Special handling for multi-part architecture names:
    - na_single → NoiseAwareSingleCausalLayer
    
    Args:
        name: Experiment folder name (e.g., "single_Lie_SM_scm6_antisym_12345678")
        
    Returns:
        ParsedName with all components extracted
    """
    parts = name.split("_")
    
    # Handle multi-part architecture names
    if len(parts) >= 2 and parts[0] == "na" and parts[1] == "single":
        architecture = "na_single"
        remaining_parts = parts[2:]
    else:
        architecture = parts[0]
        remaining_parts = parts[1:]
    
    if len(remaining_parts) < 3:
        raise ValueError(f"Name '{name}' has fewer than required parts after architecture")
    
    # Required parts after architecture
    self_attention = remaining_parts[0]
    cross_attention = remaining_parts[1]
    dataset = remaining_parts[2]
    
    # Optional parts (positions 3+)
    phi_param = None
    embeddings_comp = None
    hard = False
    orthS = False
    
    for part in remaining_parts[3:]:
        if part in ["antisym", "gated", "indep"]:
            phi_param = part
        elif part == "SVFA":
            embeddings_comp = "SVFA"
        elif part == "hard":
            hard = True
        elif part == "orthS":
            orthS = True
        elif part.isdigit():
            # This is the ID, skip it
            continue
        else:
            # Unknown suffix - might be part of dataset name or ID
            # Try to match as numeric ID
            try:
                int(part)
            except ValueError:
                # Not a number, might be additional dataset identifier
                pass
    
    return ParsedName(
        architecture=architecture,
        self_attention=self_attention,
        cross_attention=cross_attention,
        dataset=dataset,
        phi_param=phi_param,
        embeddings_composition=embeddings_comp,
        hard=hard,
        orthS=orthS
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


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name by applying aliases.
    
    E.g., "scm1_linear_gaussian" → "scm1"
    """
    if dataset_name is None:
        return None
    
    aliases = NAMING_RULES["dataset"].get("aliases", {})
    return aliases.get(dataset_name, dataset_name)


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
    
    # Check architecture
    rule = NAMING_RULES["architecture"]
    expected_value = rule["options"].get(parsed.architecture)
    if expected_value:
        actual_value = get_config_value(config, rule["config_key"])
        if actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "architecture", parsed.architecture, actual_value, rule["config_key"]
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
    
    # Check dataset (with alias normalization)
    rule = NAMING_RULES["dataset"]
    actual_value = get_config_value(config, rule["config_key"])
    normalized_actual = normalize_dataset_name(actual_value)
    if normalized_actual != parsed.dataset:
        # Only report if they don't match after normalization
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
        valid_phi_values = rule.get("valid_values", [])
        if actual_value is not None and actual_value in valid_phi_values:
            inconsistencies.append(Inconsistency(
                "PhiParametrization", 
                "None (no suffix in name)", 
                actual_value, 
                rule["config_key"]
            ))
    
    # Check embeddingsComposition
    # For na_single, SVFA is implicit (not required in name)
    rule = NAMING_RULES["embeddingsComposition"]
    actual_value = get_config_value(config, rule["config_key"])
    
    if parsed.architecture == "na_single":
        # For noise-aware, SVFA is required in config but NOT in name
        if actual_value != "svfa":
            inconsistencies.append(Inconsistency(
                "embeddingsComposition", 
                "svfa (implicit for na_single)", 
                actual_value, 
                rule["config_key"]
            ))
    else:
        # For other architectures, check based on name
        if parsed.embeddings_composition is not None:
            expected_value = rule["options"].get(parsed.embeddings_composition)
        else:
            expected_value = rule["default"]
        
        if expected_value is not None and actual_value != expected_value:
            inconsistencies.append(Inconsistency(
                "embeddingsComposition", 
                parsed.embeddings_composition or "default (summation)", 
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
    
    # Check orthS (orthogonal frozen S embedding)
    # This is trickier - we need to check if ds_embed_S has freeze: true
    # or if it's the OrthogonalMaskEmbedding format
    if parsed.orthS:
        ds_embed_S = get_config_value(config, "model.kwargs.ds_embed_S")
        is_orthogonal = False
        if ds_embed_S is not None:
            # Check if it's the orthogonal format (has freeze: true)
            if hasattr(ds_embed_S, 'freeze') or (isinstance(ds_embed_S, dict) and 'freeze' in ds_embed_S):
                freeze_val = ds_embed_S.get('freeze') if isinstance(ds_embed_S, dict) else getattr(ds_embed_S, 'freeze', None)
                is_orthogonal = freeze_val == True
            # Or check for num_variables (orthogonal format)
            elif hasattr(ds_embed_S, 'num_variables') or (isinstance(ds_embed_S, dict) and 'num_variables' in ds_embed_S):
                is_orthogonal = True
        
        if not is_orthogonal:
            inconsistencies.append(Inconsistency(
                "orthS", 
                "orthogonal frozen S embedding", 
                "learnable S embedding",
                "model.kwargs.ds_embed_S"
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

# Default directories to search for experiments
DEFAULT_EXPERIMENT_DIRS = [
    project_root / "experiments" / "single",
    project_root / "experiments" / "noise_aware_single",
    project_root / "experiments" / "stage",
]


def discover_experiments(base_dirs: List[Path] = None) -> List[Tuple[str, Path]]:
    """
    Discover all experiment directories containing config files.
    
    Recursively searches through experiment directories, looking for
    folders that contain config*.yaml files.
    
    Returns:
        List of (experiment_name, config_path) tuples
    """
    if base_dirs is None:
        base_dirs = DEFAULT_EXPERIMENT_DIRS
    
    experiments = []
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        
        # Recursively find all config files
        for config_path in base_dir.rglob("config*.yaml"):
            exp_dir = config_path.parent
            exp_name = exp_dir.name
            
            # Skip if this experiment should be skipped
            if should_skip_experiment(exp_name):
                continue
            
            # Skip euler/sweeper subdirectories that aren't actual experiments
            if any(p in str(exp_dir) for p in ['/sweeper/', '\\sweeper\\']):
                continue
            
            experiments.append((exp_name, config_path))
    
    # Remove duplicates (same folder might have multiple config files)
    seen = set()
    unique_experiments = []
    for name, path in experiments:
        key = (name, path.parent)
        if key not in seen:
            seen.add(key)
            unique_experiments.append((name, path))
    
    return sorted(unique_experiments, key=lambda x: x[0])


def pytest_generate_tests(metafunc):
    """Dynamically generate test cases for each experiment."""
    if "experiment_config" in metafunc.fixturenames:
        experiments = discover_experiments()
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
        Validate naming consistency for all experiments.
        
        Reports all inconsistencies in a summary format.
        """
        experiments = discover_experiments()
        
        if not experiments:
            pytest.skip("No experiments found in experiment directories")
        
        all_inconsistencies = {}
        consistent_count = 0
        skipped_count = 0
        
        for name, config_path in experiments:
            # Double-check skip (should already be filtered)
            if should_skip_experiment(name):
                skipped_count += 1
                continue
            
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
        print(f"Total experiments checked: {len(experiments)}")
        print(f"Consistent: {consistent_count}")
        print(f"Inconsistent: {len(all_inconsistencies)}")
        print(f"Skipped (ANS/sweep): {skipped_count}")
        
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
        experiments = discover_experiments()
        
        if not experiments:
            pytest.skip("No experiments found in experiment directories")
        
        all_warnings = {}
        standard_count = 0
        
        for name, config_path in experiments:
            # Skip ANS/sweep experiments
            if should_skip_experiment(name):
                continue
            
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
        
        # This test warns but doesn't fail
        if all_warnings:
            pytest.skip(
                f"{len(all_warnings)} experiments have non-standard values. "
                f"This is informational only."
            )
        
        print("\n✓ All experiments use standard values!")


# ============================================================================
# Standalone Runner
# ============================================================================

def check_all_experiments(base_dirs: List[Path] = None) -> Dict[str, List[Inconsistency]]:
    """
    Check all experiments for naming consistency (standalone function).
    
    Returns:
        Dictionary of {experiment_name: [Inconsistency, ...]}
    """
    experiments = discover_experiments(base_dirs)
    results = {}
    
    for name, config_path in experiments:
        if should_skip_experiment(name):
            continue
        
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
        "--experiment-dir",
        type=str,
        default=None,
        help="Specific experiment directory to check"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Check a specific experiment by name"
    )
    
    args = parser.parse_args()
    
    if args.experiment_dir:
        base_dirs = [Path(args.experiment_dir)]
        if not base_dirs[0].is_absolute():
            base_dirs = [project_root / args.experiment_dir]
    else:
        base_dirs = DEFAULT_EXPERIMENT_DIRS
    
    print(f"Checking experiments in:")
    for d in base_dirs:
        print(f"  - {d}")
    print(f"{'='*70}\n")
    
    experiments = discover_experiments(base_dirs)
    
    if args.experiment:
        # Check single experiment
        matching = [(n, p) for n, p in experiments if args.experiment in n]
        if not matching:
            print(f"ERROR: No experiment found matching '{args.experiment}'")
            return 1
        experiments = matching
    
    inconsistencies = {}
    consistent = 0
    skipped = 0
    
    for name, config_path in experiments:
        if should_skip_experiment(name):
            skipped += 1
            continue
        
        try:
            config = OmegaConf.load(config_path)
            issues = validate_experiment(name, config)
            
            if issues:
                inconsistencies[name] = issues
            else:
                consistent += 1
                
        except Exception as e:
            inconsistencies[name] = [Inconsistency("config_load", "valid", str(e), "N/A")]
    
    print(f"Results:")
    print(f"  Total checked: {len(experiments) - skipped}")
    print(f"  Consistent: {consistent}")
    print(f"  Inconsistent: {len(inconsistencies)}")
    print(f"  Skipped (ANS/sweep): {skipped}")
    
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
