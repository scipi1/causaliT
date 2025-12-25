# SCM Dataset Generation Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Quick Start](#quick-start)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Examples](#examples)
6. [Output Files Reference](#output-files-reference)
7. [Advanced Topics](#advanced-topics)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to create Structural Causal Model (SCM) datasets using the `scm_ds` module. An SCM dataset is a synthetic dataset generated from a directed acyclic graph (DAG) where:

- **Nodes** represent variables
- **Edges** represent causal relationships
- **Structural equations** define how each variable depends on its parents
- **Noise terms** introduce randomness and can be correlated to model confounding

### What Gets Generated

When you create an SCM dataset, the system produces:

- **Dataset arrays**: NumPy arrays with input and target variables
- **Attention masks**: For encoder/decoder self-attention and cross-attention
- **Graph visualization**: PDF visualization of the causal DAG
- **Metadata**: JSON files with variable mappings and dataset information

### Use Cases

- Testing causal inference algorithms
- Evaluating transformer architectures with causal structure
- Generating synthetic data with known ground truth relationships
- Studying the effects of interventions and confounding

---

## Core Components

### 1. NodeSpec

Defines a single node in the causal graph.

```python
NodeSpec(name, parents, expr)
```

**Parameters:**
- `name` (str): Unique identifier for the node (e.g., "Y", "X1", "Parent1")
- `parents` (List[str]): List of parent node names that this node depends on
- `expr` (str): Structural equation as a SymPy-compatible string

**Structural Equation Rules:**
- Use parent variable names as they appear in the `parents` list
- Include a noise term `eps_<name>` for the node (e.g., `eps_Y` for node "Y")
- Can include parameters defined in the `params` dictionary
- Standard mathematical operations: `+`, `-`, `*`, `/`, `**` (power)

**Example:**
```python
NodeSpec("Y", ["X", "Z"], "X + 2*Z + eps_Y")  # Y depends on X and Z
NodeSpec("X", [], "eps_X")                     # X is exogenous (no parents)
```

### 2. NoiseModel

Configures the noise distributions for nodes.

```python
NoiseModel(singles, groups)
```

**Parameters:**
- `singles` (Dict[str, Callable]): Independent noise samplers for individual nodes
  - Signature: `lambda rng, n: np.ndarray`
  - `rng`: NumPy random generator
  - `n`: Number of samples
  - Returns: 1D array of shape `(n,)`

- `groups` (List[GroupNoise]): Correlated noise for multiple nodes
  - Used to model latent confounding
  - Each group returns a 2D array of shape `(n, k)` where k = number of nodes in group

**Example:**
```python
# Independent noise
singles = {
    "X": lambda rng, n: rng.standard_normal(n),
    "Y": lambda rng, n: rng.standard_normal(n) * 0.5  # Lower variance
}

# Correlated noise (confounding between Z and Y)
group = GroupNoise(
    nodes=("Z", "Y"),
    sampler=lambda rng, n: rng.multivariate_normal(
        mean=[0, 0],
        cov=[[1.0, 0.3],   # Correlation of 0.3
             [0.3, 1.0]],
        size=n
    )
)

noise_model = NoiseModel(singles=singles, groups=[group])
```

### 3. SCMDataset

Main class for creating and generating SCM datasets.

```python
SCMDataset(
    name,
    description,
    tags,
    specs,
    params,
    singles,
    groups,
    input_labels,
    target_labels
)
```

**Parameters:**
- `name` (str): Dataset name
- `description` (str): Human-readable description
- `tags` (List[str]): Optional tags for categorization
- `specs` (List[NodeSpec]): List of all node specifications
- `params` (Dict[str, float]): Parameters used in structural equations
- `singles` (Dict[str, Callable]): Independent noise samplers
- `groups` (List[GroupNoise]): Correlated noise groups
- `input_labels` (List[str]): Node names to use as inputs
- `target_labels` (List[str]): Node names to use as targets

---

## Quick Start

Here's a minimal example to generate an SCM dataset:

```python
from scm_ds.scm import NodeSpec, NoiseModel, GroupNoise
from scm_ds.datasets import SCMDataset
from os.path import join

# 1. Define the causal structure
specs = [
    NodeSpec("X", [], "eps_X"),           # Exogenous input
    NodeSpec("Y", ["X"], "X + eps_Y")     # Y depends on X
]

# 2. Define noise samplers
singles = {
    "X": lambda rng, n: rng.standard_normal(n),
    "Y": lambda rng, n: rng.standard_normal(n)
}

# 3. Create the dataset
dataset = SCMDataset(
    name="simple_example",
    description="Simple X -> Y relationship",
    tags=None,
    specs=specs,
    params={},
    singles=singles,
    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)

# 4. Generate and save
dataset.generate_ds(
    mode="flat",
    n=5000,
    save_dir="data/simple_example",
    seed=42
)
```

---

## Step-by-Step Guide

### Step 1: Design Your Causal Graph

Decide on:
1. **Variables**: What nodes do you need?
2. **Relationships**: Which variables cause which others?
3. **Structure**: Draw the DAG on paper first

**Example**: One-to-one relationships with cross-talk
```
P1 → C1 ↘
P2 → C2 → Y
P3 → C3 ↗
```

### Step 2: Define Node Specifications

Create a `NodeSpec` for each node in topological order (parents before children):

```python
specs = [
    # Parents (exogenous)
    NodeSpec("P1", [], "eps_P1"),
    NodeSpec("P2", [], "eps_P2"),
    NodeSpec("P3", [], "eps_P3"),
    
    # Children (depend on parents)
    NodeSpec("C1", ["P1"], "P1 + eps_C1"),
    NodeSpec("C2", ["P2"], "P2 + eps_C2"),
    NodeSpec("C3", ["P3"], "P3 + eps_C3"),
    
    # Output (depends on children)
    NodeSpec("Y", ["C1", "C2", "C3"], "C1 + C2 + C3 + eps_Y")
]
```

### Step 3: Configure Noise Distributions

Define how randomness enters the system:

```python
# Independent noise for each node
singles = {
    "P1": lambda rng, n: rng.standard_normal(n),
    "P2": lambda rng, n: rng.standard_normal(n),
    "P3": lambda rng, n: rng.standard_normal(n),
    "C1": lambda rng, n: rng.standard_normal(n),
    "C2": lambda rng, n: rng.standard_normal(n),
    "C3": lambda rng, n: rng.standard_normal(n),
    "Y": lambda rng, n: rng.standard_normal(n)
}

# Optional: Add correlated noise for confounding
groups = []  # Empty if no confounding
```

### Step 4: Set Parameters

Define any constants used in structural equations:

```python
params = {
    "w1": 0.01,  # Weight for relationship 1
    "w2": 0.01,  # Weight for relationship 2
    # ... add more as needed
}
```

### Step 5: Create the Dataset Object

```python
dataset = SCMDataset(
    name="my_dataset",
    description="Description of what this dataset represents",
    tags=["tag1", "tag2"],  # Optional
    specs=specs,
    params=params,
    singles=singles,
    groups=groups,
    input_labels=["P1", "P2", "P3", "C1", "C2", "C3"],  # Inputs to model
    target_labels=["Y"]  # Targets to predict
)
```

### Step 6: Generate and Save

```python
dataset.generate_ds(
    mode="flat",           # Currently only "flat" mode is supported
    n=5000,                # Number of samples
    save_dir="data/my_dataset",  # Where to save
    meta_dict=None,        # Optional: Additional metadata
    seed=42                # Random seed for reproducibility
)
```

---

## Examples

### Example 1: Simple Linear Relationship

```python
from scm_ds.scm import NodeSpec, NoiseModel
from scm_ds.datasets import SCMDataset

specs = [
    NodeSpec("X", [], "eps_X"),
    NodeSpec("Y", ["X"], "2*X + eps_Y")
]

singles = {
    "X": lambda rng, n: rng.standard_normal(n),
    "Y": lambda rng, n: rng.standard_normal(n) * 0.5
}

dataset = SCMDataset(
    name="linear",
    description="Y = 2X + noise",
    tags=None,
    specs=specs,
    params={},
    singles=singles,
    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)

dataset.generate_ds(mode="flat", n=1000, save_dir="data/linear")
```

### Example 2: Confounding (Correlated Noise)

```python
from scm_ds.scm import NodeSpec, GroupNoise

specs = [
    NodeSpec("X", [], "eps_X"),
    NodeSpec("Z", [], "eps_Z"),
    NodeSpec("Y", ["X", "Z"], "X + Z + eps_Y")
]

singles = {
    "X": lambda rng, n: rng.standard_normal(n)
}

# Z and Y are confounded
group = GroupNoise(
    nodes=("Z", "Y"),
    sampler=lambda rng, n: rng.multivariate_normal(
        mean=[0, 0],
        cov=[[1.0, 0.5],   # Strong confounding
             [0.5, 1.0]],
        size=n
    )
)

dataset = SCMDataset(
    name="confounded",
    description="Z and Y share latent confounder",
    tags=["confounding"],
    specs=specs,
    params={},
    singles=singles,
    groups=[group],
    input_labels=["X", "Z"],
    target_labels=["Y"]
)

dataset.generate_ds(mode="flat", n=5000, save_dir="data/confounded")
```

### Example 3: Complex Structure with Cross-Talk

This example from `datasets.py` shows a realistic scenario:

```python
ds_scm_1_to_1_ct = SCMDataset(
    name="one-to-one_with_crosstalk",
    description="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs=[
        # Parents
        NodeSpec("P1", [], "eps_P1"),
        NodeSpec("P2", [], "eps_P2"),
        NodeSpec("P3", [], "eps_P3"),
        NodeSpec("P4", [], "eps_P4"),
        NodeSpec("P5", [], "eps_P5"),
        
        # Children (C1 has cross-talk from P2)
        NodeSpec("C1", ["P1", "P2"], "P1 - P2 + eps_C1"),
        NodeSpec("C2", ["P2"], "P2 + eps_C2"),
        NodeSpec("C3", ["P3"], "P3 + eps_C3"),
        NodeSpec("C4", ["P4"], "P4 + eps_C4"),
        NodeSpec("C5", ["P5"], "P5 + eps_C5"),
        
        # Output
        NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"], 
                 "C1 + C2 + C3 + C4 + C5 + eps_Y"),
    ],
    params={
        "w1": 0.01, "w2": 0.01, "w3": 0.01, "w4": 0.01, "w5": 0.01
    },
    singles={
        "P1": lambda rng, n: rng.standard_normal(n),
        "P2": lambda rng, n: rng.standard_normal(n),
        "P3": lambda rng, n: rng.standard_normal(n),
        "P4": lambda rng, n: rng.standard_normal(n),
        "P5": lambda rng, n: rng.standard_normal(n),
        "C1": lambda rng, n: rng.standard_normal(n),
        "C2": lambda rng, n: rng.standard_normal(n),
        "C3": lambda rng, n: rng.standard_normal(n),
        "C4": lambda rng, n: rng.standard_normal(n),
        "C5": lambda rng, n: rng.standard_normal(n),
        "Y": lambda rng, n: rng.standard_normal(n),
    },
    groups=None,
    input_labels=["P1", "P2", "P3", "P4", "P5", "C1", "C2", "C3", "C4", "C5"],
    target_labels=["Y"]
)

ds_scm_1_to_1_ct.generate_ds(
    mode="flat",
    n=5000,
    save_dir="data/example_crosstalk"
)
```

---

## Output Files Reference

After calling `generate_ds()`, the following files are created in the specified directory:

### 1. `ds.npz` - Dataset Arrays

NumPy compressed archive containing:
- `x`: Input data array, shape `(n_samples, n_input_vars, 2)`
  - Last dimension: `[value, variable_id]`
- `y`: Target data array, shape `(n_samples, n_target_vars, 2)`
  - Last dimension: `[value, variable_id]`

**Loading:**
```python
data = np.load("data/my_dataset/ds.npz")
X = data['x']  # Inputs
Y = data['y']  # Targets
```

### 2. Attention Masks

CSV files defining causal structure for attention mechanisms:

- **`enc_sef_att_mask.csv`**: Encoder self-attention mask
  - Rows: Input variables (queries)
  - Cols: Input variables (keys)
  - Value 1: Variable can attend to itself or its parents

- **`dec_self_att_mask.csv`**: Decoder self-attention mask
  - Rows: Target variables (queries)
  - Cols: Target variables (keys)
  - Value 1: Variable can attend to itself or its parents

- **`dec_cross_att_mask.csv`**: Decoder cross-attention mask
  - Rows: Target variables (queries)
  - Cols: Input variables (keys)
  - Value 1: Target can attend to this input variable

**Loading:**
```python
import pandas as pd
enc_mask = pd.read_csv("data/my_dataset/enc_sef_att_mask.csv", index_col=0)
```

### 3. Metadata Files (JSON)

- **`meta.json`**: Dataset metadata
  ```json
  {
    "name": "dataset_name",
    "created": "2025-10-29",
    "description": "Dataset description"
  }
  ```

- **`input_vars_map.json`**: Maps input variable names to integer IDs
  ```json
  {
    "X": 1,
    "Z": 2
  }
  ```

- **`input_feat_map.json`**: Maps feature indices to names
  ```json
  {
    "0": "value",
    "1": "variable"
  }
  ```

- **`target_vars_map.json`**: Maps target variable names to integer IDs
- **`target_feat_map.json`**: Maps target feature indices to names

### 4. `graph.pdf` - Causal Graph Visualization

Visual representation of the causal DAG showing:
- All nodes (variables)
- Directed edges (causal relationships)
- Graph structure in topological order

---

## Advanced Topics

### Working with Interventions

You can test interventions (do-calculus) using the underlying SCM:

```python
# Access the internal SCM
scm = dataset.scm

# Sample from observational distribution
df_obs = scm.sample(n=1000, seed=42)

# Create intervened SCM (do-operator)
scm_do = scm.do({"X": 1.0})  # Set X = 1.0
df_int = scm_do.sample(n=1000, seed=42)

# Compare distributions
print("Observational mean Y:", df_obs["Y"].mean())
print("Interventional mean Y:", df_int["Y"].mean())
```

### Custom Noise Distributions

You can use any distribution supported by NumPy:

```python
singles = {
    # Uniform distribution
    "X": lambda rng, n: rng.uniform(-1, 1, n),
    
    # Exponential distribution
    "Y": lambda rng, n: rng.exponential(scale=2.0, size=n),
    
    # Mixture distribution
    "Z": lambda rng, n: (
        rng.standard_normal(n) * 0.5 +  # Component 1
        rng.uniform(-2, 2, n) * 0.5      # Component 2
    )
}
```

### Complex Structural Equations

Structural equations support various mathematical operations:

```python
specs = [
    # Polynomial relationship
    NodeSpec("Y1", ["X"], "X**2 + 2*X + eps_Y1"),
    
    # Multiple parents with interaction
    NodeSpec("Y2", ["X", "Z"], "X*Z + X + Z + eps_Y2"),
    
    # With parameters
    NodeSpec("Y3", ["X"], "w1*X + w2*X**2 + eps_Y3"),
]

params = {"w1": 0.5, "w2": 0.1}
```

### Multivariate Confounding

Model complex confounding patterns:

```python
# Three variables with pairwise correlations
group = GroupNoise(
    nodes=("X", "Y", "Z"),
    sampler=lambda rng, n: rng.multivariate_normal(
        mean=[0, 0, 0],
        cov=[
            [1.0, 0.3, 0.2],  # X with Y, Z
            [0.3, 1.0, 0.4],  # Y with X, Z
            [0.2, 0.4, 1.0]   # Z with X, Y
        ],
        size=n
    )
)
```

### DAG Validation

The system automatically validates that your graph is acyclic:

```python
# This will raise ValueError: "Graph has a cycle"
specs = [
    NodeSpec("X", ["Y"], "Y + eps_X"),  # X depends on Y
    NodeSpec("Y", ["X"], "X + eps_Y"),  # Y depends on X (cycle!)
]
# ValueError raised during SCMDataset initialization
```

---

## Best Practices

### 1. Node Naming

- Use descriptive names: `Temperature`, `Pressure` instead of `X1`, `X2`
- For hierarchical structures: `Parent1`, `Child1`, etc.
- Consistent naming convention across datasets

### 2. Structural Equations

- Start simple, add complexity gradually
- Include noise terms `eps_<name>` in every equation
- Test equations with small samples first
- Document the meaning of parameters

### 3. Noise Configuration

- Match noise distribution to data type:
  - Continuous unbounded: `standard_normal`
  - Continuous bounded: `uniform`
  - Positive only: `exponential`, `gamma`
- Scale noise appropriately relative to signal
- Use correlated noise only when modeling confounding

### 4. Dataset Generation

- Start with small `n` (e.g., 1000) to test
- Use descriptive `save_dir` names
- Include metadata in `meta_dict`:
  ```python
  meta_dict = {
      "purpose": "Testing confounding effects",
      "parameters_tested": ["w1", "w2"],
      "notes": "Special configuration for X->Y relationship"
  }
  dataset.generate_ds(
      mode="flat",
      n=5000,
      save_dir="data/my_dataset",
      meta_dict=meta_dict
  )
  ```

### 5. Reproducibility

- Always specify `seed` parameter for reproducibility
- Document the seed value in metadata
- Version control your dataset specification files
- Keep a record of all parameters used

### 6. Performance

- For large datasets (n > 100,000), generation may take time
- Consider generating datasets in batches
- Use compressed storage (`.npz`) to save disk space
- Pre-generate and cache datasets for experiments

---

## Troubleshooting

### Issue: "Graph has a cycle"

**Cause:** Your node specifications create a cyclic dependency.

**Solution:**
1. Check your `NodeSpec` definitions for cycles
2. Ensure parents are defined before children
3. Verify that no node depends on itself through a chain

**Example of problematic code:**
```python
# BAD: Creates cycle X -> Y -> Z -> X
specs = [
    NodeSpec("X", ["Z"], "Z + eps_X"),
    NodeSpec("Y", ["X"], "X + eps_Y"),
    NodeSpec("Z", ["Y"], "Y + eps_Z")
]
```

**Fix:**
```python
# GOOD: No cycles
specs = [
    NodeSpec("X", [], "eps_X"),
    NodeSpec("Y", ["X"], "X + eps_Y"),
    NodeSpec("Z", ["Y"], "Y + eps_Z")
]
```

### Issue: "KeyError: Node not found"

**Cause:** A parent node referenced in a `NodeSpec` doesn't exist.

**Solution:**
1. Check spelling of parent names
2. Ensure all parent nodes are defined in `specs`
3. Verify parent names match exactly (case-sensitive)

**Example:**
```python
# BAD: "x" (lowercase) doesn't match "X" (uppercase)
NodeSpec("Y", ["x"], "x + eps_Y")

# GOOD: Names match
NodeSpec("Y", ["X"], "X + eps_Y")
```

### Issue: "Shape mismatch in noise sampler"

**Cause:** A `GroupNoise` sampler returns the wrong shape.

**Solution:**
1. Ensure sampler returns array of shape `(n, k)` where `k = len(nodes)`
2. Check that `mean` has length `k`
3. Verify `cov` is `k x k` matrix

**Example fix:**
```python
# BAD: Returns shape (n,) instead of (n, 2)
GroupNoise(
    nodes=("X", "Y"),
    sampler=lambda rng, n: rng.standard_normal(n)
)

# GOOD: Returns shape (n, 2)
GroupNoise(
    nodes=("X", "Y"),
    sampler=lambda rng, n: rng.multivariate_normal(
        mean=[0, 0],      # Length 2
        cov=[[1, 0.3],    # 2x2 matrix
             [0.3, 1]],
        size=n
    )
)
```

### Issue: Missing noise sampler for a node

**Cause:** A node needs a noise term but no sampler is provided.

**Solution:**
1. Add the node to `singles` dictionary
2. Or include it in a `GroupNoise` group
3. Ensure every node has a noise source

**Example:**
```python
# Ensure all nodes have noise samplers
all_nodes = ["X", "Y", "Z"]
singles = {node: lambda rng, n: rng.standard_normal(n) 
           for node in all_nodes}
```

### Issue: "AttributeError: 'dict' object has no attribute 'sample_all'"

**Cause:** Passing a dict directly instead of a `NoiseModel` when needed.

**Solution:**
The `SCMDataset` class handles this automatically, but if working with `SCM` directly:

```python
# Use NoiseModel wrapper
noise_model = NoiseModel(singles=singles, groups=groups)
scm = SCM(specs, noise_model=noise_model)
```

### Issue: Generated data looks wrong

**Debugging steps:**
1. Generate a small sample (n=10) and inspect manually
2. Check structural equations are correct
3. Verify noise scales are appropriate
4. Plot the relationships to visualize
5. Test with simple linear relationships first

**Example debug script:**
```python
# Generate small sample
dataset = SCMDataset(...)
df = dataset.sample(n=10)
print(df)

# Check relationships
import matplotlib.pyplot as plt
df = dataset.sample(n=1000)
plt.scatter(df["X"], df["Y"])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

### Issue: Graphviz not rendering

**Cause:** Graphviz system binary not installed.

**Solution:**
1. Install Graphviz: 
   - Ubuntu/Debian: `sudo apt-get install graphviz`
   - macOS: `brew install graphviz`
   - Windows: Download from https://graphviz.org/download/
2. Ensure `dot` command is in system PATH
3. Reinstall Python graphviz: `pip install graphviz`

### Performance: Dataset generation is slow

**Solutions:**
1. Reduce sample size `n` for testing
2. Simplify structural equations
3. Use vectorized NumPy operations in noise samplers
4. Pre-compile equations (automatic in the system)

### Getting Help

If you encounter issues not covered here:

1. Check the examples in `scm_ds/datasets.py`
2. Review the docstrings in `scm_ds/scm.py`
3. Test with minimal examples first
4. Verify all imports are correct

---

## Summary

You now have a complete guide to creating SCM datasets:

1. **Design** your causal graph structure
2. **Define** nodes with `NodeSpec` objects
3. **Configure** noise with `NoiseModel` (singles and groups)
4. **Create** the `SCMDataset` object
5. **Generate** and save with `generate_ds()`
6. **Use** the output files in your experiments

Key files:
- **Code**: `scm_ds/scm.py` (engine), `scm_ds/datasets.py` (examples)
- **Output**: `ds.npz` (data), `*.csv` (masks), `*.json` (metadata), `graph.pdf` (visualization)

For more examples, see the pre-defined datasets in `scm_ds/datasets.py`:
- `ds_scm_1_to_1_ct`: One-to-one with cross-talk
- `ds_scm_1_to_1_ct_2`: Alternative configuration with different parent relationships

Happy dataset generation! 🎲📊
