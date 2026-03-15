from os.path import abspath, dirname, join
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import *


# =============================================================================
# FINAL DATASETS FOR PAPER
# =============================================================================
# Structure covers variety in S→X and X→X relations:
#
# S→X relations:
#   - dangling:     S1 (no children)
#   - one-to-one:   S2 → X1
#   - one-to-many:  S3 → X2, X3
#   - many-to-one:  S4, S5 → X4
#
# X→X relations:
#   - dangling:     X3 (no X children)
#   - one-to-one:   X1 → X5
#   - one-to-many:  X2 → X4, X5
#   - many-to-one:  X1, X2 → X5
# =============================================================================

# -----------------------------------------------------------------------------
# ds_scm1: Linear Gaussian
# -----------------------------------------------------------------------------
ds_scm1 = SCMDataset(
    name="linear_gaussian",
    description="Linear SCM with Gaussian noise. Covers all S→X and X→X relation types.",
    tags=["linear", "gaussian", "paper"],
    specs=[
        # Source nodes (S)
        NodeSpec("S1", [], "eps_S1"),                           # dangling (no children)
        NodeSpec("S2", [], "eps_S2"),                           # one-to-one → X1
        NodeSpec("S3", [], "eps_S3"),                           # one-to-many → X2, X3
        NodeSpec("S4", [], "eps_S4"),                           # many-to-one (with S5) → X4
        NodeSpec("S5", [], "eps_S5"),                           # many-to-one (with S4) → X4
        # Feature nodes (X)
        NodeSpec("X1", ["S2"],           "a*S2 + eps_X1"),                      # S: one-to-one, X: one-to-one → X5
        NodeSpec("X2", ["S3"],           "b*S3 + eps_X2"),                      # S: one-to-many, X: one-to-many → X4, X5
        NodeSpec("X3", ["S3"],           "c*S3 + eps_X3"),                      # S: one-to-many, X: dangling
        NodeSpec("X4", ["S4", "S5", "X2"], "d*S4 + e*S5 + f*X2 + eps_X4"),      # S: many-to-one, X: many-to-one
        NodeSpec("X5", ["X1", "X2"],     "g*X1 + h*X2 + eps_X5"),               # X: many-to-one
    ],
    params={
        "a": 1.0,
        "b": 1.0,
        "c": 0.8,
        "d": 0.7,
        "e": 0.5,
        "f": 1.2,
        "g": 1.0,
        "h": 0.6,
    },
    singles={
        "S1": lambda rng, n: rng.uniform(-1, 1, n),
        "S2": lambda rng, n: rng.uniform(-1, 1, n),
        "S3": lambda rng, n: rng.uniform(-1, 1, n),
        "S4": lambda rng, n: rng.uniform(-1, 1, n),
        "S5": lambda rng, n: rng.uniform(-1, 1, n),
        "X1": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X2": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X3": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X4": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X5": lambda rng, n: 0.1 * rng.standard_normal(n),
    },
    groups=None,
    source_labels=["S1", "S2", "S3", "S4", "S5"],
    input_labels=["X1", "X2", "X3", "X4", "X5"],
    target_labels=[]
)


# -----------------------------------------------------------------------------
# ds_scm2: Non-linear Gaussian
# -----------------------------------------------------------------------------
ds_scm2 = SCMDataset(
    name="nonlinear_gaussian",
    description="Non-linear SCM with Gaussian noise. Uses polynomial and trigonometric functions.",
    tags=["nonlinear", "gaussian", "paper"],
    specs=[
        # Source nodes (S)
        NodeSpec("S1", [], "eps_S1"),                           # dangling (no children)
        NodeSpec("S2", [], "eps_S2"),                           # one-to-one → X1
        NodeSpec("S3", [], "eps_S3"),                           # one-to-many → X2, X3
        NodeSpec("S4", [], "eps_S4"),                           # many-to-one (with S5) → X4
        NodeSpec("S5", [], "eps_S5"),                           # many-to-one (with S4) → X4
        # Feature nodes (X) - non-linear relations
        NodeSpec("X1", ["S2"],           "a*S2**2 + eps_X1"),                                   # quadratic
        NodeSpec("X2", ["S3"],           "b*sin(S3*3.14159) + eps_X2"),                         # sinusoidal
        NodeSpec("X3", ["S3"],           "c*S3**3 + eps_X3"),                                   # cubic
        NodeSpec("X4", ["S4", "S5", "X2"], "d*S4**2 + e*S5*X2 + f*X2**2 + eps_X4"),             # interaction terms
        NodeSpec("X5", ["X1", "X2"],     "g*tanh(X1) + h*X2**2 + eps_X5"),                      # tanh + quadratic
    ],
    params={
        "a": 1.5,
        "b": 1.0,
        "c": 0.5,
        "d": 0.8,
        "e": 1.2,
        "f": 0.6,
        "g": 2.0,
        "h": 0.8,
    },
    singles={
        "S1": lambda rng, n: rng.uniform(-1, 1, n),
        "S2": lambda rng, n: rng.uniform(-1, 1, n),
        "S3": lambda rng, n: rng.uniform(-1, 1, n),
        "S4": lambda rng, n: rng.uniform(-1, 1, n),
        "S5": lambda rng, n: rng.uniform(-1, 1, n),
        "X1": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X2": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X3": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X4": lambda rng, n: 0.1 * rng.standard_normal(n),
        "X5": lambda rng, n: 0.1 * rng.standard_normal(n),
    },
    groups=None,
    source_labels=["S1", "S2", "S3", "S4", "S5"],
    input_labels=["X1", "X2", "X3", "X4", "X5"],
    target_labels=[]
)


# -----------------------------------------------------------------------------
# ds_scm3: Non-linear Non-Gaussian
# -----------------------------------------------------------------------------
ds_scm3 = SCMDataset(
    name="nonlinear_nongaussian",
    description="Non-linear SCM with non-Gaussian noise (uniform + exponential + lognormal).",
    tags=["nonlinear", "nongaussian", "paper"],
    specs=[
        # Source nodes (S)
        NodeSpec("S1", [], "eps_S1"),                           # dangling (no children)
        NodeSpec("S2", [], "eps_S2"),                           # one-to-one → X1
        NodeSpec("S3", [], "eps_S3"),                           # one-to-many → X2, X3
        NodeSpec("S4", [], "eps_S4"),                           # many-to-one (with S5) → X4
        NodeSpec("S5", [], "eps_S5"),                           # many-to-one (with S4) → X4
        # Feature nodes (X) - non-linear relations (same as ds_scm2)
        NodeSpec("X1", ["S2"],           "a*S2**2 + eps_X1"),                                   # quadratic
        NodeSpec("X2", ["S3"],           "b*sin(S3*3.14159) + eps_X2"),                         # sinusoidal
        NodeSpec("X3", ["S3"],           "c*S3**3 + eps_X3"),                                   # cubic
        NodeSpec("X4", ["S4", "S5", "X2"], "d*S4**2 + e*S5*X2 + f*X2**2 + eps_X4"),             # interaction terms
        NodeSpec("X5", ["X1", "X2"],     "g*tanh(X1) + h*X2**2 + eps_X5"),                      # tanh + quadratic
    ],
    params={
        "a": 1.5,
        "b": 1.0,
        "c": 0.5,
        "d": 0.8,
        "e": 1.2,
        "f": 0.6,
        "g": 2.0,
        "h": 0.8,
    },
    singles={
        # Source nodes: uniform noise
        "S1": lambda rng, n: rng.uniform(-1, 1, n),
        "S2": lambda rng, n: rng.uniform(-1, 1, n),
        "S3": lambda rng, n: rng.uniform(-1, 1, n),
        "S4": lambda rng, n: rng.uniform(-1, 1, n),
        "S5": lambda rng, n: rng.uniform(-1, 1, n),
        # Feature nodes: non-Gaussian noise (centered)
        "X1": lambda rng, n: 0.1 * (rng.uniform(-1, 1, n)),                           # uniform
        "X2": lambda rng, n: 0.1 * (rng.exponential(1.0, n) - 1.0),                   # exponential (centered)
        "X3": lambda rng, n: 0.1 * (rng.lognormal(0, 0.5, n) - 1.0),                  # lognormal (centered)
        "X4": lambda rng, n: 0.1 * (rng.laplace(0, 1, n)),                            # laplace
        "X5": lambda rng, n: 0.1 * (rng.uniform(-1, 1, n)),                           # uniform
    },
    groups=None,
    source_labels=["S1", "S2", "S3", "S4", "S5"],
    input_labels=["X1", "X2", "X3", "X4", "X5"],
    target_labels=[]
)


# =============================================================================
# LEGACY DATASETS (kept for backward compatibility)
# =============================================================================

ds_scm_1_to_1_ct = SCMDataset(
    name = "one-to-one_with_crosstalk",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("P1", [], "eps_P1"),                  # parent 1
        NodeSpec("P2", [], "eps_P2"),                  # parent 2
        NodeSpec("P3", [], "eps_P3"),                  # parent 3
        NodeSpec("P4", [], "eps_P4"),                  # parent 4
        NodeSpec("P5", [], "eps_P5"),                  # parent 5
        NodeSpec("C1", ["P1"], "P1 + eps_C1"),                  # child 1
        NodeSpec("C2", ["P2"], "P2 + eps_C2"),                  # child 2
        NodeSpec("C3", ["P3"], "P3 + eps_C3"),                  # child 3
        NodeSpec("C4", ["P4"], "P4 + eps_C4"),                  # child 4
        NodeSpec("C5", ["P5"], "P5 + eps_C5"),                  # child 5
        # output
        NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],    "C1 + C2 + C3 + C4 + C5 + eps_Y"),     
        ],
    params = {
        "w1": 0.01,
        "w2": 0.01,
        "w3": 0.01,
        "w4": 0.01,
        "w5": 0.01,
        },
    singles = {
        "P1": lambda rng,n: rng.standard_normal(n),
        "P2": lambda rng,n: rng.standard_normal(n),
        "P3": lambda rng,n: rng.standard_normal(n),
        "P4": lambda rng,n: rng.standard_normal(n),
        "P5": lambda rng,n: rng.standard_normal(n),
        "C1": lambda rng,n: rng.standard_normal(n),
        "C2": lambda rng,n: rng.standard_normal(n),
        "C3": lambda rng,n: rng.standard_normal(n),
        "C4": lambda rng,n: rng.standard_normal(n),
        "C5": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.standard_normal(n),
        },
    groups=None,
    input_labels=[
        "P1", "P2", "P3", "P4", "P5",
        "C1", "C2", "C3", "C4", "C5"],
    target_labels = ["Y"]
    )


ds_scm_1_to_1_ct_2 = SCMDataset(
    name = "one-to-one_with_crosstalk",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("P1", [], "eps_P1"),                  # parent 1
        NodeSpec("P2", [], "eps_P2"),                  # parent 2
        NodeSpec("P3", [], "eps_P3"),                  # parent 3
        NodeSpec("P4", [], "eps_P4"),                  # parent 4
        NodeSpec("P5", [], "eps_P5"),                  # parent 5
        NodeSpec("C1", ["P1", "P2"], "P1 - P2 + eps_C1"),                  # child 1
        NodeSpec("C2", ["P2"], "P2 + eps_C2"),                  # child 2
        NodeSpec("C3", ["P3"], "P3 + eps_C3"),                  # child 3
        NodeSpec("C4", ["P4"], "P4 + eps_C4"),                  # child 4
        NodeSpec("C5", ["P5"], "P5 + eps_C5"),                  # child 5
        # output
        NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],    "C1 + C2 + C3 + C4 + C5 + eps_Y"),     
        ],
    params = {
        "w1": 0.01,
        "w2": 0.01,
        "w3": 0.01,
        "w4": 0.01,
        "w5": 0.01,
        },
    singles = {
        "P1": lambda rng,n: rng.standard_normal(n),
        "P2": lambda rng,n: rng.standard_normal(n),
        "P3": lambda rng,n: rng.standard_normal(n),
        "P4": lambda rng,n: rng.standard_normal(n),
        "P5": lambda rng,n: rng.standard_normal(n),
        "C1": lambda rng,n: rng.standard_normal(n),
        "C2": lambda rng,n: rng.standard_normal(n),
        "C3": lambda rng,n: rng.standard_normal(n),
        "C4": lambda rng,n: rng.standard_normal(n),
        "C5": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.standard_normal(n),
        },
    groups=None,
    input_labels=[
        "P1", "P2", "P3", "P4", "P5",
        "C1", "C2", "C3", "C4", "C5"],
    target_labels = ["Y"]
    )


ds_scm4 = SCMDataset(
    name = "mid linear Gaussian",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                            # input 1
        NodeSpec("X2", [], "eps_X2"),                 # input 2
        NodeSpec("X3", ["X1", "X2"], "b*X1 + c*X2 + eps_X3"),    # input 3
        NodeSpec("Y1", ["X1", "X3"], "f*X1 + g*X3 + eps_Y1"),    # target 1
        NodeSpec("Y2", ["X3"      ], "h*X3 + eps_Y2"),           # target 2
        NodeSpec("Y3", ["X3", "Y2"], "j*X3 + k*Y2 + eps_Y3"),    # target 3
        ],
    params = {
        "a": 1,
        "b": 1,
        "c": 1,
        "f": 1,
        "g": 1,
        "h": 1,
        "j": 1,
        "k": 1,
        },
    singles = {
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        "X3": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y3": lambda rng,n: 0.05*rng.standard_normal(n),
        },
    groups=None,
    input_labels=["X1", "X2", "X3"],
    target_labels = ["Y1", "Y2", "Y3"]
    )


ds_scm5 = SCMDataset(
    name = "mid linear Gaussian",
    description ="Children with different ancestors",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                           # input 1
        NodeSpec("X2", [], "eps_X2"),                           # input 2
        NodeSpec("X3", ["X1", "X2"], "b*X1 + c*X2 + eps_X3"),   # input 3
        NodeSpec("Y1", ["X1"], "f*X1 + eps_Y1"),                # target 1
        NodeSpec("Y2", ["X3"      ], "h*X3 + eps_Y2"),          # target 2
        NodeSpec("Y3", ["X3", "Y2"], "j*X3 + k*Y2 + eps_Y3"),   # target 3
        ],
    params = {
        "a": 1,
        "b": 1,
        "c": 1,
        "f": 1,
        "g": 1,
        "h": 1,
        "j": 1,
        "k": 1,
        },
    singles = {
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        "X3": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y3": lambda rng,n: 0.05*rng.standard_normal(n),
        },
    groups=None,
    input_labels=["X1", "X2", "X3"],
    target_labels = ["Y1", "Y2", "Y3"]
    )


ds_scm6 = SCMDataset(
    name = "linear Gaussian",
    description ="Source nodes",
    tags=None,
    specs = [
        NodeSpec("S1", [], "eps_S1"),                           # source 1
        NodeSpec("S2", [], "eps_S2"),                           # source 2
        NodeSpec("S3", [], "eps_S3"),                           # source 3
        NodeSpec("X1", ["S1", "X2"], "a*S1 + b*X2 + eps_X1"),   # input 1
        NodeSpec("X2", ["S2", "S3"], "c*S2 + f*S3 + eps_X2"),   # input 2
        NodeSpec("Y1", ["X1"], "g*X1 + eps_Y1"),                # target 1
        NodeSpec("Y2", ["X2"], "h*X2 + eps_Y2"),                # target 2
        ],
    params = {
        "a": 1,
        "b": 1,
        "c": 1,
        "f": 1,
        "g": 1,
        "h": 1,
        },
    singles = {
        "S1": lambda rng,n: rng.uniform(-1, 1, n),
        "S2": lambda rng,n: rng.uniform(-1, 1, n),
        "S3": lambda rng,n: rng.uniform(-1, 1, n),
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        },
    groups=None,
    source_labels=["S1", "S2", "S3"],
    input_labels=["X1", "X2"],
    target_labels = ["Y1", "Y2"]
    )


ds_scm7 = SCMDataset(
    name = "non-linear Gaussian",
    description ="non-linear version of scm6 with different weights",
    tags=None,
    specs = [
        NodeSpec("S1", [], "eps_S1"),                               # source 1
        NodeSpec("S2", [], "eps_S2"),                               # source 2
        NodeSpec("S3", [], "eps_S3"),                               # source 3
        NodeSpec("X1", ["S1", "X2"], "a*S1^2 + b*X2^5 + eps_X1"), # input 1
        NodeSpec("X2", ["S2", "S3"], "c*S2 + f*S3^3 + eps_X2"),     # input 2
        NodeSpec("Y1", ["X1"], "g*X1 + eps_Y1"),                    # target 1
        NodeSpec("Y2", ["X2"], "h*X2 + eps_Y2"),                    # target 2
        ],
    params = {
        "a": 1,
        "b": 7,
        "c": 0.5,
        "f": 1,
        "g": 1,
        "h": 1,
        },
    singles = {
        "S1": lambda rng,n: rng.uniform(-1, 1, n),
        "S2": lambda rng,n: rng.uniform(-1, 1, n),
        "S3": lambda rng,n: rng.uniform(-1, 1, n),
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        },
    groups=None,
    source_labels=["S1", "S2", "S3"],
    input_labels=["X1", "X2"],
    target_labels = ["Y1", "Y2"]
    )


ds_scm8 = SCMDataset(
    name = "linear Gaussian",
    description ="Source nodes, fixed recipe",
    tags=None,
    specs = [
        NodeSpec("S1", [], "eps_S1"),                           # source 1
        NodeSpec("S2", [], "eps_S2"),                           # source 2
        NodeSpec("S3", [], "eps_S3"),                           # source 3
        NodeSpec("X1", ["S1", "X2"], "a*S1 + b*X2 + eps_X1"),   # input 1
        NodeSpec("X2", ["S2", "S3"], "c*S2 + f*S3 + eps_X2"),   # input 2
        NodeSpec("Y1", ["X1"], "g*X1 + eps_Y1"),                # target 1
        NodeSpec("Y2", ["X2"], "h*X2 + eps_Y2"),                # target 2
        ],
    params = {
        "a": 1,
        "b": 1,
        "c": 1,
        "f": 1,
        "g": 1,
        "h": 1,
        },
    singles = {
        "S1": lambda rng,n: rng.choice([1, 1.5, 3], size=n),
        "S2": lambda rng,n: rng.choice([2, 2.5, 3], size=n),
        "S3": lambda rng,n: rng.choice([0.4, 0.5, 0.6], size=n),
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        },
    groups=None,
    source_labels=["S1", "S2", "S3"],
    input_labels=["X1", "X2"],
    target_labels = ["Y1", "Y2"]
    )


# =============================================================================
# DATASET GENERATION (uncomment to generate)
# =============================================================================

if __name__ == "__main__":
    # Generate the three final datasets for the paper
    ds_scm1.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm1_linear_gaussian"), 
        normalize_method="minmax", 
        shared_embedding=False,
    )
    
    ds_scm2.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm2_nonlinear_gaussian"), 
        normalize_method="minmax", 
        shared_embedding=False,
    )
    
    ds_scm3.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm3_nonlinear_nongaussian"), 
        normalize_method="minmax", 
        shared_embedding=False,
    )
    
    ds_scm6.generate_ds(
        mode="flat", 
        n=50000, 
        save_dir=join(ROOT_DIR,"data/scm6"),
        normalize_method="minmax",  
        shared_embedding=False,
    )

