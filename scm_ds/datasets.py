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
    description="Linear SCM with Gaussian noise. Uniformly sampled S variables. Covers all S→X and X→X relation types.",
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

ds_scm1_discrete_sampling = SCMDataset(
    name="linear_gaussian",
    description="Linear SCM with Gaussian noise. Discrete sampling S variables. Covers all S→X and X→X relation types.",
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
        # NOTE: 0 is included in all S distributions to ensure baseline do(S=0) is in-distribution
        "S1": lambda rng, n: rng.choice([0, 0.5, -1.2], n),                                            # 3 elements (added 0)
        "S2": lambda rng, n: rng.choice([0, -1.7, 3.0, -2.0], n),                                      # 4 elements (added 0)
        "S3": lambda rng, n: rng.choice([0, 1.0, 2.5, 2, -0.5, 0.8, -0.1, -2, -2.5, -2.7, -3], n),     # 11 elements (added 0)
        "S4": lambda rng, n: rng.choice([0, -0.3, 1.5, 2.0, -1.0, 0.7], n),                            # 6 elements (added 0)
        "S5": lambda rng, n: rng.choice([0, 0.2, -0.8, 1.8, -1.5, 2.5, -2, -2.2, -2.5, -2.6, -2.7], n),# 11 elements (added 0)
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

# ds_scm2 with discrete S sampling (same discrete values as scm1 for consistency)
ds_scm2_discrete_sampling = SCMDataset(
    name="nonlinear_gaussian_discrete",
    description="Non-linear SCM with Gaussian noise. Discrete sampling S variables. Uses polynomial and trigonometric functions.",
    tags=["nonlinear", "gaussian", "paper", "discrete"],
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
        # NOTE: 0 is included in all S distributions to ensure baseline do(S=0) is in-distribution
        "S1": lambda rng, n: rng.choice([0, 0.5, -1.2], n),                                            # 3 elements (added 0)
        "S2": lambda rng, n: rng.choice([0, -1.7, 3.0, -2.0], n),                                      # 4 elements (added 0)
        "S3": lambda rng, n: rng.choice([0, 1.0, 2.5, 2, -0.5, 0.8, -0.1, -2, -2.5, -2.7, -3], n),     # 11 elements (added 0)
        "S4": lambda rng, n: rng.choice([0, -0.3, 1.5, 2.0, -1.0, 0.7], n),                            # 6 elements (added 0)
        "S5": lambda rng, n: rng.choice([0, 0.2, -0.8, 1.8, -1.5, 2.5, -2, -2.2, -2.5, -2.6, -2.7], n),# 11 elements (added 0)
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

# ds_scm3 with discrete S sampling (same discrete values as scm1/scm2 for consistency)
ds_scm3_discrete_sampling = SCMDataset(
    name="nonlinear_nongaussian_discrete",
    description="Non-linear SCM with non-Gaussian noise. Discrete sampling S variables.",
    tags=["nonlinear", "nongaussian", "paper", "discrete"],
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
        # NOTE: 0 is included in all S distributions to ensure baseline do(S=0) is in-distribution
        "S1": lambda rng, n: rng.choice([0, 0.5, -1.2], n),                                            # 3 elements (added 0)
        "S2": lambda rng, n: rng.choice([0, -1.7, 3.0, -2.0], n),                                      # 4 elements (added 0)
        "S3": lambda rng, n: rng.choice([0, 1.0, 2.5, 2, -0.5, 0.8, -0.1, -2, -2.5, -2.7, -3], n),     # 11 elements (added 0)
        "S4": lambda rng, n: rng.choice([0, -0.3, 1.5, 2.0, -1.0, 0.7], n),                            # 6 elements (added 0)
        "S5": lambda rng, n: rng.choice([0, 0.2, -0.8, 1.8, -1.5, 2.5, -2, -2.2, -2.5, -2.6, -2.7], n),# 11 elements (added 0)
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
# ds_scm_equal: EQUAL-STRENGTH CONTROL (multi-parent capacity test)
# =============================================================================
# WHY THIS DATASET EXISTS
# -----------------------
# In scm1/2/3 the multi-parent children keep only ONE parent (e.g. X4 keeps S4
# but drops S5).  Two explanations are confounded there:
#   (H1) the parents are UNEQUALLY strong, so the weak one carries too little
#        signal for HSIC/R2 to care about it;
#   (H2) the selector mechanism itself prefers in-degree 1 (directional budget,
#        gate threshold, L0), independently of the data.
# scm3 cannot separate them, because its nominal coefficients hide a large
# imbalance.  For X4 = 0.8*S4**2 + 1.2*S5*X2 + 0.6*X2**2 + eps  (S ~ U(-1,1)):
#   term 0.8*S4**2  -> Var 0.057  (18% of Var(X4))
#   term 1.2*S5*X2  -> Var 0.197  (62%)
#   term 0.6*X2**2  -> Var 0.043  (14%)
# i.e. coefficients differing by only 1.5x produce a ~4x imbalance between the
# S4 and S5 PARENTS, because Var(S**2) != Var(S*X2) and X2 is a generated node.
# This dataset removes H1 by construction so that any remaining in-degree-1
# behaviour must be H2.
#
# DESIGN RULES (each one kills a specific confound)
# ------------------------------------------------
# 1. EXCHANGEABLE CO-PARENTS.  Every multi-parent child gets parents that are
#    i.i.d. AND mutually independent, entering through the SAME function with
#    the SAME coefficient.  Their contributions are then equal *exactly, by
#    symmetry* -- no moment algebra, no calibration script, hand-checkable.
# 2. ODD + MONOTONE + BOUNDED NONLINEARITY  f(x) = tanh(2x).
#    - odd    => E[f] = 0, so parent terms are mean-zero and (being independent)
#                their variances ADD: "share of Var(child)" is exact.
#    - monotone => no information destroyed (x**2 folds +x and -x together, so a
#                parent would only ever be half-identifiable).
#    - bounded  => f(f(.)) stays O(1); x**2 composed twice is quartic, with
#                heavy tails that would also disturb the HSIC median bandwidth.
#    - gain 2 => genuinely nonlinear (tanh(2) = 0.964, visibly saturating).
# 3. NO SHARED ANCESTORS.  S1->X1, S2->X2, S3->X3 are PRIVATE chains and
#    S4, S5 feed only X4.  Therefore no variable is a partial substitute for
#    any other, and no spurious edge can be excused as "an almost-equivalent
#    parent".  (In scm3, X2 = S3 + tiny noise, so S3 is a ~sufficient stand-in
#    for X2 -- which is exactly why X5 learned the spurious S3 edge.)
# 4. NOISE SHARE 0.5 ON THE INTERMEDIATE NODES X1, X2, X3.  Each explains only
#    half of its child, so using the S-ancestor instead of the true X-parent is
#    a real, measurable loss.  (In scm3 that share was ~1%, so the substitution
#    was almost free.)
# 5. STILL 5 S + 5 X.  The query-fanin budget F = n * x_sat**2 = 68.69 scales
#    with the node count n = 10, so the F=69 arms transfer unchanged.
#
# COEFFICIENTS (all derived by hand; see scripts/build_scm_equal.py for the
# verification that these realise the intended shares)
# ----------------------------------------------------------------------------
#   S ~ U(-1,1)  =>  Var(f(S)) = 1 - tanh(2)/2 = 0.5179862  (closed form:
#                    E[f] = 0 by symmetry and int_0^1 tanh^2(2s) ds = 1 - tanh(2)/2)
#   kp = sqrt(0.50 / 0.5179862) = 0.982485   -> X1,X2,X3: parent share 0.50, noise 0.50
#   c4 = sqrt(0.45 / 0.5179862) = 0.932067   -> X4: 0.45 + 0.45 parents, 0.10 noise
#   c5 = sqrt(0.30 / Var(f(X1))) = 0.674930  -> X5: 0.30 x 3 parents, 0.10 noise
# Var(f(X1)) = 0.658572 is the ONLY quantity without a closed form (tanh of a
# bounded+Gaussian mixture); it was measured once by Monte Carlo (n = 4e6, seed 0)
# and baked here.  Note it affects only X5's overall SNR, NOT the balance among
# X1/X2/X3, which is exact by symmetry regardless of c5.
# Every node therefore has Var = 1 by construction.
#
# STRUCTURE
# ---------
#   X1 <- S1            X2 <- S2            X3 <- S3        (private chains)
#   X4 <- S4, S5        CROSS fan-in 2, parents exchangeable (0.45 each)
#   X5 <- X1, X2, X3    SELF  fan-in 3, parents exchangeable (0.30 each)
# X4 has NO X-parent and X5 has NO S-parent, so each block also gets a row that
# must stay empty.
#
# WHAT THE OUTCOME MEANS
# ----------------------
#   both S4,S5 on X4 and all of X1,X2,X3 on X5 recovered
#       -> the scm3 misses were a strength/detectability effect; the selector's
#          multi-parent capacity is fine and the open question moves to
#          weak-parent detectability.
#   still ~in-degree 1 with provably exchangeable parents
#       -> the in-degree-1 preference is INTRINSIC to the mechanism; parent
#          strength was never the cause.  This is the outcome that would most
#          change what we do next.
# NOTE: this dataset is additive and monotone, hence EASIER than scm3 (no
# interaction term).  It is a capacity CONTROL, not a difficulty benchmark:
# a failure here is conclusive, a success does not by itself prove scm3 would
# work.
# =============================================================================
ds_scm_equal = SCMDataset(
    name="equal_strength_tanh",
    description=(
        "Control SCM for multi-parent capacity: tanh(2x) mechanisms, every "
        "multi-parent child has i.i.d. mutually-independent (exchangeable) "
        "parents contributing EXACTLY equal variance shares, no shared "
        "ancestors, and 50% private noise on the intermediate nodes so that an "
        "S-ancestor is not a sufficient substitute for an X-parent."
    ),
    tags=["nonlinear", "gaussian", "control", "equal_strength"],
    specs=[
        # Source nodes (S): i.i.d. U(-1,1)
        NodeSpec("S1", [], "eps_S1"),                    # private -> X1
        NodeSpec("S2", [], "eps_S2"),                    # private -> X2
        NodeSpec("S3", [], "eps_S3"),                    # private -> X3
        NodeSpec("S4", [], "eps_S4"),                    # many-to-one (with S5) -> X4
        NodeSpec("S5", [], "eps_S5"),                    # many-to-one (with S4) -> X4
        # Private chains: parent share 0.50, noise share 0.50 (breaks substitution)
        NodeSpec("X1", ["S1"], "kp*tanh(2*S1) + eps_X1"),
        NodeSpec("X2", ["S2"], "kp*tanh(2*S2) + eps_X2"),
        NodeSpec("X3", ["S3"], "kp*tanh(2*S3) + eps_X3"),
        # CROSS fan-in 2: S4, S5 are i.i.d. U(-1,1) => exactly 0.45 each
        NodeSpec("X4", ["S4", "S5"], "c4*(tanh(2*S4) + tanh(2*S5)) + eps_X4"),
        # SELF fan-in 3: X1, X2, X3 are i.i.d. by construction => exactly 0.30 each
        NodeSpec("X5", ["X1", "X2", "X3"], "c5*(tanh(2*X1) + tanh(2*X2) + tanh(2*X3)) + eps_X5"),
    ],
    params={
        "kp": 0.982485,   # sqrt(0.50 / (1 - tanh(2)/2))
        "c4": 0.932067,   # sqrt(0.45 / (1 - tanh(2)/2))
        "c5": 0.674930,   # sqrt(0.30 / Var(tanh(2*X1))), Var = 0.658572 (MC, n=4e6, seed 0)
    },
    singles={
        # Sources: continuous uniform (as in the *_continuous variants)
        "S1": lambda rng, n: rng.uniform(-1, 1, n),
        "S2": lambda rng, n: rng.uniform(-1, 1, n),
        "S3": lambda rng, n: rng.uniform(-1, 1, n),
        "S4": lambda rng, n: rng.uniform(-1, 1, n),
        "S5": lambda rng, n: rng.uniform(-1, 1, n),
        # Intermediates: Gaussian noise with variance 0.50 (= 50% noise share)
        "X1": lambda rng, n: 0.7071068 * rng.standard_normal(n),
        "X2": lambda rng, n: 0.7071068 * rng.standard_normal(n),
        "X3": lambda rng, n: 0.7071068 * rng.standard_normal(n),
        # Multi-parent children: Gaussian noise with variance 0.10 (= 10% noise share)
        "X4": lambda rng, n: 0.3162278 * rng.standard_normal(n),
        "X5": lambda rng, n: 0.3162278 * rng.standard_normal(n),
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
    # =========================================================================
    # PAPER DATASETS: Discrete S with Holdout Split
    # These are the default datasets for the paper experiments.
    # All use the same discrete S values for consistency across SCM types.
    # Holdout: S3=1.0 and S5=2.5 are reserved for OOD test evaluation.
    # =========================================================================
    
    # SCM1: Linear Gaussian (discrete holdout)
    print("\n" + "="*60)
    print("Generating SCM1: Linear Gaussian (discrete holdout)")
    print("="*60)
    ds_scm1_discrete_sampling.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm1"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "holdout",
            "kwargs": {
                "explicit_values": {
                    "S3": [1.0],
                    "S5": [2.5],
                }
            }
        }
    )
    
    # SCM2: Non-linear Gaussian (discrete holdout)
    print("\n" + "="*60)
    print("Generating SCM2: Non-linear Gaussian (discrete holdout)")
    print("="*60)
    ds_scm2_discrete_sampling.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm2"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "holdout",
            "kwargs": {
                "explicit_values": {
                    "S3": [1.0],
                    "S5": [2.5],
                }
            }
        }
    )
    
    # SCM3: Non-linear Non-Gaussian (discrete holdout)
    print("\n" + "="*60)
    print("Generating SCM3: Non-linear Non-Gaussian (discrete holdout)")
    print("="*60)
    ds_scm3_discrete_sampling.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm3"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "holdout",
            "kwargs": {
                "explicit_values": {
                    "S3": [1.0],
                    "S5": [2.5],
                }
            }
        }
    )
    
    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)
    
    # =========================================================================
    # DIAGNOSTIC: Continuous S sampling (for HSIC kernel analysis)
    # =========================================================================
    # Same SCM structures but with uniform S instead of discrete.
    # Used to test whether discrete S is causing HSIC optimization instability.
    # See: docs/CRITICAL_EXP.md — "D1 Extended: Discrete vs Continuous"
    
    print("\n" + "="*60)
    print("Generating SCM1 Continuous: Linear Gaussian (uniform S, ratio split)")
    print("="*60)
    ds_scm1.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm1_continuous"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "ratio",
            "kwargs": {
                "test_ratio": 0.2,
                "seed": 42,
            }
        }
    )
    
    print("\n" + "="*60)
    print("Generating SCM2 Continuous: Non-linear Gaussian (uniform S, ratio split)")
    print("="*60)
    ds_scm2.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm2_continuous"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "ratio",
            "kwargs": {
                "test_ratio": 0.2,
                "seed": 42,
            }
        }
    )
    
    print("\n" + "="*60)
    print("Generating SCM3 Continuous: Non-linear Non-Gaussian (uniform S, ratio split)")
    print("="*60)
    ds_scm3.generate_ds(
        mode="flat", 
        n=50_000, 
        save_dir=join(ROOT_DIR, "data/scm3_continuous"), 
        normalize_method="minmax", 
        shared_embedding=False,
        test_split_method={
            "method": "ratio",
            "kwargs": {
                "test_ratio": 0.2,
                "seed": 42,
            }
        }
    )
    
    # =========================================================================
    # LEGACY: Uniform S sampling (not used for paper)
    # =========================================================================
    # ds_scm1.generate_ds(
    #     mode="flat", 
    #     n=50_000, 
    #     save_dir=join(ROOT_DIR, "data/scm1_linear_gaussian"), 
    #     normalize_method="minmax", 
    #     shared_embedding=False,
    # )
    
    # ds_scm2.generate_ds(
    #     mode="flat", 
    #     n=50_000, 
    #     save_dir=join(ROOT_DIR, "data/scm2_nonlinear_gaussian"), 
    #     normalize_method="minmax", 
    #     shared_embedding=False,
    # )
    
    # ds_scm3.generate_ds(
    #     mode="flat", 
    #     n=50_000, 
    #     save_dir=join(ROOT_DIR, "data/scm3_nonlinear_nongaussian"), 
    #     normalize_method="minmax", 
    #     shared_embedding=False,
    # )

