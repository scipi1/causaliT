from os.path import abspath, dirname
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import *



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

ds_scm1 = SCMDataset(
    name = "one-to-one_with_crosstalk",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                               # parent
        NodeSpec("X2", ["X1"], "w12*X1 + eps_X2"),                  # child 1
        NodeSpec("Y", ["X1", "X2"], "w13*X1 + w23*X2 + eps_Y"),   # output
        ],
    params = {
        "w12": 7,
        "w13": 1,
        "w23": 3,
        },
    singles = {
        "X1": lambda rng,n: 0.1*rng.standard_normal(n),
        "X2": lambda rng,n: 0.1*rng.standard_normal(n),
        "Y": lambda rng,n: 0.1*rng.standard_normal(n),
        },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels = ["Y"]
    )


ds_scm2 = SCMDataset(
    name = "one-to-one_with_crosstalk_lognormal",
    description ="Every parent has one child and there is cross-talk between children (log-normal noise)",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                               # parent
        NodeSpec("X2", ["X1"], "w12*X1 + eps_X2"),                  # child 1
        NodeSpec("Y", ["X1", "X2"], "w13*X1 + w23*X2 + eps_Y"),   # output
        ],
    params = {
        "w12": 7,
        "w13": 1,
        "w23": 3,
        },
    singles = {
        "X1": lambda rng,n: 0.1*rng.lognormal(mean=0, sigma=0.5, size=n),
        "X2": lambda rng,n: 0.1*rng.lognormal(mean=0, sigma=0.5, size=n),
        "Y": lambda rng,n: 0.1*rng.lognormal(mean=0, sigma=0.5, size=n),
        },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels = ["Y"]
    )


ds_scm3 = SCMDataset(
    name = "simple non-linear Gaussian",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                               # parent
        NodeSpec("X2", ["X1"], "w12*(X1**2) + eps_X2"),                  # child 1
        NodeSpec("Y", ["X1", "X2"], "w13*(X1**3) + w23*X2 + eps_Y"),   # output
        ],
    params = {
        "w12": 7,
        "w13": 1,
        "w23": 3,
        },
    singles = {
        "X1": lambda rng,n: 0.1*rng.standard_normal(n),
        "X2": lambda rng,n: 0.1*rng.standard_normal(n),
        "Y": lambda rng,n: 0.1*rng.standard_normal(n),
        },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels = ["Y"]
    )

# TODO
# - one-to-one-CT
# - one-to-many-noCT
# - one-to-many-noCT


#ds_scm_1_to_1_ct_2.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/example_2"))

#ds_scm1.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm1"), normalize_method="minmax")
#ds_scm2.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm2"), normalize_method="minmax")
ds_scm3.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm3"), normalize_method="minmax")
