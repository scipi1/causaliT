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



ds_scm3 = SCMDataset(
    name = "simple non-linear Gaussian",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                               # parent
        NodeSpec("X2", ["X1"], "w12*(X1**2) + eps_X2"),             # child 1
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


ds_scm3 = SCMDataset(
    name = "simple non-linear Gaussian",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("X1", [], "eps_X1"),                                   # parent
        NodeSpec("X2", ["X1"], "w12*(X1**2) + eps_X2"),                 # child 1
        NodeSpec("X2", ["X1"], "w12*(X1**2) + eps_X2"),                 # child 1
        NodeSpec("Y", ["X1", "X2"], "w13*(X1**3) + w23*X2 + eps_Y"),    # output
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
        # NodeSpec("X3", ["X1", "X2"], "b*X1 + c*X2 + eps_X3"), # input 3
        NodeSpec("Y1", ["X1"], "g*X1 + eps_Y1"),                # target 1
        NodeSpec("Y2", ["X2"], "h*X2 + eps_Y2"),                # target 2
        # NodeSpec("Y3", ["X3", "Y2"], "j*X3 + k*Y2 + eps_Y3"),   # target 3
        ],
    params = {
        "a": 1,
        "b": 1,
        "c": 1,
        "f": 1,
        "g": 1,
        "h": 1,
        # "j": 1,
        # "k": 1,
        },
    singles = {
        "S1": lambda rng,n: rng.uniform(-1, 1, n),
        "S2": lambda rng,n: rng.uniform(-1, 1, n),
        "S3": lambda rng,n: rng.uniform(-1, 1, n),
        "X1": lambda rng,n: 0.05*rng.standard_normal(n),
        "X2": lambda rng,n: 0.05*rng.standard_normal(n),
        # "X3": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y1": lambda rng,n: 0.05*rng.standard_normal(n),
        "Y2": lambda rng,n: 0.05*rng.standard_normal(n),
        # "Y3": lambda rng,n: 0.05*rng.standard_normal(n),
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

# TODO
# - one-to-one-CT
# - one-to-many-noCT
# - one-to-many-noCT


#ds_scm_1_to_1_ct_2.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/example_2"))

#ds_scm1.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm1"), normalize_method="minmax")
#ds_scm2.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm2"), normalize_method="minmax")
#ds_scm3.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm3"), normalize_method="minmax")
#ds_scm4.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm4"), normalize_method="minmax")
# ds_scm5.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm5"), normalize_method="minmax")

# Test with shared_embedding=True for unified variable IDs across all categories
#ds_scm6.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm6"), normalize_method="minmax", shared_embedding=True)
ds_scm7.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/scm7"), normalize_method="minmax", shared_embedding=True)
