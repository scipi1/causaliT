"""Test the user's 'shared head chose the dominant node' hypothesis on scm2.

For X4 = d*S4^2 + e*S5*X2 + f*X2^2, sweep EACH parent in turn (others held at
their observed values) and read X4.  If svfa's X4 bends to S4 (the dominant
d*S4^2 channel) but is flat in X2 (the weak f*X2^2 channel), the shared readout
allocated its capacity to the dominant parent and dropped the weak one - a
capacity/allocation bottleneck, distinct from a closed gate.

Note the tension this resolves: svfa's X2->X4 gate was ~0.276 (PARTIALLY open)
yet X4 was flat in X2.  A partially-open gate with a flat readout points at the
value/readout side, not the gate - which is the user's argument.

Usage:  python scripts/_diag_ate_dominant.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.evaluation.eval_funs.eval_interventions import get_scm_for_dataset
from causaliT.evaluation.eval_funs.helpers.eval_lib import (
    find_config_file, find_best_or_last_checkpoint)
from causaliT.evaluation.predict import create_predictor
from causaliT.training.experiment_control import update_config
from omegaconf import OmegaConf

RESULTS = project_root / "experiments" / "7_PUBLISH" / "ATE" / "results"
ARMS = {"vanilla": "vanilla_9648449", "svfa": "svfa_9425659", "cheater": "cheater_9427393"}
DS = "ds_scm2_continuous"

# Parent positions: S4 -> S-col 3, S5 -> S-col 4, X2 -> X-col 1.
PARENTS = {"S4(d*S4^2)": ("S", 3), "S5(e*S5*X2)": ("S", 4), "X2(f*X2^2)": ("X", 1)}


def first_run(arm_dir):
    d = RESULTS / arm_dir / "groups" / DS
    runs = [r for r in sorted(d.glob("sweeper/runs/combinations/*"))
            if (r / "k_0/checkpoints").is_dir()]
    return runs[0], d / "datasets"


def load(run, datadir):
    cfg = update_config(OmegaConf.load(find_config_file(str(run))))
    ck = find_best_or_last_checkpoint(str(run / "k_0/checkpoints"), "last")
    pred = create_predictor(cfg, ck, str(datadir))
    pred.model.eval()
    return pred.model


def tensors(datadir, scm, n=512):
    meta = json.load(open(Path(datadir) / DS / "dataset_metadata.json"))
    svm = json.load(open(Path(datadir) / DS / "source_vars_map.json"))
    ivm = json.load(open(Path(datadir) / DS / "input_vars_map.json"))
    norm = json.load(open(Path(datadir) / DS / "normalization.json"))
    src, inp = meta["variable_info"]["source_labels"], meta["variable_info"]["input_labels"]
    df = scm.sample(n=n, seed=1)
    s = df[src].to_numpy(np.float32)
    x = df[inp].to_numpy(np.float32)
    s = (s - norm["source"]["min"]) / (norm["source"]["max"] - norm["source"]["min"])
    x = (x - norm["input"]["min"]) / (norm["input"]["max"] - norm["input"]["min"])

    def b(a, mp, lab):
        t = np.zeros((len(a), len(lab), 2), np.float32)
        t[:, :, 0] = a
        for i, l in enumerate(lab):
            t[:, i, 1] = float(mp[l])
        return torch.from_numpy(t)
    return b(s, svm, src), b(x, ivm, inp)


def sweep(model, S, X, which, col, grid):
    dev = next(model.parameters()).device
    outs = []
    for v in grid:
        Sv, Xv = S.clone(), X.clone()
        if which == "S":
            Sv[:, col, 0] = float(v)
        else:
            Xv[:, col, 0] = float(v)
        with torch.no_grad():
            pred = model.forward(Sv, Xv)[0]
        if getattr(model, "homogeneous_nodes", False):
            pred = pred[:, S.shape[1]:, :]
        outs.append(float(pred[:, 3, 0].mean().cpu()))   # X4 row
    return np.array(outs)


def main():
    # Data is min-max normalized to [0, 1]; sweep IN-DISTRIBUTION, not [-1,1].
    grid = np.linspace(0.0, 1.0, 9)
    print("X4 response span as each parent is swept over its normalized range.")
    print("A flat row for a parent => that channel was NOT learned by the readout.\n")
    header = "  arm       " + "".join(f"{p:>16}" for p in PARENTS)
    print(header)
    for arm, arm_dir in ARMS.items():
        run, datadir = first_run(arm_dir)
        model = load(run, datadir)
        scm = get_scm_for_dataset(DS, datadir_path=str(datadir))
        S, X = tensors(datadir, scm)
        dev = next(model.parameters()).device
        S, X = S.to(dev), X.to(dev)
        spans = []
        for pname, (which, col) in PARENTS.items():
            outs = sweep(model, S, X, which, col, grid)
            spans.append(outs.max() - outs.min())
        print(f"  {arm:8s}  " + "".join(f"{s:>16.4f}" for s in spans))
    print("\nTrue X4 = d*S4^2 + e*S5*X2 + f*X2^2.  "
          "If svfa bends only for S4/S5 but is flat for X2, the shared head")
    print("learned the dominant parents and dropped the weak quadratic one.")


if __name__ == "__main__":
    main()
