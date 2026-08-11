"""Per-node test MAE on scm2: is svfa's 2x test MAE global or localized?

The user notes svfa's test MAE is ~2x vanilla/cheater on the nonlinear sets and
reads this as "fitting the data is already hard."  If the excess error is
concentrated on X4/X5 (the nodes whose parents are ALL zero-first-order), then
the fit difficulty is the closed-gate symptom, not a global capacity limit.
Sampling from the SCM (npz pruned for vanilla/svfa), normalizing, one forward.

Usage:  python scripts/_diag_ate_pernode_mae.py
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


def main():
    print(f"Per-node test MAE on {DS} (normalized units), fresh SCM sample\n")
    rows = {}
    for arm, arm_dir in ARMS.items():
        run, datadir = first_run(arm_dir)
        model = load(run, datadir)
        scm = get_scm_for_dataset(DS, datadir_path=str(datadir))
        meta = json.load(open(Path(datadir) / DS / "dataset_metadata.json"))
        svm = json.load(open(Path(datadir) / DS / "source_vars_map.json"))
        ivm = json.load(open(Path(datadir) / DS / "input_vars_map.json"))
        norm = json.load(open(Path(datadir) / DS / "normalization.json"))
        src = meta["variable_info"]["source_labels"]
        inp = meta["variable_info"]["input_labels"]
        df = scm.sample(n=4000, seed=7)
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

        S, X = b(s, svm, src), b(x, ivm, inp)
        dev = next(model.parameters()).device
        S, X = S.to(dev), X.to(dev)
        with torch.no_grad():
            pred = model.forward(S, X)[0]
        if getattr(model, "homogeneous_nodes", False):
            pred = pred[:, S.shape[1]:, :]
        mae_per_node = (pred.squeeze(-1) - X[:, :, 0]).abs().mean(0).cpu().numpy()
        rows[arm] = mae_per_node

    header = "  node  " + "".join(f"{a:>10}" for a in ARMS)
    print(header)
    for i, node in enumerate(inp):
        line = f"  {node:4s}  " + "".join(f"{rows[a][i]:>10.4f}" for a in ARMS)
        print(line)
    print("\nX4/X5 are the nodes whose parents are ALL zero-first-order.")
    print("If svfa's excess MAE is concentrated there, it is the closed-gate symptom.")


if __name__ == "__main__":
    main()
