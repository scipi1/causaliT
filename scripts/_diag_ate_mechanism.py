"""Probe WHY svfa drops the indirect effect on the nonlinear scm2.

The shortcut test already showed vanilla reads S3->X4 directly (mass 0.25) while
svfa prunes it (~0.03) and must compose S3->X2->X4.  This probe checks the last
link: does svfa's learned X4 mechanism actually RESPOND to X2?

Two reads on one run per arm (scm2):
  1. Mechanism shape: sweep X2 over a grid (other parents fixed), read X4.
     The true dependence is f*X2^2 (non-saturating quadratic in the mediator,
     E[X2]~0) - if svfa's X4 is ~flat in X2, the quadratic was never learned
     and the composition collapses to ~0.
  2. Roll-out trace under do(S3=1.5): print converged X2, X4, X5 vs baseline.

Usage:  python scripts/_diag_ate_mechanism.py
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


def tensors(datadir, scm, n=512):
    """Build normalized (S, X) observational tensors from a fresh SCM sample."""
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
    return b(s, svm, src), b(x, ivm, inp), (norm["input"]["max"] - norm["input"]["min"])


def main():
    print("=" * 74)
    print("X4 response to a mediator sweep (flat => quadratic never learned)")
    print("True X4 = d*S4^2 + e*S5*X2 + f*X2^2  (f*X2^2 is the indirect channel)")
    print("=" * 74)
    grid = np.linspace(-1.0, 1.0, 9)   # normalized X2 sweep
    for arm, arm_dir in ARMS.items():
        run, datadir = first_run(arm_dir)
        model = load(run, datadir)
        scm = get_scm_for_dataset(DS, datadir_path=str(datadir))
        S, X, rng = tensors(datadir, scm)
        dev = next(model.parameters()).device
        S, X = S.to(dev), X.to(dev)
        dev_x = next(model.parameters()).device
        outs = []
        for v in grid:
            Xv = X.clone()
            Xv[:, 1, 0] = float(v)         # X2 value column := v
            with torch.no_grad():
                pred = model.forward(S, Xv)[0]
            if getattr(model, "homogeneous_nodes", False):
                pred = pred[:, S.shape[1]:, :]
            outs.append(float(pred[:, 3, 0].mean().cpu()))   # X4 prediction
        outs = np.array(outs)
        span = outs.max() - outs.min()
        print(f"  {arm:8s} X4 over X2 in [-1,1]: span={span:.4f}  "
              f"curve={np.round(outs, 3).tolist()}")

    print()
    print("=" * 74)
    print("Roll-out trace under do(S3=1.5): converged X2 / X4 / X5 (normalized)")
    print("=" * 74)
    for arm, arm_dir in ARMS.items():
        run, datadir = first_run(arm_dir)
        model = load(run, datadir)
        scm = get_scm_for_dataset(DS, datadir_path=str(datadir))
        S, X, rng = tensors(datadir, scm)
        dev = next(model.parameters()).device
        S = S.to(dev)
        # Intervene S3 := 1.5 raw -> normalize with source stats
        norm = json.load(open(Path(datadir) / DS / "normalization.json"))
        s15 = (1.5 - norm["source"]["min"]) / (norm["source"]["max"] - norm["source"]["min"])
        s0 = (0.0 - norm["source"]["min"]) / (norm["source"]["max"] - norm["source"]["min"])
        results = {}
        for tag, sval in [("base(S3=0)", s0), ("do(S3=1.5)", s15)]:
            Si = S.clone()
            Si[:, 2, 0] = float(sval)          # S3 column
            X0 = torch.zeros_like(X).to(dev)
            for i in range(X.shape[1]):
                X0[:, i, 1] = X[0, i, 1]
            with torch.no_grad():
                xf, it, dlt = model.causal_predict(Si, X0)
            v = xf[0, :, 0].cpu().numpy()
            results[tag] = (v, it, dlt)
        vb, _, _ = results["base(S3=0)"]
        vt, it, dlt = results["do(S3=1.5)"]
        dX2, dX4, dX5 = (vt[1] - vb[1]) * rng, (vt[3] - vb[3]) * rng, (vt[4] - vb[4]) * rng
        print(f"  {arm:8s} iters={it} delta={dlt:.1e} | "
              f"dX2={dX2:+.3f} dX4={dX4:+.3f} dX5={dX5:+.3f}  (true dX4~+0.60 dX5~+0.80)")


if __name__ == "__main__":
    main()
