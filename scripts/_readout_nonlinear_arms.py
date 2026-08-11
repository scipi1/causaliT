"""Readout for the NONLINEARITIES investigation arms (baseline / larger_mlp / ...).

Run AFTER training.  For every arm folder under
experiments/6_INVESTIGATIONS/NONLINEARITIES/ that has a trained checkpoint, this
prints the four diagnostics that settle the capacity-vs-gate question:

  1. GATE      X2->X4 / X2->X5 / X1->X5 self-attention mass (does the edge open?)
  2. MECHANISM X4 response to an X2 sweep (flat vs U-shape = f*X2^2 learned?)
  3. ATE       roll-out indirect ATE on S3->X2->X4 / X5 (recovered vs still 0)
  4. SHD       structural metrics from dag_metrics.json (stays clean?)

The post-training evaluations (eval_attention_scores, eval_interventions) run
automatically (adaptive config run_final_evaluations: true), so this reads those
artifacts and adds the gate + mechanism probes on the checkpoint.

Usage:  python scripts/_readout_nonlinear_arms.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.evaluation.eval_funs.eval_interventions import get_scm_for_dataset
from causaliT.evaluation.eval_funs.helpers.eval_lib import (
    find_config_file, find_best_or_last_checkpoint)
from causaliT.evaluation.predict import create_predictor
from causaliT.training.experiment_control import update_config
from omegaconf import OmegaConf

INVEST = project_root / "experiments" / "6_INVESTIGATIONS" / "NONLINEARITIES" / "results"
DS = "scm2_continuous"          # local dataset folder name
SCM = "ds_scm2_continuous"      # registry name


def find_arms():
    arms = []
    for d in sorted(INVEST.iterdir()):
        if d.is_dir() and (d / "k_0" / "checkpoints").is_dir():
            arms.append(d)
    return arms


def load_model(arm):
    cfg = update_config(OmegaConf.load(find_config_file(str(arm))))
    ck = find_best_or_last_checkpoint(str(arm / "k_0" / "checkpoints"), "last")
    pred = create_predictor(cfg, ck, str(project_root / "data"))
    pred.model.eval()
    return pred.model


def obs_tensors(model, scm, n=2048):
    """Fresh SCM-sampled (S, X) tensors normalized to the model's space."""
    ddir = project_root / "data" / DS
    meta = json.load(open(ddir / "dataset_metadata.json"))
    svm = json.load(open(ddir / "source_vars_map.json"))
    ivm = json.load(open(ddir / "input_vars_map.json"))
    norm = json.load(open(ddir / "normalization.json"))
    src, inp = meta["variable_info"]["source_labels"], meta["variable_info"]["input_labels"]
    df = scm.sample(n=n, seed=0)
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
    return S.to(dev), X.to(dev)


def gate_masses(model, S, X):
    with torch.no_grad():
        att_sx, att_xx = model.get_split_attention(S, X)
    xx = att_xx.mean(0).cpu()   # (L_X, L_X): row=child, col=parent
    return {"X2->X4": float(xx[3, 1]), "X2->X5": float(xx[4, 1]), "X1->X5": float(xx[4, 0])}


def x4_sweep_span(model, S, X):
    """X4 response span when X2 is swept over its observed normalized range [0,1]."""
    outs = []
    for v in np.linspace(0.0, 1.0, 9):
        Xv = X.clone()
        Xv[:, 1, 0] = float(v)   # X2 value column
        with torch.no_grad():
            pred = model.forward(S, Xv)[0]
        if getattr(model, "homogeneous_nodes", False):
            pred = pred[:, S.shape[1]:, :]
        outs.append(float(pred[:, 3, 0].mean().cpu()))
    outs = np.array(outs)
    return float(outs.max() - outs.min()), np.round(outs, 3).tolist()


def read_eval(arm):
    """Pull indirect-ATE and SHD from the post-training eval artifacts."""
    out = {}
    ate = arm / "eval" / "eval_ate_mc" / "files" / "ate_metrics_mc.csv"
    if ate.is_file():
        d = pd.read_csv(ate)
        # indirect S3 effect on X4/X5 (the multi-hop paths)
        sel = d[(d.intervention.str.startswith("S3")) & (d.variable.isin(["X4", "X5"]))]
        out["indirect_abs_err"] = float(sel["abs_error"].mean()) if len(sel) else None
    dag = arm / "eval" / "eval_attention_scores" / "files" / "dag_metrics.json"
    if dag.is_file():
        j = json.load(open(dag))
        out["shd_self"] = (j.get("standard_shd_self") or {}).get("mean")
        out["shd_cross"] = (j.get("standard_shd_cross") or {}).get("mean")
    return out


def main():
    arms = find_arms()
    if not arms:
        print(f"No trained arms found under {INVEST} (need k_0/checkpoints).")
        return
    scm = get_scm_for_dataset(SCM, datadir_path=str(project_root / "data"))

    print("=" * 80)
    print("NONLINEARITIES readout  (dataset: scm2_continuous)")
    print("=" * 80)
    print(f"{'arm':16s} {'X2->X4':>7} {'X2->X5':>7} {'X1->X5':>7}  "
          f"{'X4span':>7}  {'ind_ATE_err':>11} {'shd_self':>8} {'shd_cross':>9}")
    for arm in arms:
        model = load_model(arm)
        S, X = obs_tensors(model, scm)
        g = gate_masses(model, S, X)
        span, _ = x4_sweep_span(model, S, X)
        ev = read_eval(arm)
        ind = ev.get("indirect_abs_err")
        print(f"{arm.name:16s} {g['X2->X4']:7.3f} {g['X2->X5']:7.3f} {g['X1->X5']:7.3f}  "
              f"{span:7.3f}  "
              f"{(f'{ind:11.3f}' if ind is not None else '     -     ')} "
              f"{(f'{ev.get('shd_self'):8.2f}' if ev.get('shd_self') is not None else '   -    ')} "
              f"{(f'{ev.get('shd_cross'):9.2f}' if ev.get('shd_cross') is not None else '    -     ')}")

    print()
    print("Reading: gate opens = X2->X4 mass up;  X4span>0 = U-shape learned;")
    print("ind_ATE_err down = indirect effect recovered;  shd low = DAG still clean.")


if __name__ == "__main__":
    main()
