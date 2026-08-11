"""Decisive checks for the svfa-vs-vanilla nonlinear ATE gap.

(A) Shortcut test: read the cross-attention mass from the S-ancestor onto the
    DESCENDANT (S3->X4, S3->X5).  If vanilla carries real mass there while svfa
    does not, then vanilla answers indirect queries via a one-step S->descendant
    shortcut (no mechanism composition), whereas svfa is forced to compose
    X2->X4 through the roll-out.  That - not an eval bug - is the gap.

(B) Variant A/B: run the roll-out on one svfa scm2 run with noise="none" and
    noise="residual" for the S3 interventions, and compare the indirect ATE.
    If B is not worse than A, the residual bootstrap is exonerated and the gap
    is fit/structure, not the noise model.

Usage:  python scripts/_diag_ate_shortcut.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from causaliT.evaluation.eval_funs.eval_interventions import (
    run_mc_predictions, get_scm_for_dataset, load_ate_ground_truth,
    load_normalization_stats)
from causaliT.evaluation.eval_funs.helpers.eval_lib import (
    find_config_file, find_best_or_last_checkpoint)
from causaliT.evaluation.predict import create_predictor
from causaliT.training.experiment_control import update_config
from omegaconf import OmegaConf

RESULTS = project_root / "experiments" / "7_PUBLISH" / "ATE" / "results"
ARMS = {"vanilla": "vanilla_9648449", "svfa": "svfa_9425659", "cheater": "cheater_9427393"}
DS = "ds_scm2_continuous"
N = 4000


def first_run(arm_dir, ds):
    d = RESULTS / arm_dir / "groups" / ds
    runs = [r for r in sorted(d.glob("sweeper/runs/combinations/*"))
            if (r / "k_0/checkpoints").is_dir()]
    return runs[0], d / "datasets"


def load_model(run, datadir):
    cfg = update_config(OmegaConf.load(find_config_file(str(run))))
    ck = find_best_or_last_checkpoint(str(run / "k_0/checkpoints"), "last")
    pred = create_predictor(cfg, ck, str(datadir))
    pred.model.eval()
    return pred.model


def shortcut_mass(model, scm_dataset, datadir, ds, n=2048):
    """Mean cross-attention mass S3->X4 / S3->X5 on SCM-sampled observational data.

    ds.npz is pruned for vanilla/svfa, so sample fresh (S, X) from the SCM and
    normalize to the model's space.
    """
    meta = json.load(open(Path(datadir) / ds / "dataset_metadata.json"))
    svm = json.load(open(Path(datadir) / ds / "source_vars_map.json"))
    ivm = json.load(open(Path(datadir) / ds / "input_vars_map.json"))
    norm = json.load(open(Path(datadir) / ds / "normalization.json"))
    src_lab = meta["variable_info"]["source_labels"]
    in_lab = meta["variable_info"]["input_labels"]

    df = scm_dataset.sample(n=n, seed=0)
    s_np = df[src_lab].to_numpy(np.float32)
    x_np = df[in_lab].to_numpy(np.float32)

    def normz(arr, key):
        st = norm[key]
        return (arr - st["min"]) / (st["max"] - st["min"])

    s_np, x_np = normz(s_np, "source"), normz(x_np, "input")

    def build(arr, mp, labels):
        t = np.zeros((len(arr), len(labels), 2), np.float32)
        t[:, :, 0] = arr
        for i, lab in enumerate(labels):
            t[:, i, 1] = float(mp[lab])
        return torch.from_numpy(t)

    S = build(s_np, svm, src_lab)
    X = build(x_np, ivm, in_lab)
    dev = next(model.parameters()).device
    S, X = S.to(dev), X.to(dev)
    with torch.no_grad():
        att_sx, att_xx = model.get_split_attention(S, X)
    a = att_sx.mean(0)   # (L_X, L_S)
    xx = att_xx.mean(0)  # (L_X, L_X): row=child, col=parent
    # X4 is row 3 (parents X1,X2 = cols 0,1); X5 is row 4. X2 is col 1.
    return {
        "S3->X4": float(a[3, 2]), "S3->X5": float(a[4, 2]),
        "X2->X4": float(xx[3, 1]), "X2->X5": float(xx[4, 1]),
        "X1->X5": float(xx[4, 0]),
    }


def main():
    print("=" * 70)
    print("(A) S-ancestor shortcut mass (mean attention weight, scm2)")
    print("    high on vanilla + ~0 on svfa => vanilla shortcuts the chain")
    print("=" * 70)
    print(f"  {'arm':8s}  {'S3->X4':>7} {'S3->X5':>7}   {'X2->X4':>7} {'X2->X5':>7} {'X1->X5':>7}")
    for arm, arm_dir in ARMS.items():
        run, datadir = first_run(arm_dir, DS)
        model = load_model(run, datadir)
        scm_dataset = get_scm_for_dataset(DS, datadir_path=str(datadir))
        m = shortcut_mass(model, scm_dataset, datadir, DS)
        print(f"  {arm:8s}  {m['S3->X4']:7.3f} {m['S3->X5']:7.3f}   "
              f"{m['X2->X4']:7.3f} {m['X2->X5']:7.3f} {m['X1->X5']:7.3f}")
    print("  GT edges: S3->X2; X2->X4, X2->X5, X1->X5  (S3->X4/X5 are NOT edges)")

    print()
    print("=" * 70)
    print("(B) Variant A/B on one svfa scm2 run (S3 interventions)")
    print("=" * 70)
    arm_dir = ARMS["svfa"]
    run, datadir = first_run(arm_dir, DS)
    datadir = str(datadir)
    meta = json.load(open(Path(datadir) / DS / "dataset_metadata.json"))
    svm = json.load(open(Path(datadir) / DS / "source_vars_map.json"))
    ivm = json.load(open(Path(datadir) / DS / "input_vars_map.json"))
    src_lab = meta["variable_info"]["source_labels"]
    in_lab = meta["variable_info"]["input_labels"]
    scm = get_scm_for_dataset(DS, datadir_path=datadir)
    gt = load_ate_ground_truth(datadir, DS)["monte_carlo"]["ate"]
    norm = load_normalization_stats(datadir, DS)
    rng = norm["input"]["max"] - norm["input"]["min"]

    for noise in ("none", "residual"):
        df = run_mc_predictions(
            experiment_path=str(run), scm_dataset=scm,
            intervention_config={"S3": [-0.5, 1.5]}, norm_stats=norm,
            source_vars_map=svm, input_vars_map=ivm,
            source_labels=src_lab, input_labels=in_lab,
            n_samples=N, seed=42, checkpoint_type="last",
            propagation="rollout", noise=noise, datadir_path=datadir)

        def mn(label, var):
            m = df[(df.intervention == label) & (df.variable == var)]
            return float(m["pred_feat_0"].iloc[0])

        print(f"  --- noise={noise} ---")
        for label in ("S3=-0.5", "S3=1.5"):
            for var in ("X4", "X5"):
                model_ate = (mn(label, var) - mn("S3=0_baseline", var)) * rng
                print(f"    {label} {var}: model={model_ate:+.3f}  true={gt[label][var]:+.3f}")


if __name__ == "__main__":
    main()
