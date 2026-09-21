"""Calibrate the nHSIC constraint tolerance (epsilon) via a permutation null.

Regime: residuals of the ORACLE run's final checkpoint (fitted regressor at
the true DAG => residuals ~ noise, the ideal H0 regime).  Per pair (i, j),
the null is built by permuting the residual along the BATCH dimension
(destroys the pairing, preserves marginals); epsilon per pair = 99th
percentile of the permuted nHSIC statistics.

Outputs a JSON next to the HSIC_CONSTRAINT experiment dir and prints the
summary, including the sanity comparison vs the observed nHSIC at the true
DAG (should sit around the null quantile).

Usage:  python scripts/calibrate_hsic_tolerance.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic

DATA = ROOT / "data" / "random_n20_k4_er_nonlinear_gaussian_s1"
ORACLE = (ROOT / "experiments" / "6_INVESTIGATIONS" / "HSIC_CONSTRAINT"
          / "results" / "oracle_shd_dense_14351891" / "k_0" / "checkpoints"
          / "last.ckpt")
OUT = ROOT / "experiments" / "6_INVESTIGATIONS" / "HSIC_CONSTRAINT" / "hsic_tolerance_calibration.json"

N_S, N_X = 2, 18
N = N_S + N_X
N_SAMPLES = 512
N_PERM = 100
ALPHA = 0.99
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---- GT (for the sanity contrast, not needed for the null) -------------------
cross = pd.read_csv(DATA / "dec1_cross_att_mask.csv", index_col=0).values
selfm = pd.read_csv(DATA / "dec1_self_att_mask.csv", index_col=0).values
GT = np.zeros((N, N))
GT[N_S:, :N_S] = cross
GT[N_S:, N_S:] = selfm

# ---- Residuals at the oracle's final (true-DAG, fitted) checkpoint -----------
fm = AttentionSelectorForecaster.load_from_checkpoint(ORACLE, map_location=device)
fm.eval()

d = np.load(DATA / "ds.npz")
rng = np.random.default_rng(SEED)
idx = np.sort(rng.permutation(len(d["x"]))[:N_SAMPLES])
S = torch.tensor(np.asarray(d["s"][idx]), dtype=torch.float32, device=device)
X = torch.tensor(np.asarray(d["x"][idx]), dtype=torch.float32, device=device)

with torch.no_grad():
    pred, _, _ = fm.forward(data_source=S, data_intermediate=X)
x_val = X[:, :, fm.val_idx]
if fm.homogeneous_nodes:
    x_val = torch.cat([S[:, :, fm.val_idx], x_val], dim=1)
x_target = torch.nan_to_num(x_val)
residuals = (x_target.squeeze() - pred.squeeze()).detach()   # (n, N)
combined = x_target.squeeze().detach()                        # (n, N) sources
n = residuals.shape[0]
print(f"residuals {tuple(residuals.shape)}, sources {tuple(combined.shape)}")

# ---- Observed nHSIC and permutation null per pair ----------------------------
gen = torch.Generator(device=device).manual_seed(SEED)
obs = np.zeros((N, N))
q99 = np.zeros((N, N))
for i in range(N):            # residual (child) index
    ri = residuals[:, i]
    for j in range(N):        # source index
        if i == j:
            obs[i, j] = np.nan
            q99[i, j] = np.nan
            continue
        kwargs = dict(mode="normalized", adaptive_bandwidth=True,
                      nhsic_epsilon=fm.nhsic_epsilon)
        obs[i, j] = float(hsic(combined[:, j], ri, **kwargs))
        stats = []
        for _ in range(N_PERM):
            perm = torch.randperm(n, generator=gen, device=device)
            stats.append(float(hsic(combined[:, j], ri[perm], **kwargs)))
        q99[i, j] = float(np.quantile(stats, ALPHA))
    if (i + 1) % 5 == 0:
        print(f"  row {i + 1}/{N} done")

valid = ~np.isnan(obs)
eps_mean = float(np.nanmean(q99))
eps_max = float(np.nanmax(q99))
obs_mean = float(np.nanmean(obs[valid]))

# Sanity: observed-at-GT vs null on NON-STRUCTURAL pairs (non-parents,
# non-descendants): at the true DAG these should sit AT the null.
CH = GT.T.copy()
desc = CH.copy()
for _ in range(N):
    desc = np.clip(desc + desc @ CH, 0, 1)
DESC = desc.astype(bool)
mask_np = np.zeros((N, N), bool)
for i in range(N):
    ok = (~GT[i].astype(bool)) & (~DESC[i])
    ok[i] = False
    mask_np[i] = ok

out = {
    "checkpoint": str(ORACLE.relative_to(ROOT)),
    "n_samples": N_SAMPLES,
    "n_permutations": N_PERM,
    "alpha": ALPHA,
    "eps_mean_q99": eps_mean,
    "eps_max_q99": eps_max,
    "obs_nhsic_mean_all_pairs": obs_mean,
    "obs_nhsic_mean_nonparent_nondesc": float(np.nanmean(obs[mask_np])),
    "null_q99_mean_nonparent_nondesc": float(np.nanmean(q99[mask_np])),
    "obs_nhsic_mean_parent": float(np.nanmean(obs[GT.astype(bool)])),
    "q99_per_pair": q99.tolist(),
    "obs_per_pair": obs.tolist(),
}
OUT.write_text(json.dumps(out, indent=2))
print()
print(f"eps (mean per-pair q99): {eps_mean:.4f}")
print(f"eps (max  per-pair q99): {eps_max:.4f}")
print(f"observed nHSIC at GT, non-parent/non-desc mean: {out['obs_nhsic_mean_nonparent_nondesc']:.4f} "
      f"(their null q99 mean: {out['null_q99_mean_nonparent_nondesc']:.4f})")
print(f"observed nHSIC at GT, parent mean: {out['obs_nhsic_mean_parent']:.4f}")
print(f"written: {OUT}")
