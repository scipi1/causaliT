# Temporary check: group-resolved per-pair HSIC H[i,j] over training.
# DELETE after use.
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_utils import hsic_pair_matrix

ROOT = Path(".")
DATA_DIR = ROOT / ("experiments/6_INVESTIGATIONS/LARGER_DAGS/baseline/"
                   "random_n20_k4_er_nonlinear_gaussian_s1")
N_S = 2
NODES = ["S1", "S2"] + [f"X{i}" for i in range(1, 19)]
N = len(NODES)

cross_gt = pd.read_csv(DATA_DIR / "dec1_cross_att_mask.csv", index_col=0)
self_gt = pd.read_csv(DATA_DIR / "dec1_self_att_mask.csv", index_col=0)
GT = np.zeros((N, N))
GT[N_S:, :N_S] = cross_gt.values
GT[N_S:, N_S:] = self_gt.values
EYE = np.eye(N, dtype=bool)
PA = GT.astype(bool)
CH = GT.T.astype(bool)


def closure(adj):
    r = adj.copy()
    for _ in range(int(np.ceil(np.log2(adj.shape[0]))) + 1):
        r = r | (r @ r)
    return r


ANC = closure(PA) & ~PA
DESC = closure(CH) & ~CH
SPOUSE = (CH.astype(float) @ CH.T.astype(float)) > 0

GROUPS = ["parents", "children", "ancestors", "descendants", "spouses", "other"]
MASK_G = {}
assigned = EYE.copy()
for g, mm in [("parents", PA), ("children", CH), ("ancestors", ANC), ("descendants", DESC)]:
    MASK_G[g] = mm & ~assigned
    assigned |= MASK_G[g]
MASK_G["spouses"] = SPOUSE & ~assigned
assigned |= MASK_G["spouses"]
MASK_G["other"] = ~assigned

ds = np.load(DATA_DIR / "ds.npz")
S_ALL = torch.tensor(ds["s"], dtype=torch.float32)
X_ALL = torch.tensor(ds["x"], dtype=torch.float32)
perm = torch.randperm(S_ALL.shape[0], generator=torch.Generator().manual_seed(0))
batches = [(S_ALL[b], X_ALL[b]) for b in perm.split(256)[:4]]


def H_at(ckpt, train_mode=False):
    model = AttentionSelectorForecaster.load_from_checkpoint(ckpt, map_location="cpu")
    model.train() if train_mode else model.eval()
    Hs = []
    with torch.no_grad():
        for s_b, x_b in batches:
            pred = model.forward(data_source=s_b, data_intermediate=x_b)[0]
            x_target = torch.cat([s_b, x_b], dim=1)[:, :, 0]
            residuals = x_target - pred.squeeze(-1)
            Hs.append(hsic_pair_matrix(x_target, residuals, sigma=1.0,
                                       adaptive_bandwidth=True, mode="biased",
                                       source_kernel="rbf").double().numpy())
    del model
    return np.stack(Hs).mean(axis=0)


def group_means(H):
    out = {}
    for g in GROUPS:
        m = MASK_G[g]
        out[g] = np.nanmean(np.where(m, H, np.nan)[N_S:])
    out["self"] = np.nanmean(np.where(EYE, H, np.nan)[N_S:])
    return out


for arm, ckdir, tm in [
    ("no_NT", r"experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT/k_0/checkpoints", False),
    ("no_NT_bkd_02 (train-mode)", r"experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02/k_0/checkpoints", True),
]:
    cks = {}
    for p in Path(ckdir).glob("*.ckpt"):
        m = re.match(r"epoch=(\d+)-", p.name)
        if m:
            cks[int(m.group(1))] = p
    print(f"\n==== {arm} ====")
    print(f"{'epoch':>6} | " + " | ".join(f"{g:>11}" for g in GROUPS) + " |        self")
    for e in [49, 199, 499, 999]:
        ee = min(cks, key=lambda x: abs(x - e))
        gm = group_means(H_at(cks[ee], train_mode=tm))
        print(f"{ee:>6} | " + " | ".join(f"{gm[g]:11.5f}" for g in GROUPS) + f" | {gm['self']:11.5f}")
