"""
Test 2 - key ablation on the HSIC loss: deterministic, per target.

After the warmup, each target t is ablated in six deterministic ways:

    one_parent / all_parents         true parents of t
    one_descendant / all_descendants true descendants of t
    one_spurious / all_spurious      every other key (ancestors + unrelated)

`one_*` ablations drop a single key and produce one row per dropped key;
`all_*` ablations drop the whole class in a single pass.  No BKD is used at
evaluation time: the only mask is the ablation itself, so the attention is
always the clean adjacency.

For every ablation we record the nodal contribution L_t of the target,
decomposed by SOURCE CLASS (per-edge means over parents / descendants /
spurious of att * pair_raw, and of the unweighted pair_raw), clean and
ablated, plus the MSE change.  HSIC is O(batch^2) per pair, so every
quantity is averaged over n_sub fixed sub-batches.

Run:  python test_2_loss_ablation.py --config loss_ablation
Output: results/test_2_<config>.csv  (one row per ablation x target)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIGS  # noqa: E402
from definitions import (GatedReadoutModel, load_scm, RESULTS_DIR,  # noqa: E402
                         att_weighted_hsic_parts)
from scm_ds.random_scm import RandomSCMConfig  # noqa: E402
from utils import warmup_train, append_csv, resolve_config, relations  # noqa: E402

CLASSES = ("parents", "descendants", "spurious")


def _parts_mean(model, x, sub_idx, keep, mode):
    """Loss decomposition averaged over the fixed sub-batches."""
    acc = None
    with torch.no_grad():
        for idx in sub_idx:
            p = att_weighted_hsic_parts(model, x[idx], keep=keep, mode=mode)
            cur = {k: v.detach().clone() for k, v in p.items()}
            acc = cur if acc is None else {k: acc[k] + cur[k] for k in acc}
    return {k: v / len(sub_idx) for k, v in acc.items()}


def _class_means(parts, t, classes):
    """Per-edge mean of pair_w and pair_raw of target t, per source class."""
    out = {}
    for name, idx in classes.items():
        if idx:
            out[name] = (float(parts["pair_w"][t, idx].mean()),
                         float(parts["pair_raw"][t, idx].mean()))
        else:
            out[name] = (float("nan"), float("nan"))
    return out


def _mse_t(model, x, sub_idx, keep, t):
    """MSE of target t averaged over the same sub-batches."""
    acc = 0.0
    with torch.no_grad():
        for idx in sub_idx:
            xb = x[idx]
            acc += float(((model(xb, keep=keep)[:, t] - xb[:, t]) ** 2).mean())
    return acc / len(sub_idx)


def run_cell(cfg: dict, n_nodes: int, degree: float, seed: int,
             gate_mode: str, warmup_bkd: float) -> list:
    """One warmup branch: six deterministic ablations per target."""
    scm_cfg = RandomSCMConfig(n_nodes=n_nodes, degree=degree, seed=seed,
                              linearity=cfg["linearity"], noise=cfg["noise"])
    x_train, x_test, parents_idx, _ = load_scm(
        scm_cfg, cfg["n_train"], cfg["n_test"])

    model = GatedReadoutModel(n_nodes, emb_dim=cfg["emb_dim"],
                              gate_mode=gate_mode, seed=seed)
    warmup_train(model, x_train, steps=cfg["warmup_steps"],
                 lr=cfg["warmup_lr"], seed=seed * 100 + 1, bkd_p=warmup_bkd)

    gen = torch.Generator().manual_seed(seed * 1000 + 7)
    sub_idx = [torch.randperm(len(x_test), generator=gen)[:cfg["sub_size"]]
               for _ in range(cfg["n_sub"])]
    mode = cfg["hsic_mode"]
    rel = relations(parents_idx, n_nodes)

    base = dict(gate_mode=gate_mode, warmup_bkd=warmup_bkd,
                n_nodes=n_nodes, degree=degree, seed=seed)

    clean = _parts_mean(model, x_test, sub_idx, None, mode)
    rows = []
    for t in range(n_nodes):
        P = [s for s in range(n_nodes)
             if s != t and rel[(s, t)] == "parent"]
        D = [s for s in range(n_nodes)
             if s != t and rel[(s, t)] == "descendant"]
        S = [s for s in range(n_nodes)
             if s != t and s not in P and s not in D]
        classes = {"parents": P, "descendants": D, "spurious": S}
        clean_cm = _class_means(clean, t, classes)
        nodal_clean = float(clean["nodal"][t])
        mse_clean = _mse_t(model, x_test, sub_idx, None, t)

        ablations = []
        for cls, members in (("parent", P), ("descendant", D),
                             ("spurious", S)):
            for s in members:
                ablations.append((f"one_{cls}", [s], s))
            if members:
                ablations.append((f"all_{cls}s", list(members), -1))

        for abl_name, dropped, key in ablations:
            keep = torch.ones(n_nodes)
            for s in dropped:
                keep[s] = 0.0
            abl = _parts_mean(model, x_test, sub_idx, keep, mode)
            abl_cm = _class_means(abl, t, classes)
            row = dict(
                **base, ablation=abl_name, dropped_key=key, target=t,
                n_dropped=len(dropped),
                n_parents=len(P), n_descendants=len(D), n_spurious=len(S),
                nodal_t_clean=nodal_clean,
                nodal_t_ablated=float(abl["nodal"][t]),
                mse_clean=mse_clean,
                mse_ablated=_mse_t(model, x_test, sub_idx, keep, t),
            )
            for cls in CLASSES:
                row[f"w_{cls}_clean"] = clean_cm[cls][0]
                row[f"w_{cls}_ablated"] = abl_cm[cls][0]
                row[f"raw_{cls}_clean"] = clean_cm[cls][1]
                row[f"raw_{cls}_ablated"] = abl_cm[cls][1]
            rows.append(row)
    return rows