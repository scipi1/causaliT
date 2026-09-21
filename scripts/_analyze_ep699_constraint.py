"""Epoch-699 analysis of the interrupted run
``adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd_14476278``.

The run never left the warmup phase: rung 0 of the count-based BKD ladder
(deterministic, exactly min_keys=1 key kept per batch) for all 799 logged
epochs.  This script measures, at the epoch-699 checkpoint:

PART 1 -- max(cos(-g, parents)):
    g = gradient of the LIVE constraint term (softmax-attention-weighted
    HSIC, ``model._last_hsic_reg`` after a training-path ``_step`` with the
    rung-0 BKD state) w.r.t. the free per-node query rows.  For each node we
    report the max cosine of the update direction -g with each TRUE PARENT's
    frozen key-frame vector, with the max over non-parents and the parent
    centroid as controls.

PART 2 -- key-conditioned fit:
    Rung 0 regresses every node on ONE key per batch.  We therefore force the
    BKD keep mask to a one-hot for each candidate key j in turn (eval mode,
    deterministic gates, FiLM context = the resulting one-hot applied row --
    exactly the trained regime, conditioned on j) and compute R2(j -> i) for
    every target i on the full dataset.  Targets are labelled by their causal
    relation to j (child / descendant / parent / ancestor / self / other) so
    we can check whether the nodes that fit under key j are precisely the
    ones j is causally upstream of.

PART 3 -- learned selection at epoch 699:
    dense-eval gate posterior vs ground truth (argmax key, mass on true
    edges), plus a metrics.csv trajectory summary.

Outputs are written to <run>/analysis_ep699/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_signal_diagnostics import (
    _cosine,
    free_query_weights,
    key_frame,
    load_dataset_batches,
    load_ground_truth,
    normalized_centroid,
)
import causaliT.core.modules.gated_cross_attention as gca_mod
import causaliT.core.modules.gated_self_attention as gsa_mod
import causaliT.core.modules.commutator_self_attention as csa_mod

# In homogeneous mode the single attention block is a GatedSelfAttention, so
# the BKD sampler it calls lives in the gated_self_attention namespace; patch
# ALL namespaces that imported sample_bkd_keep_mask to be safe.
BKD_NAMESPACES = (gca_mod, gsa_mod, csa_mod)

RUN = Path(
    "experiments/6_INVESTIGATIONS/HSIC_CONSTRAINT/results/"
    "adaptive_nhsic_ladder_gt_ctx_mkladder_bigmlp_film2_evalbkd_14476278"
)
CKPT = RUN / "k_0/checkpoints/epoch=699-train_loss=0.00.ckpt"
DATASET = Path("data/random_n20_k4_er_nonlinear_gaussian_s1")
OUT = RUN / "analysis_ep699"

N_GRAD_BATCHES = 8
GRAD_BATCH_SIZE = 512
SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_rung0_bkd(model, eval_mode: bool) -> None:
    """Reproduce the rung-0 ladder state the PhaseController had installed.

    p=1.0 with deterministic min_keys=1 -> exactly one key kept per forward,
    in both the training path and the seeded eval path.
    """
    for mod in model.modules():
        if hasattr(mod, "set_bkd_phase_active"):
            mod.set_bkd_schedule(1.0, 1.0, None)
            mod.set_bkd_sampling(min_keys=1, deterministic=True)
            mod.set_bkd_phase_active(True)
            if hasattr(mod, "set_bkd_eval"):
                mod.set_bkd_eval(eval_mode, seed=12345)


def forced_keep_patch(key_index: int):
    """Return a sample_bkd_keep_mask replacement keeping ONLY ``key_index``."""

    def _forced(num_keys, p, min_keys=0, deterministic=False, device=None,
                generator=None):
        keep = torch.zeros(num_keys, dtype=torch.bool, device=device)
        assert 0 <= key_index < num_keys, (key_index, num_keys)
        keep[key_index] = True
        return keep

    return _forced


def reachability(gt: np.ndarray) -> np.ndarray:
    """Bool reachability of the parent->child DAG (gt is [child, parent])."""
    a = gt.T.astype(bool)  # a[p, c] = p is parent of c
    reach = a.copy()
    for _ in range(a.shape[0]):
        new = reach | (reach @ reach)
        if new.sum() == reach.sum():
            break
        reach = new
    return reach  # reach[p, c] = p is an ancestor of c


def relation_label(i: int, j: int, gt: np.ndarray, reach: np.ndarray) -> str:
    if i == j:
        return "self"
    if gt[i, j]:
        return "child"
    if reach[j, i]:
        return "descendant"
    if gt[j, i]:
        return "parent"
    if reach[i, j]:
        return "ancestor"
    return "other"


def r2_per_node(target: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """target/pred: (N_samples, N_nodes) on CPU -> per-node R2."""
    y = target.double()
    p = pred.double()
    ss_res = (y - p).pow(2).sum(dim=0)
    ss_tot = (y - y.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    return (1.0 - ss_res / ss_tot.clamp_min(1e-12)).numpy()


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionSelectorForecaster.load_from_checkpoint(CKPT, map_location=device)
    model.eval()

    gt = load_ground_truth(DATASET)          # (20, 20) [child, parent]
    reach = reachability(gt)
    n_nodes = gt.shape[0]
    k = key_frame(model).cpu()               # (20, d) frozen key frame
    summary = {"checkpoint": str(CKPT), "device": str(device)}

    # ------------------------------------------------------------------ #
    # PART 1: max(cos(-g, parents)) of the live constraint gradient
    # ------------------------------------------------------------------ #
    set_rung0_bkd(model, eval_mode=False)    # train-path rung-0 BKD
    batches = load_dataset_batches(
        DATASET, GRAD_BATCH_SIZE, N_GRAD_BATCHES, SEED, device
    )
    weights = free_query_weights(model)
    all_grads = []
    hsic_vals = []
    model.train()
    for b, batch in enumerate(batches):
        torch.manual_seed(SEED + b)
        model._step((batch.source, batch.intermediate), stage="train")
        obj = model._last_hsic_reg
        assert obj is not None and obj.requires_grad, "no live HSIC constraint term"
        hsic_vals.append(float(obj.detach()))
        grads = torch.autograd.grad(obj, weights, allow_unused=False)
        all_grads.append([g.detach().cpu() for g in grads])
    model.eval()

    # (B, N, d) node gradients in global node order (padding row dropped)
    g = torch.stack([torch.cat([w[1:] for w in grads]) for grads in all_grads])

    records = []
    for node in range(n_nodes):
        parents = np.flatnonzero(gt[node])
        node_g = g[:, node]                                   # (B, d)
        update = -node_g
        mean_update = -node_g.mean(dim=0)
        nonparents = [j for j in range(n_nodes) if j != node and not gt[node, j]]
        if len(parents) == 0:
            records.append({"node": node, "n_parents": 0,
                            "grad_norm_mean": float(node_g.norm(dim=1).mean())})
            continue
        parent_cos_mean_g = {int(p): _cosine(mean_update, k[p]) for p in parents}
        nonpar_cos_mean_g = {int(j): _cosine(mean_update, k[j]) for j in nonparents}
        # Split non-parent directions by causal relation to the node:
        # descendants (i upstream of j), non-parent ancestors (j upstream of
        # i), and the rest ("other": spouses, unrelated, ...).
        desc_j = [j for j in nonparents if reach[node, j]]
        anc_j = [j for j in nonparents if reach[j, node]]

        def _cat_stats(idxs):
            if not idxs:
                return (math.nan, math.nan)
            vals = [nonpar_cos_mean_g[int(j)] for j in idxs]
            return (max(vals), float(np.mean(vals)))

        desc_max, desc_mean = _cat_stats(desc_j)
        anc_max, anc_mean = _cat_stats(anc_j)
        oth_j = [
            j for j in nonparents
            if not reach[node, j] and not reach[j, node]
        ]
        oth_max, oth_mean = _cat_stats(oth_j)

        parent_c = normalized_centroid(k, parents)
        second = node_g.pow(2).sum(dim=1).mean()
        signal = node_g.mean(dim=0).pow(2).sum()
        noise = (second - signal).clamp_min(0.0) / node_g.shape[1]
        records.append({
            "node": node,
            "n_parents": int(len(parents)),
            "parents": ",".join(str(int(p)) for p in parents),
            # THE metric: max over true parents of cos(-g_bar, k_p)
            "maxcos_parent_mean_g": max(parent_cos_mean_g.values()),
            "argmax_parent": int(max(parent_cos_mean_g, key=parent_cos_mean_g.get)),
            "meancos_parent_mean_g": float(np.mean(list(parent_cos_mean_g.values()))),
            # per-batch mean of per-batch max cosine (noisier view)
            "maxcos_parent_perbatch": float(np.mean([
                max(_cosine(u, k[p]) for p in parents) for u in update
            ])),
            # controls
            "maxcos_nonparent_mean_g": max(nonpar_cos_mean_g.values()),
            # non-parent directions split by causal relation
            "maxcos_descendant_mean_g": desc_max,
            "meancos_descendant_mean_g": desc_mean,
            "maxcos_ancestor_mean_g": anc_max,
            "meancos_ancestor_mean_g": anc_mean,
            "maxcos_other_mean_g": oth_max,
            "meancos_other_mean_g": oth_mean,

            "cos_centroid_mean_g": _cosine(mean_update, parent_c),
            "grad_norm_mean": float(node_g.norm(dim=1).mean()),
            "gradient_snr": float(signal / max(float(noise), 1e-12)),
        })
    grad_df = pd.DataFrame(records)
    grad_df.to_csv(OUT / "grad_cosine_per_node.csv", index=False)
    wp = grad_df[grad_df.n_parents > 0]
    summary["grad_cosine"] = {
        "constraint": "attw_softmax HSIC (_last_hsic_reg), train path, rung-0 BKD",
        "hsic_reg_per_batch": hsic_vals,
        "n_batches": N_GRAD_BATCHES,
        "batch_size": GRAD_BATCH_SIZE,
        "mean_maxcos_parent_mean_g": float(wp.maxcos_parent_mean_g.mean()),
        "median_maxcos_parent_mean_g": float(wp.maxcos_parent_mean_g.median()),
        "mean_maxcos_nonparent_mean_g": float(wp.maxcos_nonparent_mean_g.mean()),
        # Non-parent categories (NaN-safe: a node may lack a category)
        "mean_maxcos_descendant_mean_g": float(wp.maxcos_descendant_mean_g.mean()),
        "mean_meancos_descendant_mean_g": float(wp.meancos_descendant_mean_g.mean()),
        "mean_maxcos_ancestor_mean_g": float(wp.maxcos_ancestor_mean_g.mean()),
        "mean_meancos_ancestor_mean_g": float(wp.meancos_ancestor_mean_g.mean()),
        "mean_maxcos_other_mean_g": float(wp.maxcos_other_mean_g.mean()),
        "mean_meancos_other_mean_g": float(wp.meancos_other_mean_g.mean()),
        "frac_nodes_parent_beats_descendant": float(
            (wp.maxcos_parent_mean_g > wp.maxcos_descendant_mean_g)[
                wp.maxcos_descendant_mean_g.notna()
            ].mean()
        ),
        "frac_nodes_parent_beats_ancestor": float(
            (wp.maxcos_parent_mean_g > wp.maxcos_ancestor_mean_g)[
                wp.maxcos_ancestor_mean_g.notna()
            ].mean()
        ),
        "frac_nodes_parent_beats_other": float(
            (wp.maxcos_parent_mean_g > wp.maxcos_other_mean_g)[
                wp.maxcos_other_mean_g.notna()
            ].mean()
        ),

        "mean_cos_centroid_mean_g": float(wp.cos_centroid_mean_g.mean()),
        "frac_nodes_parent_beats_best_nonparent": float(
            (wp.maxcos_parent_mean_g > wp.maxcos_nonparent_mean_g).mean()
        ),
        "mean_gradient_snr": float(wp.gradient_snr.mean()),
    }


    # ------------------------------------------------------------------ #
    # PART 2: key-conditioned fit  R2(j -> i)
    # ------------------------------------------------------------------ #
    data = np.load(DATASET / "ds.npz")
    s_all = torch.tensor(np.asarray(data["s"]), dtype=torch.float32, device=device)
    x_all = torch.tensor(np.asarray(data["x"]), dtype=torch.float32, device=device)
    val_idx = model.val_idx
    target_all = torch.cat([s_all[:, :, val_idx], x_all[:, :, val_idx]], dim=1)

    set_rung0_bkd(model, eval_mode=True)     # seeded eval-BKD branch active
    model.eval()
    orig_samplers = {ns: ns.sample_bkd_keep_mask for ns in BKD_NAMESPACES}

    def patch_all(fn):
        for ns in BKD_NAMESPACES:
            ns.sample_bkd_keep_mask = fn

    def restore_all():
        for ns, fn in orig_samplers.items():
            ns.sample_bkd_keep_mask = fn

    inner = model.model.attention.inner_attention
    r2_mat = np.zeros((n_nodes, n_nodes))    # [key j, target i]
    chunk = 1000
    with torch.no_grad():
        for j in range(n_nodes):
            patch_all(forced_keep_patch(j))
            preds = []
            for lo in range(0, len(s_all), chunk):
                pred = model(
                    data_source=s_all[lo:lo + chunk],
                    data_intermediate=x_all[lo:lo + chunk],
                )[0]
                preds.append(pred.squeeze(-1).cpu())
            # Verify the forced key was actually applied by the live module.
            kept = inner.last_bkd_keep
            assert kept is not None and kept.nonzero().flatten().tolist() == [j], (
                f"forced key {j} not applied: {kept}"
            )
            r2_mat[j] = r2_per_node(target_all.cpu(), torch.cat(preds))
    restore_all()

    r2_df = pd.DataFrame(
        r2_mat,
        index=[f"key_{j}" for j in range(n_nodes)],
        columns=[f"target_{i}" for i in range(n_nodes)],
    )
    r2_df.to_csv(OUT / "r2_by_key_matrix.csv")

    long_rows = []
    for j in range(n_nodes):
        for i in range(n_nodes):
            long_rows.append({
                "key": j, "target": i, "r2": float(r2_mat[j, i]),
                "relation": relation_label(i, j, gt, reach),
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT / "r2_vs_causal_relation.csv", index=False)

    rel_mean = long_df.groupby("relation").r2.agg(["mean", "median", "max", "count"])
    summary["r2_by_relation"] = rel_mean.to_dict()

    # Per-key view: is the best-fitting non-self target causally downstream?
    key_rows = []
    for j in range(n_nodes):
        sub = long_df[(long_df.key == j) & (long_df.target != j)]
        best = sub.loc[sub.r2.idxmax()]
        downstream = sub[sub.relation.isin(["child", "descendant"])]
        other = sub[sub.relation == "other"]
        key_rows.append({
            "key": j,
            "best_target": int(best.target), "best_r2": float(best.r2),
            "best_relation": best.relation,
            "n_children": int(gt[:, j].sum()),
            "n_downstream": int(reach[j].sum()),
            "mean_r2_downstream": float(downstream.r2.mean()) if len(downstream) else math.nan,
            "max_r2_downstream": float(downstream.r2.max()) if len(downstream) else math.nan,
            "mean_r2_other": float(other.r2.mean()),
            "frac_downstream_above_0.3": float((downstream.r2 > 0.3).mean()) if len(downstream) else math.nan,
            "frac_other_above_0.3": float((other.r2 > 0.3).mean()),
        })
    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(OUT / "r2_per_key_summary.csv", index=False)
    summary["r2_per_key"] = key_rows

    # Per-target view: R2 under true-parent keys vs non-parent keys
    tgt_rows = []
    for i in range(n_nodes):
        parents = np.flatnonzero(gt[i])
        if len(parents) == 0:
            continue
        nonpar = [j for j in range(n_nodes) if j != i and not gt[i, j]]
        tgt_rows.append({
            "target": i,
            "n_parents": int(len(parents)),
            "max_r2_parent_key": float(r2_mat[parents, i].max()),
            "mean_r2_parent_key": float(r2_mat[parents, i].mean()),
            "max_r2_nonparent_key": float(r2_mat[nonpar, i].max()),
            "mean_r2_nonparent_key": float(r2_mat[nonpar, i].mean()),
        })
    tgt_df = pd.DataFrame(tgt_rows)
    tgt_df.to_csv(OUT / "r2_per_target_parent_vs_not.csv", index=False)
    summary["r2_per_target"] = {
        "mean_max_r2_parent_key": float(tgt_df.max_r2_parent_key.mean()),
        "mean_max_r2_nonparent_key": float(tgt_df.max_r2_nonparent_key.mean()),
        "frac_targets_parent_key_best": float(
            (tgt_df.max_r2_parent_key > tgt_df.max_r2_nonparent_key).mean()
        ),
    }


    # ------------------------------------------------------------------ #
    # PART 3: learned selection (dense posterior) + metrics.csv summary
    # ------------------------------------------------------------------ #
    for mod in model.modules():
        if hasattr(mod, "set_bkd_eval"):
            mod.set_bkd_eval(False)
    model.eval()
    with torch.no_grad():
        _, att_w, _ = model(
            data_source=s_all[:1024], data_intermediate=x_all[:1024]
        )
    att = att_w.mean(dim=0).cpu().numpy()    # (N, N) gate posterior

    # Query-collapse diagnostics: a flat posterior is expected if all free
    # query rows collapsed onto a single direction.
    qw = free_query_weights(model)
    qrows = torch.cat([t[1:] for t in qw]).detach().cpu()   # (N, d)
    qn = torch.nn.functional.normalize(qrows, dim=-1)
    pair_cos = qn @ qn.T
    offdiag = pair_cos[~torch.eye(n_nodes, dtype=bool)]
    q_to_key = qn @ torch.nn.functional.normalize(k, dim=-1).T  # (N, N)
    summary["query_geometry"] = {
        "mean_offdiag_query_cos": float(offdiag.mean()),
        "min_offdiag_query_cos": float(offdiag.min()),
        "max_offdiag_query_cos": float(offdiag.max()),
        "maxcos_to_any_key_per_node": [
            round(float(v), 4) for v in q_to_key.max(dim=1).values
        ],
    }
    sel_rows = []
    for i in range(n_nodes):
        parents = np.flatnonzero(gt[i])
        row = att[i].copy()
        row[i] = -np.inf                     # self-edge excluded
        am = int(np.argmax(row))
        sel_rows.append({
            "node": i,
            "n_parents": int(len(parents)),
            "argmax_key": am,
            "argmax_is_parent": bool(gt[i, am]),
            "mass_on_parents": float(att[i, parents].sum()) if len(parents) else math.nan,
            "max_posterior": float(att[i, am]),
        })
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(OUT / "selection_vs_gt.csv", index=False)
    wp_sel = sel_df[sel_df.n_parents > 0]
    summary["selection"] = {
        "frac_argmax_is_parent": float(wp_sel.argmax_is_parent.mean()),
        "mean_mass_on_parents": float(wp_sel.mass_on_parents.mean()),
    }

    m = pd.read_csv(RUN / "k_0/logs/csv/version_0/metrics.csv")
    val = m.dropna(subset=["val_x_r2"])
    summary["metrics_csv"] = {
        "epochs_logged": int(m.epoch.max()),
        "phases_seen": sorted(m.adaptive_phase.dropna().unique().tolist()),
        "bkd_min_keys_seen": sorted(m.bkd_min_keys.dropna().unique().tolist()),
        "val_x_r2_mean": float(val.val_x_r2.mean()),
        "val_x_r2_max": float(val.val_x_r2.max()),
        "val_x_r2_frac_above_0.3": float((val.val_x_r2 > 0.3).mean()),
        "val_x_r2_frac_above_0.5": float((val.val_x_r2 > 0.5).mean()),
        "corr_val_r2_val_hsic": float(val.val_x_r2.corr(val.val_hsic)),
        "dual_lambda_final": float(m["hsic/dual_lambda"].dropna().iloc[-1]),
        "constraint_ema_final": float(m["hsic/constraint_ema"].dropna().iloc[-1]),
    }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Console report
    # ------------------------------------------------------------------ #
    pd.set_option("display.width", 220)
    print("=" * 78)
    print("PART 1 - max(cos(-g, parents)) of the attw_softmax HSIC constraint")
    print(grad_df.to_string(index=False))
    print(json.dumps(summary["grad_cosine"], indent=2))
    print("=" * 78)
    print("PART 2 - key-conditioned R2 (rung-0 regime: one key at a time)")
    print("R2 by causal relation to the selected key:")
    print(rel_mean.to_string())
    print("\nPer-key summary (best target, downstream vs other):")
    print(key_df.to_string(index=False))
    print("\nPer-target: R2 under parent keys vs non-parent keys:")
    print(tgt_df.to_string(index=False))
    print(json.dumps(summary["r2_per_target"], indent=2))
    print("=" * 78)
    print("PART 3 - learned selection at epoch 699 (dense posterior)")
    print(sel_df.to_string(index=False))
    print(json.dumps(summary["selection"], indent=2))
    print("Query geometry (collapse check):")
    print(json.dumps(summary["query_geometry"], indent=2))
    print(json.dumps(summary["metrics_csv"], indent=2))
    print(f"\nWrote analysis to {OUT}")


if __name__ == "__main__":
    main()

