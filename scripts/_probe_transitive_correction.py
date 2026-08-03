"""Offline probe: does a geometric TRANSITIVE correction clean the indirect edges?

Question
========
In the HOMOGENEOUS run ``baseline_equal_ds_centroid_init_*`` the child X5
(true parents X1, X2, X3) still keeps NON-ZERO mass on its GRANDparents
S1, S2, S3 (paths S1 -> X1 -> X5).  Nothing in the loss removes them: the HSIC
term is already satisfied once the true parents are used, and lambda_l0=1e-6 is
numerically negligible.

Proposed fix (Francesco): remove from the query the key components that are
explained by a MEDIATOR.  With an ORTHONORMAL key frame the cosines
``c_ij = <u_i, k_j>`` ARE the coordinates of the query in the key basis, so
"remove the k_j component" removes exactly the logit of edge j -> i:

    raw formula   q_i <- q_i - k_j * (q_i . k_k) * (q_k . k_j)      (sum over k)

This probe applies the same geometry on a TRAINED checkpoint, but with the
weights built from the model's own DETACHED posterior ``Pi`` and with several
choices for the three design decisions.  It trains nothing.

Design axes probed
------------------
1. TRIGGER  ``t_ij`` = "how strongly is the edge j -> i explained by a mediator"
   * ``prod``   m_ij = max_k Pi_ik * Pi_kj          (product t-norm, as proposed)
   * ``min``    m_ij = max_k min(Pi_ik, Pi_kj)      (Goedel t-norm / fuzzy AND)
   * ``*_marg`` additionally gate by the MARGIN ``relu(m_ij - Pi_ij)``, which is
     exactly 0 while the direct edge is better supported than the path (this is
     what protects the centroid initialisation).

2. INSTRUMENT (how the trigger acts on the score)
   * ``q-proj``  u_i <- u_i - W_ij c_ij khat_j        -> c' = c (1 - W)   [target 0]
   * ``q-push``  u_i <- u_i - W_ij (c_ij + delta) khat_j -> c' -> -delta  [target < 0]
   * ``s-logit`` S_sym <- S_sym - eta * W             [direct logit bias]

   ``q-proj`` is the literal reading of the proposal.  ``q-push`` exists because
   of the finding below: with ``init_edge_offset`` dropped (homogeneous mode,
   T = 0) and ``gamma = -zeta`` the stretch constant is 0, so a score of ZERO
   gives ``P(z>0) = 0.5`` — a projection can only drive an edge to the 0.5
   THRESHOLD, never to 0.  Suppression REQUIRES a negative cosine.

3. STRENGTH ``alpha`` / ``eta`` and the number of fixed-point iterations.

All weights are pair-symmetrised (``W_ij <- max(W_ij, W_ji)``) so the GSA
existence part ``S_sym`` is attacked while the direction part ``A_anti`` stays
neutral (editing one row only would push orientation mass toward ``X -> S``).

Read-outs
---------
SHD is ALREADY 0 on this checkpoint, so the figure of merit is the MARGIN:
``gap = min(Pi over true edges) - max(Pi over false edges)``, plus the total
false mass and the true mass (which must not fall).

Usage
-----
    python scripts/_probe_transitive_correction.py [--exp-dir DIR]
                                                   [--phase-kind any|structure]
                                                   [--iters 1] [--delta 0.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from os.path import basename, exists, join
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as Fnn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DEFAULT_EXP = join(
    ROOT, "experiments", "6_INVESTIGATIONS", "HOMOGENEOUS", "results",
    "baseline_equal_ds_centroid_init_9494866",
)


# =============================================================================
# Trigger: how strongly is edge j -> i explained by a mediator k?
# =============================================================================

def mediation_mass(pi: torch.Tensor, tnorm: str = "min") -> torch.Tensor:
    """``m_ij = max_k T(Pi_ik, Pi_kj)`` with T a t-norm (fuzzy AND).

    ``pi[i, j] = P(j -> i)``, so ``j -> k -> i`` is supported by
    ``T(pi[i, k], pi[k, j])``.  ``prod`` is the literal product (it DEFLATES:
    0.65 * 0.80 = 0.52); ``min`` is the Goedel t-norm, which is the natural
    choice for soft posteriors because a chain of confident-but-soft edges keeps
    the confidence of its weakest link (min(0.65, 0.80) = 0.65).
    """
    n = pi.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=pi.device)
    p = pi.masked_fill(eye, 0.0)
    a = p.unsqueeze(-1).expand(n, n, n)          # a[i, k, j] = pi[i, k]
    b = p.unsqueeze(0).expand(n, n, n)           # b[i, k, j] = pi[k, j]
    if tnorm == "prod":
        joint = a * b
    elif tnorm == "min":
        joint = torch.minimum(a, b)
    else:
        raise ValueError(f"unknown tnorm={tnorm!r}")
    return joint.max(dim=1).values.masked_fill(eye, 0.0)


def transitive_weights(
    pi: torch.Tensor,
    alpha: float = 1.0,
    tnorm: str = "min",
    margin: bool = True,
    symmetric: bool = True,
) -> torch.Tensor:
    """Detached shrink weights ``W_ij`` in [0, alpha]."""
    pi = pi.detach()
    m = mediation_mass(pi, tnorm=tnorm)
    w = (m - pi).clamp_min(0.0) if margin else m
    w = float(alpha) * w
    if symmetric:
        w = torch.maximum(w, w.transpose(0, 1))
    return w.detach()


# =============================================================================
# Gate re-evaluation (eval-time GatedSelfAttention maths, no noise)
# =============================================================================

def gates_from_parts(s_sym, a_anti, beta, gamma, zeta, dir_beta) -> dict:
    l0_offset = beta * math.log(-gamma / zeta)
    p_exist = torch.sigmoid(s_sym - l0_offset)
    direction = torch.sigmoid(a_anti / dir_beta)
    pi = p_exist * direction
    eye = torch.eye(pi.shape[0], dtype=torch.bool, device=pi.device)
    return {"p_exist": p_exist.masked_fill(eye, 0.0), "direction": direction,
            "pi": pi.masked_fill(eye, 0.0), "S_sym": s_sym, "A_anti": a_anti}


def gates_from_raw(raw, beta, gamma, zeta, dir_beta) -> dict:
    return gates_from_parts(0.5 * (raw + raw.transpose(0, 1)),
                            0.5 * (raw - raw.transpose(0, 1)),
                            beta, gamma, zeta, dir_beta)


# =============================================================================
# Metrics
# =============================================================================

def edge_metrics(pi: np.ndarray, true: np.ndarray, thr: float = 0.5) -> dict:
    off = ~np.eye(len(pi), dtype=bool)
    tr, fa = (true == 1), (true == 0) & off
    pred = pi > thr
    return {
        "TP": int((pred & tr).sum()), "FP": int((pred & ~tr & off).sum()),
        "FN": int((~pred & tr).sum()),
        "soft_hamming": float(np.abs(pi - true)[off].sum()),
        "true_mass": float(pi[tr].sum()),
        "false_mass": float(pi[fa].sum()),
        "min_true": float(pi[tr].min()), "max_false": float(pi[fa].max()),
        "gap": float(pi[tr].min() - pi[fa].max()),
    }


def row_str(a: np.ndarray, i: int, labels) -> str:
    return "  ".join(f"{labels[j]}={a[i, j]:.2f}"
                     for j in range(a.shape[1]) if j != i)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=DEFAULT_EXP)
    ap.add_argument("--phase-kind", default="any", choices=("any", "structure",
                                                            "reconstruct"))
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.5,
                    help="q-push target cosine (negative side)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    exp_dir = args.exp_dir
    cfg_path = join(exp_dir, "config_atsel.yaml")
    if not exists(cfg_path):
        print(f"!! config not found: {cfg_path}")
        return 1
    dataset = OmegaConf.load(cfg_path)["data"]["dataset"]
    print(f"Experiment : {basename(exp_dir)}\nDataset    : {dataset}")

    # ---- ground truth ------------------------------------------------------
    from causaliT.evaluation.eval_funs.helpers.eval_utils import (
        load_dataset_metadata, _load_true_dag_mask, _compute_standard_shd,
    )
    meta = load_dataset_metadata(join(ROOT, "data"), dataset)
    L_S = int(meta["variable_info"]["n_source"])
    L_X = int(meta["variable_info"]["n_input"])
    N = L_S + L_X
    labels = [f"S{i+1}" for i in range(L_S)] + [f"X{i+1}" for i in range(L_X)]
    true_full = np.zeros((N, N), dtype=float)
    true_full[L_S:, :L_S] = _load_true_dag_mask(join(ROOT, "data"), dataset, "dec_cross")
    true_full[L_S:, L_S:] = _load_true_dag_mask(join(ROOT, "data"), dataset, "dec_self")
    print(f"N = {N} | {int(true_full.sum())} true edges | nodes {labels}")

    # ---- checkpoint --------------------------------------------------------
    pat = ("phase_*_end.ckpt" if args.phase_kind == "any"
           else f"phase_*_{args.phase_kind}_end.ckpt")
    ckpts = sorted(glob.glob(join(exp_dir, "stage_checkpoints", pat)),
                   key=lambda p: int(re.search(r"phase_(\d+)", basename(p)).group(1)))
    if not ckpts:
        print("!! no phase checkpoints found")
        return 1
    ckpt = ckpts[-1]
    print(f"Checkpoint : {basename(ckpt)}  ({len(ckpts)} matching phases)")

    from causaliT.training.forecasters.attention_selector_forecaster import (
        AttentionSelectorForecaster,
    )
    from causaliT.core.modules.gated_self_attention import GatedSelfAttention

    fc = AttentionSelectorForecaster.load_from_checkpoint(
        ckpt, map_location="cpu", strict=False)
    fc.to("cpu").eval()
    block: Optional[GatedSelfAttention] = next(
        (m for m in fc.model.modules() if isinstance(m, GatedSelfAttention)), None)
    if block is None:
        print("!! no GatedSelfAttention block found")
        return 1

    # ---- capture the structural query / key the block actually sees ---------
    cap = {}

    def _pre_hook(mod, pargs, kwargs):
        cap["query"] = (kwargs.get("query", pargs[0] if pargs else None)).detach()
        cap["key"] = (kwargs.get("key", pargs[1] if len(pargs) > 1 else None)).detach()

    h = block.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    data = np.load(join(ROOT, "data", dataset, "ds_test.npz"))
    with torch.no_grad():
        _, att, _ = fc.forward(
            torch.tensor(data["s"][: args.batch].astype(np.float32)),
            torch.tensor(data["x"][: args.batch].astype(np.float32)))
    h.remove()
    phi_model = att.mean(dim=0).numpy()

    q, k = cap["query"][0], cap["key"][0]                       # (N, E)
    khat = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    off_gram = float((khat @ khat.transpose(0, 1) - torch.eye(N)).abs().max())
    print(f"query {tuple(q.shape)} | key Gram max|off-diag| = {off_gram:.4f} "
          f"(orthonormal frame required for exact component removal)")

    beta, gamma, zeta, dir_beta = block.beta, block.gamma, block.zeta, block.dir_beta
    F_scale = block.query_fanin_scale
    M = (torch.exp(block.query_norm_log_scale.detach())
         if block.query_norm_log_scale is not None else torch.ones(N))
    l0_offset = beta * math.log(-gamma / zeta)
    print(f"gate: beta={beta} gamma={gamma} zeta={zeta} dir_beta={dir_beta} | "
          f"stretch c={l0_offset:.4f} | F={F_scale:.3f} sqrt(F)={math.sqrt(F_scale):.3f} "
          f"| M in [{float(M.min()):.3f}, {float(M.max()):.3f}]")

    def raw_from_u(u: torch.Tensor) -> torch.Tensor:
        uu = Fnn.normalize(u, p=2.0, dim=-1, eps=1e-8)
        return (uu * M.view(-1, 1)) @ k.transpose(0, 1) * math.sqrt(F_scale)

    u0 = Fnn.normalize(q, p=2.0, dim=-1, eps=1e-8)
    g0 = gates_from_raw(raw_from_u(u0), beta, gamma, zeta, dir_beta)
    pi0 = g0["pi"]
    err = float(np.abs(pi0.numpy() - phi_model).max())
    print(f"re-derived posterior == model forward: max|diff| = {err:.2e}"
          + ("" if err < 1e-4 else "   !! MISMATCH, results not conclusive"))

    c0 = u0 @ khat.transpose(0, 1)
    print(f"budget sum_j c_ij^2: min={float(c0.pow(2).sum(1).min()):.4f} "
          f"max={float(c0.pow(2).sum(1).max()):.4f}  (cap = 1)")

    # ---- baseline read-out -------------------------------------------------
    base = edge_metrics(pi0.numpy(), true_full)
    shd0 = _compute_standard_shd(pi0.numpy(), true_full, threshold=0.5,
                                 is_cross_attention=False)["shd"]
    print("\n=== BASELINE (trained state) " + "=" * 44)
    print(f"TP={base['TP']} FP={base['FP']} FN={base['FN']} SHD={shd0} | "
          f"softH={base['soft_hamming']:.3f} trueM={base['true_mass']:.3f} "
          f"falseM={base['false_mass']:.3f} | min_true={base['min_true']:.3f} "
          f"max_false={base['max_false']:.3f} GAP={base['gap']:+.3f}")

    x5 = N - 1
    print(f"\nX5 decomposition (row {labels[x5]}): Pi = p_exist * direction")
    for name, mat in (("cosine c", c0), ("S_sym", g0["S_sym"]),
                      ("p_exist", g0["p_exist"]), ("direction", g0["direction"]),
                      ("Pi", pi0)):
        print(f"  {name:<10s} " + row_str(mat.numpy(), x5, labels))
    print("  (a projection to c=0 gives S_sym=0 -> p_exist=sigmoid(-c)="
          f"{float(torch.sigmoid(torch.tensor(-l0_offset))):.2f}: the 0.5 THRESHOLD, "
          "not 0 -> suppression needs c<0)")

    for tnorm in ("prod", "min"):
        m = mediation_mass(pi0, tnorm=tnorm)
        print(f"\nmediation mass ({tnorm}) row X5: " + row_str(m.numpy(), x5, labels))
        w = transitive_weights(pi0, alpha=1.0, tnorm=tnorm, margin=True)
        print(f"  margin weights row X5      : " + row_str(w.numpy(), x5, labels))

    # ---- variants ----------------------------------------------------------
    print("\n=== CORRECTED " + "=" * 58)
    hdr = (f"{'instrument':<9s} {'trigger':<9s} {'str':>5s} {'it':>3s} | "
           f"{'TP':>2s} {'FP':>2s} {'FN':>2s} {'SHD':>3s} {'softH':>6s} "
           f"{'trueM':>6s} {'falsM':>6s} {'minT':>5s} {'maxF':>5s} {'GAP':>6s} | "
           f"X5: S1,S2,S3 | X1,X2,X3")
    print(hdr)
    print("-" * len(hdr))

    triggers = [("prod+marg", "prod", True), ("min+marg", "min", True),
                ("min", "min", False)]
    rows = []

    def report(inst, trig, strength, pi):
        p = pi.numpy()
        mt = edge_metrics(p, true_full)
        shd = _compute_standard_shd(p, true_full, threshold=0.5,
                                    is_cross_attention=False)["shd"]
        gp = ",".join(f"{p[x5, j]:.2f}" for j in range(3))
        pa = ",".join(f"{p[x5, L_S + j]:.2f}" for j in range(3))
        print(f"{inst:<9s} {trig:<9s} {strength:>5.2f} {args.iters:>3d} | "
              f"{mt['TP']:>2d} {mt['FP']:>2d} {mt['FN']:>2d} {shd:>3d} "
              f"{mt['soft_hamming']:>6.2f} {mt['true_mass']:>6.3f} "
              f"{mt['false_mass']:>6.3f} {mt['min_true']:>5.2f} "
              f"{mt['max_false']:>5.2f} {mt['gap']:>+6.3f} | {gp} | {pa}")
        rows.append({"instrument": inst, "trigger": trig, "strength": strength,
                     "iters": args.iters, "shd": int(shd), **mt})

    for tname, tnorm, margin in triggers:
        for inst in ("q-proj", "q-push", "s-logit"):
            strengths = (0.5, 1.0) if inst.startswith("q") else (2.0, 4.0, 8.0)
            for strength in strengths:
                u, s_sym, a_anti = u0.clone(), g0["S_sym"].clone(), g0["A_anti"]
                pi = pi0.clone()
                for _ in range(max(1, args.iters)):
                    if inst == "s-logit":
                        w = transitive_weights(pi, alpha=1.0, tnorm=tnorm,
                                               margin=margin)
                        s_sym = g0["S_sym"] - float(strength) * w
                        gg = gates_from_parts(s_sym, a_anti, beta, gamma, zeta,
                                              dir_beta)
                    else:
                        w = transitive_weights(pi, alpha=float(strength),
                                               tnorm=tnorm, margin=margin)
                        c = Fnn.normalize(u, dim=-1) @ khat.transpose(0, 1)
                        tgt = c if inst == "q-proj" else (c + float(args.delta))
                        u = Fnn.normalize(u, dim=-1) - (w * tgt) @ khat
                        gg = gates_from_raw(raw_from_u(u), beta, gamma, zeta,
                                            dir_beta)
                    pi = gg["pi"]
                report(inst, tname, strength, pi)
        print("-" * len(hdr))

    print(f"BASELINE   {'-':<9s} {'-':>5s} {'-':>3s} | "
          f"{base['TP']:>2d} {base['FP']:>2d} {base['FN']:>2d} {shd0:>3d} "
          f"{base['soft_hamming']:>6.2f} {base['true_mass']:>6.3f} "
          f"{base['false_mass']:>6.3f} {base['min_true']:>5.2f} "
          f"{base['max_false']:>5.2f} {base['gap']:>+6.3f} | "
          + ",".join(f"{pi0[x5, j]:.2f}" for j in range(3)) + " | "
          + ",".join(f"{pi0[x5, L_S + j]:.2f}" for j in range(3)))

    # ---- centroid-init safety ---------------------------------------------
    print("\n=== INIT SAFETY (all-on centroid query) " + "=" * 32)
    q_c = khat.sum(dim=0, keepdim=True).expand(N, -1).clone()
    g_c = gates_from_raw(raw_from_u(q_c), beta, gamma, zeta, dir_beta)
    print(f"  Pi_init ~ {float(g_c['pi'][0, 1]):.3f}")
    for tnorm in ("prod", "min"):
        for margin in (True, False):
            w_c = transitive_weights(g_c["pi"], alpha=1.0, tnorm=tnorm,
                                     margin=margin)
            print(f"  tnorm={tnorm:<4s} margin={str(margin):<5s} -> max W = "
                  f"{float(w_c.max()):.4f}  m_init = "
                  f"{float(mediation_mass(g_c['pi'], tnorm=tnorm)[0, 1]):.3f}")

    out = join(exp_dir, "transitive_correction_probe.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": basename(ckpt),
                   "baseline": {**base, "shd": int(shd0)},
                   "delta": args.delta, "variants": rows}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
