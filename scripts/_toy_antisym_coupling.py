"""Toy study: does the antisymmetric direction term couple queries across rows?

Hypothesis (observed in the HSIC_OPT runs): the direction gate logit

    A_anti[i, j] = (raw[i, j] - raw[j, i]) / 2 ,  raw[i, j] = <q_i, k_j> * scale

contains ``p_ji`` with a MINUS sign.  A loss that wants to raise ``p_ij``
(e.g. HSIC(x_j, r_i) > 0 asks node i to attend to j) can therefore descend by
pushing ``q_j`` AWAY from ``k_i`` instead of pulling ``q_i`` toward ``k_j``.
That reverse path couples the learning of queries j and i and can produce
HSIC gradients that point against the true-parent centroid of node j - which
in an ANM should not happen.

Minimal, controlled replica of the gate math of
``causaliT.core.modules.gated_self_attention.GatedSelfAttention``
(deterministic eval-mode form, same as the offline probe in
``evaluate_updates.ipynb``):

    p_ij = sigmoid(S_sym - l0_offset) * sigmoid(A_anti / dir_beta)

on a small nonlinear Gaussian ANM, trained with MSE + HSIC (the repo's own
``hsic_cross_per_pair``).  Three arms share init and seed:

    A  current : full gradient through A_anti (and S_sym)
    B  fix     : A_anti = 0.5 * (raw - raw^T.detach())   (p_ji path cut)
    C  control : additionally detach raw^T inside S_sym  (symmetric coupling)

Evidence (per arm, over seeds): gradient decomposition of d HSIC / d q_j into
own-row / reverse-antisym / reverse-sym parts with per-part cosine to the
true-parent key centroid; fraction of anti-aligned steps; alignment and SHD
trajectories; final HSIC/MSE.

Outputs (figures + summary.json):
``experiments/6_INVESTIGATIONS/HSIC_OPT/toy_antisym/``.

Run from the repo root:  python scripts/_toy_antisym_coupling.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from causaliT.utils.hsic_utils import hsic_cross_per_pair

# ---- Constants (mirroring the HSIC_OPT run configs) -------------------------
DIR_BETA = 2.0 / 3.0        # dir_tau_self
L0_OFFSET = 0.0             # beta * log(-gamma / zeta) = 0 for the runs' gates
HSIC_KW = dict(sigma=1.0, adaptive_bandwidth=True, mode="biased",
               nhsic_epsilon=0.01, source_kernel="rbf", pair_mask=None)

# ---- Toy-study hyperparameters ----------------------------------------------
N = 8                       # nodes
D_MODEL = 40                # key/query width (as the runs)
N_SAMPLES = 4096            # dataset size
BATCH = 256                 # HSIC batch per step
STEPS = 400                 # optimizer steps per arm
LOG_EVERY = 20              # gradient-decomposition cadence
LR = 5e-3
LAMBDA_HSIC = 1.0
SEEDS = [0, 1, 2]
NOISE_SCALE = 1.0           # ANM noise scale (s1, as the dataset)
EDGE_PROB = 2.0 / (N - 1)   # ER, expected fan-in ~2

OUT_DIR = (Path(__file__).resolve().parent.parent
           / "experiments" / "6_INVESTIGATIONS" / "HSIC_OPT" / "toy_antisym")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
             "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})




# ---- Toy SCM -----------------------------------------------------------------
def make_scm(seed):
    """Nonlinear Gaussian ANM on N nodes.  Returns (X (n, N), GT (N, N)) with
    GT[i, j] = 1 <=> j -> i (model convention)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    GT = np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            if rng.random() < EDGE_PROB:
                GT[order[b], order[a]] = 1.0     # earlier -> later
    # Random per-edge nonlinearities f(x) = a * sin(b * x)
    coef_a = rng.uniform(0.5, 1.5, (N, N)) * rng.choice([-1, 1], (N, N))
    coef_b = rng.uniform(0.5, 1.5, (N, N))
    X = np.zeros((N_SAMPLES, N))
    for b in range(N):
        i = order[b]
        pa = np.where(GT[i] == 1)[0]
        if len(pa):
            f = (coef_a[i, pa] * np.sin(coef_b[i, pa] * X[:, pa])).sum(axis=1)
            f = f / np.sqrt(len(pa))
        else:
            f = 0.0
        X[:, i] = f + NOISE_SCALE * rng.standard_normal(N_SAMPLES)
    return torch.tensor(X, dtype=torch.float32), GT


# ---- Toy model: the gate math of GatedSelfAttention, deterministic -----------
class ToyGate(nn.Module):
    """Fixed orthonormal keys, free unit queries with a norm budget, plus a
    per-node value map.  ``posterior(detach_anti, detach_sym)`` reproduces the
    eval-mode directed posterior of GatedSelfAttention."""

    def __init__(self, K, scale):
        super().__init__()
        N_, d = K.shape
        self.register_buffer("K", K)                       # (N, d) orthonormal
        self.scale = scale
        c_all = K.mean(dim=0)
        c_all = c_all / c_all.norm()
        q0 = c_all + 0.1 * torch.randn(N_, d)              # centroid init
        self.q = nn.Parameter(q0)
        self.log_M = nn.Parameter(torch.zeros(N_))         # norm budget M_i
        self.value = nn.ModuleList(
            nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(N_))
        with torch.no_grad():                              # small init
            for m in self.value:
                m[-1].weight.mul_(0.1)
                m[-1].bias.zero_()

    def posterior(self, detach_anti=False, detach_sym=False):
        qh = self.q / self.q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q_s = qh * self.log_M.exp()[:, None]
        raw = (q_s @ self.K.T) * self.scale                # (N, N)
        rawT = raw.T
        rawT_anti = rawT.detach() if detach_anti else rawT
        rawT_sym = rawT.detach() if detach_sym else rawT
        s_sym = 0.5 * (raw + rawT_sym)
        a_anti = 0.5 * (raw - rawT_anti)
        p = torch.sigmoid(s_sym - L0_OFFSET) * torch.sigmoid(a_anti / DIR_BETA)
        return p * (1.0 - torch.eye(self.K.shape[0]))      # zero diagonal

    def predict(self, x, detach_anti=False, detach_sym=False):
        v = torch.stack([m(x[:, j:j + 1]).squeeze(-1)
                         for j, m in enumerate(self.value)], dim=1)  # (B, N)
        p = self.posterior(detach_anti, detach_sym)
        return torch.einsum("ij,bj->bi", p, v)


def hsic_term(model, x, detach_anti=False, detach_sym=False):
    pred = model.predict(x, detach_anti, detach_sym)
    return hsic_cross_per_pair(x, x - pred, **HSIC_KW)


# ---- Gradient decomposition ---------------------------------------------------
def grad_decomposition(model, x):
    """Split g = d HSIC / d q into own-row / reverse-antisym / reverse-sym."""
    def g(da, ds):
        return torch.autograd.grad(hsic_term(model, x, da, ds), model.q,
                                   retain_graph=False)[0]
    g_full = g(False, False)
    g_no_anti = g(True, False)          # A_anti transpose detached
    g_no_both = g(True, True)           # + S_sym transpose detached
    return g_no_both, g_full - g_no_anti, g_no_anti - g_no_both


def cos_to(a, c):
    """Row-wise cosine of (N, d) against per-row centroid c (NaN rows)."""
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (an * c).sum(-1)


def parent_centroids(K, GT):
    C = torch.full((N, K.shape[1]), float("nan"), dtype=K.dtype)
    for i in range(N):
        pa = np.where(GT[i] == 1)[0]
        if len(pa):
            c = K[pa].mean(dim=0)
            C[i] = c / c.norm()
    return C


def shd(p, GT):
    pred = (p >= 0.5).numpy()
    return int((pred != GT.astype(bool)).sum())


# ---- One arm -------------------------------------------------------------------
def run_arm(X, GT, K, scale, detach_anti, detach_sym, seed):
    torch.manual_seed(seed)
    model = ToyGate(K, scale)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    C = parent_centroids(K, GT)
    has_par = torch.isfinite(C[:, 0])
    log = {"step": [], "cos_full": [], "cos_app": [], "cos_own": [], "cos_anti": [],
           "cos_sym": [], "norm_own": [], "norm_anti": [], "norm_sym": [],
           "align": [], "shd": [], "hsic": [], "mse": []}
    for t in range(STEPS + 1):
        idx = torch.randint(0, len(X), (BATCH,))
        xb = X[idx]
        if t % LOG_EVERY == 0 or t == STEPS:
            model.eval()
            with torch.enable_grad():
                g_own, g_anti, g_sym = grad_decomposition(model, xb)
                g_full = g_own + g_anti + g_sym
                g_app = (g_own + (0.0 if detach_anti else g_anti)
                         + (0.0 if detach_sym else g_sym))
            with torch.no_grad():
                def cos(g):
                    return cos_to(-g, C)               # -g: update direction
                log["step"].append(t)
                log["cos_full"].append(cos(g_full))
                log["cos_app"].append(cos(g_app))
                log["cos_own"].append(cos(g_own))
                log["cos_anti"].append(cos(g_anti))
                log["cos_sym"].append(cos(g_sym))
                gn = g_full.norm(dim=-1).clamp_min(1e-12)
                log["norm_own"].append(g_own.norm(dim=-1) / gn)
                log["norm_anti"].append(g_anti.norm(dim=-1) / gn)
                log["norm_sym"].append(g_sym.norm(dim=-1) / gn)
                qh = model.q / model.q.norm(dim=-1, keepdim=True)
                log["align"].append((qh * C).sum(-1))
                log["shd"].append(shd(model.posterior().detach(), GT))
                log["hsic"].append(float(hsic_term(model, xb)))
                log["mse"].append(float(((xb - model.predict(xb)) ** 2).mean()))
            model.train()
        if t == STEPS:
            break
        opt.zero_grad()
        pred = model.predict(xb, detach_anti, detach_sym)
        loss = ((xb - pred) ** 2).mean() + LAMBDA_HSIC * hsic_term(
            model, xb, detach_anti, detach_sym)
        loss.backward()
        opt.step()
    for k, v in log.items():
        log[k] = np.stack([np.asarray(r, dtype=float) for r in v])
    log["has_par"] = has_par.numpy()
    return log


# ---- Main -----------------------------------------------------------------------
def main():
    arms = [("A current", False, False),
            ("B detach A_anti", True, False),
            ("C detach both", True, True)]
    all_logs = {name: [] for name, _, _ in arms}
    final_rows = []
    for seed in SEEDS:
        X, GT = make_scm(seed)
        g = torch.Generator().manual_seed(10_000 + seed)
        K = torch.linalg.qr(torch.randn(N, D_MODEL, generator=g))[0]
        scale = 1.0
        print(f"seed {seed}: GT edges = {int(GT.sum())}", flush=True)
        for name, da, ds in arms:
            log = run_arm(X, GT, K, scale, da, ds, seed=7_000 + seed)
            all_logs[name].append(log)
            hp = log["has_par"]
            frac_anti = float((log["cos_app"][:, hp] < 0).mean())
            print(f"  {name:18s} final SHD={int(log['shd'][-1]):2d}  "
                  f"HSIC={log['hsic'][-1]:.4f}  MSE={log['mse'][-1]:.4f}  "
                  f"align={np.nanmean(log['align'][-1][hp]):+.3f}  "
                  f"anti-aligned frac={frac_anti:.2%}", flush=True)
            final_rows.append({"seed": seed, "arm": name,
                               "shd": int(log["shd"][-1]),
                               "hsic": float(log["hsic"][-1]),
                               "mse": float(log["mse"][-1]),
                               "align_final":
                                   float(np.nanmean(log["align"][-1][hp])),
                               "anti_aligned_frac": frac_anti})

    # ---- Aggregate ------------------------------------------------------------
    def agg(name, key):
        hp = all_logs[name][0]["has_par"]
        return np.stack([np.nanmean(l[key][:, hp], axis=1)
                         for l in all_logs[name]])        # (seeds, steps)

    steps = all_logs["A current"][0]["step"]
    summary = {"arms": {}, "per_seed": final_rows,
               "config": {"N": N, "d_model": D_MODEL, "steps": STEPS,
                          "batch": BATCH, "lr": LR, "lambda_hsic": LAMBDA_HSIC,
                          "seeds": SEEDS, "dir_beta": DIR_BETA}}
    for name, _, _ in arms:
        anti_frac = np.stack([(l["cos_app"][:, l["has_par"]] < 0).mean(axis=1)
                              for l in all_logs[name]])   # (seeds, steps)
        summary["arms"][name] = {
            "anti_aligned_frac_mean": float(anti_frac.mean()),
            "anti_aligned_frac_per_seed": [float(a.mean()) for a in anti_frac],
            "final_shd_mean":
                float(np.mean([l["shd"][-1] for l in all_logs[name]])),
            "final_hsic_mean":
                float(np.mean([l["hsic"][-1] for l in all_logs[name]])),
            "final_mse_mean":
                float(np.mean([l["mse"][-1] for l in all_logs[name]])),
            "final_align_mean": float(np.mean(
                [np.nanmean(l["align"][-1][l["has_par"]])
                 for l in all_logs[name]])),
        }
    # Reverse-path evidence in arm A: alignment and norm share of each part.
    A = all_logs["A current"]
    for part in ["own", "anti", "sym"]:
        c = np.stack([np.nanmean(l[f"cos_{part}"][:, l["has_par"]], axis=1)
                      for l in A])
        nrm = np.stack([np.nanmean(l[f"norm_{part}"][:, l["has_par"]], axis=1)
                        for l in A])
        summary["arms"]["A current"][f"part_{part}_cos_mean"] = float(c.mean())
        summary["arms"]["A current"][f"part_{part}_cos_final"] = \
            float(c[:, -1].mean())
        summary["arms"]["A current"][f"part_{part}_norm_share"] = \
            float(nrm.mean())

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Figure 1: anti-aligned fraction over training -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, (name, _, _) in enumerate(arms):
        af = np.stack([(l["cos_app"][:, l["has_par"]] < 0).mean(axis=1)
                       for l in all_logs[name]])
        ax.plot(steps, af.mean(axis=0), color=OKABE_ITO[k], lw=1.4, label=name)
        ax.fill_between(steps, af.min(axis=0), af.max(axis=0),
                        color=OKABE_ITO[k], alpha=0.15)
    ax.set_xlabel("step [-]")
    ax.set_ylabel("fraction of nodes [-]")
    ax.set_title("Anti-aligned HSIC gradient: fraction of nodes with "
                 "cos(-g, parent centroid) < 0")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_anti_aligned_fraction.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: gradient decomposition in arm A ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = {"own": "own-row (via p_j.)", "anti": "reverse A_anti (via p_.j)",
              "sym": "reverse S_sym (via p_.j)"}
    for k, part in enumerate(["own", "anti", "sym"]):
        c = np.stack([np.nanmean(l[f"cos_{part}"][:, l["has_par"]], axis=1)
                      for l in A])
        nrm = np.stack([np.nanmean(l[f"norm_{part}"][:, l["has_par"]], axis=1)
                        for l in A])
        axes[0].plot(steps, c.mean(axis=0), color=OKABE_ITO[k], lw=1.4,
                     label=labels[part])
        axes[1].plot(steps, nrm.mean(axis=0), color=OKABE_ITO[k], lw=1.4,
                     label=labels[part])
    axes[0].axhline(0, color="#999999", lw=0.8)
    axes[0].set_ylabel("cos(-g_part, parent centroid) [-]")
    axes[0].set_title("Arm A: direction of each gradient part")
    axes[1].set_ylabel("||g_part|| / ||g_full|| [-]")
    axes[1].set_title("Arm A: norm share of each part")
    for ax in axes:
        ax.set_xlabel("step [-]")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_grad_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: alignment and SHD trajectories --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (name, _, _) in enumerate(arms):
        al = agg(name, "align")
        sd = np.stack([l["shd"] for l in all_logs[name]])
        axes[0].plot(steps, al.mean(axis=0), color=OKABE_ITO[k], lw=1.4,
                     label=name)
        axes[1].plot(steps, sd.mean(axis=0), color=OKABE_ITO[k], lw=1.4,
                     label=name)
    axes[0].set_ylabel("cos(q, parent centroid) [-]")
    axes[0].set_title("Query-parent-centroid alignment (mean over nodes)")
    axes[1].set_ylabel("SHD [edges]")
    axes[1].set_title("Structural recovery")
    for ax in axes:
        ax.set_xlabel("step [-]")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_alignment_shd.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Verdict ------------------------------------------------------------------
    print("\n==== VERDICT ====")
    for name, _, _ in arms:
        s = summary["arms"][name]
        print(f"{name:18s} anti-aligned frac={s['anti_aligned_frac_mean']:.2%}  "
              f"final SHD={s['final_shd_mean']:.1f}  "
              f"HSIC={s['final_hsic_mean']:.4f}  MSE={s['final_mse_mean']:.4f}  "
              f"align={s['final_align_mean']:+.3f}")
    sA = summary["arms"]["A current"]
    print(f"Arm A parts: own cos={sA['part_own_cos_mean']:+.3f} "
          f"(share {sA['part_own_norm_share']:.2f}), "
          f"rev-anti cos={sA['part_anti_cos_mean']:+.3f} "
          f"(share {sA['part_anti_norm_share']:.2f}), "
          f"rev-sym cos={sA['part_sym_cos_mean']:+.3f} "
          f"(share {sA['part_sym_norm_share']:.2f})")
    print(f"outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
