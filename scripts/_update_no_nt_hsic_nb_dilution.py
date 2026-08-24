"""Revise analyze_no_NT_hsic_signal.ipynb: the HSIC dilution readout (per-node
true-parent vs wrong-candidate HSIC ratio).

Context: the first pass added a gradient-SNR probe for the BKD sweep, but the
dilution question is about the LOSS landscape: per node i, with
H[i, j] = HSIC(X_j, r_i),

    R_i = mean_{j in PA_i} H[i, j] / mean_{j not in PA_i} H[i, j]

and the true-parent mass share F_i = sum_{PA} H / sum_{all} H.  Many wrong
pairs with small HSIC dilute the mean; the descendant mask removes the BIASED
wrong pairs (self + children + descendants), raising the post-mask contrast.

Edits:
1) Section 7: insert the dilution subsection (definition, per-pair H matrices
   from checkpoints, R/F trajectories per arm, final-epoch mass decomposition,
   pre- vs post-mask contrast for the nodesc arms).
2) Section 8: replace the gradient-SNR probe with the dilution matrix vs
   dropout (train mode = BKD active vs eval mode = clean reference); rewrite
   the intro and answer templates with the correct definition.
3) Synthesis: updated prompts.

Run:  python scripts/_update_no_nt_hsic_nb_dilution.py
"""
import json
from pathlib import Path

NB = Path("experiments/6_INVESTIGATIONS/LARGER_DAGS/analyze_no_NT_hsic_signal.ipynb")


def md_cell(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code_cell(cell_id, text):
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


DILUTION_INTRO = """\
### The HSIC dilution, per node

The structural loss is the MEAN over all (child, candidate-parent) pairs,
`hsic = mean_{i,j} H[i, j]` with `H[i, j] = HSIC(X_j, r_i)`.  For child i the
pairs split into the true parents (j in PA_i) and the wrong candidates.
**HSIC dilution**: the per-node contrast

    R_i = mean_{j in PA_i} H[i, j] / mean_{j not in PA_i} H[i, j]

and the true-parent mass share `F_i = sum_{PA_i} H / sum_{all} H`.  Many
wrong pairs each carry a small HSIC, but there are MANY of them: when R_i ~ 1
(and F_i ~ the chance level |PA_i| / N) the mean cannot tell the true parents
from the background - the gradient is spent on the wrong pairs.  The wrong
pairs split into the BIASED ones (self, children, descendants - irreducibly
dependent on r_i under an ANM, see the descendant_mask module) and the clean
background (ancestors, spouses, other).  The descendant mask removes the
biased part, so the POST-MASK ratio R_kept (true parents vs kept wrong pairs)
is the contrast the masked loss actually sees.

**Readouts.**  H recomputed from checkpoints (eval mode, training-time HSIC
settings, fixed batches, no backward): the R / F trajectories per arm, the
final-epoch per-node mass decomposition, and the pre- vs post-mask ratio for
the nodesc arms.  Note: R -> 1 at the end of training only means the true
parents no longer stand out in the residual - either because they are attended
(the signal was consumed) or because the residuals carry no contrast at all
(the signal never materialized).  The trajectory disambiguates.
"""

DILUTION_COMPUTE = """\
# ---- Per-pair HSIC matrices from checkpoints (eval mode, no backward) --------------------------
from causaliT.utils.hsic_utils import hsic_pair_matrix

DILUTE_EPOCHS = [49, 199, 499, 999]   # same readout points as the Section-4 probe


def hsic_matrix_at(ckpt_path, batches, train_mode=False):
    # Per-pair HSIC matrix H[i, j] = HSIC(X_j, r_i), averaged over batches.
    # train_mode=True runs the training-time forward (gate sampling + BKD).
    model = AttentionSelectorForecaster.load_from_checkpoint(ckpt_path, map_location="cpu")
    if train_mode:
        model.train()
    else:
        model.eval()
    Hs = []
    with torch.no_grad():
        for s_b, x_b in batches:
            pred = model.forward(data_source=s_b, data_intermediate=x_b)[0]
            x_target = torch.cat([s_b, x_b], dim=1)[:, :, 0]          # (B, N)
            residuals = x_target - pred.squeeze(-1)
            Hs.append(hsic_pair_matrix(x_target, residuals, sigma=1.0,
                                       adaptive_bandwidth=True, mode="biased",
                                       source_kernel="rbf").double().numpy())
    del model
    return np.stack(Hs).mean(axis=0)                                  # (N, N)


# Shared batches (common random numbers across arms).
perm = torch.randperm(S_ALL.shape[0], generator=torch.Generator().manual_seed(SEED))
BATCHES = [(S_ALL[b], X_ALL[b]) for b in perm.split(BATCH_SIZE)[:N_BATCHES]]

ARM_CKPTS = {"no_NT": ckpts}
for name, d in MARGIN_ARM_DIRS.items():
    ARM_CKPTS[name] = {e: p for p in (d / "k_0" / "checkpoints").glob("*.ckpt")
                       if (e := ckpt_epoch(p)) is not None}

HMAT = {}   # HMAT[arm][epoch] = (N, N) per-pair HSIC, eval mode
for name in MARGIN_ARMS:
    HMAT[name] = {}
    for target in DILUTE_EPOCHS:
        e = min(ARM_CKPTS[name], key=lambda x: abs(x - target))
        HMAT[name][e] = hsic_matrix_at(ARM_CKPTS[name][e], BATCHES)
        print(f"{name} epoch {e}: mean H (off-diag) = {HMAT[name][e][~EYE].mean():.5f}")

# ---- The dilution ratio per node ---------------------------------------------------------------
WRONG_ALL = ~PA                                   # every non-parent candidate (incl. self)
BIASED = EYE | MASK_G["children"] | MASK_G["descendants"]   # removed by the mask
KEPT_WRONG = WRONG_ALL & ~BIASED                  # clean background (anc/spouse/other)


def ratio_per_node(H, denom_mask):
    # R_i = mean_{j in PA_i} H[i, j] / mean_{j in denom_mask_i} H[i, j].
    num = np.nanmean(np.where(PA, H, np.nan), axis=1)
    den = np.nanmean(np.where(denom_mask, H, np.nan), axis=1)
    return num / den                                                    # (N,)


def mass_share_per_node(H):
    # F_i = sum_{j in PA_i} H[i, j] / sum_{all j} H[i, j] (the raw mean's share).
    return (np.where(PA, H, 0.0).sum(axis=1)
            / np.maximum(H.sum(axis=1), 1e-12))                         # (N,)


for name in ["no_NT_bkd_02_nodesc", "no_NT_bkd_02_nodesc_budget"]:
    e = max(HMAT[name])
    H = HMAT[name][e]
    print(f"{name} @ {e}: R_raw={np.nanmean(ratio_per_node(H, WRONG_ALL)[N_S:]):.2f} "
          f"R_kept={np.nanmean(ratio_per_node(H, KEPT_WRONG)[N_S:]):.2f} "
          f"F={np.nanmean(mass_share_per_node(H)[N_S:]):.2%}")
"""

DILUTION_TRAJ = """\
# ---- Dilution trajectories per arm ---------------------------------------------------------------
chance_F = np.mean([PA[i].sum() for i in range(N_S, N)]) / N   # all pairs equal

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for name in MARGIN_ARMS:
    ep = sorted(HMAT[name])
    R = np.stack([ratio_per_node(HMAT[name][e], WRONG_ALL) for e in ep])   # (E, N)
    F = np.stack([mass_share_per_node(HMAT[name][e]) for e in ep])
    axes[0].plot(ep, np.nanmean(R[:, N_S:], axis=1), "o-", lw=1.4,
                 color=MARGIN_ARM_COLORS[name], label=name)
    axes[1].plot(ep, np.nanmean(F[:, N_S:], axis=1), "o-", lw=1.4,
                 color=MARGIN_ARM_COLORS[name], label=name)
axes[0].axhline(1, color="#999999", lw=0.8, ls="--")
axes[0].set_xlim(min(DILUTE_EPOCHS), max(DILUTE_EPOCHS))
axes[0].set_xlabel("epoch [-]")
axes[0].set_ylabel("R [-]")
axes[0].set_title("HSIC contrast R = H(true parents) / H(wrong)\\n(mean over X nodes)")
axes[0].legend(fontsize=8)
axes[1].axhline(chance_F, color="#999999", lw=0.8, ls="--", label="chance")
axes[1].set_xlim(min(DILUTE_EPOCHS), max(DILUTE_EPOCHS))
axes[1].set_xlabel("epoch [-]")
axes[1].set_ylabel("F [-]")
axes[1].set_title("True-parent share of the HSIC mass F\\n(mean over X nodes)")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "dilution_trajectories")
plt.show()
"""

DILUTION_DECOMP = """\
# ---- Final-epoch decomposition of the HSIC mass (budget arm) --------------------------------------
DECOMP_GROUPS = [
    ("true parents", MASK_G["parents"], OKABE_ITO[0]),
    ("self", EYE, OKABE_ITO[7]),
    ("children+descendants (masked)", MASK_G["children"] | MASK_G["descendants"], OKABE_ITO[3]),
    ("clean background (anc/spouse/other)",
     MASK_G["ancestors"] | MASK_G["spouses"] | MASK_G["other"], OKABE_ITO[2]),
]
arm = "no_NT_bkd_02_nodesc_budget"
e = max(HMAT[arm])
H = HMAT[arm][e]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
xs = np.arange(N_S, N)
bottom = np.zeros(N - N_S)
for label, mask, color in DECOMP_GROUPS:
    mass = np.where(mask, H, 0.0)[N_S:].sum(axis=1)
    axes[0].bar(xs, mass, bottom=bottom, color=color, label=label)
    bottom += mass
axes[0].set_xticks(list(xs), NODES[N_S:], rotation=90)
axes[0].set_xlabel("child i")
axes[0].set_ylabel("HSIC mass per row [-]")
axes[0].set_title(f"HSIC mass decomposition per node\\n({arm}, epoch {e})")
axes[0].legend(fontsize=8)

vmax = np.quantile(H[~EYE], 0.99)   # the diagonal may saturate
im = axes[1].imshow(H, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
mark_gt(axes[1])
axes[1].set_xticks(range(N), NODES, rotation=90)
axes[1].set_yticks(range(N), NODES)
axes[1].set_xlabel("source j")
axes[1].set_ylabel("child i")
axes[1].set_title(f"Per-pair HSIC H[i, j] (epoch {e})\\nGT edges framed")
fig.colorbar(im, ax=axes[1], fraction=0.046, label="HSIC [-]")
fig.tight_layout()
save_fig(fig, "dilution_decomposition")
plt.show()
"""

SECTION8_INTRO = """\
## 8. BKD dose-response: do we dilute the HSIC less with high dropout?

**Sweep:** `bkd_sweep_11437173` - `batch_key_dropout` (BKD) in {0.0, 0.05,
0.1, 0.2, 0.3} on top of the no_NT config (same dataset, same seed; p = 0.0
reproduces no_NT; no descendant exclusion here: the plain HSIC mean).  BKD
zeroes entire key COLUMNS of the attention matrix with a single mask drawn
once per batch (batch-consistent), train mode only, no 1/(1-p) rescaling.

**The dilution question (per node).**  With `H[i, j] = HSIC(X_j, r_i)`, the
structural loss averages all pairs of a child row, and the true parents stand
out only if their per-pair HSIC exceeds the wrong-candidate background:
`R_i = mean_{PA_i} H / mean_{wrong} H` (Section 7).  Many wrong pairs with a
small HSIC dilute the mean.  What does BKD do to R?  When a true parent's key
is dropped, its contribution stays in the residual, so `HSIC(X_j, r_i)` spikes
exactly where an edge is missing (leave-one-out); a dropped WRONG key that was
not attended changes nothing.  If this selective inflation dominates, R_train
RISES with p: high dropout dilutes the HSIC LESS.

**Readouts.**
- logged metrics: val_hsic (clean, eval mode) vs train_hsic (computed UNDER
  dropout) - the train/val gap measures the dropout inflation of the
  training-time signal;
- structure learning: the Section-5 direction margins per level;
- the dilution matrix: H recomputed at matched checkpoints in TRAIN mode
  (BKD + gate sampling active: the loss the optimizer actually sees) and in
  eval mode (the clean reference) - R(p) and the mean true-parent vs
  wrong-pair HSIC per level.
"""

SECTION8_DILUTION = """\
# ---- The dilution matrix vs dropout: train-mode (BKD active) vs eval-mode -------------------------
# H[i, j] = HSIC(X_j, r_i) per pair, averaged over the shared batches.  Train mode:
# the residual is computed with a p-fraction of keys blanked - a dropped true parent
# leaves its contribution in the residual (leave-one-out inflation), a dropped wrong
# key that was not attended changes nothing.  If the true-parent HSIC rises relative
# to the wrong-pair background with p, high dropout dilutes the HSIC LESS.
BKD_H = {}   # BKD_H[p][epoch][mode] = (N, N) per-pair HSIC
for p in BKD_LEVELS:
    BKD_H[p] = {}
    for target in [500, 999]:
        e = min(BKD[p]["ckpts"], key=lambda x: abs(x - target))
        out = {}
        for mode, tm in [("eval", False), ("train", True)]:
            torch.manual_seed(SEED)   # comparable gate/BKD draws across levels
            out[mode] = hsic_matrix_at(BKD[p]["ckpts"][e], BATCHES, train_mode=tm)
        BKD_H[p][e] = out
        print(f"p={p}: epoch {e} done")

e_last = max(e for p in BKD_LEVELS for e in BKD_H[p])
rows = []
for p in BKD_LEVELS:
    for e, modes in BKD_H[p].items():
        for mode, H in modes.items():
            rows.append(dict(p=p, epoch=e, mode=mode,
                             R=np.nanmean(ratio_per_node(H, WRONG_ALL)[N_S:]),
                             F=np.nanmean(mass_share_per_node(H)[N_S:]),
                             H_true=np.nanmean(np.where(PA, H, np.nan)[N_S:]),
                             H_wrong=np.nanmean(np.where(WRONG_ALL, H, np.nan)[N_S:])))
bkd_dilution = pd.DataFrame(rows)
print(bkd_dilution.round(4).to_string(index=False))

sub = bkd_dilution[bkd_dilution.epoch == e_last]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for mode, mkr, col, lab in [("train", "o", OKABE_ITO[0], "train (BKD active)"),
                            ("eval", "s", OKABE_ITO[2], "eval (clean)")]:
    s = sub[sub["mode"] == mode].sort_values("p")
    axes[0].plot(s.p, s.R, mkr + "-", color=col, lw=1.4, label=lab)
axes[0].axhline(1, color="#999999", lw=0.8, ls="--")
axes[0].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[0].set_xlabel("batch key dropout p [-]")
axes[0].set_ylabel("R [-]")
axes[0].set_title(f"HSIC contrast R vs dropout (epoch {e_last})")
axes[0].legend(fontsize=9)

s = sub[sub["mode"] == "train"].sort_values("p")
axes[1].plot(s.p, s.H_true, "o-", color=OKABE_ITO[0], lw=1.4, label="true parents (train)")
axes[1].plot(s.p, s.H_wrong, "o-", color=OKABE_ITO[3], lw=1.4, label="wrong (train)")
se = sub[sub["mode"] == "eval"].sort_values("p")
axes[1].plot(se.p, se.H_true, "s--", color=OKABE_ITO[0], lw=1.0, alpha=0.6,
             label="true parents (eval)")
axes[1].plot(se.p, se.H_wrong, "s--", color=OKABE_ITO[3], lw=1.0, alpha=0.6,
             label="wrong (eval)")
axes[1].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[1].set_xlabel("batch key dropout p [-]")
axes[1].set_ylabel("mean per-pair HSIC [-]")
axes[1].set_title(f"Leave-one-out inflation (epoch {e_last})")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "bkd_sweep_dilution")
plt.show()
"""

SECTION8_ANSWER = """\
### Do we dilute the HSIC less with high dropout? (fill in after running)

- R_train(p) vs R_eval(p) at the final epoch (does the training-time contrast
  rise with p?): ...
- The leave-one-out split: does BKD inflate H on the true-parent pairs
  selectively (H_true up, H_wrong flat)?: ...
- The train/val HSIC gap vs p (dropout inflation of the logged signal): ...
- Structure learning (margins / oriented fraction) vs p: ...
- Verdict: ...
"""

SYNTHESIS = """\
## 9. Synthesis (fill in after running)

### Is HSIC pushing in the correct dimension?
- cos(-g, parent centroid) vs the other groups, per checkpoint: ...
- Contaminating groups (children / descendants / spouses): ...

### Is the signal dying?
- Gradient magnitude per checkpoint: ...

### The momentum question
- Per-batch consistency (SNR): ... -> if the per-batch gradient is consistent,
  more momentum / a larger structural LR helps; if it is noisy, the bottleneck
  is the signal itself.

### The budgeted descendant mask (Section 7)
- kept_frac trajectory (does the cap bind? closure ties?): ...
- The end-of-training soft cycle (cyclic flag + checkpoint reproduction): ...
- Budget vs threshold arm on the Section-5 margins: ...
- The dilution ratio: R / F trajectories per arm, pre- vs post-mask contrast: ...

### The BKD dose-response (Section 8)
- Do we dilute the HSIC less with high dropout? (R_train(p) vs R_eval(p),
  leave-one-out split H_true vs H_wrong): ...
- Best dropout level (signal vs reconstruction trade-off): ...

### Notes
- The Section-4 probe runs in eval mode (deterministic gates): the mean-field
  gradient of the stochastic training gates.
- The dilution readout (Sections 7-8) is the per-pair HSIC matrix
  H[i, j] = HSIC(X_j, r_i) recomputed from checkpoints (no backward); Section 8
  adds the train-mode forward (gate sampling + BKD active) for the
  training-time loss matrix.
- The budget M ~ 1.02 is not the constraint; the direction is.
- The displacement attribution (Section 3) is the realized movement; the
  gradient probe (Section 4) is the force - compare them.
- TODO (future arm): safe NOTEARS integration - see Section 7.
"""


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    cells = nb["cells"]
    by_id = {c["id"]: i for i, c in enumerate(cells) if "id" in c}

    # ---- 1) Section 7: insert the dilution subsection after the metrics table -------------------
    new_cells = [
        md_cell("dilution_intro", DILUTION_INTRO),
        code_cell("dilution_compute", DILUTION_COMPUTE),
        code_cell("dilution_traj", DILUTION_TRAJ),
        code_cell("dilution_decomp", DILUTION_DECOMP),
    ]
    pos = by_id["nodesc_budget_table"] + 1
    cells[pos:pos] = new_cells

    # ---- 2) Section 8: correct the dilution definition, replace the probe -----------------------
    by_id = {c["id"]: i for i, c in enumerate(cells) if "id" in c}   # re-index
    cells[by_id["bkd_sweep_intro"]]["source"] = SECTION8_INTRO.splitlines(keepends=True)
    cells[by_id["bkd_sweep_probe"]]["source"] = SECTION8_DILUTION.splitlines(keepends=True)
    cells[by_id["bkd_sweep_answer"]]["source"] = SECTION8_ANSWER.splitlines(keepends=True)

    # ---- 3) Synthesis ----------------------------------------------------------------------------
    cells[by_id["fd9b2158"]]["source"] = SYNTHESIS.splitlines(keepends=True)

    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    nb2 = json.load(open(NB, encoding="utf-8"))
    ids = [c["id"] for c in nb2["cells"] if "id" in c]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    import nbformat
    nbformat.validate(nb2)
    print(f"wrote {NB}: {len(nb2['cells'])} cells; nbformat.validate OK")


if __name__ == "__main__":
    main()
