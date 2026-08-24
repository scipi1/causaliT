"""Fix the dilution readout: resolve the "wrong candidates" aggregate by group.

The aggregate wrong-parent line averaged the BIASED pairs (self, children,
descendants, spouses - irreducibly dependent on r_i, high HSIC) with the large
clean background ("other", at the noise floor), hiding the contamination the
user cares about.  Resolve by group:

1) dilution_compute: print the group-resolved per-pair HSIC at the final epoch.
2) dilution_decomp: add the group-resolved HSIC trajectory (no_NT) - the biased
   groups score high early, everything converges to the floor.
3) bkd_sweep_dilution: right panel becomes the group-resolved per-pair HSIC vs
   dropout (train mode), with the eval true-parent reference for the
   leave-one-out gap.
4) dilution_intro: note that "wrong" is heterogeneous.

Run:  python scripts/_fix_dilution_groups.py
"""
import json
from pathlib import Path

NB = Path("experiments/6_INVESTIGATIONS/LARGER_DAGS/analyze_no_NT_hsic_signal.ipynb")


def append_source(cell, text):
    cell["source"] = ("".join(cell["source"]) + text).splitlines(keepends=True)


def edit_source(cell, old, new):
    src = "".join(cell["source"])
    assert old in src, f"pattern not found in cell {cell['id']!r}:\n{old}"
    cell["source"] = src.replace(old, new, 1).splitlines(keepends=True)


# Group colour map shared by the new figures (Okabe-Ito; self in black).
GROUP_COLORS = {
    "parents": "#0072B2", "children": "#E69F00", "ancestors": "#009E73",
    "descendants": "#D55E00", "spouses": "#CC79A7", "other": "#56B4E9",
    "self": "#000000",
}

GROUP_PRINT = '''

# ---- Group-resolved per-pair HSIC at the final epoch (the "wrong" pairs are heterogeneous) ------
# The biased groups (self, children, descendants, spouses) are irreducibly dependent on
# r_i and score HIGH; the clean background ("other") sits at the noise floor.  The
# aggregate "wrong" mean mixes the two and hides the contamination.
GROUP_COLORS = {"parents": "#0072B2", "children": "#E69F00", "ancestors": "#009E73",
                "descendants": "#D55E00", "spouses": "#CC79A7", "other": "#56B4E9",
                "self": "#000000"}
print("\\nMean per-pair HSIC by candidate group at the final epoch (eval mode):")
print(f"{'arm':>28} | " + " | ".join(f"{g:>11}" for g in GROUPS) + " |        self")
for name in MARGIN_ARMS:
    e = max(HMAT[name])
    H = HMAT[name][e]
    gm = {g: np.nanmean(np.where(MASK_G[g], H, np.nan)[N_S:]) for g in GROUPS}
    gm["self"] = np.nanmean(np.where(EYE, H, np.nan)[N_S:])
    print(f"{name:>28} | " + " | ".join(f"{gm[g]:11.5f}" for g in GROUPS)
          + f" | {gm['self']:11.5f}")
'''

GROUP_TRAJ = '''

# ---- Group-resolved HSIC trajectory (no_NT): the biased pairs DO score high early -----------------
# Read against the aggregate in Section 8: the "wrong candidates" mean is dominated by
# the clean "other" background; the biased groups (self, children, descendants, spouses)
# sit ABOVE the true parents early and converge to the floor only as the residuals shrink.
fig, ax = plt.subplots(figsize=(7.2, 4.0))
arm = "no_NT"
ep = sorted(HMAT[arm])
for g in GROUPS + ["self"]:
    mask = EYE if g == "self" else MASK_G[g]
    traj = [np.nanmean(np.where(mask, HMAT[arm][e], np.nan)[N_S:]) for e in ep]
    ax.plot(ep, traj, "o-", color=GROUP_COLORS[g], lw=1.4, label=g)
ax.set_xlim(min(ep), max(ep))
ax.set_xlabel("epoch [-]")
ax.set_ylabel("mean per-pair HSIC [-]")
ax.set_title(f"Per-pair HSIC by candidate group ({arm})\\n"
             "biased groups (self, children, descendants, spouses) score high early")
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "dilution_groups_trajectory")
plt.show()
'''

# Replace ONLY the right panel (axes[1]); the left R panel (axes[0]) is untouched.
NEW_BKD_FIG = '''# Right panel: the wrong candidates resolved by group (train mode).  The aggregate
# "wrong" mean is dominated by the clean "other" background; the biased groups
# (self, children+descendants, spouses) are the ones that out-score the true parents.
GROUP_H = [("true parents", MASK_G["parents"], GROUP_COLORS["parents"]),
           ("self", EYE, GROUP_COLORS["self"]),
           ("children+descendants", MASK_G["children"] | MASK_G["descendants"],
            GROUP_COLORS["descendants"]),
           ("spouses", MASK_G["spouses"], GROUP_COLORS["spouses"]),
           ("clean background", MASK_G["ancestors"] | MASK_G["other"],
            GROUP_COLORS["other"])]
for label, mask, col in GROUP_H:
    vals = [np.nanmean(np.where(mask, BKD_H[p][e_last]["train"], np.nan)[N_S:])
            for p in BKD_LEVELS]
    axes[1].plot(BKD_LEVELS, vals, "o-", color=col, lw=1.4, label=label)
# eval reference on the true parents: the train-eval gap is the leave-one-out inflation.
vals = [np.nanmean(np.where(MASK_G["parents"], BKD_H[p][e_last]["eval"], np.nan)[N_S:])
        for p in BKD_LEVELS]
axes[1].plot(BKD_LEVELS, vals, "s--", color=GROUP_COLORS["parents"], lw=1.0,
             alpha=0.6, label="true parents (eval)")
axes[1].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[1].set_xlabel("batch key dropout p [-]")
axes[1].set_ylabel("mean per-pair HSIC [-]")
axes[1].set_title(f"Per-pair HSIC by candidate group vs dropout\\n(train mode, epoch {e_last})")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "bkd_sweep_dilution")
plt.show()'''

OLD_BKD_FIG = '''s = sub[sub["mode"] == "train"].sort_values("p")
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
plt.show()'''

INTRO_NOTE_OLD = """pairs split into the true parents (j in PA_i) and the wrong candidates.
**HSIC dilution**"""
INTRO_NOTE_NEW = """pairs split into the true parents (j in PA_i) and the wrong candidates.  The
wrong candidates are HETEROGENEOUS: the biased pairs (self, children,
descendants, spouses) are irreducibly dependent on r_i and score HIGH, while
the clean background ("other") sits at the noise floor - so the aggregate
"wrong" mean mixes the two and must be read by group (below).
**HSIC dilution**"""


def replace_from_marker(cell, marker, new_tail):
    # Replace everything from `marker` to the end of the cell with new_tail.
    # Robust to cosmetic drift (e.g. a title line break) inside the old block.
    src = "".join(cell["source"])
    start = src.find(marker)
    assert start != -1, f"marker not found in cell {cell['id']!r}:\n{marker}"
    cell["source"] = (src[:start] + new_tail).splitlines(keepends=True)


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    cells = nb["cells"]
    by_id = {c["id"]: i for i, c in enumerate(cells) if "id" in c}

    edit_source(cells[by_id["dilution_intro"]], INTRO_NOTE_OLD, INTRO_NOTE_NEW)
    append_source(cells[by_id["dilution_compute"]], GROUP_PRINT)
    append_source(cells[by_id["dilution_decomp"]], GROUP_TRAJ)
    # The right panel is the last block of the cell (ends in plt.show()).
    replace_from_marker(cells[by_id["bkd_sweep_probe"]],
                        's = sub[sub["mode"] == "train"].sort_values("p")',
                        NEW_BKD_FIG)

    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    nb2 = json.load(open(NB, encoding="utf-8"))
    import nbformat
    nbformat.validate(nb2)
    print(f"wrote {NB}: {len(nb2['cells'])} cells; nbformat.validate OK")


if __name__ == "__main__":
    main()
