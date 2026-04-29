"""One-shot helper: append the Stage-2 stability section to
``notebooks/nb_edge_decisiveness.ipynb``.

Run from repo root:

    python scripts/_add_stage2_cells.py

Idempotent: if the section already exists (detected by a marker comment),
it is replaced rather than duplicated.
"""

from __future__ import annotations

import json
import pathlib
import sys

NB = pathlib.Path("notebooks/nb_edge_decisiveness.ipynb")
MARKER = "# === Stage-2 stability section (auto-inserted) ==="


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


HEADER_MD = """\
---

## 6. Stage-2 stability — `C* = C3_dmodel_96`, two seeds per arm

The two duplicate `C3_dmodel_96` runs (same config, byte-for-byte)
ended up in two visibly different HSIC basins. To probe how chaotic
the optimizer is, every Stage-2 arm is launched twice with different
seeds (42 / 43) and we report:

* **per-seed metrics** (the usual `test_*` headline columns), and
* **cross-seed Δ** — `|metric(seed42) − metric(seed43)|`,

so that an arm that *both* sits on the (decisiveness, MAE) Pareto
front *and* has a small Δ wins.
"""

CODE_LOAD = """\
# === Stage-2 stability section (auto-inserted) ===
# Pairs `<arm>_seed42_<jobid>` with `<arm>_seed43_<jobid>` under
# `RESULTS_ROOT` and computes a per-arm cross-seed Δ table.

STAGE2_ARMS = {
    'L1_lr_3e-4':  'L1 — struct lr 3e-4 (= C*)',
    'L2_lr_1e-3':  'L2 — struct lr 1e-3',
    'L3_lr_3e-3':  'L3 — struct lr 3e-3',
    'L4_lr_1e-2':  'L4 — struct lr 1e-2',
    'S1_noise':    'S1 — grad noise (Langevin)',
    'S2_restart':  'S2 — cosine warm restarts',
}
STAGE2_SEEDS = (42, 43)

def find_seeded_arm_dirs(arm: str, seeds=STAGE2_SEEDS):
    \"\"\"Return {seed: Path} resolving `<arm>_seed{seed}_<jobid>` under RESULTS_ROOT.\"\"\"
    out = {}
    for s in seeds:
        matches = sorted(RESULTS_ROOT.glob(f'{arm}_seed{s}_*'))
        out[s] = matches[-1] if matches else None
    return out

STAGE2_DIRS = {arm: find_seeded_arm_dirs(arm) for arm in STAGE2_ARMS}
for arm, by_seed in STAGE2_DIRS.items():
    line = ', '.join(f'seed{s} -> {p.name if p else "MISSING"}' for s, p in by_seed.items())
    print(f'{arm:14s} {line}')
"""

CODE_TABLE = """\
# Cross-seed Δ table for the headline metrics.
STAGE2_METRICS = METRICS  # reuse the same list defined earlier in the notebook.

def _row_for(folder):
    if folder is None or not (folder / 'kfold_summary.json').exists():
        return {k: np.nan for k in STAGE2_METRICS}
    j = json.loads((folder / 'kfold_summary.json').read_text())
    m = j['fold_results']['0']['metrics']
    return {k: m.get(k, np.nan) for k in STAGE2_METRICS}

records = []
for arm, label in STAGE2_ARMS.items():
    by_seed = STAGE2_DIRS[arm]
    a = _row_for(by_seed.get(42))
    b = _row_for(by_seed.get(43))
    rec = {'arm': label}
    for k in STAGE2_METRICS:
        va, vb = a[k], b[k]
        rec[f'{k}__s42'] = va
        rec[f'{k}__s43'] = vb
        rec[f'{k}__delta'] = abs(va - vb) if (va is not None and vb is not None and not (np.isnan(va) or np.isnan(vb))) else np.nan
    records.append(rec)

df_stage2 = pd.DataFrame(records).set_index('arm')
df_stage2.round(4)
"""

CODE_DELTA_VIEW = """\
# Compact Δ-only view, colour-coded: lower delta = greener.
delta_cols = [c for c in df_stage2.columns if c.endswith('__delta')]
df_delta = df_stage2[delta_cols].copy()
df_delta.columns = [c.replace('__delta', '') for c in df_delta.columns]
df_delta.style.background_gradient(axis=0, cmap='RdYlGn_r').format('{:.4f}')
"""

CODE_PARETO = """\
# Per-seed (test_x_mae, test_self_score_sparse) scatter with arm-mean
# anchored by an error bar to make stability visible at a glance.
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap('tab10')

for i, (arm, label) in enumerate(STAGE2_ARMS.items()):
    color = cmap(i % 10)
    xs, ys = [], []
    for s in STAGE2_SEEDS:
        f = STAGE2_DIRS[arm].get(s)
        if f is None or not (f / 'kfold_summary.json').exists():
            continue
        j = json.loads((f / 'kfold_summary.json').read_text())
        m = j['fold_results']['0']['metrics']
        xs.append(m.get('test_x_mae'))
        ys.append(m.get('test_self_score_sparse'))
    if not xs:
        continue
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    ax.scatter(xs, ys, color=color, s=60, alpha=0.55, label=None)
    ax.errorbar(
        xs.mean(), ys.mean(),
        xerr=xs.std(ddof=0), yerr=ys.std(ddof=0),
        fmt='o', mfc='white', mec=color, ecolor=color, capsize=4, lw=2,
        label=label,
    )

ax.set_xlabel('test_x_mae  (lower = better fit)')
ax.set_ylabel('test_self_score_sparse  (higher = more decisive)')
ax.set_title('Stage-2 Pareto: fit vs. decisiveness, error bars = seed std')
ax.legend(loc='best', fontsize=8, frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""

CODE_HEATMAP_DIFF = """\
# For each arm, side-by-side heatmaps of the final mean dec_self
# attention from seed42 vs seed43, plus their absolute difference.
# This is the most direct chaos-vs-stability visual.
def _load_self(folder):
    f = folder / 'eval' / 'eval_attention_scores' / 'files' / 'learned_dag_edges.json'
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    # try common keys; fall back to first dec_self_* entry.
    for key in ('dec_self_L0_mean', 'dec_self_mean', 'dec_self_L0'):
        if key in j:
            return np.array(j[key])
    for k, v in j.items():
        if k.startswith('dec_self'):
            return np.array(v)
    return None

n_arms = len(STAGE2_ARMS)
fig, axes = plt.subplots(n_arms, 3, figsize=(11, 3.2 * n_arms), squeeze=False)
for r, (arm, label) in enumerate(STAGE2_ARMS.items()):
    f42 = STAGE2_DIRS[arm].get(42)
    f43 = STAGE2_DIRS[arm].get(43)
    A42 = _load_self(f42) if f42 is not None else None
    A43 = _load_self(f43) if f43 is not None else None
    titles = [f'{label} | seed42', f'{label} | seed43', '|Δ|  (lower = stable)']
    mats = [A42, A43, (np.abs(A42 - A43) if (A42 is not None and A43 is not None and A42.shape == A43.shape) else None)]
    for c, (mat, title) in enumerate(zip(mats, titles)):
        ax = axes[r, c]
        if mat is None:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
        else:
            cmap = 'YlOrRd' if c < 2 else 'Greys'
            sns.heatmap(mat, ax=ax, vmin=0, vmax=1, cmap=cmap, cbar=False,
                        annot=True, fmt='.2f', annot_kws={'fontsize': 7}, square=True)
        ax.set_title(title, fontsize=9)
plt.suptitle('Stage-2: dec_self attention — seed42 vs seed43 vs |Δ|', y=1.01, fontweight='bold')
plt.tight_layout()
plt.show()
"""

NEW_CELLS = [
    md(HEADER_MD),
    code(CODE_LOAD),
    code(CODE_TABLE),
    code(CODE_DELTA_VIEW),
    code(CODE_PARETO),
    code(CODE_HEATMAP_DIFF),
]


def main() -> int:
    if not NB.is_file():
        print(f"ERROR: notebook not found at {NB}", file=sys.stderr)
        return 1
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # find / drop a previously inserted block (any cell whose source contains MARKER)
    keep = []
    drop_after_idx = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if MARKER in src:
            # drop this cell + the preceding markdown header (last item in keep)
            if keep and keep[-1]["cell_type"] == "markdown" and "Stage-2 stability" in "".join(keep[-1]["source"]):
                keep.pop()
            drop_after_idx = i
            continue
        keep.append(c)

    # If a marker was found, we also drop subsequent auto-section cells (until
    # next top-level "## " markdown header that's not from this section).
    # Simpler: rebuild fresh. We assume the auto section is always tail-only.
    if drop_after_idx is not None:
        # ``keep`` already excludes the cell containing MARKER. We additionally
        # strip the trailing Stage-2 cells we recognise by their content.
        sigs = (
            "df_stage2",
            "df_delta",
            "STAGE2_ARMS",
            "_load_self(folder",
            "Stage-2 Pareto",
        )
        while keep and any(s in "".join(keep[-1].get("source", [])) for s in sigs):
            keep.pop()

    keep.extend(NEW_CELLS)
    nb["cells"] = keep
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"appended {len(NEW_CELLS)} cells to {NB} (total cells now: {len(keep)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
