"""Build the evaluation notebook for experiments/7_PUBLISH/ATE/results.

Generates evaluate_ate_results.ipynb via nbformat (re-runnable).
"""
from pathlib import Path

import nbformat as nbf

OUT = Path("experiments/7_PUBLISH/ATE/results/evaluate_ate_results.ipynb")

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip()))


# ---------------------------------------------------------------------------
md('''
# ATE Benchmark - Evaluation

Evaluation of the ATE sweep archived in this folder: several model arms x three datasets (`ds_scm1/2/3_continuous`) x five model seeds.

- **Auto-discovery**: the notebook scans the `<model>_<jobid>/` folders next to itself, so it can be copy-pasted into any compatible results folder and re-run as-is.
- **Collected per run**: ATE metrics (`eval/eval_ate_mc/files/ate_metrics_mc.csv`), structural DAG metrics (`eval/eval_attention_scores/files/dag_metrics.json`), training metrics (`kfold_summary.json`, `k_0/logs/csv/version_0/metrics.csv`).
- **Aggregations**: over model seeds, then over ATE categories - intervention regime (in/out-of-distribution) x edge path type (direct/indirect/zero).
- **Outputs**: training sanity check, structural metrics, final comparison table (display + LaTeX), and a 1x3 spiderweb figure. All figures are written to `./img/` as `.png` + `.pdf`, sized for the A4 short side.

> Do NOT compare absolute errors across datasets (different normalizations); compare models **within** a dataset.
''')

code(r'''
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The notebook lives next to the <model>_<jobid> folders.
RESULTS_ROOT = Path.cwd()

# Figures: png + pdf into ./img, sized for the A4 SHORT side (portrait width).
IMG_DIR = RESULTS_ROOT / "img"
IMG_DIR.mkdir(exist_ok=True)

A4_SHORT_IN = 8.27          # 210 mm
FIG_W = A4_SHORT_IN - 1.38  # ~175 mm text width (1.75 cm margins)

# Colorblind-safe palette everywhere (seaborn 'colorblind' = Okabe-Ito style).
PALETTE = sns.color_palette("colorblind")
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})


def save_fig(fig, name: str, exts=(".png", ".pdf")) -> None:
    # Write a figure to ./img in every requested format.
    for ext in exts:
        fig.savefig(IMG_DIR / f"{name}{ext}", bbox_inches="tight")


# Display labels per model arm (drives the table columns and the legend).
ROLE_MAP = {"vanilla": "Vanilla", "svfa": "Ours", "cheater": "Cheater"}
BASELINE_MODEL = "vanilla"   # reference arm for the Delta% columns

# Human-readable dataset descriptions (figure titles use these alone).
DATASET_LABELS = {
    "ds_scm1_continuous": "Linear Gaussian",
    "ds_scm2_continuous": "Nonlinear Gaussian",
    "ds_scm3_continuous": "Nonlinear non-Gaussian",
}


def ds_label(ds: str, full: bool = False) -> str:
    # full=True prefixes the folder name ('scm1_continuous: Linear Gaussian') for
    # tables, where traceability matters. Unmapped datasets fall back to the raw name.
    desc = DATASET_LABELS.get(ds)
    if desc is None:
        return ds
    return f"{ds.removeprefix('ds_')}: {desc}" if full else desc


# Per-run artifact locations (relative to a run folder).
ATE_CSV = "eval/eval_ate_mc/files/ate_metrics_mc.csv"
DAG_JSON = "eval/eval_attention_scores/files/dag_metrics.json"
KFOLD_JSON = "kfold_summary.json"
METRICS_CSV = "k_0/logs/csv/version_0/metrics.csv"
''')

md('''
## 1. Run discovery

Index every sweep run `<model>_<jobid>/groups/<dataset>/sweeper/runs/combinations/<run>` and flag which artifacts are present. Runs with missing artifacts are skipped (with a warning) in the corresponding aggregates.
''')

code(r'''
def discover_runs(root: Path) -> pd.DataFrame:
    # Index all sweep runs under the <model>_<jobid> folders of `root`.
    rows = []
    model_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and re.search(r"_\d+$", p.name)
    )
    for model_dir in model_dirs:
        model = re.sub(r"_\d+$", "", model_dir.name)
        for run_dir in sorted(model_dir.glob("groups/*/sweeper/runs/combinations/*")):
            if not run_dir.is_dir():
                continue
            m = re.match(
                rf"{re.escape(model)}_(?P<dataset>.+)_dag_(?P<dag>\d+)_model_(?P<seed>\d+)$",
                run_dir.name,
            )
            if m is None:
                continue
            rows.append({
                "model": model,
                "dataset": m.group("dataset"),
                "dag_seed": int(m.group("dag")),
                "model_seed": int(m.group("seed")),
                "run_dir": run_dir,
                "has_ate": (run_dir / ATE_CSV).is_file(),
                "has_dag": (run_dir / DAG_JSON).is_file(),
                "has_train": (run_dir / KFOLD_JSON).is_file(),
                "has_curves": (run_dir / METRICS_CSV).is_file(),
            })
    return pd.DataFrame(rows)


runs = discover_runs(RESULTS_ROOT)
assert not runs.empty, f"No runs found under {RESULTS_ROOT} - set RESULTS_ROOT manually."

flag_cols = ["has_ate", "has_dag", "has_train", "has_curves"]
print(
    f"Discovered {len(runs)} runs: "
    f"{runs['model'].nunique()} models x {runs['dataset'].nunique()} datasets x "
    f"{int(runs.groupby(['model', 'dataset'])['model_seed'].nunique().max())} seeds"
)
for model, sub in runs.groupby("model"):
    for flag in flag_cols:
        n_missing = int((~sub[flag]).sum())
        if n_missing:
            warnings.warn(f"{model}: {n_missing}/{len(sub)} runs missing '{flag[4:]}' data")

print("artifact counts per (model, dataset):")
display(runs.groupby(["model", "dataset"])[flag_cols].sum().astype(int))
runs.head(3)
''')

md('''
## 2. Data loading

The ground-truth DAG is read from the dataset artifacts (`dec1_cross_att_mask.csv` = S->X edges, `dec1_self_att_mask.csv` = X->X edges), so the path-type categories adapt automatically to any experiment with the same layout. Each (intervention, variable) ATE pair is classified by:

- **dist_group**: `OOD` when |intervention value| > 1 (training support is [-1, 1]), else `ID`;
- **path_type**: `direct` (S is a direct parent of X), `indirect` (ancestor but not parent), `zero` (no causal path).

Two error metrics are carried through the whole notebook:

- `abs_error` = `|model_ate - true_ate|`, as produced by the evaluation (units of the normalized target);
- `scaled_error` = `abs_error / |s|`, where `s` is the intervention value. `|s|` is never 0, so this is always defined. It removes the trivial "a 3x larger intervention produces a 3x larger effect and hence a 3x larger error" scale factor, which is what makes the ID and OOD axes comparable on a single radar.

The CSV also carries `rel_error = abs_error / |true_ate|`; it is **undefined for ~60% of the pairs** (all zero-effect pairs) and explodes for near-zero true effects, so it is never used as the headline metric here.
''')

code(r'''
def load_ground_truth_masks(root: Path):
    # Return (cross, self) GT masks; rows = X targets, cols = S / X sources.
    for cross_csv in sorted(root.glob("*/groups/*/datasets/*/dec1_cross_att_mask.csv")):
        self_csv = cross_csv.with_name("dec1_self_att_mask.csv")
        if self_csv.is_file():
            return pd.read_csv(cross_csv, index_col=0), pd.read_csv(self_csv, index_col=0)
    raise FileNotFoundError("No GT mask CSVs found - check the datasets folders.")


cross_gt, self_gt = load_ground_truth_masks(RESULTS_ROOT)

# Direct children per source, and all descendants via the X->X transitive closure.
DIRECT_CHILDREN = {s: [x for x in cross_gt.index if cross_gt.loc[x, s]] for s in cross_gt.columns}
_X_CHILDREN = {x: [t for t in self_gt.index if self_gt.loc[t, x]] for x in self_gt.columns}


def _descendants(s: str) -> list:
    seen, stack = set(), list(DIRECT_CHILDREN.get(s, []))
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(_X_CHILDREN.get(x, []))
    return sorted(seen)


DESCENDANTS = {s: _descendants(s) for s in cross_gt.columns}
print("direct children:", DIRECT_CHILDREN)
print("descendants:    ", DESCENDANTS)


def assign_ate_groups(intervention: str, variable: str):
    # (dist_group, struct_group, path_type) for an (intervention, variable) pair.
    src, value_str = intervention.split("=")
    dist = "OOD" if abs(float(value_str)) > 1.0 else "ID"
    has_effect = variable in DESCENDANTS.get(src, [])
    struct = "nonzero" if has_effect else "zero"
    if not has_effect:
        path = "zero"
    elif variable in DIRECT_CHILDREN.get(src, []):
        path = "direct"
    else:
        path = "indirect"
    return dist, struct, path


def add_ate_groups(df: pd.DataFrame) -> pd.DataFrame:
    # Category columns + intervention magnitude + the scaled error.
    df = df.copy()
    groups = [assign_ate_groups(i, v) for i, v in zip(df["intervention"], df["variable"])]
    df["dist_group"] = [g[0] for g in groups]
    df["struct_group"] = [g[1] for g in groups]
    df["path_type"] = [g[2] for g in groups]
    df["intv_value"] = df["intervention"].astype(str).str.extract(r"=(-?[\d.]+)")[0].astype(float)
    df["intv_mag"] = df["intv_value"].abs()
    # |s| is a fixed sweep value (never 0), but guard anyway.
    df["scaled_error"] = df["abs_error"] / df["intv_mag"].replace(0.0, np.nan)
    df["pct_error"] = 100.0 * df["scaled_error"]   # same quantity, in % of |s|
    return df
''')

code(r'''
def load_ate(runs: pd.DataFrame) -> pd.DataFrame:
    # Long ATE dataframe: one row per (run, intervention, variable).
    frames = []
    for r in runs.itertuples():
        if not r.has_ate:
            continue
        df = pd.read_csv(Path(r.run_dir) / ATE_CSV)
        df["model"], df["dataset"], df["model_seed"] = r.model, r.dataset, r.model_seed
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return add_ate_groups(pd.concat(frames, ignore_index=True))


def load_struct(runs: pd.DataFrame) -> pd.DataFrame:
    # One row per run with the structural DAG metrics.
    rows = []
    for r in runs.itertuples():
        if not r.has_dag:
            continue
        with open(Path(r.run_dir) / DAG_JSON) as f:
            j = json.load(f)
        rows.append({
            "model": r.model, "dataset": r.dataset, "model_seed": r.model_seed,
            "shd_cross": (j.get("standard_shd_cross") or {}).get("mean"),
            "shd_self": (j.get("standard_shd_self") or {}).get("mean"),
            "soft_hamming_cross": (j.get("soft_hamming_cross") or {}).get("mean"),
            "soft_hamming_self": (j.get("soft_hamming_self") or {}).get("mean"),
            "zeroness_contrast_cross": (j.get("zeroness_cross") or {}).get("contrast"),
            "zeroness_contrast_self": (j.get("zeroness_self") or {}).get("contrast"),
            "mec_distance": (j.get("mec_distance") or {}).get("mean"),
            "mec_membership_rate": j.get("mec_membership_rate"),
        })
    return pd.DataFrame(rows)


def load_train(runs: pd.DataFrame) -> pd.DataFrame:
    # One row per run with the final training/validation/test metrics.
    keep = ["val_loss", "val_x_mae", "val_x_rmse", "val_x_r2", "val_x_r2_macro",
            "test_x_mae", "test_x_rmse", "test_x_r2", "test_x_r2_macro",
            "trainable_params", "total_training_time"]
    rows = []
    for r in runs.itertuples():
        if not r.has_train:
            continue
        with open(Path(r.run_dir) / KFOLD_JSON) as f:
            j = json.load(f)
        metrics = j.get("fold_results", {}).get("0", {}).get("metrics", {})
        rows.append({"model": r.model, "dataset": r.dataset, "model_seed": r.model_seed,
                     **{k: metrics.get(k) for k in keep}})
    return pd.DataFrame(rows)


df_ate = load_ate(runs)
df_struct = load_struct(runs)
df_train = load_train(runs)

print(f"df_ate:    {df_ate.shape}  models={sorted(df_ate['model'].unique()) if len(df_ate) else '-'}")
print(f"df_struct: {df_struct.shape}  models={sorted(df_struct['model'].unique()) if len(df_struct) else '-'}")
print(f"df_train:  {df_train.shape}  models={sorted(df_train['model'].unique()) if len(df_train) else '-'}")
display(df_ate.head(3))
display(df_struct.head(3))
df_train.head(3)
''')

md('''
## 3. Training sanity check

Prerequisite before reading any causal number: the reconstruction fit must be good. **Read R2 first** - it is the quickest fit indicator; `x_r2_macro` (mean per-node R2) is the pooling-free variant to trust. A low R2 means the ATE/SHD numbers below are not meaningful. The train/val curves show generalization (overfit = train >> val, underfit = both bad).
''')

code(r'''
def pm(mean, std, decimals: int = 3) -> str:
    # Format 'mean +/- std' tolerating NaN.
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


_fit_metrics = ["test_x_r2", "test_x_r2_macro", "test_x_mae", "test_x_rmse", "val_loss"]
train_agg = df_train.groupby(["model", "dataset"]).agg(
    n_seeds=("model_seed", "nunique"),
    **{m: (m, "mean") for m in _fit_metrics},
    **{f"{m}_std": (m, "std") for m in _fit_metrics},
).reset_index()

train_disp = train_agg[["model", "dataset", "n_seeds"]].copy()
for m in _fit_metrics:
    train_disp[m] = [pm(a, b) for a, b in zip(train_agg[m], train_agg[f"{m}_std"])]
display(train_disp)

low = train_agg[train_agg["test_x_r2_macro"] < 0.9]
if len(low):
    warnings.warn(
        "Low test R2-macro (<0.9): "
        + ", ".join(f"{m}/{d}" for m, d in low[["model", "dataset"]].values)
        + " - inspect the fit before trusting the ATE numbers."
    )
''')

code(r'''
CURVE_COLS = {"epoch", "train_x_r2_macro", "val_x_r2_macro", "train_loss_x", "val_loss_x"}


def load_curves(runs: pd.DataFrame) -> pd.DataFrame:
    # Per-epoch train/val curves for every run.
    frames = []
    for r in runs.itertuples():
        if not r.has_curves:
            continue
        df = pd.read_csv(Path(r.run_dir) / METRICS_CSV, usecols=lambda c: c in CURVE_COLS)
        # train and val metrics are logged in separate rows of the same epoch
        df = df.groupby("epoch", as_index=False).first()
        df["model"], df["dataset"], df["model_seed"] = r.model, r.dataset, r.model_seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


df_curves = load_curves(runs)

_datasets = sorted(df_curves["dataset"].unique())
fig, axes = plt.subplots(2, len(_datasets), figsize=(FIG_W, 0.62 * FIG_W), sharex=True)
axes = np.atleast_2d(axes)
for col, ds in enumerate(_datasets):
    sub = df_curves[df_curves["dataset"] == ds]
    for row, (val_col, train_col, logy) in enumerate([
        ("val_x_r2_macro", "train_x_r2_macro", False),
        ("val_loss_x", "train_loss_x", True),
    ]):
        ax = axes[row, col]
        for model, msub in sub.groupby("model"):
            curve = msub.groupby("epoch")[[val_col, train_col]].agg(["mean", "std"])
            for colname, ls, tag in [(val_col, "-", "val"), (train_col, "--", "train")]:
                mu = curve[(colname, "mean")]
                sd = curve[(colname, "std")].fillna(0.0)
                (line,) = ax.plot(mu.index, mu.values, ls, lw=1.0,
                                  label=f"{ROLE_MAP.get(model, model)} ({tag})")
                if ls == "-":
                    ax.fill_between(mu.index, (mu - sd).values, (mu + sd).values,
                                    alpha=0.15, color=line.get_color())
        if logy:
            ax.set_yscale("log")
        if row == 0:
            ax.set_title(ds_label(ds))
        else:
            ax.set_xlabel("epoch")
axes[0, 0].set_ylabel("R2 macro (mean +/- std)")
axes[1, 0].set_ylabel("loss (log)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)),
           bbox_to_anchor=(0.5, 1.05), frameon=False)
plt.tight_layout()
save_fig(fig, "fig_training_curves")
plt.show()
''')

md('''
## 4. Structural metrics (attention DAG)

From `dag_metrics.json`: standard SHD (missing + extra + reversed edges vs the true DAG), soft Hamming (probabilistic distance), zeroness contrast (edge vs non-edge attention separation) and MEC distance. Aggregated as mean +/- std over model seeds.

Note: an arm evaluated with GT hard masks (`cheater`) does not perform discovery; its numbers reflect thresholding of the masked attention.
''')

code(r'''
_struct_metrics = ["shd_cross", "shd_self", "soft_hamming_cross", "soft_hamming_self",
                   "zeroness_contrast_cross", "zeroness_contrast_self", "mec_distance"]
struct_agg = df_struct.groupby(["model", "dataset"])[_struct_metrics].agg(["mean", "std"])

struct_disp = pd.DataFrame(index=struct_agg.index)
for m in _struct_metrics:
    struct_disp[m] = [pm(mu, sd, 2)
                      for mu, sd in zip(struct_agg[(m, "mean")], struct_agg[(m, "std")])]
display(struct_disp.reset_index())

_long = df_struct.melt(id_vars=["model", "dataset", "model_seed"],
                       value_vars=["shd_cross", "shd_self"],
                       var_name="block", value_name="SHD")
_long["arm"] = _long["model"].map(lambda m: ROLE_MAP.get(m, m))
_long["dataset_label"] = _long["dataset"].map(ds_label)
_n_ds = _long["dataset"].nunique()
g = sns.catplot(_long, x="arm", y="SHD", hue="block", col="dataset_label", kind="bar",
                col_order=[ds_label(d) for d in sorted(_long["dataset"].unique())],
                palette="colorblind", height=FIG_W / _n_ds, aspect=1.0)
g.set_axis_labels("", "SHD (mean over seeds)")
g.set_titles("{col_name}")
g.figure.set_size_inches(FIG_W, FIG_W / _n_ds)
save_fig(g.figure, "fig_structural_shd")
plt.show()
''')

md('''
## 5. ATE aggregation over seeds and categories

Per (intervention, variable) pair, per-seed errors are averaged (mean/std over seeds). Category means then combine the pairs of each `dist_group x path_type` cell via the **law of total variance**: `mu = mean(mu_i)`, `var = mean(sigma_i^2) + var(mu_i)`.

Three metrics are aggregated: `abs` (raw absolute error), `scaled` (`abs_error / |s|`) and `pct` (the same as a percentage, `100 * abs_error / |s|`, used by the spiderweb). They differ in how ID and OOD compare: with `abs`, OOD looks ~3x worse simply because `|s|` is 3x larger there.
''')

code(r'''
df_ate_s = (
    df_ate.groupby(["model", "dataset", "intervention", "variable", "intv_value", "intv_mag",
                    "dist_group", "struct_group", "path_type"], as_index=False)
    .agg(abs_error_mean=("abs_error", "mean"), abs_error_std=("abs_error", "std"),
         scaled_error_mean=("scaled_error", "mean"), scaled_error_std=("scaled_error", "std"),
         pct_error_mean=("pct_error", "mean"), pct_error_std=("pct_error", "std"),
         rel_error_mean=("rel_error", "mean"), rel_error_std=("rel_error", "std"),
         n_seeds=("abs_error", "size"))
)

# Ordered by edge type (zero -> indirect -> direct), ID then OOD within each type.
# The radar walks this list clockwise starting at the top.
PATH_ORDER = ["zero", "indirect", "direct"]
CATEGORIES = [f"{d}_{p}" for p in PATH_ORDER for d in ["ID", "OOD"]]
METRICS = ["abs", "scaled", "pct"]


def _total_variance(mu: np.ndarray, sd: np.ndarray) -> float:
    # within-pair variance + between-pair variance
    n = len(mu)
    return float((sd ** 2).mean() + (mu.var(ddof=1) if n > 1 else 0.0))


def aggregate_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Mean ATE error per (model, dataset, dist_group x path_type) category.
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        for dist in ["ID", "OOD"]:
            for path in ["direct", "indirect", "zero"]:
                cell = sub[(sub["dist_group"] == dist) & (sub["path_type"] == path)]
                if cell.empty:
                    continue
                row = {"model": model, "dataset": dataset,
                       "dist_group": dist, "path_type": path,
                       "category": f"{dist}_{path}", "n_pairs": len(cell)}
                for met in METRICS:
                    mu = cell[f"{met}_error_mean"].to_numpy(float)
                    sd = cell[f"{met}_error_std"].fillna(0).to_numpy(float)
                    row[f"{met}_mean"] = float(mu.mean())
                    row[f"{met}_std"] = float(np.sqrt(_total_variance(mu, sd)))
                rows.append(row)
    return pd.DataFrame(rows)


df_cat = aggregate_categories(df_ate_s)

_cats = [c for c in CATEGORIES if c in set(df_cat["category"])]
for met, title in [("abs", "abs ATE error"), ("pct", "abs ATE error in % of |s|")]:
    print(f"===== mean {title} per category =====")
    for ds in sorted(df_cat["dataset"].unique()):
        print(f"--- {ds_label(ds)} ---")
        display(
            df_cat[df_cat["dataset"] == ds]
            .assign(arm=lambda d: d["model"].map(lambda m: ROLE_MAP.get(m, m)))
            .pivot_table(index="arm", columns="category", values=f"{met}_mean")
            .reindex(columns=_cats)
            .round(4)
        )
''')

md(r'''
## 6. Summary tables

**6.1** is the compact main-text table: one row per model arm, three columns per dataset. **6.2** is the full per-cell breakdown, kept for the appendix.

Per dataset:

- **MAE** - `test_x_mae` on the test split, mean over model seeds. Lower = better. This is the fit-quality prerequisite; it is *not* comparable across datasets (different normalizations).
- **ATE% causal** - mean ATE error in % of `|s|`, macro-averaged over the four `direct`/`indirect` x `ID`/`OOD` categories. This is the estimation-accuracy number.
- **ATE% zero** - the same, over the two `zero` categories: a spurious effect attributed to a non-descendant, i.e. **false-effect leakage**.

The causal and zero columns are kept **separate on purpose**. Arms that apply a structural mask return *exactly* 0 on the zero categories by construction, so folding those two cells into a single headline average would credit a definitional zero as estimation accuracy and flatter the masked arms by ~1/3 of the score. Read the zero column as a structural property, the causal column as accuracy.

Macro-averaging (equal weight per category) rather than pooling over pairs keeps the numbers consistent with the spiderweb and prevents the 42 zero-effect pairs from outvoting the 21 direct ones. Uncertainties combine via the same law of total variance used throughout.
''')

code(r'''
# Which path types feed each ATE column. Config, so the split is easy to change.
CATEGORY_GROUPS = {"causal": ["direct", "indirect"], "zero": ["zero"]}


def model_order(models, role_map: dict) -> list:
    # ROLE_MAP key order first (the intended reading order), then any extras.
    known = [m for m in role_map if m in set(models)]
    return known + sorted(set(models) - set(known))


def macro_average_categories(df_cat: pd.DataFrame, metric: str = "pct",
                             groups: dict = CATEGORY_GROUPS) -> pd.DataFrame:
    # Macro-average the per-category means into one value per (model, dataset, group).
    rows = []
    for (model, dataset), sub in df_cat.groupby(["model", "dataset"]):
        for gname, paths in groups.items():
            cell = sub[sub["path_type"].isin(paths)]
            if cell.empty:
                continue
            mu = cell[f"{metric}_mean"].to_numpy(float)
            sd = cell[f"{metric}_std"].fillna(0).to_numpy(float)
            rows.append({"model": model, "dataset": dataset, "group": gname,
                         "n_cat": len(mu), "mean": float(mu.mean()),
                         "std": float(np.sqrt(_total_variance(mu, sd)))})
    return pd.DataFrame(rows)


def build_compact_table(df_cat: pd.DataFrame, df_train: pd.DataFrame, role_map: dict,
                        metric: str = "pct", compact: bool = True,
                        mae_decimals: int = 3, ate_decimals: int = 1):
    # Rows = model arms; per dataset: test MAE, ATE% causal, ATE% zero (all lower=better).
    # compact=True prints means only (fits the A4 text width); False adds +/- std.
    macro = macro_average_categories(df_cat, metric)
    mae = (df_train.groupby(["model", "dataset"])["test_x_mae"]
           .agg(["mean", "std"]).reset_index())

    datasets = sorted(df_cat["dataset"].unique())
    models = model_order(df_cat["model"].unique(), role_map)
    # (column name, path-type group or None for MAE, decimals)
    columns = ([("MAE", None, mae_decimals)]
               + [(f"ATE% {g}", g, ate_decimals) for g in CATEGORY_GROUPS])

    # cell[(dataset, column)] -> {"mean": Series over models, "std": Series}
    cell = {}
    for ds in datasets:
        for cname, group, _ in columns:
            src = (mae[mae["dataset"] == ds] if group is None
                   else macro[(macro["dataset"] == ds) & (macro["group"] == group)])
            src = src.set_index("model")
            cell[(ds, cname)] = {"mean": src["mean"].reindex(models),
                                 "std": src["std"].reindex(models)}

    def fmt(mu, sd, dec):
        if pd.isna(mu):
            return "--"
        if compact or pd.isna(sd):
            return f"{mu:.{dec}f}"
        return f"{mu:.{dec}f} +/- {sd:.{dec}f}"

    # Best (= lowest) arm per column, for bolding. All three columns are lower-better.
    best = {k: (v["mean"].idxmin() if v["mean"].notna().any() else None)
            for k, v in cell.items()}

    raw = pd.concat({k: v["mean"] for k, v in cell.items()}, axis=1)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)

    disp = pd.DataFrame(index=pd.Index([role_map.get(m, m) for m in models], name="Model"))
    for ds in datasets:
        for cname, _, dec in columns:
            c = cell[(ds, cname)]
            disp[(ds_label(ds), cname)] = [
                (lambda s, m=m: f"**{s}**" if m == best[(ds, cname)] else s)(
                    fmt(c["mean"][m], c["std"][m], dec)) for m in models]
    disp.columns = pd.MultiIndex.from_tuples(disp.columns)

    # LaTeX (booktabs): one multicolumn block per dataset.
    n_c = len(columns)
    pct, dn = r"\%", r"$\downarrow$"
    lines = ["\\begin{tabular}{l" + (" " + "c" * n_c) * len(datasets) + "}", "\\toprule"]
    lines.append(" & ".join([""] + [f"\\multicolumn{{{n_c}}}{{c}}{{{ds_label(ds)}}}"
                                    for ds in datasets]) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + i * n_c}-{1 + (i + 1) * n_c}}}"
                          for i in range(len(datasets))))
    lines.append(" & ".join(["Model"] + [f"{c.replace('%', pct)} {dn}"
                                         for _ in datasets for c, _, _ in columns]) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = [role_map.get(m, m)]
        for ds in datasets:
            for cname, _, dec in columns:
                c = cell[(ds, cname)]
                s = fmt(c["mean"][m], c["std"][m], dec)
                if s != "--":
                    s = "$" + s.replace("+/-", r"\pm") + "$"
                cells.append(f"\\textbf{{{s}}}" if m == best[(ds, cname)] else s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return raw, disp, "\n".join(lines)


compact_raw, compact_disp, compact_latex = build_compact_table(df_cat, df_train, ROLE_MAP)
print("MAE = test_x_mae (lower better) | ATE% = mean abs ATE error in % of |s| (lower better)")
display(compact_disp)
print(compact_latex)

# Same table with uncertainties, for the record.
_, compact_disp_std, _ = build_compact_table(df_cat, df_train, ROLE_MAP, compact=False)
display(compact_disp_std)
''')

md(r'''
### 6.2 Appendix: full breakdown by (dataset, path type, |s|)

Abs. ATE error per (dataset, path_type, |s|) cell, formatted `mean +/- std`, with **Delta% vs the baseline arm** = `100 * (mu_base - mu_method) / mu_base` (positive = method better than the baseline; first-order error propagation). `|s|` is the intervention magnitude annotated `(ID)`/`(OOD)`. Because every row fixes `|s|`, the raw `abs` metric is the right choice here (dividing by a constant within a row would change nothing but the units). The booktabs LaTeX is printed below the table.

This is the detailed view: too many rows for the main text, but it is where the per-magnitude behaviour is visible.
''')

code(r'''
def aggregate_error(df: pd.DataFrame, group_cols=("dataset", "path_type", "intv_mag"),
                    metric: str = "abs") -> pd.DataFrame:
    # Aggregate per-pair (mean, std) into per-cell (mean, std): law of total variance.
    rows = []
    for keys, sub in df.groupby(list(group_cols)):
        mu = sub[f"{metric}_error_mean"].astype(float).to_numpy()
        sd = sub[f"{metric}_error_std"].fillna(0.0).astype(float).to_numpy()
        if len(mu) == 0:
            continue
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"n_pairs": len(mu), "mean": float(mu.mean()),
                    "std": float(np.sqrt(_total_variance(mu, sd)))})
        rows.append(row)
    return pd.DataFrame(rows)


def compute_delta_pct(base: pd.DataFrame, method: pd.DataFrame,
                      group_cols=("dataset", "path_type", "intv_mag")) -> pd.DataFrame:
    # Delta% = 100 * (mu_base - mu_method) / mu_base with error propagation.
    m = base.merge(method, on=list(group_cols), suffixes=("_base", "_meth"))
    denom = m["mean_base"].abs().replace(0.0, np.nan)
    m["delta_pct"] = 100.0 * (m["mean_base"] - m["mean_meth"]) / denom
    m["delta_pct_std"] = 100.0 * np.sqrt(m["std_base"] ** 2 + m["std_meth"] ** 2) / denom
    return m[list(group_cols) + ["delta_pct", "delta_pct_std"]]


def _fmt_pm(mean, std, decimals: int = 2) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _fmt_pm_latex(mean, std, decimals: int = 2, bold: bool = False) -> str:
    if pd.isna(mean):
        return "--"
    s = f"{mean:.{decimals}f}" if pd.isna(std) else f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    s = f"${s}$"
    return f"\\textbf{{{s}}}" if bold else s


def _fmt_delta(mean, std, decimals: int = 1, bold: bool = False, latex: bool = False) -> str:
    if pd.isna(mean):
        return "--"
    sign = "+" if mean >= 0 else ""
    body = f"{sign}{mean:.{decimals}f} +/- {std:.{decimals}f}" if not pd.isna(std) else f"{sign}{mean:.{decimals}f}"
    if latex:
        body = "$" + body.replace("+/-", "\\pm") + "$"
        return f"\\textbf{{{body}}}" if bold else body
    return f"**{body}**" if bold else body
''')

code(r'''
def build_summary_table(df_ate_s: pd.DataFrame, role_map: dict,
                        baseline_model: str = BASELINE_MODEL, metric: str = "abs",
                        path_type_order=("direct", "indirect", "zero"),
                        decimals: int = 2, mag_decimals: int = 1):
    # ATE error per (dataset, path_type, |s|) for every arm, vs the baseline arm.
    # Returns (raw_df, display_df, latex_str).  Column labels come from role_map.
    group_cols = ["dataset", "path_type", "intv_mag"]
    df = df_ate_s.copy()

    models = sorted(df["model"].unique(), key=lambda m: role_map.get(m, m))
    if baseline_model not in models:
        warnings.warn(f"Baseline '{baseline_model}' absent; using '{models[0]}'.")
        baseline_model = models[0]
    label_of = {m: role_map.get(m, m) for m in models}
    base_lab = label_of[baseline_model]
    methods = [m for m in models if m != baseline_model]
    method_labels = [label_of[m] for m in methods]

    plain = {m: aggregate_error(df[df["model"] == m], group_cols, metric) for m in models}
    raw = plain[baseline_model][group_cols + ["n_pairs", "mean", "std"]].rename(
        columns={"mean": f"{base_lab}_mean", "std": f"{base_lab}_std"})
    for m in methods:
        lab = label_of[m]
        raw = raw.merge(
            plain[m][group_cols + ["mean", "std"]].rename(
                columns={"mean": f"{lab}_mean", "std": f"{lab}_std"}),
            on=group_cols, how="left")
        d = compute_delta_pct(plain[baseline_model], plain[m], group_cols)
        raw = raw.merge(d.rename(columns={"delta_pct": f"{lab}_delta",
                                          "delta_pct_std": f"{lab}_delta_std"}),
                        on=group_cols, how="left")

    # |s| label with (ID)/(OOD) tag (mode of dist_group inside the cell)
    dist_lookup = (df.groupby(group_cols)["dist_group"]
                   .agg(lambda s: s.mode().iloc[0]).reset_index())
    raw = raw.merge(dist_lookup, on=group_cols, how="left")
    raw["mag_label"] = [f"{v:.{mag_decimals}f} ({d})" for v, d in zip(raw["intv_mag"], raw["dist_group"])]

    # Row order: dataset x path_type x |s|
    raw["dataset"] = pd.Categorical(raw["dataset"], sorted(raw["dataset"].unique()), ordered=True)
    raw["path_type"] = pd.Categorical(raw["path_type"], list(path_type_order), ordered=True)
    raw = raw.sort_values(group_cols).reset_index(drop=True)

    best = (raw[[f"{lab}_mean" for lab in method_labels]].idxmin(axis=1)
            .str.replace("_mean", "", regex=False) if len(method_labels) >= 2 else None)

    disp = pd.DataFrame()
    disp["Dataset"] = [ds_label(d, full=True) for d in raw["dataset"].astype(str)]
    disp["Path type"] = raw["path_type"].astype(str)
    disp["|s|"] = raw["mag_label"]
    disp["n"] = raw["n_pairs"].astype(int)
    disp[base_lab] = [_fmt_pm(a, b, decimals) for a, b in zip(raw[f"{base_lab}_mean"], raw[f"{base_lab}_std"])]
    for lab in method_labels:
        disp[lab] = [_fmt_pm(a, b, decimals) for a, b in zip(raw[f"{lab}_mean"], raw[f"{lab}_std"])]
    for lab in method_labels:
        disp[f"Delta% {lab}"] = [_fmt_delta(a, b) for a, b in zip(raw[f"{lab}_delta"], raw[f"{lab}_delta_std"])]
    if best is not None:
        for i, bm in enumerate(best):
            if isinstance(bm, str) and bm in disp.columns:
                disp.at[i, bm] = f"**{disp.at[i, bm]}**"

    # LaTeX (booktabs)
    n_index = 4
    n_meth = 1 + len(method_labels)
    col_spec = "lll r " + "c" * n_meth + (" " + "c" * len(method_labels) if method_labels else "")
    lines = ["\\begin{tabular}{" + col_spec + "}", "\\toprule"]
    head1 = [""] * n_index + [f"\\multicolumn{{{n_meth}}}{{c}}{{Abs. ATE error $\\downarrow$}}"]
    if method_labels:
        head1.append(f"\\multicolumn{{{len(method_labels)}}}{{c}}{{Delta\\% vs {base_lab} $\\uparrow$}}")
    lines.append(" & ".join(head1) + " \\\\")
    end = n_index + n_meth
    cmids = [f"\\cmidrule(lr){{{n_index + 1}-{end}}}"]
    if method_labels:
        cmids.append(f"\\cmidrule(lr){{{end + 1}-{end + len(method_labels)}}}")
    lines.append(" ".join(cmids))
    head2 = (["Dataset", "Path type", "$|s|$", "$n$", base_lab] + method_labels
             + [f"Delta\\% {lab}" for lab in method_labels])
    lines.append(" & ".join(head2) + " \\\\")
    lines.append("\\midrule")

    last_ds, last_pt = None, None
    for i, row in raw.iterrows():
        ds, pt = str(row["dataset"]), str(row["path_type"])
        if ds != last_ds and last_ds is not None:
            lines.append("\\midrule")
        if ds != last_ds:
            last_pt = None
        bm = best.iloc[i] if best is not None else None
        ds_tex = ds_label(ds, full=True).replace("_", r"\_")
        cells = [ds_tex if ds != last_ds else "", pt if pt != last_pt else "",
                 str(row["mag_label"]), f"{int(row['n_pairs'])}"]
        cells.append(_fmt_pm_latex(row[f"{base_lab}_mean"], row[f"{base_lab}_std"], decimals))
        for lab in method_labels:
            cells.append(_fmt_pm_latex(row[f"{lab}_mean"], row[f"{lab}_std"], decimals, bold=(lab == bm)))
        for lab in method_labels:
            cells.append(_fmt_delta(row[f"{lab}_delta"], row[f"{lab}_delta_std"],
                                    bold=(lab == bm), latex=True))
        lines.append(" & ".join(cells) + " \\\\")
        last_ds, last_pt = ds, pt
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return raw, disp, "\n".join(lines)


raw_df, display_df, latex_str = build_summary_table(df_ate_s, ROLE_MAP, metric="abs")
display(display_df)
print(latex_str)
''')

md('''
## 7. Spiderweb figure

One radar panel per dataset, six axes = the categories `ID/OOD x direct/indirect/zero`. The radius is the seed- and pair-aggregated ATE error, so **a smaller polygon is a better model**.

**Radius = ATE error as a percentage of the intervention size** (`metric="pct"`, i.e. `100 * abs_error / |s|`). This is a real, interpretable unit - "the estimated effect is off by 12% of the intervention" - so the panels need no per-axis normalization and the numbers mean the same thing on the ID and OOD axes, whose `|s|` differ by 3x. `metric="abs"` (raw error) and `metric="scaled"` (fraction of `|s|`) remain available. `rel_error` is deliberately not offered: it is undefined on the three `zero` axes.

The radius stays **logarithmic** (`rscale="log"`, spanning `log_decades` decades below the outer ring), which is what keeps a 10x gap visible instead of collapsing it onto the centre.

**Radial axis**: a single black spine on the right-hand side, between two category spokes. Only the outer value is labelled numerically - the intermediate decades and the 2..9 minor steps are drawn as tick marks on that spine and echoed in the web as light-gray circles, so the scale stays readable without a column of competing numbers. Ticks are logarithmic, hence unevenly spaced by design.

**All panels share one outer ring** (`shared_scale=True`): the ring is the rounded maximum over *all* datasets, so polygon sizes can be compared across panels at a glance and the single label applies everywhere. This is a presentation choice, not a licence to compare datasets numerically - the normalizations still differ (see the note at the top). Set `shared_scale=False` for per-panel autoscaling, which resolves more detail in the lower-error panels at the cost of three different rings.

**The centre is a censoring bound, not zero.** `log(0)` is undefined, and on this data several cells are *exactly* 0: the `zero` categories of the masked arms, whose mask removes the non-ancestor path entirely, so the estimated effect is 0 by construction. Everything at or below the floor (`outer / 10^log_decades`) is clipped onto the centre rather than dropped, so a polygon touching the centre means "at or below the floor, possibly exactly 0".

`log_decades=3` is the default because 2 decades was **hiding real data**: it put the floor at ~1.4% and swallowed the baseline `ID_zero` cells (0.8% on scm2, 1.3% on scm3), which are precisely the cells that separate the arms. Any nonzero value that lands below the floor now raises a warning, so this can never happen silently - widen `log_decades` if it fires.

**Reading the `zero` axes**: the masked arms sit pinned at the centre on both of them, which is a structural fact (a hard zero from the mask), not an estimation win in the same sense as the other four axes. The interesting comparison lives on `direct` and `indirect`; check the section 5 tables to see which centre-touching cells are exact zeros.

Axes run **clockwise from the top**, grouped by edge type: `zero (ID, OOD) -> indirect (ID, OOD) -> direct (ID, OOD)`, so the ID/OOD pair of each edge type sits side by side (`CATEGORIES` / `PATH_ORDER` control this).

The headline figure plus per-axis-normalized and raw-error variants are saved to `./img/`.
''')

code(r'''
GRID_MAJOR, GRID_MINOR = "0.80", "0.91"   # light gray log circles
UNIT = {"pct": "% of |s|", "scaled": "fraction of |s|", "abs": "abs. error"}
AXIS_UNIT = {"pct": "%", "scaled": "", "abs": ""}   # terse label on the radial spine


def _nice_ceil(v: float) -> float:
    # Round up to the next 1/1.5/2/3/5/7 x 10^k so the single axis label stays
    # clean without wasting most of the outer decade (140 -> 150, not 200).
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = np.floor(np.log10(v))
    m = v / 10.0 ** e
    for c in (1.0, 1.5, 2.0, 3.0, 5.0, 7.0):
        if m <= c * (1 + 1e-9):
            return c * 10.0 ** e
    return 10.0 ** (e + 1)


def _radial_scale(vmax: float, decades: int, rscale: str):
    # Return (top, r(v), major ticks, minor ticks) mapping values onto [0, 1].
    top = _nice_ceil(vmax)
    if rscale != "log":
        return top, (lambda v: np.asarray(v, float) / top), [top * f for f in (0.25, 0.5, 0.75, 1.0)], []
    floor = top * 10.0 ** (-decades)
    lo = np.log10(floor)

    def r(v):
        # clip pulls everything at/below the floor (incl. exact 0) onto the centre
        return (np.log10(np.clip(np.asarray(v, float), floor, top)) - lo) / decades

    majors, minors = [], []
    for k in range(int(np.floor(lo + 1e-9)), int(np.ceil(np.log10(top) - 1e-9)) + 1):
        for m in range(1, 10):
            v = m * 10.0 ** k
            if floor * (1 - 1e-9) <= v <= top * (1 + 1e-9):
                (majors if m == 1 else minors).append(v)
    return top, r, majors, minors


def _draw_radial_axis(ax, rfun, majors, minors, top: float, unit: str):
    # Black radial spine on the right-hand side, log ticks, single numeric label.
    th = np.pi / 2  # with theta_offset=pi/2 and clockwise direction this is screen-right
    ax.plot([th, th], [0, 1], color="black", lw=0.8, zorder=6)
    for vals, length, lw in [(majors, 0.055, 0.8), (minors, 0.030, 0.6)]:
        for v in vals:
            rr = float(rfun(v))
            dth = length / max(rr, 0.12)          # keep the tick length ~constant
            ax.plot(np.linspace(th, th + dth, 6), np.full(6, rr),
                    color="black", lw=lw, zorder=6)
    # Offset in points from the outer tip of the spine: clear of the ticks and of
    # the neighbouring category label.
    ax.annotate(f"{top:g}{unit}", xy=(th, 1.0), xytext=(7, 9),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=6.5, color="black", annotation_clip=False)


def plot_spiderweb(df_cat: pd.DataFrame, metric: str = "pct", normalize_axes: bool = False,
                   rscale: str = "log", log_decades: int = 3, shared_scale: bool = True,
                   save_stem: str | None = None):
    # One radar panel per dataset; radius = ATE error (lower = better).
    cats = [c for c in CATEGORIES if c in set(df_cat["category"])]
    datasets = sorted(df_cat["dataset"].unique())
    models = sorted(df_cat["model"].unique(), key=lambda m: ROLE_MAP.get(m, m))
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    # One outer ring for every panel: polygon sizes become comparable at a glance
    # (still only within a dataset for interpretation - see the note above).
    shared_max = None
    if shared_scale and not normalize_axes:
        _all = df_cat[f"{metric}_mean"].to_numpy(float)
        shared_max = float(np.nanmax(_all)) if np.isfinite(_all).any() else 1.0

    fig, axes = plt.subplots(1, len(datasets), figsize=(FIG_W, FIG_W / len(datasets) + 0.7),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        piv = (df_cat[df_cat["dataset"] == ds]
               .pivot_table(index="model", columns="category", values=f"{metric}_mean")
               .reindex(index=models, columns=cats))
        if normalize_axes:
            mx = piv.max(axis=0)
            for c in piv.columns:
                # all-zero axis -> keep zeros instead of 0/0; NaN cells stay NaN
                piv[c] = piv[c] / mx[c] if (pd.notna(mx[c]) and mx[c] > 0) else piv[c] * 0.0

        vals_all = piv.to_numpy(float)
        vmax = float(np.nanmax(vals_all)) if np.isfinite(vals_all).any() else 1.0
        top, rfun, majors, minors = _radial_scale(
            vmax if shared_max is None else shared_max, log_decades, rscale)
        if rscale == "log":
            # Loud about censoring: a NONZERO value pushed onto the centre is
            # silent data loss, unlike an exact 0 which has nowhere else to go.
            floor = top * 10.0 ** (-log_decades)
            hidden = int(((vals_all > 0) & (vals_all < floor)).sum())
            if hidden:
                warnings.warn(f"{ds}: {hidden} nonzero cell(s) below the {floor:.3g} floor "
                              f"are clipped onto the centre - raise log_decades.")
        with np.errstate(divide="ignore", invalid="ignore"):
            piv = pd.DataFrame(rfun(vals_all), index=piv.index, columns=piv.columns)

        # Log ticks echoed in the web as light-gray circles (drawn under the data).
        circle = np.linspace(0, 2 * np.pi, 181)
        for vals, color, lw in [(minors, GRID_MINOR, 0.4), (majors, GRID_MAJOR, 0.6)]:
            for v in vals:
                ax.plot(circle, np.full_like(circle, float(rfun(v))),
                        color=color, lw=lw, zorder=0)

        for model in models:
            if model not in piv.index:
                continue
            vals = piv.loc[model]
            if vals.isna().all():
                continue
            if vals.isna().any():
                warnings.warn(f"{ds}/{model}: no data for {list(vals[vals.isna()].index)}"
                              " - arm skipped in this panel")
                continue
            v = vals.tolist() + [vals.iloc[0]]
            (line,) = ax.plot(angles, v, marker="o", ms=2.5, lw=1.0,
                              label=ROLE_MAP.get(model, model))
            ax.fill(angles, v, alpha=0.08, color=line.get_color())
        # Clockwise, first category at the top.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=6)
        ax.set_rlim(0, 1)
        ax.set_yticks([])                       # no numbers inside the web
        ax.xaxis.grid(True, color=GRID_MAJOR, lw=0.5)
        ax.spines["polar"].set_color(GRID_MAJOR)
        ax.spines["polar"].set_linewidth(0.6)
        _draw_radial_axis(ax, rfun, majors, minors, top,
                          "" if normalize_axes else AXIS_UNIT.get(metric, ""))
        ax.set_title(ds_label(ds), pad=12)
    unit = "normalized per axis" if normalize_axes else UNIT.get(metric, metric)
    fig.suptitle(f"ATE error ({unit}) by category $\\downarrow$"
                 + (f" - log radius, {log_decades} decades" if rscale == "log" else ""),
                 fontsize=9)
    plt.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=len(labels), frameon=False)

    if save_stem is None:
        save_stem = (f"fig_ate_spiderweb_{metric}"
                     + ("_norm" if normalize_axes else "_raw") + f"_{rscale}")
    save_fig(fig, save_stem)
    plt.show()


# Headline figure: error as % of |s|, real units, log radius.
plot_spiderweb(df_cat, metric="pct", normalize_axes=False, rscale="log",
               save_stem="fig_ate_spiderweb")
# Variants for reference.
plot_spiderweb(df_cat, metric="pct", normalize_axes=True, rscale="log")
plot_spiderweb(df_cat, metric="abs", normalize_axes=False, rscale="log")
''')

md('''
## Notes

- **Reuse**: copy this notebook into any sweep results folder with the same `<model>_<jobid>/groups/<dataset>/sweeper/runs/combinations/<run>` layout and re-run; discovery, GT-DAG extraction and the ATE categories are all derived from the folder contents. Only `ROLE_MAP` / `BASELINE_MODEL` are naming choices.
- **Figures** go to `./img/` as `.png` + `.pdf`, sized to the A4 short side (`FIG_W`).
- **Missing artifacts** (e.g. a run whose post-training evaluations failed) are flagged in section 1 and skipped in the aggregates, so partial result sets still work.
- Compare models **within** a dataset only: datasets have different normalizations, so errors are not comparable across datasets.
''')

nb = nbf.v4.new_notebook(cells=CELLS, metadata={
    "kernelspec": {"display_name": "venv (3.13.12.final.0)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.12"},
})
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(CELLS)} cells)")

nb2 = nbf.read(OUT, as_version=4)
nbf.validate(nb2)
print("validation OK")
