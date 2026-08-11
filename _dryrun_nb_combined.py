def display(x):
    print(x.to_string() if hasattr(x, 'to_string') else x)


# ===== cell 1 =====
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# The notebook lives next to the <model>_<jobid> folders.
RESULTS_ROOT = Path.cwd()

# Display roles for the final table / spiderweb legend.
ROLE_MAP = {"vanilla": "Baseline", "svfa": "Ours", "cheater": "Oracle"}

# Per-run artifact locations (relative to a run folder).
ATE_CSV = "eval/eval_ate_mc/files/ate_metrics_mc.csv"
DAG_JSON = "eval/eval_attention_scores/files/dag_metrics.json"
KFOLD_JSON = "kfold_summary.json"
METRICS_CSV = "k_0/logs/csv/version_0/metrics.csv"

# ===== cell 3 =====
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

# ===== cell 5 =====
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
    df = df.copy()
    groups = [assign_ate_groups(i, v) for i, v in zip(df["intervention"], df["variable"])]
    df["dist_group"] = [g[0] for g in groups]
    df["struct_group"] = [g[1] for g in groups]
    df["path_type"] = [g[2] for g in groups]
    return df

# ===== cell 6 =====
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

# ===== cell 8 =====
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

# ===== cell 9 =====
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
fig, axes = plt.subplots(2, len(_datasets), figsize=(5.2 * len(_datasets), 7), sharex=True)
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
                (line,) = ax.plot(mu.index, mu.values, ls, label=f"{model} ({tag})")
                if ls == "-":
                    ax.fill_between(mu.index, (mu - sd).values, (mu + sd).values,
                                    alpha=0.15, color=line.get_color())
        if logy:
            ax.set_yscale("log")
        if row == 0:
            ax.set_title(ds)
        ax.set_xlabel("epoch")
axes[0, 0].set_ylabel("R2 macro (mean +/- std over seeds)")
axes[1, 0].set_ylabel("loss (log)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           ncol=max(1, len(labels)), bbox_to_anchor=(0.5, 1.06), fontsize=8)
plt.tight_layout()
plt.show()

# ===== cell 11 =====
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
g = sns.catplot(_long, x="model", y="SHD", hue="block", col="dataset",
                kind="bar", height=4, aspect=0.9)
g.set_axis_labels("", "SHD (mean over seeds)")
plt.show()

# ===== cell 13 =====
df_ate_s = (
    df_ate.groupby(["model", "dataset", "intervention", "variable",
                    "dist_group", "struct_group", "path_type"], as_index=False)
    .agg(abs_error_mean=("abs_error", "mean"), abs_error_std=("abs_error", "std"),
         rel_error_mean=("rel_error", "mean"), rel_error_std=("rel_error", "std"),
         n_seeds=("abs_error", "size"))
)

CATEGORIES = [f"{d}_{p}" for d in ["ID", "OOD"] for p in ["direct", "indirect", "zero"]]


def aggregate_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Mean abs ATE error per (model, dataset, dist_group x path_type) category.
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        for dist in ["ID", "OOD"]:
            for path in ["direct", "indirect", "zero"]:
                cell = sub[(sub["dist_group"] == dist) & (sub["path_type"] == path)]
                if cell.empty:
                    continue
                mu = cell["abs_error_mean"].to_numpy(float)
                sd = cell["abs_error_std"].fillna(0).to_numpy(float)
                n = len(mu)
                var = (sd ** 2).mean() + (mu.var(ddof=1) if n > 1 else 0.0)
                rows.append({"model": model, "dataset": dataset,
                             "dist_group": dist, "path_type": path,
                             "category": f"{dist}_{path}", "n_pairs": n,
                             "mean": float(mu.mean()), "std": float(np.sqrt(var))})
    return pd.DataFrame(rows)


df_cat = aggregate_categories(df_ate_s)

for ds in sorted(df_cat["dataset"].unique()):
    print(f"--- {ds}: mean abs ATE error per category ---")
    display(
        df_cat[df_cat["dataset"] == ds]
        .pivot_table(index="model", columns="category", values="mean")
        .reindex(columns=[c for c in CATEGORIES if c in df_cat["category"].unique()])
    )

# ===== cell 15 =====
def add_intervention_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "intv_value" not in df.columns:
        df["intv_value"] = df["intervention"].astype(str).str.extract(r"=(-?[\d.]+)")[0].astype(float)
    if "intv_mag" not in df.columns:
        df["intv_mag"] = df["intv_value"].abs()
    return df


def aggregate_abs_error(df: pd.DataFrame, group_cols=("dataset", "path_type", "intv_mag")) -> pd.DataFrame:
    # Aggregate per-pair (mean, std) into per-cell (mean, std): total variance.
    rows = []
    for keys, sub in df.groupby(list(group_cols)):
        mu = sub["abs_error_mean"].astype(float).values
        sd = sub["abs_error_std"].fillna(0.0).astype(float).values
        n = len(mu)
        if n == 0:
            continue
        mu_g = float(np.mean(mu))
        var = float(np.mean(sd ** 2)) + (float(np.var(mu, ddof=1)) if n > 1 else 0.0)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"n_pairs": n, "mean": mu_g, "std": float(np.sqrt(var))})
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

# ===== cell 16 =====
def build_summary_table(df_ate_s: pd.DataFrame, role_map: dict,
                        path_type_order=("direct", "indirect", "zero"),
                        decimals: int = 2, mag_decimals: int = 1):
    # Baseline / Oracle / Ours abs. ATE error per (dataset, path_type, |s|).
    # Returns (raw_df, display_df, latex_str).  Models missing from df_ate_s are
    # simply absent; role labels come from role_map (fallback: model name).
    group_cols = ["dataset", "path_type", "intv_mag"]
    df = add_intervention_magnitude(df_ate_s)

    models = sorted(df["model"].unique(), key=lambda m: role_map.get(m, m))
    baseline = next((m for m in models if role_map.get(m) == "Baseline"), models[0])
    label_of = {m: role_map.get(m, m) for m in models}
    methods = [m for m in models if m != baseline]
    if role_map.get(baseline) != "Baseline":
        warnings.warn(f"No model mapped to 'Baseline'; using '{baseline}'.")

    plain = {m: aggregate_abs_error(df[df["model"] == m]) for m in models}
    raw = plain[baseline][group_cols + ["n_pairs", "mean", "std"]].rename(
        columns={"mean": "Baseline_mean", "std": "Baseline_std"})
    for m in methods:
        lab = label_of[m]
        raw = raw.merge(
            plain[m][group_cols + ["mean", "std"]].rename(
                columns={"mean": f"{lab}_mean", "std": f"{lab}_std"}),
            on=group_cols, how="left")
        d = compute_delta_pct(plain[baseline], plain[m], group_cols)
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

    method_labels = [label_of[m] for m in methods]
    best = (raw[[f"{lab}_mean" for lab in method_labels]].idxmin(axis=1)
            .str.replace("_mean", "", regex=False) if len(method_labels) >= 2 else None)

    disp = pd.DataFrame()
    disp["Dataset"] = raw["dataset"].astype(str)
    disp["Path type"] = raw["path_type"].astype(str)
    disp["|s|"] = raw["mag_label"]
    disp["n"] = raw["n_pairs"].astype(int)
    disp["Baseline"] = [_fmt_pm(a, b, decimals) for a, b in zip(raw["Baseline_mean"], raw["Baseline_std"])]
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
        head1.append(f"\\multicolumn{{{len(method_labels)}}}{{c}}{{Delta\\% vs Baseline $\\uparrow$}}")
    lines.append(" & ".join(head1) + " \\\\")
    end = n_index + n_meth
    cmids = [f"\\cmidrule(lr){{{n_index + 1}-{end}}}"]
    if method_labels:
        cmids.append(f"\\cmidrule(lr){{{end + 1}-{end + len(method_labels)}}}")
    lines.append(" ".join(cmids))
    head2 = (["Dataset", "Path type", "$|s|$", "$n$", "Baseline"] + method_labels
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
        cells = [ds if ds != last_ds else "", pt if pt != last_pt else "",
                 str(row["mag_label"]), f"{int(row['n_pairs'])}"]
        cells.append(_fmt_pm_latex(row["Baseline_mean"], row["Baseline_std"], decimals))
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


raw_df, display_df, latex_str = build_summary_table(df_ate_s, ROLE_MAP)
display(display_df)
print(latex_str)

# ===== cell 18 =====
def plot_spiderweb(df_cat: pd.DataFrame, normalize_axes: bool = True,
                   save_stem: str = "fig_ate_spiderweb"):
    cats = CATEGORIES
    datasets = sorted(df_cat["dataset"].unique())
    models = sorted(df_cat["model"].unique(), key=lambda m: ROLE_MAP.get(m, m))
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 5.6),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        piv = (df_cat[df_cat["dataset"] == ds]
               .pivot_table(index="model", columns="category", values="mean")
               .reindex(index=models, columns=cats))
        if normalize_axes:
            piv = piv / piv.max(axis=0)
        for model in models:
            if model not in piv.index:
                continue
            vals = piv.loc[model]
            if vals.isna().all():
                continue
            if vals.isna().any():
                warnings.warn(f"{ds}/{model}: incomplete categories, skipped in spider plot")
                continue
            v = vals.tolist() + [vals.iloc[0]]
            (line,) = ax.plot(angles, v, marker="o", ms=3, label=ROLE_MAP.get(model, model))
            ax.fill(angles, v, alpha=0.08, color=line.get_color())
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=9)
        ax.set_title(ds, pad=18)
        ax.grid(True, alpha=0.4)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.28, -0.20),
                   ncol=len(models), fontsize=9)
    fig.suptitle("Mean abs ATE error by category"
                 + (" (per-axis normalized)" if normalize_axes else ""))
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(f"{save_stem}{ext}", dpi=200, bbox_inches="tight")
    plt.show()


plot_spiderweb(df_cat)