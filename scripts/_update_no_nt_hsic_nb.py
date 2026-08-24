"""Update analyze_no_NT_hsic_signal.ipynb with the nodesc-budget arm + BKD sweep.

1) Section 5: add the ``no_NT_bkd_02_nodesc_budget_11435503`` arm to the margin
   comparison (4 arms; the final-epoch scatter grid grows to 3x2).
2) Section 4 probe: add a ``train_mode`` flag to ``hsic_grad_on_queries`` (the
   training-time gradient with gate sampling + batch key dropout active).
3) New Section 7: the budgeted descendant mask - what ``kept_frac`` / ``cyclic``
   log, their trajectories, a from-checkpoint reproduction of the mask, the
   final-metrics table across arms, and the safe-NOTEARS todo arm.
4) New Section 8: the BKD dose-response - do we dilute the HSIC less with high
   dropout?  (logged metrics, margins, and the train-vs-eval dilution probe).
5) The synthesis becomes Section 9 with the new fill-in prompts.

Run:  python scripts/_update_no_nt_hsic_nb.py
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


def edit_source(cell, old, new):
    # Targeted replacement inside a cell source (asserts the pattern is found).
    src = "".join(cell["source"])
    assert old in src, f"pattern not found in cell {cell['id']!r}:\n{old}"
    cell["source"] = src.replace(old, new, 1).splitlines(keepends=True)


def reset_outputs(cell):
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


# =============================================================================
# New cell sources
# =============================================================================

SECTION7_INTRO = """\
## 7. The budgeted descendant mask: reading kept_frac / cyclic

**Arm:** `no_NT_bkd_02_nodesc_budget_11435503` - the no_NT_bkd_02_nodesc arm
with the BUDGETED descendant mask (`hsic_descendant_mode: budget`,
`budget_frac=0.25`, `per_row=True`, `tnorm=min`, `exclude_self=True`,
warmup 0). The threshold variant hardens the posterior at 0.5 and hopes the
result is a DAG; the budgeted variant ranks the pairs by the SOFT descendant
score (the fuzzy transitive closure of the detached posterior, min t-norm)
and excludes the top ~25% per child row, so the mask triggers every step.

**What `train_hsic_desc_kept_frac` indicates.** The epoch-mean fraction of
the N x N = 400 (child, candidate-parent) HSIC pairs that SURVIVE the mask on
train batches. Per child row the top `ceil(0.25 * 20) = 5` pairs are excluded,
the diagonal always ranked first (`exclude_self`). Readings:

- `1.0` - the mask is INACTIVE (fallback: feature off / no score tensor /
  warmup /, in threshold mode only, the min_kept_frac collapse guard);
- `0.75` - the cap binds exactly (5 of 20 per row);
- `< 0.75` - ties in the soft closure push exclusion PAST the cap (the budget
  is not a hard cap: every pair tied at the k-th score is excluded);
- `0.95` - only the diagonal excluded (a degenerate, near-zero posterior).

It is the activity/collapse indicator of the descendant exclusion: whether
the mask fired, and how hard it bit.

**What `train_hsic_desc_cyclic` indicates.** 1 when any node is reachable
from itself with min-link confidence > 0.5 in the soft closure - the learned
posterior contains a soft cycle. Diagnostic only; the mask still acts.

**Caveats.**
- With masking on, `{stage}_hsic` is a MASKED mean whose normalisation set
  changes step to step - compare HSIC levels across arms with care.
- In THIS run the logged val-side diagnostics flip to `kept_frac = 1.0`,
  `cyclic = 0` from epoch ~982 (and for most mid-training val epochs), while
  the train-side mask stays active. The from-checkpoint reproduction below
  (current code) gives an ACTIVE, cycle-flagging mask at the final epoch -
  the late val logs are a run-time fallback the local code does not reproduce
  (likely a cluster/local code drift). Conclusions rely on the train-side
  logs and the checkpoint reproduction.
"""

SECTION7_LOGS = """\
# ---- The budget arm's mask diagnostics from the training logs ------------------------------
BUDGET_ARM = ROOT / ("experiments/6_INVESTIGATIONS/LARGER_DAGS/"
                     "no_NT_bkd_02_nodesc_budget_11435503")
BUDGET_METRICS = BUDGET_ARM / "k_0" / "logs" / "csv" / "version_0" / "metrics.csv"

bm = pd.read_csv(BUDGET_METRICS)
btr = bm.dropna(subset=["train_hsic_desc_kept_frac"])
bva = bm.dropna(subset=["val_hsic_desc_kept_frac"])
bsummary = json.load(open(BUDGET_ARM / "adaptive_training_summary.json"))
btrans = bsummary["transitions"]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for a, b in zip(btrans, btrans[1:] + [{"global_epoch": bsummary["total_epoch_budget"]}]):
    if a["to_phase"] == "structure":
        for ax in axes:
            ax.axvspan(a["global_epoch"], b["global_epoch"], color="#009E73", alpha=0.08)
axes[0].plot(btr.epoch, btr.train_hsic_desc_kept_frac, color=OKABE_ITO[0], lw=1.0,
             label="train")
axes[0].plot(bva.epoch, bva.val_hsic_desc_kept_frac, color=OKABE_ITO[3], lw=1.0,
             alpha=0.7, label="val")
axes[0].axhline(0.75, color="#999999", lw=0.8, ls="--")   # the 0.25 per-row cap binding
axes[0].axhline(0.95, color="#999999", lw=0.8, ls=":")    # diagonal-only exclusion
axes[0].set_xlim(0, btr.epoch.max())
axes[0].set_ylim(0, 1.02)
axes[0].set_xlabel("epoch [-]")
axes[0].set_ylabel("kept fraction [-]")
axes[0].set_title("HSIC pairs kept by the descendant mask\\n(structure phases shaded)")
axes[0].legend(fontsize=9)

ROLL = 21   # rolling-mean window for the 0/1 cyclic flag
axes[1].plot(btr.epoch, btr.train_hsic_desc_cyclic.rolling(ROLL, center=True).mean(),
             color=OKABE_ITO[0], lw=1.2, label="train (21-ep mean)")
axes[1].plot(bva.epoch, bva.val_hsic_desc_cyclic.rolling(ROLL, center=True).mean(),
             color=OKABE_ITO[3], lw=1.2, alpha=0.7, label="val (21-ep mean)")
axes[1].set_xlim(0, btr.epoch.max())
axes[1].set_ylim(-0.02, 1.02)
axes[1].set_xlabel("epoch [-]")
axes[1].set_ylabel("cyclic flag [-]")
axes[1].set_title("Soft-cycle detection (closure diagonal > 0.5)")
axes[1].legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "nodesc_budget_mask_logs")
plt.show()

on = btr[btr.train_hsic_desc_cyclic > 0.5]
off_epochs = btr.epoch[btr.train_hsic_desc_cyclic <= 0.5]
sustained_from = int(off_epochs.max() + 1) if len(off_epochs) else int(btr.epoch.min())
print(f"train: kept_frac min={btr.train_hsic_desc_kept_frac.min():.3f} "
      f"mean(last 100 ep)={btr.train_hsic_desc_kept_frac.tail(100).mean():.3f}")
print(f"train: cyclic on {len(on)}/{len(btr)} epochs, first={int(on.epoch.min())}, "
      f"sustained from ~{sustained_from}")
print(f"val:   last epoch with the mask active = "
      f"{int(bva.epoch[bva.val_hsic_desc_kept_frac < 1.0].max())} "
      f"(kept_frac = 1.0 afterwards = the run-time fallback; see below)")
"""

SECTION7_REPRO = """\
# ---- Ground-truth the mask at the final checkpoint (eval mode, current code) ----------------
from omegaconf import OmegaConf
from causaliT.utils.descendant_mask import (build_hsic_pair_mask_budgeted,
                                            soft_transitive_closure)

bcfg = OmegaConf.load(BUDGET_ARM / "config.yaml")["training"]
mask_kwargs = dict(s_seq_len=N_S, homogeneous_nodes=True,
                   budget_frac=float(bcfg.hsic_descendant_budget_frac),
                   per_row=bool(bcfg.hsic_descendant_per_row),
                   exclude_self=bool(bcfg.get("hsic_descendant_exclude_self", True)),
                   excluded_weight=float(bcfg.hsic_descendant_weight),
                   tnorm=str(bcfg.hsic_descendant_tnorm), hops=None)

budget_ckpts = {e: p for p in (BUDGET_ARM / "k_0" / "checkpoints").glob("*.ckpt")
                if (e := ckpt_epoch(p)) is not None}
e_last = max(budget_ckpts)
model = AttentionSelectorForecaster.load_from_checkpoint(budget_ckpts[e_last],
                                                         map_location="cpu")
model.eval()
with torch.no_grad():
    model.forward(data_source=S_ALL[:256], data_intermediate=X_ALL[:256])
score = model.model.get_score_tensor_for_sparsity().detach().double()
mask, kept_frac, is_cyclic = build_hsic_pair_mask_budgeted(score, **mask_kwargs)
desc = soft_transitive_closure(score, tnorm=mask_kwargs["tnorm"])
self_reach = desc.diagonal().numpy()
del model
print(f"epoch {e_last}: reproduced mask kept_frac={kept_frac:.3f} cyclic={is_cyclic}")
print("(run-time log at the final epochs: val kept_frac=1.0, cyclic=0 - fallback artifact)")
print(f"score tensor: min={score.min():.3f} max={score.max():.3f} "
      f"nan={bool(score.isnan().any())}")

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2))
im = axes[0].imshow(desc.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
mark_gt(axes[0])
axes[0].set_xticks(range(N), NODES, rotation=90)
axes[0].set_yticks(range(N), NODES)
axes[0].set_xlabel("source j")
axes[0].set_ylabel("child i")
axes[0].set_title(f"Soft descendant score (epoch {e_last})\\nGT edges framed")
fig.colorbar(im, ax=axes[0], fraction=0.046, label="soft reachability [-]")

order = np.argsort(-self_reach)
axes[1].bar(range(N), self_reach[order],
            color=[OKABE_ITO[3] if v > 0.5 else OKABE_ITO[0] for v in self_reach[order]])
axes[1].axhline(0.5, color="#999999", lw=0.8, ls="--")
axes[1].set_xticks(range(N), [NODES[i] for i in order], rotation=90)
axes[1].set_ylabel("self-reachability [-]")
axes[1].set_title("Soft-closure diagonal: nodes on a cycle (> 0.5)\\n"
                  "(GT is a DAG: any bar above the line is spurious)")
fig.tight_layout()
save_fig(fig, "nodesc_budget_mask_repro")
plt.show()
"""

SECTION7_TABLE = """\
# ---- Final metrics across the four arms ------------------------------------------------------
# Note: the nodesc arms' final val_hsic_desc_* are the fallback values (see the caveat above).
ARM_FINAL = {
    "no_NT": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT",
    "no_NT_bkd_02": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02",
    "no_NT_bkd_02_nodesc": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02_nodesc",
    "no_NT_bkd_02_nodesc_budget": BUDGET_ARM,
}
METRIC_COLS = ["val_hsic", "val_x_mae", "val_x_r2", "val_score_sparse",
               "query_norm/mean_M", "test_hsic", "test_x_mae", "test_x_r2",
               "val_hsic_desc_kept_frac", "val_hsic_desc_cyclic"]
rows = {}
for name, d in ARM_FINAL.items():
    m = json.load(open(d / "kfold_summary.json"))["fold_results"]["0"]["metrics"]
    rows[name] = {c: m.get(c, np.nan) for c in METRIC_COLS}
    rows[name]["final_margin"] = MARGIN_ARMS[name]["margin"][-1].mean()
    rows[name]["oriented_frac"] = (MARGIN_ARMS[name]["margin"][-1] > 0).mean()
table = pd.DataFrame(rows).T
print(table.to_string(float_format=lambda v: f"{v:.4g}"))
"""

SECTION7_TODO = """\
### TODO (future arm): safe NOTEARS integration

The budget arm's end state flags a soft cycle: `train_hsic_desc_cyclic = 1`
for the last ~400 epochs, reproduced at the final checkpoint
(self-reachability up to ~0.72; several 2-cycles tied at exactly 0.5 on the
direction gate). The descendant mask EXCLUDES the biased pairs but does not
PREVENT the cycle.

**Todo arm:** reintroduce the NOTEARS acyclicity term (`kappa > 0`;
`_notears_acyclicity` h(A) = tr(exp(A * A)) - d is already implemented and
logged as `{stage}_notears`) in a signal-safe way:

- small kappa, structure phases only, so it cannot fight the reconstruction
  warmup (the baseline ill region is the counterexample);
- gate it on the cyclic flag / HSIC plateau (the `l0_gate_on_hsic` pattern is
  the template) so the penalty only fires when a cycle is actually present;
- keep the budget mask ON: NOTEARS removes the cycle, the mask removes the
  descendant bias - complementary, not redundant.

Watch: `train_hsic_desc_cyclic` (should turn off), `kept_frac` (should rise
toward the cap), the Section-5 margins (should widen), and val_hsic (should
NOT regress vs this arm).
"""

SECTION8_INTRO = """\
## 8. BKD dose-response: do we dilute the HSIC less with high dropout?

**Sweep:** `bkd_sweep_11437173` - `batch_key_dropout` (BKD) in {0.0, 0.05,
0.1, 0.2, 0.3} on top of the no_NT config (same dataset, same seed; p = 0.0
reproduces no_NT; no descendant exclusion here: the plain HSIC mean). BKD
zeroes entire key COLUMNS of the attention matrix with a single mask drawn
once per batch (batch-consistent), train mode only, no 1/(1-p) rescaling.

**The dilution question.** With key j dropped in a p-fraction of batches, the
pair-(i, j) HSIC gradient on q_i only flows in the surviving (1-p) fraction -
the per-pair structural signal is thinned, and the dropped-parent batches add
noise (the parent's contribution sits unexplained in the residual). Naive
prediction: the usable signal scales like (1-p). The counter-mechanism:
dropping a true parent's key EXPOSES its contribution in the residual, so
HSIC(X_j, r_i) spikes exactly where an edge is missing - a leave-one-out
signal that REINFORCES true parents and breaks the symmetric descendant
coupling. If the measured dilution is sub-linear in p, high dropout dilutes
the HSIC less than feared.

**Readouts.**
- logged metrics: val_hsic (clean, eval mode) vs train_hsic (computed UNDER
  dropout) - the train/val gap measures the dropout inflation of the
  training-time signal;
- structure learning: the Section-5 direction margins per level;
- the dilution probe: the Section-4 HSIC gradient probe at matched
  checkpoints, run in TRAIN mode (BKD + gate sampling active: the actual
  training-time gradient) and in eval mode (the mean-field reference) -
  per-batch consistency (SNR), direction cos(-g, parent centroid), magnitude.
"""

SECTION8_LOAD = """\
# ---- Load the sweep runs ---------------------------------------------------------------------
SWEEP_DIR = ROOT / ("experiments/6_INVESTIGATIONS/LARGER_DAGS/bkd_sweep_11437173/"
                    "sweeper/runs/combinations")
BKD_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
BKD_DIRS = {p: SWEEP_DIR / f"bkd_sweep_combo_batch_key_dropout_{p}" for p in BKD_LEVELS}
BKD_COLORS = {p: c for p, c in zip(BKD_LEVELS, OKABE_ITO)}

BKD = {}
for p, d in BKD_DIRS.items():
    ck = {e: f for f in (d / "k_0" / "checkpoints").glob("*.ckpt")
          if (e := ckpt_epoch(f)) is not None}
    ep = sorted(ck)
    QH_p, M_p, K_p = [], [], None
    for e in ep:
        q, k, m = load_qkm(ck[e])
        if K_p is None:
            K_p = k
        assert np.allclose(k, K_p), "key frame must be frozen"
        QH_p.append(q / np.linalg.norm(q, axis=1, keepdims=True))
        M_p.append(m)
    assert np.allclose(K_p, K), "the key frame must be shared with the no_NT arm"
    BKD[p] = dict(
        epochs=ep, ckpts=ck, QH=np.stack(QH_p), M=np.stack(M_p),
        metrics=pd.read_csv(d / "k_0" / "logs" / "csv" / "version_0" / "metrics.csv"),
        summary=json.load(open(d / "adaptive_training_summary.json")),
        final=json.load(open(d / "kfold_summary.json"))["fold_results"]["0"]["metrics"],
    )
    print(f"bkd={p}: {len(ep)} checkpoints, epochs {ep[0]}..{ep[-1]}, "
          f"final val_hsic={BKD[p]['final']['val_hsic']:.5f} "
          f"val_x_mae={BKD[p]['final']['val_x_mae']:.5f}")
"""

SECTION8_METRICS = """\
# ---- Logged metrics vs dropout level ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for p in BKD_LEVELS:
    va = BKD[p]["metrics"].dropna(subset=["val_hsic"])
    axes[0].plot(va.epoch, va.val_hsic.rolling(5, center=True).mean(),
                 color=BKD_COLORS[p], lw=1.2, label=f"p={p}")
axes[0].set_xlim(0, max(BKD[p]["epochs"][-1] for p in BKD_LEVELS))
axes[0].set_xlabel("epoch [-]")
axes[0].set_ylabel("val HSIC [-]")
axes[0].set_title("Clean (eval-mode) HSIC\\n(5-epoch smoothed)")
axes[0].legend(fontsize=9)


def phase_in(summary, ep):
    # Active phase at epoch ep for an arbitrary arm's summary.
    phase = summary["start_phase"]
    for t in summary["transitions"]:
        if ep >= t["global_epoch"]:
            phase = t["to_phase"]
    return phase


# train/val HSIC gap during structure phases: the dropout inflation of the signal
gap = {}
for p in BKD_LEVELS:
    met = BKD[p]["metrics"]
    tr = met.dropna(subset=["train_hsic"])[["epoch", "train_hsic"]]
    va = met.dropna(subset=["val_hsic"])[["epoch", "val_hsic"]]
    j = tr.merge(va, on="epoch")
    is_struct = j.epoch.map(lambda e: phase_in(BKD[p]["summary"], e) == "structure")
    gap[p] = (j.train_hsic[is_struct] / j.val_hsic[is_struct]).mean()
axes[1].plot(BKD_LEVELS, [gap[p] for p in BKD_LEVELS], "o-", color=OKABE_ITO[0], lw=1.4)
axes[1].axhline(1, color="#999999", lw=0.8, ls="--")
axes[1].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[1].set_xlabel("batch key dropout p [-]")
axes[1].set_ylabel("train HSIC / val HSIC [-]")
axes[1].set_title("Dropout inflation of the training-time HSIC\\n(mean over structure epochs)")
fig.tight_layout()
save_fig(fig, "bkd_sweep_metrics")
plt.show()

rows = {}
for p in BKD_LEVELS:
    r = {c: BKD[p]["final"].get(c, np.nan) for c in
         ["val_hsic", "val_x_mae", "val_x_r2", "val_score_sparse",
          "query_norm/mean_M", "test_hsic", "test_x_mae"]}
    r["n_cycles"] = BKD[p]["summary"]["n_cycles"]
    rows[f"p={p}"] = r
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4g}"))
"""

SECTION8_MARGINS = """\
# ---- Structure learning vs dropout: direction margins on the true edges -----------------------
for p in BKD_LEVELS:
    S = BKD[p]
    coords = np.einsum("end,md->enm", S["QH"], K)          # (E, N, N)
    S["c_child"] = coords[:, edges[:, 0], edges[:, 1]]     # cos(q_i, k_j)
    S["c_parent"] = coords[:, edges[:, 1], edges[:, 0]]    # cos(q_j, k_i)
    S["margin"] = (S["M"][:, edges[:, 0]] * S["c_child"]
                   - S["M"][:, edges[:, 1]] * S["c_parent"]) * np.sqrt(F_FANIN)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for p in BKD_LEVELS:
    S = BKD[p]
    axes[0].plot(S["epochs"], S["margin"].mean(axis=1), color=BKD_COLORS[p],
                 lw=1.2, label=f"p={p}")
    axes[1].plot(S["epochs"], (S["margin"] > 0).mean(axis=1), color=BKD_COLORS[p],
                 lw=1.2, label=f"p={p}")
axes[0].axhline(0, color="#999999", lw=0.8)
axes[0].set_xlim(left=0)
axes[0].set_xlabel("epoch [-]")
axes[0].set_ylabel("direction margin [logit]")
axes[0].set_title("Mean direction margin on true edges")
axes[0].legend(fontsize=8)
axes[1].set_xlim(left=0)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("epoch [-]")
axes[1].set_ylabel("fraction with margin > 0 [-]")
axes[1].set_title("Fraction of true edges oriented")
axes[1].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "bkd_sweep_margins")
plt.show()

for p in BKD_LEVELS:
    a = np.einsum("nd,nd->n", BKD[p]["QH"][-1], CENT[GROUPS.index("parents")])
    coords = BKD[p]["QH"][-1] @ K.T
    mass = ((coords ** 2) * PA.astype(float)).sum(axis=1)[N_S:].mean()
    print(f"p={p}: final margin={BKD[p]['margin'][-1].mean():+.3f} "
          f"oriented={(BKD[p]['margin'][-1] > 0).mean():.2%} "
          f"align={np.nanmean(a[N_S:]):+.3f} parent mass={mass:.3f}")
"""

SECTION8_PROBE = """\
# ---- The dilution probe: train-mode vs eval-mode HSIC gradient --------------------------------
# Train mode = the actual training-time gradient (gate sampling + BKD active at the
# arm's rate); eval mode = the mean-field reference on the SAME checkpoint.  Same
# batches and the same RNG seed across arms for comparability.
BKD_PROBE_TARGETS = [500, 999]   # resolved to the closest saved checkpoint per run
BKD_PROBE = {}
for p in BKD_LEVELS:
    BKD_PROBE[p] = {}
    for target in BKD_PROBE_TARGETS:
        e = min(BKD[p]["ckpts"], key=lambda x: abs(x - target))
        perm = torch.randperm(S_ALL.shape[0], generator=torch.Generator().manual_seed(SEED))
        batches = [(S_ALL[b], X_ALL[b]) for b in perm.split(BATCH_SIZE)[:N_BATCHES]]
        out = {}
        for mode, train_mode in [("eval", False), ("train", True)]:
            torch.manual_seed(SEED)   # comparable gate/BKD draws across arms
            out[mode] = hsic_grad_on_queries(BKD[p]["ckpts"][e], batches,
                                             train_mode=train_mode)
        BKD_PROBE[p][e] = out
        print(f"p={p}: probed epoch {e} (target {target})")


def probe_stats(G):
    # (per-node SNR, mean descent direction) from per-batch gradients (B, N, d).
    Gn = G / np.maximum(np.linalg.norm(G, axis=2, keepdims=True), 1e-12)
    sim = np.einsum("bnd,cnd->bcn", Gn, Gn)
    snr = sim[~np.eye(G.shape[0], dtype=bool)].mean(axis=0)          # (N,)
    gmean = -G.mean(axis=0)                                          # (N, d)
    gmean = gmean / np.maximum(np.linalg.norm(gmean, axis=1, keepdims=True), 1e-12)
    return snr, gmean


rows = []
for p in BKD_LEVELS:
    for e, modes in BKD_PROBE[p].items():
        for mode, G in modes.items():
            snr, gmean = probe_stats(G)
            cos_pa = np.einsum("nd,nd->n", gmean,
                               np.nan_to_num(CENT[GROUPS.index("parents")]))
            rows.append(dict(p=p, epoch=e, mode=mode,
                             snr=np.nanmean(snr[N_S:]),
                             cos_parent=np.nanmean(cos_pa[N_S:]),
                             gnorm=np.linalg.norm(G, axis=2)[:, N_S:].mean()))
probe_df = pd.DataFrame(rows)
print(probe_df.round(4).to_string(index=False))

e_last = max(e for p in BKD_LEVELS for e in BKD_PROBE[p])
sub = probe_df[probe_df.epoch == e_last]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
for mode, mkr, col, lab in [("train", "o", OKABE_ITO[0], "train (BKD active)"),
                            ("eval", "s", OKABE_ITO[2], "eval (mean-field)")]:
    s = sub[sub["mode"] == mode].sort_values("p")   # (\"mode\" col: .mode is a DataFrame method)
    axes[0].plot(s.p, s.snr, mkr + "-", color=col, lw=1.4, label=lab)
    axes[1].plot(s.p, s.cos_parent, mkr + "-", color=col, lw=1.4, label=lab)
s0 = sub[(sub["mode"] == "train") & (sub.p == 0.0)].snr.iloc[0]
axes[0].plot(BKD_LEVELS, [s0 * (1 - p) for p in BKD_LEVELS], ls="--", color="#999999",
             lw=1.0, label="naive (1-p) thinning")
axes[0].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[0].set_xlabel("batch key dropout p [-]")
axes[0].set_ylabel("mean pairwise cos [-]")
axes[0].set_title(f"Per-batch gradient consistency (epoch {e_last})")
axes[0].legend(fontsize=9)
axes[1].axhline(0, color="#999999", lw=0.8)
axes[1].set_xlim(min(BKD_LEVELS), max(BKD_LEVELS))
axes[1].set_xlabel("batch key dropout p [-]")
axes[1].set_ylabel("cos(-g, parent centroid) [-]")
axes[1].set_title(f"HSIC push direction (epoch {e_last})")
axes[1].legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "bkd_sweep_probe")
plt.show()
"""

SECTION8_ANSWER = """\
### Do we dilute the HSIC less with high dropout? (fill in after running)

- SNR_train(p) vs the naive (1-p) thinning reference: ...
- Direction cos(-g, parent centroid), train vs eval per level: ...
- The train/val HSIC gap vs p (dropout inflation of the training signal): ...
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

### The BKD dose-response (Section 8)
- Do we dilute the HSIC less with high dropout? (SNR vs the naive (1-p)
  reference, direction vs p): ...
- Best dropout level (signal vs reconstruction trade-off): ...

### Notes
- The Section-4 probe runs in eval mode (deterministic gates): the mean-field
  gradient of the stochastic training gates. Section 8 adds the train-mode
  probe (gate sampling + BKD active) for the dilution readout.
- The budget M ~ 1.02 is not the constraint; the direction is.
- The displacement attribution (Section 3) is the realized movement; the
  gradient probe (Section 4) is the force - compare them.
- TODO (future arm): safe NOTEARS integration - see Section 7.
"""


# =============================================================================
# Apply the edits
# =============================================================================

def main():
    nb = json.load(open(NB, encoding="utf-8"))
    cells = nb["cells"]
    by_id = {c["id"]: i for i, c in enumerate(cells) if "id" in c}

    # ---- 1) Section 5: four arms -------------------------------------------------------------
    edit_source(
        cells[by_id["00472563"]],
        """Compared across three arms:

- **no_NT** (this notebook's arm),
- **no_NT_bkd_02** (+ batch key dropout 0.1),
- **no_NT_bkd_02_nodesc** (+ descendant exclusion in the HSIC mean).
""",
        """Compared across four arms:

- **no_NT** (this notebook's arm),
- **no_NT_bkd_02** (+ batch key dropout 0.1),
- **no_NT_bkd_02_nodesc** (+ descendant exclusion in the HSIC mean,
  threshold mode),
- **no_NT_bkd_02_nodesc_budget** (descendant exclusion, BUDGETED mode: the
  top 25% pairs by soft descendant score excluded per child row, so the mask
  fires every step; run `no_NT_bkd_02_nodesc_budget_11435503`).
""",
    )

    edit_source(
        cells[by_id["9a3ada56"]],
        '''MARGIN_ARM_DIRS = {
    "no_NT_bkd_02": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02",
    "no_NT_bkd_02_nodesc": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02_nodesc",
}
MARGIN_ARM_COLORS = {"no_NT": "#0072B2", "no_NT_bkd_02": "#009E73",
                     "no_NT_bkd_02_nodesc": "#CC79A7"}''',
        '''MARGIN_ARM_DIRS = {
    "no_NT_bkd_02": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02",
    "no_NT_bkd_02_nodesc": ROOT / "experiments/6_INVESTIGATIONS/LARGER_DAGS/no_NT_bkd_02_nodesc",
    "no_NT_bkd_02_nodesc_budget": ROOT / ("experiments/6_INVESTIGATIONS/LARGER_DAGS/"
                                          "no_NT_bkd_02_nodesc_budget_11435503"),
}
MARGIN_ARM_COLORS = {"no_NT": "#0072B2", "no_NT_bkd_02": "#009E73",
                     "no_NT_bkd_02_nodesc": "#CC79A7",
                     "no_NT_bkd_02_nodesc_budget": "#E69F00"}''',
    )

    edit_source(
        cells[by_id["b072a278"]],
        "fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.4), squeeze=False)",
        "fig, axes = plt.subplots(3, 2, figsize=(7.6, 10.8), squeeze=False)",
    )
    edit_source(
        cells[by_id["b072a278"]],
        "ax = axes.flat[-1]",
        "ax = axes.flat[len(MARGIN_ARMS)]",
    )
    edit_source(
        cells[by_id["b072a278"]],
        '''ax.set_title("Final margin distribution")
ax.legend(fontsize=8)
fig.suptitle(''',
        '''ax.set_title("Final margin distribution")
ax.legend(fontsize=8)
for ax in axes.flat[len(MARGIN_ARMS) + 1:]:
    ax.axis("off")
fig.suptitle(''',
    )

    # ---- 2) Section 4 probe: train_mode flag ---------------------------------------------------
    edit_source(
        cells[by_id["b6bd9904"]],
        '''def hsic_grad_on_queries(ckpt_path, batches):
    # Per-batch HSIC gradient on the query embeddings. Returns (B, N, d).
    model = AttentionSelectorForecaster.load_from_checkpoint(ckpt_path,
                                                             map_location="cpu")
    model.eval()   # deterministic gates: the mean-field gradient''',
        '''def hsic_grad_on_queries(ckpt_path, batches, train_mode=False):
    # Per-batch HSIC gradient on the query embeddings. Returns (B, N, d).
    # train_mode=False: eval mode (deterministic gates) - the mean-field
    # gradient of the stochastic training gates.  train_mode=True: the actual
    # training-time forward (gate sampling + batch key dropout active).
    model = AttentionSelectorForecaster.load_from_checkpoint(ckpt_path,
                                                             map_location="cpu")
    if train_mode:
        model.train()   # stochastic gates + BKD: the training-time gradient
    else:
        model.eval()    # deterministic gates: the mean-field gradient''',
    )

    # ---- 5) Synthesis (before inserting, the index is still valid) ------------------------------
    cells[by_id["fd9b2158"]]["source"] = SYNTHESIS.splitlines(keepends=True)

    # ---- 3)+4) New Sections 7 and 8, inserted before the synthesis ------------------------------
    new_cells = [
        md_cell("nodesc_budget_intro", SECTION7_INTRO),
        code_cell("nodesc_budget_logs", SECTION7_LOGS),
        code_cell("nodesc_budget_repro", SECTION7_REPRO),
        code_cell("nodesc_budget_table", SECTION7_TABLE),
        md_cell("nodesc_budget_todo", SECTION7_TODO),
        md_cell("bkd_sweep_intro", SECTION8_INTRO),
        code_cell("bkd_sweep_load", SECTION8_LOAD),
        code_cell("bkd_sweep_metrics", SECTION8_METRICS),
        code_cell("bkd_sweep_margins", SECTION8_MARGINS),
        code_cell("bkd_sweep_probe", SECTION8_PROBE),
        md_cell("bkd_sweep_answer", SECTION8_ANSWER),
    ]
    cells[by_id["fd9b2158"]:by_id["fd9b2158"]] = new_cells

    # Modified code cells get stale outputs: reset them (plus the margin
    # trajectory cell, whose output changes with the 4th arm).
    for cid in ["9a3ada56", "26eb31bd", "b072a278", "b6bd9904"]:
        reset_outputs(cells[by_id[cid]])

    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---- Validate -------------------------------------------------------------------------------
    nb2 = json.load(open(NB, encoding="utf-8"))
    ids = [c["id"] for c in nb2["cells"] if "id" in c]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    try:
        import nbformat
        nbformat.validate(nb2)
        print("nbformat.validate: OK")
    except ImportError:
        print("nbformat not installed; skipped validation")
    print(f"wrote {NB}: {len(nb2['cells'])} cells "
          f"({len(new_cells)} new, sections renumbered 7/8/9)")


if __name__ == "__main__":
    main()
