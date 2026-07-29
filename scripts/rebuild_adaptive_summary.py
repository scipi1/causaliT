"""
Rebuild a missing ``adaptive_training_summary.json`` for an INTERRUPTED adaptive
run (Option B — lightweight, CSV-only reconstruction).

The adaptive trainer (``causaliT/training/adaptive_trainer.py``) only writes
``adaptive_training_summary.json`` at the very end of ``adaptive_trainer()``.  If
the process is killed before that final write, the per-phase checkpoints
(``stage_checkpoints/phase_*_end.ckpt``) and the per-epoch log
(``k_0/logs/csv/version_0/metrics.csv``) still exist, but the summary the
evaluation notebook expects is gone.

This script reconstructs a summary with the SAME schema the notebook consumes,
using ONLY the surviving CSV + checkpoint filenames (no model loading):

* ``transitions``    — one record per discovered ``phase_*_end.ckpt``. The phase
  index and ``from_phase`` come from the checkpoint filename; ``global_epoch``,
  ``monitor_value`` and ``phase_best`` come from the matching contiguous
  ``adaptive_phase`` segment in ``metrics.csv``. ``reason`` is a best-effort
  label matching the trainer's naming; ``dag_diagnostics`` score margins are set
  to ``null`` (not recoverable without loading the checkpoints — the notebook
  tolerates this: the score-margin plot renders empty and the DAG heatmaps in
  Sections 2–3 load the checkpoints directly).
* ``final_metrics`` — the ``val_*`` fields from the LAST logged epoch. Because
  the CSV has no ``test_*`` columns or timing fields, ``test_*`` mirror the
  ``val_*`` proxy and the timing fields are NaN (interrupted-run recovery).

Usage::

    python scripts/rebuild_adaptive_summary.py experiments/6_INVESTIGATIONS/SELF_ATTENTION/results/GSA_shared_qk_qk_inj_7812192

The summary is written to ``<experiment_dir>/adaptive_training_summary.json``.
"""

import argparse
import glob
import json
import math
import re
from os.path import basename, exists, join

import pandas as pd

try:
    from omegaconf import OmegaConf
    _HAVE_OMEGACONF = True
except Exception:  # pragma: no cover
    _HAVE_OMEGACONF = False


PHASE_NAME = {0.0: "reconstruct", 1.0: "structure"}


def _find_metrics_csv(experiment_dir: str) -> str:
    """Locate the per-epoch metrics.csv under the fold (``k_*``) tree."""
    pattern = join(experiment_dir, "k_*", "logs", "csv", "*", "metrics.csv")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No metrics.csv found under {pattern}")
    return hits[0]


def _load_epoch_frame(metrics_csv: str) -> pd.DataFrame:
    """Collapse the train/val split rows into one row per epoch."""
    df = pd.read_csv(metrics_csv)
    g = df.groupby("epoch").first().reset_index()
    g["adaptive_phase"] = g["adaptive_phase"].ffill().bfill()
    return g


def _phase_segments(g: pd.DataFrame):
    """Return contiguous [(phase_code, epoch_start, epoch_end), ...] runs."""
    segs = []
    d = g[["epoch", "adaptive_phase"]].dropna(subset=["adaptive_phase"]).reset_index(drop=True)
    if d.empty:
        return segs
    run_code = d["adaptive_phase"].iloc[0]
    run_start = int(d["epoch"].iloc[0])
    prev_ep = run_start
    for i in range(1, len(d)):
        code = d["adaptive_phase"].iloc[i]
        ep = int(d["epoch"].iloc[i])
        if code != run_code:
            segs.append((run_code, run_start, prev_ep))
            run_code, run_start = code, ep
        prev_ep = ep
    segs.append((run_code, run_start, prev_ep))
    return segs


def _parse_phase_ckpt(path: str):
    m = re.search(r"phase_(\d+)_([a-zA-Z]+)_end", basename(path))
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def _val_at_epoch(g: pd.DataFrame, epoch: int, col: str):
    row = g[g["epoch"] == epoch]
    if row.empty or col not in g:
        return None
    v = row[col].iloc[0]
    return None if pd.isna(v) else float(v)


def _best_over_segment(g: pd.DataFrame, e0: int, e1: int, col: str):
    seg = g[(g["epoch"] >= e0) & (g["epoch"] <= e1)]
    if seg.empty or col not in g:
        return None
    v = seg[col].min()  # monitor is val_x_mae (lower is better)
    return None if pd.isna(v) else float(v)


def rebuild_summary(experiment_dir: str) -> dict:
    metrics_csv = _find_metrics_csv(experiment_dir)
    g = _load_epoch_frame(metrics_csv)
    segments = _phase_segments(g)

    # --- config (best-effort; only used for a few metadata fields) ---
    monitor = "val_x_mae"
    total_budget = None
    start_phase = "reconstruct"
    data_split_ratio = None
    cfg_path = join(experiment_dir, "config_atsel.yaml")
    if _HAVE_OMEGACONF and exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        ad = cfg.get("adaptive_training", {}) or {}
        monitor = str(ad.get("monitor", monitor))
        total_budget = ad.get("total_epoch_budget", None)
        start_phase = str(ad.get("start_phase", start_phase)).lower()
        data_split_ratio = ad.get("data_split_ratio", None)

    # --- discover ordered phase-end checkpoints ---
    ckpts = sorted(glob.glob(join(experiment_dir, "stage_checkpoints", "phase_*_end.ckpt")))
    parsed = [(_parse_phase_ckpt(p)[0], _parse_phase_ckpt(p)[1], p) for p in ckpts]
    parsed = [t for t in parsed if t[0] is not None]
    parsed.sort(key=lambda t: t[0])

    transitions = []
    n_cycles = 0
    for k, (idx, from_phase, ckpt) in enumerate(parsed):
        # Segment k corresponds to checkpoint k (phases alternate in lockstep).
        seg = segments[k] if k < len(segments) else None
        if seg is not None:
            _, e0, e1 = seg
            global_epoch = int(e1)
            monitor_value = _val_at_epoch(g, e1, "val_x_mae")
            phase_best = _best_over_segment(g, e0, e1, "val_x_mae")
        else:
            global_epoch, monitor_value, phase_best = None, None, None

        # to_phase = the next checkpoint's from_phase, else the alternate phase.
        if k + 1 < len(parsed):
            to_phase = parsed[k + 1][1]
        else:
            to_phase = "structure" if from_phase == "reconstruct" else "reconstruct"

        if from_phase == "reconstruct":
            reason = "recon_plateau"
        else:
            reason = "struct_hsic_plateau"
            n_cycles += 1

        transitions.append({
            "phase_index": idx,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "reason": reason,
            "global_epoch": global_epoch,
            "phase_epochs": (int(e1 - e0 + 1) if seg is not None else None),
            "monitor": monitor,
            "monitor_value": monitor_value,
            "phase_best": phase_best,
            "checkpoint": ckpt.replace("\\", "/"),
            "dag_diagnostics": {
                "phase": from_phase,
                "phase_index": idx,
                "epoch": global_epoch,
                "label": f"end_{from_phase}",
                "phi_cross": None,
                "phi_cross_decisiveness": None,
                "phi_cross_stats": None,
                "phi_self": None,
                "phi_self_decisiveness": None,
                "phi_self_stats": None,
                "soft_hamming_cross": None,
                "soft_hamming_self": None,
                "score_margin_cross": None,   # not recoverable from CSV alone
                "score_margin_self": None,
            },
        })

    # --- final_metrics from the last logged epoch (val_* proxy for test_*) ---
    last = g.iloc[-1]

    def _lv(col):
        v = last.get(col)
        return None if (v is None or pd.isna(v)) else float(v)

    val_x_r2 = _lv("val_x_r2")
    val_x_mae = _lv("val_x_mae")
    val_x_rmse = _lv("val_x_rmse")
    val_loss = _lv("val_loss")
    val_loss_x = _lv("val_loss_x")
    val_hsic = _lv("val_hsic")
    NAN = float("nan")

    final_metrics = {
        "val_loss_x": val_loss_x,
        "val_hsic": val_hsic,
        "val_x_mae": val_x_mae,
        "val_x_rmse": val_x_rmse,
        "val_x_r2": val_x_r2,
        "val_loss": val_loss,
        "val_notears": _lv("val_notears"),
        "adaptive_phase": _lv("adaptive_phase"),
        "adaptive_cycle": _lv("adaptive_cycle"),
        # No test_* columns in metrics.csv -> mirror the val proxy so the
        # notebook's final table renders (interrupted-run recovery).
        "test_loss_x": val_loss_x,
        "test_hsic": val_hsic,
        "test_x_mae": val_x_mae,
        "test_x_rmse": val_x_rmse,
        "test_x_r2": val_x_r2,
        "test_loss": val_loss,
        "test_notears": _lv("val_notears"),
        # Timing not logged in the CSV.
        "total_training_time": NAN,
        "avg_time_per_epoch": NAN,
        "_reconstructed": True,
        "_reconstruction_note": (
            "Rebuilt from metrics.csv after an interrupted run: test_* mirror "
            "val_* (no test columns logged), timing is NaN, and per-transition "
            "score margins are null (not recoverable without loading checkpoints)."
        ),
    }

    summary = {
        "experiment_tag": "NA",
        "total_epoch_budget": (int(total_budget) if total_budget is not None else None),
        "start_phase": start_phase,
        "monitor": monitor,
        "cross_fitting": (data_split_ratio is not None),
        "data_split_ratio": (float(data_split_ratio) if data_split_ratio is not None else None),
        "n_train_reconstruct": None,
        "n_train_structure": None,
        "n_transitions": len(transitions),
        "n_cycles": n_cycles,
        "final_metrics": final_metrics,
        "transitions": transitions,
        "_reconstructed": True,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment_dir", help="Path to the experiment folder (contains stage_checkpoints/ and k_*/).")
    ap.add_argument("--out", default=None, help="Output path (default: <experiment_dir>/adaptive_training_summary.json).")
    args = ap.parse_args()

    summary = rebuild_summary(args.experiment_dir)
    out_path = args.out or join(args.experiment_dir, "adaptive_training_summary.json")
    with open(out_path, "w") as fh:
        # allow_nan=True (default) keeps NaN readable by Python's json.load.
        json.dump(summary, fh, indent=2)

    print(f"Wrote {out_path}")
    print(f"  transitions : {summary['n_transitions']}")
    print(f"  cycles      : {summary['n_cycles']}")
    fm = summary["final_metrics"]
    print(f"  final val_x_r2 : {fm['val_x_r2']}")
    print(f"  final val_x_mae: {fm['val_x_mae']}")


if __name__ == "__main__":
    main()
