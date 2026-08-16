"""
Anneal Reconstruction: final reconstruction-only phase for adaptive SVFA runs.

Motivation
----------
The adaptive trainer alternates reconstruct/structure phases and cross-fits on
50% of the train split per phase (``data_split_ratio: 0.5``).  The ATE benchmark
baselines (vanilla, cheater) instead train on 100% of the train split with a
pure reconstruction objective, so the published SVFA numbers carry a
reconstruction gap (e.g. test R2-macro ~0.71 vs ~0.93 on ds_scm2_continuous).

This script tests whether the gap closes when the *learned structure* is kept
and only the predictor is re-annealed:

1. Load a run's ``k_0/checkpoints/best_causal_checkpoint.ckpt`` (best HSIC).
2. Freeze the structural parameter group (``_structural_params``); keep the
   reconstruction group (``_reconstruction_params``) trainable.
3. Anneal with a pure MSE reconstruction loss on the FULL train split (both
   cross-fit halves merged; val/test stay held out), cosine LR decay to 0.
4. Write ``<run_dir>/anneal_recon/`` with per-epoch metrics, curves, a summary
   JSON, and a pseudo-experiment (``pseudo_exp/``) holding the annealed
   checkpoint so the standard ATE evaluation can run on it unchanged.

The comparison this enables: cheater = handed the TRUE structure, vanilla = no
structure at all, svfa-annealed = its LEARNED structure, frozen, with the same
reconstruction-only full-data training the baselines enjoyed.

Usage
-----
Single run:
    python scripts/anneal_reconstruction.py --run_dir <path/to/run>

Whole SVFA sweep (all datasets x seeds):
    python scripts/anneal_reconstruction.py --sweep_root experiments/7_PUBLISH/ATE/results/svfa_10687808
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from causaliT.training.trainer import (
    get_dataloader,
    _make_fold_splits,
    create_model_instance,
    resolve_seeds,
)
from causaliT.training.config_utils import populate_seq_lengths_from_dataset
from causaliT.training.experiment_control import update_config


# =============================================================================
# Run loading
# =============================================================================

def _resolve_run_data_dir(run_dir: Path, dataset: str) -> Path:
    """Find the ``datasets/`` root containing ``<dataset>/ds.npz`` by walking up.

    The run config carries a stale cluster ``data_root``; the relocated results
    tree keeps the datasets at ``groups/<dataset>/datasets/<dataset>``.
    """
    run_dir = run_dir.resolve()
    for anc in [run_dir] + list(run_dir.parents):
        cand = anc / "datasets" / dataset / "ds.npz"
        if cand.is_file():
            return cand.parent.parent
    raise FileNotFoundError(
        f"Could not locate datasets/{dataset}/ds.npz above {run_dir}"
    )


def _apply_overrides(config, overrides):
    """Apply dotted-path key=value overrides to the config (OmegaConf).

    Values are parsed with yaml.safe_load, so "1e-3" -> float, "true" -> bool,
    "separate" -> str.  Example: --override experiment.value_structure_injection=separate
    """
    import yaml
    for kv in overrides or []:
        key, sep, raw = kv.partition("=")
        if not key or not sep:
            raise ValueError(f"bad override {kv!r}, expected key=value")
        OmegaConf.update(config, key, yaml.safe_load(raw), merge=False)
    return config


def _load_run(run_dir: Path, overrides=None):
    """Return (config, data_dir, best_causal_ckpt, kfold_summary) for a run."""
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"{run_dir}: no config.yaml")
    config = OmegaConf.load(config_path)
    # Resolve the multiplier-derived fields (experiment.d_ff / d_qk) - the
    # sweeper writes them as null and the model constructor needs numbers.
    config = update_config(config)
    config = _apply_overrides(config, overrides)
    dataset = config["data"]["dataset"]
    data_dir = _resolve_run_data_dir(run_dir, dataset)

    ckpt = run_dir / "k_0" / "checkpoints" / "best_causal_checkpoint.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{run_dir}: missing {ckpt.name}")

    kfold_path = run_dir / "kfold_summary.json"
    kfold_summary = json.load(open(kfold_path)) if kfold_path.is_file() else None
    return config, data_dir, ckpt, kfold_summary


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def _eval_split(model, dl, device) -> dict:
    """Reconstruction metrics on a split: MSE, MAE, pooled R2, macro R2."""
    model.eval()
    val_idx = model.val_idx
    n = 0
    abs_sum = 0.0
    sse = None          # per-node sum of squared errors
    y_sum = None        # per-node target sum
    y2_sum = None       # per-node target squared sum
    for batch in dl:
        S, X = batch[0].to(device), batch[1].to(device)
        pred = model.forward(data_source=S, data_intermediate=X)[0].squeeze(-1)
        targ = torch.nan_to_num(X[:, :, val_idx])
        if model.homogeneous_nodes:
            targ = torch.cat([S[:, :, val_idx], targ], dim=1)
        err = pred - targ                                    # (B, L)
        b = err.shape[0]
        n += b
        abs_sum += err.abs().sum().item()
        sse = err.pow(2).sum(dim=0) if sse is None else sse + err.pow(2).sum(dim=0)
        y_sum = targ.sum(dim=0) if y_sum is None else y_sum + targ.sum(dim=0)
        y2_sum = targ.pow(2).sum(dim=0) if y2_sum is None else y2_sum + targ.pow(2).sum(dim=0)

    mse = float(sse.sum().item() / max(n * sse.numel(), 1))
    mae = abs_sum / max(n * sse.numel(), 1)
    sst = y2_sum - y_sum.pow(2) / max(n, 1)                  # per-node SST
    r2_node = 1.0 - sse / sst.clamp_min(1e-12)
    r2_pooled = 1.0 - sse.sum() / sst.sum().clamp_min(1e-12)
    return {
        "loss_x": mse,
        "x_mae": mae,
        "x_r2": float(r2_pooled.item()),
        "x_r2_macro": float(r2_node.mean().item()),
    }


# =============================================================================
# Annealing
# =============================================================================

def _override_slug(overrides) -> str:
    """Short filesystem-safe suffix encoding the overrides ("" when none)."""
    parts = []
    for kv in overrides or []:
        key, _, raw = kv.partition("=")
        parts.append(f"{key.split('.')[-1]}-{raw}")
    return ("_" + "_".join(parts)) if parts else ""


def anneal_run(
    run_dir: Path,
    epochs: int = 300,
    patience: int = 40,
    min_delta: float = 1e-5,
    lr: float = None,
    mode: str = "frozen",
    overrides=None,
    eval_ate: bool = True,
    eval_dag: bool = True,
    overwrite: bool = False,
    device_str: str = "auto",
) -> dict:
    """Anneal one run's best-causal checkpoint on the full train split.

    Modes:
        frozen:     structural params frozen, pure MSE (reconstruction-only).
        joint:      ALL params trainable, pure MSE (structure may drift).
        joint_full: ALL params trainable, full original loss (MSE + HSIC + L0
                    + NOTEARS + query-norm) - the training objective, joint.

    ``overrides`` is a list of dotted-path key=value strings applied to the run
    config before the model is built (e.g. value_structure_injection=separate).
    New modules introduced by an override are absent from the checkpoint and
    start randomly initialised (state_dict loads with strict=False).
    """
    assert mode in ("frozen", "joint", "joint_full"), mode
    run_dir = Path(run_dir)
    slug = _override_slug(overrides)
    out_dir = run_dir / (
        ("anneal_recon" if mode == "frozen" else f"anneal_{mode}") + slug
    )
    summary_path = out_dir / "anneal_summary.json"
    if summary_path.is_file() and not overwrite:
        print(f"  [skip] {run_dir.name}: {out_dir.name} already exists")
        return json.load(open(summary_path))

    config, data_dir, ckpt_path, kfold_summary = _load_run(run_dir, overrides)
    dataset = config["data"]["dataset"]
    seed, data_seed = resolve_seeds(config)
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Resolve seq lengths / auto knobs (init_edge_offset) before building.
    config = populate_seq_lengths_from_dataset(config, str(data_dir))

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    if device.type == "cpu" and "device" in config["model"]["kwargs"]:
        config["model"]["kwargs"]["device"] = "cpu"

    # --- Data: FULL train split (both cross-fit halves), same val/test -------
    dm = get_dataloader(config, str(data_dir), cluster=False, seed=data_seed)
    dm.prepare_data()
    fold_splits, test_idx, train_val_idx = _make_fold_splits(
        config, dm, data_seed, data_dir=str(data_dir)
    )
    train_local_idx, val_local_idx = fold_splits[0]
    dm.update_idx(train_idx=train_local_idx, val_idx=val_local_idx, test_idx=test_idx)

    # --- Model: build, load best-causal weights, freeze structure ------------
    model = create_model_instance(config, str(data_dir))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] state_dict mismatch: missing={missing[:3]} "
              f"unexpected={unexpected[:3]}")
    # Raw state_dict load bypasses on_load_checkpoint: re-arm the centroid
    # latch manually so the first batch does NOT re-initialise the learned
    # X query embedding at the key centroid.
    if hasattr(model, "_query_centroid_init_done"):
        # setattr (not direct assignment): nn.Module.__setattr__ is typed for
        # Tensor/Module values only.
        setattr(model, "_query_centroid_init_done", True)

    struct_params = getattr(model, "_structural_params", None)
    recon_params = getattr(model, "_reconstruction_params", None)
    if struct_params is None or recon_params is None:
        raise RuntimeError(
            "Run was not trained with use_gradient_routing=True; cannot freeze "
            "the structural group."
        )
    if mode == "frozen":
        for p in struct_params:
            p.requires_grad_(False)
        for p in recon_params:
            p.requires_grad_(True)
        train_params = recon_params
    else:
        # joint / joint_full: every parameter trains.
        for p in model.parameters():
            p.requires_grad_(True)
        train_params = list(model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)

    # joint_full trains on the ORIGINAL total loss via model._step, which calls
    # self.log(...) - mute it (no Lightning trainer is attached in this loop).
    if mode == "joint_full":
        model.log = lambda *a, **k: None

    # --- Optimizer: fresh AdamW + cosine decay to 0 (the "anneal") -----------
    lr = float(lr if lr is not None else config["training"].get("lr", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=0.0
    )

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    print(f"  {run_dir.name}: dataset={dataset} seed={seed} "
          f"train={len(train_local_idx)} val={len(val_local_idx)} "
          f"test={len(test_idx) if test_idx is not None else 0} | "
          f"frozen={n_frozen:,d} trainable={n_train:,d} | lr={lr} epochs<={epochs}")

    # --- Loop -----------------------------------------------------------------
    val_idx = model.val_idx
    history = []
    best = {"val_x_mae": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        if mode == "joint_full":
            # Advance the fan-in squeeze clock (no-op when fanin_prior unset).
            model.fanin_schedule.in_structure_phase = True
            model.fanin_schedule.on_epoch_start(model.model)
        train_se, train_n = 0.0, 0
        for batch in train_dl:
            S, X = batch[0].to(device), batch[1].to(device)
            if mode == "joint_full":
                # Original total loss: MSE + HSIC + L0 + NOTEARS + query-norm.
                total_loss, pred_x, _ = model._step(batch=(S, X), stage="train")
                opt.zero_grad()
                total_loss.backward()
                opt.step()
                loss_x = float(model._last_loss_components["loss_recon"].detach())
                train_se += loss_x * pred_x.numel()
                train_n += pred_x.numel()
            else:
                pred = model.forward(data_source=S, data_intermediate=X)[0].squeeze(-1)
                targ = torch.nan_to_num(X[:, :, val_idx])
                if model.homogeneous_nodes:
                    targ = torch.cat([S[:, :, val_idx], targ], dim=1)
                loss = F.mse_loss(pred, targ)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_se += loss.item() * pred.numel()
                train_n += pred.numel()
        sched.step()

        val_m = _eval_split(model, val_dl, device)
        row = {
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss_x": train_se / max(train_n, 1),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)

        if val_m["x_mae"] < best["val_x_mae"] - min_delta:
            best = {
                "val_x_mae": val_m["x_mae"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"    ep {epoch:4d} | train_mse {row['train_loss_x']:.3e} | "
                  f"val_mse {val_m['loss_x']:.3e} | val_mae {val_m['x_mae']:.4f} | "
                  f"val_r2_macro {val_m['x_r2_macro']:.4f}")
        if bad_epochs >= patience:
            print(f"    early stop at epoch {epoch} (no val_x_mae improvement "
                  f"for {patience} epochs)")
            break

    # --- Restore best weights, final val/test metrics -------------------------
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    val_m = _eval_split(model, val_dl, device)
    test_m = _eval_split(model, test_dl, device) if test_dl is not None else {}
    elapsed = time.time() - t0

    # --- Artifacts -------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(out_dir / "anneal_metrics.csv", index=False)

    before = {}
    if kfold_summary is not None:
        m = kfold_summary.get("fold_results", {}).get("0", {}).get("metrics", {})
        before = {k: m.get(k) for k in (
            "val_loss_x", "val_x_mae", "val_x_r2", "val_x_r2_macro",
            "test_x_mae", "test_x_r2", "test_x_r2_macro")}

    summary = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "mode": mode,
        "model_seed": seed,
        "data_seed": data_seed,
        "checkpoint": ckpt_path.name,
        "n_train": int(len(train_local_idx)),
        "n_val": int(len(val_local_idx)),
        "n_test": int(len(test_idx)) if test_idx is not None else 0,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs_run": int(len(history)),
        "best_epoch": int(best["epoch"]),
        "elapsed_sec": elapsed,
        "before": before,
        "after": {
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        },
    }

    # Curves: val R2-macro and val MSE vs the run's original final values.
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True)
    axes[0].plot(df_hist["epoch"], df_hist["val_x_r2_macro"], lw=1.2)
    if before.get("val_x_r2_macro") is not None:
        axes[0].axhline(before["val_x_r2_macro"], color="k", ls="--", lw=0.9,
                        label="original final")
        axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("val R2 macro")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].plot(df_hist["epoch"], df_hist["val_loss_x"], lw=1.2)
    if before.get("val_loss_x") is not None:
        axes[1].axhline(before["val_loss_x"], color="k", ls="--", lw=0.9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("val MSE (log)")
    axes[1].set_xlabel("anneal epoch")
    axes[0].set_title(f"{run_dir.name} - recon-only anneal of best_causal",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "anneal_curves.png", dpi=200)
    plt.close(fig)

    # --- Pseudo-experiment for the standard eval suite -------------------------
    pseudo = out_dir / "pseudo_exp"
    pseudo_ckpt_dir = pseudo / "k_0" / "checkpoints"
    pseudo_ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Patch the annealed weights into the ORIGINAL checkpoint so that
    # load_from_checkpoint keeps the saved hyperparameters.
    patched = {k: v for k, v in ckpt.items() if k != "state_dict"}
    patched["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    patched.pop("optimizer_states", None)
    patched.pop("lr_schedulers", None)
    patched["epoch"] = 9999
    torch.save(patched, pseudo_ckpt_dir / "epoch=9999-annealed.ckpt")
    # Point the pseudo-experiment's data_root at the resolved local datasets
    # dir (the run config carries a stale cluster path; the DAG evaluation
    # resolves the data root from the config).
    config["data"]["data_root"] = str(data_dir)
    OmegaConf.save(config, str(pseudo / "config.yaml"), resolve=True)
    # Minimal kfold summary so the pseudo-experiment is notebook-ingestible.
    with open(pseudo / "kfold_summary.json", "w") as fh:
        json.dump({
            "total_folds": 1,
            "completed_folds": 1,
            "fold_results": {"0": {"metrics": {
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "trainable_params": n_train + n_frozen,
                "total_training_time": elapsed,
            }, "best_checkpoint_path": str(pseudo_ckpt_dir / "epoch=9999-annealed.ckpt"),
                "fold_dir": str(pseudo / "k_0")}},
        }, fh, indent=2)

    # --- ATE re-evaluation on the annealed checkpoint --------------------------
    if eval_ate:
        try:
            from causaliT.evaluation.eval_funs.eval_interventions import eval_ate_mc
            print("    running ATE eval on the annealed checkpoint...")
            eval_ate_mc(experiment=str(pseudo), datadir_path=str(data_dir))
            summary["ate"] = _ate_before_after(run_dir, pseudo)
        except Exception as exc:
            print(f"    [warn] ATE eval failed: {exc}")
            summary["ate_error"] = str(exc)

    # --- DAG drift: SHD of the annealed model vs the original run ------------
    if eval_dag:
        try:
            from causaliT.evaluation.eval_funs.eval_attention import eval_attention_scores
            print("    running DAG eval on the annealed checkpoint...")
            eval_attention_scores(str(pseudo), show_plots=False)
            summary["dag"] = _dag_before_after(run_dir, pseudo)
        except Exception as exc:
            print(f"    [warn] DAG eval failed: {exc}")
            summary["dag_error"] = str(exc)

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    b_r2 = before.get("test_x_r2_macro")
    a_r2 = summary["after"].get("test_x_r2_macro")
    print(f"  done in {elapsed:.0f}s | test R2-macro: "
          f"{b_r2 if b_r2 is not None else float('nan'):.4f} -> {a_r2:.4f} | "
          f"val MSE: {before.get('val_loss_x', float('nan')):.3e} -> "
          f"{val_m['loss_x']:.3e}")
    ate = summary.get("ate") or {}
    if ate.get("before") and ate.get("after"):
        print(f"  ATE mean abs err: {ate['before']['mean_abs_error']:.4f} -> "
              f"{ate['after']['mean_abs_error']:.4f}")
    dag = summary.get("dag") or {}
    if dag.get("before") and dag.get("after"):
        print(f"  SHD cross/self: "
              f"{dag['before'].get('shd_cross')}/{dag['before'].get('shd_self')} -> "
              f"{dag['after'].get('shd_cross')}/{dag['after'].get('shd_self')}")
    return summary


# =============================================================================
# ATE before/after helper
# =============================================================================

def _ate_summary(ate_csv: Path) -> dict:
    """Compact ATE summary: mean abs error overall and by ID/OOD regime."""
    df = pd.read_csv(ate_csv)
    df = df.dropna(subset=["abs_error"])
    mag = df["intervention"].astype(str).str.extract(r"=(-?[\d.]+)")[0].astype(float).abs()
    out = {
        "mean_abs_error": float(df["abs_error"].mean()),
        "mean_pct_error": float((100 * df["abs_error"] / mag.replace(0.0, np.nan)).mean()),
    }
    for tag, mask in [("ID", mag <= 1.0), ("OOD", mag > 1.0)]:
        sub = df[mask]
        out[f"mean_abs_error_{tag}"] = float(sub["abs_error"].mean()) if len(sub) else None
    return out


def _ate_before_after(run_dir: Path, pseudo: Path) -> dict:
    before_csv = run_dir / "eval" / "eval_ate_mc" / "files" / "ate_metrics_mc.csv"
    after_csv = pseudo / "eval" / "eval_ate_mc" / "files" / "ate_metrics_mc.csv"
    out = {}
    out["before"] = _ate_summary(before_csv) if before_csv.is_file() else None
    out["after"] = _ate_summary(after_csv) if after_csv.is_file() else None
    return out


def _dag_summary(dag_json: Path) -> dict:
    """Compact DAG summary: standard SHD on the cross (S->X) and self (X->X) blocks."""
    j = json.load(open(dag_json))
    return {
        "shd_cross": (j.get("standard_shd_cross") or {}).get("mean"),
        "shd_self": (j.get("standard_shd_self") or {}).get("mean"),
    }


def _dag_before_after(run_dir: Path, pseudo: Path) -> dict:
    before_json = run_dir / "eval" / "eval_attention_scores" / "files" / "dag_metrics.json"
    after_json = pseudo / "eval" / "eval_attention_scores" / "files" / "dag_metrics.json"
    out = {}
    out["before"] = _dag_summary(before_json) if before_json.is_file() else None
    out["after"] = _dag_summary(after_json) if after_json.is_file() else None
    return out


# =============================================================================
# CLI
# =============================================================================

def _iter_sweep_runs(sweep_root: Path):
    """Yield run dirs under a <model>_<jobid> sweep root."""
    yield from sorted(
        p for p in sweep_root.glob("groups/*/sweeper/runs/combinations/*")
        if p.is_dir()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", type=str, help="Single run directory.")
    src.add_argument("--sweep_root", type=str,
                     help="Sweep root (<model>_<jobid>) - anneal every run below it.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=None,
                    help="Anneal LR (default: the run config's training.lr).")
    ap.add_argument("--mode", choices=["frozen", "joint", "joint_full"],
                    default="frozen",
                    help="frozen: structure frozen, pure MSE. joint: all params, "
                         "pure MSE. joint_full: all params, full original loss.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Sweep mode only: restrict to runs of this dataset "
                         "(e.g. ds_scm2_continuous).")
    ap.add_argument("--override", action="append", default=None,
                    help="Dotted-path config override, repeatable. E.g. "
                         "--override experiment.value_structure_injection=separate")
    ap.add_argument("--no_ate", action="store_true",
                    help="Skip the ATE re-evaluation on the annealed checkpoint.")
    ap.add_argument("--no_dag", action="store_true",
                    help="Skip the DAG-drift evaluation on the annealed checkpoint.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run even if the mode's anneal folder already exists.")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.run_dir:
        anneal_run(Path(args.run_dir), epochs=args.epochs, patience=args.patience,
                   lr=args.lr, mode=args.mode, overrides=args.override,
                   eval_ate=not args.no_ate, eval_dag=not args.no_dag,
                   overwrite=args.overwrite, device_str=args.device)
        return

    # Sweep mode: anneal every run, then write a compact before/after table.
    sweep_root = Path(args.sweep_root)
    rows = []
    for run_dir in _iter_sweep_runs(sweep_root):
        if args.dataset and args.dataset not in run_dir.parts:
            continue
        try:
            s = anneal_run(run_dir, epochs=args.epochs, patience=args.patience,
                           lr=args.lr, mode=args.mode, overrides=args.override,
                           eval_ate=not args.no_ate, eval_dag=not args.no_dag,
                           overwrite=args.overwrite, device_str=args.device)
        except Exception as exc:
            print(f"  [FAIL] {run_dir.name}: {exc}")
            continue
        row = {
            "run": run_dir.name,
            "dataset": s["dataset"],
            "model_seed": s["model_seed"],
            "test_r2_macro_before": (s.get("before") or {}).get("test_x_r2_macro"),
            "test_r2_macro_after": s["after"].get("test_x_r2_macro"),
            "val_mse_before": (s.get("before") or {}).get("val_loss_x"),
            "val_mse_after": s["after"].get("val_loss_x"),
            "ate_abs_before": ((s.get("ate") or {}).get("before") or {}).get("mean_abs_error"),
            "ate_abs_after": ((s.get("ate") or {}).get("after") or {}).get("mean_abs_error"),
            "ate_pct_before": ((s.get("ate") or {}).get("before") or {}).get("mean_pct_error"),
            "ate_pct_after": ((s.get("ate") or {}).get("after") or {}).get("mean_pct_error"),
            "shd_cross_before": ((s.get("dag") or {}).get("before") or {}).get("shd_cross"),
            "shd_cross_after": ((s.get("dag") or {}).get("after") or {}).get("shd_cross"),
            "shd_self_before": ((s.get("dag") or {}).get("before") or {}).get("shd_self"),
            "shd_self_after": ((s.get("dag") or {}).get("after") or {}).get("shd_self"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = sweep_root / (
        ("anneal_summary.csv" if args.mode == "frozen"
         else f"anneal_summary_{args.mode}.csv").replace(
            ".csv", f"{_override_slug(args.override)}.csv")
    )
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    print(f"\n===== ANNEAL SUMMARY ({args.mode}) =====")
    print(df.to_string(index=False))
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
