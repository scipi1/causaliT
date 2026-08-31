"""Run node-wise HSIC objective/gradient diagnostics on a checkpoint.

Example:
    python scripts/diagnose_hsic_signal.py \
      --checkpoint experiments/.../k_0/checkpoints/epoch=1359-train_loss=0.00.ckpt \
      --dataset_dir data/random_n20_k4_er_nonlinear_gaussian_s1 \
      --output_dir experiments/6_INVESTIGATIONS/HSIC_OPT_2/diagnostics/d20_epoch1359
"""

import argparse
import json
from pathlib import Path

import torch

from causaliT.training.forecasters.attention_selector_forecaster import (
    AttentionSelectorForecaster,
)
from causaliT.utils.hsic_signal_diagnostics import (
    gradient_probe,
    load_dataset_batches,
    load_ground_truth,
    query_intervention_probe,
)


KERNELS = {
    "single": None,
    "msrbf": [0.5, 1.0, 2.0],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_batches", default=4, type=int)
    parser.add_argument("--n_wrong", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--kernels", default="single,msrbf")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionSelectorForecaster.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.eval()
    batches = load_dataset_batches(
        args.dataset_dir, args.batch_size, args.n_batches, args.seed, device
    )
    gt = load_ground_truth(args.dataset_dir)

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "n_wrong": args.n_wrong,
        "seed": args.seed,
        "kernels": {},
    }

    for name in [x.strip() for x in args.kernels.split(",") if x.strip()]:
        if name not in KERNELS:
            raise ValueError(f"unknown kernel {name!r}; choices: {list(KERNELS)}")
        multipliers = KERNELS[name]
        interventions = query_intervention_probe(
            model, batches, gt, args.n_wrong, args.seed, multipliers
        )
        eval_grads = gradient_probe(
            model, batches, gt, False, multipliers, args.seed
        )
        train_grads = gradient_probe(
            model, batches, gt, True, multipliers, args.seed
        )

        interventions.to_csv(
            args.output_dir / f"{name}_query_interventions.csv", index=False
        )
        eval_grads.to_csv(args.output_dir / f"{name}_gradient_eval.csv", index=False)
        train_grads.to_csv(args.output_dir / f"{name}_gradient_train.csv", index=False)

        summary["kernels"][name] = {
            "bandwidth_multipliers": multipliers,
            "n_nodes_with_parents": int(len(interventions)),
            "frac_true_centroid_improves_node_hsic": float(
                (interventions.true_delta > 0).mean()
            ),
            "frac_wrong_centroid_improves_node_hsic": float(
                (interventions.wrong_delta_mean > 0).mean()
            ),
            "mean_true_delta": float(interventions.true_delta.mean()),
            "mean_wrong_delta": float(interventions.wrong_delta_mean.mean()),
            "eval_mean_parent_alignment": float(
                eval_grads.parent_alignment_mean.mean()
            ),
            "eval_mean_parent_minus_wrong": float(
                eval_grads.parent_minus_wrong_mean.mean()
            ),
            "eval_frac_snr_above_1": float((eval_grads.gradient_snr > 1).mean()),
            "train_mean_parent_alignment": float(
                train_grads.parent_alignment_mean.mean()
            ),
            "train_mean_parent_minus_wrong": float(
                train_grads.parent_minus_wrong_mean.mean()
            ),
            "train_frac_snr_above_1": float((train_grads.gradient_snr > 1).mean()),
        }

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["kernels"], indent=2))
    print(f"wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
