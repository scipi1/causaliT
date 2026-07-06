"""Temporary diagnostic: check for eval-leak kink in HC stage_1/stage_2 HSIC curves."""
import os
import pandas as pd

base = r"experiments/2_ARCH_STUDY/BKD/BKD_HC_scm3c_5827447/sweeper/runs/combinations"
combo = "BKD_HC_scm3c_combo_orthogonal_fixed_False_BKD_0.0_seed_1"


def load_stage(stage):
    p = os.path.join(base, combo, "anm_stages", stage, "k_0",
                     "logs", "csv", "version_0", "metrics.csv")
    if not os.path.exists(p):
        print("MISSING", p)
        return None
    df = pd.read_csv(p)
    tc = next((c for c in df.columns if c.startswith("train_")), None)
    df = df[df[tc].notna()].copy() if tc else df
    return df


for stage in ["00_stage_1", "01_stage_2"]:
    df = load_stage(stage)
    if df is None:
        continue
    print("\n===", stage)
    lo, hi = (5, 40) if stage == "00_stage_1" else (100, 160)

    cols = [c for c in ["train_hsic_reg", "train_loss_x", "train_l0_penalty",
                        "train_score_sparse"] if c in df.columns]
    sub = df[["epoch"] + cols].copy()
    print(sub[(sub["epoch"] >= lo) & (sub["epoch"] <= hi)].to_string(index=False))


