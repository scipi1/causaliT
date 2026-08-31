"""Create the warmcos arm config from the sgd_frozenbw base."""
import shutil
from pathlib import Path

SRC = Path("experiments/6_INVESTIGATIONS/HSIC_OPT_2/bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw")
DST = Path("experiments/6_INVESTIGATIONS/HSIC_OPT_2/bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw_warmcos")

if not DST.exists():
    shutil.copytree(SRC, DST)

p = DST / "config.yaml"
c = p.read_text(encoding="utf-8")


def rep(old, new, label):
    global c
    assert c.count(old) == 1, f"{label}: {c.count(old)}x"
    c = c.replace(old, new)


rep("bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw\n",
    "bkd_warmup_06_global_nonorm_hsicbkd_d20_base_sgd_frozenbw_warmcos\n", "title")

rep(
    "#   L0=0, NOTEARS kappa=0, PCGrad off, nodewise off, centroid-commit off.",
    "#   L0=0, NOTEARS kappa=0, PCGrad off, nodewise off, centroid-commit off.\n"
    "#   WARMCOS arm (Arm 1): a dedicated WARMUP phase runs once before the\n"
    "#   alternating schedule (adaptive_training.warmup, start_phase=warmup).\n"
    "#   Reconstruction-only (structure frozen) under a COSINE BKD curriculum\n"
    "#   oscillating in [0.1, 0.9] with a decaying envelope, landing exactly on\n"
    "#   the structure phase's fixed BKD 0.05; the gates are FORCED OPEN with a\n"
    "#   BKD-coupled constant c(t) = c_end + (1-c_end)(p-p_end)/(p_max-p_end)\n"
    "#   (~1 at high dropout, c_end = the frozen learned-gate init value at\n"
    "#   p_end), so the decoder learns to fit under EVERY key-subset regime\n"
    "#   (sparse+large gates, dense+small gates) and the warmup->structure\n"
    "#   switch is continuous by construction.  Motivation: the structure phase\n"
    "#   should select freely without paying the co-adaptation misfit price\n"
    "#   (teleportation tests: true parents raise HSIC/MSE against a stale\n"
    "#   decoder).  Everything else identical to ..._sgd_frozenbw.",
    "header",
)

rep("  start_phase: reconstruct\n", "  start_phase: warmup\n", "start_phase")

rep(
    "  reconstruct:\n",
    "  warmup:\n"
    "    enabled: true\n"
    "    max_epochs: 800\n"
    "    # Cosine BKD curriculum: oscillation center 0.5, amplitude 0.4 (spans\n"
    "    # 0.1..0.9 early), 6 full periods, landing on 0.05 = the structure\n"
    "    # phase's fixed BKD.  Anneal clock in batches: ~2500 samples on the\n"
    "    # reconstruct split / batch 1024 ~ 3 batches/epoch * 800 epochs.\n"
    "    batch_key_dropout: 0.5\n"
    "    batch_key_dropout_final: 0.05\n"
    "    batch_key_dropout_annealing_batches: 2400\n"
    "    bkd_schedule: cosine\n"
    "    bkd_amplitude: 0.4\n"
    "    bkd_cycles: 6\n"
    "    open_gate_bkd_coupled: true\n"
    "  reconstruct:\n",
    "warmup block",
)

# Structure phase: fixed dense BKD 0.05 (the warmup landing point).
rep(
    "    batch_key_dropout: 0.6\n"
    "    batch_key_dropout_final: 0.05\n"
    "    batch_key_dropout_annealing_batches: 3000\n"
    "    # Freeze the HSIC kernel bandwidth",
    "    # Fixed dense BKD 0.05 (the warmup curriculum's landing point).\n"
    "    batch_key_dropout: 0.05\n"
    "    batch_key_dropout_final: 0.05\n"
    "    batch_key_dropout_annealing_batches: null\n"
    "    # Freeze the HSIC kernel bandwidth",
    "structure bkd",
)

p.write_text(c, encoding="utf-8")
print("warmcos config created")
