"""Stage-2 generator for `experiments/tests/edge_decisiveness/`.

Once Stage-1 (C1..C5 capacity sweep) has identified the winning capacity arm
(call it ``C*``), this script writes Stage-2 arms that take ``C*`` as their
template and apply a single delta:

* L1..L4 — LR sweep on ``training.structural_lr``.
* S1..S2 — *stability* arms inspired by `experiments/tests/structure_opt`:
  Langevin-style structural gradient noise (S1) and cosine-warm-restarts on
  the structural LR (S2). These probe whether the optimization chaos that
  produced two visibly different ``C3_dmodel_96`` outcomes from identical
  configs can be tamed without changing the LR.

Each arm is emitted **twice**, once per seed in ``--seeds`` (default ``42 43``),
as sibling directories named ``<arm>_seed<seed>``. This lets us measure
seed-to-seed stability per arm with the existing launcher loop.

Decision rule for picking C* (per `experiments/tests/edge_decisiveness/README.md`):
    pick the C-arm with the best ``test_x_mae`` whose
    ``test_self_score_sparse`` did NOT collapse (> 0.05 floor).

Usage
-----
From the repo root:

    python scripts/build_stage2_lr_arms.py --c-star C3_dmodel_96
    # (optional) also include the S3 = noise + restart combination:
    python scripts/build_stage2_lr_arms.py --c-star C3_dmodel_96 --include-s3
    # (optional) more / fewer seeds:
    python scripts/build_stage2_lr_arms.py --c-star C3_dmodel_96 --seeds 42 43 44

Idempotent — overwrites existing ``L*_*_seed*`` and ``S*_*_seed*`` dirs.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
EDGE = ROOT / "experiments" / "tests" / "edge_decisiveness"

# ---------------------------------------------------------------------------
# Stage-2A: structural-lr sweep
# Single delta per arm = ``training.structural_lr``.
# ---------------------------------------------------------------------------
L_ARMS: List[Tuple[str, str, str]] = [
    ("L1_lr_3e-4", "3.0e-4", "reference (== C* structural_lr)"),
    ("L2_lr_1e-3", "1.0e-3", "gentle structural boost"),
    ("L3_lr_3e-3", "3.0e-3", "iter-11 / B4-style aggressive boost"),
    ("L4_lr_1e-2", "1.0e-2", "far-end regime"),
]

# ---------------------------------------------------------------------------
# Stage-2B: stability arms (single delta vs C*, structural_lr untouched).
# Each arm is described by a list of (yaml_key, new_value) edits applied to
# the C*-template body.
# ---------------------------------------------------------------------------
S_ARMS: List[Tuple[str, List[Tuple[str, str]], str]] = [
    (
        "S1_noise",
        [
            ("structural_gradient_noise", "0.01"),
            ("structural_gradient_noise_decay", "0.995"),
        ],
        "Langevin-style noise injection on structural grads — pushes toward "
        "flatter / more stable minima. Tests whether the seed-to-seed "
        "variance observed in duplicate C3 runs can be reduced.",
    ),
    (
        "S2_restart",
        [
            ("structural_scheduler", "cosine_warm_restarts"),
            (
                "structural_scheduler_kwargs",
                "{T_0: 100, T_mult: 1, eta_min: 1.0e-5}",
            ),
        ],
        "Cosine warm restarts on structural LR — periodic momentum reset to "
        "escape bad basins without changing the average LR.",
    ),
]

S3_ARM: Tuple[str, List[Tuple[str, str]], str] = (
    "S3_noise_restart",
    [
        ("structural_gradient_noise", "0.01"),
        ("structural_gradient_noise_decay", "0.995"),
        ("structural_scheduler", "cosine_warm_restarts"),
        (
            "structural_scheduler_kwargs",
            "{T_0: 100, T_mult: 1, eta_min: 1.0e-5}",
        ),
    ],
    "S1 + S2 combined: noise injection AND warm restarts. Run only after "
    "S1 / S2 individually have shown promise.",
)


HEADER_LR = """# =============================================================================
# tests / edge_decisiveness / {arm}_seed{seed}   (Stage-2 structural-lr sweep)
# Template: {c_star}    Single delta:
#   training.structural_lr : <C* default> -> {lr}
#   training.seed          : 42            -> {seed}
# Hypothesis: {hyp}
# =============================================================================
"""

HEADER_STAB = """# =============================================================================
# tests / edge_decisiveness / {arm}_seed{seed}   (Stage-2 stability arm)
# Template: {c_star}    Single delta vs C*:
{delta_block}#   training.seed          : 42 -> {seed}
# Hypothesis: {hyp}
# =============================================================================
"""


def _strip_header(s: str) -> str:
    idx = s.find("\nexperiment:")
    if idx < 0:
        raise RuntimeError("Could not locate 'experiment:' top-level key in template.")
    return s[idx + 1 :]


def _replace_yaml_scalar(body: str, key: str, new_value: str) -> str:
    """Replace ``  <key>: <anything>`` with ``  <key>: <new_value>`` (first match)."""
    pattern = rf"^(\s*{re.escape(key)}:\s*).*$"
    new_body, n = re.subn(
        pattern,
        lambda m: f"{m.group(1)}{new_value}",
        body,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise RuntimeError(
            f"Could not find key '{key}' in template body — refusing to silently miss the edit."
        )
    return new_body


def _set_seed(body: str, seed: int) -> str:
    return _replace_yaml_scalar(body, "seed", str(seed))


def _write_lr_arm(
    body: str, arm: str, lr: str, hyp: str, c_star: str, seed: int
) -> None:
    new_body = _replace_yaml_scalar(body, "structural_lr", lr)
    new_body = _set_seed(new_body, seed)
    out_dir = EDGE / f"{arm}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    header = HEADER_LR.format(arm=arm, seed=seed, c_star=c_star, lr=lr, hyp=hyp)
    (out_dir / "config_single_causal_svfa.yaml").write_text(
        header.rstrip() + "\n\n" + new_body, encoding="utf-8"
    )
    print(f"  wrote {out_dir.relative_to(ROOT)}")


def _write_stab_arm(
    body: str,
    arm: str,
    edits: List[Tuple[str, str]],
    hyp: str,
    c_star: str,
    seed: int,
) -> None:
    new_body = body
    for key, val in edits:
        new_body = _replace_yaml_scalar(new_body, key, val)
    new_body = _set_seed(new_body, seed)
    delta_block = "".join(
        f"#   training.{key:<35s}: <C* default> -> {val}\n" for key, val in edits
    )
    out_dir = EDGE / f"{arm}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    header = HEADER_STAB.format(
        arm=arm,
        seed=seed,
        c_star=c_star,
        delta_block=delta_block,
        hyp=hyp,
    )
    (out_dir / "config_single_causal_svfa.yaml").write_text(
        header.rstrip() + "\n\n" + new_body, encoding="utf-8"
    )
    print(f"  wrote {out_dir.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--c-star",
        required=True,
        help="Name of the Stage-1 winner directory (e.g. C3_dmodel_96).",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43],
        help="Seed list — one config dir is emitted per (arm, seed). Default: 42 43.",
    )
    ap.add_argument(
        "--include-s3",
        action="store_true",
        help="Also emit S3_noise_restart (noise + warm restarts combined).",
    )
    ap.add_argument(
        "--skip-lr",
        action="store_true",
        help="Skip the L1..L4 LR-sweep arms (only emit stability arms).",
    )
    ap.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip the S1..S2 stability arms (only emit LR-sweep arms).",
    )
    args = ap.parse_args()

    template = EDGE / args.c_star / "config_single_causal_svfa.yaml"
    if not template.is_file():
        print(f"ERROR: template config not found: {template}", file=sys.stderr)
        return 1

    body = _strip_header(template.read_text(encoding="utf-8"))

    if not args.skip_lr:
        print("Stage-2A: structural-lr sweep")
        for arm, lr, hyp in L_ARMS:
            for seed in args.seeds:
                _write_lr_arm(body, arm, lr, hyp, args.c_star, seed)

    if not args.skip_stability:
        print("Stage-2B: stability arms")
        s_arms = list(S_ARMS)
        if args.include_s3:
            s_arms.append(S3_ARM)
        for arm, edits, hyp in s_arms:
            for seed in args.seeds:
                _write_stab_arm(body, arm, edits, hyp, args.c_star, seed)

    print(f"Stage-2 arms ready (template = {args.c_star}, seeds = {args.seeds}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
