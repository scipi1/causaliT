"""Generate the HSIC_OPT_5_BILEVEL experiment arms from BILEVEL_GATE/gate_d20.

Arms (all: centroid_commit OFF, hsic_cross_fit ON — the continuous bi-level
structural gradient is isolated, and both arms evaluate HSIC on fold B):

* bilevel_first_d20          structural_grad: hsic           (first-order A/B control)
* bilevel_unrolled_d20       structural_grad: hsic_unrolled  (DARTS 2nd-order)
* bilevel_unrolled_d20_smoke short crash-hunting variant of the unrolled arm
                             (epoch budget from gate_d20_smoke).

Run:  python scripts/_gen_hsic_opt5_bilevel.py
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "experiments/6_INVESTIGATIONS/BILEVEL_GATE/gate_d20/config.yaml"
SMOKE = REPO / "experiments/6_INVESTIGATIONS/BILEVEL_GATE/gate_d20_smoke/config.yaml"
OUT = REPO / "experiments/6_INVESTIGATIONS/HSIC_OPT_5_BILEVEL"

HEADER = """# ===========================================================================
# HSIC_OPT_5_BILEVEL / {name}
# ===========================================================================
# {desc}
# ===========================================================================
#"""


def _patch_common(text: str, structural_grad: str) -> str:
    """Shared edits vs gate_d20: commit off, cross-fit on, structural_grad."""
    # 1. Disable the centroid-commit machinery (isolate the continuous grad).
    old = "  centroid_commit:\n    enabled: true"
    new = ("  centroid_commit:\n    enabled: false         # HSIC_OPT_5: continuous gradient, no commits")
    assert old in text
    text = text.replace(old, new)
    # ... and the gate cannot stay on without the controller (forecaster
    # raises "bilevel_gate requires centroid_commit.enabled: true").
    old = "    bilevel_gate:\n      enabled: true"
    new = "    bilevel_gate:\n      enabled: false        # no commits -> no gate"
    assert old in text
    text = text.replace(old, new)

    # 2. Structural gradient source + HSIC cross-fitting + unrolled block,
    #    inserted after the gradient_surgery key.
    anchor = "  gradient_surgery: false\n"
    assert anchor in text
    bilevel = (
        anchor
        + f'  structural_grad: "{structural_grad}"\n'
        + "  # Both arms cross-fit: fold A drives the MSE, fold B the HSIC\n"
        + "  # (and, for hsic_unrolled, the virtual refit + FD Hessian too).\n"
        + "  hsic_cross_fit: true\n"
        + "  hsic_cross_fit_ratio: 0.5\n"
        + "  unrolled:\n"
        + "    inner_lr: null            # virtual refit step = training.lr\n"
        + "    fd_epsilon: 0.01          # finite-difference mixed-Hessian scale\n"
        + "    every: 1\n"
    )
    text = text.replace(anchor, bilevel)
    return text


def _retitle(text: str, name: str, desc: str) -> str:
    lines = text.split("\n")
    end = next(i for i, l in enumerate(lines[1:], start=1)
               if not l.startswith("#"))
    return HEADER.format(name=name, desc=desc) + "\n" + "\n".join(lines[end:])


def main() -> None:
    base = BASE.read_text(encoding="utf-8")

    first = _patch_common(base, "hsic")
    first = _retitle(
        first, "bilevel_first_d20",
        "A/B CONTROL: identical to BILEVEL_GATE/gate_d20 except (a) the\n"
        "# centroid-commit dynamics are OFF and (b) HSIC cross-fitting is ON\n"
        "# (fold A = MSE, fold B = HSIC).  The structural gradient is the\n"
        "# first-order frozen-theta_R HSIC gradient (structural_grad: hsic).")

    unrolled = _patch_common(base, "hsic_unrolled")
    unrolled = _retitle(
        unrolled, "bilevel_unrolled_d20",
        "DARTS second-order STRUCTURAL gradient (docs/ideas/BILEVEL_CENTROID_\n"
        "# COMMIT.md, Tier-1 math lifted to the optimizer): virtual 1-step\n"
        "# refit of theta_R on fold B + HSIC at the refit point - FD mixed-\n"
        "# Hessian correction (eta = training.lr, eps = 0.01, every step).\n"
        "# Commits OFF, cross-fit ON — paired with bilevel_first_d20.")

    smoke = _patch_common(SMOKE.read_text(encoding="utf-8"), "hsic_unrolled")
    smoke = _retitle(
        smoke, "bilevel_unrolled_d20_smoke",
        "SMOKE TEST of bilevel_unrolled_d20 (crash-hunting, NOT science):\n"
        "# warmup 20 epochs, structure ~100 epochs (from gate_d20_smoke), so\n"
        "# the warmup->structure transition and the first bi-level steps run\n"
        "# within minutes on the cluster.")

    for name, text in (("bilevel_first_d20", first),
                       ("bilevel_unrolled_d20", unrolled),
                       ("bilevel_unrolled_d20_smoke", smoke)):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.yaml").write_text(text, encoding="utf-8")
        print(f"wrote {d / 'config.yaml'}")


if __name__ == "__main__":
    main()
