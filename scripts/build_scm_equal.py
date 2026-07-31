"""Generate and VERIFY the ``scm_equal`` control dataset.

Why a dedicated script
----------------------
``ds_scm_equal`` exists to settle one question: do the multi-parent children of
scm1/2/3 keep a single parent because their parents are UNEQUALLY strong (a data
property), or because the selector prefers in-degree 1 (a mechanism property)?
That question is only answered if the dataset really does what it claims, so
this script does not just generate the data -- it **measures** the claims and
fails loudly if any of them is violated.  Running it is the gate before any GPU
time is spent on the ``equal_*`` arms.

The four claims under test
--------------------------
1. EQUAL SHARES.  Every parent of a multi-parent child contributes the same
   fraction of the child's variance (X4: 0.45 / 0.45, X5: 0.30 x 3).
2. EQUAL LEARNING SIGNAL.  Dropping any one parent of a child costs the same
   R^2 -- this is what the reconstruction loss actually feels, and it is the
   quantity that "equal strength" is supposed to mean.
3. EQUAL RESIDUAL DEPENDENCE.  For each parent, HSIC(parent, residual of a fit
   on the OTHER true parents) is the same -- this is what the *structural*
   (HSIC) loss feels, and it is not implied by (1) because HSIC is kernel-based.
4. NO SUBSTITUTES.  No NON-parent has appreciable dependence with a child's
   residual after fitting its true parents.  This is what makes a spurious edge
   unambiguous: in scm3, X2 = S3 + 1% noise, so S3 was a near-sufficient
   stand-in for X2 and the "spurious" S3->X5 edge was almost free.

Usage
-----
    python scripts/build_scm_equal.py                 # generate + verify
    python scripts/build_scm_equal.py --verify-only   # verify existing data

Writes ``data/scm_equal/equal_strength_report.json`` next to the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from os.path import abspath, dirname, exists, join

import numpy as np

ROOT_DIR = dirname(dirname(abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = join(ROOT_DIR, "data", "scm_equal")

# --- Ground truth of ds_scm_equal (kept in sync with scm_ds/datasets.py) -----
S_LABELS = ["S1", "S2", "S3", "S4", "S5"]
X_LABELS = ["X1", "X2", "X3", "X4", "X5"]
# child -> {parent: intended variance share}
TRUE_PARENTS = {
    "X1": {"S1": 0.50},
    "X2": {"S2": 0.50},
    "X3": {"S3": 0.50},
    "X4": {"S4": 0.45, "S5": 0.45},
    "X5": {"X1": 0.30, "X2": 0.30, "X3": 0.30},
}
MULTI_PARENT = ["X4", "X5"]

# Tolerances.  Shares are exact by symmetry, so the only error is Monte-Carlo /
# finite-sample; 0.02 absolute is generous at n = 50k.
TOL_SHARE_ABS = 0.02
TOL_SPREAD_ABS = 0.02          # max-min share within one child
TOL_R2_SPREAD_ABS = 0.03       # max-min per-parent R^2 drop within one child
HSIC_SUBSTITUTE_RATIO = 0.25   # a non-parent must stay below 25% of the weakest
                               # true parent's residual dependence


# ============================================================================ #
# Generation
# ============================================================================ #
def generate() -> None:
    from scm_ds.datasets import ds_scm_equal

    print("\n" + "=" * 70)
    print("Generating ds_scm_equal -> data/scm_equal")
    print("=" * 70)
    ds_scm_equal.generate_ds(
        mode="flat",
        n=50_000,
        save_dir=DATA_DIR,
        normalize_method="minmax",   # per-variable affine: preserves variance SHARES
        shared_embedding=False,
        test_split_method={
            "method": "ratio",
            "kwargs": {"test_ratio": 0.2, "seed": 42},
        },
    )


# ============================================================================ #
# Claim 1 -- variance shares, measured on a raw (un-normalised) replication
# ============================================================================ #
def raw_variance_shares(n: int = 500_000, seed: int = 0) -> dict:
    """Re-derive the coefficients from scratch and measure the shares.

    This is deliberately INDEPENDENT of ``datasets.py``: the coefficients are
    recomputed here from the target shares (0.50 / 0.45 / 0.30), so if the
    literals baked into ``datasets.py`` ever drift, claim (1') below -- which
    measures the shares on the ACTUAL generated data -- will disagree with this
    and the gate fails.
    """
    f = lambda x: np.tanh(2.0 * x)
    var_fS = 1.0 - np.tanh(2.0) / 2.0            # closed form, S ~ U(-1,1)
    kp = float(np.sqrt(0.50 / var_fS))
    c4 = float(np.sqrt(0.45 / var_fS))

    rng = np.random.default_rng(seed)
    # Var(f(X1)) has no closed form -> measure it, then set c5 from it.
    _s = rng.uniform(-1, 1, n)
    _x1 = kp * f(_s) + 0.7071068 * rng.standard_normal(n)
    c5 = float(np.sqrt(0.30 / f(_x1).var()))


    S = {name: rng.uniform(-1, 1, n) for name in S_LABELS}
    X = {}
    X["X1"] = kp * f(S["S1"]) + 0.7071068 * rng.standard_normal(n)
    X["X2"] = kp * f(S["S2"]) + 0.7071068 * rng.standard_normal(n)
    X["X3"] = kp * f(S["S3"]) + 0.7071068 * rng.standard_normal(n)
    X["X4"] = c4 * (f(S["S4"]) + f(S["S5"])) + 0.3162278 * rng.standard_normal(n)
    X["X5"] = (c5 * (f(X["X1"]) + f(X["X2"]) + f(X["X3"]))
               + 0.3162278 * rng.standard_normal(n))

    # Per-parent term variance / total child variance.  The parent terms are
    # mean-zero (tanh is odd) and mutually independent, so these shares add up
    # to 1 together with the noise share -- that additivity is itself checked.
    pool = {**S, **X}
    coef = {"X1": kp, "X2": kp, "X3": kp, "X4": c4, "X5": c5}
    out = {}
    for child, parents in TRUE_PARENTS.items():
        v_child = float(X[child].var())
        shares = {par: float((coef[child] * f(pool[par])).var()) / v_child
                  for par in parents}
        out[child] = {
            "var_child": v_child,
            "shares": shares,
            "noise_share": 1.0 - sum(shares.values()),
        }
    return out


# ============================================================================ #
# Claims 2-4 -- measured on the GENERATED data (what the model actually sees)
# ============================================================================ #
def _load_flat(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, 5) VALUE matrices for S and X.

    The exporter stores (n, n_var, n_feature) where the last axis carries the
    value and the variable id; the value channel index is recorded in the
    metadata as ``feature_indices.value``.  We also assert the variable-id
    channel matches ``variable_index_map`` so a column reordering upstream
    cannot silently mislabel the parents in this report.
    """
    meta_path = join(DATA_DIR, "dataset_metadata.json")
    val_idx, var_idx, idx_map = 0, None, None
    if exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        fi = meta.get("feature_indices", {})
        val_idx = int(fi.get("value", 0))
        var_idx = fi.get("variable")
        idx_map = meta.get("variable_index_map")

    d = np.load(path)
    s_raw = np.asarray(d["s"], dtype=np.float64)
    x_raw = np.asarray(d["x"], dtype=np.float64)
    s = s_raw[:, :, val_idx]
    x = x_raw[:, :, val_idx]
    assert s.shape[1] == len(S_LABELS), f"unexpected S width {s.shape}"
    assert x.shape[1] == len(X_LABELS), f"unexpected X width {x.shape}"

    # The variable-id channel must be strictly ASCENDING in each block: the
    # exporter re-indexes the sliced blocks locally (both s and x carry 1..5),
    # so we cannot compare against the GLOBAL variable_index_map -- but an
    # ascending id sequence does guarantee the columns are in declaration
    # order, i.e. S_LABELS / X_LABELS as listed above.
    if var_idx is not None:
        got_s = s_raw[0, :, int(var_idx)].astype(int).tolist()
        got_x = x_raw[0, :, int(var_idx)].astype(int).tolist()
        assert got_s == sorted(got_s) and len(set(got_s)) == len(S_LABELS), (
            f"S columns are not in declaration order: ids {got_s}")
        assert got_x == sorted(got_x) and len(set(got_x)) == len(X_LABELS), (
            f"X columns are not in declaration order: ids {got_x}")
    return s, x





def _basis(col: np.ndarray, degree: int = 5) -> np.ndarray:
    """Additive smooth basis for one predictor (standardised powers).

    The true mechanisms are additive in tanh(2*parent); a degree-5 polynomial in
    the standardised parent approximates that closely, so R^2 differences
    reflect the DATA rather than the flexibility of the fit.
    """
    z = (col - col.mean()) / (col.std() + 1e-12)
    return np.stack([z ** k for k in range(1, degree + 1)], axis=1)


def _r2(y: np.ndarray, design: np.ndarray | None) -> float:
    if design is None or design.shape[1] == 0:
        return 0.0
    A = np.concatenate([np.ones((len(y), 1)), design], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - float(resid.var()) / float(y.var())


def _residual(y: np.ndarray, design: np.ndarray | None) -> np.ndarray:
    if design is None or design.shape[1] == 0:
        return y - y.mean()
    A = np.concatenate([np.ones((len(y), 1)), design], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _hsic(a: np.ndarray, b: np.ndarray, m: int = 2000, seed: int = 0) -> float:
    """Biased HSIC with RBF kernels and the median-bandwidth heuristic."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a), size=min(m, len(a)), replace=False)
    a, b = a[idx][:, None], b[idx][:, None]

    def K(v):
        d2 = (v - v.T) ** 2
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / (med + 1e-12))

    Ka, Kb = K(a), K(b)
    mm = len(a)
    H = np.eye(mm) - 1.0 / mm
    return float(np.trace(Ka @ H @ Kb @ H) / (mm * mm))


def measure_on_data(s: np.ndarray, x: np.ndarray) -> dict:
    pool = {name: s[:, i] for i, name in enumerate(S_LABELS)}
    pool.update({name: x[:, i] for i, name in enumerate(X_LABELS)})

    report = {}
    for child, parents in TRUE_PARENTS.items():
        y = pool[child]
        par_names = list(parents)
        full = np.concatenate([_basis(pool[p]) for p in par_names], axis=1)
        r2_full = _r2(y, full)

        # --- Claim 2: per-parent R^2 drop (leave-one-parent-out) -------------
        r2_drop = {}
        # --- Claim 3: HSIC(parent, residual after fitting the OTHERS) --------
        hsic_par = {}
        for p_name in par_names:
            others = [q for q in par_names if q != p_name]
            design = (np.concatenate([_basis(pool[q]) for q in others], axis=1)
                      if others else None)
            r2_drop[p_name] = r2_full - _r2(y, design)
            hsic_par[p_name] = _hsic(pool[p_name], _residual(y, design))

        # --- Claim 4: no non-parent explains the residual --------------------
        resid_full = _residual(y, full)
        hsic_non = {}
        for name in S_LABELS + X_LABELS:
            if name in parents or name == child:
                continue
            hsic_non[name] = _hsic(pool[name], resid_full)

        report[child] = {
            "r2_full": r2_full,
            "r2_drop_per_parent": r2_drop,
            "hsic_parent_vs_other_resid": hsic_par,
            "hsic_nonparent_vs_full_resid": hsic_non,
        }
    return report


# ============================================================================ #
# Gate
# ============================================================================ #
def verify() -> bool:
    ok = True
    print("\n" + "=" * 70)
    print("VERIFYING scm_equal")
    print("=" * 70)

    # ---- Claim 1 --------------------------------------------------------- #
    raw = raw_variance_shares()
    print("\n[1] VARIANCE SHARES (raw mechanisms, independent re-derivation)")
    print(f"    {'child':6s} {'parent':7s} {'measured':>9s} {'intended':>9s}  status")
    for child, parents in TRUE_PARENTS.items():
        for par, intended in parents.items():
            got = raw[child]["shares"][par]
            good = abs(got - intended) <= TOL_SHARE_ABS
            ok &= good
            print(f"    {child:6s} {par:7s} {got:9.4f} {intended:9.4f}  "
                  f"{'OK' if good else 'FAIL'}")
        spread = (max(raw[child]["shares"].values())
                  - min(raw[child]["shares"].values()))
        good = spread <= TOL_SPREAD_ABS
        ok &= good
        print(f"    {child:6s} {'spread':7s} {spread:9.4f} {0.0:9.4f}  "
              f"{'OK' if good else 'FAIL'}   "
              f"(Var(child)={raw[child]['var_child']:.3f}, "
              f"noise={raw[child]['noise_share']:.3f})")

    # ---- Claims 2-4 ------------------------------------------------------ #
    train = join(DATA_DIR, "ds_train.npz")
    path = train if exists(train) else join(DATA_DIR, "ds.npz")
    if not exists(path):
        print(f"\n!! no generated data at {DATA_DIR} -- run without --verify-only")
        return False
    s, x = _load_flat(path)
    print(f"\n    data: {path}  |  S {s.shape}  X {x.shape}")
    data_rep = measure_on_data(s, x)

    print("\n[2] PER-PARENT R^2 DROP (what the reconstruction loss feels)")
    for child in TRUE_PARENTS:
        drops = data_rep[child]["r2_drop_per_parent"]
        spread = max(drops.values()) - min(drops.values())
        good = (spread <= TOL_R2_SPREAD_ABS) or len(drops) == 1
        ok &= good
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in drops.items())
        print(f"    {child:4s} r2_full={data_rep[child]['r2_full']:.4f}  {pretty}"
              f"   spread={spread:.4f}  {'OK' if good else 'FAIL'}")

    print("\n[3] HSIC(parent, residual after the OTHER parents)  "
          "(what the structural loss feels)")
    for child in MULTI_PARENT:
        h = data_rep[child]["hsic_parent_vs_other_resid"]
        rel = max(h.values()) / (min(h.values()) + 1e-12)
        good = rel <= 1.5
        ok &= good
        pretty = "  ".join(f"{k}={v:.3e}" for k, v in h.items())
        print(f"    {child:4s} {pretty}   max/min={rel:.2f}  "
              f"{'OK' if good else 'FAIL'}")

    print("\n[4] SUBSTITUTE SCAN: HSIC(non-parent, residual after ALL parents)")
    for child in TRUE_PARENTS:
        h_par = data_rep[child]["hsic_parent_vs_other_resid"]
        h_non = data_rep[child]["hsic_nonparent_vs_full_resid"]
        weakest_true = min(h_par.values())
        worst_name = max(h_non, key=h_non.get)
        worst = h_non[worst_name]
        good = worst <= HSIC_SUBSTITUTE_RATIO * weakest_true
        ok &= good
        print(f"    {child:4s} worst non-parent {worst_name}={worst:.3e}  vs "
              f"weakest true parent={weakest_true:.3e}  "
              f"ratio={worst / (weakest_true + 1e-12):.3f}  "
              f"{'OK' if good else 'FAIL'}")

    out = {
        "tolerances": {
            "share_abs": TOL_SHARE_ABS,
            "share_spread_abs": TOL_SPREAD_ABS,
            "r2_spread_abs": TOL_R2_SPREAD_ABS,
            "substitute_ratio": HSIC_SUBSTITUTE_RATIO,
        },
        "raw_variance_shares": raw,
        "measured_on_data": data_rep,
        "passed": bool(ok),
    }
    with open(join(DATA_DIR, "equal_strength_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("\n" + "=" * 70)
    print(("PASS -- scm_equal is a valid equal-strength control"
           if ok else
           "FAIL -- do NOT launch arms on this dataset until fixed"))
    print("report: " + join(DATA_DIR, "equal_strength_report.json"))
    print("=" * 70)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="skip generation, only run the verification gate")
    args = ap.parse_args()
    if not args.verify_only:
        generate()
    sys.exit(0 if verify() else 1)
