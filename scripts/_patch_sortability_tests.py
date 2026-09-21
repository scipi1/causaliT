"""Fix tests/test_sortability.py: R2-chain expectation + _normalize calls."""
from pathlib import Path

P = Path("tests/test_sortability.py")
t = P.read_text(encoding="utf-8")


def rep(old, new, label):
    global t
    assert t.count(old) == 1, f"{label}: {t.count(old)}x"
    t = t.replace(old, new)


rep(
    """    for i in range(1, d):
        # noise share shrinks along the chain -> R2 strictly increasing
        X[:, i] = X[:, i - 1] + rng.standard_normal(n) * (0.5 ** i)
""",
    """    for i in range(1, d):
        # constant additive noise: the noise SHARE of the marginal variance
        # shrinks along the chain, so R2 (regression on all others) strictly
        # increases -> perfect R2-sortability.
        X[:, i] = X[:, i - 1] + rng.standard_normal(n) * 0.3
""",
    "r2 chain",
)

# _normalize is an instance method (calls self._check_finite_values); build a
# bare instance for the unit tests.
rep(
    """def test_nodewise_standardize():
    from scm_ds.scm import SCMDataset
""",
    """def _bare_scm_dataset():
    from scm_ds.scm import SCMDataset

    inst = SCMDataset.__new__(SCMDataset)
    inst.name = "unit-test"
    return inst


def test_nodewise_standardize():
""",
    "helper",
)

for meth in ("standardize_node", "minmax_node", "minmax"):
    rep(
        f'SCMDataset._normalize(None, data, method="{meth}")',
        f'_bare_scm_dataset()._normalize(data, method="{meth}")',
        f"call {meth}",
    )

P.write_text(t, encoding="utf-8")
print("tests patched")
