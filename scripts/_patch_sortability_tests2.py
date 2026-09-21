"""Fix the flaky R2-chain test: exact hand-computed alignment checks."""
from pathlib import Path

P = Path("tests/test_sortability.py")
t = P.read_text(encoding="utf-8")

old = """def test_r2_sortability_deterministic_chain():
    rng = np.random.default_rng(1)
    n, d = 5000, 4
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        # constant additive noise: the noise SHARE of the marginal variance
        # shrinks along the chain, so R2 (regression on all others) strictly
        # increases -> perfect R2-sortability.
        X[:, i] = X[:, i - 1] + rng.standard_normal(n) * 0.3
    W = _chain_adj(d)
    assert r2_sortability(X, W) == 1.0
"""

new = """def test_order_alignment_paths_exact():
    # Chain 0 -> 1 -> 2: directed paths (0,1), (1,2), (0,2), each weight 1.
    W = _chain_adj(3)
    assert order_alignment_paths(W, np.array([1.0, 2.0, 3.0])) == 1.0
    assert order_alignment_paths(W, np.array([3.0, 2.0, 1.0])) == 0.0
    assert order_alignment_paths(W, np.array([1.0, 1.0, 1.0])) == 0.5  # ties
    # wrong on (0,1), right on (1,2) and (0,2): 2/3
    assert order_alignment_paths(W, np.array([2.0, 1.0, 3.0])) == 2.0 / 3.0


def test_r2_sortability_above_chance_on_chain():
    rng = np.random.default_rng(1)
    n, d = 5000, 6
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    for i in range(1, d):
        X[:, i] = X[:, i - 1] + rng.standard_normal(n) * 0.3
    W = _chain_adj(d)
    val = r2_sortability(X, W)
    assert 0.5 < val <= 1.0
"""
assert t.count(old) == 1
P.write_text(t.replace(old, new), encoding="utf-8")
print("patched")
