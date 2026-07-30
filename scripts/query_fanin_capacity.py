"""Pick / audit ``query_fanin_scale`` for the shared-query Hard-Concrete gate.

Context: with ``normalize_query=true``, ``remove_query_projection`` +
``remove_key_projection`` and an orthonormal structural key frame, the edge logit is

    log_alpha_ij = M_i * cos(q_hat_i, k_j) * sqrt(F)        F = query_fanin_scale

and the query provably lives in span(K), so ``sum_j cos^2(q_hat_i, k_j) == 1``
EXACTLY (not "<= 1").  The row therefore has a hard budget: F fixes how much
cos^2 each decision costs.  With T = init_edge_offset, the sharpest row holding
m parents at sigmoid(d) and n-m non-parents at sigmoid(-d) costs

    F(m, d) = [ m*(T + d)^2 + (n - m)*(T - d)^2 ] / M^2

This script inverts that (best affordable margin per in-degree) and reports the
state of the CENTROID initialisation, whose logit is sqrt(F/n) at M = 1.

See docs/experimental_elaborations/QUERY_FANIN_SCALE_BUDGET.md

Usage
-----
    python scripts/query_fanin_capacity.py                       # scm3: n=10
    python scripts/query_fanin_capacity.py --n 20 --max-indegree 5
    python scripts/query_fanin_capacity.py --n 10 --F 12.07 68.69
"""
from __future__ import annotations

import argparse
import math


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def gate_z(log_alpha: float, T: float, beta: float, gamma: float, zeta: float) -> float:
    """Deterministic (eval-time) Hard-Concrete gate value."""
    la = log_alpha - T
    return min(max(sig(la / beta) * (zeta - gamma) + gamma, 0.0), 1.0)


def F_needed(m: int, d: float, n: int, T: float, M: float = 1.0) -> float:
    """Budget consumed by a row with m parents at margin +d, n-m at margin -d."""
    return (m * (T + d) ** 2 + (n - m) * (T - d) ** 2) / M ** 2


def best_margin(F: float, m: int, n: int, T: float, M: float = 1.0) -> float | None:
    """Largest symmetric margin d affordable with budget F (None = not affordable)."""
    if F_needed(m, 0.0, n, T, M) > F:
        return None
    lo, hi = 0.0, 50.0
    for _ in range(200):                       # bisection (F is monotone in d)
        mid = 0.5 * (lo + hi)
        if F_needed(m, mid, n, T, M) > F:
            hi = mid
        else:
            lo = mid
    return lo


def F_for_saturated_gate(n: int, T: float, beta: float, gamma: float,
                         zeta: float) -> tuple[float, float]:
    """F such that the CENTROID query already drives the gate to z = 1."""
    x_sat = beta * math.log((1.0 - gamma) / (zeta - 1.0)) + T
    return n * x_sat ** 2, x_sat


def F_for_centroid_posterior(n: int, T: float, p: float) -> tuple[float, float]:
    """F such that P(edge) at the centroid equals p."""
    x = math.log(p / (1.0 - p)) + T
    return n * x ** 2, x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=10,
                    help='number of candidate parents per row = L_S + L_X (default 10)')
    ap.add_argument('--offset', type=float, default=math.log(3.0),
                    help='init_edge_offset T (default ln 3)')
    ap.add_argument('--beta', type=float, default=0.5, help='init_tau (default 0.5)')
    ap.add_argument('--gamma', type=float, default=-1.1)
    ap.add_argument('--zeta', type=float, default=1.1)
    ap.add_argument('--M', type=float, default=1.0,
                    help='learnable query-norm budget M_i (default 1.0)')
    ap.add_argument('--max-indegree', type=int, default=4)
    ap.add_argument('--F', type=float, nargs='*', default=None,
                    help='extra query_fanin_scale values to tabulate')
    a = ap.parse_args()
    n, T, beta, gamma, zeta, M = a.n, a.offset, a.beta, a.gamma, a.zeta, a.M

    F_sat, x_sat = F_for_saturated_gate(n, T, beta, gamma, zeta)
    F_p90, x_p90 = F_for_centroid_posterior(n, T, 0.90)
    F_dead = n * T ** 2                        # centroid sits ON the decision line

    print('gate: T=%.4f beta=%.3f gamma=%.2f zeta=%.2f | n=%d | M=%.3f'
          % (T, beta, gamma, zeta, n, M))
    print('stretch term beta*ln(-gamma/zeta) = %+.4f' % (beta * math.log(-gamma / zeta)))
    print('\nRECOMMENDED VALUES')
    print('  F = %8.2f  -> centroid gate SATURATES (z=1), log_alpha=%.4f, P=%.4f'
          % (F_sat, x_sat, sig(x_sat - T)))
    print('  F = %8.2f  -> centroid posterior P = 0.90, log_alpha=%.4f'
          % (F_p90, x_p90))
    print('  F = %8.2f  -> DEGENERATE: centroid exactly on the decision line'
          % F_dead)
    print('                  (P = 0.5 but z = 0 -> the eval gate passes NOTHING)')

    grid = sorted({round(F_dead, 2), round(F_sat, 2), round(F_p90, 2)}
                  | set(a.F or []))

    print('\ncentroid state vs F  (cos_j = 1/sqrt(n))')
    print('%9s %10s %8s %8s %11s' % ('F', 'log_alpha', 'P_cross', 'z_cross', 'P_self_dir'))
    for F in grid:
        x = M * math.sqrt(F / n)
        print('%9.2f %10.4f %8.4f %8.4f %11.4f'
              % (F, x, sig(x - T), gate_z(x, T, beta, gamma, zeta), sig(x) * 0.5))

    ms = list(range(1, a.max_indegree + 1))
    print('\ncapacity: best affordable P_on / P_off per in-degree m')
    print('%9s ' % 'F' + ' '.join('%13s' % ('m=%d' % m) for m in ms))
    for F in grid:
        cells = []
        for m in ms:
            d = best_margin(F, m, n, T, M)
            cells.append('%.3f/%.3f' % (sig(d), sig(-d)) if d is not None else '     ---     ')
        print('%9.2f ' % F + ' '.join('%13s' % c for c in cells))

    print('\ndiagnostics per F')
    for F in grid:
        m = min(3, a.max_indegree)
        d = best_margin(F, m, n, T, M)
        frac = (m * (T + d) ** 2 / (F * M ** 2)) if d is not None else float('nan')
        x = M * math.sqrt(F / n)
        print('  F=%8.2f | dP/dcos = M*sqrt(F)/4 = %5.3f | m=%d optimum spends %.2f of '
              'the budget on parents | matched init_edge_offset = %.4f'
              % (F, M * math.sqrt(F) / 4, m, frac, math.log(2 + math.exp(x))))
    print('\nNOTE: F scales with the node count (F = n * x^2).  Recompute per dataset.')


if __name__ == '__main__':
    main()
