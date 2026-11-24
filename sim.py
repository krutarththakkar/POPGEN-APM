"""Blue/white population with alternating sexual and asexual generations.

Model (fixed population size N, state = number of white individuals W):

  Sexual generation   Each child picks 2 parents uniformly at random (they may
                      be the same individual). Blue is dominant at the phenotype
                      level: the child is white only if both parents are white.
  Asexual generation  Each child copies one parent. Every parent has weight 1,
                      except that with probability q (once per generation, if any
                      whites exist) a single white individual, the "handsome
                      guy", has weight N*h.

One cycle = one sexual generation followed by one asexual generation. W = 0 and
W = N are absorbing, so every run ends with white lost or white fixed.

Tipping point = the starting white fraction x at which P(white fixes) = 1/2.
"""

import numpy as np


def run_cycle(W, N, h, q, rng):
    """Advance an int64 array of white counts through one sexual + one asexual generation."""
    W = rng.binomial(N, (W / N) ** 2)
    extra = (N * h - 1.0) * ((rng.random(W.shape) < q) & (W > 0))
    return rng.binomial(N, (W + extra) / (N + extra))


def q_critical(N, h):
    """q above which blue shrinks when rare (so a real tipping point exists).

    A rare blue type doubles in the sexual generation and is diluted by
    N / (N + N*h - 1) in a handsome guy generation, so its long-run log growth per
    cycle is ln 2 - q ln(1 + h - 1/N). Values > 1 mean blue always invades.
    """
    return np.log(2) / np.log1p(h - 1 / N) if N * h > 1 else np.inf


def fixation_prob(N, h, q, W0, reps, rng, max_cycles=20_000):
    """Simulate `reps` populations from each starting white count in W0.

    Returns (p_fix, unresolved): per-W0 fraction of runs where white fixed, and
    the fraction still polymorphic after max_cycles.
    """
    W0 = np.atleast_1d(W0)
    W = np.repeat(W0, reps).astype(np.int64)
    active = np.flatnonzero((W > 0) & (W < N))
    for _ in range(max_cycles):
        if active.size == 0:
            break
        w = run_cycle(W[active], N, h, q, rng)
        W[active] = w
        active = active[(w > 0) & (w < N)]
    W = W.reshape(len(W0), reps)
    return (W == N).mean(axis=1), ((W > 0) & (W < N)).mean(axis=1)


def _isotonic(y):
    """Least-squares non-decreasing fit (pool adjacent violators, equal weights)."""
    vals, wts = [], []
    for v in y:
        vals.append(float(v))
        wts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
            wts[-2:] = [w]
    return np.repeat(vals, wts)


def _crossing(x, y):
    """Where a non-decreasing curve with y[0] = 0, y[-1] = 1 crosses 1/2.

    Returns (x at the crossing, index below, index above).
    """
    lo = np.flatnonzero(y < 0.5)[-1]
    hi = np.flatnonzero(y > 0.5)[0]
    x_half = x[lo] + (0.5 - y[lo]) * (x[hi] - x[lo]) / (y[hi] - y[lo])
    return x_half, lo, hi


def tipping_point(N, h, q, reps, rng, n_grid=33, max_cycles=20_000, max_unresolved=0.05):
    """Estimate the tipping point by simulation.

    Scans a logit-spaced grid of starting white counts, then refines around the
    P(white fixes) = 1/2 crossing. Returns (tipping point as a white fraction,
    largest unresolved fraction seen). The tipping point is NaN when too many
    runs never absorb (long-lived coexistence, e.g. q near 1 with h <= 1).
    """
    z = np.linspace(-np.log(N - 1), np.log(N - 1), n_grid)
    grid = np.unique(np.clip(np.rint(N / (1 + np.exp(-z))), 1, N - 1).astype(np.int64))
    p, unresolved = fixation_prob(N, h, q, grid, reps, rng, max_cycles)
    worst = unresolved.max()
    if worst > max_unresolved:
        return np.nan, worst

    x = np.concatenate([[0], grid, [N]])
    y = np.concatenate([[0.0], _isotonic(p), [1.0]])
    _, lo, hi = _crossing(x, y)
    # Noise can shift the bracket by a grid step, so widen it by one each side.
    a, b = x[max(lo - 1, 0)], x[min(hi + 1, len(x) - 1)]
    fine = np.setdiff1d(np.unique(np.rint(np.linspace(a, b, n_grid)).astype(np.int64)), x)
    if fine.size:
        p2, unresolved2 = fixation_prob(N, h, q, fine, reps, rng, max_cycles)
        worst = max(worst, unresolved2.max())
        grid = np.concatenate([grid, fine])
        order = np.argsort(grid)
        grid, p = grid[order], np.concatenate([p, p2])[order]
        x = np.concatenate([[0], grid, [N]])
        y = np.concatenate([[0.0], _isotonic(p), [1.0]])
    return _crossing(x, y)[0] / N, worst


def exact_fixation(N, h, q):
    """Exact P(white fixes) for every W0 = 0..N from the Markov chain (O(N^3))."""
    from scipy.stats import binom

    k = np.arange(N + 1)
    sexual = binom.pmf(k[None, :], N, ((k / N) ** 2)[:, None])
    plain = binom.pmf(k[None, :], N, (k / N)[:, None])
    handsome = binom.pmf(k[None, :], N, ((k + N * h - 1.0) / (N + N * h - 1.0))[:, None])
    qq = np.where(k > 0, q, 0.0)[:, None]
    T = sexual @ ((1 - qq) * plain + qq * handsome)
    inner = slice(1, N)
    u = np.linalg.solve(np.eye(N - 1) - T[inner, inner], T[inner, N])
    return np.concatenate([[0.0], u, [1.0]])


def exact_tipping_point(N, h, q):
    """Tipping point (white fraction) from the exact fixation probabilities."""
    u = np.clip(exact_fixation(N, h, q), 0.0, 1.0)
    u[0], u[-1] = 0.0, 1.0
    return _crossing(np.arange(N + 1), np.maximum.accumulate(u))[0] / N
