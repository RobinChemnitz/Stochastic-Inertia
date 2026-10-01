"""Additive-noise stochastic Heun, killed on original or Lienard rectangles.

For curved boundaries w=const use the local normal volatility
sqrt((x*x-1)**2+epsilon**2) for the inward crossing correction.
Paired refinement shares Brownian increments at h and h/2.
"""
import numpy as np
from numba import njit, prange
BETA = 0.5825971579390107


@njit(cache=True)
def validate_parameters(initial, sigma, h, n_pairs, box, kind, tmax, mu):
    """Validate simulation inputs before entering the parallel loop."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    if not np.isfinite(h) or h <= 0 or not np.isfinite(tmax) or tmax <= 0:
        raise ValueError('h and tmax must be finite and positive')
    if not np.isfinite(mu) or mu <= 0 or n_pairs < 2:
        raise ValueError('mu must be finite and positive; n_pairs must be at least two')
    if kind != 0 and kind != 1:
        raise ValueError('kind must be 0 (original) or 1 (Lienard)')
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(box)):
        raise ValueError('Initial point and box must be finite')
    if box[0] >= box[1] or box[2] >= box[3]:
        raise ValueError('Box bounds must be strictly increasing')


@njit(cache=True, inline='always')
def face(x, y, box, kind, shift, epsilon):
    """Exit face: kind=0 means (x,y), kind=1 means (x,w).

    The shift on w faces includes their normal volatility.
    """
    if x <= box[0] + shift:
        return 0
    if x >= box[1] - shift:
        return 1
    v = y
    vshift = shift
    if kind == 1:
        v = epsilon * y + x * x * x / 3 - x
        vshift = shift * np.sqrt((x * x - 1) ** 2 + epsilon ** 2)
    if v <= box[2] + vshift:
        return 2
    if v >= box[3] - vshift:
        return 3
    return -1

@njit(cache=True, inline='always')
def step(x, y, h, wx, wy, mu):
    """Stochastic Heun step in original coordinates with additive noise increments wx, wy."""
    vx = y
    vy = mu * (1 - x * x) * y - x
    px = x + h * vx + wx
    py = y + h * vy + wy
    return (x + 0.5 * h * (vx + py) + wx, y + 0.5 * h * (vy + mu * (1 - px * px) * py - px) + wy)

@njit(cache=True, parallel=True)
def simulate(initial, sigma, h, n_pairs, box, kind, seed, tmax=100.0, mu=10.0):
    """Antithetic pairs with deterministic per-pair seeds.

    Negative exit faces indicate censored paths, which summarize() rejects.
    """
    validate_parameters(initial, sigma, h, n_pairs, box, kind, tmax, mu)
    times = np.full((n_pairs, 2), tmax)
    faces = np.full((n_pairs, 2), -1, np.int64)
    shift = BETA * np.sqrt(2 * sigma * h)
    amplitude = np.sqrt(2 * sigma * h)
    if face(initial[0], initial[1], box, kind, shift, 1 / mu) >= 0:
        raise ValueError('Start outside corrected domain')
    for k in prange(n_pairs):
        np.random.seed((seed + 104729 * k) % 4294967295)
        xp, yp = (initial[0], initial[1])
        xm, ym = (xp, yp)
        ap = True
        am = True
        for j in range(int(np.ceil(tmax / h))):
            wx = amplitude * np.random.randn()
            wy = amplitude * np.random.randn()
            if ap:
                xp, yp = step(xp, yp, h, wx, wy, mu)
                f = face(xp, yp, box, kind, shift, 1 / mu)
                if f >= 0:
                    ap = False
                    times[k, 0] = (j + 1) * h
                    faces[k, 0] = f
            if am:
                xm, ym = step(xm, ym, h, -wx, -wy, mu)
                f = face(xm, ym, box, kind, shift, 1 / mu)
                if f >= 0:
                    am = False
                    times[k, 1] = (j + 1) * h
                    faces[k, 1] = f
            if not ap and (not am):
                break
    return (times, faces)

@njit(cache=True, parallel=True)
def coupled(initial, sigma, h, n_pairs, box, kind, seed, tmax=100.0, mu=10.0):
    """Compare h and h/2 using the same Brownian path and antithetic pairs."""
    validate_parameters(initial, sigma, h, n_pairs, box, kind, tmax, mu)
    times = np.full((n_pairs, 2, 2), tmax)
    faces = np.full((n_pairs, 2, 2), -1, np.int64)
    shift = BETA * np.sqrt(2 * sigma * h)
    smallshift = shift / np.sqrt(2.0)
    amplitude = np.sqrt(sigma * h)
    if face(initial[0], initial[1], box, kind, shift, 1 / mu) >= 0:
        raise ValueError('Start outside corrected domain')
    for k in prange(n_pairs):
        np.random.seed((seed + 104729 * k) % 4294967295)
        z = np.empty((2, 2, 2))
        for level in range(2):
            for member in range(2):
                z[level, member, :] = initial
        for j in range(int(np.ceil(tmax / h))):
            wx1 = amplitude * np.random.randn()
            wy1 = amplitude * np.random.randn()
            wx2 = amplitude * np.random.randn()
            wy2 = amplitude * np.random.randn()
            for member in range(2):
                sign = 1.0 if member == 0 else -1.0
                if faces[k, 0, member] < 0:
                    x, y = step(z[0, member, 0], z[0, member, 1], h, sign * (wx1 + wx2), sign * (wy1 + wy2), mu)
                    z[0, member, 0] = x
                    z[0, member, 1] = y
                    f = face(x, y, box, kind, shift, 1 / mu)
                    if f >= 0:
                        times[k, 0, member] = (j + 1) * h
                        faces[k, 0, member] = f
                for half in range(2):
                    if faces[k, 1, member] < 0:
                        wx = wx1 if half == 0 else wx2
                        wy = wy1 if half == 0 else wy2
                        x, y = step(z[1, member, 0], z[1, member, 1], h / 2, sign * wx, sign * wy, mu)
                        z[1, member, 0] = x
                        z[1, member, 1] = y
                        f = face(x, y, box, kind, smallshift, 1 / mu)
                        if f >= 0:
                            times[k, 1, member] = (j + (half + 1) / 2) * h
                            faces[k, 1, member] = f
            if np.all(faces[k] >= 0):
                break
    return (times, faces)

def summarize(times, faces):
    """Estimate the mean and its standard error from independent pair means."""
    if times.ndim != 2 or times.shape[1] != 2 or faces.shape != times.shape:
        raise ValueError('Expected matching arrays of shape (n_pairs, 2).')
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError('At least two finite, nonnegative exit-time pairs are required.')
    censored = int(np.sum(faces < 0))
    if censored:
        raise RuntimeError(f'{censored} censored paths: increase tmax and rerun.')
    if not np.all(np.isin(faces, (0, 1, 2, 3))):
        raise ValueError('Exit faces must be 0 (left), 1 (right), 2 (bottom), or 3 (top).')
    pairs = times.mean(axis=1)
    return {
        'mean': float(pairs.mean()),
        'se': float(pairs.std(ddof=1) / np.sqrt(len(pairs))),
        'n_pairs': len(pairs),
        'n_paths': int(times.size),
        'censored': censored,
        'face_fractions': [float(np.mean(faces == i)) for i in range(4)],
        'max_exit_time': float(times.max()),
    }
