import numpy as np


def generate_diverse_population(n_points, x_L, x_U, rng=None, seed=None):
    """
    Generate an initial diverse population inside [x_L, x_U].

    If rng is not provided, a new Generator is created from seed.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    x_L = np.asarray(x_L, dtype=float)
    x_U = np.asarray(x_U, dtype=float)
    dim = len(x_L)

    freq = np.ones((dim, 4), dtype=float)
    S = np.empty((n_points, dim), dtype=float)

    first_block = min(4, n_points)
    S[:first_block, :] = (rng.random((first_block, dim)) + np.arange(first_block)[:, None]) / 4.0

    for i in range(4, n_points):
        inv = 1.0 / freq
        prob = inv / inv.sum(axis=1, keepdims=True)
        cdf = prob.cumsum(axis=1)
        u = rng.random(dim)
        m = (u[:, None] >= cdf).sum(axis=1)
        S[i, :] = (rng.random(dim) + m) / 4.0
        freq[np.arange(dim), m] += 1

    return S * (x_U - x_L) + x_L
