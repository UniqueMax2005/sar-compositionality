"""SAE Additivity Residuals (SAR).

Keep or remove K selected SAE contributions, read one behavior g(S) for each coalition S, fit the best
first order and pairwise models on some coalitions, and measure what they miss on the others.
SAR_r is the held-out fraction of behavioral variance left unexplained at interaction order r.
"""
from itertools import combinations

import numpy as np


def coalitions(K=16, M=384, p=0.5, n_fit=256, seed=0):
    """M random coalitions (each feature kept with probability p) and a fit / held-out split."""
    rng = np.random.default_rng(seed)
    masks = rng.random((M, K)) < p
    order = rng.permutation(M)
    return masks, order[:n_fit], order[n_fit:]


def rotation(K, seed=0):
    """Random orthogonal Q with Q @ 1 = 1: rotates the K contributions but keeps their span and their sum."""
    e = np.ones(K) / np.sqrt(K)
    U = np.column_stack([e, np.linalg.qr(np.eye(K) - np.outer(e, e))[0][:, : K - 1]])  # [1/sqrt(K), its complement]
    R, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((K - 1, K - 1)))
    Q = np.eye(K)
    Q[1:, 1:] = R * np.sign(np.diag(r))                 # Haar rotation of the complement
    return U @ Q @ U.T


def design(masks, order):
    """Columns 1{B in S} for |B| <= order: intercept, single features, then pairs."""
    s = np.asarray(masks, dtype=np.float64)
    cols = [np.ones((len(s), 1)), s]
    if order == 2:
        cols += [s[:, [i]] * s[:, [j]] for i, j in combinations(range(s.shape[1]), 2)]
    return np.hstack(cols)


def _spearman(a, b):
    def rank(x):                                        # average ranks over ties
        r = np.argsort(np.argsort(x, kind="stable")).astype(np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=r) / cnt)[inv]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def sar(g, masks, fit, held, g_full=None, eps=1e-8):
    """SAR_1, SAR_2, Delta_2, rho_2 and held-out coalition metrics of the first order and pairwise fits.

    g: behavior per coalition, shape (M,); masks: (M, K) bool; fit / held: index arrays.
    g_full: behavior of the full coalition g(N), only needed for the sign error.
    """
    g, masks = np.asarray(g, dtype=np.float64), np.asarray(masks, dtype=bool)
    gh = g[held]
    V = float(np.mean((gh - gh.mean()) ** 2))
    out = {}
    for r in (1, 2):
        X = design(masks, r)
        theta = np.linalg.lstsq(X[fit], g[fit], rcond=None)[0]
        a = X[held] @ theta
        dg, da = gh[:, None] - gh[None, :], a[:, None] - a[None, :]
        out[f"sar{r}"] = float(np.mean((gh - a) ** 2)) / (V + eps)
        out[f"rank{r}"] = _spearman(a, gh)                                          # rank fidelity
        out[f"E{r}"] = float(np.mean(dg**2 * (dg * da <= 0))) / (2 * V + 1e-12)     # ordering error
        out[f"regret{r}"] = float((gh.max() - gh[np.argmax(a)]) / (gh.max() - gh.min() + 1e-8))
        if g_full is not None:
            a_full = design(np.ones((1, masks.shape[1]), bool), r) @ theta
            out[f"sign{r}"] = float(np.mean(np.sign(gh - g_full) != np.sign(a - a_full)))
    out["delta2"] = out["sar1"] - out["sar2"]
    out["rho2"] = out["delta2"] / out["sar1"] if out["sar1"] > 1e-4 else float("nan")
    return out
