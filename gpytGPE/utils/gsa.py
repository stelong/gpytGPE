from itertools import combinations
from typing import Callable, Optional

import numpy as np
import torch


def srs(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    X = np.random.random_sample(size=(n, d))
    return X


def lhd(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    X = np.zeros((n, d), dtype=float)
    for j in range(d):
        a = np.random.random_sample(n)
        b = np.arange(n)
        np.random.shuffle(b)
        c = (a + b) / n
        X[:, j] = c
    return X


def sobol(
    n: int, d: int, seed: Optional[int] = None, skip: bool = True
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)

    soboleng = torch.quasirandom.SobolEngine(dimension=d, seed=seed)
    if skip:
        m = np.floor(np.log2(n + 1))
        n_skip = int(np.power(2, m) - 1)
        X = soboleng.draw(n_skip + n)
        X = X[n_skip:]
    else:
        X = soboleng.draw(n)
    return X.numpy()


def scale(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    X_scaled = np.copy(X)
    for j in range(d):
        X_scaled[:, j] = lb[j] + (ub[j] - lb[j]) * X_scaled[:, j]
    return X_scaled


def build_space(
    n: int,
    d: int,
    seed: Optional[int] = None,
    sampler: str = "sobol",
    scaling: Optional[np.ndarray] = None,
) -> np.ndarray:
    if sampler == "srs":
        ab = srs(n, 2 * d, seed)
    elif sampler == "lhd":
        ab = lhd(n, 2 * d, seed)
    elif sampler == "sobol":
        ab = sobol(n, 2 * d, seed, skip=True)
    else:
        raise ValueError(
            "Not a valid sampler! Available samplers are: 'srs', 'lhd', 'sobol'."
        )

    a = ab[:, :d]
    b = ab[:, d:]

    if scaling is not None:
        lb, ub = scaling[:, 0], scaling[:, 1]
        a = scale(a, lb, ub)
        b = scale(b, lb, ub)

    X = np.vstack((a, b))

    for i in range(d):
        ab_i = np.copy(a)
        ab_i[:, i] = b[:, i]
        X = np.vstack((X, ab_i))

    for i in range(d):
        ba_i = np.copy(b)
        ba_i[:, i] = a[:, i]
        X = np.vstack((X, ba_i))

    return X


def evaluate(
    n: int, d: int, f: Callable[[np.ndarray], np.ndarray], X: np.ndarray
) -> list:
    Y = f(X)
    length = len(Y.shape)
    if length == 1:
        Y = list(Y)

    y_dct_list = []
    for y in Y:
        y_dct = {}
        y_dct["a"] = y[:n]
        y_dct["b"] = y[n : 2 * n]

        for i in range(d):
            y_dct[f"ab_{i+1}"] = y[(2 + i) * n : (2 + i + 1) * n]

        if X.shape[0] == n * (2 * d + 2):
            for i in range(d):
                y_dct[f"ba_{i+1}"] = y[(2 + d + i) * n : (2 + d + i + 1) * n]

        y_dct_list.append(y_dct)
        if length == 1:
            break

    return y_dct_list


def si_estimator(fa: np.ndarray, fab_i: np.ndarray, fb: np.ndarray) -> float:
    var = np.var(fa, axis=0, ddof=1)
    si = np.mean(fb * (fab_i - fa), axis=0) / var
    return si


def sti_estimator(fa: np.ndarray, fab_i: np.ndarray, fb: np.ndarray) -> float:
    var = np.var(fa, axis=0, ddof=1)
    sti = 0.5 * np.mean(np.power(fa - fab_i, 2), axis=0) / var
    return sti


def sij_estimator(
    fa: np.ndarray,
    fab_i: np.ndarray,
    fab_j: np.ndarray,
    fba_i: np.ndarray,
    fb: np.ndarray,
) -> float:
    var_ij = np.mean(fba_i * fab_j - fa * fb, axis=0) / np.var(
        fa, axis=0, ddof=1
    )
    si = si_estimator(fa, fab_i, fb)
    sj = si_estimator(fa, fab_j, fb)
    sij = var_ij - si - sj
    return sij


def compute_Sobol_indices(y: dict, n_bootstrap: int = 100) -> tuple:
    n = y["a"].shape[0]
    d = int((len(y) - 2) / 2)

    si = np.zeros((d,), dtype=float)
    sti = np.zeros((d,), dtype=float)
    si_bootstrap = np.zeros((d,), dtype=float)
    sti_bootstrap = np.zeros((d,), dtype=float)

    r = np.random.randint(n, size=(n, n_bootstrap))

    for k in range(d):
        si[k] = si_estimator(y["a"], y[f"ab_{k+1}"], y["b"])
        sti[k] = sti_estimator(y["a"], y[f"ab_{k+1}"], y["b"])
        si_bootstrap[k] = si_estimator(
            y["a"][r], y[f"ab_{k+1}"][r], y["b"][r]
        ).std(axis=0, ddof=1)
        sti_bootstrap[k] = sti_estimator(
            y["a"][r], y[f"ab_{k+1}"][r], y["b"][r]
        ).std(axis=0, ddof=1)

    ij_list = [c for c in combinations(range(d), 2)]
    sij = np.zeros((len(ij_list),), dtype=float)
    sij_bootstrap = np.zeros((len(ij_list),), dtype=float)

    for k, (i, j) in enumerate(ij_list):
        sij[k] = sij_estimator(
            y["a"], y[f"ab_{i+1}"], y[f"ab_{j+1}"], y[f"ba_{i+1}"], y["b"]
        )
        sij_bootstrap[k] = sij_estimator(
            y["a"][r],
            y[f"ab_{i+1}"][r],
            y[f"ab_{j+1}"][r],
            y[f"ba_{i+1}"][r],
            y["b"][r],
        ).std(axis=0, ddof=1)

    return sti, sti_bootstrap, si, si_bootstrap, sij, sij_bootstrap
