"""Rank correlations and bootstrap intervals in plain NumPy (no SciPy)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks along the last axis of a 2-D array, ties sharing their average rank."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    order = np.argsort(x, axis=-1, kind="stable")
    r = np.empty_like(x)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(x.shape[-1], dtype=np.float64), x.shape), axis=-1)
    for i in range(x.shape[0]):
        s = np.sort(x[i])
        if np.any(s[1:] == s[:-1]):  # average the ranks of tied values
            _, inv, counts = np.unique(x[i], return_inverse=True, return_counts=True)
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, r[i])
            r[i] = (sums / counts)[inv]
    return r


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching rows of a and b, shape (n,). A constant row gives 0."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    out: np.ndarray = np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
    return out


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman rho between matching rows of a and b."""
    return pearson_rows(ranks(a), ranks(b))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Share of each row's top-k entries of a that are among the same row's top-k of b."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ta = np.argsort(-a, axis=-1)[:, :k]
    tb = np.argsort(-b, axis=-1)[:, :k]
    return np.array([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(ta, tb, strict=True)])


def bootstrap(
    stat: Callable[[Sequence[np.ndarray]], float],
    samples: Sequence[np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """`stat` on the samples, and a percentile interval from resampling each sample independently."""
    value = stat(samples)
    if n_boot <= 0 or any(len(s) == 0 for s in samples):
        return value, float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat([s[rng.integers(0, len(s), len(s))] for s in samples])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return value, float(lo), float(hi)
