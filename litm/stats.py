"""Paired statistics for position contrasts (plan section 5).

All comparisons are paired: a[i] and b[i] are 0/1 scores for the same question at two
positions.
"""
from math import comb, sqrt
from statistics import NormalDist
from typing import Dict, List, Sequence

import numpy as np

N_BOOTSTRAP = 10_000
SEED = 0
# Pre-registered equivalence margin: a contrast counts as "no effect" only if its whole
# 95% CI lies within +/- 3 points. (Using the MDE here would call tiny samples "Flat",
# because the MDE widens as fast as the CI.)
FLAT_MARGIN = 0.03


def accuracy_ci(scores: Sequence[int], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    x = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"acc": x.mean(), "low": low, "high": high, "n": len(x)}


def paired_bootstrap(a: Sequence[int], b: Sequence[int], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    """95% CI for mean(a) - mean(b), resampling questions."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"diff": diff.mean(), "low": low, "high": high}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict:
    """Two-sided exact McNemar test on discordant pairs."""
    a, b = np.asarray(a), np.asarray(b)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
        p = min(1.0, 2 * tail)
    return {"only_a": only_a, "only_b": only_b, "discordance": n / len(a), "p": p}


def holm(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm step-down: which hypotheses are rejected."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rejected = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        rejected[i] = True
    return rejected


def mde(discordance: float, n: int, alpha: float = 0.025, power: float = 0.8) -> float:
    """Minimum detectable paired difference (Card et al. 2020 approximation).

    alpha defaults to 0.05 / 2, the stricter Holm threshold for two contrasts.
    """
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * sqrt(discordance / n)


def contrast(a: Sequence[int], b: Sequence[int]) -> Dict:
    result = {**paired_bootstrap(a, b), **mcnemar_exact(a, b)}
    result["mde"] = mde(result["discordance"], len(a))
    return result


def verdict(first: Sequence[int], middle: Sequence[int], last: Sequence[int]) -> Dict:
    """Pre-registered shape verdict from the first-vs-middle and last-vs-middle contrasts."""
    first_vs_middle = contrast(first, middle)
    last_vs_middle = contrast(last, middle)
    first_sig, last_sig = holm([first_vs_middle["p"], last_vs_middle["p"]])
    # A significant contrast only counts toward the shape if the middle is the lower one.
    first_up = first_sig and first_vs_middle["diff"] > 0
    last_up = last_sig and last_vs_middle["diff"] > 0

    if first_up and last_up:
        shape = "U shape"
    elif first_up and not last_sig:
        shape = "Primacy only"
    elif last_up and not first_sig:
        shape = "Recency only"
    elif not first_sig and not last_sig:
        within = all(
            -FLAT_MARGIN <= c["low"] and c["high"] <= FLAT_MARGIN for c in (first_vs_middle, last_vs_middle)
        )
        shape = "Flat" if within else "Inconclusive"
    else:
        shape = "Other"  # e.g. the middle is significantly *better* than an end
    return {"shape": shape, "first_vs_middle": first_vs_middle, "last_vs_middle": last_vs_middle}
