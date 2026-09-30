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


# ---------------------------------------------------------------------------
# Map-reduce comparison (spec §6). Inputs are [questions, positions] arrays of 0/1 scores,
# rows aligned on the same question IDs, so every resample carries both methods together.

NI_MARGIN = 0.02  # pre-registered non-inferiority margin: 2 points


def bootstrap_mean(x: Sequence[float], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    """Mean of a per-question vector, with its percentile 95% CI and the bootstrap draws."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(x.mean()), "low": float(low), "high": float(high), "boot": boot}


def one_sided_p(boot: np.ndarray, threshold: float) -> float:
    """Bootstrap p-value for H0: mean <= threshold."""
    return float((np.sum(boot <= threshold) + 1) / (len(boot) + 1))


def u_contrast(scores) -> np.ndarray:
    """Per question: mean(first, last position) - mean(the positions in between)."""
    s = np.asarray(scores, dtype=float)
    return s[:, [0, -1]].mean(axis=1) - s[:, 1:-1].mean(axis=1)


def non_inferiority(new, ref, margin: float = NI_MARGIN) -> Dict:
    """Test 1: average over positions per question, then require CI low > -margin."""
    diff = np.asarray(new, dtype=float).mean(axis=1) - np.asarray(ref, dtype=float).mean(axis=1)
    r = bootstrap_mean(diff)
    return {"diff": r["mean"], "low": r["low"], "high": r["high"], "margin": margin,
            "p": one_sided_p(r["boot"], -margin), "pass": r["low"] > -margin}


def delta_u(base, new) -> Dict:
    """Test 2: U(baseline) - U(new) per question; pass if its CI is above 0 (new is flatter)."""
    u_base, u_new = u_contrast(base), u_contrast(new)
    r = bootstrap_mean(u_base - u_new)
    return {"u_base": float(u_base.mean()), "u_new": float(u_new.mean()), "diff": r["mean"],
            "low": r["low"], "high": r["high"], "p": one_sided_p(r["boot"], 0.0), "pass": r["low"] > 0}


def recovered_share(base, new, middle: int, first: int = 0, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> Dict:
    """Test 3: (new_mid - base_mid) / (base_first - base_mid), unstable if the denominator's CI touches 0."""
    b, n = np.asarray(base, dtype=float), np.asarray(new, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(b), size=(n_boot, len(b)))
    num = n[:, middle][idx].mean(axis=1) - b[:, middle][idx].mean(axis=1)
    den = b[:, first][idx].mean(axis=1) - b[:, middle][idx].mean(axis=1)
    point_den = b[:, first].mean() - b[:, middle].mean()
    point = (n[:, middle].mean() - b[:, middle].mean()) / point_den if point_den else float("nan")
    ok = den != 0
    low, high = np.percentile(num[ok] / den[ok], [2.5, 97.5]) if ok.any() else (float("nan"), float("nan"))
    den_low, den_high = np.percentile(den, [2.5, 97.5])
    return {"share": float(point), "low": float(low), "high": float(high), "unstable": bool(den_low <= 0 <= den_high)}


def range_minus_null(scores, n_perm: int = N_BOOTSTRAP, seed: int = SEED, chunk: int = 250) -> Dict:
    """Best-minus-worst position accuracy, and its mean when position labels are shuffled
    within each question (the "no position effect" value). Descriptive only."""
    s = np.asarray(scores, dtype=float)
    acc = s.mean(axis=0)
    rng = np.random.default_rng(seed)
    null = []
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        order = np.argsort(rng.random((k, *s.shape)), axis=2)
        means = np.take_along_axis(np.broadcast_to(s, (k, *s.shape)), order, axis=2).mean(axis=1)
        null.append(means.max(axis=1) - means.min(axis=1))
    null_mean = float(np.concatenate(null).mean())
    observed = float(acc.max() - acc.min())
    return {"range": observed, "null_mean": null_mean, "excess": observed - null_mean}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (ties count half); nan if either class is missing."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ranks = (np.bincount(inverse, weights=ranks) / counts)[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
