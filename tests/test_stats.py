import numpy as np
import pytest

from litm.stats import accuracy_ci, holm, mcnemar_exact, mde, paired_bootstrap, verdict


def pairs(n_both, n_only_a, n_only_b, n_neither):
    a = [1] * n_both + [1] * n_only_a + [0] * n_only_b + [0] * n_neither
    b = [1] * n_both + [0] * n_only_a + [1] * n_only_b + [0] * n_neither
    return a, b


def test_mcnemar_exact_known_value():
    a, b = pairs(50, 10, 2, 38)
    result = mcnemar_exact(a, b)
    # 2 * P(X <= 2), X ~ Binomial(12, 0.5) = 2 * (1 + 12 + 66) / 4096
    assert result["p"] == pytest.approx(158 / 4096)
    assert result["discordance"] == pytest.approx(0.12)


def test_mcnemar_no_discordance():
    assert mcnemar_exact([1, 0, 1], [1, 0, 1])["p"] == 1.0


def test_holm():
    assert holm([0.01, 0.04]) == [True, True]
    assert holm([0.03, 0.04]) == [False, False]
    assert holm([0.001, 0.2]) == [True, False]


def test_mde_matches_report_table():
    # Plan-check report: n=2655, d=0.20, two Holm contrasts -> about 2.7 points
    assert mde(0.20, 2655) == pytest.approx(0.027, abs=0.001)
    assert mde(0.20, 500) == pytest.approx(0.062, abs=0.001)


def test_bootstrap_is_deterministic_and_centred():
    a, b = pairs(300, 60, 20, 120)
    first, second = paired_bootstrap(a, b), paired_bootstrap(a, b)
    assert first == second
    assert first["diff"] == pytest.approx(40 / 500)
    assert first["low"] < first["diff"] < first["high"]
    acc = accuracy_ci(a)
    assert acc["low"] < acc["acc"] < acc["high"]


def make_curve(rng, n, p_first, p_middle, p_last):
    return [(rng.random(n) < p).astype(int) for p in (p_first, p_middle, p_last)]


@pytest.mark.parametrize(
    "probs, expected",
    [
        ((0.75, 0.55, 0.70), "U shape"),
        ((0.75, 0.55, 0.55), "Primacy only"),
        ((0.55, 0.55, 0.75), "Recency only"),
    ],
)
def test_verdict_shapes(probs, expected):
    rng = np.random.default_rng(1)
    assert verdict(*make_curve(rng, 2000, *probs))["shape"] == expected


def test_verdict_flat_versus_inconclusive():
    # Identical answers at every position: no difference, tight CI -> Flat
    same = [1, 0] * 1000
    assert verdict(same, same, same)["shape"] == "Flat"
    # Tiny sample with a real but unproven gap -> Inconclusive
    a, b = pairs(5, 3, 1, 5)
    assert verdict(a, b, a)["shape"] == "Inconclusive"
