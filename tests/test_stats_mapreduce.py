import numpy as np
import pytest

from litm.stats import auroc, delta_u, non_inferiority, one_sided_p, range_minus_null, recovered_share, u_contrast


def test_u_contrast():
    s = np.array([[1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    assert u_contrast(s).tolist() == pytest.approx([1.0, 0.0, -1.0])


def test_non_inferiority_needs_the_lower_bound_above_minus_margin():
    rng = np.random.default_rng(0)
    ref = (rng.random((2000, 5)) < 0.6).astype(int)
    worse = ref.copy()
    worse[(rng.random(ref.shape) < 0.05) & (ref == 1)] = 0  # about 3 points worse
    result = non_inferiority(worse, ref)
    assert result["diff"] < -0.02 and not result["pass"]
    assert non_inferiority(ref, ref)["pass"]


def test_non_inferiority_does_not_reward_noise():
    rng = np.random.default_rng(1)
    ref = (rng.random((30, 5)) < 0.6).astype(int)
    new = (rng.random((30, 5)) < 0.6).astype(int)
    result = non_inferiority(new, ref)
    # The old rule ("CI not entirely below 0") would pass this tiny, noisy comparison.
    assert result["high"] > 0 and not result["pass"]


def test_delta_u_detects_a_flattened_curve():
    rng = np.random.default_rng(2)
    p_base = np.array([0.75, 0.55, 0.55, 0.55, 0.70])
    base = (rng.random((2000, 5)) < p_base).astype(int)
    flat = (rng.random((2000, 5)) < 0.65).astype(int)
    result = delta_u(base, flat)
    assert result["diff"] == pytest.approx(0.175, abs=0.05)
    assert result["pass"] and result["p"] < 0.001
    assert not delta_u(base, base)["pass"]


def test_one_sided_p():
    assert one_sided_p(np.array([1.0, 2.0, 3.0]), 0.0) == pytest.approx(0.25)
    assert one_sided_p(np.array([-1.0, -2.0, 3.0]), 0.0) == pytest.approx(0.75)


def test_recovered_share():
    base = np.array([[1, 0]] * 50 + [[1, 1]] * 50)  # first 100%, middle 50%
    new = np.array([[1, 1]] * 75 + [[1, 0]] * 25)   # middle 75%
    r = recovered_share(base, new, middle=1)
    assert r["share"] == pytest.approx(0.5) and not r["unstable"]
    flat = np.array([[1, 1]] * 50 + [[0, 0]] * 50)
    assert recovered_share(flat, new, middle=1)["unstable"]


def test_range_minus_null_is_near_zero_without_a_position_effect():
    rng = np.random.default_rng(3)
    scores = (rng.random((500, 5)) < 0.6).astype(int)
    r = range_minus_null(scores, n_perm=2000)
    assert r["null_mean"] > 0.02            # the raw range is biased upwards
    assert abs(r["excess"]) < 0.04
    assert range_minus_null(scores, n_perm=2000) == r


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert np.isnan(auroc([0.3], [1]))
