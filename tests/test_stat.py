"""Unit tests for the statistical validation tests."""
import numpy as np
import pytest
from scipy import stats

from riskengine.tests_stat import (
    basel_traffic_light,
    basel_zone_shares,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)


def test_kupiec_perfect_frequency():
    """10 violations in 1000 days at 99%: exactly the expected rate."""
    v = np.zeros(1000, dtype=bool)
    v[:10] = True
    lr, p, x, expected = kupiec_pof(v, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)
    assert x == 10
    assert expected == pytest.approx(10.0)


def test_kupiec_zero_violations():
    """No violation at all: LR = -2 * T * ln(1 - p), no crash, no NaN."""
    v = np.zeros(250, dtype=bool)
    lr, p, x, _ = kupiec_pof(v, 0.99)
    assert x == 0
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))
    assert p == pytest.approx(stats.chi2.sf(lr, df=1))


def test_kupiec_rejects_too_many_violations():
    """30 violations in 1000 days at 99% (3% instead of 1%): rejected."""
    v = np.zeros(1000, dtype=bool)
    v[:30] = True
    _, p, _, _ = kupiec_pof(v, 0.99)
    assert p < 0.001


def test_independence_accepts_spread_violations():
    """10 violations spaced 100 days apart: no clustering."""
    v = np.zeros(1000, dtype=bool)
    v[::100] = True
    _, p = christoffersen_independence(v)
    assert p > 0.3


def test_independence_rejects_clustered_violations():
    """10 violations in a row: strong clustering, even if the frequency is right."""
    v = np.zeros(1000, dtype=bool)
    v[100:110] = True
    _, p = christoffersen_independence(v)
    assert p < 0.001


def test_conditional_coverage_adds_both_tests():
    """With the right frequency, LR_uc = 0, so LR_cc equals LR_ind.

    This is exactly the case Kupiec cannot see: 1% of violations,
    but all concentrated in 10 consecutive days.
    """
    v = np.zeros(1000, dtype=bool)
    v[100:110] = True
    lr_ind, _ = christoffersen_independence(v)
    lr_cc, p_cc = conditional_coverage(v, 0.99)
    assert lr_cc == pytest.approx(lr_ind, abs=1e-6)
    assert p_cc < 0.001


def test_basel_zone_boundaries():
    """Green up to 4, yellow from 5 to 9, red from 10; only the last 250 days count."""
    cases = [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"),
             (10, "red"), (12, "red")]
    for n_viol, expected_zone in cases:
        v = np.zeros(300, dtype=bool)
        if n_viol > 0:
            v[-n_viol:] = True
        x, zone = basel_traffic_light(v)
        assert (x, zone) == (n_viol, expected_zone)

    # Violations older than 250 days are ignored.
    v = np.zeros(300, dtype=bool)
    v[:50] = True
    assert basel_traffic_light(v) == (0, "green")


def test_basel_zone_shares():
    """No violation: all windows green. A cluster of 10: some windows red."""
    calm = basel_zone_shares(np.zeros(500, dtype=bool))
    assert calm["green"] == 1.0
    assert calm["max_violations"] == 0

    v = np.zeros(500, dtype=bool)
    v[250:260] = True
    shares = basel_zone_shares(v)
    assert shares["max_violations"] == 10
    assert 0 < shares["red"] < 1
    assert shares["green"] + shares["yellow"] + shares["red"] == pytest.approx(1.0)