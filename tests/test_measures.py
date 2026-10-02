"""Unit tests for the VaR / ES functions."""
import numpy as np
import pytest

from riskengine.measures import historical_var_es


def test_var_known_normal():
    """On a large Gaussian sample, the 99% VaR must be close to 2.326 sigma."""
    rng = np.random.default_rng(42)
    pnl = rng.normal(0, 1, 500_000)
    var, es = historical_var_es(pnl, alpha=0.99)
    assert abs(var - 2.326) < 0.02
    assert es > var  # universal property


def test_es_greater_than_var():
    """ES >= VaR at every level, even for fat-tailed data (Student t, 4 dof)."""
    rng = np.random.default_rng(0)
    pnl = rng.standard_t(4, 100_000)
    for a in (0.95, 0.975, 0.99):
        var, es = historical_var_es(pnl, a)
        assert es >= var


def test_exact_small_example():
    """Hand-checkable case: P&L = -100, -99, ..., -1.

    The 5% quantile (linear interpolation) is -95.05, so VaR = 95.05.
    The tail is {-100, -99, -98, -97, -96}, whose mean is -98, so ES = 98.
    """
    pnl = np.arange(-100, 0, dtype=float)
    var, es = historical_var_es(pnl, alpha=0.95)
    assert var == pytest.approx(95.05)
    assert es == pytest.approx(98.0)