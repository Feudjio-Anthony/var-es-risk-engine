"""Unit tests for the VaR / ES functions."""
import numpy as np
import pandas as pd
import pytest

from riskengine.measures import historical_var_es, parametric_var_es


# ----------------------------------------------------------------------
# Historical simulation
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Parametric (Gaussian)
# ----------------------------------------------------------------------
def _simulate_gaussian_returns(n=500_000, seed=123):
    """Simulate truly Gaussian returns for 5 correlated assets."""
    rng = np.random.default_rng(seed)
    cols = ["A", "B", "C", "D", "E"]
    vols = np.array([0.012, 0.015, 0.004, 0.009, 0.006])   # daily volatilities
    # Correlation matrix: 0.3 everywhere off the diagonal (positive definite).
    corr = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    cov = np.outer(vols, vols) * corr
    mean = np.array([0.0003, 0.0004, 0.0001, 0.0002, 0.0000])
    data = rng.multivariate_normal(mean, cov, size=n)
    return pd.DataFrame(data, columns=cols)


def test_parametric_matches_historical_on_gaussian_data():
    """On truly Gaussian data, both methods must agree within 2%.

    If they differ here, it is a BUG in the code.
    If they only differ on real data, it is the MODEL (fat tails).
    """
    v0 = 1_000_000.0
    returns = _simulate_gaussian_returns()
    weights = np.full(5, 0.2)
    pnl = v0 * (returns.to_numpy() @ weights)

    for alpha in (0.95, 0.975, 0.99):
        var_h, es_h = historical_var_es(pnl, alpha)
        var_p, es_p = parametric_var_es(returns, weights, alpha, v0=v0)
        assert var_p == pytest.approx(var_h, rel=0.02)
        assert es_p == pytest.approx(es_h, rel=0.02)


def test_parametric_properties():
    """ES >= VaR, and both grow with alpha."""
    returns = _simulate_gaussian_returns(n=50_000)
    weights = np.full(5, 0.2)
    results = [parametric_var_es(returns, weights, a) for a in (0.95, 0.975, 0.99)]
    for var, es in results:
        assert es >= var
    assert results[0][0] < results[1][0] < results[2][0]
    assert results[0][1] < results[1][1] < results[2][1]