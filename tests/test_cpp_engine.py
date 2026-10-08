"""Tests of the compiled C++ Monte Carlo module (skipped if it is not built)."""
import numpy as np
import pytest
from scipy import stats

# If the module has not been compiled, every test below is skipped
# (the rest of the project does not depend on it).
mc_engine = pytest.importorskip("riskengine.mc_engine")

# A small, realistic setup: 5 assets with 30% correlation.
VOLS = np.array([0.012, 0.015, 0.004, 0.009, 0.006])
CORR = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
COV = np.outer(VOLS, VOLS) * CORR
MU = np.array([0.0003, 0.0004, 0.0001, 0.0002, 0.0])
W = np.full(5, 0.2)
V0 = 1_000_000.0


def test_mean_and_std_match_theory():
    """Simulated P&L has the theoretical mean and standard deviation.

    A bug in the correlation step (for example a forgotten transpose)
    would shift the standard deviation by several percent.
    """
    pnl = mc_engine.simulate_pnl(MU, COV, W, 1_000_000, V0, 42)
    assert pnl.shape == (1_000_000,)
    assert pnl.mean() == pytest.approx(V0 * W @ MU, abs=40.0)
    assert pnl.std() == pytest.approx(V0 * np.sqrt(W @ COV @ W), rel=0.005)


def test_correlation_is_applied():
    """Long asset 0, short asset 1: the variance depends on the correlation.

    Var = s0^2 + s1^2 - 2*rho*s0*s1. With independent assets (rho = 0)
    it would be about 25% larger, so this test detects missing correlation.
    """
    w = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    pnl = mc_engine.simulate_pnl(np.zeros(5), COV, w, 1_000_000, 1.0, 7)
    theory = np.sqrt(w @ COV @ w)
    independent = np.sqrt(VOLS[0] ** 2 + VOLS[1] ** 2)
    assert independent / theory > 1.1                    # the test can discriminate
    assert pnl.std() == pytest.approx(theory, rel=0.005)


def test_normal_sampler_is_standard_normal():
    """Pure N(0, 1) draws: Kolmogorov-Smirnov test, kurtosis and far tail.

    With sigma = identity and weights = (1, 0, 0, 0, 0), the output is
    exactly one standard normal number per scenario.
    """
    z = mc_engine.simulate_pnl(np.zeros(5), np.eye(5), np.eye(5)[0],
                               1_000_000, 1.0, 123)
    assert stats.kstest(z, "norm").pvalue > 0.01
    assert abs(stats.kurtosis(z)) < 0.03
    # Far tail (beyond 3.5 sigma): about 465 points expected out of 1,000,000.
    assert abs(np.sum(np.abs(z) > 3.5) - 2 * stats.norm.sf(3.5) * z.size) < 105


def test_var_close_to_analytical():
    """99% VaR within 1% of the closed-form Gaussian VaR."""
    pnl = mc_engine.simulate_pnl(MU, COV, W, 1_000_000, V0, 3)
    var_mc = -np.quantile(pnl, 0.01)
    var_th = V0 * (stats.norm.ppf(0.99) * np.sqrt(W @ COV @ W) - W @ MU)
    assert var_mc == pytest.approx(var_th, rel=0.01)


def test_reproducible():
    """Same seed gives exactly the same numbers; another seed does not."""
    a = mc_engine.simulate_pnl(MU, COV, W, 1000, V0, 7)
    b = mc_engine.simulate_pnl(MU, COV, W, 1000, V0, 7)
    c = mc_engine.simulate_pnl(MU, COV, W, 1000, V0, 8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_accepts_non_contiguous_and_float32_input():
    """Fortran-ordered or float32 arrays are converted, not misread."""
    ref = mc_engine.simulate_pnl(MU, COV, W, 1000, V0, 1)
    fortran = mc_engine.simulate_pnl(MU, np.asfortranarray(COV), W, 1000, V0, 1)
    assert np.array_equal(ref, fortran)
    f32 = mc_engine.simulate_pnl(MU.astype(np.float32), COV.astype(np.float32),
                                 W.astype(np.float32), 1000, V0, 1)
    assert f32.shape == (1000,)
    assert np.std(f32) == pytest.approx(np.std(ref), rel=0.15)


def test_bad_inputs_raise_clear_errors():
    with pytest.raises(ValueError):
        mc_engine.simulate_pnl(MU, COV, np.ones(3), 100, V0, 1)    # weights length
    with pytest.raises(ValueError):
        mc_engine.simulate_pnl(MU, np.eye(4), W, 100, V0, 1)       # sigma shape
    with pytest.raises(ValueError):
        mc_engine.simulate_pnl(MU, COV, W, 0, V0, 1)               # n_sims
    with pytest.raises(RuntimeError, match="positive definite"):
        mc_engine.simulate_pnl(MU, np.ones((5, 5)), W, 100, V0, 1)  # singular