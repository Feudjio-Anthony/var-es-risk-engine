"""Validation of the C++ backend against the Python (NumPy) reference.

The two backends use different random number generators, so their outputs
can never be identical, even with the same seed. The right question is:
"is the gap compatible with the Monte Carlo sampling error?"
We answer it by running both with 20 different seeds.
"""
import numpy as np
import pandas as pd
import pytest

from riskengine.measures import monte_carlo_var_es, parametric_var_es

# The C++ tests are skipped (not failed) if the module is not compiled.
try:
    from riskengine import mc_engine  # noqa: F401
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="C++ module not compiled")

N_SEEDS = 20
N_SIMS = 50_000
WEIGHTS = np.full(5, 0.2)


@pytest.fixture(scope="module")
def sample_returns():
    """Simulated daily returns of 5 correlated assets (about 1% volatility)."""
    rng = np.random.default_rng(2024)
    vols = np.array([0.012, 0.015, 0.004, 0.009, 0.006])
    corr = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    cov = np.outer(vols, vols) * corr
    data = rng.multivariate_normal(np.full(5, 0.0003), cov, size=2_000)
    return pd.DataFrame(data, columns=list("ABCDE"))


def _run(returns, backend, alpha):
    """VaR and ES for 20 seeds with one backend, as two arrays."""
    out = [monte_carlo_var_es(returns, WEIGHTS, alpha, n_sims=N_SIMS,
                              seed=s, backend=backend) for s in range(N_SEEDS)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def test_unknown_backend_is_rejected(sample_returns):
    """A typo in the backend name must fail loudly, not silently fall back."""
    with pytest.raises(ValueError, match="backend"):
        monte_carlo_var_es(sample_returns, WEIGHTS, backend="fortran")


@requires_cpp
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_backends_are_equivalent(sample_returns, alpha):
    """Mean over 20 seeds: VaR and ES of both backends agree within 1%.

    The gap between two 20-seed averages has a standard error of about
    0.2%, so 1% leaves a wide safety margin without hiding a real bug.
    """
    var_py, es_py = _run(sample_returns, "python", alpha)
    var_cpp, es_cpp = _run(sample_returns, "cpp", alpha)

    gap_var = abs(var_py.mean() - var_cpp.mean()) / var_py.mean()
    gap_es = abs(es_py.mean() - es_cpp.mean()) / es_py.mean()
    assert gap_var < 0.01, f"VaR gap of {gap_var:.2%}"
    assert gap_es < 0.01, f"ES gap of {gap_es:.2%}"

    # The seed-to-seed spread must be of the same order in both backends
    # (a sampler with the wrong variance or correlated draws would break this).
    assert 0.5 < var_cpp.std() / var_py.std() < 2.0


@requires_cpp
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_cpp_converges_to_analytical_solution(sample_returns, alpha):
    """Second, independent validation: the closed-form Gaussian formula.

    Same model, so the Monte Carlo average must approach the parametric
    VaR / ES. This does not rely on the Python backend at all.
    """
    var_th, es_th = parametric_var_es(sample_returns, WEIGHTS, alpha)
    var_cpp, es_cpp = _run(sample_returns, "cpp", alpha)

    assert var_cpp.mean() == pytest.approx(var_th, rel=0.01)
    assert es_cpp.mean() == pytest.approx(es_th, rel=0.01)


@requires_cpp
def test_cpp_backend_is_reproducible(sample_returns):
    """Same seed gives exactly the same result with the C++ backend."""
    a = monte_carlo_var_es(sample_returns, WEIGHTS, seed=5, backend="cpp")
    b = monte_carlo_var_es(sample_returns, WEIGHTS, seed=5, backend="cpp")
    c = monte_carlo_var_es(sample_returns, WEIGHTS, seed=6, backend="cpp")
    assert a == b
    assert a != c