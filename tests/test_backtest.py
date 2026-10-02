"""Unit tests for the rolling-window backtest engine."""
import numpy as np
import pandas as pd

from riskengine.backtest import count_violations, rolling_backtest
from riskengine.portfolio import DEFAULT_WEIGHTS, portfolio_pnl


def _make_returns(n=160, seed=0):
    """Small simulated dataset: 5 assets with a daily volatility of 1%."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    cols = list(DEFAULT_WEIGHTS.keys())
    data = rng.normal(0.0, 0.01, size=(n, len(cols)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _run(returns):
    """Run a small, fast backtest."""
    pnl = portfolio_pnl(returns)
    return rolling_backtest(returns, pnl, DEFAULT_WEIGHTS,
                            window=100, n_sims=2_000, verbose=False)


def test_output_shape():
    """One forecast per day after the first `window` days, 19 columns."""
    returns = _make_returns()
    results = _run(returns)
    assert len(results) == len(returns) - 100
    # pnl + 3 methods x 2 measures x 3 confidence levels
    assert results.shape[1] == 1 + 3 * 2 * 3


def test_no_lookahead_bias():
    """A shock on the LAST day must not change ANY forecast.

    Forecasts for day i only use days up to i - 1, so the last day's return
    can influence nothing except the realised P&L of that same day.
    """
    returns = _make_returns()
    base = _run(returns)

    shocked_returns = returns.copy()
    shocked_returns.iloc[-1] = shocked_returns.iloc[-1] * 20   # absurd shock
    shocked = _run(shocked_returns)

    forecast_cols = [c for c in base.columns if c != "pnl"]
    pd.testing.assert_frame_equal(base[forecast_cols], shocked[forecast_cols])

    # The realised P&L of the last day DID change (the shock is real).
    assert base["pnl"].iloc[-1] != shocked["pnl"].iloc[-1]


def test_lookahead_test_is_sensitive():
    """A shock INSIDE the last estimation window must change the last forecast.

    This proves the previous test is not vacuous: the forecasts do react
    to past data, they just do not react to future data.
    """
    returns = _make_returns()
    base = _run(returns)

    shocked_returns = returns.copy()
    shocked_returns.iloc[-2] = shocked_returns.iloc[-2] * 20   # day before last
    shocked = _run(shocked_returns)

    assert base["param_var_990"].iloc[-1] != shocked["param_var_990"].iloc[-1]


def test_count_violations_convention():
    """A violation happens when pnl < -var (pnl negative, var positive)."""
    results = pd.DataFrame({
        "pnl": [-10.0, -5.0, -20.0, 3.0],
        "hist_var_990": [8.0, 8.0, 8.0, 8.0],
    })
    violations = count_violations(results, "hist", 990)
    assert violations.tolist() == [True, False, True, False]