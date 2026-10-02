"""Smoke tests for the figures: they must run and produce a PNG file."""
import matplotlib

matplotlib.use("Agg")   # no window, before importing the plotting module

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from riskengine.backtest import rolling_backtest  # noqa: E402
from riskengine.plotting import (  # noqa: E402
    plot_covid_zoom,
    plot_es_var_ratio,
    plot_pnl_vs_var,
    plot_violation_rates,
)
from riskengine.portfolio import DEFAULT_WEIGHTS, portfolio_pnl  # noqa: E402


@pytest.fixture(scope="module")
def results():
    """Small simulated backtest whose dates cover March 2020."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2019-06-03", periods=600)
    cols = list(DEFAULT_WEIGHTS.keys())
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(600, 5)),
                           index=dates, columns=cols)
    pnl = portfolio_pnl(returns)
    return rolling_backtest(returns, pnl, DEFAULT_WEIGHTS, window=100,
                            n_sims=1_000, verbose=False)


@pytest.mark.parametrize("func, name", [
    (plot_pnl_vs_var, "fig1.png"),
    (plot_es_var_ratio, "fig2.png"),
    (plot_covid_zoom, "fig3.png"),
    (plot_violation_rates, "fig4.png"),
])
def test_figure_is_created(results, tmp_path, func, name):
    out = tmp_path / name
    fig = func(results, out=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert isinstance(fig, plt.Figure)
    plt.close(fig)