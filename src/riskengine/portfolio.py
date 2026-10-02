"""Build the daily P&L series of the portfolio.

Pipeline of this module:
    prices  ->  log-returns  ->  weighted portfolio return  ->  P&L in euros
"""
import numpy as np
import pandas as pd

# Equally weighted portfolio: 20% in each of the five assets.
# This choice is deliberately simple and neutral: the goal of the project
# is to measure risk, not to optimise the allocation.
DEFAULT_WEIGHTS = {
    "SPY": 0.20,
    "EEM": 0.20,
    "IEF": 0.20,
    "GLD": 0.20,
    "FXE": 0.20,
}

# Portfolio value in euros. It is kept constant every day, which means
# we assume daily rebalancing back to the target weights.
V0 = 1_000_000.0


def load_prices(path="data/raw/prices.csv"):
    """Load the price file, using the date as the index.

    parse_dates=True turns the first column into real dates (instead of
    plain text), which is needed for time-based operations later.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def compute_log_returns(prices):
    """Compute daily log-returns: r_t = ln(P_t / P_{t-1}).

    prices.shift(1) moves every row one day down, so each row is divided
    by the previous day's price. The first day has no previous day, so its
    result is NaN ("not a number"); dropna() removes that row.
    """
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_pnl(returns, weights=None, v0=V0):
    """Compute the daily portfolio P&L, in euros.

    P&L_t = V0 * sum_i( w_i * r_i,t )

    We build the column order from the weights dictionary so that each
    weight is always multiplied by the right asset. This avoids silent
    mix-ups if the columns of the data file are in a different order.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Safety check: weights must sum to 1 (small float errors are allowed).
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    cols = list(weights.keys())
    w = np.array([weights[c] for c in cols])

    # returns[cols] has shape (n_days, n_assets); "@" is the matrix product
    # with the weight vector, giving one portfolio return per day.
    port_ret = returns[cols].to_numpy() @ w

    return pd.Series(v0 * port_ret, index=returns.index, name="pnl")