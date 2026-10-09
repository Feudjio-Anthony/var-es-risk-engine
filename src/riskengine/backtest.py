"""Rolling-window backtesting engine.

For every date i (starting after `window` days of history), we:
    1. take the estimation window = the `window` days BEFORE day i,
    2. fit the three models on that window only,
    3. produce a VaR and an ES forecast for day i,
    4. record the P&L actually realised on day i.

This gives genuine out-of-sample forecasts.
"""
import numpy as np
import pandas as pd

from .measures import historical_var_es, monte_carlo_var_es, parametric_var_es

# Short names of the three methods, used to build the column names.
METHODS = ("hist", "param", "mc")


def rolling_backtest(returns, pnl, weights, window=500,
                     alphas=(0.95, 0.975, 0.99), v0=1_000_000.0,
                     n_sims=50_000, backend="python", verbose=True):
    """Produce out-of-sample VaR and ES forecasts for the three methods.

    Parameters
    ----------
    returns : DataFrame (days x assets) of daily log-returns.
    pnl : Series of the realised daily portfolio P&L, same index as returns.
    weights : dict {asset name: weight}.
    window : number of past days used to estimate the models.
    alphas : confidence levels.
    v0 : portfolio value in euros.
    n_sims : number of Monte Carlo scenarios per forecast.
    backend : "python" (NumPy) or "cpp" (compiled module) for Monte Carlo.
        Only the Monte Carlo columns depend on it.
    verbose : print a progress message every 500 forecasts.

    Returns
    -------
    DataFrame indexed by date, with one column per (method, measure, level),
    for example "param_var_990", plus the realised "pnl".
    """
    # Both series must describe exactly the same days, in the same order.
    if not returns.index.equals(pnl.index):
        raise ValueError("returns and pnl must have the same index.")

    # Weight vector in the SAME order as the columns of returns.
    w = np.asarray([weights[c] for c in returns.columns], dtype=float)

    dates = returns.index
    n_forecasts = len(dates) - window
    if n_forecasts <= 0:
        raise ValueError("Not enough data for the chosen window.")

    rows = []

    # We start at index `window`: before that, there is not enough history.
    for i in range(window, len(dates)):
        # Estimation window: STRICTLY BEFORE day i.
        # The slice [i - window : i] stops at i - 1 (the end is excluded).
        # THIS LINE is what prevents look-ahead bias.
        # Never write [i - window : i + 1]: it would include day i itself.
        win_ret = returns.iloc[i - window:i]
        win_pnl = pnl.iloc[i - window:i].to_numpy()

        # The realised P&L of day i is only stored to be compared later;
        # it is never used to compute a forecast.
        row = {"date": dates[i], "pnl": float(pnl.iloc[i])}

        for a in alphas:
            # 0.99 -> 990, 0.975 -> 975, 0.95 -> 950 (used in column names).
            tag = round(a * 1000)

            v, e = historical_var_es(win_pnl, a)
            row[f"hist_var_{tag}"], row[f"hist_es_{tag}"] = v, e

            v, e = parametric_var_es(win_ret, w, a, v0)
            row[f"param_var_{tag}"], row[f"param_es_{tag}"] = v, e

            # seed=i: a different but reproducible seed for each date.
            v, e = monte_carlo_var_es(win_ret, w, a, v0, n_sims=n_sims,
                                      seed=i, backend=backend)
            row[f"mc_var_{tag}"], row[f"mc_es_{tag}"] = v, e

        rows.append(row)

        done = i - window + 1
        if verbose and (done % 500 == 0 or done == n_forecasts):
            print(f"  forecast {done}/{n_forecasts}  ({dates[i].date()})")

    return pd.DataFrame(rows).set_index("date")


def count_violations(results, method, alpha_tag):
    """Boolean series: True on the days when the loss exceeds the VaR.

    Reminder of the convention: pnl is NEGATIVE for a loss, var is POSITIVE.
    A violation happens when pnl < -var.
    """
    var = results[f"{method}_var_{alpha_tag}"]
    return results["pnl"] < -var


def violation_summary(results, alphas=(0.95, 0.975, 0.99)):
    """Observed vs theoretical violation rate, for every method and level."""
    rows = []
    for method in METHODS:
        for a in alphas:
            tag = round(a * 1000)
            violations = count_violations(results, method, tag)
            rows.append({
                "method": method,
                "alpha": a,
                "forecasts": len(violations),
                "violations": int(violations.sum()),
                "observed %": 100 * violations.mean(),
                "expected %": 100 * (1 - a),
            })
    return pd.DataFrame(rows).set_index(["method", "alpha"])