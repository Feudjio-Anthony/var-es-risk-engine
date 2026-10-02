"""Value at Risk (VaR) and Expected Shortfall (ES): the three methods.

SIGN CONVENTION (valid for the whole module):
    - pnl : signed series. Negative = loss.
    - var, es : POSITIVE numbers expressing a loss.
      A VaR of 15,000 means "we can lose 15,000 euros".

COMMON SIGNATURE (all three methods will follow it):
    method(pnl, alpha) -> (var, es)
"""
import numpy as np


def historical_var_es(pnl, alpha=0.99):
    """VaR and ES by historical simulation.

    Idea: the best forecast of tomorrow's distribution is the distribution
    observed over the recent past. No law is assumed, no parameter is fitted.

    Parameters
    ----------
    pnl : array-like
        P&L observed over the estimation window (signed, in euros).
    alpha : float
        Confidence level (0.99 means 99%).

    Returns
    -------
    (var, es) : two positive floats.
    """
    # Convert to a NumPy array of floats so every input type works
    # (list, pandas Series, array...).
    pnl = np.asarray(pnl, dtype=float)

    # The quantile of order (1 - alpha) of the P&L distribution.
    # For alpha = 99%, it is the 1% quantile: a very negative P&L.
    # We flip the sign to get a positive loss.
    q = np.quantile(pnl, 1.0 - alpha)
    var = -q

    # ES = average of the tail, i.e. of all scenarios at least as bad as
    # the quantile. Again we flip the sign to get a positive loss.
    tail = pnl[pnl <= q]

    # Safety: the tail always contains at least one point (the quantile
    # lies within the data range), but we guard against an empty tail anyway.
    es = -tail.mean() if tail.size > 0 else var

    return float(var), float(es)