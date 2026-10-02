"""Value at Risk (VaR) and Expected Shortfall (ES): the three methods.

SIGN CONVENTION (valid for the whole module):
    - pnl : signed series. Negative = loss.
    - var, es : POSITIVE numbers expressing a loss.
      A VaR of 15,000 means "we can lose 15,000 euros".

Every method returns a tuple (var, es).
"""
import numpy as np
from scipy import stats


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


def parametric_var_es(returns, weights, alpha=0.99, v0=1_000_000.0):
    """Gaussian VaR and ES (variance-covariance approach).

    We assume the asset returns follow a multivariate normal law.
    A weighted sum of Gaussian variables is Gaussian, so the portfolio
    return is Gaussian too, and VaR / ES have closed-form formulas.

    Parameters
    ----------
    returns : DataFrame (days x assets) of daily log-returns.
    weights : vector of weights, in the SAME order as the columns of returns.
    alpha : confidence level (0.99 means 99%).
    v0 : portfolio value in euros.

    Returns
    -------
    (var, es) : two positive floats, in euros.
    """
    mu = returns.mean().to_numpy()    # mean return of each asset
    cov = returns.cov().to_numpy()    # covariance matrix between assets
    w = np.asarray(weights, dtype=float)

    # Mean and standard deviation of the PORTFOLIO return.
    mu_p = float(w @ mu)

    # w' * Sigma * w is the portfolio variance. This is where
    # diversification lives: negative or low cross-covariances
    # make sigma_p smaller than the weighted average of the asset vols.
    sigma_p = float(np.sqrt(w @ cov @ w))

    # z = quantile of the standard normal law (about 2.326 for 99%).
    z = stats.norm.ppf(alpha)

    # VaR = V0 * (z * sigma_p - mu_p). The minus sign on mu_p is because
    # a positive average return reduces the loss.
    var = v0 * (z * sigma_p - mu_p)

    # ES: inverse Mills ratio. For a standard normal Z,
    # E[Z | Z > z] = pdf(z) / (1 - alpha). Then we scale by sigma_p
    # and shift by mu_p.
    es = v0 * (sigma_p * stats.norm.pdf(z) / (1 - alpha) - mu_p)

    return float(var), float(es)