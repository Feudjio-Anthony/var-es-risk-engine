"""Statistical tests to validate VaR models (backtesting).

Notation: I_t is the violation indicator on day t (1 if violation, else 0).
A correct VaR model at level alpha must satisfy TWO properties:
    1. Unconditional coverage: E[I_t] = 1 - alpha  (the frequency is right).
    2. Independence: violations have no memory (no clustering).
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy  # xlogy(a, b) = a * ln(b), with 0 * ln(0) = 0


def kupiec_pof(violations, alpha):
    """Kupiec test (Proportion of Failures): is the FREQUENCY correct?

    H0: the true violation rate is exactly 1 - alpha.
    The likelihood-ratio statistic compares the likelihood under the
    theoretical rate (H0) with the likelihood under the observed rate
    (the maximum-likelihood estimate). It follows a chi-square with
    1 degree of freedom.

    Returns (LR statistic, p-value, observed violations, expected violations).
    """
    v = np.asarray(violations, dtype=bool)
    T, x = v.size, int(v.sum())
    p = 1.0 - alpha        # theoretical violation rate
    pi = x / T             # observed violation rate

    # Log-likelihood of x violations in T days, for a violation rate r:
    #   (T - x) * ln(1 - r) + x * ln(r)
    # xlogy handles the edge cases x = 0 and x = T without any special code.
    ll_h0 = xlogy(T - x, 1 - p) + xlogy(x, p)
    ll_h1 = xlogy(T - x, 1 - pi) + xlogy(x, pi)

    lr = max(-2.0 * (ll_h0 - ll_h1), 0.0)   # max() guards against -1e-16

    # sf = survival function = 1 - cdf, but accurate for tiny p-values.
    return float(lr), float(stats.chi2.sf(lr, df=1)), x, T * p


def transition_counts(violations):
    """2x2 matrix n[i, j] = number of days where state i was followed by j.

    State 0 = no violation, state 1 = violation.
    """
    v = np.asarray(violations, dtype=int)
    prev, cur = v[:-1], v[1:]
    n = np.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            n[i, j] = np.sum((prev == i) & (cur == j))
    return n


def transition_probabilities(violations):
    """(pi01, pi11): P(violation after a calm day), P(violation after a violation).

    If violations cluster, pi11 is much larger than pi01.
    pi11 is NaN when there was no violation that could be followed by a next day.
    """
    n = transition_counts(violations)
    pi01 = n[0, 1] / (n[0, 0] + n[0, 1]) if (n[0, 0] + n[0, 1]) > 0 else np.nan
    pi11 = n[1, 1] / (n[1, 0] + n[1, 1]) if (n[1, 0] + n[1, 1]) > 0 else np.nan
    return float(pi01), float(pi11)


def christoffersen_independence(violations):
    """Christoffersen independence test: do violations follow each other?

    Model: first-order Markov chain with two states. Under H0 (independence),
    the probability of a violation does not depend on yesterday's state:
    pi01 = pi11. Statistic ~ chi-square with 1 degree of freedom.

    Returns (LR statistic, p-value).
    """
    n = transition_counts(violations)
    n00, n01, n10, n11 = n[0, 0], n[0, 1], n[1, 0], n[1, 1]
    total = n.sum()
    if total == 0:
        return 0.0, 1.0

    # Violation probability after a calm day / after a violation.
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    # Overall violation probability, the common value under H0.
    pi = (n01 + n11) / total

    ll_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_h1 = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
             + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))

    lr = max(-2.0 * (ll_h0 - ll_h1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def conditional_coverage(violations, alpha):
    """Joint test: right frequency AND independence. Chi-square, 2 d.o.f.

    LR_cc = LR_uc + LR_ind (Christoffersen, 1998).
    Returns (LR statistic, p-value).
    """
    lr_uc, _, _, _ = kupiec_pof(violations, alpha)
    lr_ind, _ = christoffersen_independence(violations)
    lr = lr_uc + lr_ind
    return float(lr), float(stats.chi2.sf(lr, df=2))


def _basel_zone(x):
    """Basel zone for x violations over 250 days (99% VaR)."""
    if x <= 4:
        return "green"
    if x <= 9:
        return "yellow"
    return "red"


def basel_traffic_light(violations, window=250):
    """Basel zone on the LAST `window` days. Returns (violations, zone).

    Green: 0-4 violations (model accepted). Yellow: 5-9 (capital surcharge).
    Red: 10 or more (model rejected). Defined for a 99% VaR.
    """
    x = int(np.asarray(violations, dtype=bool)[-window:].sum())
    return x, _basel_zone(x)


def basel_zone_shares(violations, window=250):
    """Basel zones over ALL rolling windows of `window` days.

    The last window alone can be misleading (a calm period at the end of
    the sample hides past crises). This function scans every window and
    returns the share of windows in each zone, plus the worst count.
    """
    v = pd.Series(np.asarray(violations, dtype=int))
    counts = v.rolling(window).sum().dropna()
    zones = counts.map(lambda c: _basel_zone(int(c)))
    return {
        "green": float((zones == "green").mean()),
        "yellow": float((zones == "yellow").mean()),
        "red": float((zones == "red").mean()),
        "max_violations": int(counts.max()),
    }