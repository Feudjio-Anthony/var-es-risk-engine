"""Compare historical and parametric VaR / ES on the full period.

Run from the project root:
    python scripts/compare_methods.py
"""
import pandas as pd

from riskengine.measures import historical_var_es, parametric_var_es
from riskengine.portfolio import (
    DEFAULT_WEIGHTS,
    V0,
    compute_log_returns,
    load_prices,
    portfolio_pnl,
)

ALPHAS = [0.95, 0.975, 0.99]


def main():
    prices = load_prices()
    returns = compute_log_returns(prices)
    pnl = portfolio_pnl(returns)

    # Build the weight vector in the same order as the returns columns we use.
    cols = list(DEFAULT_WEIGHTS.keys())
    weights = [DEFAULT_WEIGHTS[c] for c in cols]
    returns = returns[cols]

    rows = []
    for alpha in ALPHAS:
        var_h, es_h = historical_var_es(pnl, alpha)
        var_p, es_p = parametric_var_es(returns, weights, alpha, v0=V0)
        rows.append({
            "alpha": alpha,
            "VaR hist": var_h,
            "VaR param": var_p,
            "VaR gap %": 100 * (var_p / var_h - 1),
            "ES hist": es_h,
            "ES param": es_p,
            "ES gap %": 100 * (es_p / es_h - 1),
            "ES/VaR hist": es_h / var_h,
            "ES/VaR param": es_p / var_p,
        })

    table = pd.DataFrame(rows).set_index("alpha")
    print("Historical vs parametric (full period, 1-day horizon, EUR)\n")
    print(table.round(2).to_string())
    print("\n'gap %' = parametric relative to historical "
          "(negative = parametric is too optimistic).")
    print("Gaussian theory: ES/VaR is about 1.15 at 99%.")


if __name__ == "__main__":
    main()