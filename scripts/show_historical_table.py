"""Compute historical VaR and ES on the full P&L series and check basic properties.

Run from the project root:
    python scripts/show_historical_table.py
"""
import pandas as pd

from riskengine.measures import historical_var_es

ALPHAS = [0.95, 0.975, 0.99]


def main():
    # Load the P&L saved in session 2. squeeze() turns the one-column
    # table into a Series.
    pnl = pd.read_csv("data/processed/pnl.csv", index_col=0,
                      parse_dates=True).squeeze("columns")

    rows = []
    for alpha in ALPHAS:
        var, es = historical_var_es(pnl, alpha)
        rows.append({"alpha": alpha, "VaR (EUR)": var, "ES (EUR)": es})

    table = pd.DataFrame(rows).set_index("alpha")
    print(f"Historical VaR / ES on {len(pnl)} days (1-day horizon)\n")
    print(table.round(0).to_string())

    # Two properties that must ALWAYS hold. If one fails, there is a sign bug.
    assert (table["ES (EUR)"] >= table["VaR (EUR)"]).all(), "ES must be >= VaR"
    assert table["VaR (EUR)"].is_monotonic_increasing, "VaR must grow with alpha"
    assert table["ES (EUR)"].is_monotonic_increasing, "ES must grow with alpha"
    print("\nChecks passed: ES >= VaR, and both grow with alpha.")


if __name__ == "__main__":
    main()