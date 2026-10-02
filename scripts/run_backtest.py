"""Run the rolling-window backtest and save the results.

Run from the project root:
    python scripts/run_backtest.py --fast     (quick test, 2018-2022)
    python scripts/run_backtest.py            (full run, takes several minutes)
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from riskengine.backtest import rolling_backtest, violation_summary
from riskengine.portfolio import (
    DEFAULT_WEIGHTS,
    compute_log_returns,
    load_prices,
    portfolio_pnl,
)

RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser(description="Rolling-window VaR/ES backtest")
    parser.add_argument("--fast", action="store_true",
                        help="short period and fewer simulations, for quick tests")
    args = parser.parse_args()

    prices = load_prices()
    returns = compute_log_returns(prices)

    # Keep the columns in the same order as the weights dictionary.
    returns = returns[list(DEFAULT_WEIGHTS.keys())]

    if args.fast:
        # Starting in 2016 leaves 500 days of history, so the first forecast
        # falls around the end of 2017 and the test covers 2018-2022.
        returns = returns.loc["2016-01-01":"2022-12-31"]
        n_sims = 5_000
        output_file = RESULTS_DIR / "backtest_fast.csv"
    else:
        n_sims = 50_000
        output_file = RESULTS_DIR / "backtest.csv"

    # Recompute the P&L from the (possibly shortened) returns so both
    # series always share exactly the same index.
    pnl = portfolio_pnl(returns)

    print(f"Backtest on {len(returns)} days, window = 500, n_sims = {n_sims:,}")
    start = time.time()
    results = rolling_backtest(returns, pnl, DEFAULT_WEIGHTS,
                               window=500, n_sims=n_sims)
    elapsed = time.time() - start

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file)
    print(f"\nDone in {elapsed:.0f} s. Saved: {output_file}")
    print(f"Period: {results.index[0].date()} -> {results.index[-1].date()}")

    summary = violation_summary(results)
    pd.options.display.float_format = "{:.2f}".format
    print("\n=== Violation rates (observed vs expected) ===\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()