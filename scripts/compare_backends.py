"""Compare two backtest result files: Python backend vs C++ backend.

Run from the project root, after both backtests have been produced:
    python scripts/compare_backends.py            (full results)
    python scripts/compare_backends.py --fast     (quick results)

What we expect:
    - hist_* and param_* columns IDENTICAL (they do not use the backend);
    - mc_* columns slightly different (different random numbers);
    - violation counts nearly identical.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.backtest import count_violations

RESULTS_DIR = Path("results")
ALPHAS = (0.95, 0.975, 0.99)


def main():
    parser = argparse.ArgumentParser(description="Compare Python and C++ backtests")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    stem = "backtest_fast" if args.fast else "backtest"
    py = pd.read_csv(RESULTS_DIR / f"{stem}.csv", index_col="date", parse_dates=True)
    cpp = pd.read_csv(RESULTS_DIR / f"{stem}_cpp.csv", index_col="date", parse_dates=True)

    if not py.index.equals(cpp.index):
        raise SystemExit("The two files do not cover the same dates.")

    # ---- 1. Columns that must NOT change ----
    untouched = [c for c in py.columns if not c.startswith("mc_")]
    same = np.allclose(py[untouched], cpp[untouched], rtol=0, atol=1e-9)
    print(f"hist / param / pnl columns identical: {same}")

    # ---- 2. Monte Carlo columns: relative gap forecast by forecast ----
    mc_cols = [c for c in py.columns if c.startswith("mc_")]
    gap = (cpp[mc_cols] / py[mc_cols] - 1).abs() * 100
    print("\nMonte Carlo forecasts, |C++ / Python - 1| in % (day by day):")
    print(gap.describe().loc[["mean", "50%", "max"]].T
          .rename(columns={"50%": "median"}).round(3).to_string())

    # ---- 3. Violation counts ----
    rows = []
    for a in ALPHAS:
        tag = round(a * 1000)
        v_py = count_violations(py, "mc", tag)
        v_cpp = count_violations(cpp, "mc", tag)
        rows.append({
            "alpha": a,
            "python": int(v_py.sum()),
            "cpp": int(v_cpp.sum()),
            "difference": int(v_cpp.sum() - v_py.sum()),
            "days that differ": int((v_py != v_cpp).sum()),
            "expected": round((1 - a) * len(py), 1),
        })
    print("\nMonte Carlo violations (out of "
          f"{len(py)} forecasts):")
    print(pd.DataFrame(rows).set_index("alpha").to_string())


if __name__ == "__main__":
    main()