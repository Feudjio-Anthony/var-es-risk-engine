"""Print the numbers needed to write the commentary of the figures.

Run from the project root:
    python scripts/figure_facts.py
"""
import pandas as pd

from riskengine.backtest import METHODS, count_violations


def main():
    results = pd.read_csv("results/backtest.csv", index_col=0, parse_dates=True)
    years = results.index.year

    print("=== 99% violations per year (clustering) ===\n")
    per_year = pd.DataFrame({
        m: count_violations(results, m, 990).groupby(years).sum()
        for m in METHODS
    })
    per_year["days"] = results.groupby(years).size()
    print(per_year.to_string())

    print("\n=== Smoothness: daily relative change of the 99% VaR ===\n")
    for m in METHODS:
        change = results[f"{m}_var_990"].pct_change().abs()
        print(f"{m:>6}: mean {100 * change.mean():.2f}% per day, "
              f"max {100 * change.max():.1f}%")

    print("\n=== Ghost effect: 5 biggest one-day DROPS of the historical "
          "99% VaR ===\n")
    drops = results["hist_var_990"].pct_change().nsmallest(5)
    for date, drop in drops.items():
        print(f"{date.date()}   {100 * drop:.1f}%")

    print("\n=== Mean ES/VaR ratio per method and level ===\n")
    for m in METHODS:
        parts = []
        for tag in (950, 975, 990):
            ratio = (results[f"{m}_es_{tag}"] / results[f"{m}_var_{tag}"]).mean()
            parts.append(f"{tag / 10:g}%: {ratio:.2f}")
        print(f"{m:>6}: " + "   ".join(parts))

    print("\n=== March 2020 timeline (99% VaR, EUR) ===\n")
    zoom = results.loc["2020-02-19":"2020-04-03",
                       ["pnl", "hist_var_990", "param_var_990"]].copy()
    zoom["hist viol"] = zoom["pnl"] < -zoom["hist_var_990"]
    zoom["param viol"] = zoom["pnl"] < -zoom["param_var_990"]
    zoom[["pnl", "hist_var_990", "param_var_990"]] = (
        zoom[["pnl", "hist_var_990", "param_var_990"]].round(0))
    print(zoom.to_string())


if __name__ == "__main__":
    main()