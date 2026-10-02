"""Run the statistical validation tests on the saved backtest.

Run from the project root:
    python scripts/run_stat_tests.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine.backtest import METHODS, count_violations
from riskengine.tests_stat import (
    basel_traffic_light,
    basel_zone_shares,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    transition_probabilities,
)

ALPHAS = (0.95, 0.975, 0.99)
RESULTS_FILE = Path("results/backtest.csv")
OUTPUT_FILE = Path("results/stat_tests.csv")
SIGNIFICANCE = 0.05


def format_p(p):
    """Readable p-value: scientific notation when it is very small."""
    if pd.isna(p):
        return "-"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def main():
    results = pd.read_csv(RESULTS_FILE, index_col=0, parse_dates=True)

    rows = []
    for method in METHODS:
        for alpha in ALPHAS:
            tag = round(alpha * 1000)
            v = count_violations(results, method, tag).to_numpy()

            _, p_uc, x, expected = kupiec_pof(v, alpha)
            _, p_ind = christoffersen_independence(v)
            _, p_cc = conditional_coverage(v, alpha)
            pi01, pi11 = transition_probabilities(v)

            row = {
                "method": method,
                "alpha": alpha,
                "T": len(v),
                "violations": x,
                "expected": expected,
                "observed %": 100 * x / len(v),
                "p_kupiec": p_uc,
                "p_indep": p_ind,
                "p_cc": p_cc,
                "pi01": pi01,
                "pi11": pi11,
                "last250_viol": np.nan,
                "last250_zone": None,
                "share_yellow": np.nan,
                "share_red": np.nan,
                "worst250": np.nan,
            }

            # The Basel traffic light is defined for a 99% VaR only.
            if alpha == 0.99:
                last_x, zone = basel_traffic_light(v)
                shares = basel_zone_shares(v)
                row.update({
                    "last250_viol": last_x,
                    "last250_zone": zone,
                    "share_yellow": shares["yellow"],
                    "share_red": shares["red"],
                    "worst250": shares["max_violations"],
                })

            rows.append(row)

    table = pd.DataFrame(rows).set_index(["method", "alpha"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE)

    # ---- Display (strings, so tiny p-values stay readable) ----
    display = pd.DataFrame(index=table.index)
    display["viol"] = table["violations"]
    display["expected"] = table["expected"].round(1)
    display["obs %"] = table["observed %"].round(2)
    display["p Kupiec"] = table["p_kupiec"].map(format_p)
    display["p Indep"] = table["p_indep"].map(format_p)
    display["p CondCov"] = table["p_cc"].map(format_p)
    display["pi01"] = table["pi01"].round(3)
    display["pi11"] = table["pi11"].round(3)

    def rejected(r):
        names = [name for name, col in
                 (("Kupiec", "p_kupiec"), ("Indep", "p_indep"), ("CC", "p_cc"))
                 if r[col] < SIGNIFICANCE]
        return ", ".join(names) if names else "none"

    display["rejected at 5%"] = table.apply(rejected, axis=1)

    print(f"=== Statistical validation ({int(table['T'].iloc[0])} forecasts) ===\n")
    print(display.to_string())

    print("\n=== Basel traffic light (99% VaR, 250-day windows) ===\n")
    basel = table.xs(0.99, level="alpha")[
        ["last250_viol", "last250_zone", "share_yellow", "share_red", "worst250"]
    ].copy()
    basel["share_yellow"] = (100 * basel["share_yellow"]).round(1)
    basel["share_red"] = (100 * basel["share_red"]).round(1)
    basel.columns = ["last window viol", "last window zone",
                     "% windows yellow", "% windows red", "worst window viol"]
    print(basel.to_string())

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()