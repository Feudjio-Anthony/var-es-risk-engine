"""Build the four figures of the risk report from results/backtest.csv.

Run from the project root:
    python scripts/make_figures.py
"""
import matplotlib

# Non-interactive backend: we only save PNG files, we never open a window.
# It must be set BEFORE importing riskengine.plotting (which imports pyplot).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from riskengine.plotting import (  # noqa: E402
    plot_covid_zoom,
    plot_es_var_ratio,
    plot_pnl_vs_var,
    plot_violation_rates,
)


def main():
    results = pd.read_csv("results/backtest.csv", index_col=0, parse_dates=True)

    for name, func in [
        ("fig1_pnl_var.png", plot_pnl_vs_var),
        ("fig2_es_var_ratio.png", plot_es_var_ratio),
        ("fig3_covid_zoom.png", plot_covid_zoom),
        ("fig4_violation_rates.png", plot_violation_rates),
    ]:
        fig = func(results)
        plt.close(fig)       # free the memory before building the next one
        print(f"Saved: figures/{name}")


if __name__ == "__main__":
    main()