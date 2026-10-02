"""Build the portfolio P&L, run sanity checks and save two figures.

Run from the project root:
    python scripts/build_pnl.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from riskengine.portfolio import (
    compute_log_returns,
    load_prices,
    portfolio_pnl,
)

PNL_FILE = Path("data/processed/pnl.csv")
RETURNS_FILE = Path("data/processed/returns.csv")
FIGURES_DIR = Path("figures")


def print_sanity_checks(pnl, v0):
    """Print the numbers that tell us whether the data looks right."""
    print("=== Sanity checks ===")
    print(f"Observations : {len(pnl)}")
    print(f"Period       : {pnl.index[0].date()} -> {pnl.index[-1].date()}")

    # Annualised volatility = daily std * sqrt(252 trading days).
    # We divide by v0 to express it as a percentage of portfolio value.
    daily_vol = pnl.std() / v0
    print(f"Annualised volatility : {daily_vol * np.sqrt(252):.2%}  "
          f"(expected: roughly 8% to 12%)")

    # scipy's kurtosis() returns the EXCESS kurtosis (normal law = 0).
    print(f"Skewness        : {stats.skew(pnl):.3f}  (expected: negative)")
    print(f"Excess kurtosis : {stats.kurtosis(pnl):.3f}  "
          f"(expected: clearly positive)")

    print("\nFive worst days (P&L in EUR):")
    print(pnl.nsmallest(5).round(0).to_string())


def plot_pnl_series(pnl):
    """Figure 1: the daily P&L over time."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pnl.index, pnl.values, linewidth=0.6, color="tab:blue")
    ax.set_title("Daily portfolio P&L")
    ax.set_xlabel("Date")
    ax.set_ylabel("P&L (EUR)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_series.png", dpi=150)
    plt.close(fig)


def plot_pnl_histogram(pnl):
    """Figure 2: histogram of the P&L vs a normal density.

    The normal curve has the SAME mean and standard deviation as the data.
    Left panel: normal scale. Right panel: log scale on the y axis, which
    makes the fat tails easy to see.
    """
    mu, sigma = pnl.mean(), pnl.std()
    x = np.linspace(pnl.min(), pnl.max(), 500)
    normal_density = stats.norm.pdf(x, loc=mu, scale=sigma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log_scale in zip(axes, [False, True]):
        ax.hist(pnl, bins=120, density=True, alpha=0.6,
                color="tab:blue", label="Empirical P&L")
        ax.plot(x, normal_density, color="tab:red", linewidth=2,
                label="Normal (same mean and std)")
        ax.set_xlabel("Daily P&L (EUR)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-8)
            ax.set_title("Log scale: the tails")
        else:
            ax.set_title("Distribution of daily P&L")
        ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_histogram.png", dpi=150)
    plt.close(fig)


def main():
    prices = load_prices()
    returns = compute_log_returns(prices)
    pnl = portfolio_pnl(returns)

    # Create the output folders if they do not exist, then save.
    PNL_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_csv(RETURNS_FILE)
    pnl.to_csv(PNL_FILE)

    print_sanity_checks(pnl, v0=1_000_000.0)
    plot_pnl_series(pnl)
    plot_pnl_histogram(pnl)
    print(f"\nSaved: {PNL_FILE}, {RETURNS_FILE} and 2 figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()