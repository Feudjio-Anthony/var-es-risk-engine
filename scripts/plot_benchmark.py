"""Draw the performance figure from the benchmark results.

Run from the project root, after scripts/benchmark_cpp.py:
    python scripts/plot_benchmark.py

The figure shows SPEED-UP RATIOS only. On a laptop the absolute speed drifts
(turbo boost, heat) by a factor of two between runs, but the ratios between
functions timed back to back stay stable, so ratios are what we can trust.

Two scopes of measurement are shown, from narrow to wide:
    - kernel   : scenario generation only (curves, one point per N);
    - pipeline : the full rolling backtest (dotted line).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # no window: write the file only
import matplotlib.pyplot as plt             # noqa: E402
import pandas as pd                         # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

BENCH = Path("results/benchmark.csv")
PIPELINE = Path("results/benchmark_pipeline.csv")
OUTPUT = Path("figures/fig5_benchmark.png")

BLUE, GREEN, ORANGE = "#2a6fbb", "#3a9a5b", "#d9822b"
NOISE_LIMIT = 1.5     # median / minimum above this = noisy measurement


def _thousands(x, _pos):
    """12000 -> '12,000' (and 1000000 -> '1,000,000')."""
    return f"{x:,.0f}"


def _draw(ax, n, ratio, noisy, color, marker, label, style):
    """One curve; noisy points are drawn hollow so nobody over-reads them."""
    ax.plot(n, ratio, style, color=color, label=label)
    for x, y, bad in zip(n, ratio, noisy):
        ax.plot(x, y, marker, color=color,
                markerfacecolor="white" if bad else color,
                markersize=7, zorder=3)


def main():
    df = pd.read_csv(BENCH)
    pipe = pd.read_csv(PIPELINE).set_index("backend")
    pipe_speedup = pipe.loc["python", "total_s"] / pipe.loc["cpp", "total_s"]
    n_forecasts = int(pipe.loc["cpp", "forecasts"])

    n = df["n_sims"]
    noisy = (df["noise_numpy"] > NOISE_LIMIT) | (df["noise_cpp"] > NOISE_LIMIT)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    _draw(ax, n, df["speedup_vs_numpy"], noisy, BLUE, "o",
          "Kernel: C++ vs NumPy", "-")
    _draw(ax, n, df["speedup_vs_folded"], noisy, GREEN, "s",
          "Kernel: C++ vs NumPy with weights folded into Cholesky", "--")
    ax.axhline(pipe_speedup, color=ORANGE, lw=2, ls=":",
               label=f"Full backtest ({n_forecasts:,} forecasts): {pipe_speedup:.2f}x")
    ax.axhline(1.0, color="grey", lw=1)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Number of scenarios N")
    ax.set_ylabel("Speed-up of the C++ module (x)")
    ax.set_title("C++ vs NumPy: the gain shrinks as the scope widens")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    # Legend below the plot, so it hides no data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.text(0.01, 0.01,
             "Hollow marker = noisy measurement (median/minimum > "
             f"{NOISE_LIMIT}). 1x = same speed.",
             fontsize=8, color="grey")
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()