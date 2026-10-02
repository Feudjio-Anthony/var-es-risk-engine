"""Validate Monte Carlo against the parametric method and study convergence.

Run from the project root:
    python scripts/validate_monte_carlo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskengine.measures import monte_carlo_var_es, parametric_var_es
from riskengine.portfolio import (
    DEFAULT_WEIGHTS,
    V0,
    compute_log_returns,
    load_prices,
)

ALPHAS = [0.95, 0.975, 0.99]
N_VALUES = [1_000, 5_000, 10_000, 50_000, 200_000]
N_REPEATS = 30          # number of different seeds per N
FIGURES_DIR = Path("figures")


def cross_validation(returns, weights):
    """Compare Monte Carlo with the parametric (analytical) solution."""
    rows = []
    for alpha in ALPHAS:
        var_p, es_p = parametric_var_es(returns, weights, alpha, v0=V0)
        for n_sims in (50_000, 1_000_000):
            var_mc, es_mc = monte_carlo_var_es(
                returns, weights, alpha, v0=V0, n_sims=n_sims)
            rows.append({
                "alpha": alpha,
                "n_sims": n_sims,
                "VaR param": var_p,
                "VaR MC": var_mc,
                "VaR gap %": 100 * (var_mc / var_p - 1),
                "ES param": es_p,
                "ES MC": es_mc,
                "ES gap %": 100 * (es_mc / es_p - 1),
            })
    table = pd.DataFrame(rows).set_index(["alpha", "n_sims"])
    print("=== Cross-validation: Monte Carlo vs parametric ===\n")
    print(table.round(2).to_string())


def convergence_study(returns, weights):
    """Study how the 99% VaR estimate fluctuates as N grows."""
    alpha = 0.99
    var_param, _ = parametric_var_es(returns, weights, alpha, v0=V0)

    # results[n] = list of VaR estimates, one per seed.
    results = {}
    for n_sims in N_VALUES:
        results[n_sims] = np.array([
            monte_carlo_var_es(returns, weights, alpha, v0=V0,
                               n_sims=n_sims, seed=seed)[0]
            for seed in range(N_REPEATS)
        ])

    # Spread (standard deviation) of the estimate across seeds.
    stds = np.array([results[n].std(ddof=1) for n in N_VALUES])

    print("\n=== Convergence of the 99% VaR (30 seeds per N) ===\n")
    print(f"Parametric reference: {var_param:,.0f} EUR\n")
    for n_sims, std in zip(N_VALUES, stds):
        print(f"N = {n_sims:>8,d}   std across seeds = {std:8.1f} EUR "
              f"({100 * std / var_param:.2f}% of VaR)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: every estimate, with the analytical value as a red line.
    ax = axes[0]
    for n_sims in N_VALUES:
        ax.scatter([n_sims] * N_REPEATS, results[n_sims],
                   alpha=0.4, color="tab:blue", s=15)
    ax.axhline(var_param, color="tab:red", linewidth=2,
               label="Parametric (exact)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulations N")
    ax.set_ylabel("99% VaR (EUR)")
    ax.set_title("Monte Carlo estimates (30 seeds per N)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Right: the spread on log-log axes, against a 1/sqrt(N) reference line.
    ax = axes[1]
    n_arr = np.array(N_VALUES, dtype=float)
    reference = stds[0] * np.sqrt(n_arr[0]) / np.sqrt(n_arr)
    ax.loglog(n_arr, stds, "o-", color="tab:blue", label="Observed spread")
    ax.loglog(n_arr, reference, "--", color="tab:red",
              label="Theory: 1/sqrt(N)")
    ax.set_xlabel("Number of simulations N")
    ax.set_ylabel("Std of VaR estimate (EUR)")
    ax.set_title("Error decreases as 1/sqrt(N)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "mc_convergence.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIGURES_DIR / 'mc_convergence.png'}")


def main():
    prices = load_prices()
    returns = compute_log_returns(prices)

    # Same order for the returns columns and the weight vector.
    cols = list(DEFAULT_WEIGHTS.keys())
    weights = [DEFAULT_WEIGHTS[c] for c in cols]
    returns = returns[cols]

    cross_validation(returns, weights)
    convergence_study(returns, weights)


if __name__ == "__main__":
    main()