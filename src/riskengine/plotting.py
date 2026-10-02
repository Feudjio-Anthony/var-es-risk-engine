"""Figures of the risk report.

Each function builds ONE figure, saves it as a PNG (150 dpi) and returns the
matplotlib Figure object. All figures share the same sober palette.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .backtest import METHODS, violation_summary

# One colour per method, the same on every figure.
COLORS = {"hist": "#1f4e79", "param": "#c0392b", "mc": "#1e8449"}
LABELS = {"hist": "Historical", "param": "Parametric", "mc": "Monte Carlo"}

# One colour per confidence level (used by the violation-rate figure).
ALPHA_COLORS = {0.95: "#9db4c9", 0.975: "#5b84b1", 0.99: "#1f3b57"}

# Global style: light grid only, no top/right frame, readable font size.
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 9,
    "savefig.dpi": 150,
})


def _pct(alpha_tag):
    """990 -> '99%', 975 -> '97.5%'."""
    return f"{alpha_tag / 10:g}%"


def _format_dates(ax):
    """Readable date labels on the x axis."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save(fig, out):
    """Create the output folder if needed, then save the figure."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


# ----------------------------------------------------------------------
# Figure 1: P&L and VaR lines over the whole period
# ----------------------------------------------------------------------
def plot_pnl_vs_var(results, alpha_tag=990, violation_method="hist",
                    out="figures/fig1_pnl_var.png"):
    """Daily P&L, the VaR of each method (plotted as a NEGATIVE number,
    so it can be compared with the signed P&L) and the violations."""
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(results.index, results["pnl"], lw=0.5, color="#95a5a6",
            label="Daily P&L", zorder=1)

    for m in METHODS:
        # Monte Carlo is dashed, otherwise it hides under the parametric line.
        linestyle = "--" if m == "mc" else "-"
        ax.plot(results.index, -results[f"{m}_var_{alpha_tag}"],
                lw=1.1, ls=linestyle, color=COLORS[m], zorder=2,
                label=f"{_pct(alpha_tag)} VaR, {LABELS[m]}")

    # A violation = the loss is larger than the VaR: pnl < -var.
    viol = results["pnl"] < -results[f"{violation_method}_var_{alpha_tag}"]
    ax.scatter(results.index[viol.to_numpy()], results.loc[viol, "pnl"],
               s=14, color="#e74c3c", zorder=3,
               label=f"Violations of {LABELS[violation_method]} VaR "
                     f"({int(viol.sum())})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily P&L (EUR)")
    ax.set_title(f"Realised P&L and {_pct(alpha_tag)} VaR "
                 "- multi-asset portfolio, 1-day horizon")
    _format_dates(ax)
    ax.legend(loc="lower left")
    _save(fig, out)
    return fig


# ----------------------------------------------------------------------
# Figure 2: ES / VaR ratio, empirical vs Gaussian theory
# ----------------------------------------------------------------------
def plot_es_var_ratio(results, alphas=(0.95, 0.975, 0.99),
                      out="figures/fig2_es_var_ratio.png"):
    """Average ES/VaR ratio of each method, against the Gaussian theory.

    A ratio of 1.0 would mean 'ES = VaR' (no tail beyond the VaR).
    The Gaussian theory is pdf(z) / ((1 - alpha) * z).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(alphas))
    offsets = {"hist": -0.22, "param": 0.0, "mc": 0.22}

    for m in METHODS:
        ratios = np.array([
            (results[f"{m}_es_{round(a * 1000)}"]
             / results[f"{m}_var_{round(a * 1000)}"]).mean()
            for a in alphas
        ])
        pos = x + offsets[m]
        ax.vlines(pos, 1.0, ratios, color=COLORS[m], lw=2, alpha=0.5, zorder=2)
        ax.scatter(pos, ratios, s=90, color=COLORS[m], zorder=3,
                   label=LABELS[m])
        for p, r in zip(pos, ratios):
            ax.annotate(f"{r:.2f}", (p, r), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)

    # Gaussian theory for each confidence level.
    for i, a in enumerate(alphas):
        z = stats.norm.ppf(a)
        theory = stats.norm.pdf(z) / ((1 - a) * z)
        extra = {"label": "Gaussian theory"} if i == 0 else {}
        ax.hlines(theory, x[i] - 0.38, x[i] + 0.38, colors="black",
                  linestyles="--", lw=1.2, zorder=1, **extra)
        ax.annotate(f"{theory:.2f}", (x[i] + 0.38, theory),
                    textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a * 100:g}%" for a in alphas])
    ax.set_xlabel("Confidence level")
    ax.set_ylabel("Mean ES / VaR ratio (1.0 = ES equals VaR)")
    ax.set_title("Heavy tails: empirical ES/VaR vs Gaussian theory")
    ax.set_ylim(1.0, None)
    ax.legend(loc="upper right")
    _save(fig, out)
    return fig


# ----------------------------------------------------------------------
# Figure 3: zoom on the March 2020 crash
# ----------------------------------------------------------------------
def plot_covid_zoom(results, alpha_tag=990, start="2020-02-03",
                    end="2020-04-17", out="figures/fig3_covid_zoom.png"):
    """Two months around the COVID crash, with a dilated time scale.

    The historical VaR has not 'seen' the crash yet: it stays low during
    the first days, then adjusts too late.
    """
    zoom = results.loc[start:end]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(zoom.index, zoom["pnl"], width=0.8, color="#bdc3c7",
           label="Daily P&L", zorder=1)

    for m in METHODS:
        linestyle = "--" if m == "mc" else "-"
        ax.plot(zoom.index, -zoom[f"{m}_var_{alpha_tag}"], lw=1.8,
                ls=linestyle, color=COLORS[m], drawstyle="steps-mid",
                zorder=2, label=f"{_pct(alpha_tag)} VaR, {LABELS[m]}")

    # Mark the violations of the historical and parametric VaR.
    marker_style = {
        "hist": dict(marker="o", s=70, facecolors="none",
                     edgecolors=COLORS["hist"], linewidths=1.6),
        "param": dict(marker="x", s=55, color=COLORS["param"], linewidths=1.8),
    }
    for m, style in marker_style.items():
        viol = zoom["pnl"] < -zoom[f"{m}_var_{alpha_tag}"]
        ax.scatter(zoom.index[viol.to_numpy()], zoom.loc[viol, "pnl"],
                   zorder=3, label=f"Violations, {LABELS[m]} "
                                   f"({int(viol.sum())})", **style)

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily P&L (EUR)")
    ax.set_title("Zoom on the March 2020 crash: how fast does each VaR react?")
    _format_dates(ax)
    ax.legend(loc="lower left")
    _save(fig, out)
    return fig


# ----------------------------------------------------------------------
# Figure 4: observed vs expected violation rates
# ----------------------------------------------------------------------
def plot_violation_rates(results, alphas=(0.95, 0.975, 0.99),
                         out="figures/fig4_violation_rates.png"):
    """Observed violation rate per method and level, with the expected rate
    (1 - alpha) drawn as a dashed horizontal line."""
    summary = violation_summary(results, alphas)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(METHODS))
    width = 0.25

    for k, a in enumerate(alphas):
        observed = [summary.loc[(m, a), "observed %"] for m in METHODS]
        bars = ax.bar(x + (k - 1) * width, observed, width,
                      color=ALPHA_COLORS[a], zorder=2,
                      label=f"{a * 100:g}% VaR")
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8.5, padding=2)

        expected = 100 * (1 - a)
        ax.axhline(expected, color=ALPHA_COLORS[a], ls="--", lw=1.4, zorder=1)
        ax.text(len(METHODS) - 0.42, expected, f"expected {expected:g}%",
                va="bottom", ha="right", fontsize=8.5,
                color=ALPHA_COLORS[a])

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_xlabel("Method")
    ax.set_ylabel("Violation rate (% of days)")
    ax.set_title("Observed vs expected violation rates (out-of-sample)")
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.legend(loc="upper left")
    _save(fig, out)
    return fig