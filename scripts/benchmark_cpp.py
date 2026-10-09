"""Benchmark of the two Monte Carlo backends (NumPy vs C++).

Run from the project root, with the machine idle and plugged in:
    python scripts/benchmark_cpp.py            (full benchmark, a few minutes)
    python scripts/benchmark_cpp.py --quick    (smoke test, seconds)

Four measurements, from the narrowest to the widest:
    1. KERNEL    : only "generate scenarios -> portfolio P&L".
    2. FUNCTION  : the whole monte_carlo_var_es call (adds estimation of the
                   mean/covariance with pandas, and the quantile / ES step).
    3. BREAKDOWN : where the time of ONE backtest forecast goes.
    4. PIPELINE  : the real rolling backtest, run end to end with each backend.

Benchmark rules: warm-up call discarded, MINIMUM of several repeats (least
polluted by the operating system), identical workload for every backend,
and the compared functions INTERLEAVED so they share the same conditions.
"""
import argparse
import ctypes
import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from riskengine import mc_engine
from riskengine.backtest import rolling_backtest
from riskengine.measures import monte_carlo_var_es
from riskengine.portfolio import (
    DEFAULT_WEIGHTS,
    compute_log_returns,
    load_prices,
    portfolio_pnl,
)

RESULTS_DIR = Path("results")
V0 = 1_000_000.0
WINDOW = 500


def chrono_group(fns, n_repeat):
    """Time several functions fairly, and return {name: (minimum, median)}.

    The functions are INTERLEAVED: in each round, every function runs once,
    one after the other. On a laptop the CPU speed drifts (turbo boost,
    heat, background tasks); interleaving makes all the functions go through
    the same conditions, so their RATIOS stay meaningful even when the
    absolute speed moves. Each function is first called once as a warm-up
    (allocation, caches, lazy imports) and that call is thrown away.

    The MINIMUM is the least polluted by the operating system. The MEDIAN
    is kept too: a median far above the minimum means a noisy measurement.
    """
    for fn in fns.values():
        fn()
    times = {name: [] for name in fns}
    gc.disable()          # the garbage collector must not fire mid-measure
    try:
        for _ in range(n_repeat):
            for name, fn in fns.items():
                t0 = time.perf_counter()
                fn()
                times[name].append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return {name: (min(t), statistics.median(t)) for name, t in times.items()}


def repeats_for(n):
    """More repeats for small (fast, noisy) calls, fewer for big ones."""
    return max(10, min(50, 2_000_000 // n))


def raise_priority():
    """Ask Windows to schedule this process with a higher priority.

    Fewer interruptions from background tasks. Needs no administrator
    rights; silently ignored on other systems or if refused.
    """
    try:
        HIGH_PRIORITY_CLASS = 0x00000080
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
    except Exception:
        pass


# ----------------------------------------------------------------------
# The three scenario generators compared in the KERNEL measurement
# ----------------------------------------------------------------------
def kernel_numpy(mu, cov_reg, w, n, seed=1):
    """The Python backend, exactly as in measures.py (reference)."""
    chol = np.linalg.cholesky(cov_reg)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(w)))
    return V0 * ((mu + z @ chol.T) @ w)


def kernel_numpy_folded(mu, cov_reg, w, n, seed=1):
    """NumPy with the same algebraic trick as the C++ code.

    The weights are folded into the Cholesky factor once (c = L' w), so the
    N x 5 matrix of simulated returns is never built: one matrix-vector
    product replaces two. This isolates how much of the C++ gain comes
    from the algebra and how much from the language / loop fusion.
    """
    chol = np.linalg.cholesky(cov_reg)
    c = chol.T @ w
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(w)))
    return V0 * (mu @ w + z @ c)


def kernel_cpp(mu, cov_reg, w, n, seed=1):
    return mc_engine.simulate_pnl(mu, cov_reg, w, n, V0, seed)


# ----------------------------------------------------------------------
# Hardware and software context
# ----------------------------------------------------------------------
def _powershell(command):
    """Run a PowerShell one-liner and return its output ('' on failure)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def describe_machine():
    """Text description of the machine, saved next to the results."""
    cpu = ram = ""
    if platform.system() == "Windows":
        cpu = _powershell("(Get-CimInstance Win32_Processor).Name")
        ram_bytes = _powershell(
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        if ram_bytes.isdigit():
            ram = f"{int(ram_bytes) / 1024 ** 3:.0f} GB"
    lines = [
        f"CPU            : {cpu or platform.processor() or 'unknown'}",
        f"Logical cores  : {os.cpu_count()}",
        f"RAM            : {ram or 'unknown'}",
        f"OS             : {platform.platform()}",
        f"Python         : {sys.version.split()[0]}",
        f"NumPy          : {np.__version__}",
        f"pandas         : {pd.__version__}",
        f"C++ compiler   : MSVC, flags /O2 /GL /DNDEBUG /MD (see build log)",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def measure_kernel_and_function(returns, sizes):
    """Measurements 1 and 2, for every N (all variants interleaved)."""
    window = returns.iloc[-WINDOW:]
    mu = window.mean().to_numpy()
    cov_reg = window.cov().to_numpy() + 1e-12 * np.eye(window.shape[1])
    w = np.asarray([DEFAULT_WEIGHTS[c] for c in window.columns], dtype=float)

    rows = []
    for n in sizes:
        t = chrono_group({
            "numpy": lambda: kernel_numpy(mu, cov_reg, w, n),
            "folded": lambda: kernel_numpy_folded(mu, cov_reg, w, n),
            "cpp": lambda: kernel_cpp(mu, cov_reg, w, n),
            "f_py": lambda: monte_carlo_var_es(
                window, w, 0.99, V0, n_sims=n, backend="python"),
            "f_cpp": lambda: monte_carlo_var_es(
                window, w, 0.99, V0, n_sims=n, backend="cpp"),
        }, repeats_for(n))

        t_np, t_fold, t_cpp = t["numpy"][0], t["folded"][0], t["cpp"][0]
        f_py, f_cpp = t["f_py"][0], t["f_cpp"][0]
        rows.append({
            "n_sims": n,
            "numpy_ms": 1000 * t_np,
            "numpy_folded_ms": 1000 * t_fold,
            "cpp_ms": 1000 * t_cpp,
            "speedup_vs_numpy": t_np / t_cpp,
            "speedup_vs_folded": t_fold / t_cpp,
            "cpp_Mscen_per_s": n / t_cpp / 1e6,
            "numpy_Mscen_per_s": n / t_np / 1e6,
            "function_python_ms": 1000 * f_py,
            "function_cpp_ms": 1000 * f_cpp,
            "function_speedup": f_py / f_cpp,
            # median / minimum: close to 1 = quiet machine, >> 1 = noisy.
            "noise_numpy": t["numpy"][1] / t_np,
            "noise_cpp": t["cpp"][1] / t_cpp,
        })
        print(f"  N = {n:>9,d} done")
    return pd.DataFrame(rows)


def measure_breakdown(returns, n_sims, n_repeat=40):
    """Measurement 3: the cost of each stage of ONE backtest forecast."""
    window = returns.iloc[-WINDOW:]
    w = np.asarray([DEFAULT_WEIGHTS[c] for c in window.columns], dtype=float)
    mu = window.mean().to_numpy()
    cov_reg = window.cov().to_numpy() + 1e-12 * np.eye(window.shape[1])
    pnl_np = kernel_numpy(mu, cov_reg, w, n_sims)

    def quantile_and_tail():
        q = np.quantile(pnl_np, 0.01)
        return -pnl_np[pnl_np <= q].mean()

    stages = {
        "pandas mean + cov (500 days)": lambda: (window.mean(), window.cov()),
        "simulate scenarios, NumPy": lambda: kernel_numpy(mu, cov_reg, w, n_sims),
        "simulate scenarios, C++": lambda: kernel_cpp(mu, cov_reg, w, n_sims),
        "quantile + ES tail": quantile_and_tail,
    }
    t = chrono_group(stages, n_repeat)
    return pd.DataFrame([{"stage": name, "ms": 1000 * t[name][0]}
                         for name in stages])


def measure_pipeline(returns, n_sims, n_rounds):
    """Measurement 4: the real rolling backtest, backends alternated.

    Alternating python / cpp / python / cpp ... inside the same session
    means both see the same machine conditions (temperature, background
    load). The minimum over the rounds is kept for each backend.
    """
    pnl = portfolio_pnl(returns)
    n_forecasts = len(returns) - WINDOW
    best = {"python": np.inf, "cpp": np.inf}
    for r in range(n_rounds + 1):                       # round 0 = warm-up
        for backend in ("python", "cpp"):
            t0 = time.perf_counter()
            rolling_backtest(returns, pnl, DEFAULT_WEIGHTS, window=WINDOW,
                             n_sims=n_sims, backend=backend, verbose=False)
            dt = time.perf_counter() - t0
            if r > 0:
                best[backend] = min(best[backend], dt)
            print(f"  round {r} {backend:>6}: {dt:6.1f} s"
                  + ("  (warm-up, discarded)" if r == 0 else ""))
    return pd.DataFrame([{
        "backend": b,
        "forecasts": n_forecasts,
        "total_s": best[b],
        "ms_per_forecast": 1000 * best[b] / n_forecasts,
    } for b in ("python", "cpp")])


def main():
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs C++")
    parser.add_argument("--quick", action="store_true",
                        help="few sizes, few repeats, short pipeline (smoke test)")
    args = parser.parse_args()

    prices = load_prices()
    returns = compute_log_returns(prices)[list(DEFAULT_WEIGHTS.keys())]

    if args.quick:
        sizes = (1_000, 100_000)
        # About 250 forecasts (2 years of history for the 500-day window).
        pipe_returns, pipe_rounds = returns.loc["2017-01-01":"2019-12-31"], 1
    else:
        sizes = (1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000)
        # 2016-2022 = about 1,250 forecasts at the production setting
        # (50,000 simulations): representative, without waiting 2 minutes
        # per run. The cost per forecast does not depend on the period.
        pipe_returns, pipe_rounds = returns.loc["2016-01-01":"2022-12-31"], 3

    raise_priority()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = describe_machine()
    print("=== Machine ===\n" + env + "\n")
    (RESULTS_DIR / "benchmark_env.txt").write_text(env + "\n", encoding="utf-8")

    print("=== 1-2. Kernel and function, by number of scenarios ===")
    bench = measure_kernel_and_function(returns, sizes)
    bench.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    show = ["n_sims", "numpy_ms", "numpy_folded_ms", "cpp_ms",
            "speedup_vs_numpy", "speedup_vs_folded", "cpp_Mscen_per_s",
            "noise_numpy", "noise_cpp"]
    print("\nKERNEL (scenarios -> P&L only)")
    print(bench[show].to_string(index=False, float_format="{:.2f}".format))
    print("\nFUNCTION (monte_carlo_var_es, includes pandas + quantile + ES)")
    print(bench[["n_sims", "function_python_ms", "function_cpp_ms",
                 "function_speedup"]]
          .to_string(index=False, float_format="{:.2f}".format))

    print("\n=== 3. Breakdown of one forecast (50,000 scenarios) ===")
    breakdown = measure_breakdown(returns, 50_000)
    breakdown.to_csv(RESULTS_DIR / "benchmark_breakdown.csv", index=False)
    print(breakdown.to_string(index=False, float_format="{:.3f}".format))

    print("\n=== 4. Full pipeline (rolling backtest, 50,000 scenarios) ===")
    pipeline = measure_pipeline(pipe_returns, 50_000, pipe_rounds)
    pipeline.to_csv(RESULTS_DIR / "benchmark_pipeline.csv", index=False)
    print(pipeline.to_string(index=False, float_format="{:.2f}".format))
    ratio = (pipeline.loc[0, "total_s"] / pipeline.loc[1, "total_s"])
    print(f"\nPipeline speed-up (Python / C++): {ratio:.2f}x")


if __name__ == "__main__":
    main()