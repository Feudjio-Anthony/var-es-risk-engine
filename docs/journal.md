# Project journal

## Session 1
- Done: repo structure, virtual environment, price download (5 ETFs, 2006-2025).
- Blocking: nothing.
- Next: session 2, compute log-returns and the portfolio P&L series.

## Session 2
- Done: log-returns, equal-weight portfolio P&L, sanity checks, two figures.
- Blocking: nothing.
- Next: session 3, historical-simulation VaR and ES.

## Session 3
- Done: historical VaR/ES in measures.py (sign convention documented), summary table, 3 unit tests.
- Blocking: nothing.
- Next: session 4, parametric (Gaussian) VaR and ES.

## Session 4
- Done: parametric Gaussian VaR/ES (variance-covariance), comparison with historical, 2 new tests (agreement on simulated Gaussian data).
- Blocking: nothing.
- Next: session 5, Monte Carlo VaR/ES.

## Session 5
- Done: Monte Carlo VaR/ES with Cholesky, cross-validation vs parametric, convergence study (figure), 2 new tests.
- Blocking: nothing.
- Next: session 6, rolling-window backtesting engine.

## Session 6
- Done: rolling-window backtest engine (window 500, three methods, three levels), violation counting, no-look-ahead test, results/backtest.csv.
- Blocking: nothing.
- Next: session 7, statistical validation tests (Kupiec, Christoffersen).

## Session 7
- Done: Kupiec, Christoffersen independence, conditional coverage, Basel traffic light (last window + rolling shares), results/stat_tests.csv, 8 new tests.
- Blocking: nothing.
- Next: session 8, visualisation and reading of the results.

## Session 8
- Done: four figures (P&L vs VaR, ES/VaR ratio, March 2020 zoom, violation rates), smoke tests, written commentary in docs/methodology.md.
- Blocking: nothing.
- Next: session 9, C++ Monte Carlo engine.


## Session 9
- Done: C++ Monte Carlo engine with pybind11 (fused Cholesky + sampling + portfolio projection), ziggurat/xoshiro sampler after profiling showed std::normal_distribution slower than NumPy, build script, 7 tests.
- Blocking: nothing (if the compiler is missing: skip C++, see guide).
- Next: session 10, plug the C++ backend into measures.py and validate it statistically over 20 seeds.

## Session 9 - C++ Monte Carlo engine (pybind11)
- Built `cpp/mc_engine.cpp`: Cholesky + one-pass P&L (weights folded into the Cholesky factor), ziggurat normal sampler on xoshiro256++. Tests (KS, tails, correlation, reproducibility, error handling): 7 passed.
- Profiling: `std::normal_distribution` + mt19937_64 was slower than NumPy's sampler, so I replaced it. Measured on MSVC: ~2.65x faster than NumPy for >= 100k scenarios (1.4x at 10k).
- NumPy and C++ use different generators, so validation has to be statistical, not bit-for-bit.

