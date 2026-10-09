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


## Session 9 - C++ Monte Carlo engine (pybind11)
- Built `cpp/mc_engine.cpp`: Cholesky + one-pass P&L (weights folded into the Cholesky factor), ziggurat normal sampler on xoshiro256++. Tests (KS, tails, correlation, reproducibility, error handling): 7 passed.
- Profiling: `std::normal_distribution` + mt19937_64 was slower than NumPy's sampler, so I replaced it. Measured on MSVC: ~2.65x faster than NumPy for >= 100k scenarios (1.4x at 10k).
- NumPy and C++ use different generators, so validation has to be statistical, not bit-for-bit.


## Session 10 - C++ backend integration and validation
- Added a switchable `backend="python"|"cpp"` parameter to `monte_carlo_var_es` and `rolling_backtest`; both backends use the same regularised covariance matrix.
- 20-seed validation (VaR and ES, 95% and 99%): backends agree within 1%, and the C++ mean converges to the analytical Gaussian value. Exact equality is impossible (different RNGs).
- Full backtest with C++: hist/param columns identical; Monte Carlo violations 217/137/87 (C++) vs 217/137/86 (Python) at 95/97.5/99%. The day-by-day gap (0.7-0.9% on average) matches the expected Monte Carlo noise of two independent runs.
- Timings were measured under different machine loads, so no speed claim yet; proper benchmark in session 11.