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