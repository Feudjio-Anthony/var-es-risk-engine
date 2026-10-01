"""Download adjusted closing prices for the portfolio assets.

Run from the project root:
    python scripts/download_data.py
"""
from pathlib import Path

import yfinance as yf

# Five assets, four asset classes. All are US-listed ETFs quoted in USD,
# so they share the same currency and the same trading calendar.
TICKERS = {
    "SPY": "US equities (S&P 500)",
    "EEM": "Emerging market equities",
    "IEF": "US Treasuries 7-10 years",
    "GLD": "Gold",
    "FXE": "EUR/USD currency",
}

# FXE is the youngest ETF (listed since late 2005), so we start in 2006.
START = "2006-01-01"
END = "2026-01-01"

OUTPUT_FILE = Path("data/raw/prices.csv")


def main() -> None:
    # auto_adjust=True corrects prices for dividends and splits.
    # Without it, an ex-dividend date would look like a market loss.
    raw = yf.download(list(TICKERS), start=START, end=END, auto_adjust=True)

    # We only keep the "Close" columns (one per ticker).
    # dropna(how="any") removes any day where at least one asset has no price,
    # so every remaining row is a complete day for all 5 assets.
    prices = raw["Close"].dropna(how="any")

    # Sort the columns alphabetically so the file layout is always the same.
    prices = prices.sort_index(axis=1)

    # Create the data/raw folder if it does not exist yet, then save.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_FILE)

    print(f"{len(prices)} trading days, "
          f"from {prices.index[0].date()} to {prices.index[-1].date()}")
    print("\nFirst rows:")
    print(prices.head())
    print("\nLast rows:")
    print(prices.tail())


if __name__ == "__main__":
    main()