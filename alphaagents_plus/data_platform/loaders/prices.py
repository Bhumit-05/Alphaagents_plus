"""Daily price loader (Yahoo Finance via yfinance).

Output schema (long format, one row per ticker per trading day):
    ticker, date, open, high, low, close, adj_close, volume, dividends, splits

`adj_close` is split- and dividend-adjusted and is what we compute returns from.
`close` is the split-adjusted price without dividend adjustment.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger(__name__)

_RENAME = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "splits",
}
COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]


def download_prices(tickers: list, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf  # imported here so tests do not need it

    frames = []
    for ticker in tickers:
        df = None
        for attempt in range(1, retries + 1):
            try:
                df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
                if df is not None and len(df):
                    break
            except Exception as err:  # network / parsing problems
                log.warning("%s: price download failed (%s), attempt %d/%d", ticker, err, attempt, retries)
            time.sleep(2 * attempt)
        if df is None or df.empty:
            log.error("%s: no price data returned", ticker)
            continue

        df = df.rename(columns=_RENAME)
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
        df.index.name = "date"
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        for col in ("dividends", "splits"):
            if col not in df.columns:
                df[col] = 0.0
        df = df.reset_index()
        df["ticker"] = ticker
        frames.append(df[COLUMNS])
        log.info("%s: %d price rows (%s to %s)", ticker, len(df), df["date"].min().date(), df["date"].max().date())

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
