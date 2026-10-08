"""Company news loader (Finnhub).

Output schema:
    ticker, published_at (UTC, naive), headline, summary, source, url, article_id

Two safeguards against silently losing articles:
  * A request that returns `split_at` or more articles is probably truncated by the API, so its date
    range is split in half and fetched again, down to single days.
  * Failed requests are counted and reported instead of being treated as "no news".

Check Finnhub's current free-tier limits (history depth, calls per minute) before long downloads.
To use another provider, write a function that returns the same columns.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from ._http import get_with_retry

log = logging.getLogger(__name__)

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
COLUMNS = ["ticker", "published_at", "headline", "summary", "source", "url", "article_id"]
DAY = pd.Timedelta(days=1)


def _fetch_window(session, ticker: str, d0: pd.Timestamp, d1: pd.Timestamp, api_key: str):
    """One API call for the dates d0..d1 (inclusive). Returns a list, or None if the call failed."""
    params = {"symbol": ticker, "from": d0.strftime("%Y-%m-%d"),
              "to": d1.strftime("%Y-%m-%d"), "token": api_key}
    try:
        items = get_with_retry(session, FINNHUB_URL, params=params).json()
    except Exception as err:
        log.error("%s %s..%s failed: %s", ticker, params["from"], params["to"], err)
        return None
    return items if isinstance(items, list) else []


def _fetch_adaptive(session, ticker, d0, d1, api_key, pause, split_at, stats) -> list:
    items = _fetch_window(session, ticker, d0, d1, api_key)
    stats["calls"] += 1
    time.sleep(pause)
    if items is None:
        stats["failed_windows"] += 1
        return []
    if len(items) >= split_at:
        span = (d1 - d0).days
        if span >= 1:
            mid = d0 + pd.Timedelta(days=span // 2)
            stats["splits"] += 1
            return (_fetch_adaptive(session, ticker, d0, mid, api_key, pause, split_at, stats)
                    + _fetch_adaptive(session, ticker, mid + DAY, d1, api_key, pause, split_at, stats))
        stats["saturated_days"] += 1
        log.warning("%s %s: %d articles in a single day, may be truncated", ticker, d0.date(), len(items))
    return items


def _rows(ticker: str, items: list) -> list:
    rows = []
    for it in items:
        ts = pd.to_datetime(it.get("datetime"), unit="s", utc=True)
        if pd.isna(ts):
            continue
        rows.append({
            "ticker": ticker,
            "published_at": ts.tz_localize(None),
            "headline": it.get("headline", ""),
            "summary": it.get("summary", ""),
            "source": it.get("source", ""),
            "url": it.get("url", ""),
            "article_id": str(it.get("id", "")),
        })
    return rows


def download_news_ticker(ticker: str, start: str, end: str | None, api_key: str, chunk_days: int = 7,
                         pause: float = 1.1, split_at: int = 200, session=None):
    """Download one ticker's news. Returns (DataFrame, stats dict)."""
    if not api_key:
        raise ValueError("FINNHUB_API_KEY is missing. Put it in your .env file.")
    session = session or requests.Session()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    stats = {"calls": 0, "splits": 0, "failed_windows": 0, "saturated_days": 0}

    rows, cursor = [], pd.Timestamp(start)
    while cursor <= end_ts:
        window_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end_ts)
        rows += _rows(ticker, _fetch_adaptive(session, ticker, cursor, window_end, api_key, pause, split_at, stats))
        cursor = window_end + DAY

    if not rows:
        return pd.DataFrame(columns=COLUMNS), stats
    df = pd.DataFrame(rows).drop_duplicates(["ticker", "url"]).drop_duplicates(["ticker", "article_id", "published_at"])
    return df.sort_values(["ticker", "published_at"]).reset_index(drop=True), stats


def download_news(tickers: list, start: str, end: str | None, api_key: str,
                  chunk_days: int = 7, pause: float = 1.1, split_at: int = 200) -> pd.DataFrame:
    """Convenience wrapper: all tickers in one DataFrame."""
    frames = []
    for t in tickers:
        df, stats = download_news_ticker(t, start, end, api_key, chunk_days, pause, split_at)
        log.info("%s: %d articles, %s", t, len(df), stats)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
