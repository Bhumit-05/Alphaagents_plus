"""Data quality report: find gaps and oddities BEFORE the agents run, so they are handled on purpose."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def price_quality(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    calendar = pd.DatetimeIndex(prices.loc[prices["ticker"] == benchmark, "date"].sort_values().unique())
    rows = []
    for t, g in prices.groupby("ticker"):
        g = g.sort_values("date")
        have = pd.DatetimeIndex(g["date"])
        expected = calendar[(calendar >= have.min()) & (calendar <= have.max())] if len(calendar) else have
        ret = g["adj_close"].astype(float).pct_change().abs()
        rows.append({
            "ticker": t, "first_date": have.min().date(), "last_date": have.max().date(), "n_days": len(g),
            "missing_vs_benchmark": int(len(expected.difference(have))) if len(calendar) else np.nan,
            "zero_volume_days": int((g["volume"] == 0).sum()),
            "unchanged_close_days": int((g["close"].diff() == 0).sum()),
            "max_abs_daily_move": float(ret.max()) if len(ret) > 1 else np.nan,
            "moves_over_25pct": int((ret > 0.25).sum()),
        })
    return pd.DataFrame(rows)


def filing_quality(filings: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame(columns=["ticker", "n_10K", "n_10Q", "latest_filing", "missing_files", "flag"])
    filings = filings.copy()
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    rows = []
    for t, g in filings.groupby("ticker"):
        missing = 0
        for rel in g["local_path"]:
            if rel is None or (isinstance(rel, float) and pd.isna(rel)):
                missing += 1
            elif data_dir is not None and not (Path(data_dir) / rel).exists():
                missing += 1
        n_k, n_q = int((g["form"] == "10-K").sum()), int((g["form"] == "10-Q").sum())
        rows.append({"ticker": t, "n_10K": n_k, "n_10Q": n_q, "latest_filing": g["filing_date"].max().date(),
                     "missing_files": missing, "flag": "CHECK" if (n_k < 3 or n_q < 9 or missing) else ""})
    return pd.DataFrame(rows)


def news_quality(news: pd.DataFrame, tickers: list) -> pd.DataFrame:
    news = news.copy()
    if not news.empty:
        news["published_at"] = pd.to_datetime(news["published_at"])
    rows = []
    for t in tickers:
        g = news[news["ticker"] == t] if not news.empty else news
        if g.empty:
            rows.append({"ticker": t, "n_articles": 0, "first": None, "last": None,
                         "share_of_days_with_news": 0.0, "low_coverage": True})
            continue
        days = g["published_at"].dt.normalize()
        span = max((days.max() - days.min()).days + 1, 1)
        share = days.nunique() / span
        rows.append({"ticker": t, "n_articles": len(g), "first": days.min().date(), "last": days.max().date(),
                     "share_of_days_with_news": round(float(share), 3), "low_coverage": bool(share < 0.5)})
    return pd.DataFrame(rows)


def macro_quality(macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return pd.DataFrame(columns=["series", "n_obs", "first_obs", "last_obs", "last_value"])
    macro = macro.copy()
    macro["obs_date"] = pd.to_datetime(macro["obs_date"])
    rows = []
    for s, g in macro.groupby("series"):
        g = g.sort_values("obs_date")
        rows.append({"series": s, "n_obs": len(g), "first_obs": g["obs_date"].min().date(),
                     "last_obs": g["obs_date"].max().date(), "last_value": float(g["value"].iloc[-1])})
    return pd.DataFrame(rows)


def build_report(prices, filings, news, macro, benchmark: str, tickers: list, data_dir: Path | None = None) -> dict:
    return {
        "prices": price_quality(prices, benchmark),
        "filings": filing_quality(filings, data_dir),
        "news": news_quality(news, tickers),
        "macro": macro_quality(macro),
    }


def save_report(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in report.items():
        df.to_csv(out_dir / f"{name}_quality.csv", index=False)
