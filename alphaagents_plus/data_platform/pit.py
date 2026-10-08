"""Point-in-time data access.

RULE: every read takes an `as_of` date and returns only what was knowable at the CLOSE of that day.
No method in this class returns unfiltered data, and nothing else in the project reads the raw files.

What "knowable at the close of day T" means for each dataset:
  prices   rows with date <= T (the close of T is known at the close of T)
  filings  rows with filing_date + filing_lag_days <= T (conservative: a filing is usable the day after)
  news     articles published at or before 4pm New York time on day T
  macro    observations with available_date <= T (observation date + a publication lag)

A `cutoff` makes the object PHYSICALLY drop everything later than the cutoff, and any request beyond the
cutoff raises FutureAccessError. The tests use this to prove that nothing leaks: results computed with
all the data must be identical to results computed with the future deleted.
"""
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class FutureAccessError(ValueError):
    """Raised when a request asks for a date later than the object's cutoff."""


def _day(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def market_close_utc(day) -> pd.Timestamp:
    """4pm New York time on `day`, as a naive UTC timestamp (handles daylight saving time)."""
    d = _day(day)
    local = datetime.combine(d.date(), time(16, 0), tzinfo=NY)
    return pd.Timestamp(local.astimezone(UTC).replace(tzinfo=None))


def monthly_decision_dates(trading_days: pd.DatetimeIndex, start, end) -> list:
    """First trading day of each calendar month between start and end (calendar helper, no market data)."""
    idx = pd.DatetimeIndex(trading_days)
    idx = idx[(idx >= _day(start)) & (idx <= _day(end))]
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).first().values)


class PointInTimeData:
    def __init__(self, prices: pd.DataFrame, filings: pd.DataFrame, news: pd.DataFrame,
                 macro: pd.DataFrame, benchmark: str = "SPY", filing_lag_days: int = 1,
                 cutoff=None, data_dir=None):
        self.benchmark = benchmark
        self.filing_lag_days = int(filing_lag_days)
        self.cutoff = None if cutoff is None else _day(cutoff)
        self.data_dir = Path(data_dir) if data_dir else None
        self._raw = (prices, filings, news, macro)  # kept so with_cutoff() can rebuild

        p = prices.copy()
        p["date"] = pd.to_datetime(p["date"])
        f = filings.copy()
        f["filing_date"] = pd.to_datetime(f["filing_date"])
        f["available_date"] = f["filing_date"] + pd.Timedelta(days=self.filing_lag_days)
        n = news.copy()
        n["published_at"] = pd.to_datetime(n["published_at"])
        m = macro.copy()
        m["obs_date"] = pd.to_datetime(m["obs_date"])
        m["available_date"] = pd.to_datetime(m["available_date"])

        if self.cutoff is not None:  # physically remove the future
            p = p[p["date"] <= self.cutoff]
            f = f[f["available_date"] <= self.cutoff]
            n = n[n["published_at"] <= market_close_utc(self.cutoff)]
            m = m[m["available_date"] <= self.cutoff]

        self._prices = {t: g.sort_values("date").set_index("date") for t, g in p.groupby("ticker")}
        self._filings = f.sort_values("filing_date").reset_index(drop=True)
        self._news = n.sort_values("published_at").reset_index(drop=True)
        self._macro = m.sort_values(["series", "obs_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg, cutoff=None) -> "PointInTimeData":
        for path in (cfg.prices_path, cfg.filings_path, cfg.news_path, cfg.macro_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run: python -m scripts.download_data --all")
        return cls(
            prices=pd.read_parquet(cfg.prices_path),
            filings=pd.read_parquet(cfg.filings_path),
            news=pd.read_parquet(cfg.news_path),
            macro=pd.read_parquet(cfg.macro_path),
            benchmark=cfg.benchmark,
            filing_lag_days=cfg.filing_lag_days,
            cutoff=cutoff,
            data_dir=cfg.data_dir,
        )

    def with_cutoff(self, cutoff) -> "PointInTimeData":
        """A copy that physically contains no data later than `cutoff`."""
        new = _day(cutoff)
        if self.cutoff is not None and new > self.cutoff:
            raise FutureAccessError(f"cannot extend cutoff {self.cutoff.date()} to {new.date()}")
        prices, filings, news, macro = self._raw
        return PointInTimeData(prices, filings, news, macro, self.benchmark,
                               self.filing_lag_days, cutoff=new, data_dir=self.data_dir)

    # ------------------------------------------------------------------ guard
    def _check(self, as_of) -> pd.Timestamp:
        ts = _day(as_of)
        if self.cutoff is not None and ts > self.cutoff:
            raise FutureAccessError(f"as_of {ts.date()} is after the cutoff {self.cutoff.date()}")
        return ts

    # ------------------------------------------------------------------ prices
    @property
    def tickers(self) -> list:
        return sorted(self._prices)

    def prices(self, ticker: str, as_of, lookback_days: int | None = None) -> pd.DataFrame:
        """Daily bars up to and including `as_of`, oldest first, optionally only the last N rows."""
        ts = self._check(as_of)
        if ticker not in self._prices:
            raise KeyError(f"no price data for {ticker}")
        df = self._prices[ticker].loc[:ts]
        return (df.tail(lookback_days) if lookback_days else df).copy()

    def trading_days(self, as_of, lookback_days: int | None = None) -> pd.DatetimeIndex:
        """Trading calendar up to `as_of`, taken from the benchmark's price history."""
        idx = self.prices(self.benchmark, as_of).index
        return idx[-lookback_days:] if lookback_days else idx

    # ------------------------------------------------------------------ filings
    def filings(self, ticker: str, as_of, forms: list | None = None, n: int | None = None) -> pd.DataFrame:
        """Filings usable at `as_of`, newest first."""
        ts = self._check(as_of)
        df = self._filings
        df = df[(df["ticker"] == ticker) & (df["available_date"] <= ts)]
        if forms:
            df = df[df["form"].isin(forms)]
        df = df.sort_values("filing_date", ascending=False)
        return (df.head(n) if n else df).reset_index(drop=True)

    def latest_filing(self, ticker: str, as_of, form: str = "10-K"):
        """The most recent usable filing of a given form, or None."""
        df = self.filings(ticker, as_of, forms=[form], n=1)
        return None if df.empty else df.iloc[0]

    def filing_path(self, filing) -> Path | None:
        """Where the filing's HTML sits on disk (None if it was not downloaded)."""
        if filing is None or self.data_dir is None:
            return None
        rel = filing.get("local_path") if hasattr(filing, "get") else None
        if rel is None or (isinstance(rel, float) and pd.isna(rel)):
            return None
        return self.data_dir / rel

    # ------------------------------------------------------------------ news
    def news(self, ticker: str, as_of, days: int = 30) -> pd.DataFrame:
        """Articles published in the `days` before the close of `as_of`, newest first."""
        ts = self._check(as_of)
        end = market_close_utc(ts)
        start = end - pd.Timedelta(days=days)
        df = self._news
        df = df[(df["ticker"] == ticker) & (df["published_at"] <= end) & (df["published_at"] > start)]
        return df.sort_values("published_at", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ macro
    def macro(self, series: str, as_of, lookback_days: int | None = None) -> pd.Series:
        """Observations of a macro series that were public at `as_of`, indexed by observation date."""
        ts = self._check(as_of)
        df = self._macro
        df = df[(df["series"] == series) & (df["available_date"] <= ts)]
        if lookback_days:
            df = df[df["obs_date"] > ts - pd.Timedelta(days=lookback_days)]
        return pd.Series(df["value"].values, index=pd.DatetimeIndex(df["obs_date"]), name=series)

    def macro_latest(self, series: str, as_of):
        s = self.macro(series, as_of)
        return None if s.empty else float(s.iloc[-1])

    def risk_free_rate(self, as_of, series: str = "DGS1MO"):
        """Latest known 1-month Treasury yield as an annual decimal (4.5% -> 0.045), or None."""
        v = self.macro_latest(series, as_of)
        return None if v is None else v / 100.0
