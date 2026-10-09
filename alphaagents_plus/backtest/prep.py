"""Turn a PointInTimeData object into the inputs the engine needs (realized prices and risk-free rates).

The engine is the EVALUATOR, so it is allowed to look at what happened after each decision.
Strategies are never given this data; they only get truncated views (see strategies.build_schedules).
"""
from __future__ import annotations

import pandas as pd

from data_platform.pit import PointInTimeData


def price_matrix(pit: PointInTimeData, tickers: list, end) -> pd.DataFrame:
    """Adjusted closes, one column per ticker, up to and including `end`."""
    return pd.DataFrame({t: pit.prices(t, end)["adj_close"] for t in tickers}).sort_index()


def risk_free_daily(pit: PointInTimeData, days: pd.DatetimeIndex, end, series: str = "DGS1MO") -> pd.Series:
    """Daily risk-free return for each trading day. Day d uses the rate seen on the previous trading day."""
    s = pit.macro(series, end) / 100.0 / 252.0
    daily = s.reindex(days.union(s.index)).ffill().reindex(days)
    return daily.shift(1).fillna(0.0)
