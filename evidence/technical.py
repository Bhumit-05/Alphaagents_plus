"""Technical tool: chart-based indicators. Pure rules on past prices, no judgement involved."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.features import _ma_ratio, _rsi
from data_platform.pit import PointInTimeData

from .record import Collector


def macd_values(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line and histogram at the latest day (None if there are too few prices)."""
    if len(close) < slow + signal:
        return None
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(line.iloc[-1] - sig.iloc[-1])


def breakout_20d(close: pd.Series) -> float:
    """+1 if the latest close is above the highest close of the previous 20 days, -1 if below the lowest, else 0."""
    if len(close) < 21:
        return np.nan
    prior, last = close.iloc[-21:-1], close.iloc[-1]
    return 1.0 if last > prior.max() else (-1.0 if last < prior.min() else 0.0)


def compute_technical(close: pd.Series) -> dict:
    """All technical numbers from a series of past closes (oldest first). NaN where history is too short."""
    out = {k: np.nan for k in ["rsi_14", "ma50_ratio", "ma200_ratio", "ma50_above_ma200", "macd_line",
                               "macd_signal_line", "macd_hist", "macd_hist_pct", "breakout_20d", "pct_from_52w_high"]}
    out["rsi_14"] = _rsi(close, 14)
    out["ma50_ratio"] = _ma_ratio(close, 50)
    out["ma200_ratio"] = _ma_ratio(close, 200)
    if len(close) >= 200:
        s50, s200 = close.tail(50).mean(), close.tail(200).mean()
        out["ma50_above_ma200"] = 1.0 if s50 > s200 else (-1.0 if s50 < s200 else 0.0)
    m = macd_values(close)
    if m is not None:
        out["macd_line"], out["macd_signal_line"], out["macd_hist"] = m
        out["macd_hist_pct"] = m[2] / float(close.iloc[-1])
    out["breakout_20d"] = breakout_20d(close)
    if len(close) >= 252:
        out["pct_from_52w_high"] = float(close.iloc[-1] / close.tail(252).max() - 1.0)
    return out


_SPEC = [  # name, unit, window, meaning
    ("rsi_14", "rsi_points", 14, "Relative Strength Index over 14 days, from 0 to 100. Above 70 is often read as overbought, below 30 as oversold."),
    ("ma50_ratio", "fraction", 50, "Price compared with its 50-day average. 0.05 means 5% above the average."),
    ("ma200_ratio", "fraction", 200, "Price compared with its 200-day average. 0.05 means 5% above the average."),
    ("ma50_above_ma200", "flag", 200, "+1 if the 50-day average is above the 200-day average (uptrend), -1 if below (downtrend)."),
    ("macd_line", "price", 26, "MACD line: 12-day exponential average of the price minus the 26-day one, in price units."),
    ("macd_signal_line", "price", 9, "9-day exponential average of the MACD line."),
    ("macd_hist", "price", 9, "MACD line minus its signal line. Positive suggests strengthening momentum, negative weakening."),
    ("macd_hist_pct", "fraction", 9, "MACD histogram as a fraction of the price, so it can be compared across stocks."),
    ("breakout_20d", "flag", 20, "+1 if today's close is above the highest close of the previous 20 days, -1 if below the lowest, 0 otherwise."),
    ("pct_from_52w_high", "fraction", 252, "Distance from the 52-week high. 0 means at the high, -0.10 means 10% below it."),
]


def technical_evidence(pit: PointInTimeData, ticker: str, as_of):
    as_of = pd.Timestamp(as_of).normalize()
    c = Collector("TEC", "technical", ticker, as_of)
    close = pit.prices(ticker, as_of, lookback_days=300)["adj_close"].astype(float)
    values = compute_technical(close)
    for name, unit, window, what in _SPEC:
        c.put(name, values[name], unit, window, what)
    return c.items, c.missing
