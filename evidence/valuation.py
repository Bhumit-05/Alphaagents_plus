"""Valuation tool: how the price and trading volume have behaved (the paper's Valuation Agent inputs).

Formulas from the paper:  annualized return = (1 + R) ** (252 / n) - 1,   annualized volatility = daily std * sqrt(252).
All numbers come from data_platform.features, so the whole project computes them one way.
"""
from __future__ import annotations

import pandas as pd

from data_platform.features import compute_features
from data_platform.pit import PointInTimeData

from .record import Collector

_SPEC = [  # feature key, evidence name, unit, window, meaning
    ("ret_21d", "ret_1m", "fraction", 21, "Price return over the last 21 trading days (about 1 month). 0.05 means +5%."),
    ("ret_63d", "ret_3m", "fraction", 63, "Price return over the last 63 trading days (about 3 months). 0.05 means +5%."),
    ("ret_126d", "ret_6m", "fraction", 126, "Price return over the last 126 trading days (about 6 months)."),
    ("ret_252d", "ret_12m", "fraction", 252, "Price return over the last 252 trading days (about 1 year)."),
    ("ann_ret_63d", "ann_ret_63d", "fraction", 63, "The last 3 months' return scaled to a full year: (1 + R) ** (252 / 63) - 1."),
    ("ann_vol_63d", "ann_vol_63d", "fraction", 63, "Volatility: standard deviation of daily returns over 63 days times sqrt(252). 0.27 means 27% a year. Higher means a bumpier ride."),
    ("max_dd_252d", "max_drawdown_252d", "fraction", 252, "Largest fall from a peak to a later low over the last year. A negative number: -0.15 means a 15% fall."),
    ("beta_252d", "beta_252d", "ratio", 252, "How strongly the stock moves with the benchmark (SPY) over the last year. 1.0 moves with the market, above 1 more, below 1 less."),
    ("avg_volume_63d", "avg_volume_63d", "shares", 63, "Average number of shares traded per day over the last 63 days."),
    ("volume_z_63d", "volume_z_63d", "z_score", 63, "Latest day's volume compared with the 63-day average, in standard deviations. Above 2 is unusually busy."),
]


def valuation_evidence(pit: PointInTimeData, ticker: str, as_of):
    """Returns (list of Evidence, names that could not be computed, date of the latest price used)."""
    as_of = pd.Timestamp(as_of).normalize()
    c = Collector("VAL", "valuation", ticker, as_of)
    f = compute_features(pit, ticker, as_of)
    for key, name, unit, window, what in _SPEC:
        c.put(name, f[key], unit, window, what)

    last = pit.prices(ticker, as_of, lookback_days=1)
    price_date = None if last.empty else f"{last.index[-1]:%Y-%m-%d}"
    c.put("last_close", float(last["close"].iloc[-1]) if len(last) else None, "price", None,
          "Latest closing price (adjusted for splits, not for dividends), in the stock's trading currency.")

    bench = compute_features(pit, pit.benchmark, as_of)
    c.put("rel_strength_3m", f["ret_63d"] - bench["ret_63d"], "fraction", 63,
          "The stock's 3-month return minus the benchmark's. Positive means it beat the market over 3 months.")
    return c.items, c.missing, price_date
