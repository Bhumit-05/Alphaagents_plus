"""Macro tool: market-wide conditions (volatility, interest rates, inflation, unemployment).

These readings are the same for every stock on a given date, so they carry the ticker "MKT".
Monthly series (CPI, unemployment) are only used once their publication lag has passed (see config.yaml).
"""
from __future__ import annotations

import pandas as pd

from data_platform.pit import PointInTimeData

from .record import Collector


def macro_evidence(pit: PointInTimeData, as_of):
    as_of = pd.Timestamp(as_of).normalize()
    c = Collector("MAC", "macro", "MKT", as_of)

    vix = pit.macro("VIXCLS", as_of, lookback_days=400)
    if len(vix):
        c.put("vix_level", float(vix.iloc[-1]), "index_points", None,
              "The VIX 'fear index' of expected US stock volatility. Around 15 is calm, above 30 is stressed.")
        if len(vix) >= 100:
            year = vix.tail(252)
            c.put("vix_percentile_1y", float((year <= year.iloc[-1]).mean()), "percentile", 252,
                  "Share of the last year's VIX readings at or below today's. 0.9 means volatility is higher than on 90% of days.")
        else:
            c.missing.append("vix_percentile_1y")
        if len(vix) >= 22:
            c.put("vix_change_21d", float(vix.iloc[-1] - vix.iloc[-22]), "index_points", 21,
                  "Change in the VIX over the last 21 trading days. Positive means rising fear.")
        else:
            c.missing.append("vix_change_21d")
    else:
        c.missing += ["vix_level", "vix_percentile_1y", "vix_change_21d"]

    c.put("rate_1m", pit.macro_latest("DGS1MO", as_of), "percent", None,
          "1-month US Treasury yield, in percent (4.5 means 4.5%). The risk-free rate for cash.")

    y2 = pit.macro("DGS2", as_of, lookback_days=200)
    y10 = pit.macro("DGS10", as_of, lookback_days=200)
    c.put("yield_2y", float(y2.iloc[-1]) if len(y2) else None, "percent", None, "2-year US Treasury yield, in percent.")
    c.put("yield_10y", float(y10.iloc[-1]) if len(y10) else None, "percent", None, "10-year US Treasury yield, in percent.")
    slope = (y10 - y2).dropna()
    c.put("yield_curve_10y_2y", float(slope.iloc[-1]) if len(slope) else None, "percent", None,
          "10-year yield minus 2-year yield, in percentage points. Negative (inverted) has often come before slowdowns.")
    c.put("yield_curve_change_63d", float(slope.iloc[-1] - slope.iloc[-64]) if len(slope) >= 64 else None, "percent", 63,
          "Change in that gap over the last 63 trading days, in percentage points.")

    cpi = pit.macro("CPIAUCSL", as_of)
    c.put("cpi_yoy", float(cpi.iloc[-1] / cpi.iloc[-13] - 1.0) if len(cpi) >= 13 else None, "fraction", 252,
          "Consumer price inflation over the latest 12 months, as a fraction. 0.03 means 3%.")

    un = pit.macro("UNRATE", as_of)
    c.put("unemployment_rate", float(un.iloc[-1]) if len(un) else None, "percent", None,
          "US unemployment rate, in percent (4.2 means 4.2%).")
    c.put("unemployment_change_3m", float(un.iloc[-1] - un.iloc[-4]) if len(un) >= 4 else None, "percent", 63,
          "Change in the unemployment rate over 3 months, in percentage points. Positive means rising unemployment.")
    return c.items, c.missing
