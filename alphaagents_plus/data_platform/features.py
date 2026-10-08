"""Features computed ONLY from point-in-time data.

Feature names tie back to the paper's valuation agent:
    ann_ret_63d = (1 + cumulative return) ** (252 / n) - 1      with n = 63 trading days
    ann_vol_63d = daily volatility * sqrt(252)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pit import PointInTimeData

TRADING_DAYS = 252
FEATURE_COLUMNS = [
    "ticker", "date", "n_obs", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "ann_ret_63d", "ann_vol_63d", "max_dd_252d", "beta_252d",
    "avg_volume_63d", "volume_z_63d", "rsi_14", "ma50_ratio", "ma200_ratio",
]


def _window_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)


def _max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    return float((close / close.cummax() - 1.0).min())


def _rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff().dropna().tail(n)
    if len(delta) < n:
        return np.nan
    gain = delta.clip(lower=0).mean()
    loss = (-delta.clip(upper=0)).mean()
    return 100.0 if loss == 0 else float(100.0 - 100.0 / (1.0 + gain / loss))


def _ma_ratio(close: pd.Series, window: int) -> float:
    if len(close) < window:
        return np.nan
    return float(close.iloc[-1] / close.tail(window).mean() - 1.0)


def compute_features(pit: PointInTimeData, ticker: str, as_of, lookback_days: int = 300) -> dict:
    """All features for one ticker at one decision date. Uses data up to `as_of` only."""
    df = pit.prices(ticker, as_of, lookback_days=lookback_days)
    close = df["adj_close"].astype(float)
    daily = close.pct_change().dropna()
    out = {c: np.nan for c in FEATURE_COLUMNS}
    out.update({"ticker": ticker, "date": pd.Timestamp(as_of).normalize(), "n_obs": len(df)})

    for w in (21, 63, 126, 252):
        out[f"ret_{w}d"] = _window_return(close, w)

    if not np.isnan(out["ret_63d"]):
        out["ann_ret_63d"] = float((1.0 + out["ret_63d"]) ** (TRADING_DAYS / 63) - 1.0)
    if len(daily) >= 63:
        out["ann_vol_63d"] = float(daily.tail(63).std() * np.sqrt(TRADING_DAYS))

    out["max_dd_252d"] = _max_drawdown(close.tail(TRADING_DAYS)) if len(close) >= 2 else np.nan

    if ticker != pit.benchmark:
        bench = pit.prices(pit.benchmark, as_of, lookback_days=lookback_days)["adj_close"].astype(float)
        both = pd.concat([daily, bench.pct_change().dropna()], axis=1, join="inner").tail(TRADING_DAYS)
        if len(both) >= 60 and both.iloc[:, 1].var() > 0:
            out["beta_252d"] = float(both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var())

    vol = df["volume"].astype(float).tail(63)
    if len(vol) == 63:
        out["avg_volume_63d"] = float(vol.mean())
        if vol.std() > 0:
            out["volume_z_63d"] = float((vol.iloc[-1] - vol.mean()) / vol.std())

    out["rsi_14"] = _rsi(close)
    out["ma50_ratio"] = _ma_ratio(close, 50)
    out["ma200_ratio"] = _ma_ratio(close, 200)
    return out


def build_feature_table(pit: PointInTimeData, tickers: list, dates: list) -> pd.DataFrame:
    """One row per (ticker, decision date)."""
    rows = []
    for d in dates:
        for t in tickers:
            if t in pit.tickers:
                rows.append(compute_features(pit, t, d))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
