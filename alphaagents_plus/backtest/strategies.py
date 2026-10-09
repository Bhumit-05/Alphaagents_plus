"""Baseline strategies (no AI). Each one is handed a VIEW of the data that physically stops at the decision date.

A strategy returns target weights {ticker: weight}: long-only, adding up to at most 1 (the rest is cash).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.pit import PointInTimeData


def eligible_tickers(view: PointInTimeData, as_of, tickers: list, min_history: int = 1) -> list:
    """Stocks that traded on `as_of` and have at least `min_history` days of prices."""
    as_of = pd.Timestamp(as_of)
    out = []
    for t in tickers:
        if t not in view.tickers:
            continue
        df = view.prices(t, as_of)
        if len(df) >= min_history and df.index[-1] == as_of:
            out.append(t)
    return out


class Strategy:
    name = "strategy"
    min_history = 1

    def pick(self, view: PointInTimeData, as_of, eligible: list) -> dict:
        raise NotImplementedError

    def weights(self, view: PointInTimeData, as_of, tickers: list) -> dict:
        eligible = eligible_tickers(view, as_of, tickers, self.min_history)
        return self.pick(view, pd.Timestamp(as_of), eligible) if eligible else {}


class EqualWeight(Strategy):
    """Hold every stock in the universe in equal amounts."""
    name = "Equal-weight universe"

    def pick(self, view, as_of, eligible):
        return {t: 1.0 / len(eligible) for t in eligible}


class SingleAsset(Strategy):
    """Put everything in one asset (used for the SPY benchmark)."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.name = f"{ticker} buy-and-hold"

    def weights(self, view, as_of, tickers):
        return {self.ticker: 1.0} if eligible_tickers(view, as_of, [self.ticker]) else {}


class Momentum(Strategy):
    """12-1 momentum: hold the top k stocks by return over the past year, skipping the most recent month."""
    min_history = 253

    def __init__(self, k: int = 4):
        self.k = k
        self.name = f"Momentum top-{k}"

    def pick(self, view, as_of, eligible):
        score = {}
        for t in eligible:
            c = view.prices(t, as_of, lookback_days=253)["adj_close"]
            score[t] = float(c.iloc[-22] / c.iloc[0] - 1.0)
        top = sorted(score, key=score.get, reverse=True)[: self.k]
        return {t: 1.0 / len(top) for t in top}


class LowVolatility(Strategy):
    """Hold the k calmest stocks (lowest 63-day volatility): a defensive baseline to compare with risk-averse agents."""
    min_history = 64

    def __init__(self, k: int = 4):
        self.k = k
        self.name = f"Low-volatility top-{k}"

    def pick(self, view, as_of, eligible):
        vol = {t: float(view.prices(t, as_of, lookback_days=64)["adj_close"].pct_change().dropna().std())
               for t in eligible}
        calm = sorted(vol, key=vol.get)[: self.k]
        return {t: 1.0 / len(calm) for t in calm}


class RandomPicks(Strategy):
    """k random stocks each month (repeatable: the same seed always gives the same picks)."""

    def __init__(self, k: int = 4, seed: int = 0):
        self.k, self.seed = k, seed
        self.name = f"Random {k} picks (seed {seed})"

    def pick(self, view, as_of, eligible):
        rng = np.random.default_rng(self.seed + int(as_of.strftime("%Y%m%d")))
        chosen = list(rng.choice(sorted(eligible), size=min(self.k, len(eligible)), replace=False))
        return {t: 1.0 / len(chosen) for t in chosen}


def build_schedules(pit: PointInTimeData, strategies: list, tickers: list, decision_dates: list) -> dict:
    """{strategy name: {decision date: weights}}.

    For each date, every strategy is given pit.with_cutoff(date): a copy of the data that has the future
    PHYSICALLY removed. A strategy therefore cannot see later prices even if it tried.
    """
    schedules = {s.name: {} for s in strategies}
    for d in decision_dates:
        d = pd.Timestamp(d)
        view = pit.with_cutoff(d)
        for s in strategies:
            schedules[s.name][d] = s.weights(view, d, tickers)
        del view
    return schedules
