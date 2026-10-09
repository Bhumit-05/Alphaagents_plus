"""Backtest engine: replays a schedule of target weights through historical prices.

TIMING (the part that prevents cheating)
  * A decision is made at the CLOSE of day T, using only data up to T (strategies get a view of the data that
    physically contains nothing later).
  * The trade happens at the close of day T + execution_lag trading days (default 1: the next day's close).
    The new position earns returns starting the day AFTER the trade.
  * Between rebalances nothing is traded: each position simply grows or shrinks with its price (weights drift).

COSTS
  turnover = sum of absolute changes in weight at a rebalance (buying and selling both count).
  cost     = turnover * cost_bps / 10,000, taken out of the portfolio value on the trade day.

CASH
  Weights may add up to less than 1. The rest is cash and earns the risk-free rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name: str
    nav: pd.Series          # portfolio value at each close (1.0 = starting money)
    returns: pd.Series      # daily returns after costs
    turnover: pd.Series     # one value per rebalance
    costs: pd.Series        # cost per rebalance, as a fraction of portfolio value
    weights: pd.DataFrame   # target weights chosen at each rebalance (rows = trade dates)
    n_missing_returns: int = 0


def _validate(weights: dict, tickers: list, when) -> None:
    for t, w in weights.items():
        if t not in tickers:
            raise ValueError(f"{when.date()}: weight given for {t}, which has no price data")
        if w < -1e-12:
            raise ValueError(f"{when.date()}: negative weight for {t} (this engine is long-only)")
    if sum(weights.values()) > 1 + 1e-9:
        raise ValueError(f"{when.date()}: weights add up to {sum(weights.values()):.4f}, more than 1")


def run_backtest(name: str, schedule: dict, prices: pd.DataFrame, rf_daily: pd.Series | None = None,
                 cost_bps: float = 10.0, execution_lag: int = 1) -> BacktestResult:
    """schedule: {decision_date: {ticker: target_weight}}.  prices: dates x tickers (adjusted close)."""
    if execution_lag < 0:
        raise ValueError("execution_lag must be 0 or more")
    prices = prices.sort_index()
    days = prices.index
    tickers = list(prices.columns)
    rets = prices.pct_change()
    rf = rf_daily.reindex(days).ffill().fillna(0.0) if rf_daily is not None else pd.Series(0.0, index=days)

    exec_targets = {}
    for d in sorted(schedule):
        d = pd.Timestamp(d)
        if d not in days:
            raise ValueError(f"decision date {d.date()} is not a trading day in the price data")
        pos = days.get_loc(d) + execution_lag
        if pos >= len(days):
            log.warning("%s: decision on %s cannot be executed (no later price data), skipped", name, d.date())
            continue
        _validate(schedule[d], tickers, d)
        exec_targets[days[pos]] = schedule[d]
    if not exec_targets:
        raise ValueError(f"{name}: no executable decision dates")

    first = min(exec_targets)
    start_pos = days.get_loc(first)
    hold = pd.Series(0.0, index=tickers)   # money in each stock
    cash = 1.0
    nav_vals, nav_idx = [], []
    turnovers, costs, w_rows, w_idx = [], [], [], []
    n_missing = 0

    for pos in range(start_pos, len(days)):
        day = days[pos]
        if pos > start_pos:
            r = rets.iloc[pos]
            n_missing += int((r.isna() & (hold > 0)).sum())
            hold = hold * (1.0 + r.fillna(0.0))
            cash = cash * (1.0 + rf.iloc[pos])
        nav = hold.sum() + cash

        if day in exec_targets:
            tw = pd.Series(0.0, index=tickers)
            for t, w in exec_targets[day].items():
                tw[t] = w
            current_w = hold / nav
            turnover = float((tw - current_w).abs().sum())
            cost = turnover * cost_bps / 1e4
            nav = nav * (1.0 - cost)
            hold = tw * nav
            cash = nav * (1.0 - tw.sum())
            turnovers.append(turnover)
            costs.append(cost)
            w_rows.append(tw)
            w_idx.append(day)

        nav_vals.append(nav)
        nav_idx.append(day)

    nav_s = pd.Series(nav_vals, index=nav_idx, name=name)
    ret_s = nav_s / nav_s.shift(1).fillna(1.0) - 1.0          # the first day's return is the initial trading cost
    return BacktestResult(
        name=name, nav=nav_s, returns=ret_s,
        turnover=pd.Series(turnovers, index=w_idx), costs=pd.Series(costs, index=w_idx),
        weights=pd.DataFrame(w_rows, index=w_idx), n_missing_returns=n_missing,
    )
