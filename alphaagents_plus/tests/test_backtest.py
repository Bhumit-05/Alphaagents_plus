"""Tests for the backtest engine, the metrics and the baseline strategies.

The engine tests use tiny price tables where the right answer can be worked out by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest import metrics as M
from backtest.engine import run_backtest
from backtest.prep import price_matrix, risk_free_daily
from backtest.strategies import (EqualWeight, LowVolatility, Momentum, RandomPicks, SingleAsset,
                                 build_schedules)
from tests.synthetic import make_pit, make_raw, poison_future


def toy(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.bdate_range("2024-01-01", periods=n))


# ----------------------------------------------------------------------------- engine
def test_buy_and_hold_follows_the_price():
    p = toy(A=[100, 110, 121, 133.1])
    res = run_backtest("x", {p.index[0]: {"A": 1.0}}, p, cost_bps=0, execution_lag=0)
    assert math.isclose(res.nav.iloc[-1], 1.331, rel_tol=1e-9)
    assert math.isclose(M.cumulative_return(res.returns), 0.331, rel_tol=1e-9)


def test_execution_lag_means_the_next_days_return_is_missed():
    p = toy(A=[100, 100, 200, 200])                  # the price doubles on day 2
    decide_day1 = {p.index[1]: {"A": 1.0}}
    same_day = run_backtest("x", decide_day1, p, cost_bps=0, execution_lag=0)    # trade at day 1 close
    next_day = run_backtest("x", decide_day1, p, cost_bps=0, execution_lag=1)    # trade at day 2 close
    assert math.isclose(M.cumulative_return(same_day.returns), 1.0)             # catches the jump
    assert math.isclose(M.cumulative_return(next_day.returns), 0.0, abs_tol=1e-12)  # misses it


def test_costs_are_charged_on_turnover_only():
    p = toy(A=[100] * 4, B=[100] * 4)
    sched = {p.index[0]: {"A": 1.0}, p.index[2]: {"B": 1.0}}
    res = run_backtest("x", sched, p, cost_bps=100, execution_lag=0)             # 1% per unit of turnover
    assert list(res.turnover.round(9)) == [1.0, 2.0]                              # buy A; then sell A and buy B
    assert list(res.costs.round(9)) == [0.01, 0.02]
    assert math.isclose(res.nav.iloc[-1], 0.99 * 0.98, rel_tol=1e-12)
    assert math.isclose(M.cumulative_return(res.returns), 0.99 * 0.98 - 1, rel_tol=1e-12)


def test_keeping_the_same_weights_costs_nothing_when_nothing_moved():
    p = toy(A=[100] * 4, B=[100] * 4)
    same = {"A": 0.5, "B": 0.5}
    res = run_backtest("x", {p.index[0]: same, p.index[2]: same}, p, cost_bps=50, execution_lag=0)
    assert res.turnover.iloc[1] < 1e-12


def test_weights_drift_between_rebalances():
    p = toy(A=[100, 110, 110], B=[100, 100, 100])
    same = {"A": 0.5, "B": 0.5}
    res = run_backtest("x", {p.index[0]: same, p.index[1]: same}, p, cost_bps=0, execution_lag=0)
    expected = 2 * abs(0.5 - 0.55 / 1.05)                                        # A grew, so it must be trimmed back
    assert math.isclose(res.turnover.iloc[1], expected, rel_tol=1e-9)


def test_cash_earns_the_risk_free_rate():
    p = toy(A=[100] * 4)
    rf = pd.Series(0.0001, index=p.index)
    res = run_backtest("x", {p.index[0]: {}}, p, rf_daily=rf, cost_bps=0, execution_lag=0)
    assert math.isclose(res.nav.iloc[-1], 1.0001 ** 3, rel_tol=1e-12)


def test_half_invested_half_cash():
    p = toy(A=[100, 110])
    rf = pd.Series(0.01, index=p.index)
    res = run_backtest("x", {p.index[0]: {"A": 0.5}}, p, rf_daily=rf, cost_bps=0, execution_lag=0)
    assert math.isclose(res.nav.iloc[-1], 0.5 * 1.10 + 0.5 * 1.01, rel_tol=1e-12)


def test_bad_inputs_are_rejected():
    p = toy(A=[100] * 4, B=[100] * 4)
    d = p.index[0]
    with pytest.raises(ValueError):
        run_backtest("x", {d: {"A": 0.7, "B": 0.7}}, p)          # adds up to more than 1
    with pytest.raises(ValueError):
        run_backtest("x", {d: {"A": -0.1}}, p)                    # shorting is not supported
    with pytest.raises(ValueError):
        run_backtest("x", {d: {"ZZZ": 1.0}}, p)                   # no price data for that ticker
    with pytest.raises(ValueError):
        run_backtest("x", {pd.Timestamp("2024-01-06"): {"A": 1.0}}, p)   # a Saturday


def test_decisions_with_no_later_price_are_skipped():
    p = toy(A=[100, 101, 102, 103])
    sched = {p.index[0]: {"A": 1.0}, p.index[-1]: {"A": 1.0}}    # the last one cannot be traded with lag 1
    res = run_backtest("x", sched, p, cost_bps=0, execution_lag=1)
    assert len(res.turnover) == 1
    with pytest.raises(ValueError):
        run_backtest("x", {p.index[-1]: {"A": 1.0}}, p, execution_lag=1)


# ----------------------------------------------------------------------------- metrics
def test_sharpe_by_hand():
    r = pd.Series([0.02, 0.0, 0.02, 0.0], index=pd.bdate_range("2024-01-01", periods=4))
    expected = (0.01 / (0.01 * math.sqrt(4 / 3))) * math.sqrt(252)
    assert math.isclose(M.sharpe(r), expected, rel_tol=1e-9)


def test_sharpe_is_lower_when_the_risk_free_rate_is_higher():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 200), index=pd.bdate_range("2024-01-01", periods=200))
    rf = pd.Series(0.0005, index=r.index)
    assert M.sharpe(r, rf) < M.sharpe(r)


def test_max_drawdown_by_hand():
    idx = pd.bdate_range("2024-01-01", periods=3)
    assert math.isclose(M.max_drawdown(pd.Series([0.1, -0.2, 0.05], index=idx)), -0.2, rel_tol=1e-9)
    assert math.isclose(M.max_drawdown(pd.Series([-0.1, 0.05], index=idx[:2])), -0.1, rel_tol=1e-9)  # start counts as a peak


def test_annualized_return_and_flat_volatility():
    r = pd.Series(0.001, index=pd.bdate_range("2023-01-02", periods=252))
    assert math.isclose(M.annualized_return(r), 1.001 ** 252 - 1, rel_tol=1e-9)
    assert M.annualized_vol(r) < 1e-12
    assert math.isnan(M.sharpe(r))                                  # no variation, Sharpe is undefined


def test_alpha_and_beta():
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(0.0005, 0.01, 300), index=pd.bdate_range("2023-01-02", periods=300))
    alpha, beta = M.alpha_beta(2 * b, b)
    assert math.isclose(beta, 2.0, rel_tol=1e-9) and abs(alpha) < 1e-9


def test_rolling_sharpe_needs_a_full_window():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 100), index=pd.bdate_range("2024-01-01", periods=100))
    rs = M.rolling_sharpe(r, None, 63)
    assert rs.iloc[:62].isna().all() and rs.iloc[62:].notna().all()


# ----------------------------------------------------------------------------- strategies
D = pd.Timestamp("2024-06-03")


def view_at(d=D):
    pit = make_pit()
    return pit, pit.with_cutoff(d)


def test_equal_weight_and_single_asset():
    _, v = view_at()
    assert EqualWeight().weights(v, D, ["AAA", "BBB"]) == {"AAA": 0.5, "BBB": 0.5}
    assert SingleAsset("SPY").weights(v, D, []) == {"SPY": 1.0}
    assert SingleAsset("NOPE").weights(v, D, []) == {}


def test_momentum_picks_the_best_twelve_minus_one_month_return():
    _, v = view_at()
    score = {}
    for t in ["AAA", "BBB"]:
        c = v.prices(t, D, lookback_days=253)["adj_close"]
        score[t] = c.iloc[-22] / c.iloc[0] - 1
    best = max(score, key=score.get)
    assert Momentum(k=1).weights(v, D, ["AAA", "BBB"]) == {best: 1.0}


def test_low_volatility_picks_the_calmest_stock():
    _, v = view_at()
    vol = {t: v.prices(t, D, lookback_days=64)["adj_close"].pct_change().dropna().std() for t in ["AAA", "BBB"]}
    assert LowVolatility(k=1).weights(v, D, ["AAA", "BBB"]) == {min(vol, key=vol.get): 1.0}


def test_momentum_needs_enough_history():
    early = pd.Timestamp("2023-06-01")
    _, v = view_at(early)
    assert Momentum(k=1).weights(v, early, ["AAA", "BBB"]) == {}


def test_random_picks_are_repeatable():
    _, v = view_at()
    a = RandomPicks(k=1, seed=5).weights(v, D, ["AAA", "BBB"])
    assert a == RandomPicks(k=1, seed=5).weights(v, D, ["AAA", "BBB"])
    assert sum(a.values()) == 1.0 and len(a) == 1
    assert len(RandomPicks(k=2, seed=5).weights(v, D, ["AAA", "BBB"])) == 2


def test_strategy_weights_do_not_depend_on_the_future():
    """Fill the future with garbage: every strategy's weights at the decision date must stay identical."""
    raw = make_raw()
    strategies = [SingleAsset("SPY"), EqualWeight(), Momentum(1), LowVolatility(1), RandomPicks(1, 0)]
    for d in ["2024-03-01", "2024-06-03", "2024-09-02"]:
        clean = build_schedules(make_pit(raw=raw), strategies, ["AAA", "BBB"], [d])
        dirty = build_schedules(make_pit(raw=poison_future(raw, d)), strategies, ["AAA", "BBB"], [d])
        assert clean == dirty, d


# ----------------------------------------------------------------------------- whole pipeline
def test_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    from backtest.metrics import summary_table
    from data_platform.pit import monthly_decision_dates

    pit = make_pit()
    end = pit.trading_days("2100-01-01")[-1] if False else pd.Timestamp("2024-12-31")
    dates = monthly_decision_dates(pit.trading_days(end), "2024-02-01", "2024-11-30")
    strategies = [SingleAsset("SPY"), EqualWeight(), Momentum(1), LowVolatility(1), RandomPicks(1, 0)]
    schedules = build_schedules(pit, strategies, ["AAA", "BBB"], dates)
    prices = price_matrix(pit, ["AAA", "BBB", "SPY"], end)
    rf = risk_free_daily(pit, prices.index, end)
    results = {s.name: run_backtest(s.name, schedules[s.name], prices, rf, 10, 1) for s in strategies}

    table = summary_table(results, rf, bench_name="SPY buy-and-hold")
    assert len(table) == 5 and table["days"].nunique() == 1       # same period for every strategy
    assert math.isclose(table.loc["SPY buy-and-hold", "beta"], 1.0, rel_tol=1e-9)
    assert (table["total_cost"] >= 0).all()
    assert rf.iloc[0] == 0.0 and rf.iloc[5] > 0                   # rates are shifted by one day

    pytest.importorskip("matplotlib")
    from backtest.report import plot_results
    out = tmp_path / "plot.png"
    plot_results(results, out, rf)
    assert out.exists() and out.stat().st_size > 1000


def test_momentum_skips_the_most_recent_month():
    """AAA soars for 11 months then crashes in the last month; BBB climbs steadily.
    12-1 momentum (skip the last month) must pick AAA. Plain 12-month momentum would pick BBB."""
    from data_platform.pit import PointInTimeData

    idx = pd.bdate_range("2023-01-02", periods=300)
    i = np.arange(300)
    aaa = 100 * np.exp(0.004 * np.minimum(i, 278))
    aaa[279:] = aaa[278] * 0.97 ** np.arange(1, 22)
    series = {"AAA": aaa, "BBB": 100 * np.exp(0.002 * i), "SPY": 100 * np.exp(0.001 * i)}
    prices = pd.concat([pd.DataFrame({"ticker": t, "date": idx, "open": v, "high": v, "low": v, "close": v,
                                      "adj_close": v, "volume": 1_000_000, "dividends": 0.0, "splits": 0.0})
                        for t, v in series.items()], ignore_index=True)
    _, filings, news, macro = make_raw()
    d = idx[-1]
    view = PointInTimeData(prices, filings, news, macro, benchmark="SPY").with_cutoff(d)

    c = {t: view.prices(t, d, lookback_days=253)["adj_close"] for t in ["AAA", "BBB"]}
    plain_12m = max(c, key=lambda t: c[t].iloc[-1] / c[t].iloc[0])
    assert plain_12m == "BBB"                                         # the naive version would be fooled
    assert Momentum(k=1).weights(view, d, ["AAA", "BBB"]) == {"AAA": 1.0}
