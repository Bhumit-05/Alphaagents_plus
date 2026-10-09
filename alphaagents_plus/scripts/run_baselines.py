"""Run the no-AI baseline strategies and print how they performed.

Usage (from the project root):
    python -m scripts.run_baselines
    python -m scripts.run_baselines --start 2025-11-01          # the period where news exists
    python -m scripts.run_baselines --cost-bps 0 --lag 0         # see how much costs and timing matter
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summary_table
from backtest.prep import price_matrix, risk_free_daily
from backtest.report import plot_results
from backtest.strategies import EqualWeight, LowVolatility, Momentum, RandomPicks, SingleAsset, build_schedules
from data_platform.config import Config
from data_platform.pit import PointInTimeData, monthly_decision_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01", help="first decision date")
    ap.add_argument("--end", default=None, help="last decision date (default: latest price date)")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="trading cost per unit of turnover, in basis points")
    ap.add_argument("--lag", type=int, default=1, help="trading days between decision and trade")
    ap.add_argument("--k", type=int, default=4, help="stocks held by the momentum / low-vol / random strategies")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = Config.load(args.config)
    pit = PointInTimeData.from_config(cfg)
    last_day = pit.trading_days("2100-01-01")[-1]
    end = pd.Timestamp(args.end) if args.end else last_day
    all_days = pit.trading_days(last_day)
    dates = monthly_decision_dates(all_days, args.start, end)
    print(f"{len(dates)} monthly decision dates, {dates[0].date()} to {dates[-1].date()}; "
          f"cost {args.cost_bps} bps, execution lag {args.lag} day(s)")

    strategies = [SingleAsset(cfg.benchmark), EqualWeight(), Momentum(args.k), LowVolatility(args.k),
                  RandomPicks(args.k, seed=0)]
    schedules = build_schedules(pit, strategies, cfg.tickers, dates)

    prices = price_matrix(pit, cfg.price_tickers, last_day)
    rf = risk_free_daily(pit, prices.index, last_day)
    results = {s.name: run_backtest(s.name, schedules[s.name], prices, rf, args.cost_bps, args.lag) for s in strategies}

    table = summary_table(results, rf, bench_name=strategies[0].name)
    out_dir = cfg.data_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "baselines_summary.csv")
    pd.DataFrame({n: r.nav for n, r in results.items()}).to_csv(out_dir / "baselines_nav.csv")

    show = table.copy()
    for col in ["cumulative_return", "annualized_return", "annualized_vol", "max_drawdown", "pct_days_up",
                "avg_turnover", "total_cost", "alpha_yearly"]:
        show[col] = (show[col].astype(float) * 100).round(1).astype(str) + "%"
    for col in ["days", "rebalances"]:
        show[col] = show[col].astype(int)
    for col in ["sharpe", "sortino", "calmar", "beta"]:
        show[col] = show[col].astype(float).round(2)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + show.T.to_string())
    print(f"\nSaved: {out_dir / 'baselines_summary.csv'}")
    if not args.no_plots:
        try:
            plot_results(results, out_dir / "baselines.png", rf)
            print(f"Saved: {out_dir / 'baselines.png'}")
        except ImportError:
            print("matplotlib is not installed, skipping the plot (pip install matplotlib)")


if __name__ == "__main__":
    main()
