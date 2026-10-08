"""Print the data quality report and run a point-in-time smoke test.

Usage: python -m scripts.check_data [--as-of 2025-03-03]
"""
from __future__ import annotations

import argparse

import pandas as pd

from data_platform.config import Config
from data_platform.features import compute_features
from data_platform.pit import PointInTimeData
from data_platform.quality import build_report, save_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="decision date for the smoke test (default: latest price date)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    pit = PointInTimeData.from_config(cfg)
    prices, filings, news, macro = (pd.read_parquet(p) for p in
                                    (cfg.prices_path, cfg.filings_path, cfg.news_path, cfg.macro_path))

    report = build_report(prices, filings, news, macro, cfg.benchmark, cfg.tickers, cfg.data_dir)
    save_report(report, cfg.quality_dir)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for name, df in report.items():
        print(f"\n=== {name} quality ===")
        print(df.to_string(index=False))

    as_of = pd.Timestamp(args.as_of) if args.as_of else pit.trading_days("2100-01-01")[-1]
    ticker = cfg.tickers[0]
    print(f"\n=== smoke test: {ticker} as of {as_of.date()} ===")
    print("price rows visible :", len(pit.prices(ticker, as_of)))
    f = pit.latest_filing(ticker, as_of, "10-K")
    print("latest 10-K        :", None if f is None else f"filed {f['filing_date'].date()} (usable from {f['available_date'].date()})")
    print("news items (30d)   :", len(pit.news(ticker, as_of, 30)))
    print("risk-free rate     :", pit.risk_free_rate(as_of))
    print("features           :", {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in compute_features(pit, ticker, as_of).items()})
    print(f"\nSaved CSV reports to {cfg.quality_dir}")


if __name__ == "__main__":
    main()
