"""Download all raw data into data/.

Usage (from the project root):
    python -m scripts.download_data --all
    python -m scripts.download_data --prices --macro
    python -m scripts.download_data --filings --tickers XOM
    python -m scripts.download_data --news --news-start 2025-10-01

Running with --tickers REPLACES only those tickers' rows and keeps everything else on disk.
News is saved after every ticker, so an interrupted run keeps the tickers already finished.
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from data_platform.config import Config
from data_platform.loaders.filings import download_filings
from data_platform.loaders.macro import download_macro
from data_platform.loaders.news import download_news_ticker
from data_platform.loaders.prices import download_prices
from data_platform.store import save_replacing

log = logging.getLogger("download")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--filings", action="store_true")
    ap.add_argument("--news", action="store_true")
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--no-filing-docs", action="store_true", help="only list filings, skip HTML downloads")
    ap.add_argument("--news-start", default=None, help="first news date (default: start in config.yaml)")
    ap.add_argument("--tickers", nargs="*", help="override the tickers in config.yaml")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load(args.config)
    subset = [t.upper() for t in args.tickers] if args.tickers else None
    tickers = subset or cfg.tickers
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    do = lambda flag: args.all or flag  # noqa: E731
    if not (args.all or args.prices or args.filings or args.news or args.macro):
        ap.error("choose --all or at least one of --prices --filings --news --macro")

    if do(args.prices):
        wanted = tickers + ([cfg.benchmark] if cfg.benchmark not in tickers else [])
        df = download_prices(wanted, cfg.start, cfg.end)
        merged = save_replacing(df, cfg.prices_path, wanted)
        log.info("prices saved: %d new rows, %d total -> %s", len(df), len(merged), cfg.prices_path)

    if do(args.macro):
        df = download_macro(cfg.macro_series, cfg.start)
        df.to_parquet(cfg.macro_path, index=False)
        log.info("macro saved: %d rows -> %s", len(df), cfg.macro_path)

    if do(args.filings):
        df = download_filings(tickers, cfg.filing_forms, cfg.start, cfg.sec_user_agent,
                              cfg.data_dir, download_docs=not args.no_filing_docs,
                              cik_overrides=cfg.cik_overrides)
        merged = save_replacing(df, cfg.filings_path, tickers)
        log.info("filings metadata saved: %d new rows, %d total -> %s", len(df), len(merged), cfg.filings_path)

    if do(args.news):
        key = os.getenv("FINNHUB_API_KEY", "")
        if not key:
            log.warning("FINNHUB_API_KEY not set: skipping news. Add it to .env and rerun with --news")
        else:
            start = args.news_start or cfg.start
            for t in tickers:
                df, stats = download_news_ticker(t, start, cfg.end, key, cfg.news_chunk_days)
                save_replacing(df, cfg.news_path, [t])      # saved per ticker, so progress is kept
                log.info("news %s: %d articles saved. %s", t, len(df), stats)
                if stats["failed_windows"]:
                    log.warning("%s: %d requests failed, so this ticker has gaps. Rerun with --tickers %s",
                                t, stats["failed_windows"], t)


if __name__ == "__main__":
    main()
