"""Small synthetic dataset used by the tests (no internet needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_platform.pit import PointInTimeData

DAYS = pd.bdate_range("2023-01-02", "2024-12-31")


def make_raw(seed: int = 0):
    rng = np.random.default_rng(seed)

    price_frames = []
    for t in ["AAA", "BBB", "SPY"]:
        ret = rng.normal(0.0004, 0.01, len(DAYS))
        close = 100 * np.cumprod(1 + ret)
        price_frames.append(pd.DataFrame({
            "ticker": t, "date": DAYS, "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close, "volume": rng.integers(1_000_000, 2_000_000, len(DAYS)),
            "dividends": 0.0, "splits": 0.0,
        }))
    prices = pd.concat(price_frames, ignore_index=True)

    filings = pd.DataFrame([
        ("AAA", "0001", "10-K", "2023-02-21", "2022-12-31", "aaa_2023_10k.htm"),
        ("AAA", "0002", "10-Q", "2023-05-05", "2023-03-31", "aaa_2023_q1.htm"),
        ("AAA", "0003", "10-K", "2024-02-20", "2023-12-31", "aaa_2024_10k.htm"),
        ("AAA", "0004", "10-Q", "2024-05-03", "2024-03-31", "aaa_2024_q1.htm"),
        ("BBB", "0005", "10-K", "2024-03-01", "2023-12-31", "bbb_2024_10k.htm"),
    ], columns=["ticker", "accession", "form", "filing_date", "report_date", "primary_doc"])
    filings["cik"] = 1
    filings["acceptance_datetime"] = None
    filings["local_path"] = "filings/" + filings["ticker"] + "/" + filings["primary_doc"]

    news = pd.DataFrame({
        "ticker": "AAA",
        "published_at": pd.to_datetime([
            "2024-02-10 15:00", "2024-02-28 14:00",
            "2024-03-01 20:30",   # 3:30pm New York: public before the close of Mar 1
            "2024-03-01 22:00",   # 5:00pm New York: after the close of Mar 1
            "2024-03-04 14:00", "2024-04-02 12:00",
        ]),
    })
    news["headline"] = ["h%d" % i for i in range(len(news))]
    news["summary"], news["source"] = "s", "src"
    news["url"] = ["http://x/%d" % i for i in range(len(news))]
    news["article_id"] = [str(i) for i in range(len(news))]

    rf = pd.DataFrame({"series": "DGS1MO", "obs_date": DAYS, "value": 5.0 + rng.normal(0, 0.05, len(DAYS))})
    rf["available_date"] = rf["obs_date"] + pd.Timedelta(days=1)
    cpi_dates = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    cpi = pd.DataFrame({"series": "CPIAUCSL", "obs_date": cpi_dates, "value": np.linspace(300, 320, len(cpi_dates))})
    cpi["available_date"] = cpi["obs_date"] + pd.Timedelta(days=45)
    extra = []
    for sid, base in [("VIXCLS", 18.0), ("DGS2", 4.0), ("DGS10", 4.4)]:
        df = pd.DataFrame({"series": sid, "obs_date": DAYS, "value": base + rng.normal(0, 0.3, len(DAYS))})
        df["available_date"] = df["obs_date"] + pd.Timedelta(days=1)
        extra.append(df)
    un_dates = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    un = pd.DataFrame({"series": "UNRATE", "obs_date": un_dates, "value": 4.0 + rng.normal(0, 0.1, len(un_dates))})
    un["available_date"] = un["obs_date"] + pd.Timedelta(days=38)
    macro = pd.concat([rf, cpi, *extra, un], ignore_index=True)

    return prices, filings, news, macro


def make_pit(seed: int = 0, raw=None, **kw) -> PointInTimeData:
    prices, filings, news, macro = raw if raw is not None else make_raw(seed)
    return PointInTimeData(prices, filings, news, macro, benchmark="SPY", filing_lag_days=1, **kw)


def poison_future(raw, after):
    """Return a copy of the raw data in which everything that becomes public AFTER `after` holds garbage.

    If any code path peeks into the future, its output will change and the tests will fail.
    """
    after = pd.Timestamp(after).normalize()
    prices, filings, news, macro = (d.copy() for d in raw)

    future = prices["date"] > after
    for col in ("open", "high", "low", "close", "adj_close"):
        prices.loc[future, col] = prices.loc[future, col] * 1000.0
    prices.loc[future, "volume"] = 1

    filings["available_date_tmp"] = pd.to_datetime(filings["filing_date"]) + pd.Timedelta(days=1)
    fut_f = filings["available_date_tmp"] > after
    filings.loc[fut_f, "primary_doc"] = "POISON"
    filings.loc[fut_f, "local_path"] = "POISON"
    filings = filings.drop(columns="available_date_tmp")

    close_utc = pd.Timestamp(after) + pd.Timedelta(hours=21)  # generous: after the NY close in any season
    fut_n = news["published_at"] > close_utc
    news.loc[fut_n, "headline"] = "FUTURE"

    fut_m = macro["available_date"] > after
    macro.loc[fut_m, "value"] = 99999.0
    return prices, filings, news, macro
