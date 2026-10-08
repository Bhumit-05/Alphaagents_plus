"""Offline tests for the news splitting logic and the safe-save helper (no internet, no API key)."""
from __future__ import annotations

import pandas as pd

from data_platform.loaders import news as news_mod
from data_platform.store import merge_replace

CAP = 250  # pretend the API returns at most this many articles per request


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def make_server(per_day: int, first="2025-01-01", last="2025-01-31", fail_on=None):
    """A fake news API: `per_day` articles per day, truncated to the first CAP articles per request."""
    days = pd.date_range(first, last)
    articles = []
    for d in days:
        for k in range(per_day):
            ts = d + pd.Timedelta(minutes=k * (1 if per_day > 100 else 10))   # keep every article inside its own day
            articles.append({"id": len(articles), "datetime": int(ts.timestamp()), "headline": "h",
                             "summary": "s", "source": "x", "url": f"http://x/{len(articles)}"})

    def fake_get(session, url, params=None, **kw):
        if fail_on and params["from"] == fail_on:
            raise RuntimeError("boom")
        lo = pd.Timestamp(params["from"])
        hi = pd.Timestamp(params["to"]) + pd.Timedelta(days=1)
        hits = [a for a in articles if lo.timestamp() <= a["datetime"] < hi.timestamp()]
        return FakeResponse(hits[:CAP])

    return fake_get, len(articles)


def run(fake_get, **kw):
    orig_get, orig_sleep = news_mod.get_with_retry, news_mod.time.sleep
    news_mod.get_with_retry, news_mod.time.sleep = fake_get, lambda s: None
    try:
        return news_mod.download_news_ticker("AAA", "2025-01-01", "2025-01-31", "key", pause=0, **kw)
    finally:
        news_mod.get_with_retry, news_mod.time.sleep = orig_get, orig_sleep


def test_busy_ticker_is_not_truncated():
    fake_get, total = make_server(per_day=60)           # 420 articles per week, above the API cap
    df, stats = run(fake_get)
    assert len(df) == total and stats["splits"] > 0 and stats["saturated_days"] == 0


def test_quiet_ticker_needs_no_splitting():
    fake_get, total = make_server(per_day=5)
    df, stats = run(fake_get)
    assert len(df) == total and stats["splits"] == 0


def test_failed_requests_are_reported_not_hidden():
    fake_get, total = make_server(per_day=5, fail_on="2025-01-08")
    df, stats = run(fake_get)
    assert stats["failed_windows"] == 1 and len(df) < total


def test_single_day_over_cap_is_flagged():
    fake_get, _ = make_server(per_day=300, first="2025-01-01", last="2025-01-03")
    df, stats = run(fake_get)
    assert stats["saturated_days"] >= 1 and len(df) <= 3 * CAP


def test_merge_replace_keeps_other_tickers():
    old = pd.DataFrame({"ticker": ["A", "A", "B", "C"], "v": [1, 2, 3, 4]})
    new = pd.DataFrame({"ticker": ["B"], "v": [99]})
    out = merge_replace(old, new, ["B"])
    assert sorted(out["ticker"]) == ["A", "A", "B", "C"] and out.loc[out.ticker == "B", "v"].item() == 99
    assert len(merge_replace(None, new, ["B"])) == 1
    assert len(merge_replace(old, new.iloc[0:0], ["B"])) == 3   # a failed refresh does not add rows
