"""Tests for the point-in-time guarantees. The first two are the important ones."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from data_platform.features import compute_features
from data_platform.pit import FutureAccessError, market_close_utc, monthly_decision_dates
from tests.synthetic import make_pit, make_raw, poison_future

AS_OFS = ["2023-06-30", "2024-02-20", "2024-02-21", "2024-03-01", "2024-03-18", "2024-04-15", "2024-09-30"]


def snapshot(pit, as_of):
    """Everything an agent could read about ticker AAA at `as_of`."""
    filing = pit.latest_filing("AAA", as_of, "10-K")
    return {
        "prices": pit.prices("AAA", as_of, 300),
        "bench": pit.prices("SPY", as_of, 300),
        "filing": None if filing is None else filing.drop(labels=["available_date"]).to_dict(),
        "all_filings": pit.filings("AAA", as_of),
        "news": pit.news("AAA", as_of, 45),
        "cpi": pit.macro("CPIAUCSL", as_of),
        "rf_series": pit.macro("DGS1MO", as_of, 60),
        "rf": pit.risk_free_rate(as_of),
        "features": compute_features(pit, "AAA", as_of),
    }


def same(a, b, path="") -> None:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, obj=path)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, obj=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_truncation_invariance():
    """Deleting the future must not change anything an agent can read."""
    full = make_pit()
    for as_of in AS_OFS:
        same(snapshot(full, as_of), snapshot(full.with_cutoff(as_of), as_of), as_of)


def test_poisoned_future_does_not_leak():
    """Fill every future row with garbage: outputs at `as_of` must not change."""
    raw = make_raw()
    for as_of in AS_OFS:
        clean = make_pit(raw=raw)
        poisoned = make_pit(raw=poison_future(raw, as_of))
        same(snapshot(clean, as_of), snapshot(poisoned, as_of), as_of)


def test_cutoff_blocks_future_requests():
    pit = make_pit().with_cutoff("2024-03-01")
    with pytest.raises(FutureAccessError):
        pit.prices("AAA", "2024-03-04")
    with pytest.raises(FutureAccessError):
        pit.news("AAA", "2024-03-04")
    with pytest.raises(FutureAccessError):
        pit.macro("CPIAUCSL", "2024-03-04")
    with pytest.raises(FutureAccessError):
        pit.with_cutoff("2024-06-01")  # cannot widen the window again


def test_prices_never_after_as_of():
    pit = make_pit()
    df = pit.prices("AAA", "2024-03-01")
    assert df.index.max() == pd.Timestamp("2024-03-01")
    assert len(pit.prices("AAA", "2024-03-01", lookback_days=10)) == 10


def test_filing_usable_only_after_lag():
    pit = make_pit()  # filing lag = 1 day; AAA's FY2023 10-K was filed 2024-02-20
    assert pit.latest_filing("AAA", "2024-02-20", "10-K")["filing_date"] == pd.Timestamp("2023-02-21")
    assert pit.latest_filing("AAA", "2024-02-21", "10-K")["filing_date"] == pd.Timestamp("2024-02-20")
    assert pit.latest_filing("BBB", "2024-03-01", "10-K") is None   # filed that day, not yet usable
    assert pit.latest_filing("BBB", "2024-03-04", "10-K") is not None


def test_news_cutoff_is_new_york_close():
    pit = make_pit()
    mar1 = set(pit.news("AAA", "2024-03-01", 30)["headline"])
    assert "h2" in mar1 and "h3" not in mar1        # 3:30pm ET is in, 5:00pm ET is not
    mar4 = set(pit.news("AAA", "2024-03-04", 30)["headline"])
    assert "h3" in mar4 and "h4" in mar4
    assert market_close_utc("2024-03-01") == pd.Timestamp("2024-03-01 21:00")   # winter: UTC-5
    assert market_close_utc("2024-07-01") == pd.Timestamp("2024-07-01 20:00")   # summer: UTC-4


def test_macro_publication_lag():
    pit = make_pit()  # CPI for 2024-02-01 is public from 2024-03-17 (45-day lag)
    assert pit.macro("CPIAUCSL", "2024-03-15").index.max() == pd.Timestamp("2024-01-01")
    assert pit.macro("CPIAUCSL", "2024-03-18").index.max() == pd.Timestamp("2024-02-01")
    assert pit.macro_latest("NOPE", "2024-03-18") is None


def test_features_match_hand_calculation():
    pit = make_pit()
    as_of = "2024-06-28"
    close = pit.prices("AAA", as_of)["adj_close"]
    f = compute_features(pit, "AAA", as_of)
    assert math.isclose(f["ret_63d"], close.iloc[-1] / close.iloc[-64] - 1)
    assert math.isclose(f["ann_vol_63d"], close.pct_change().dropna().tail(63).std() * 252 ** 0.5)
    assert math.isclose(f["ann_ret_63d"], (1 + f["ret_63d"]) ** (252 / 63) - 1)
    assert 0 <= f["rsi_14"] <= 100


def test_features_are_nan_when_history_is_short():
    pit = make_pit()
    f = compute_features(pit, "AAA", "2023-02-01")   # only ~22 rows exist
    assert math.isnan(f["ret_252d"]) and math.isnan(f["ma200_ratio"])


def test_monthly_decision_dates():
    pit = make_pit()
    dates = monthly_decision_dates(pit.trading_days("2024-12-31"), "2024-01-01", "2024-06-30")
    assert len(dates) == 6
    assert dates[0] == pd.Timestamp("2024-01-01") and dates[1] == pd.Timestamp("2024-02-01")
