from __future__ import annotations

import pandas as pd

from data_platform.quality import build_report
from tests.synthetic import make_raw


def test_quality_report_runs_and_finds_gaps():
    prices, filings, news, macro = make_raw()
    gap = prices[~((prices["ticker"] == "AAA") & (prices["date"].between("2024-03-04", "2024-03-08")))]
    report = build_report(gap, filings, news, macro, "SPY", ["AAA", "BBB"])
    pq = report["prices"].set_index("ticker")
    assert pq.loc["AAA", "missing_vs_benchmark"] == 5
    assert pq.loc["BBB", "missing_vs_benchmark"] == 0
    nq = report["news"].set_index("ticker")
    assert nq.loc["AAA", "n_articles"] == 6
    assert nq.loc["BBB", "n_articles"] == 0 and bool(nq.loc["BBB", "low_coverage"])
    assert set(report["macro"]["series"]) == {"DGS1MO", "CPIAUCSL", "VIXCLS", "DGS2", "DGS10", "UNRATE"}


def test_parquet_round_trip(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("pyarrow")
    prices, *_ = make_raw()
    path = tmp_path / "p.parquet"
    prices.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), prices)
