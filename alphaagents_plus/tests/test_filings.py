"""Offline tests for combining filings that appear under more than one SEC number (CIK)."""
from __future__ import annotations

import pandas as pd

from data_platform.config import Config
from data_platform.loaders.filings import COLUMNS, combine_cik_filings


def frame(rows, cik):
    df = pd.DataFrame(rows, columns=["accession", "form", "filing_date", "report_date"])
    df["ticker"], df["cik"], df["acceptance_datetime"], df["primary_doc"], df["local_path"] = "XOM", cik, None, "d.htm", None
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df[COLUMNS]


OLD = frame([("o1", "10-K", "2026-02-18", "2025-12-31"), ("o2", "10-Q", "2026-05-04", "2026-03-31"),
             ("o3", "10-Q", "2026-08-03", "2026-06-30")], cik=34088)
NEW = frame([("n1", "10-Q", "2026-08-03", "2026-06-30")], cik=2115436)   # the same Q2 report, filed again


def test_same_report_under_two_ciks_is_kept_once():
    out = combine_cik_filings([OLD, NEW])
    assert len(out) == 3
    assert (out["report_date"] == pd.Timestamp("2026-06-30")).sum() == 1


def test_first_listed_cik_wins_a_same_day_tie():
    out = combine_cik_filings([OLD, NEW])
    assert out.loc[out["report_date"] == pd.Timestamp("2026-06-30"), "cik"].item() == 34088
    out2 = combine_cik_filings([NEW, OLD])
    assert out2.loc[out2["report_date"] == pd.Timestamp("2026-06-30"), "cik"].item() == 2115436


def test_distinct_filings_from_both_numbers_are_all_kept():
    later = frame([("n2", "10-Q", "2026-11-02", "2026-09-30")], cik=2115436)
    assert len(combine_cik_filings([OLD, later])) == 4


def test_rows_without_report_date_are_not_collapsed():
    a = frame([("a1", "10-K", "2026-01-01", None), ("a2", "10-K", "2026-01-02", None)], cik=1)
    assert len(combine_cik_filings([a, frame([], cik=2)])) == 2


def test_config_reads_cik_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text('tickers: [XOM]\nstart: "2022-01-01"\ncik_overrides:\n  XOM: [34088, 2115436]\n  abc: 5\n')
    cfg = Config.load(path)
    assert cfg.cik_overrides == {"XOM": [34088, 2115436], "ABC": [5]}
    path.write_text('tickers: [XOM]\nstart: "2022-01-01"\n')
    assert Config.load(path).cik_overrides == {}
