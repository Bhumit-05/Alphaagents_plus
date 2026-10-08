"""SEC EDGAR filing loader (10-K / 10-Q).

We store, for every filing, the date it was FILED (not the fiscal period it covers).
The filing date is what decides when the document became public.

Metadata schema:
    ticker, cik, accession, form, filing_date, report_date, acceptance_datetime,
    primary_doc, local_path

SEC rules: send a descriptive User-Agent with your contact email and stay under 10 requests/second.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from ._http import get_with_retry

log = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_EXTRA_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

COLUMNS = ["ticker", "cik", "accession", "form", "filing_date", "report_date",
           "acceptance_datetime", "primary_doc", "local_path"]


def make_session(user_agent: str) -> requests.Session:
    if not user_agent or "example.com" in user_agent:
        raise ValueError("Set sec_user_agent in config.yaml to your real name and email (SEC requirement).")
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def get_cik_map(session: requests.Session) -> dict:
    data = get_with_retry(session, SEC_TICKERS_URL).json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _frame_from_block(block: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "accession": block.get("accessionNumber", []),
        "form": block.get("form", []),
        "filing_date": block.get("filingDate", []),
        "report_date": block.get("reportDate", []),
        "acceptance_datetime": block.get("acceptanceDateTime", []),
        "primary_doc": block.get("primaryDocument", []),
    })


def list_filings(session: requests.Session, ticker: str, cik: int, forms: list, start: str) -> pd.DataFrame:
    """List filings of the wanted forms, going back far enough that the first decision date has a 10-K."""
    since = pd.Timestamp(start) - pd.Timedelta(days=450)
    data = get_with_retry(session, SUBMISSIONS_URL.format(cik=cik)).json()
    frames = [_frame_from_block(data["filings"]["recent"])]

    # Older filings live in extra JSON pages. Fetch them while we still need earlier history.
    for extra in data["filings"].get("files", []):
        if pd.Timestamp(extra.get("filingTo", "1900-01-01")) < since:
            continue
        time.sleep(0.2)
        block = get_with_retry(session, SUBMISSIONS_EXTRA_URL.format(name=extra["name"])).json()
        frames.append(_frame_from_block(block))

    df = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    df = df[df["form"].isin(forms)].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df[df["filing_date"] >= since]
    df["ticker"] = ticker
    df["cik"] = cik
    df["local_path"] = None
    return df[COLUMNS].sort_values("filing_date").reset_index(drop=True)


def download_filing_docs(session: requests.Session, meta: pd.DataFrame, out_dir: Path,
                         data_dir: Path, pause: float = 0.2) -> pd.DataFrame:
    """Download each filing's main HTML document. Already-downloaded files are skipped."""
    meta = meta.copy()
    paths = []
    for row in meta.itertuples():
        rel = Path("filings") / row.ticker / f"{row.filing_date:%Y-%m-%d}_{row.form}_{row.accession}.htm"
        full = data_dir / rel
        if full.exists():
            paths.append(str(rel))
            continue
        url = ARCHIVE_URL.format(cik=row.cik, acc=row.accession.replace("-", ""), doc=row.primary_doc)
        try:
            resp = get_with_retry(session, url, timeout=60)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(resp.content)
            paths.append(str(rel))
            log.info("saved %s", rel)
        except Exception as err:
            log.error("could not download %s: %s", url, err)
            paths.append(None)
        time.sleep(pause)
    meta["local_path"] = paths
    return meta


def combine_cik_filings(frames: list) -> pd.DataFrame:
    """Merge the filing lists of several SEC numbers (CIKs) that belong to the same stock.

    After a restructuring (for example a new holding company) the same report can be filed under both the
    old and the new number. We keep ONE copy per (form, report period): the earliest filed, and the number
    listed first in config.yaml when both were filed on the same day. Rows without a report date are kept.
    """
    parts = []
    for order, df in enumerate(frames):
        df = df.copy()
        df["_order"] = order
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    meta = pd.concat(parts, ignore_index=True).sort_values(["filing_date", "_order"])
    has_date = meta["report_date"].notna()
    meta = pd.concat([meta[has_date].drop_duplicates(["form", "report_date"], keep="first"), meta[~has_date]])
    return meta.drop(columns="_order").sort_values("filing_date").reset_index(drop=True)[COLUMNS]


def download_filings(tickers: list, forms: list, start: str, user_agent: str, data_dir: Path,
                     download_docs: bool = True, cik_overrides: dict | None = None) -> pd.DataFrame:
    session = make_session(user_agent)
    cik_map = get_cik_map(session)
    overrides = {k.upper(): list(v) for k, v in (cik_overrides or {}).items()}
    frames = []
    for ticker in tickers:
        ciks = overrides.get(ticker.upper()) or ([cik_map[ticker.upper()]] if ticker.upper() in cik_map else [])
        if not ciks:
            log.error("%s: no CIK found (ETFs such as SPY have no 10-K/10-Q)", ticker)
            continue
        parts = []
        for cik in ciks:
            time.sleep(0.2)
            parts.append(list_filings(session, ticker, cik, forms, start))
        meta = combine_cik_filings(parts) if len(parts) > 1 else parts[0]
        log.info("%s: %d filings listed (CIKs %s)", ticker, len(meta), ciks)
        if download_docs:
            meta = download_filing_docs(session, meta, data_dir / "filings", data_dir)
        frames.append(meta)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)
