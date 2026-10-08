"""Macro data loader (FRED public CSV endpoint, no API key needed).

Output schema:
    series, obs_date, available_date, value

FRED returns today's values for every date, including later revisions. We cannot recover the exact
release date of every number, so `available_date = obs_date + lag_days` (a conservative approximation set
in config.yaml). For exact vintages, FRED's ALFRED service would be needed. State this as a limitation.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from ._http import get_with_retry

log = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
COLUMNS = ["series", "obs_date", "available_date", "value"]


def download_macro(series_cfg: dict, start: str) -> pd.DataFrame:
    session = requests.Session()
    since = pd.Timestamp(start) - pd.Timedelta(days=800)  # extra history for year-over-year features
    frames = []
    for sid, cfg in series_cfg.items():
        lag = int(cfg.get("lag_days", 1))
        try:
            text = get_with_retry(session, FRED_CSV_URL, params={"id": sid}).text
        except Exception as err:
            log.error("%s: FRED download failed: %s", sid, err)
            continue
        raw = pd.read_csv(io.StringIO(text))
        date_col, value_col = raw.columns[0], raw.columns[1]
        df = pd.DataFrame({
            "series": sid,
            "obs_date": pd.to_datetime(raw[date_col]),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),  # FRED uses "." for missing
        }).dropna(subset=["value"])
        df = df[df["obs_date"] >= since]
        df["available_date"] = df["obs_date"] + pd.Timedelta(days=lag)
        frames.append(df[COLUMNS])
        log.info("%s: %d observations", sid, len(df))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)
