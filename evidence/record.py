"""Evidence records: every number an agent may use is wrapped in one, with a unique, readable ID.

ID format:  GROUP|TICKER|DATE|name     e.g.  VAL|AAPL|2025-11-03|ann_vol_63d
Later, an agent's claim ("volatility is 27%") cites the ID, and the fact-checker compares the claim with the stored value.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def make_id(group: str, ticker: str, as_of, name: str) -> str:
    return f"{group}|{ticker}|{pd.Timestamp(as_of):%Y-%m-%d}|{name}"


@dataclass(frozen=True)
class Evidence:
    id: str
    tool: str
    ticker: str            # "MKT" for market-wide numbers
    as_of: str             # YYYY-MM-DD
    name: str
    value: float
    unit: str              # fraction, ratio, percent, index_points, price, shares, z_score, flag, percentile, rsi_points
    what: str              # plain-English meaning, written so a language model reads it correctly
    window_days: int | None = None

    def to_dict(self, ndigits: int = 6) -> dict:
        d = asdict(self)
        d["value"] = round(float(self.value), ndigits)
        return d


class Collector:
    """Gathers the evidence one tool produces and remembers which numbers could not be computed."""

    def __init__(self, group: str, tool: str, ticker: str, as_of):
        self.group, self.tool, self.ticker = group, tool, ticker
        self.as_of = pd.Timestamp(as_of).normalize()
        self.items: list[Evidence] = []
        self.missing: list[str] = []

    def put(self, name: str, value, unit: str, window_days: int | None, what: str) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.missing.append(name)
            return
        self.items.append(Evidence(
            id=make_id(self.group, self.ticker, self.as_of, name), tool=self.tool, ticker=self.ticker,
            as_of=f"{self.as_of:%Y-%m-%d}", name=name, value=float(value), unit=unit, what=what,
            window_days=window_days))


class EvidenceStore:
    """Every piece of evidence handed to an agent in one run. Used later to fact-check claims."""

    def __init__(self):
        self._items: dict[str, Evidence] = {}

    def add(self, ev: Evidence) -> Evidence:
        self._items[ev.id] = ev
        return ev

    def add_many(self, evs) -> None:
        for e in evs:
            self.add(e)

    def get(self, evidence_id: str) -> Evidence:
        if evidence_id not in self._items:
            raise KeyError(f"unknown evidence id: {evidence_id}")
        return self._items[evidence_id]

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> list:
        return sorted(self._items)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([e.to_dict() for e in self._items.values()], indent=1))
