"""Configuration loading for the data platform."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    tickers: list
    benchmark: str
    start: str
    end: str | None
    data_dir: Path
    sec_user_agent: str
    filing_forms: list
    filing_lag_days: int
    news_provider: str
    news_chunk_days: int
    macro_series: dict
    cik_overrides: dict = field(default_factory=dict)   # ticker -> list of SEC numbers (CIKs)

    # ---- file locations -------------------------------------------------
    @property
    def prices_path(self) -> Path:
        return self.data_dir / "prices.parquet"

    @property
    def filings_path(self) -> Path:
        return self.data_dir / "filings_meta.parquet"

    @property
    def filings_dir(self) -> Path:
        return self.data_dir / "filings"

    @property
    def news_path(self) -> Path:
        return self.data_dir / "news.parquet"

    @property
    def macro_path(self) -> Path:
        return self.data_dir / "macro.parquet"

    @property
    def features_path(self) -> Path:
        return self.data_dir / "features.parquet"

    @property
    def quality_dir(self) -> Path:
        return self.data_dir / "quality"

    @property
    def price_tickers(self) -> list:
        """Universe plus the benchmark, without duplicates."""
        out = list(self.tickers)
        if self.benchmark not in out:
            out.append(self.benchmark)
        return out

    @staticmethod
    def load(path: str | Path | None = None) -> "Config":
        from dotenv import load_dotenv  # imported here so tests do not need it

        load_dotenv(ROOT / ".env")
        path = Path(path) if path else ROOT / "config.yaml"
        raw = yaml.safe_load(path.read_text())
        data_dir = Path(raw.get("data_dir", "data"))
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        news = raw.get("news", {})
        return Config(
            tickers=[t.upper() for t in raw["tickers"]],
            benchmark=raw.get("benchmark", "SPY").upper(),
            start=str(raw["start"]),
            end=None if raw.get("end") in (None, "null") else str(raw["end"]),
            data_dir=data_dir,
            sec_user_agent=raw.get("sec_user_agent", ""),
            filing_forms=list(raw.get("filing_forms", ["10-K", "10-Q"])),
            filing_lag_days=int(raw.get("filing_lag_days", 1)),
            news_provider=news.get("provider", "finnhub"),
            news_chunk_days=int(news.get("chunk_days", 7)),
            macro_series=dict(raw.get("macro_series", {})),
            cik_overrides={str(k).upper(): [int(x) for x in (v if isinstance(v, list) else [v])]
                           for k, v in (raw.get("cik_overrides") or {}).items()},
        )
