# AlphaAgents+ : Phase 1, point-in-time data platform

Goal of this phase: one data-access layer that **cannot leak the future**, plus the data downloads,
a quality report, and a feature builder, all proven by tests.

## Setup (macOS, Apple Silicon)

```bash
cd alphaagents_plus
python3 -m venv .venv && source .venv/bin/activate
python -c "import platform; print(platform.machine())"     # should print arm64
pip install -r requirements.txt

cp .env.example .env            # then paste your FINNHUB_API_KEY (free key from finnhub.io)
# edit config.yaml: set sec_user_agent to "Your Name your.email@domain.com" (SEC requires it)
```

## Run

```bash
pytest -q                                   # 1. prove the guarantees on synthetic data (no internet)
python -m scripts.download_data --prices --macro      # 2. start with the quick downloads
python -m scripts.download_data --filings --tickers AAPL    # 3. try ONE ticker before the full run
python -m scripts.download_data --all       # 4. everything (filings and news take a while)
python -m scripts.check_data                # 5. quality report + smoke test
```

Downloads are written to `data/` (git-ignored). Filing documents are skipped if already downloaded,
so you can stop and rerun safely.

## Layout

```
config.yaml                 universe, dates, lags
data_platform/
  config.py                 loads config.yaml, defines file paths
  loaders/                  prices.py  filings.py  news.py  macro.py   (raw data in, parquet out)
  pit.py                    PointInTimeData: the ONLY way the rest of the project reads data
  features.py               features computed from point-in-time data only
  quality.py                gaps and oddities report
scripts/download_data.py    download everything
scripts/check_data.py       quality report + smoke test
tests/                      synthetic data + leak tests
```

## The rules (what to tell your professor)

| Data | Usable at the close of day T when... | Why |
|---|---|---|
| Prices | date <= T | The day's close is known at the close |
| Filings | filing date + 1 day <= T | Use the date filed, not the fiscal period it covers; conservative one-day lag |
| News | published by 4pm New York time on T | Handles daylight saving time |
| Macro | observation date + publication lag <= T | Monthly data like CPI is published weeks later |

## How the tests prove "no leakage"

1. **Truncation test:** physically delete everything after T and recompute. Results must be identical.
2. **Poison test:** replace everything after T with garbage (prices x1000, fake headlines). Results must not change.
3. **Rule tests:** one test per rule above (filing lag, New York close, macro lag, hard error when asking
   for a date past the cutoff).

If any code path peeks past T, tests 1 and 2 fail.

## Known limitations (state these in your report)

- FRED returns today's (possibly revised) values; the lag is an approximation, not the true release date.
- Dividend-adjusted prices (`adj_close`) are rescaled whenever a new dividend is paid. Returns are
  unaffected, but absolute price levels in old rows differ slightly from what was seen then.
- Free news APIs limit history and call rates; coverage is checked in the quality report.
- The universe in config.yaml is today's large caps, which carries some survivorship bias.
- The downloaders have to be run against live APIs on your machine; the tests do not touch the internet.
