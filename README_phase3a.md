# Phase 3A: the agents' calculators (evidence tools)

Goal: every number an agent may use comes from a tool, carries a unique ID, and is dated on or before the decision date.

## Install the add-on

From the project root, with the environment on (`source .venv/bin/activate`):

```bash
unzip -o ~/Downloads/alphaagents_plus_phase3a_addon.zip
pytest -q
python -m scripts.try_tools --ticker AAPL --as-of 2025-11-03
python -m scripts.try_tools --schemas
```

The zip replaces `tests/synthetic.py` and `tests/test_quality.py` (they gained extra macro series for the new tests).
It does not touch `config.yaml` or your data.

## What is new

```
evidence/record.py      Evidence record, ID format, EvidenceStore
evidence/valuation.py   returns, volatility, drawdown, beta, volume, strength vs the market (paper formulas)
evidence/technical.py   RSI, moving averages, MACD, breakout flag, distance from the 52-week high
evidence/macro.py       VIX, rates, yield curve, inflation, unemployment
evidence/tools.py       AgentContext (one date's view of the data), Tool, ToolRegistry, per-agent tool lists
scripts/try_tools.py    call the tools by hand and print what an agent would see
tests/test_evidence.py  20 tests
```

## How it works

1. `AgentContext.create(pit, as_of, universe)` makes a copy of the data with everything after `as_of` deleted.
2. A tool (for example `get_valuation_metrics("AAPL")`) reads only from that copy.
3. It returns a list of numbers. Each has an ID such as `VAL|AAPL|2025-11-03|ann_vol_63d`, a unit, and a plain-English meaning.
4. The same numbers are saved in the context's `EvidenceStore`, so a fact-checker can later compare an agent's claim with the real value.
5. Bad input (an unknown ticker, broken JSON) returns `{"error": ...}` instead of crashing, so an agent can recover.
6. Each agent gets only its own tools (`AGENT_TOOLS`), as in the paper.

## Limitations to state in your report

- Tools return numbers, never recommendations. Judgement is the agents' job.
- Technical indicators use the usual textbook settings (RSI 14, MACD 12/26/9, 50 and 200-day averages). Other settings give other readings.
- Monthly macro series use an approximate publication lag (see config.yaml), not exact release dates.
- `last_close` is split-adjusted but not dividend-adjusted; returns use the dividend-adjusted series.
- If history is too short for a number, the tool lists it under `unavailable` instead of guessing.
