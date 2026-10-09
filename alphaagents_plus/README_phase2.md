# Phase 2: backtest engine and baselines (no AI yet)

Goal: a scoring machine. It takes "hold these stocks on these dates" and replays history to show the
return, the risk and the cost. We test it on simple strategies first, so every AI agent later has numbers to beat.

## Install the add-on

Unzip it from the project root (it only adds files, plus a fixed `data_platform/pit.py`):

```bash
cd alphaagents_plus
unzip -o ~/Downloads/alphaagents_plus_phase2_addon.zip
pip install matplotlib
pytest -q
python -m scripts.run_baselines
```

## What is new

```
backtest/engine.py       replays weights through prices (timing, costs, drift, cash)
backtest/metrics.py      return, volatility, Sharpe, Sortino, drawdown, Calmar, alpha, beta
backtest/strategies.py   baselines: SPY, equal-weight, momentum, low-volatility, random picks
backtest/prep.py         price table and risk-free rate for the engine
backtest/report.py       cumulative-return and rolling-Sharpe plot
scripts/run_baselines.py runs everything and prints a table
tests/test_backtest.py   hand-checked tests
```

## The rules of the engine

| Topic | Rule |
|---|---|
| Decision | Made at the close of day T, using a copy of the data that physically stops at T |
| Trade | At the close of T + 1 trading day (`--lag 1`); the position earns returns from the next day on |
| Between rebalances | No trading: weights drift with prices |
| Cost | turnover x cost_bps / 10,000, where turnover = sum of absolute weight changes (buys and sells) |
| Cash | Weights below 100% leave cash, which earns the risk-free rate (1-month Treasury, prior day's value) |
| Long only | No shorting, no leverage (weights add up to at most 1) |
| Sharpe | Annualized: mean daily excess return / its std x sqrt(252). The paper does not annualize. |

## Limitations to state in your report

- Trades happen at closing prices with no slippage or market impact; costs are a flat basis-point assumption.
- Decisions are monthly (first trading day of each month). Results depend on that choice.
- Few monthly decisions means noisy statistics. Phase 8 adds confidence intervals and the random-portfolio test.
- The 12 stocks are today's large caps (survivorship bias), so absolute returns look better than a real investor's.
- `adj_close` includes dividends; cash earns the rate shown, so dividend-heavy stocks are treated fairly.
