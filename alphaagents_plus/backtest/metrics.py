"""Performance numbers. All inputs are DAILY returns (a pandas Series indexed by date).

Sharpe here is ANNUALIZED (mean daily excess return / its standard deviation * sqrt(252)).
The AlphaAgents paper does not annualize; say which one you use when you compare numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _excess(r: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return r
    return r - rf.reindex(r.index).ffill().fillna(0.0)


def cumulative_return(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def annualized_return(r: pd.Series) -> float:
    n = len(r)
    return float((1.0 + r).prod() ** (TRADING_DAYS / n) - 1.0) if n else np.nan


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan


def sharpe(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if len(ex) > 1 and sd > 1e-12 else np.nan


def sortino(r: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(r, rf)
    downside = np.sqrt(np.mean(np.minimum(ex.values, 0.0) ** 2))
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if len(ex) > 1 and downside > 1e-12 else np.nan


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the portfolio value, counting the starting value as a peak."""
    nav = (1.0 + r).cumprod().values
    running_peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    return float((nav / running_peak - 1.0).min()) if len(nav) else np.nan


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return float(annualized_return(r) / abs(mdd)) if mdd < 0 else np.nan


def alpha_beta(r: pd.Series, bench: pd.Series, rf: pd.Series | None = None):
    """Beta = how much the strategy moves with the benchmark. Alpha = yearly return beyond what beta explains."""
    both = pd.concat([_excess(r, rf), _excess(bench, rf)], axis=1, join="inner").dropna()
    if len(both) < 30 or both.iloc[:, 1].var() < 1e-18:
        return np.nan, np.nan
    beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
    alpha = (both.iloc[:, 0].mean() - beta * both.iloc[:, 1].mean()) * TRADING_DAYS
    return float(alpha), float(beta)


def rolling_sharpe(r: pd.Series, rf: pd.Series | None = None, window: int = 63) -> pd.Series:
    ex = _excess(r, rf)
    return ex.rolling(window).mean() / ex.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def summarize(result, rf: pd.Series | None = None, bench: pd.Series | None = None) -> dict:
    r = result.returns
    out = {
        "days": len(r),
        "cumulative_return": cumulative_return(r),
        "annualized_return": annualized_return(r),
        "annualized_vol": annualized_vol(r),
        "sharpe": sharpe(r, rf),
        "sortino": sortino(r, rf),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "pct_days_up": float((r > 0).mean()),
        "rebalances": len(result.turnover),
        "avg_turnover": float(result.turnover.iloc[1:].mean()) if len(result.turnover) > 1 else np.nan,
        "total_cost": float(result.costs.sum()),
    }
    if bench is not None:
        out["alpha_yearly"], out["beta"] = alpha_beta(r, bench, rf)
    return out


def summary_table(results: dict, rf: pd.Series | None = None, bench_name: str | None = None) -> pd.DataFrame:
    bench = results[bench_name].returns if bench_name in results else None
    return pd.DataFrame({name: summarize(res, rf, bench) for name, res in results.items()}).T
