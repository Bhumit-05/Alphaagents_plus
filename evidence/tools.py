"""The agents' toolbox.

* AgentContext: ONE decision date's view of the data. It is built with pit.with_cutoff(as_of), so everything
  after that date is physically deleted. Tools reading from it cannot see the future.
* Tool / ToolRegistry: each tool has a name, a plain description and a parameter schema in the standard
  "function calling" format, which Qwen and most other model APIs understand.
* Tools never raise errors at the model. A bad ticker or bad arguments produce {"error": ...} so the agent can recover.
* Every number returned is also stored in the context's EvidenceStore, keyed by its ID, for fact-checking later.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from data_platform.pit import PointInTimeData

from .macro import macro_evidence
from .record import EvidenceStore
from .technical import technical_evidence
from .valuation import valuation_evidence

log = logging.getLogger(__name__)


@dataclass
class AgentContext:
    pit: PointInTimeData          # a view that ends at as_of (nothing later exists in it)
    as_of: pd.Timestamp
    store: EvidenceStore
    universe: list

    @classmethod
    def create(cls, pit: PointInTimeData, as_of, universe: list, store: EvidenceStore | None = None) -> "AgentContext":
        as_of = pd.Timestamp(as_of).normalize()
        return cls(pit.with_cutoff(as_of), as_of, store or EvidenceStore(), list(universe))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[AgentContext, dict], dict]

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {t.name: t for t in tools}

    def names(self) -> list:
        return sorted(self._tools)

    def schemas(self, names: list | None = None) -> list:
        """Tool descriptions to send to the model. Pass `names` to give an agent only its own tools."""
        chosen = names if names is not None else self.names()
        return [self._tools[n].schema() for n in chosen]

    def call(self, ctx: AgentContext, name: str, args) -> dict:
        """Run a tool. `args` may be a dict or the JSON string a model sends. Never raises."""
        if name not in self._tools:
            return {"error": f"Unknown tool '{name}'. Available: {self.names()}"}
        try:
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            args = args or {}
        except json.JSONDecodeError:
            return {"error": "Arguments must be valid JSON."}
        tool = self._tools[name]
        missing = [p for p in tool.parameters.get("required", []) if p not in args]
        if missing:
            return {"error": f"Missing required argument(s): {missing}"}
        try:
            return tool.fn(ctx, args)
        except Exception as err:  # a tool bug must not crash an agent run; it is logged and reported
            log.warning("tool %s failed: %s: %s", name, type(err).__name__, err)
            return {"error": f"{type(err).__name__}: {err}"}


# ----------------------------------------------------------------------------- the three tools
def _clean_ticker(ctx: AgentContext, args: dict):
    t = str(args.get("ticker", "")).strip().upper()
    if t not in ctx.universe and t != ctx.pit.benchmark:
        return None, {"error": f"Unknown ticker '{t}'. Available: {sorted(ctx.universe)}"}
    return t, None


def _package(ctx: AgentContext, tool: str, ticker: str, items: list, missing: list, **extra) -> dict:
    ctx.store.add_many(items)
    out = {"tool": tool, "ticker": ticker, "as_of": f"{ctx.as_of:%Y-%m-%d}",
           "metrics": [{"id": e.id, "name": e.name, "value": round(e.value, 6), "unit": e.unit, "meaning": e.what}
                       for e in items],
           "unavailable": missing}
    out.update(extra)
    return out


def _valuation(ctx: AgentContext, args: dict) -> dict:
    t, err = _clean_ticker(ctx, args)
    if err:
        return err
    items, missing, price_date = valuation_evidence(ctx.pit, t, ctx.as_of)
    return _package(ctx, "valuation", t, items, missing, latest_price_date=price_date)


def _technical(ctx: AgentContext, args: dict) -> dict:
    t, err = _clean_ticker(ctx, args)
    if err:
        return err
    items, missing = technical_evidence(ctx.pit, t, ctx.as_of)
    return _package(ctx, "technical", t, items, missing)


def _macro(ctx: AgentContext, args: dict) -> dict:
    items, missing = macro_evidence(ctx.pit, ctx.as_of)
    return _package(ctx, "macro", "MKT", items, missing)


_TICKER_PARAM = {"type": "object", "required": ["ticker"],
                 "properties": {"ticker": {"type": "string", "description": "Stock symbol from the universe, e.g. AAPL"}}}

DEFAULT_REGISTRY = ToolRegistry([
    Tool("get_valuation_metrics",
         "Price and volume statistics for one stock as of the decision date: returns, volatility, drawdown, beta, "
         "volume, and strength versus the market. Use it to judge how the price has behaved.",
         _TICKER_PARAM, _valuation),
    Tool("get_technical_indicators",
         "Chart indicators for one stock as of the decision date: RSI, moving averages, MACD, breakout flag, "
         "distance from the 52-week high. Use it to judge momentum and trend.",
         _TICKER_PARAM, _technical),
    Tool("get_macro_snapshot",
         "Market-wide conditions as of the decision date: VIX, interest rates, yield curve, inflation, unemployment. "
         "Takes no arguments.",
         {"type": "object", "properties": {}}, _macro),
])

# Which tools each agent may use. The fundamental and sentiment agents get theirs in Phases 3B to 3D.
AGENT_TOOLS = {
    "valuation": ["get_valuation_metrics"],
    "technical": ["get_technical_indicators"],
    "macro": ["get_macro_snapshot"],
}
