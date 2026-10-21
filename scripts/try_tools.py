"""Call the agents' tools by hand and see exactly what an agent would see.

Usage (from the project root):
    python -m scripts.try_tools --ticker AAPL --as-of 2025-11-03
    python -m scripts.try_tools --schemas
"""
from __future__ import annotations

import argparse
import json

from data_platform.config import Config
from data_platform.pit import PointInTimeData
from evidence.tools import AGENT_TOOLS, DEFAULT_REGISTRY, AgentContext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--as-of", default="2025-11-03")
    ap.add_argument("--schemas", action="store_true", help="print the tool descriptions sent to the model")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.schemas:
        for agent, names in AGENT_TOOLS.items():
            print(f"\n=== tools for the {agent} agent ===")
            print(json.dumps(DEFAULT_REGISTRY.schemas(names), indent=2))
        return

    cfg = Config.load(args.config)
    ticker = (args.ticker or cfg.tickers[0]).upper()
    ctx = AgentContext.create(PointInTimeData.from_config(cfg), args.as_of, cfg.tickers)
    for name, call_args in [("get_valuation_metrics", {"ticker": ticker}),
                            ("get_technical_indicators", {"ticker": ticker}),
                            ("get_macro_snapshot", {})]:
        result = DEFAULT_REGISTRY.call(ctx, name, call_args)
        print(f"\n=== {name}({json.dumps(call_args)}) ===")
        for m in result.get("metrics", []):
            print(f"  {m['id']:<52} {m['value']:>14,.4f}   {m['unit']}")
        if result.get("unavailable"):
            print("  could not compute:", result["unavailable"])
        if "error" in result:
            print("  ERROR:", result["error"])
    print(f"\n{len(ctx.store)} pieces of evidence stored, all dated on or before {ctx.as_of.date()}.")


if __name__ == "__main__":
    main()
