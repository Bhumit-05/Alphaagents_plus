"""Tests for the evidence tools. Most use data where the right answer can be worked out by hand."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from data_platform.features import compute_features
from data_platform.pit import FutureAccessError, PointInTimeData
from evidence.macro import macro_evidence
from evidence.record import EvidenceStore, make_id
from evidence.technical import compute_technical, technical_evidence
from evidence.tools import AGENT_TOOLS, DEFAULT_REGISTRY, AgentContext, Tool, ToolRegistry
from evidence.valuation import valuation_evidence
from tests.synthetic import make_pit, make_raw, poison_future

AS_OFS = ["2023-06-30", "2024-03-01", "2024-06-03", "2024-09-30"]
D = "2024-06-03"


def values(items):
    return {e.id: e.value for e in items}


def all_evidence(pit, as_of, ticker="AAA"):
    v, v_missing, _ = valuation_evidence(pit, ticker, as_of)
    t, t_missing = technical_evidence(pit, ticker, as_of)
    m, m_missing = macro_evidence(pit, as_of)
    return {"vals": values(v) | values(t) | values(m), "missing": [v_missing, t_missing, m_missing]}


# ----------------------------------------------------------------------------- ids and records
def test_ids_are_well_formed_unique_and_repeatable():
    pit = make_pit()
    v, _, _ = valuation_evidence(pit, "AAA", D)
    t, _ = technical_evidence(pit, "AAA", D)
    m, _ = macro_evidence(pit, D)
    assert all(e.id.startswith("VAL|AAA|2024-06-03|") for e in v)
    assert all(e.id.startswith("TEC|AAA|2024-06-03|") for e in t)
    assert all(e.id.startswith("MAC|MKT|2024-06-03|") for e in m)
    ids = [e.id for e in v + t + m]
    assert len(ids) == len(set(ids)) and len(ids) > 20
    assert all(e.id == make_id(e.id.split("|")[0], e.ticker, e.as_of, e.name) for e in v + t + m)
    v2, _, _ = valuation_evidence(pit, "AAA", D)
    assert values(v) == values(v2)                                  # same inputs, same outputs


def test_store_keeps_and_saves_evidence(tmp_path):
    store = EvidenceStore()
    v, _, _ = valuation_evidence(make_pit(), "AAA", D)
    store.add_many(v)
    assert len(store) == len(v) and v[0].id in store
    assert store.get(v[0].id).value == v[0].value
    with pytest.raises(KeyError):
        store.get("VAL|AAA|2024-06-03|nonsense")
    store.save(tmp_path / "e.json")
    assert len(json.loads((tmp_path / "e.json").read_text())) == len(v)


# ----------------------------------------------------------------------------- valuation
def test_valuation_matches_hand_calculation():
    pit = make_pit()
    close = pit.prices("AAA", D)["adj_close"]
    v = {e.name: e.value for e in valuation_evidence(pit, "AAA", D)[0]}
    daily = close.pct_change().dropna()
    assert math.isclose(v["ann_vol_63d"], daily.tail(63).std() * 252 ** 0.5, rel_tol=1e-9)
    assert math.isclose(v["ret_3m"], close.iloc[-1] / close.iloc[-64] - 1, rel_tol=1e-9)
    assert math.isclose(v["ann_ret_63d"], (1 + v["ret_3m"]) ** (252 / 63) - 1, rel_tol=1e-9)   # the paper's formula
    assert math.isclose(v["last_close"], pit.prices("AAA", D)["close"].iloc[-1], rel_tol=1e-12)
    spy = compute_features(pit, "SPY", D)["ret_63d"]
    assert math.isclose(v["rel_strength_3m"], v["ret_3m"] - spy, rel_tol=1e-9)


def test_valuation_reports_what_it_cannot_compute():
    items, missing, price_date = valuation_evidence(make_pit(), "AAA", "2023-02-01")   # only ~22 days of history
    assert "ret_12m" in missing and "ann_vol_63d" in missing
    assert price_date == "2023-02-01"
    assert all(not math.isnan(e.value) for e in items)


# ----------------------------------------------------------------------------- technical
def test_technical_on_a_steady_rise():
    out = compute_technical(pd.Series(np.arange(1, 301, dtype=float)))
    assert out["rsi_14"] == 100.0 and out["breakout_20d"] == 1.0 and out["ma50_above_ma200"] == 1.0
    assert out["macd_line"] > 0 and out["pct_from_52w_high"] == 0.0 and out["ma50_ratio"] > 0


def test_technical_on_a_steady_fall():
    out = compute_technical(pd.Series(np.arange(300, 0, -1, dtype=float)))
    assert out["rsi_14"] == 0.0 and out["breakout_20d"] == -1.0 and out["ma50_above_ma200"] == -1.0
    assert out["macd_line"] < 0 and math.isclose(out["pct_from_52w_high"], 1 / 252 - 1, rel_tol=1e-9)


def test_breakout_flag_by_hand():
    assert compute_technical(pd.Series([1.0] * 20 + [2.0]))["breakout_20d"] == 1.0
    assert compute_technical(pd.Series([5.0] * 20 + [1.0]))["breakout_20d"] == -1.0
    assert compute_technical(pd.Series([3.0] * 20 + [3.0]))["breakout_20d"] == 0.0


def test_macd_is_zero_for_a_flat_price_and_nan_for_a_short_history():
    out = compute_technical(pd.Series([100.0] * 60))
    assert out["macd_line"] == 0.0 and out["macd_hist"] == 0.0 and out["ma50_ratio"] == 0.0
    short = compute_technical(pd.Series(np.arange(1, 11, dtype=float)))
    assert math.isnan(short["macd_line"]) and math.isnan(short["breakout_20d"]) and math.isnan(short["rsi_14"])


# ----------------------------------------------------------------------------- macro
def macro_pit(drop=()):
    days = pd.bdate_range("2023-06-01", "2024-06-28")
    n = len(days)
    daily = {"VIXCLS": np.arange(1, n + 1, dtype=float), "DGS1MO": np.full(n, 5.0),
             "DGS2": np.full(n, 4.0), "DGS10": np.full(n, 4.5)}
    frames = []
    for sid, vals in daily.items():
        f = pd.DataFrame({"series": sid, "obs_date": days, "value": vals})
        f["available_date"] = f["obs_date"] + pd.Timedelta(days=1)
        frames.append(f)
    months = pd.date_range("2023-01-01", periods=17, freq="MS")
    for sid, vals, lag in [("CPIAUCSL", 100 + np.arange(17.0), 45), ("UNRATE", 4.0 + 0.1 * np.arange(17), 38)]:
        f = pd.DataFrame({"series": sid, "obs_date": months, "value": vals})
        f["available_date"] = f["obs_date"] + pd.Timedelta(days=lag)
        frames.append(f)
    macro = pd.concat(frames, ignore_index=True)
    macro = macro[~macro["series"].isin(drop)]
    prices, filings, news, _ = make_raw()
    return PointInTimeData(prices, filings, news, macro, benchmark="SPY"), n


def test_macro_by_hand():
    pit, n = macro_pit()
    m = {e.name: e.value for e in macro_evidence(pit, "2024-06-28")[0]}
    assert m["vix_level"] == n - 1                      # the 28 June reading only becomes public on the 29th
    assert m["vix_percentile_1y"] == 1.0 and m["vix_change_21d"] == 21
    assert m["rate_1m"] == 5.0 and m["yield_2y"] == 4.0 and m["yield_10y"] == 4.5
    assert math.isclose(m["yield_curve_10y_2y"], 0.5) and abs(m["yield_curve_change_63d"]) < 1e-12
    assert math.isclose(m["cpi_yoy"], 116 / 104 - 1, rel_tol=1e-9)
    assert math.isclose(m["unemployment_rate"], 5.6) and math.isclose(m["unemployment_change_3m"], 0.3, abs_tol=1e-9)


def test_macro_waits_for_the_publication_lag():
    pit, _ = macro_pit()
    early = {e.name: e.value for e in macro_evidence(pit, "2024-06-10")[0]}     # May CPI (lag 45 days) is not out yet
    late = {e.name: e.value for e in macro_evidence(pit, "2024-06-28")[0]}
    assert math.isclose(early["cpi_yoy"], 115 / 103 - 1, rel_tol=1e-9)
    assert math.isclose(late["cpi_yoy"], 116 / 104 - 1, rel_tol=1e-9)


def test_macro_reports_missing_series_and_keeps_going():
    pit, _ = macro_pit(drop=("VIXCLS",))
    items, missing = macro_evidence(pit, "2024-06-28")
    assert {"vix_level", "vix_percentile_1y", "vix_change_21d"} <= set(missing)
    assert "yield_curve_10y_2y" in {e.name for e in items}


# ----------------------------------------------------------------------------- no peeking
def test_tools_ignore_a_poisoned_future():
    """Fill the future with garbage. The functions are called with the FULL data, so only the as_of filter protects them."""
    raw = make_raw()
    for as_of in AS_OFS:
        clean = all_evidence(make_pit(raw=raw), as_of)
        dirty = all_evidence(make_pit(raw=poison_future(raw, as_of)), as_of)
        assert clean == dirty, as_of


def test_tools_give_the_same_answer_when_the_future_is_deleted():
    full = make_pit()
    for as_of in AS_OFS:
        assert all_evidence(full, as_of) == all_evidence(full.with_cutoff(as_of), as_of), as_of


def test_context_physically_removes_the_future():
    ctx = AgentContext.create(make_pit(), D, ["AAA", "BBB"])
    assert ctx.pit.cutoff == pd.Timestamp(D)
    with pytest.raises(FutureAccessError):
        ctx.pit.prices("AAA", "2024-06-04")


# ----------------------------------------------------------------------------- registry
def new_ctx():
    return AgentContext.create(make_pit(), D, ["AAA", "BBB"])


def test_valuation_tool_end_to_end():
    ctx = new_ctx()
    r = DEFAULT_REGISTRY.call(ctx, "get_valuation_metrics", {"ticker": "aaa"})      # lower case is accepted
    assert r["ticker"] == "AAA" and r["as_of"] == D and r["latest_price_date"] == D
    assert len(r["metrics"]) >= 10 and all(m["id"] in ctx.store for m in r["metrics"])
    json.dumps(r)                                                                  # must be JSON-serializable
    assert all(set(m) == {"id", "name", "value", "unit", "meaning"} for m in r["metrics"])


def test_all_three_tools_fill_the_store():
    ctx = new_ctx()
    for name, args in [("get_valuation_metrics", {"ticker": "BBB"}), ("get_technical_indicators", {"ticker": "BBB"}),
                       ("get_macro_snapshot", {})]:
        r = DEFAULT_REGISTRY.call(ctx, name, args)
        assert "error" not in r and r["metrics"], name
    prefixes = {i.split("|")[0] for i in ctx.store.ids()}
    assert prefixes == {"VAL", "TEC", "MAC"}


def test_benchmark_is_an_allowed_ticker():
    r = DEFAULT_REGISTRY.call(new_ctx(), "get_valuation_metrics", {"ticker": "SPY"})
    rel = [m for m in r["metrics"] if m["name"] == "rel_strength_3m"][0]
    assert abs(rel["value"]) < 1e-12


def test_bad_calls_return_errors_instead_of_raising():
    ctx = new_ctx()
    assert "Unknown ticker" in DEFAULT_REGISTRY.call(ctx, "get_valuation_metrics", {"ticker": "ZZZ"})["error"]
    assert "Missing" in DEFAULT_REGISTRY.call(ctx, "get_valuation_metrics", {})["error"]
    assert "Unknown tool" in DEFAULT_REGISTRY.call(ctx, "buy_everything", {})["error"]
    assert "valid JSON" in DEFAULT_REGISTRY.call(ctx, "get_valuation_metrics", "{not json")["error"]
    assert "error" not in DEFAULT_REGISTRY.call(ctx, "get_valuation_metrics", '{"ticker": "AAA"}')   # a JSON string works
    assert "error" not in DEFAULT_REGISTRY.call(ctx, "get_macro_snapshot", "")                        # empty args work


def test_a_crashing_tool_is_reported_not_raised():
    boom = Tool("boom", "always fails", {"type": "object", "properties": {}}, lambda ctx, a: 1 / 0)
    r = ToolRegistry([boom]).call(new_ctx(), "boom", {})
    assert "ZeroDivisionError" in r["error"]


def test_each_agent_sees_only_its_own_tools():
    for agent, names in AGENT_TOOLS.items():
        schemas = DEFAULT_REGISTRY.schemas(names)
        assert [s["function"]["name"] for s in schemas] == names, agent
        for s in schemas:
            assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"
            json.dumps(s)
    assert len(DEFAULT_REGISTRY.schemas()) == 3
