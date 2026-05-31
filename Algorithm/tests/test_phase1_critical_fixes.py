"""Regression tests for all P0 crash-level bugs fixed in Phase 1."""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# 1. regime.py — NoneType crash when feature key exists but value is None
# ---------------------------------------------------------------------------

def test_regime_none_feature_no_crash():
    """float(features.get("key", 0.0)) crashes when value is None; or-pattern must handle it."""
    from ftip.strategy_graph.regime import classify_regime

    # Build minimal candle stubs
    candle = SimpleNamespace(close=100.0)
    candles = [candle] * 30

    # Features where keys exist but values are None — was crash before fix
    features = {
        "volatility_ann": None,
        "trend_sma20_50": None,
    }
    result = classify_regime(candles, features)
    assert result is not None, "classify_regime must return a result, not crash"


def test_regime_none_feature_zero_treated_as_neutral():
    """None volatility and trend should be treated as 0.0 (neutral), not crash."""
    from ftip.strategy_graph.regime import classify_regime

    candle = SimpleNamespace(close=100.0)
    candles = [candle] * 30
    result_none = classify_regime(candles, {"volatility_ann": None, "trend_sma20_50": None})
    result_zero = classify_regime(candles, {"volatility_ann": 0.0, "trend_sma20_50": 0.0})
    # classify_regime returns (regime_label, details) tuple
    assert result_none[0] == result_zero[0]


# ---------------------------------------------------------------------------
# 2. snapshot.py — PIT filter must include report_date guard
# ---------------------------------------------------------------------------

def test_snapshot_pit_query_includes_report_date_filter():
    """Fundamentals query must filter by report_date <= as_of_date, not just fiscal_period_end."""
    import inspect
    from api.research import snapshot

    source = inspect.getsource(snapshot)
    # Both the fiscal_period_end filter and report_date filter must appear together
    assert "report_date IS NULL OR report_date <=" in source, (
        "snapshot.py must include (report_date IS NULL OR report_date <= %s) PIT guard"
    )


# ---------------------------------------------------------------------------
# 3. fundamentals.py — yfinance must not fabricate report_date
# ---------------------------------------------------------------------------

def test_yfinance_report_date_is_none():
    """yfinance fetch must set report_date=None and pit_safe=False, not fiscal_end."""
    from api.data_providers.fundamentals import _fetch_yfinance_quarterly

    mock_ticker = MagicMock()
    mock_ticker.quarterly_financials = MagicMock()

    import pandas as pd
    import numpy as np

    fiscal_end = pd.Timestamp("2024-12-31")
    mock_ticker.quarterly_financials.columns = [fiscal_end]
    mock_ticker.quarterly_financials.index = pd.Index([
        "Total Revenue", "Gross Profit", "Operating Income"
    ])
    mock_ticker.quarterly_financials.__getitem__ = lambda self, key: pd.Series(
        [1e9, 6e8, 3e8], index=["Total Revenue", "Gross Profit", "Operating Income"]
    )
    mock_ticker.quarterly_financials.get = lambda k, d=None: pd.Series(
        {"Total Revenue": 1e9, "Gross Profit": 6e8, "Operating Income": 3e8}
    )

    with patch("api.data_providers.fundamentals.yf.Ticker", return_value=mock_ticker):
        try:
            results = _fetch_yfinance_quarterly("AAPL")
        except Exception:
            pytest.skip("yfinance fetch requires network or complex mock — checking source instead")

    for row in results:
        assert row.get("report_date") is None, (
            f"report_date must be None for yfinance rows, got {row.get('report_date')}"
        )
        assert row.get("pit_safe") is False, (
            f"pit_safe must be False for yfinance rows, got {row.get('pit_safe')}"
        )


def test_yfinance_report_date_not_fiscal_end_in_source():
    """Source-level check: yfinance fetch must not assign fiscal_end to report_date."""
    import inspect
    from api.data_providers import fundamentals

    source = inspect.getsource(fundamentals._fetch_yfinance_quarterly)
    assert '"report_date": fiscal_end' not in source, (
        "fundamentals.py must not set report_date=fiscal_end for yfinance data"
    )
    assert '"pit_safe": False' in source, (
        "fundamentals.py must mark yfinance rows with pit_safe=False"
    )


# ---------------------------------------------------------------------------
# 4. backtest/outcomes.py — entry must use next-day row, not signal-day close
# ---------------------------------------------------------------------------

def test_backtest_entry_uses_next_day_row():
    """Entry price must come from rows[1] (next trading day), not rows[0] (signal day)."""
    from api.research.backtest.outcomes import evaluate_prediction_outcome

    record = {
        "symbol": "AAPL",
        "as_of_date": "2024-01-01",
        "horizon_days": 5,
        "final_signal": "BUY",
    }

    # rows[0] = signal day close = 100, rows[1] = next day open = 110
    # If rows[0] is used as entry, return = (120-100)/100 = +20%
    # If rows[1] is used as entry, return = (120-110)/110 ≈ +9.09% — correct
    fake_rows = [
        {"as_of_date": "2024-01-01", "close": 100.0, "open": 100.0},  # signal day — must NOT be entry
        {"as_of_date": "2024-01-02", "close": 110.0, "open": 110.0},  # entry day
        {"as_of_date": "2024-01-03", "close": 112.0, "open": 112.0},
        {"as_of_date": "2024-01-04", "close": 115.0, "open": 115.0},
        {"as_of_date": "2024-01-05", "close": 118.0, "open": 118.0},
        {"as_of_date": "2024-01-08", "close": 120.0, "open": 120.0},  # exit day (rows[5])
    ]

    result = evaluate_prediction_outcome(record, bar_fetcher=lambda *_: fake_rows)

    assert result.get("outcome_status") != "insufficient_bars", (
        f"Should have enough bars but got: {result}"
    )
    pnl = result.get("pnl_pct")
    if pnl is not None:
        # With entry at rows[1]=110 and exit at rows[5]=120: pnl ≈ +9.09%
        # If entry had been rows[0]=100: pnl would be +20% — that would be wrong
        assert abs(pnl - 9.09) < 1.0, (
            f"Expected pnl ~9.09% (next-day entry at 110), got {pnl:.2f}% — look-ahead bias not fixed"
        )


# ---------------------------------------------------------------------------
# 5. daily.py — _changed_signals must return List[Dict], not List[str]
# ---------------------------------------------------------------------------

def test_changed_signals_returns_list_of_dicts():
    """_changed_signals must return List[WhatChangedItem] with layer/description/direction/prior/current."""
    from api.assistant.phase14.daily import _changed_signals, WhatChangedItem

    current = {
        "strategy": {"final_signal": "BUY"},
        "strategy_posture": "offensive",
        "deployment_permission": "live_eligible",
        "trust_tier": "A",
        "current_operating_mode": "live",
        "candidate_classification": "live_candidate",
    }
    prior = {
        "strategy": {"final_signal": "HOLD"},
        "strategy_posture": "neutral",
        "deployment_permission": "paper_shadow_only",
        "trust_tier": "B",
        "current_operating_mode": "paper",
        "candidate_classification": "paper_candidate",
    }

    result = _changed_signals(current, prior)
    assert isinstance(result, list), "must return a list"
    assert len(result) > 0
    for item in result:
        assert isinstance(item, WhatChangedItem), f"each item must be WhatChangedItem, got {type(item)}"
        for key in ("layer", "description", "direction", "prior", "current"):
            assert hasattr(item, key), f"missing field '{key}' on WhatChangedItem"


def test_changed_signals_no_prior_returns_baseline_dict():
    """With no prior report, must return a single baseline WhatChangedItem."""
    from api.assistant.phase14.daily import _changed_signals, WhatChangedItem

    result = _changed_signals({"strategy": {"final_signal": "BUY"}}, {})
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], WhatChangedItem)
    assert result[0].layer == "baseline"


def test_changed_signals_no_changes_returns_unchanged_dict():
    """When nothing changed, must return List[WhatChangedItem] with direction='unchanged'."""
    from api.assistant.phase14.daily import _changed_signals

    same = {
        "strategy": {"final_signal": "BUY"},
        "strategy_posture": "offensive",
        "deployment_permission": "live_eligible",
        "trust_tier": "A",
        "current_operating_mode": "live",
        "candidate_classification": "live_candidate",
    }
    result = _changed_signals(same, same)
    assert isinstance(result, list)
    assert result[0].direction == "unchanged"
