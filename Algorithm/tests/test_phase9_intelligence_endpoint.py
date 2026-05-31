"""Regression tests for Phase 9 — symbol intelligence endpoint and WhatChangedItem model."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. WhatChangedItem model
# ---------------------------------------------------------------------------

def test_what_changed_item_importable():
    from api.assistant.phase14.daily import WhatChangedItem
    assert WhatChangedItem


def test_what_changed_item_has_required_fields():
    from api.assistant.phase14.daily import WhatChangedItem
    item = WhatChangedItem(
        layer="signal",
        description="Signal changed from HOLD to BUY.",
        direction="changed",
        prior="HOLD",
        current="BUY",
    )
    assert item.layer == "signal"
    assert item.direction == "changed"
    assert item.prior == "HOLD"
    assert item.current == "BUY"


def test_changed_signals_returns_what_changed_items():
    """_changed_signals must return List[WhatChangedItem], not List[Dict]."""
    from api.assistant.phase14.daily import _changed_signals, WhatChangedItem
    current = {
        "strategy": {"final_signal": "BUY"},
        "deployment_permission": "live_eligible",
        "trust_tier": "T1",
        "current_operating_mode": "live",
        "candidate_classification": "core",
    }
    prior = {
        "strategy": {"final_signal": "HOLD"},
        "deployment_permission": "paper_shadow_only",
        "trust_tier": "T2",
        "current_operating_mode": "shadow",
        "candidate_classification": "candidate",
    }
    items = _changed_signals(current, prior)
    assert len(items) > 0
    for item in items:
        assert isinstance(item, WhatChangedItem)


def test_changed_signals_detects_signal_change():
    from api.assistant.phase14.daily import _changed_signals
    current = {"strategy": {"final_signal": "BUY"}}
    prior = {"strategy": {"final_signal": "SELL"}}
    items = _changed_signals(current, prior)
    signal_items = [i for i in items if i.layer == "signal"]
    assert len(signal_items) == 1
    assert signal_items[0].prior == "SELL"
    assert signal_items[0].current == "BUY"
    assert signal_items[0].direction == "changed"


def test_changed_signals_returns_unchanged_when_no_diff():
    from api.assistant.phase14.daily import _changed_signals
    report = {
        "strategy": {"final_signal": "BUY"},
        "deployment_permission": "live_eligible",
        "trust_tier": "T1",
        "current_operating_mode": "live",
        "candidate_classification": "core",
    }
    items = _changed_signals(report, report)
    assert len(items) == 1
    assert items[0].layer == "none"
    assert items[0].direction == "unchanged"


def test_changed_signals_baseline_when_no_prior():
    from api.assistant.phase14.daily import _changed_signals
    items = _changed_signals({"strategy": {"final_signal": "BUY"}}, {})
    assert len(items) == 1
    assert items[0].layer == "baseline"
    assert items[0].direction == "new"


def test_build_daily_workflow_what_changed_panel_is_list_of_dicts():
    """what_changed_panel must contain serialized dicts with all WhatChangedItem fields."""
    from api.assistant.phase14.daily import build_daily_workflow
    current = {
        "symbol": "AAPL",
        "strategy": {"final_signal": "BUY"},
        "deployment_permission": "live_eligible",
        "trust_tier": "T1",
        "current_operating_mode": "live",
        "candidate_classification": "core",
    }
    prior = {
        "symbol": "AAPL",
        "strategy": {"final_signal": "HOLD"},
        "deployment_permission": "paper_shadow_only",
        "trust_tier": "T2",
        "current_operating_mode": "shadow",
        "candidate_classification": "candidate",
    }
    result = build_daily_workflow(
        current,
        recent_reports=[current, prior],
        recent_shadow_records=[],
        recent_incidents=[],
    )
    panel = result["what_changed_panel"]
    assert isinstance(panel, list)
    assert len(panel) > 0
    for item in panel:
        assert isinstance(item, dict)
        assert "layer" in item
        assert "direction" in item
        assert "prior" in item
        assert "current" in item
        assert "description" in item


def test_build_daily_workflow_changed_signals_is_text_list():
    """changed_signals key must be a plain list of description strings."""
    from api.assistant.phase14.daily import build_daily_workflow
    report = {
        "symbol": "AAPL",
        "strategy": {"final_signal": "BUY"},
        "deployment_permission": "live_eligible",
        "trust_tier": "T1",
    }
    result = build_daily_workflow(
        report,
        recent_reports=[report],
        recent_shadow_records=[],
        recent_incidents=[],
    )
    for entry in result["changed_signals"]:
        assert isinstance(entry, str)


# ---------------------------------------------------------------------------
# 2. GET /signals/intelligence endpoint
# ---------------------------------------------------------------------------

def _make_unified_signal(symbol="AAPL", as_of="2024-06-01"):
    return {
        "signal": "BUY", "action": "BUY",
        "score": 0.72, "confidence": 0.65,
        "regime": "TRENDING", "thresholds": {},
        "score_mode": "stacked", "base_score": 0.68, "stacked_score": 0.72,
        "entry_low": 180.0, "entry_high": 185.0,
        "stop_loss": 175.0, "take_profit_1": 195.0, "take_profit_2": 205.0,
        "reason_codes": ["STRONG_TREND"], "reason_details": {},
        "signal_version": "phase9_v1", "feature_version": "feats_v1",
        "meta": {}, "as_of": as_of,
        "source_table": "prosperity_signals_daily",
    }


def test_signal_intelligence_route_exists():
    """GET /signals/intelligence route must be registered."""
    from api.signals import routes
    paths = [r.path for r in routes.router.routes]
    assert "/signals/intelligence" in paths


def test_signal_intelligence_returns_explanation_keys():
    """intelligence endpoint must return signal fields + explanation block."""
    from api.signals.routes import signal_intelligence

    unified = _make_unified_signal()
    with patch("api.signals.routes.get_unified_signal", return_value=unified), \
         patch("api.signals.routes.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None  # features_daily empty

        import asyncio
        result = asyncio.run(signal_intelligence("AAPL", dt.date(2024, 6, 1)))

    assert result["symbol"] == "AAPL"
    assert result["signal"] == "BUY"
    assert result["score"] == 0.72
    assert result["source_table"] == "prosperity_signals_daily"
    assert "explanation" in result
    assert "top_drivers" in result["explanation"]
    assert "staleness_label" in result["explanation"]
    assert "reason_codes" in result["explanation"]


def test_signal_intelligence_includes_system_confidence():
    """system_confidence must be present as a top-level field."""
    from api.signals.routes import signal_intelligence

    unified = _make_unified_signal()
    with patch("api.signals.routes.get_unified_signal", return_value=unified), \
         patch("api.signals.routes.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None

        import asyncio
        result = asyncio.run(signal_intelligence("AAPL", dt.date(2024, 6, 1)))

    assert "system_confidence" in result
    assert isinstance(result["system_confidence"], float)


def test_signal_intelligence_features_enrich_drivers():
    """When features_daily row is present, top_drivers must be populated."""
    from api.signals.routes import signal_intelligence

    unified = _make_unified_signal()
    # Provide mom_21=0.12 (strong supporter for BUY) and rsi14=65 (bullish)
    feature_row = (
        0.02,  # mom_5
        0.12,  # mom_21
        0.18,  # mom_63
        None, None,  # mom_126, mom_252
        65.0,  # rsi14
        0.08,  # trend_sma20_50
        0.85,  # trend_r2_63d
        None, None, None, None, None,  # volume_z20 and rest
    )

    with patch("api.signals.routes.get_unified_signal", return_value=unified), \
         patch("api.signals.routes.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = feature_row

        import asyncio
        result = asyncio.run(signal_intelligence("AAPL", dt.date(2024, 6, 1)))

    drivers = result["explanation"]["top_drivers"]
    assert len(drivers) > 0
    features_seen = {d["feature"] for d in drivers}
    assert "mom_21" in features_seen or "rsi14" in features_seen


def test_signal_intelligence_404_when_no_signal():
    """intelligence endpoint must return 404 when unified signal not found."""
    from fastapi import HTTPException
    from api.signals.routes import signal_intelligence

    with patch("api.signals.routes.get_unified_signal", return_value=None), \
         patch("api.signals.routes.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True

        import asyncio
        try:
            asyncio.run(signal_intelligence("ZZZZ", dt.date(2024, 6, 1)))
            assert False, "should have raised HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 404


def test_signal_intelligence_staleness_fields_present():
    """Explanation block must include staleness fields."""
    from api.signals.routes import signal_intelligence

    unified = _make_unified_signal(as_of="2024-01-01")
    with patch("api.signals.routes.get_unified_signal", return_value=unified), \
         patch("api.signals.routes.db") as mock_db, \
         patch("api.assistant.explanation.dt") as mock_dt:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None
        mock_dt.date.today.return_value = dt.date(2024, 1, 5)
        mock_dt.date.fromisoformat = dt.date.fromisoformat

        import asyncio
        result = asyncio.run(signal_intelligence("AAPL", dt.date(2024, 1, 1)))

    expl = result["explanation"]
    assert expl.get("signal_age_days") == 4
    assert expl.get("staleness_label") == "aging"
