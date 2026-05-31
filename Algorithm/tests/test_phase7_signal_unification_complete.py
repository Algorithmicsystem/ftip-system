"""Regression tests for Phase 7 — complete signal table unification."""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prosperity_signal_row(signal="BUY", score=0.72, confidence=0.65,
                            regime="TRENDING", as_of=None, meta=None):
    """12-column tuple for prosperity_signals_daily."""
    return (
        signal, score, confidence, regime,
        {"buy": 0.3, "sell": -0.3},
        "stacked", 0.68, 0.72,
        "phase9_v1", "feats_v1",
        meta or {},
        as_of or dt.date(2024, 6, 1),
    )


# ---------------------------------------------------------------------------
# 1. fetch_intelligence_context uses prosperity as primary source
# ---------------------------------------------------------------------------

def test_fetch_canonical_core_record_uses_prosperity_first():
    """fetch_canonical_core_record must use get_unified_signal (prosperity-first)."""
    from api.assistant import orchestrator

    unified_result = {
        "signal": "BUY", "action": "BUY",
        "score": 0.72, "confidence": 0.65,
        "regime": "TRENDING", "thresholds": {},
        "score_mode": "stacked", "base_score": 0.68, "stacked_score": 0.72,
        "entry_low": None, "entry_high": None,
        "stop_loss": None, "take_profit_1": None, "take_profit_2": None,
        "reason_codes": [], "reason_details": {},
        "signal_version": "phase9_v1", "feature_version": "feats_v1",
        "meta": {"signal_version": "phase9_v1"},
        "as_of": "2024-06-01",
        "source_table": "prosperity_signals_daily",
    }

    with patch("api.assistant.orchestrator.get_unified_signal", return_value=unified_result) as mock_get, \
         patch("api.assistant.orchestrator.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None  # features_daily empty

        result = orchestrator.fetch_canonical_core_record("AAPL", dt.date(2024, 6, 1))

    mock_get.assert_called_once_with("AAPL", dt.date(2024, 6, 1))
    assert result["signal_payload"]["action"] == "BUY"
    assert result["signal_payload"]["score"] == 0.72


def test_fetch_canonical_core_record_signal_payload_has_depth_fields():
    """signal_payload must include suppression_flags and penalty fields from meta."""
    from api.assistant import orchestrator

    depth = {
        "suppression_flags": ["LOW_CONFIDENCE"],
        "event_penalties": {"earnings_window": 0.3},
        "adjusted_confidence_notes": ["Near earnings window"],
    }
    unified_result = {
        "signal": "HOLD", "action": "HOLD",
        "score": 0.1, "confidence": 0.4,
        "regime": "CHOPPY", "thresholds": {}, "score_mode": "stacked",
        "base_score": None, "stacked_score": None,
        "entry_low": None, "entry_high": None,
        "stop_loss": None, "take_profit_1": None, "take_profit_2": None,
        "reason_codes": [], "reason_details": {},
        "signal_version": "v1", "feature_version": None,
        "meta": {"depth_adjustments": depth},
        "as_of": "2024-06-01",
        "source_table": "prosperity_signals_daily",
    }

    with patch("api.assistant.orchestrator.get_unified_signal", return_value=unified_result), \
         patch("api.assistant.orchestrator.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None

        result = orchestrator.fetch_canonical_core_record("AAPL", dt.date(2024, 6, 1))

    sp = result["signal_payload"]
    assert sp["suppression_flags"] == ["LOW_CONFIDENCE"]
    assert sp["event_penalties"] == {"earnings_window": 0.3}
    assert sp["adjusted_confidence_notes"] == ["Near earnings window"]


def test_fetch_canonical_core_record_returns_empty_when_no_data():
    """fetch_canonical_core_record returns {} when DB is disabled."""
    from api.assistant import orchestrator

    with patch("api.assistant.orchestrator.db") as mock_db:
        mock_db.db_enabled.return_value = False

        result = orchestrator.fetch_canonical_core_record("AAPL", dt.date(2024, 6, 1))

    assert result == {}


# ---------------------------------------------------------------------------
# 2. fetch_top_picks uses prosperity as primary source
# ---------------------------------------------------------------------------

def test_fetch_top_picks_uses_prosperity_when_available():
    """fetch_top_picks must read prosperity_signals_daily first."""
    from api.assistant import orchestrator

    def _fake_fetchone(query, params=None):
        if "MAX(as_of) FROM prosperity_signals_daily" in query:
            return (dt.date(2024, 6, 1),)
        return None

    prosperity_rows = [
        ("AAPL", "BUY", 0.85, 0.70, {"reason_codes": ["STRONG_MOM"]}),
        ("MSFT", "BUY", 0.75, 0.65, {}),
    ]

    def _fake_fetchall(query, params=None):
        if "FROM prosperity_signals_daily" in query:
            return prosperity_rows
        return []

    with patch.object(orchestrator.db, "safe_fetchone", side_effect=_fake_fetchone), \
         patch.object(orchestrator.db, "safe_fetchall", side_effect=_fake_fetchall):
        as_of, picks = orchestrator.fetch_top_picks(10)

    assert as_of == dt.date(2024, 6, 1)
    assert len(picks) == 2
    assert picks[0]["symbol"] == "AAPL"
    assert picks[0]["direction"] == "long"
    assert picks[0]["score"] == 0.85


def test_fetch_top_picks_falls_back_to_legacy_when_prosperity_empty():
    """fetch_top_picks must fall back to signals_daily when prosperity has no rows."""
    from api.assistant import orchestrator

    def _fake_fetchone(query, params=None):
        if "MAX(as_of) FROM prosperity_signals_daily" in query:
            return (dt.date(2024, 6, 1),)
        return None

    def _fake_fetchall(query, params=None):
        if "FROM prosperity_signals_daily" in query:
            return []  # empty — triggers fallback
        if "FROM signals_daily" in query:
            return [("TSLA", "SELL", -0.60, 0.55, ["HIGH_VOL"])]
        return []

    with patch.object(orchestrator.db, "safe_fetchone", side_effect=_fake_fetchone), \
         patch.object(orchestrator.db, "safe_fetchall", side_effect=_fake_fetchall):
        as_of, picks = orchestrator.fetch_top_picks(10)

    assert len(picks) == 1
    assert picks[0]["symbol"] == "TSLA"
    assert picks[0]["direction"] == "short"


def test_fetch_top_picks_returns_empty_when_both_tables_empty():
    """fetch_top_picks returns (None, []) when both tables have no data."""
    from api.assistant import orchestrator

    with patch.object(orchestrator.db, "safe_fetchone", return_value=(None,)), \
         patch.object(orchestrator.db, "safe_fetchall", return_value=[]):
        as_of, picks = orchestrator.fetch_top_picks(10)

    assert as_of is None
    assert picks == []


# ---------------------------------------------------------------------------
# 3. universe_coverage uses GREATEST across both tables
# ---------------------------------------------------------------------------

def test_universe_coverage_uses_prosperity_count():
    """universe_coverage must count signals from both tables via GREATEST."""
    from api.assistant import orchestrator

    def _fake_fetchone(query, params=None):
        if "GREATEST" in query:
            return (8, 10)  # 8 signals found, 10 active symbols
        return None

    with patch.object(orchestrator.db, "safe_fetchone", side_effect=_fake_fetchone):
        coverage = orchestrator.universe_coverage(dt.date(2024, 6, 1))

    assert abs(coverage - 0.8) < 1e-9


def test_universe_coverage_zero_when_no_date():
    """universe_coverage returns 0.0 when as_of_date is None."""
    from api.assistant import orchestrator
    assert orchestrator.universe_coverage(None) == 0.0


def test_universe_coverage_zero_when_no_symbols():
    """universe_coverage returns 0.0 when market_symbols is empty."""
    from api.assistant import orchestrator

    with patch.object(orchestrator.db, "safe_fetchone", return_value=(5, 0)):
        coverage = orchestrator.universe_coverage(dt.date(2024, 6, 1))

    assert coverage == 0.0


def test_universe_coverage_handles_db_none():
    """universe_coverage returns 0.0 when DB query returns None."""
    from api.assistant import orchestrator

    with patch.object(orchestrator.db, "safe_fetchone", return_value=None):
        coverage = orchestrator.universe_coverage(dt.date(2024, 6, 1))

    assert coverage == 0.0
