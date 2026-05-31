"""Regression tests for Phase 6 — unified signal read path."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prosperity_row(signal="BUY", score=0.72, confidence=0.65,
                          regime="TRENDING", signal_version="phase9_v1",
                          feature_version="feats_v1", meta=None, as_of=None):
    """Build a tuple matching the 12-column prosperity SELECT."""
    return (
        signal, score, confidence, regime,
        {"buy": 0.3, "sell": -0.3},  # thresholds
        "stacked",                   # score_mode
        0.68,                        # base_score
        0.72,                        # stacked_score
        signal_version,
        feature_version,
        meta or {},
        as_of or dt.date(2024, 6, 1),
    )


def _make_legacy_row(action="SELL", score=-0.45, confidence=0.50,
                     signal_version=2, as_of=None):
    """Build a tuple matching the 18-column legacy signals_daily SELECT."""
    return (
        action, score, confidence,
        "CHOPPY",                    # regime
        {"buy": 0.3, "sell": -0.3},  # thresholds
        "stacked",                   # score_mode
        -0.40,                       # base_score
        -0.45,                       # stacked_score
        signal_version,              # INT in legacy table
        100.0,                       # entry_low
        110.0,                       # entry_high
        95.0,                        # stop_loss
        125.0,                       # take_profit_1
        140.0,                       # take_profit_2
        ["LOW_LIQUIDITY"],           # reason_codes
        {"details": "test"},         # reason_details
        {},                          # signal_meta
        as_of or dt.date(2024, 6, 1),
    )


# ---------------------------------------------------------------------------
# 1. Module-level API
# ---------------------------------------------------------------------------

def test_get_unified_signal_is_importable():
    """get_unified_signal must be importable from api.signals.query."""
    from api.signals.query import get_unified_signal
    assert callable(get_unified_signal)


def test_get_unified_signal_returns_none_when_both_tables_empty():
    """Returns None when neither table has a row."""
    from api.signals.query import get_unified_signal
    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.return_value = None
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))
    assert result is None


# ---------------------------------------------------------------------------
# 2. Canonical source (prosperity) wins when available
# ---------------------------------------------------------------------------

def test_prosperity_wins_when_row_exists():
    """When prosperity_signals_daily has a row, it must be returned."""
    from api.signals.query import get_unified_signal
    prosperity_row = _make_prosperity_row(signal="BUY", score=0.72)

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [prosperity_row, None]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert result is not None
    assert result["source_table"] == "prosperity_signals_daily"
    assert result["signal"] == "BUY"
    assert result["score"] == 0.72


def test_legacy_only_reached_when_prosperity_empty():
    """Legacy signals_daily must only be read when prosperity returns nothing."""
    from api.signals.query import get_unified_signal
    legacy_row = _make_legacy_row(action="SELL", score=-0.45)

    with patch("api.signals.query.db") as mock_db:
        # First two calls: prosperity exact + range → both None
        # Third/fourth calls: legacy exact + range → legacy_row
        mock_db.safe_fetchone.side_effect = [None, None, legacy_row]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert result is not None
    assert result["source_table"] == "signals_daily"
    assert result["signal"] == "SELL"


# ---------------------------------------------------------------------------
# 3. Field normalisation
# ---------------------------------------------------------------------------

def test_normalised_output_has_both_signal_and_action_keys():
    """Normalised dict must expose both 'signal' and 'action' keys."""
    from api.signals.query import get_unified_signal
    prosperity_row = _make_prosperity_row(signal="HOLD")

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [prosperity_row, None]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert "signal" in result
    assert "action" in result
    assert result["signal"] == result["action"] == "HOLD"


def test_signal_version_always_string_for_legacy_int():
    """signal_version from signals_daily (INT) must be normalised to str."""
    from api.signals.query import get_unified_signal
    legacy_row = _make_legacy_row(signal_version=3)

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [None, None, legacy_row]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert result is not None
    assert isinstance(result["signal_version"], str)
    assert result["signal_version"] == "3"


def test_signal_version_string_for_prosperity():
    """signal_version from prosperity (TEXT) must remain a str."""
    from api.signals.query import get_unified_signal
    prosperity_row = _make_prosperity_row(signal_version="phase9_canonical_signal_v1")

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [prosperity_row, None]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert isinstance(result["signal_version"], str)
    assert result["signal_version"] == "phase9_canonical_signal_v1"


def test_legacy_entry_points_surfaced():
    """Entry/stop/take-profit from signals_daily must appear in normalised result."""
    from api.signals.query import get_unified_signal
    legacy_row = _make_legacy_row()

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [None, None, legacy_row]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert result["entry_low"] == 100.0
    assert result["entry_high"] == 110.0
    assert result["stop_loss"] == 95.0
    assert result["take_profit_1"] == 125.0
    assert result["take_profit_2"] == 140.0


def test_source_table_key_present():
    """Every returned dict must carry a source_table field."""
    from api.signals.query import get_unified_signal

    with patch("api.signals.query.db") as mock_db:
        mock_db.safe_fetchone.side_effect = [_make_prosperity_row(), None]
        result = get_unified_signal("AAPL", dt.date(2024, 6, 1))

    assert result["source_table"] in ("prosperity_signals_daily", "signals_daily")


# ---------------------------------------------------------------------------
# 4. fetch_signal() in orchestrator uses unified helper
# ---------------------------------------------------------------------------

def test_fetch_signal_delegates_to_unified():
    """orchestrator.fetch_signal must call get_unified_signal."""
    from api.assistant import orchestrator

    unified_result = {
        "signal": "BUY", "action": "BUY", "score": 0.6, "confidence": 0.7,
        "entry_low": None, "entry_high": None, "stop_loss": None,
        "take_profit_1": None, "take_profit_2": None,
        "reason_codes": [], "reason_details": {},
        "regime": "TRENDING", "signal_version": "v1", "feature_version": "fv1",
        "source_table": "prosperity_signals_daily", "meta": {},
    }

    with patch("api.assistant.orchestrator.get_unified_signal", return_value=unified_result) as mock_get:
        result = orchestrator.fetch_signal("AAPL", dt.date(2024, 6, 1))

    mock_get.assert_called_once_with("AAPL", dt.date(2024, 6, 1))
    assert result is not None
    assert result["action"] == "BUY"
    assert result["signal"] == "BUY"
    assert result["regime"] == "TRENDING"


def test_fetch_signal_returns_none_when_unified_returns_none():
    """fetch_signal must return None when get_unified_signal returns None."""
    from api.assistant import orchestrator

    with patch("api.assistant.orchestrator.get_unified_signal", return_value=None):
        result = orchestrator.fetch_signal("AAPL", dt.date(2024, 6, 1))

    assert result is None


def test_fetch_signal_exposes_signal_and_source_table_fields():
    """fetch_signal result must expose 'signal' and 'source_table' keys."""
    from api.assistant import orchestrator

    unified_result = {
        "signal": "SELL", "action": "SELL", "score": -0.5, "confidence": 0.6,
        "entry_low": 100.0, "entry_high": None, "stop_loss": 95.0,
        "take_profit_1": None, "take_profit_2": None,
        "reason_codes": [], "reason_details": {},
        "regime": "CHOPPY", "signal_version": "v2", "feature_version": None,
        "source_table": "signals_daily", "meta": {},
    }

    with patch("api.assistant.orchestrator.get_unified_signal", return_value=unified_result):
        result = orchestrator.fetch_signal("AAPL", dt.date(2024, 6, 1))

    assert result["signal"] == "SELL"
    assert result["source_table"] == "signals_daily"
