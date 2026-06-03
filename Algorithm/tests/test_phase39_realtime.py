"""Phase 10: Real-Time Intelligence tests.

Tests for intraday engine, WebSocket manager, morning briefing,
intraday IC tracking, and scheduler.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(open_=100.0, high=105.0, low=98.0, close=103.0, volume=10000.0) -> Dict[str, Any]:
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _make_bars(n=10, volume=10000.0) -> list:
    return [_make_bar(volume=volume) for _ in range(n)]


# ---------------------------------------------------------------------------
# TestIntradayEngine
# ---------------------------------------------------------------------------

class TestIntradayEngine:
    def test_vwap_deviation_zero_for_empty_bars(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        assert compute_vwap_deviation([]) == 0.0

    def test_vwap_deviation_positive_above_vwap(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        # Last close = 110, VWAP ≈ (95+90+110)/3 = 98.33
        bars = [
            {"high": 95.0, "low": 90.0, "close": 92.0, "volume": 1000.0},
            {"high": 95.0, "low": 90.0, "close": 92.0, "volume": 1000.0},
            {"high": 115.0, "low": 105.0, "close": 110.0, "volume": 500.0},  # big close
        ]
        dev = compute_vwap_deviation(bars)
        assert dev > 0.0

    def test_vwap_deviation_negative_below_vwap(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        bars = [
            {"high": 105.0, "low": 100.0, "close": 104.0, "volume": 1000.0},
            {"high": 105.0, "low": 100.0, "close": 104.0, "volume": 1000.0},
            {"high": 100.0, "low": 92.0, "close": 93.0, "volume": 500.0},  # big down close
        ]
        dev = compute_vwap_deviation(bars)
        assert dev < 0.0

    def test_volume_surge_neutral_at_average(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        avg_daily = 390_000.0  # 390k shares/day
        expected_rate = avg_daily / 390.0  # 1000 shares/min
        score = compute_volume_surge_score(expected_rate, avg_daily)
        # surge_ratio = 1.0 → score = 0.0
        assert score == pytest.approx(0.0, abs=0.1)

    def test_volume_surge_high_when_elevated(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        avg_daily = 390_000.0
        expected_rate = avg_daily / 390.0  # 1000 shares/min
        score = compute_volume_surge_score(expected_rate * 4, avg_daily)  # 4× = surge_ratio 4
        assert score > 70.0

    def test_volume_surge_zero_at_zero_rate(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        score = compute_volume_surge_score(0.0, 390_000.0)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_intraday_composite_bounded(self):
        from api.axiom.intraday.intraday_engine import compute_intraday_composite
        # Even extreme values must be in [0, 100]
        c1 = compute_intraday_composite(200.0, 200.0, 200.0)
        assert 0.0 <= c1 <= 100.0
        c2 = compute_intraday_composite(-50.0, -50.0, -50.0)
        assert 0.0 <= c2 <= 100.0

    def test_intraday_composite_anchored_to_daily(self):
        from api.axiom.intraday.intraday_engine import compute_intraday_composite, _INTRADAY_BASE_WEIGHT
        # With intraday = 50 (neutral), composite should track daily × base_weight + 50 × remainder
        daily = 70.0
        composite = compute_intraday_composite(daily, 50.0, 50.0)
        # = 70 * 0.50 + 50 * 0.30 + 50 * 0.20 = 35 + 15 + 10 = 60
        assert composite == pytest.approx(60.0, abs=0.5)

    def test_run_intraday_graceful_empty(self):
        from api.axiom.intraday.intraday_engine import run_intraday_update
        snapshot = run_intraday_update("AAPL", [], 65.0, 1_000_000.0)
        assert snapshot.symbol == "AAPL"
        assert snapshot.intraday_flow_score is None
        assert snapshot.intraday_composite is None
        assert snapshot.alert_eligible is False

    def test_run_intraday_with_bars(self):
        from api.axiom.intraday.intraday_engine import run_intraday_update
        bars = _make_bars(n=15)
        snapshot = run_intraday_update("MSFT", bars, 65.0, 1_000_000.0)
        assert snapshot.symbol == "MSFT"
        assert snapshot.intraday_composite is not None
        assert 0.0 <= snapshot.intraday_composite <= 100.0

    def test_vwap_deviation_zero_volume(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        bars = [{"high": 100.0, "low": 90.0, "close": 95.0, "volume": 0.0}]
        assert compute_vwap_deviation(bars) == 0.0

    def test_volume_surge_zero_avg(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        # avg_daily_volume = 0 → return 50.0
        score = compute_volume_surge_score(1000.0, 0.0)
        assert score == pytest.approx(50.0, abs=0.1)


# ---------------------------------------------------------------------------
# TestWebSocketManager
# ---------------------------------------------------------------------------

class TestWebSocketManager:
    def test_connection_count_starts_zero(self):
        from api.realtime.websocket_manager import WebSocketManager
        mgr = WebSocketManager()
        assert mgr.connection_count() == 0

    def test_opportunity_alert_structure(self):
        from api.realtime.websocket_manager import build_opportunity_alert
        alert = build_opportunity_alert("AAPL", 78.0, "BULL_TRENDING", "momentum_surge")
        assert alert["alert_type"] == "opportunity"
        assert alert["symbol"] == "AAPL"
        assert "severity" in alert
        assert "timestamp" in alert
        assert "payload" in alert
        assert alert["payload"]["dau"] == 78.0

    def test_risk_alert_structure(self):
        from api.realtime.websocket_manager import build_risk_alert
        alert = build_risk_alert("TSLA", "bubble_risk", "warning", "SCPS elevated")
        assert alert["alert_type"] == "risk"
        assert alert["payload"]["risk_type"] == "bubble_risk"
        assert alert["severity"] == "warning"

    def test_regime_change_alert_structure(self):
        from api.realtime.websocket_manager import build_regime_change_alert
        alert = build_regime_change_alert("TRENDING", "HIGH_VOL", 15)
        assert alert["alert_type"] == "regime_change"
        assert alert["payload"]["from_regime"] == "TRENDING"
        assert alert["payload"]["to_regime"] == "HIGH_VOL"
        assert alert["payload"]["symbol_count"] == 15

    def test_bubble_warning_structure(self):
        from api.realtime.websocket_manager import build_bubble_warning
        alert = build_bubble_warning("GME", 75.0, 68.0)
        assert alert["alert_type"] == "bubble_warning"
        assert alert["payload"]["scps_score"] == 75.0
        assert alert["payload"]["bfs_score"] == 68.0

    def test_earnings_stress_alert_structure(self):
        from api.realtime.websocket_manager import build_earnings_stress_alert
        alert = build_earnings_stress_alert("AAPL", 82.0, 12)
        assert alert["alert_type"] == "earnings_stress"
        assert alert["payload"]["pess_score"] == 82.0
        assert alert["payload"]["days_to_earnings"] == 12

    def test_opportunity_alert_severity_info(self):
        from api.realtime.websocket_manager import build_opportunity_alert
        alert = build_opportunity_alert("X", 60.0, "CHOPPY", "value")
        assert alert["severity"] == "info"

    def test_opportunity_alert_severity_warning(self):
        from api.realtime.websocket_manager import build_opportunity_alert
        alert = build_opportunity_alert("X", 80.0, "TRENDING", "momentum")
        assert alert["severity"] == "warning"

    def test_bubble_warning_critical_severity(self):
        from api.realtime.websocket_manager import build_bubble_warning
        alert = build_bubble_warning("MEME", 85.0, 82.0)
        assert alert["severity"] == "critical"

    def test_earnings_stress_critical_severity(self):
        from api.realtime.websocket_manager import build_earnings_stress_alert
        alert = build_earnings_stress_alert("X", 85.0, 5)
        assert alert["severity"] == "critical"


# ---------------------------------------------------------------------------
# TestMorningBriefing
# ---------------------------------------------------------------------------

class TestMorningBriefing:
    def test_sri_neutral_no_data(self):
        from api.jobs.morning_briefing import compute_systemic_risk_index
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            sri = compute_systemic_risk_index(dt.date.today())
        assert sri == pytest.approx(50.0, abs=0.01)

    def test_sri_bounded(self):
        from api.jobs.morning_briefing import compute_systemic_risk_index
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            for _ in range(5):
                sri = compute_systemic_risk_index(dt.date.today())
                assert 0.0 <= sri <= 100.0

    def test_generate_briefing_returns_dataclass(self):
        from api.jobs.morning_briefing import MorningBriefing, generate_morning_briefing
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            briefing = generate_morning_briefing(dt.date.today())
        assert isinstance(briefing, MorningBriefing)
        assert briefing.briefing_date == dt.date.today()

    def test_briefing_text_three_paragraphs(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            briefing = generate_morning_briefing(dt.date.today())
        # 3 paragraphs separated by \n\n → at least 2 double-newlines = 4 \n chars
        assert briefing.briefing_text.count("\n") >= 3

    def test_briefing_text_mentions_regime(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            briefing = generate_morning_briefing(dt.date.today())
        assert "regime" in briefing.briefing_text.lower()

    def test_systemic_risk_label(self):
        from api.jobs.morning_briefing import _sri_label
        assert _sri_label(85.0) == "critical"
        assert _sri_label(60.0) == "elevated"
        assert _sri_label(30.0) == "normal"

    def test_briefing_text_mentions_ic_state(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            briefing = generate_morning_briefing(dt.date.today())
        assert "IC" in briefing.briefing_text or "ic" in briefing.briefing_text.lower()

    def test_briefing_has_systemic_risk_index(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        with patch("api.jobs.morning_briefing.db.db_read_enabled", return_value=False):
            briefing = generate_morning_briefing(dt.date.today())
        assert 0.0 <= briefing.systemic_risk_index <= 100.0


# ---------------------------------------------------------------------------
# TestIntradayIC
# ---------------------------------------------------------------------------

class TestIntradayIC:
    def test_intraday_ic_insufficient_no_data(self):
        from api.jobs.intraday_ic import compute_intraday_ic
        with patch("api.jobs.intraday_ic.db.db_read_enabled", return_value=False):
            result = compute_intraday_ic(dt.date.today(), 16)
        assert result["ic_state"] == "INSUFFICIENT"
        assert result["sample_count"] == 0

    def test_time_of_day_calendar_structure(self):
        from api.jobs.intraday_ic import compute_time_of_day_ic_calendar
        with patch("api.jobs.intraday_ic.db.db_read_enabled", return_value=False):
            cal = compute_time_of_day_ic_calendar()
        for h in ("10", "12", "14", "16"):
            assert h in cal
        assert "best_hour" in cal

    def test_best_hour_is_one_of_four(self):
        from api.jobs.intraday_ic import compute_time_of_day_ic_calendar
        with patch("api.jobs.intraday_ic.db.db_read_enabled", return_value=False):
            cal = compute_time_of_day_ic_calendar()
        assert cal["best_hour"] in {"10", "12", "14", "16"}

    def test_intraday_ic_returns_isodate(self):
        from api.jobs.intraday_ic import compute_intraday_ic
        today = dt.date.today()
        with patch("api.jobs.intraday_ic.db.db_read_enabled", return_value=False):
            result = compute_intraday_ic(today, 10)
        assert result["session_date"] == today.isoformat()
        assert result["update_hour"] == 10


# ---------------------------------------------------------------------------
# TestScheduler
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_scheduler_not_started_by_default(self):
        from api.config import scheduler_enabled
        # Default: FTIP_SCHEDULER_ENABLED not set → False
        assert scheduler_enabled() is False

    def test_get_status_when_stopped(self):
        from api.jobs.scheduler import SchedulerManager
        mgr = SchedulerManager()
        status = mgr.get_status()
        assert status["running"] is False

    def test_job_ids_defined(self):
        from api.jobs.scheduler import _JOB_IDS
        expected = {
            "morning_briefing",
            "intraday_update_10",
            "intraday_ic_10",
            "intraday_update_12",
            "intraday_update_14",
            "intraday_update_16",
            "full_daily_pipeline",
            "ml_training_check",
            "memory_consolidation",
        }
        assert _JOB_IDS == expected

    def test_trigger_job_unknown_id(self):
        from api.jobs.scheduler import SchedulerManager
        mgr = SchedulerManager()
        result = mgr.trigger_job("nonexistent_job_xyz")
        assert result["status"] == "error"
        assert "error" in result

    def test_trigger_job_not_running(self):
        from api.jobs.scheduler import SchedulerManager
        mgr = SchedulerManager()
        # Any valid job ID should fail gracefully when not running
        result = mgr.trigger_job("morning_briefing")
        assert result["status"] == "error"

    def test_get_status_structure(self):
        from api.jobs.scheduler import SchedulerManager
        mgr = SchedulerManager()
        status = mgr.get_status()
        assert "running" in status
        assert "next_run_times" in status
        assert "last_run_results" in status

    def test_job_count_is_nine(self):
        from api.jobs.scheduler import _JOB_IDS
        assert len(_JOB_IDS) == 9


# ---------------------------------------------------------------------------
# TestMarketOpenAlert
# ---------------------------------------------------------------------------

class TestMarketOpenAlert:
    def test_market_open_alert_structure(self):
        from api.jobs.market_open_alert import generate_market_open_alert
        with patch("api.jobs.market_open_alert.db.db_read_enabled", return_value=False):
            result = generate_market_open_alert(dt.date.today())
        assert "high_conviction_longs" in result
        assert "high_conviction_shorts" in result
        assert "watch_list" in result
        assert "alert_text" in result
        assert isinstance(result["alert_text"], str)

    def test_market_open_alert_text_nonempty(self):
        from api.jobs.market_open_alert import generate_market_open_alert
        with patch("api.jobs.market_open_alert.db.db_read_enabled", return_value=False):
            result = generate_market_open_alert(dt.date.today())
        assert len(result["alert_text"]) > 0
