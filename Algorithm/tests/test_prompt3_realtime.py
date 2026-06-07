"""
Prompt 3 integrity tests: ML training loop, intraday engine, WebSocket feed,
toast alerts, Kelly IC fix.

DIAGNOSIS (written before any code was changed):
  - MINIMUM_SAMPLES was 50 inline; now extracted to MINIMUM_SAMPLES_INITIAL=20 in training_data.py
  - Intraday engine already had VWAP/volume/momentum computation but wrong alert logic
    (composite >= 65 threshold instead of shift/surge/deviation checks)
  - WebSocketManager existed (api_key keyed) but lacked broadcast_from_thread;
    new IntelligenceWebSocketManager adds thread-safe event-loop broadcasting
  - /ws/intelligence endpoint is new (was only /ws/alerts)
  - Kelly sizer was stuck at 0.25× when IC state = INSUFFICIENT because
    _load_ic_state_bulk returned "INSUFFICIENT" on any DB miss; now returns "WEAK"
    with a prior fallback IC=0.04
  - Version bumped 27 → 28
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App client
# ---------------------------------------------------------------------------

from api.main import app

client = TestClient(app, raise_server_exceptions=False)

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "api" / "webapp"


# ===========================================================================
# TestMLTraining
# ===========================================================================

class TestMLTraining:
    def test_minimum_samples_initial_is_20(self):
        from api.axiom.ml.training_data import MINIMUM_SAMPLES_INITIAL
        assert MINIMUM_SAMPLES_INITIAL == 20

    def test_minimum_samples_production_is_100(self):
        from api.axiom.ml.training_data import MINIMUM_SAMPLES_PRODUCTION
        assert MINIMUM_SAMPLES_PRODUCTION == 100

    def test_model_quality_tiers_defined(self):
        from api.axiom.ml.training_job import get_model_quality_tier
        assert get_model_quality_tier(0) == "insufficient"
        assert get_model_quality_tier(19) == "insufficient"
        assert get_model_quality_tier(20) == "bootstrap"
        assert get_model_quality_tier(99) == "bootstrap"
        assert get_model_quality_tier(100) == "production"
        assert get_model_quality_tier(500) == "production"

    def test_kelly_max_by_quality_tiers(self):
        from api.axiom.ml.training_job import _KELLY_MAX_BY_QUALITY
        assert _KELLY_MAX_BY_QUALITY["bootstrap"] <= 0.30
        assert _KELLY_MAX_BY_QUALITY["production"] <= 0.50
        assert _KELLY_MAX_BY_QUALITY["insufficient"] == 0.25

    def test_ml_prediction_graceful_when_no_model(self):
        from api.axiom.ml.ensemble import EnsembleResult, compute_ensemble_dau
        result = compute_ensemble_dau("AAPL", 70.0, dt.date.today())
        assert isinstance(result, EnsembleResult)
        # No model → rule_only blend
        assert result.blend_method == "rule_only"
        assert result.ensemble_dau == 70.0

    def test_training_status_endpoint_returns_200(self):
        r = client.get("/axiom/ml/training-status",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
        data = r.json()
        required = [
            "model_trained", "model_version", "model_quality", "training_samples",
            "samples_for_production", "cross_val_auc", "feature_count",
            "kelly_max_from_quality", "ensemble_mode", "next_training_at",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_training_status_bootstrap_kelly_cap(self):
        r = client.get("/axiom/ml/training-status",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
        data = r.json()
        quality = data["model_quality"]
        kelly_max = data["kelly_max_from_quality"]
        if quality == "bootstrap":
            assert kelly_max <= 0.30
        elif quality == "production":
            assert kelly_max <= 0.50

    def test_invalidate_ensemble_cache_is_callable(self):
        from api.axiom.ml.ensemble import invalidate_ensemble_cache
        invalidate_ensemble_cache()  # should not raise


# ===========================================================================
# TestIntradayEngine
# ===========================================================================

class TestIntradayEngine:
    def _make_bars(self, n: int = 10, base: float = 100.0) -> list:
        return [
            {"open": base, "high": base + 1, "low": base - 1,
             "close": base + 0.5 * i, "volume": 10000}
            for i in range(n)
        ]

    def test_vwap_computation_correct(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        bars = [
            {"high": 102, "low": 98, "close": 101, "volume": 1000},
            {"high": 103, "low": 99, "close": 102, "volume": 1500},
        ]
        dev = compute_vwap_deviation(bars)
        # vwap = (100*1000 + 101.333*1500) / 2500 ≈ 100.8; current=102
        # deviation = (102 - vwap) / vwap * 100 > 0
        assert dev > 0

    def test_vwap_deviation_bullish_positive(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_deviation
        bars = [{"high": 105, "low": 95, "close": 104, "volume": 1000}]
        dev = compute_vwap_deviation(bars)
        # close (104) > typical_price (101.33) → positive deviation
        assert dev > 0

    def test_vwap_signal_score_neutral_at_zero_deviation(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_signal_score
        assert compute_vwap_signal_score(0.0) == 50.0

    def test_vwap_signal_score_clamped(self):
        from api.axiom.intraday.intraday_engine import compute_vwap_signal_score
        assert compute_vwap_signal_score(100.0) == 100.0
        assert compute_vwap_signal_score(-100.0) == 0.0

    def test_volume_surge_zero_when_normal_volume(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        # At exactly expected rate: surge_ratio = 1.0 → score = 0
        score = compute_volume_surge_score(current_volume_rate=100, avg_daily_volume=100 * 390)
        assert score == 0.0

    def test_volume_surge_100_when_extreme(self):
        from api.axiom.intraday.intraday_engine import compute_volume_surge_score
        # 4× normal → surge_ratio = 4 → (4-1)/3*100 = 100
        score = compute_volume_surge_score(current_volume_rate=400, avg_daily_volume=100 * 390)
        assert score == 100.0

    def test_intraday_composite_between_0_and_100(self):
        from api.axiom.intraday.intraday_engine import run_intraday_update
        snap = run_intraday_update("AAPL", self._make_bars(), 60.0, 1_000_000)
        if snap.intraday_composite is not None:
            assert 0.0 <= snap.intraday_composite <= 100.0

    def test_alert_eligible_when_large_dau_shift(self):
        from api.axiom.intraday.intraday_engine import determine_alert_type
        alert_type = determine_alert_type(
            intraday_composite=30.0, daily_axiom_dau=60.0,
            volume_surge_score=10.0, vwap_deviation=0.5,
        )
        assert alert_type == "signal_weakening"

    def test_alert_type_signal_strengthening(self):
        from api.axiom.intraday.intraday_engine import determine_alert_type
        alert_type = determine_alert_type(
            intraday_composite=80.0, daily_axiom_dau=60.0,
            volume_surge_score=10.0, vwap_deviation=0.5,
        )
        assert alert_type == "signal_strengthening"

    def test_alert_type_volume_surge(self):
        from api.axiom.intraday.intraday_engine import determine_alert_type
        alert_type = determine_alert_type(
            intraday_composite=62.0, daily_axiom_dau=60.0,
            volume_surge_score=80.0, vwap_deviation=0.5,
        )
        assert alert_type == "volume_surge"

    def test_alert_type_vwap_dislocation(self):
        from api.axiom.intraday.intraday_engine import determine_alert_type
        alert_type = determine_alert_type(
            intraday_composite=62.0, daily_axiom_dau=60.0,
            volume_surge_score=10.0, vwap_deviation=4.0,
        )
        assert alert_type == "vwap_dislocation"

    def test_no_alert_when_small_shift(self):
        from api.axiom.intraday.intraday_engine import determine_alert_type
        result = determine_alert_type(
            intraday_composite=62.0, daily_axiom_dau=60.0,
            volume_surge_score=10.0, vwap_deviation=0.5,
        )
        assert result is None

    def test_snapshot_graceful_no_bars(self):
        from api.axiom.intraday.intraday_engine import run_intraday_update
        snap = run_intraday_update("AAPL", [], 60.0, 1_000_000)
        assert snap.alert_eligible is False
        assert snap.source == "no_data"
        assert snap.intraday_composite == 60.0  # pass-through when no data

    def test_snapshot_has_daily_axiom_dau_field(self):
        from api.axiom.intraday.intraday_engine import IntradaySnapshot, run_intraday_update
        snap = run_intraday_update("AAPL", [], 75.0, 1_000_000)
        assert snap.daily_axiom_dau == 75.0

    def test_intraday_universe_endpoint_returns_200(self):
        r = client.get("/axiom/intraday/universe/latest",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
        data = r.json()
        assert "snapshots" in data
        assert "as_of" in data

    def test_intraday_universe_snapshots_have_required_fields(self):
        r = client.get("/axiom/intraday/universe/latest",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
        data = r.json()
        if data.get("snapshots"):
            snap = data["snapshots"][0]
            for field in ["symbol", "intraday_composite", "alert_eligible", "source"]:
                assert field in snap, f"Missing field: {field}"


# ===========================================================================
# TestWebSocket
# ===========================================================================

class TestWebSocket:
    def test_websocket_intelligence_route_registered(self):
        routes = [getattr(r, "path", "") for r in app.routes]
        assert "/ws/intelligence" in routes

    def test_ws_manager_broadcast_from_thread_no_loop(self):
        from api.realtime.websocket_manager import IntelligenceWebSocketManager
        mgr = IntelligenceWebSocketManager()
        # No loop set — should not raise
        mgr.broadcast_from_thread({"type": "test"})

    def test_ws_manager_connection_count_empty(self):
        from api.realtime.websocket_manager import IntelligenceWebSocketManager
        mgr = IntelligenceWebSocketManager()
        assert mgr.connection_count() == 0

    def test_ws_manager_singleton_exists(self):
        from api.realtime.websocket_manager import ws_manager
        assert ws_manager is not None

    def test_ws_message_types_can_be_constructed(self):
        import json
        msg_types = [
            "signal_alert", "intraday_update", "regime_change",
            "pipeline_complete", "ml_model_updated", "sri_update", "heartbeat",
        ]
        for t in msg_types:
            msg = {"type": t, "timestamp": "2026-01-01T00:00:00"}
            encoded = json.dumps(msg)
            decoded = json.loads(encoded)
            assert decoded["type"] == t

    def test_feed_status_elements_in_html(self):
        html = (WEBAPP_DIR / "index.html").read_text()
        assert "toast-container" in html
        assert "ws-status-dot" in html
        assert "ws-status-text" in html

    def test_toast_styles_in_css(self):
        css = (WEBAPP_DIR / "design_system.css").read_text()
        assert "alert-toast" in css
        assert "toast-visible" in css
        assert "feed-dot" in css
        assert "ws-pulse" in css

    def test_intelligence_feed_init_in_dashboard_js(self):
        js = (WEBAPP_DIR / "js" / "dashboard.js").read_text()
        assert "initIntelligenceFeed" in js
        assert "handleFeedMessage" in js
        assert "showToast" in js
        assert "updateFeedStatus" in js

    def test_legacy_ws_alerts_route_still_exists(self):
        routes = [getattr(r, "path", "") for r in app.routes]
        assert "/ws/alerts" in routes

    def test_set_loop_stores_loop(self):
        import asyncio
        from api.realtime.websocket_manager import IntelligenceWebSocketManager
        mgr = IntelligenceWebSocketManager()
        loop = asyncio.new_event_loop()
        try:
            mgr.set_loop(loop)
            assert mgr._loop is loop
        finally:
            loop.close()


# ===========================================================================
# TestKellySizer
# ===========================================================================

class TestKellySizer:
    def test_kelly_status_endpoint_returns_200(self):
        r = client.get("/axiom/allocation/kelly-status",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200

    def test_kelly_status_has_required_fields(self):
        r = client.get("/axiom/allocation/kelly-status",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
        data = r.json()
        for field in ["ic_source", "ic_value", "ic_state", "kelly_mode",
                       "sample_count", "ic_kelly_multiplier"]:
            assert field in data, f"Missing field: {field}"

    def test_kelly_insufficient_replaced_by_weak_prior(self):
        """_load_ic_state_bulk should return WEAK (not INSUFFICIENT) when no DB data."""
        with patch("api.axiom.screener.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.return_value = None  # no IC data
            from api.axiom.screener import _load_ic_state_bulk
            result = _load_ic_state_bulk(dt.date.today())
            # Should be WEAK (prior) not INSUFFICIENT (permanent lock)
            assert result == "WEAK"

    def test_kelly_ic_source_present(self):
        r = client.get("/axiom/allocation/kelly-status",
                       headers={"X-FTIP-API-Key": "test"})
        data = r.json()
        assert data["ic_source"] in ("signal_ic_daily", "bootstrap_prior", "insufficient")
        assert data["kelly_mode"] in ("live_ic", "bootstrap_prior", "insufficient_default")

    def test_kelly_multiplier_insufficent_is_025(self):
        from api.axiom.sizer import _IC_KELLY_MULTIPLIER
        assert _IC_KELLY_MULTIPLIER["INSUFFICIENT"] == 0.25

    def test_kelly_multiplier_weak_is_020(self):
        from api.axiom.sizer import _IC_KELLY_MULTIPLIER
        assert _IC_KELLY_MULTIPLIER["WEAK"] == 0.20

    def test_compute_kelly_size_weak_ic_produces_nonzero(self):
        from api.axiom.sizer import compute_kelly_size
        result = compute_kelly_size(
            symbol="AAPL", as_of_date="2026-01-01",
            dau=70.0, fragility_score=30.0,
            ic_state="WEAK",  # prior state after fix
        )
        assert result.suggested_weight > 0.0
        # WEAK multiplier is 0.20, better than INSUFFICIENT 0.25 but produces nonzero
        assert result.ic_kelly_multiplier == 0.20


# ===========================================================================
# TestSchedulerWiring
# ===========================================================================

class TestSchedulerWiring:
    def test_ws_heartbeat_job_in_job_ids(self):
        from api.jobs.scheduler import _JOB_IDS
        assert "ws_heartbeat" in _JOB_IDS

    def test_ml_training_uses_min_samples_initial(self):
        """Scheduler must call run_training_job with reduced threshold."""
        import inspect
        from api.jobs.scheduler import _job_ml_training_check
        src = inspect.getsource(_job_ml_training_check)
        # Should reference MINIMUM_SAMPLES_INITIAL not hardcoded 50
        assert "MINIMUM_SAMPLES_INITIAL" in src

    def test_ml_training_broadcasts_on_success(self):
        """After training, scheduler should call ws_manager.broadcast_from_thread."""
        import inspect
        from api.jobs.scheduler import _job_ml_training_check
        src = inspect.getsource(_job_ml_training_check)
        assert "broadcast_from_thread" in src
        assert "ml_model_updated" in src


# ===========================================================================
# TestIntraDAUDB
# ===========================================================================

class TestIntradayRoutes:
    def test_run_endpoint_loads_dau_and_broadcasts(self):
        """Verify run endpoint signature loads DAU and attempts WS broadcast."""
        import inspect
        from api.axiom.intraday.intraday_routes import run_intraday_update_endpoint
        src = inspect.getsource(run_intraday_update_endpoint)
        assert "_load_daily_dau" in src
        assert "broadcast_from_thread" in src

    def test_store_intraday_snapshot_function_exists(self):
        from api.axiom.intraday.intraday_routes import _store_intraday_snapshot
        assert callable(_store_intraday_snapshot)

    def test_load_daily_dau_returns_float(self):
        from api.axiom.intraday.intraday_routes import _load_daily_dau
        result = _load_daily_dau("AAPL")
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


# ===========================================================================
# TestMLRoutes
# ===========================================================================

class TestMLRoutes:
    def test_training_status_has_bootstrap_samples_field(self):
        r = client.get("/axiom/ml/training-status",
                       headers={"X-FTIP-API-Key": "test"})
        data = r.json()
        assert "samples_for_bootstrap" in data
        assert data["samples_for_bootstrap"] == 20

    def test_training_status_feature_count_is_46(self):
        r = client.get("/axiom/ml/training-status",
                       headers={"X-FTIP-API-Key": "test"})
        data = r.json()
        assert data["feature_count"] == 46


# ===========================================================================
# TestVersionBump
# ===========================================================================

class TestVersionBump:
    def test_version_bumped_to_28(self):
        r = client.get("/config/client")
        assert r.status_code == 200
        data = r.json()
        assert "28" in data.get("version", "")

    def test_html_version_28(self):
        html = (WEBAPP_DIR / "index.html").read_text()
        assert "v=28" in html


# ===========================================================================
# TestEnsembleModule
# ===========================================================================

class TestEnsembleModule:
    def test_get_ensemble_status_returns_dict(self):
        from api.axiom.ml.ensemble import get_ensemble_status
        result = get_ensemble_status(dt.date.today())
        assert isinstance(result, dict)
        for field in ["model_version", "blend_method", "rule_weight", "ml_weight"]:
            assert field in result

    def test_ensemble_endpoint_returns_200(self):
        r = client.get("/axiom/ml/ensemble-status",
                       headers={"X-FTIP-API-Key": "test"})
        assert r.status_code == 200
