"""Session 13: Regime-gated alert engine tests."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import security
from api.jobs import alerts as alerts_module
from api.jobs.alerts import compute_conviction_score, run_alert_scan
from api.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in ["FTIP_API_KEY", "FTIP_DB_ENABLED", "FTIP_DB_READ_ENABLED", "FTIP_DB_WRITE_ENABLED"]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()
    yield
    for key in ["FTIP_API_KEY", "FTIP_DB_ENABLED", "FTIP_DB_READ_ENABLED", "FTIP_DB_WRITE_ENABLED"]:
        monkeypatch.delenv(key, raising=False)
    security.reset_auth_cache()


def _db_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


AUTH   = {"X-FTIP-API-Key": "secret"}
TODAY  = dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# compute_conviction_score: pure logic
# ---------------------------------------------------------------------------

class TestConvictionScore:

    def _base(self, **overrides):
        defaults = dict(
            dau=75.0, signal_label="BUY",
            regime_label="fundamental_convergence",
            breadth_state="EXPANDING",
            ic_state="MODERATE",
            min_dau=65.0,
            favorable_regimes=None,
            require_breadth_alignment=True,
        )
        defaults.update(overrides)
        return compute_conviction_score(**defaults)

    def test_all_aligned_gives_positive_score(self):
        score = self._base()
        assert score > 0

    def test_dau_below_threshold_returns_zero(self):
        assert self._base(dau=60.0, min_dau=65.0) == 0.0

    def test_ic_degraded_returns_zero(self):
        assert self._base(ic_state="DEGRADED") == 0.0

    def test_veto_regime_buy_returns_zero(self):
        assert self._base(signal_label="BUY", regime_label="euphoria_critical") == 0.0
        assert self._base(signal_label="BUY", regime_label="liquidity_fracture") == 0.0

    def test_veto_regime_sell_returns_zero(self):
        assert self._base(
            signal_label="SELL", regime_label="fundamental_convergence",
            breadth_state="CONTRACTING",
        ) == 0.0

    def test_breadth_misaligned_buy_returns_zero(self):
        # BUY but breadth is CONTRACTING
        score = self._base(signal_label="BUY", breadth_state="CONTRACTING",
                           require_breadth_alignment=True)
        assert score == 0.0

    def test_breadth_misaligned_ok_when_not_required(self):
        score = self._base(signal_label="BUY", breadth_state="CONTRACTING",
                           require_breadth_alignment=False)
        assert score > 0.0

    def test_breadth_aligned_sell_gives_score(self):
        score = self._base(
            signal_label="SELL",
            regime_label="liquidity_fracture",
            breadth_state="CONTRACTING",
            favorable_regimes=frozenset({"liquidity_fracture"}),
        )
        assert score > 0.0

    def test_higher_dau_gives_higher_score(self):
        low  = self._base(dau=70.0)
        high = self._base(dau=95.0)
        assert high > low

    def test_strong_ic_gives_higher_score_than_weak(self):
        strong = self._base(ic_state="STRONG")
        weak   = self._base(ic_state="WEAK")
        assert strong >= weak

    def test_favorable_regime_beats_neutral(self):
        favorable = self._base(regime_label="fundamental_convergence")
        neutral   = self._base(regime_label="some_other_regime")
        assert favorable > neutral

    def test_score_capped_at_100(self):
        score = self._base(dau=100.0, ic_state="STRONG",
                           regime_label="fundamental_convergence",
                           breadth_state="EXPANDING")
        assert score <= 100.0

    def test_score_is_non_negative(self):
        for regime in ["fundamental_convergence", "indeterminate", "recovery_reset"]:
            score = self._base(regime_label=regime)
            assert score >= 0.0

    def test_hold_signal_breadth_always_aligned(self):
        # HOLD signal should not require breadth alignment
        score = self._base(signal_label="HOLD", breadth_state="CONTRACTING",
                           require_breadth_alignment=True)
        assert score > 0.0

    def test_custom_favorable_regimes(self):
        custom = frozenset({"my_custom_regime"})
        score_with = self._base(
            regime_label="my_custom_regime", favorable_regimes=custom
        )
        score_without = self._base(regime_label="my_custom_regime")
        assert score_with >= score_without


# ---------------------------------------------------------------------------
# deliver_webhook
# ---------------------------------------------------------------------------

def test_deliver_webhook_returns_zero_on_connection_error(monkeypatch):
    monkeypatch.setattr("api.jobs.alerts.httpx.post",
                        lambda *a, **kw: (_ for _ in ()).throw(Exception("conn refused")))
    result = alerts_module.deliver_webhook("http://fake", {"x": 1})
    assert result == 0


def test_deliver_webhook_returns_status_code(monkeypatch):
    class FakeResp:
        status_code = 200
    monkeypatch.setattr("api.jobs.alerts.httpx.post", lambda *a, **kw: FakeResp())
    result = alerts_module.deliver_webhook("http://fake", {"x": 1})
    assert result == 200


# ---------------------------------------------------------------------------
# run_alert_scan
# ---------------------------------------------------------------------------

def _make_rule(symbol="AAPL", min_dau=65.0, signal_filter=None,
               require_breadth=True, min_conv=35.0, webhook_url=None):
    return {
        "rule_id": f"rule-{symbol}",
        "symbol": symbol,
        "min_dau": min_dau,
        "signal_filter": signal_filter or [],
        "favorable_regimes": None,
        "require_breadth_alignment": require_breadth,
        "min_conviction_score": min_conv,
        "webhook_url": webhook_url,
        "meta": {},
    }


def _patch_scan(monkeypatch, rules, scores, breadth="EXPANDING", ic="MODERATE",
                already_fired=False):
    monkeypatch.setattr("api.jobs.alerts._load_active_rules", lambda: rules)
    monkeypatch.setattr("api.jobs.alerts._load_axiom_scores_batch",
                        lambda symbols, date: scores)
    monkeypatch.setattr("api.jobs.alerts._load_breadth_state", lambda d: breadth)
    monkeypatch.setattr("api.jobs.alerts._load_ic_state", lambda d: ic)
    monkeypatch.setattr("api.jobs.alerts._already_fired", lambda r, d: already_fired)
    monkeypatch.setattr("api.jobs.alerts._store_event", lambda e: None)


def test_scan_empty_rules_returns_zero_fired(monkeypatch):
    _patch_scan(monkeypatch, rules=[], scores={})
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0
    assert result.rules_evaluated == 0


def test_scan_fires_when_all_conditions_met(monkeypatch):
    rules = [_make_rule("AAPL")]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 1
    assert result.suppressed == 0


def test_scan_suppresses_when_dau_too_low(monkeypatch):
    rules = [_make_rule("AAPL", min_dau=80.0)]
    scores = {"AAPL": {
        "dau": 60.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0
    assert result.suppressed == 1


def test_scan_suppresses_not_actionable_tier(monkeypatch):
    rules = [_make_rule("AAPL")]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "not_actionable",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0
    assert result.suppressed == 1


def test_scan_skips_already_fired(monkeypatch):
    rules = [_make_rule("AAPL")]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores, already_fired=True)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0
    assert result.already_fired_today == 1


def test_scan_applies_signal_filter(monkeypatch):
    rules = [_make_rule("AAPL", signal_filter=["SELL"])]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0
    assert result.suppressed == 1


def test_scan_veto_regime_suppresses(monkeypatch):
    rules = [_make_rule("AAPL")]
    scores = {"AAPL": {
        "dau": 85.0, "regime_label": "euphoria_critical",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0


def test_scan_degraded_ic_suppresses(monkeypatch):
    rules = [_make_rule("AAPL")]
    scores = {"AAPL": {
        "dau": 85.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores, ic="DEGRADED")
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 0


def test_scan_webhook_delivery_tracked(monkeypatch):
    rules = [_make_rule("AAPL", webhook_url="http://fake/hook")]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)

    class FakeResp:
        status_code = 200
    monkeypatch.setattr("api.jobs.alerts.httpx.post", lambda *a, **kw: FakeResp())

    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 1
    assert result.webhook_delivered == 1


def test_scan_webhook_failure_tracked(monkeypatch):
    rules = [_make_rule("AAPL", webhook_url="http://fake/hook")]
    scores = {"AAPL": {
        "dau": 80.0, "regime_label": "fundamental_convergence",
        "deployability_tier": "live_candidate",
        "signal_label": "BUY", "confidence": 70.0,
    }}
    _patch_scan(monkeypatch, rules, scores)
    monkeypatch.setattr("api.jobs.alerts.httpx.post",
                        lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))

    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 1           # event stored even if webhook fails
    assert result.webhook_failed == 1


def test_scan_events_list_populated(monkeypatch):
    rules = [_make_rule("AAPL"), _make_rule("MSFT")]
    scores = {
        "AAPL": {"dau": 80.0, "regime_label": "fundamental_convergence",
                 "deployability_tier": "live_candidate", "signal_label": "BUY", "confidence": 70.0},
        "MSFT": {"dau": 85.0, "regime_label": "compensation_capture",
                 "deployability_tier": "live_candidate", "signal_label": "BUY", "confidence": 75.0},
    }
    _patch_scan(monkeypatch, rules, scores)
    result = run_alert_scan(dt.date(2024, 1, 10))
    assert result.fired == 2
    assert len(result.events) == 2


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class TestAlertEndpoints:

    def test_daily_scan_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        client = TestClient(app)
        resp = client.post("/jobs/alerts/daily-scan", json={})
        assert resp.status_code == 401

    def test_daily_scan_returns_ok(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.jobs.alerts._load_active_rules", lambda: [])
        client = TestClient(app)
        resp = client.post("/jobs/alerts/daily-scan",
                           json={"as_of_date": TODAY}, headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["fired"] == 0

    def test_rules_endpoint_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        client = TestClient(app)
        resp = client.post("/jobs/alerts/rules",
                           json={"symbol": "AAPL"})
        assert resp.status_code == 401

    def test_rules_endpoint_write_disabled(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.jobs.alerts.db.db_write_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post("/jobs/alerts/rules",
                           json={"symbol": "AAPL"}, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["status"] == "db_write_disabled"

    def test_rules_endpoint_creates_rule(self, monkeypatch):
        _db_env(monkeypatch)
        calls = []
        monkeypatch.setattr("api.jobs.alerts.db.db_write_enabled", lambda: True)
        monkeypatch.setattr("api.jobs.alerts.db.safe_execute",
                            lambda sql, params: calls.append(params))
        client = TestClient(app)
        resp = client.post(
            "/jobs/alerts/rules",
            json={"symbol": "AAPL", "min_dau": 70.0, "webhook_url": "http://x/hook"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["symbol"] == "AAPL"
        assert "rule_id" in body
        assert calls  # DB was called

    def test_recent_alerts_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        client = TestClient(app)
        resp = client.get("/jobs/alerts/recent")
        assert resp.status_code == 401

    def test_recent_alerts_db_disabled(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.jobs.alerts.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.get("/jobs/alerts/recent", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["status"] == "db_read_disabled"

    def test_recent_alerts_returns_events(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.jobs.alerts.db.db_read_enabled", lambda: True)
        rows = [
            ("evt-1", "rule-AAPL", "AAPL", dt.date(2024, 1, 10), "BUY",
             80.0, "fundamental_convergence", "EXPANDING", "MODERATE", 72.5,
             True, 200),
        ]
        monkeypatch.setattr("api.jobs.alerts.db.safe_fetchall",
                            lambda sql, params: rows)
        client = TestClient(app)
        resp = client.get("/jobs/alerts/recent?symbol=AAPL", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["events"][0]["symbol"] == "AAPL"

    def test_routes_in_openapi(self):
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/jobs/alerts/daily-scan" in paths
        assert "/jobs/alerts/rules" in paths
        assert "/jobs/alerts/recent" in paths
