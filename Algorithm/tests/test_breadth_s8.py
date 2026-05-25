"""Session 8: market_breadth_daily job and depth-layer injection."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import security
from api.jobs import breadth as breadth_module
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


# ---------------------------------------------------------------------------
# compute_market_breadth: pure computation
# ---------------------------------------------------------------------------

def _fake_fetchall(signals: List[tuple]):
    def _fetchall(sql, params):
        return signals
    return _fetchall


def test_compute_empty_returns_empty(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall([]))
    result = breadth_module.compute_market_breadth(dt.date(2024, 1, 10))
    assert result == {}


def test_compute_all_buy_gives_high_confirmation(monkeypatch):
    rows = [(0.8, "BUY")] * 8 + [(0.7, "BUY")] * 2
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    result = breadth_module.compute_market_breadth(dt.date(2024, 1, 10))
    assert result["breadth_confirmation_score"] == pytest.approx(100.0)
    assert result["participation_breadth_score"] == pytest.approx(100.0)
    assert result["breadth_state"] == "EXPANDING"
    assert result["broad_participation_confirmation"] is True
    assert result["universe_size"] == 10


def test_compute_all_sell_gives_contracting(monkeypatch):
    rows = [(-0.6, "SELL")] * 8 + [(-0.3, "SELL")] * 2
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    result = breadth_module.compute_market_breadth(dt.date(2024, 1, 10))
    assert result["breadth_confirmation_score"] == pytest.approx(0.0)
    assert result["breadth_state"] == "CONTRACTING"
    assert result["broad_participation_confirmation"] is False


def test_compute_mixed_universe(monkeypatch):
    rows = [(0.6, "BUY")] * 5 + [(-0.4, "SELL")] * 3 + [(0.05, "HOLD")] * 2
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    result = breadth_module.compute_market_breadth(dt.date(2024, 1, 10))
    assert 0 < result["breadth_confirmation_score"] < 100
    assert result["universe_size"] == 10
    assert result["breadth_thrust_proxy"] is not None


def test_compute_narrow_leadership_warning(monkeypatch):
    # A few high scorers, many low scorers → narrow leadership
    rows = [(0.95, "BUY")] * 2 + [(-0.5, "SELL")] * 8
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    result = breadth_module.compute_market_breadth(dt.date(2024, 1, 10))
    assert result["participation_breadth_score"] == pytest.approx(20.0)
    assert result["narrow_leadership_warning"] is True


# ---------------------------------------------------------------------------
# store_market_breadth
# ---------------------------------------------------------------------------

def test_store_market_breadth_calls_safe_execute(monkeypatch):
    calls = []
    monkeypatch.setattr("api.jobs.breadth.db.safe_execute", lambda sql, params: calls.append(params))
    payload = {"breadth_confirmation_score": 60.0, "breadth_state": "NEUTRAL", "universe_size": 10}
    result = breadth_module.store_market_breadth(dt.date(2024, 1, 10), payload)
    assert result is True
    assert calls


def test_store_market_breadth_empty_payload_returns_false():
    result = breadth_module.store_market_breadth(dt.date(2024, 1, 10), {})
    assert result is False


# ---------------------------------------------------------------------------
# load_market_breadth
# ---------------------------------------------------------------------------

def test_load_market_breadth_returns_none_when_db_disabled(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: False)
    assert breadth_module.load_market_breadth(dt.date(2024, 1, 10)) is None


def test_load_market_breadth_returns_dict_from_row(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: True)
    row = (60.0, 55.0, 52.0, 20.0, 55.0, 30.0, 70.0, 40.0, False, True, "NEUTRAL", 12)
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchone", lambda sql, params: row)
    result = breadth_module.load_market_breadth(dt.date(2024, 1, 10))
    assert result is not None
    assert result["breadth_confirmation_score"] == 60.0
    assert result["breadth_state"] == "NEUTRAL"
    assert result["universe_size"] == 12


def test_load_market_breadth_returns_none_on_no_row(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchone", lambda sql, params: None)
    assert breadth_module.load_market_breadth(dt.date(2024, 1, 10)) is None


# ---------------------------------------------------------------------------
# /jobs/breadth/daily-snapshot endpoint
# ---------------------------------------------------------------------------

def test_endpoint_requires_auth(monkeypatch):
    _db_env(monkeypatch)
    client = TestClient(app)
    resp = client.post("/jobs/breadth/daily-snapshot", json={})
    assert resp.status_code == 401


def test_endpoint_returns_no_data_when_no_signals(monkeypatch):
    _db_env(monkeypatch)
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall([]))
    client = TestClient(app)
    resp = client.post(
        "/jobs/breadth/daily-snapshot",
        json={"as_of_date": "2024-01-10"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_data"


def test_endpoint_returns_ok_with_signals(monkeypatch):
    _db_env(monkeypatch)
    rows = [(0.7, "BUY")] * 6 + [(0.1, "HOLD")] * 4
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    monkeypatch.setattr("api.jobs.breadth.db.safe_execute", lambda sql, params: None)
    client = TestClient(app)
    resp = client.post(
        "/jobs/breadth/daily-snapshot",
        json={"as_of_date": "2024-01-10", "lookback": 252},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["universe_size"] == 10
    assert body["stored"] is True


def test_endpoint_skips_store_when_write_disabled(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.delenv("FTIP_DB_WRITE_ENABLED", raising=False)
    security.reset_auth_cache()

    rows = [(0.7, "BUY")] * 6 + [(0.1, "HOLD")] * 4
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", _fake_fetchall(rows))
    client = TestClient(app)
    resp = client.post(
        "/jobs/breadth/daily-snapshot",
        json={"as_of_date": "2024-01-10"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["stored"] is False


# ---------------------------------------------------------------------------
# Depth-layer injection: _overlay_stored_breadth
# ---------------------------------------------------------------------------

def test_overlay_fills_none_fields(monkeypatch):
    stored = {
        "breadth_confirmation_score": 65.0,
        "participation_breadth_score": 58.0,
        "breadth_state": "EXPANDING",
    }
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchone", lambda sql, params: (
        65.0, 58.0, None, None, None, None, None, None, None, None, "EXPANDING", 10
    ))

    from api.assistant.intelligence import _overlay_stored_breadth
    domain: Dict[str, Any] = {
        "breadth_confirmation_score": None,
        "participation_breadth_score": None,
        "breadth_state": None,
    }
    _overlay_stored_breadth(domain, "2024-01-10")
    assert domain["breadth_confirmation_score"] == 65.0
    assert domain["breadth_state"] == "EXPANDING"


def test_overlay_does_not_replace_existing_values(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: True)
    monkeypatch.setattr("api.jobs.breadth.db.safe_fetchone", lambda sql, params: (
        99.0, 99.0, None, None, None, None, None, None, None, None, "EXPANDING", 10
    ))

    from api.assistant.intelligence import _overlay_stored_breadth
    domain: Dict[str, Any] = {
        "breadth_confirmation_score": 42.0,
        "participation_breadth_score": 38.0,
        "breadth_state": "CONTRACTING",
    }
    _overlay_stored_breadth(domain, "2024-01-10")
    assert domain["breadth_confirmation_score"] == 42.0
    assert domain["breadth_state"] == "CONTRACTING"


def test_overlay_silent_when_no_date():
    from api.assistant.intelligence import _overlay_stored_breadth
    domain: Dict[str, Any] = {"breadth_confirmation_score": None}
    _overlay_stored_breadth(domain, None)
    assert domain["breadth_confirmation_score"] is None


def test_overlay_silent_when_db_disabled(monkeypatch):
    monkeypatch.setattr("api.jobs.breadth.db.db_read_enabled", lambda: False)
    from api.assistant.intelligence import _overlay_stored_breadth
    domain: Dict[str, Any] = {"breadth_confirmation_score": None}
    _overlay_stored_breadth(domain, "2024-01-10")
    assert domain["breadth_confirmation_score"] is None
