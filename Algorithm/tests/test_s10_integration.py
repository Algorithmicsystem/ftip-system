"""Session 10: route contract tests for endpoints added in sessions 4–8.

These tests verify route registration, auth enforcement, and response shape.
They never touch a real DB or external APIs.
"""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import security
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


@pytest.fixture()
def auth_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "testkey")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


AUTH_HEADER = {"X-FTIP-API-Key": "testkey"}

TODAY = dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_401(resp):
    assert resp.status_code == 401, f"expected 401, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# /jobs/breadth/daily-snapshot  (Session 8)
# ---------------------------------------------------------------------------

class TestBreadthDailySnapshotAuth:
    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.post("/jobs/breadth/daily-snapshot", json={})
        _assert_401(resp)

    def test_wrong_key_returns_401(self, client, auth_env):
        resp = client.post(
            "/jobs/breadth/daily-snapshot",
            json={"as_of_date": TODAY},
            headers={"X-FTIP-API-Key": "wrong"},
        )
        _assert_401(resp)

    def test_no_signals_returns_no_data(self, monkeypatch, client, auth_env):
        monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", lambda sql, params: [])
        resp = client.post(
            "/jobs/breadth/daily-snapshot",
            json={"as_of_date": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"

    def test_with_signals_returns_ok_and_result(self, monkeypatch, client, auth_env):
        rows = [(0.7, "BUY")] * 6 + [(0.1, "HOLD")] * 4
        monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", lambda sql, params: rows)
        monkeypatch.setattr("api.jobs.breadth.db.safe_execute", lambda sql, params: None)
        resp = client.post(
            "/jobs/breadth/daily-snapshot",
            json={"as_of_date": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "result" in body
        assert body["result"]["universe_size"] == 10

    def test_response_has_stored_field(self, monkeypatch, client, auth_env):
        rows = [(0.7, "BUY")] * 5 + [(-0.3, "SELL")] * 5
        monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", lambda sql, params: rows)
        monkeypatch.setattr("api.jobs.breadth.db.safe_execute", lambda sql, params: None)
        resp = client.post(
            "/jobs/breadth/daily-snapshot",
            json={"as_of_date": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert "stored" in resp.json()

    def test_stored_false_when_write_disabled(self, monkeypatch, client):
        monkeypatch.setenv("FTIP_API_KEY", "testkey")
        monkeypatch.setenv("FTIP_DB_ENABLED", "1")
        monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
        monkeypatch.delenv("FTIP_DB_WRITE_ENABLED", raising=False)
        security.reset_auth_cache()

        rows = [(0.7, "BUY")] * 6 + [(0.1, "HOLD")] * 4
        monkeypatch.setattr("api.jobs.breadth.db.safe_fetchall", lambda sql, params: rows)
        resp = client.post(
            "/jobs/breadth/daily-snapshot",
            json={"as_of_date": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert resp.json()["stored"] is False


# ---------------------------------------------------------------------------
# /jobs/prosperity/nightly-recalibrate  (Session 4/5)
# ---------------------------------------------------------------------------

class TestProsperityNightlyRecalibrateAuth:
    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.post("/jobs/prosperity/nightly-recalibrate", json={})
        _assert_401(resp)

    def test_route_registered(self, client, auth_env):
        resp = client.post(
            "/jobs/prosperity/nightly-recalibrate",
            json={},
            headers=AUTH_HEADER,
        )
        assert resp.status_code != 404, "route not registered"


class TestProsperityDailySnapshotStatus:
    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.get("/jobs/prosperity/daily-snapshot/status")
        _assert_401(resp)

    def test_route_registered(self, client, auth_env):
        resp = client.get(
            "/jobs/prosperity/daily-snapshot/status",
            headers=AUTH_HEADER,
        )
        assert resp.status_code != 404


# ---------------------------------------------------------------------------
# /narrator/signal  (Session 4/5)
# ---------------------------------------------------------------------------

class TestNarratorSignalAuth:
    _VALID_BODY = {"symbol": "AAPL", "as_of": TODAY}

    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.post("/narrator/signal", json=self._VALID_BODY)
        _assert_401(resp)

    def test_missing_symbol_returns_422(self, monkeypatch, client, auth_env):
        resp = client.post(
            "/narrator/signal",
            json={"as_of": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422

    def test_missing_as_of_returns_422(self, monkeypatch, client, auth_env):
        resp = client.post(
            "/narrator/signal",
            json={"symbol": "AAPL"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422

    def test_route_registered(self, monkeypatch, client, auth_env):
        # Patch the underlying narrator call so we don't need an LLM key
        monkeypatch.setattr(
            "api.narrator.routes.narrator_client.complete_chat",
            lambda *a, **kw: "Test narrative.",
        )
        resp = client.post(
            "/narrator/signal",
            json=self._VALID_BODY,
            headers=AUTH_HEADER,
        )
        # 200 or 500 (if further deps missing) — but not 404 or 401
        assert resp.status_code not in (404, 401)


# ---------------------------------------------------------------------------
# /narrator/portfolio  (Session 4/5)
# ---------------------------------------------------------------------------

class TestNarratorPortfolioAuth:
    _VALID_BODY = {
        "symbols": ["AAPL", "MSFT"],
        "from_date": "2024-01-01",
        "to_date": TODAY,
    }

    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.post("/narrator/portfolio", json=self._VALID_BODY)
        _assert_401(resp)

    def test_missing_symbols_returns_422(self, client, auth_env):
        resp = client.post(
            "/narrator/portfolio",
            json={"from_date": "2024-01-01", "to_date": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422

    def test_route_registered(self, monkeypatch, client, auth_env):
        monkeypatch.setattr(
            "api.narrator.routes.narrator_client.complete_chat",
            lambda *a, **kw: "Portfolio narrative.",
        )
        resp = client.post(
            "/narrator/portfolio",
            json=self._VALID_BODY,
            headers=AUTH_HEADER,
        )
        assert resp.status_code not in (404, 401)


# ---------------------------------------------------------------------------
# /narrator/ask/legacy  (Session 4/5)
# ---------------------------------------------------------------------------

class TestNarratorAskLegacyAuth:
    _VALID_BODY = {
        "question": "What is the trend?",
        "symbols": ["AAPL"],
        "as_of": TODAY,
    }

    def test_no_auth_returns_401(self, client, auth_env):
        resp = client.post("/narrator/ask/legacy", json=self._VALID_BODY)
        _assert_401(resp)

    def test_missing_question_returns_422(self, client, auth_env):
        resp = client.post(
            "/narrator/ask/legacy",
            json={"symbols": ["AAPL"], "as_of": TODAY},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 422

    def test_route_registered(self, monkeypatch, client, auth_env):
        monkeypatch.setattr(
            "api.narrator.routes.narrator_client.complete_chat",
            lambda *a, **kw: "Legacy answer.",
        )
        resp = client.post(
            "/narrator/ask/legacy",
            json=self._VALID_BODY,
            headers=AUTH_HEADER,
        )
        assert resp.status_code not in (404, 401)


# ---------------------------------------------------------------------------
# Cross-session smoke: all new routes are registered in the OpenAPI schema
# ---------------------------------------------------------------------------

class TestRouteRegistration:
    EXPECTED_ROUTES = [
        ("POST", "/jobs/breadth/daily-snapshot"),
        ("POST", "/jobs/prosperity/nightly-recalibrate"),
        ("GET", "/jobs/prosperity/daily-snapshot/status"),
        ("POST", "/narrator/signal"),
        ("POST", "/narrator/portfolio"),
        ("POST", "/narrator/ask/legacy"),
    ]

    def test_all_routes_present_in_openapi(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        for method, path in self.EXPECTED_ROUTES:
            assert path in paths, f"{path} not in OpenAPI paths"
            assert method.lower() in paths[path], (
                f"{method} not registered on {path}"
            )
