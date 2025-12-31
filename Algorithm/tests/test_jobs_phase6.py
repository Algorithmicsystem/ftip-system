import datetime as dt
import os
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import security


def _set_db_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch):
    for key in [
        "FTIP_API_KEY",
        "FTIP_API_KEYS",
        "FTIP_RETENTION_DAYS",
        "FTIP_UNIVERSE",
        "FTIP_LOOKBACK",
        "FTIP_SNAPSHOT_WINDOW_DAYS",
        "FTIP_SNAPSHOT_CONCURRENCY",
        "FTIP_DB_ENABLED",
        "FTIP_DB_WRITE_ENABLED",
        "FTIP_DB_READ_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)
    security._API_KEYS = None
    yield


def test_daily_snapshot_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.post("/jobs/prosperity/daily-snapshot")

    assert resp.status_code == 401


def test_daily_snapshot_runs_with_defaults(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    called: Dict[str, Any] = {}
    fixed_today = dt.date(2024, 1, 10)

    async def fake_snapshot(req, request):
        called["req"] = req
        return {
            "status": "ok",
            "result": {
                "symbols_ok": ["AAPL"],
                "symbols_failed": [],
                "rows_written": {"signals": 1, "features": 1},
            },
            "timings": {"total": 1.0},
        }

    monkeypatch.setattr("api.jobs.prosperity._utc_today", lambda: fixed_today)
    monkeypatch.setattr("api.jobs.prosperity.snapshot_run", fake_snapshot)

    client = TestClient(app)
    resp = client.post(
        "/jobs/prosperity/daily-snapshot",
        headers={"X-FTIP-API-Key": "secret"},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert called["req"].as_of_date == dt.date(2024, 1, 9)
    assert called["req"].from_date == dt.date(2023, 1, 9)
    assert called["req"].to_date == dt.date(2024, 1, 9)
    assert called["req"].lookback == 252
    assert called["req"].concurrency == 3
    assert called["req"].compute_strategy_graph is True

    assert body["symbols_ok"] == ["AAPL"]
    assert body["rows_written"]["signals"] == 1
    assert body["timings"]["total"] == 1.0


def test_daily_snapshot_invokes_retention(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_RETENTION_DAYS", "730")

    called: Dict[str, Any] = {}
    async def fake_snapshot(req, request):
        return {
            "status": "ok",
            "result": {
                "symbols_ok": ["AAPL"],
                "symbols_failed": [],
                "rows_written": {"signals": 1, "features": 1},
            },
            "timings": {"total": 1.0},
        }

    def fake_cleanup(as_of_date, retention_days):
        called["args"] = (as_of_date, retention_days)
        return {"prosperity_features_daily": 5}

    monkeypatch.setattr("api.jobs.prosperity._utc_today", lambda: dt.date(2024, 5, 20))
    monkeypatch.setattr("api.jobs.prosperity.snapshot_run", fake_snapshot)
    monkeypatch.setattr("api.jobs.prosperity.cleanup_retention", fake_cleanup)

    client = TestClient(app)
    resp = client.post(
        "/jobs/prosperity/daily-snapshot",
        headers={"X-FTIP-API-Key": "secret"},
    )

    assert resp.status_code == 200
    assert called["args"] == (dt.date(2024, 5, 19), 730)
    assert resp.json()["retention_deleted"] == {"prosperity_features_daily": 5}
