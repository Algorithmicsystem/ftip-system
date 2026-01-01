import datetime as dt
from typing import Any, Dict, List

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
        "FTIP_JOB_LOCK_TTL_SECONDS",
        "FTIP_JOB_LOCK_WINDOW_SEC",
    ]:
        monkeypatch.delenv(key, raising=False)
    security._API_KEYS = None
    yield


def test_daily_snapshot_lock_conflict(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    lock_calls: List[Dict[str, Any]] = []

    def fake_acquire(run_id: str, job_name: str, as_of_date, requested, ttl_seconds: int, lock_owner: str):
        if not lock_calls:
            lock_calls.append(
                {
                    "job_name": job_name,
                    "ttl": ttl_seconds,
                    "owner": lock_owner,
                    "run_id": run_id,
                    "as_of_date": as_of_date,
                }
            )
            return True, {"started_at": "now", "lock_owner": lock_owner, "run_id": run_id}
        return False, {"started_at": "later", "lock_owner": lock_owner, "run_id": run_id}

    monkeypatch.setattr("api.jobs.prosperity._acquire_job_lock", fake_acquire)
    monkeypatch.setattr("api.jobs.prosperity._update_job_run", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.jobs.prosperity._utc_today", lambda: dt.date(2024, 2, 10))

    async def fake_snapshot(req, request):
        return {
            "status": "ok",
            "result": {
                "symbols_ok": ["AAPL"],
                "symbols_failed": [],
                "rows_written": {"signals": 1, "features": 1},
            },
            "timings": {"total": 0.5},
        }

    monkeypatch.setattr("api.jobs.prosperity.snapshot_run", fake_snapshot)

    client = TestClient(app)
    first = client.post(
        "/jobs/prosperity/daily-snapshot",
        headers={"X-FTIP-API-Key": "secret"},
    )
    second = client.post(
        "/jobs/prosperity/daily-snapshot",
        headers={"X-FTIP-API-Key": "secret"},
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json().get("error") == "locked"
    assert lock_calls and lock_calls[0]["ttl"] == 1200


def test_daily_snapshot_status_endpoint(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    last_run = {
        "run_id": "1234",
        "job_name": "prosperity_daily_snapshot",
        "started_at": "2024-01-01T00:00:00+00:00",
        "finished_at": "2024-01-01T00:05:00+00:00",
        "status": "success",
        "requested": {"symbols": ["AAPL"]},
        "result": {"symbols_ok": ["AAPL"], "rows_written": {"signals": 1}},
        "error": None,
    }

    monkeypatch.setattr("api.jobs.prosperity._fetch_last_job_run", lambda job: last_run)

    client = TestClient(app)
    resp = client.get(
        "/jobs/prosperity/daily-snapshot/status",
        headers={"X-FTIP-API-Key": "secret"},
    )

    assert resp.status_code == 200
    assert resp.json()["run_id"] == "1234"
