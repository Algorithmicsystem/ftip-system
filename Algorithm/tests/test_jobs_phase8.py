import datetime as dt
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from api import security
from api.main import app
from api.jobs import prosperity
from api.prosperity import ingest, query, routes
from api.prosperity.models import SnapshotRunRequest


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


def test_coverage_endpoints_require_auth(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    def fake_coverage_response(*, as_of_date=None, run_id=None):
        payload: Dict[str, Any] = {
            "attempted": 1,
            "ok": 0,
            "failed": 1,
            "skipped": 0,
            "by_reason_code": {"API_ERROR": 1},
            "failed_symbols": [
                {
                    "symbol": "AAPL",
                    "reason": "API error",
                    "reason_code": "API_ERROR",
                    "reason_detail": "API error",
                }
            ],
        }
        if as_of_date:
            payload["as_of_date"] = as_of_date.isoformat()
        if run_id:
            payload["run_id"] = run_id
        return payload

    monkeypatch.setattr("api.jobs.prosperity._coverage_response", fake_coverage_response)
    monkeypatch.setattr("api.jobs.prosperity._require_db_enabled", lambda *args, **kwargs: None)

    client = TestClient(app)

    unauthorized = client.get("/jobs/prosperity/daily-snapshot/coverage", params={"as_of_date": "2024-01-01"})
    assert unauthorized.status_code == 401

    resp = client.get(
        "/jobs/prosperity/daily-snapshot/coverage",
        params={"as_of_date": "2024-01-01"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["attempted"] == 1
    assert payload["failed_symbols"][0]["reason_code"] == "API_ERROR"

    run_resp = client.get(
        "/jobs/prosperity/daily-snapshot/runs/test-run/coverage",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert run_resp.status_code == 200
    assert run_resp.json().get("run_id") == "test-run"


@pytest.mark.anyio
async def test_snapshot_run_reports_reason_code(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setattr(routes, "_require_db_enabled", lambda *args, **kwargs: None)
    monkeypatch.setattr(routes.metrics_tracker, "record_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "upsert_universe", lambda symbols: symbols)
    monkeypatch.setattr(ingest, "ingest_bars", lambda *args, **kwargs: None)

    def raise_failure(*_args, **_kwargs):
        raise routes.SymbolFailure("API_ERROR", "")

    monkeypatch.setattr(query, "fetch_bars", raise_failure)
    monkeypatch.setattr(routes, "_log_symbol_coverage", lambda *args, **kwargs: None)

    req = SnapshotRunRequest(
        symbols=["AAPL"],
        from_date=dt.date(2024, 1, 1),
        to_date=dt.date(2024, 1, 2),
        as_of_date=dt.date(2024, 1, 3),
        lookback=10,
        concurrency=1,
        force_refresh=False,
        compute_strategy_graph=False,
    )

    result = await routes.snapshot_run(req, Request(scope={"type": "http"}), run_id="run-1", job_name="job-1")
    failures: List[Dict[str, Any]] = result.get("result", {}).get("symbols_failed", [])

    assert failures
    assert failures[0]["reason_code"] == "API_ERROR"
    assert failures[0]["reason"] == "API_ERROR"
    assert failures[0]["reason_detail"] == "API_ERROR"


def test_daily_snapshot_lock_conflict(monkeypatch: pytest.MonkeyPatch):
    _set_db_flags(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    call_count = {"count": 0}

    def fake_acquire(run_id: str, job_name: str, as_of_date, requested, ttl_seconds: int, lock_owner: str):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return True, {"run_id": run_id, "started_at": "now", "lock_owner": lock_owner}
        return False, {"run_id": "existing", "started_at": "later", "lock_owner": lock_owner}

    async def fake_snapshot(req, request, **_kwargs):
        return {
            "status": "ok",
            "result": {
                "symbols_ok": ["AAPL"],
                "symbols_failed": [],
                "rows_written": {"signals": 1, "features": 1},
            },
            "timings": {"total": 0.1},
        }

    monkeypatch.setattr("api.jobs.prosperity._acquire_job_lock", fake_acquire)
    monkeypatch.setattr("api.jobs.prosperity._update_job_run", lambda *args, **kwargs: None)
    monkeypatch.setattr("api.jobs.prosperity._utc_today", lambda: dt.date(2024, 1, 5))
    monkeypatch.setattr("api.jobs.prosperity.snapshot_run", fake_snapshot)

    client = TestClient(app)
    first = client.post("/jobs/prosperity/daily-snapshot", headers={"X-FTIP-API-Key": "secret"})
    second = client.post("/jobs/prosperity/daily-snapshot", headers={"X-FTIP-API-Key": "secret"})

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json().get("error") == "locked"
