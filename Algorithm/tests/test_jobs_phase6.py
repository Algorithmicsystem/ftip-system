import datetime as dt
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import security
from api.jobs import prosperity


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


@pytest.fixture(autouse=True)
def stub_job_run_tables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "api.jobs.prosperity._acquire_job_lock",
        lambda *args, **kwargs: (True, {"locked_until": None, "lock_owner": "test"}),
    )
    monkeypatch.setattr(
        "api.jobs.prosperity._update_job_run", lambda *args, **kwargs: None
    )


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

    async def fake_snapshot(req, request, **_kwargs):
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
    assert called["req"].from_date == dt.date(2024, 1, 9) - dt.timedelta(days=420)
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

    async def fake_snapshot(req, request, **_kwargs):
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
        return {
            "deleted": {"prosperity_features_daily": 5},
            "skipped": {},
            "warnings": [],
        }

    monkeypatch.setattr("api.jobs.prosperity._utc_today", lambda: dt.date(2024, 5, 20))
    monkeypatch.setattr("api.jobs.prosperity.snapshot_run", fake_snapshot)
    monkeypatch.setattr("api.jobs.prosperity.cleanup_retention_report", fake_cleanup)

    client = TestClient(app)
    resp = client.post(
        "/jobs/prosperity/daily-snapshot",
        headers={"X-FTIP-API-Key": "secret"},
    )

    assert resp.status_code == 200
    assert called["args"] == (dt.date(2024, 5, 19), 730)
    assert resp.json()["retention_deleted"] == {"prosperity_features_daily": 5}


def test_cleanup_retention_targets_v1_strategy_tables(
    monkeypatch: pytest.MonkeyPatch,
):
    called = []

    def fake_delete(table: str, cutoff: dt.date) -> int:
        called.append((table, cutoff))
        return 0

    monkeypatch.setattr("api.jobs.prosperity._delete_older_than", fake_delete)

    as_of_date = dt.date(2024, 5, 19)
    retention_days = 30
    cutoff = dt.date(2024, 4, 19)

    result = prosperity.cleanup_retention(as_of_date, retention_days)

    assert called == [
        ("prosperity_features_daily", cutoff),
        ("prosperity_signals_daily", cutoff),
        ("prosperity_strategy_signals_daily", cutoff),
        ("prosperity_ensemble_signals_daily", cutoff),
    ]
    assert result == {
        "prosperity_features_daily": 0,
        "prosperity_signals_daily": 0,
        "prosperity_strategy_signals_daily": 0,
        "prosperity_ensemble_signals_daily": 0,
    }


def test_delete_older_than_uses_schema_aware_date_column(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: Dict[str, Any] = {}

    monkeypatch.setattr(
        "api.jobs.prosperity.db.safe_fetchone",
        lambda sql, params: ("prosperity_strategy_signals_daily",),
    )
    monkeypatch.setattr(
        "api.jobs.prosperity._retention_table_columns",
        lambda table: ["symbol", "as_of_date", "lookback"],
    )

    def fake_fetchall(sql, params):
        recorded["sql"] = sql
        recorded["params"] = params
        return []

    monkeypatch.setattr("api.jobs.prosperity.db.safe_fetchall", fake_fetchall)

    deleted = prosperity._delete_older_than(
        "prosperity_strategy_signals_daily", dt.date(2024, 4, 1)
    )

    assert deleted == 0
    assert "as_of_date" in recorded["sql"]


def test_cleanup_retention_report_skips_missing_retention_column(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_delete(table: str, cutoff: dt.date) -> int:
        if table == "prosperity_strategy_signals_daily":
            raise ValueError("no retention date column available for prosperity_strategy_signals_daily")
        return 0

    monkeypatch.setattr("api.jobs.prosperity._delete_older_than", fake_delete)

    report = prosperity.cleanup_retention_report(dt.date(2024, 5, 19), 30)

    assert report["deleted"]["prosperity_features_daily"] == 0
    assert (
        "prosperity_strategy_signals_daily" in report["skipped"]
    )
    assert report["warnings"]
