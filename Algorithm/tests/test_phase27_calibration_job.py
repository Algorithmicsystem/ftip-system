"""Phase 27: Calibration Auto-Update Job tests."""
from __future__ import annotations

import datetime as dt

import pytest


def _pnl_rows(*entries):
    """(signal_label, horizon_days, regime_label, return_pct, hit)"""
    return list(entries)


def _patch_db(monkeypatch, rows):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: rows)


# ---------------------------------------------------------------------------
# compute_calibration_snapshot tests
# ---------------------------------------------------------------------------

def test_db_disabled_returns_db_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    from api.jobs.pnl import compute_calibration_snapshot
    result = compute_calibration_snapshot(dt.date(2026, 1, 5))
    assert result["status"] == "db_disabled"


def test_no_rows_returns_no_data(monkeypatch):
    _patch_db(monkeypatch, [])
    from api.jobs.pnl import compute_calibration_snapshot
    result = compute_calibration_snapshot(dt.date(2026, 1, 5))
    assert result["status"] == "no_data"


def test_overall_hit_rate_computed(monkeypatch):
    rows = _pnl_rows(
        ("BUY", 21, "regime_a", 2.0, True),
        ("BUY", 21, "regime_a", 1.0, True),
        ("BUY", 21, "regime_a", -1.0, False),
        ("BUY", 21, "regime_a", -0.5, False),
    )
    _patch_db(monkeypatch, rows)
    from api.jobs.pnl import compute_calibration_snapshot
    result = compute_calibration_snapshot(dt.date(2026, 1, 5))
    assert result["status"] == "ok"
    assert result["payload"]["overall_hit_rate"] == pytest.approx(0.5)
    assert result["payload"]["sample_count"] == 4


def test_bucket_stats_per_signal_horizon(monkeypatch):
    rows = _pnl_rows(
        ("BUY",  21, "r1", 2.0, True),
        ("BUY",  21, "r1", 1.0, True),
        ("SELL",  5, "r2", -3.0, True),
    )
    _patch_db(monkeypatch, rows)
    from api.jobs.pnl import compute_calibration_snapshot
    result = compute_calibration_snapshot(dt.date(2026, 1, 5))
    payload = result["payload"]
    buckets = {b["bucket_key"]: b for b in payload["buckets"]}
    assert "BUY_21d" in buckets
    assert "SELL_5d" in buckets
    assert buckets["BUY_21d"]["hit_rate"] == pytest.approx(1.0)
    assert buckets["SELL_5d"]["n"] == 1


def test_payload_contains_required_keys(monkeypatch):
    rows = _pnl_rows(("BUY", 21, "r1", 1.5, True))
    _patch_db(monkeypatch, rows)
    from api.jobs.pnl import compute_calibration_snapshot
    result = compute_calibration_snapshot(dt.date(2026, 1, 5))
    p = result["payload"]
    for key in ("overall_hit_rate", "overall_avg_return_pct", "sample_count", "buckets", "calibration_version"):
        assert key in p


# ---------------------------------------------------------------------------
# store_calibration_snapshot tests
# ---------------------------------------------------------------------------

def test_store_skipped_when_db_write_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_write_enabled", lambda: False)
    from api.jobs.pnl import store_calibration_snapshot
    result = store_calibration_snapshot(dt.date(2026, 1, 5), {"overall_hit_rate": 0.6})
    assert result is False


def test_store_calls_safe_execute(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    executed = []
    monkeypatch.setattr(db, "safe_execute", lambda sql, params: executed.append(params))
    from api.jobs.pnl import store_calibration_snapshot
    result = store_calibration_snapshot(dt.date(2026, 1, 5), {"overall_hit_rate": 0.6})
    assert result is True
    assert len(executed) == 1
    # snapshot_key should contain the date
    assert "2026-01-05" in executed[0][0]


# ---------------------------------------------------------------------------
# Endpoint test
# ---------------------------------------------------------------------------

def test_calibration_daily_endpoint_returns_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    rows = _pnl_rows(
        ("BUY", 21, "r1", 2.0, True),
        ("BUY", 21, "r1", -1.0, False),
    )
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "db_write_enabled", lambda: False)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: rows)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.post(
        "/jobs/calibration/daily",
        json={"as_of_date": "2026-01-05", "store": False},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "overall_hit_rate" in body
    assert "buckets" in body
