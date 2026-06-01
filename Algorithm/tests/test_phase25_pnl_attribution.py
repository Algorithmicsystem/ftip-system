"""Phase 25: P&L Attribution tests."""
from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rows(*entries) -> List[Tuple]:
    """Build fake signal_pnl_daily rows: (regime, sector, sig_ver, ret_pct, hit)."""
    return list(entries)


def _patch_db(monkeypatch, rows):
    import api.jobs.pnl as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: rows)


# ---------------------------------------------------------------------------
# Unit: load_pnl_attribution
# ---------------------------------------------------------------------------

def test_db_disabled_returns_db_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    from api.jobs.pnl import load_pnl_attribution
    result = load_pnl_attribution(dt.date(2026, 1, 5))
    assert result["status"] == "db_disabled"
    assert result["buckets"] == []


def test_group_by_regime_buckets(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("fundamental_convergence", "Technology", "v1.0", 2.5,  True),
        ("fundamental_convergence", "Financials",  "v1.0", 1.2,  True),
        ("liquidity_fracture",       "Technology", "v1.0", -3.0, False),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="regime")
    assert result["status"] == "ok"
    assert result["group_by"] == "regime"
    buckets = {b["bucket"]: b for b in result["buckets"]}
    assert "fundamental_convergence" in buckets
    assert "liquidity_fracture" in buckets
    fc = buckets["fundamental_convergence"]
    assert fc["n"] == 2
    assert fc["win_rate"] == pytest.approx(1.0)
    assert fc["avg_return_pct"] == pytest.approx(1.85, abs=0.01)


def test_group_by_sector(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("any_regime", "Technology", "v1.0",  3.0,  True),
        ("any_regime", "Technology", "v1.0",  1.0,  True),
        ("any_regime", "Financials",  "v1.0", -1.0, False),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="sector")
    buckets = {b["bucket"]: b for b in result["buckets"]}
    assert "Technology" in buckets
    assert buckets["Technology"]["n"] == 2
    assert buckets["Technology"]["win_rate"] == pytest.approx(1.0)
    assert buckets["Financials"]["win_rate"] == pytest.approx(0.0)


def test_group_by_signal_version(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("r1", "Tech", "v2.0", 4.0, True),
        ("r1", "Tech", "v2.0", 2.0, True),
        ("r1", "Tech", "v1.0", -2.0, False),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="signal_version")
    buckets = {b["bucket"]: b for b in result["buckets"]}
    assert "v2.0" in buckets
    assert "v1.0" in buckets
    assert buckets["v2.0"]["avg_return_pct"] == pytest.approx(3.0)


def test_max_drawdown_is_worst_return(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("regime_a", "Tech", "v1", 5.0,  True),
        ("regime_a", "Tech", "v1", -8.0, False),
        ("regime_a", "Tech", "v1", 1.0,  True),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="regime")
    bucket = result["buckets"][0]
    assert bucket["max_drawdown_pct"] == pytest.approx(-8.0)


def test_rows_without_return_still_counted_in_n(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("regime_a", "Tech", "v1", None, None),
        ("regime_a", "Tech", "v1", 2.0,  True),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="regime")
    bucket = result["buckets"][0]
    assert bucket["n"] == 2
    assert bucket["avg_return_pct"] == pytest.approx(2.0)


def test_invalid_group_by_falls_back_to_regime(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    rows = _rows(
        ("some_regime", "Tech", "v1", 1.0, True),
    )
    _patch_db(monkeypatch, rows)
    result = load_pnl_attribution(dt.date(2026, 1, 5), group_by="nonsense")
    assert result["group_by"] == "regime"


def test_empty_rows_returns_empty_buckets(monkeypatch):
    from api.jobs.pnl import load_pnl_attribution
    _patch_db(monkeypatch, [])
    result = load_pnl_attribution(dt.date(2026, 1, 5))
    assert result["status"] == "ok"
    assert result["bucket_count"] == 0
    assert result["buckets"] == []


# ---------------------------------------------------------------------------
# Endpoint integration test
# ---------------------------------------------------------------------------

def test_attribution_endpoint_returns_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get(
        "/jobs/pnl/attribution?as_of_date=2026-01-05&group_by=regime",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "buckets" in body
    assert "group_by" in body
    assert body["group_by"] == "regime"


def test_attribution_endpoint_sector_grouping(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get(
        "/jobs/pnl/attribution?group_by=sector&horizon_days=5",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["group_by"] == "sector"
