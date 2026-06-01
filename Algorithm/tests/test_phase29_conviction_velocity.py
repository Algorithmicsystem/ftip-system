"""Phase 29: Conviction Velocity tests."""
from __future__ import annotations

import datetime as dt

import pytest


# ---------------------------------------------------------------------------
# _linear_slope unit tests
# ---------------------------------------------------------------------------

def test_slope_flat_series():
    from api.axiom.routes import _linear_slope
    assert _linear_slope([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_slope_monotone_increasing():
    from api.axiom.routes import _linear_slope
    slope = _linear_slope([10.0, 20.0, 30.0, 40.0, 50.0])
    assert slope == pytest.approx(10.0)


def test_slope_monotone_decreasing():
    from api.axiom.routes import _linear_slope
    slope = _linear_slope([50.0, 40.0, 30.0, 20.0, 10.0])
    assert slope == pytest.approx(-10.0)


def test_slope_single_value_returns_zero():
    from api.axiom.routes import _linear_slope
    assert _linear_slope([42.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_conviction_trends unit tests
# ---------------------------------------------------------------------------

def _patch_trends(monkeypatch, rows):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: rows)


def _row(sym, day_offset, conf, dau=75.0):
    import datetime as dt2
    return (sym, dt2.date(2026, 1, 1) + dt2.timedelta(days=day_offset), conf, dau)


def test_db_disabled_returns_db_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5))
    assert result["status"] == "db_disabled"


def test_accelerating_symbol_detected(monkeypatch):
    rows = [
        _row("NVDA", 0, 50.0),
        _row("NVDA", 1, 55.0),
        _row("NVDA", 2, 60.0),
        _row("NVDA", 3, 65.0),
        _row("NVDA", 4, 70.0),
    ]
    _patch_trends(monkeypatch, rows)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5))
    assert result["status"] == "ok"
    nvda = result["trends"][0]
    assert nvda["symbol"] == "NVDA"
    assert nvda["conviction_velocity"] == pytest.approx(5.0)
    assert nvda["trend"] == "accelerating"


def test_decelerating_symbol_detected(monkeypatch):
    rows = [
        _row("AAPL", 0, 70.0),
        _row("AAPL", 1, 65.0),
        _row("AAPL", 2, 60.0),
        _row("AAPL", 3, 55.0),
        _row("AAPL", 4, 50.0),
    ]
    _patch_trends(monkeypatch, rows)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5))
    aapl = result["trends"][0]
    assert aapl["trend"] == "decelerating"
    assert aapl["conviction_velocity"] < 0


def test_min_dau_filter(monkeypatch):
    rows = [
        _row("HIGH", 0, 60.0, dau=80.0),
        _row("LOW",  0, 60.0, dau=20.0),
    ]
    _patch_trends(monkeypatch, rows)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5), min_dau=50.0)
    syms = [t["symbol"] for t in result["trends"]]
    assert "HIGH" in syms
    assert "LOW" not in syms


def test_trends_sorted_by_velocity_desc(monkeypatch):
    rows = (
        [_row("FAST", i, 50.0 + 10 * i) for i in range(5)] +
        [_row("SLOW", i, 60.0 + 1 * i) for i in range(5)]
    )
    _patch_trends(monkeypatch, rows)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5))
    velocities = [t["conviction_velocity"] for t in result["trends"]]
    assert velocities == sorted(velocities, reverse=True)


def test_change_over_window_computed(monkeypatch):
    rows = [_row("SYM", i, 50.0 + 2 * i) for i in range(5)]
    _patch_trends(monkeypatch, rows)
    from api.axiom.routes import _compute_conviction_trends
    result = _compute_conviction_trends(dt.date(2026, 1, 5))
    sym = result["trends"][0]
    assert sym["change_over_window"] == pytest.approx(8.0)  # 58 - 50


# ---------------------------------------------------------------------------
# Endpoint test
# ---------------------------------------------------------------------------

def test_conviction_trends_endpoint_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get(
        "/axiom/conviction/trends?as_of_date=2026-01-05&window=5",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "trends" in body
    assert body["status"] in ("ok", "db_disabled")
