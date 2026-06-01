"""Phase 28: Regime Transition Detection tests."""
from __future__ import annotations

import datetime as dt

import pytest


# ---------------------------------------------------------------------------
# detect_regime_transition unit tests
# ---------------------------------------------------------------------------

def _patch_dominant(monkeypatch, today_regime, prior_regime, context=None):
    """Patch _dominant_regime to return different values for two dates."""
    import api.jobs.regime_monitor as mod
    from api import db

    dates_seen = []

    def fake_dominant(d):
        dates_seen.append(d)
        return today_regime if len(dates_seen) == 1 else prior_regime

    monkeypatch.setattr(mod, "_dominant_regime", fake_dominant)

    # Patch DB calls for breadth/IC/symbol_count
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)

    ctx = context or {}
    call_count = [0]

    def fake_fetchone(sql, params):
        call_count[0] += 1
        if "breadth_state" in sql:
            return (ctx.get("breadth", "EXPANDING"),)
        if "ic_state" in sql:
            return (ctx.get("ic", "STRONG"),)
        return (ctx.get("symbol_count", 10),)

    monkeypatch.setattr(db, "safe_fetchone", fake_fetchone)


def test_no_transition_same_regime(monkeypatch):
    import api.jobs.regime_monitor as mod
    monkeypatch.setattr(mod, "_dominant_regime", lambda _d: "fundamental_convergence")
    result = mod.detect_regime_transition(dt.date(2026, 1, 5))
    assert result is None


def test_transition_detected_on_regime_change(monkeypatch):
    import api.jobs.regime_monitor as mod
    _patch_dominant(monkeypatch, "liquidity_fracture", "fundamental_convergence")
    result = mod.detect_regime_transition(dt.date(2026, 1, 5))
    assert result is not None
    assert result["from_regime"] == "fundamental_convergence"
    assert result["to_regime"] == "liquidity_fracture"
    assert "transition_id" in result


def test_no_transition_when_today_regime_none(monkeypatch):
    import api.jobs.regime_monitor as mod
    _patch_dominant(monkeypatch, None, "fundamental_convergence")
    result = mod.detect_regime_transition(dt.date(2026, 1, 5))
    assert result is None


def test_transition_includes_breadth_and_ic(monkeypatch):
    import api.jobs.regime_monitor as mod
    _patch_dominant(
        monkeypatch, "compensation_capture", "behavioral_continuation",
        context={"breadth": "CONTRACTING", "ic": "WEAK", "symbol_count": 25},
    )
    result = mod.detect_regime_transition(dt.date(2026, 1, 5))
    assert result["breadth_state"] == "CONTRACTING"
    assert result["ic_state"] == "WEAK"
    assert result["symbol_count"] == 25


# ---------------------------------------------------------------------------
# store_regime_transition tests
# ---------------------------------------------------------------------------

def test_store_skipped_when_db_write_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_write_enabled", lambda: False)
    from api.jobs.regime_monitor import store_regime_transition
    result = store_regime_transition({
        "transition_id": "t1", "as_of_date": dt.date(2026, 1, 5),
        "from_regime": "regime_a", "to_regime": "regime_b",
        "symbol_count": 10, "breadth_state": None, "ic_state": None, "meta": {},
    })
    assert result is False


def test_store_calls_safe_execute(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_write_enabled", lambda: True)
    executed = []
    monkeypatch.setattr(db, "safe_execute", lambda sql, params: executed.append(params))
    from api.jobs.regime_monitor import store_regime_transition
    result = store_regime_transition({
        "transition_id": "t1", "as_of_date": dt.date(2026, 1, 5),
        "from_regime": "regime_a", "to_regime": "regime_b",
        "symbol_count": 10, "breadth_state": "EXPANDING", "ic_state": "STRONG", "meta": {},
    })
    assert result is True
    assert executed[0][0] == "t1"
    assert executed[0][2] == "regime_a"
    assert executed[0][3] == "regime_b"


# ---------------------------------------------------------------------------
# load_regime_transitions tests
# ---------------------------------------------------------------------------

def test_load_returns_empty_when_db_disabled(monkeypatch):
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    from api.jobs.regime_monitor import load_regime_transitions
    assert load_regime_transitions() == []


def test_load_serializes_rows(monkeypatch):
    import datetime as dt2
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    fake_rows = [(
        "tid-1", dt2.date(2026, 1, 5), "regime_a", "regime_b",
        12, "EXPANDING", "STRONG", dt2.datetime(2026, 1, 5, 10, 0, 0),
    )]
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: fake_rows)
    from api.jobs.regime_monitor import load_regime_transitions
    result = load_regime_transitions()
    assert len(result) == 1
    assert result[0]["from_regime"] == "regime_a"
    assert result[0]["to_regime"] == "regime_b"
    assert result[0]["as_of_date"] == "2026-01-05"


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

def test_transitions_endpoint_returns_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get("/ops/regime/transitions", headers={"X-FTIP-API-Key": "secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert "transitions" in body
    assert "count" in body


def test_detect_endpoint_no_transition(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    import api.jobs.regime_monitor as mod

    monkeypatch.setattr(mod, "_dominant_regime", lambda _d: "fundamental_convergence")
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.post(
        "/ops/regime/detect?as_of_date=2026-01-05&store=false",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_transition"
