"""Phase 24: Intelligence Digest API tests."""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_ic_health_db_disabled(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    result = mod._ic_health(dt.date(2026, 1, 5))
    assert result["ic_state"] == "UNKNOWN"


def test_ic_health_no_row(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: None)
    result = mod._ic_health(dt.date(2026, 1, 5))
    assert result["ic_state"] == "INSUFFICIENT"


def test_ic_health_with_row(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: ("STRONG", 42, 0.07, dt.date(2026, 1, 5)))
    result = mod._ic_health(dt.date(2026, 1, 5))
    assert result["ic_state"] == "STRONG"
    assert result["sample_count"] == 42
    assert result["mean_ic"] == pytest.approx(0.07)


def test_market_posture_db_disabled(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    result = mod._market_posture(dt.date(2026, 1, 5))
    assert result["breadth_state"] == "UNKNOWN"


def test_market_posture_with_rows(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    call_count = [0]

    def fake_fetchone(sql, params):
        call_count[0] += 1
        if call_count[0] == 1:
            return ("EXPANDING",)
        return ("fundamental_convergence", 12)

    monkeypatch.setattr(db, "safe_fetchone", fake_fetchone)
    result = mod._market_posture(dt.date(2026, 1, 5))
    assert result["breadth_state"] == "EXPANDING"
    assert result["dominant_regime"] == "fundamental_convergence"


def test_top_opportunities_db_disabled(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    assert mod._top_opportunities(dt.date(2026, 1, 5)) == []


def test_top_opportunities_returns_ranked(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: [
        ("NVDA", 90.0, "fundamental_convergence", "live_candidate", "BUY", "Technology"),
        ("AAPL", 75.0, "compensation_capture",    "live_candidate", "BUY", "Technology"),
    ])
    result = mod._top_opportunities(dt.date(2026, 1, 5), limit=5)
    assert len(result) == 2
    assert result[0]["symbol"] == "NVDA"
    assert result[0]["dau"] == pytest.approx(90.0)


def test_sector_rotation_db_disabled(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    assert mod._sector_rotation(dt.date(2026, 1, 5)) == []


def test_sector_rotation_aggregates_by_sector(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchall", lambda *_a, **_k: [
        ("Technology", 3, 1, 0, 78.5),
        ("Financials",  1, 2, 1, 65.0),
    ])
    result = mod._sector_rotation(dt.date(2026, 1, 5))
    assert result[0]["sector"] == "Technology"
    assert result[0]["buy_count"] == 3
    assert result[1]["sector"] == "Financials"
    assert result[1]["sell_count"] == 2


def test_provider_status_db_disabled(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    assert mod._provider_status(dt.date(2026, 1, 5)) == []


def test_calibration_quality_no_row(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: None)
    result = mod._calibration_quality(dt.date(2026, 1, 5))
    assert result["quality_score"] is None


def test_calibration_quality_with_row(monkeypatch):
    import api.ops as mod
    from api import db
    monkeypatch.setattr(db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db, "safe_fetchone", lambda *_a, **_k: ("21d", 0.62, 200))
    result = mod._calibration_quality(dt.date(2026, 1, 5))
    assert result["quality_score"] == pytest.approx(62.0)
    assert result["sample_count"] == 200


# ---------------------------------------------------------------------------
# Endpoint integration test
# ---------------------------------------------------------------------------

def test_intelligence_endpoint_returns_200(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get(
        "/ops/intelligence?as_of_date=2026-01-05",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "market_posture" in body
    assert "ic_health" in body
    assert "top_opportunities" in body
    assert "sector_rotation" in body
    assert "provider_status" in body
    assert "calibration_quality" in body
    assert body["as_of_date"] == "2026-01-05"


def test_intelligence_endpoint_top_n_param(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from api import db

    monkeypatch.setattr(db, "db_read_enabled", lambda: False)
    monkeypatch.setenv("FTIP_API_KEY", "secret")

    client = TestClient(app)
    resp = client.get(
        "/ops/intelligence?as_of_date=2026-01-05&top_n=3",
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
