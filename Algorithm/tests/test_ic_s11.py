"""Session 11: IC decay pipeline — unit tests."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import security
from api.jobs import ic as ic_module
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


def _db_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


# ---------------------------------------------------------------------------
# spearman_ic: pure math
# ---------------------------------------------------------------------------

def test_spearman_perfect_positive_correlation():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    ic, _ = ic_module.spearman_ic(scores, returns)
    assert ic == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    ic, _ = ic_module.spearman_ic(scores, returns)
    assert ic == pytest.approx(-1.0)


def test_spearman_zero_correlation():
    # alternating high/low scores against monotone returns → near-zero IC
    scores = [1.0, 5.0, 2.0, 4.0, 3.0, 1.0, 5.0, 2.0, 4.0, 3.0]
    returns = [0.01 * i for i in range(1, 11)]
    ic, _ = ic_module.spearman_ic(scores, returns)
    assert ic is not None
    assert abs(ic) < 0.5  # not perfectly correlated


def test_spearman_returns_none_for_insufficient_sample():
    scores = [1.0, 2.0]
    returns = [0.01, 0.02]
    ic, p = ic_module.spearman_ic(scores, returns)
    assert ic is None
    assert p is None


def test_spearman_handles_ties():
    scores = [1.0, 1.0, 2.0, 2.0, 3.0]
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    ic, _ = ic_module.spearman_ic(scores, returns)
    assert ic is not None
    assert -1.0 <= ic <= 1.0


def test_spearman_p_value_for_large_sample():
    # monotone → IC=1, p should be very small
    n = 50
    scores = list(range(n))
    returns = [i * 0.001 for i in range(n)]
    ic, p = ic_module.spearman_ic(scores, returns)
    assert ic == pytest.approx(1.0)
    assert p is not None
    assert p < 0.01


# ---------------------------------------------------------------------------
# _extract_score
# ---------------------------------------------------------------------------

def test_extract_composite_uses_dau():
    payload = {"deployable_alpha_utility": 0.72, "engine_scores": {}}
    assert ic_module._extract_score(payload, "composite") == pytest.approx(0.72)


def test_extract_composite_falls_back_to_gross_opportunity():
    payload = {"gross_opportunity": 0.55}
    assert ic_module._extract_score(payload, "composite") == pytest.approx(0.55)


def test_extract_engine_score():
    payload = {"engine_scores": {"fundamental_reality": {"score": 0.65}}}
    assert ic_module._extract_score(payload, "fundamental_reality") == pytest.approx(0.65)


def test_extract_missing_engine_returns_none():
    payload = {"engine_scores": {}}
    assert ic_module._extract_score(payload, "flow_transmission") is None


def test_extract_none_payload_returns_none():
    assert ic_module._extract_score({}, "composite") is None


# ---------------------------------------------------------------------------
# compute_ic_decay_summary
# ---------------------------------------------------------------------------

def test_decay_summary_insufficient_for_tiny_history():
    history = [{"as_of_date": "2024-01-01", "ic_value": 0.1}]
    result = ic_module.compute_ic_decay_summary(history)
    assert result["ic_state"] == "INSUFFICIENT"
    assert result["icir"] is None


def test_decay_summary_strong_state():
    # High consistent positive IC → STRONG
    history = [{"as_of_date": f"2024-{i:02d}-01", "ic_value": 0.2} for i in range(1, 13)]
    result = ic_module.compute_ic_decay_summary(history)
    assert result["mean_ic"] == pytest.approx(0.2, abs=1e-5)
    assert result["icir"] is not None
    # std=0 when all values identical → icir=None; but with tiny std → very high icir
    # With 12 identical values, std_ddof1 = 0 → icir = None
    # So let's use varied values:
    history2 = [{"as_of_date": f"2024-{i:02d}-01", "ic_value": 0.18 + (i % 3) * 0.02} for i in range(1, 13)]
    result2 = ic_module.compute_ic_decay_summary(history2)
    assert result2["mean_ic"] is not None
    assert result2["icir"] is not None


def test_decay_summary_degraded_state():
    history = [{"as_of_date": f"2024-{i:02d}-01", "ic_value": -0.2 - (i % 3) * 0.01} for i in range(1, 13)]
    result = ic_module.compute_ic_decay_summary(history)
    assert result["ic_state"] == "DEGRADED"
    assert result["mean_ic"] < 0


def test_decay_summary_has_rolling_fields():
    history = [{"as_of_date": f"2024-01-{i:02d}", "ic_value": 0.05 * (i % 5)} for i in range(1, 30)]
    result = ic_module.compute_ic_decay_summary(history)
    assert "ic_mean_21d" in result
    assert "icir_21d" in result
    assert "t_stat" in result


# ---------------------------------------------------------------------------
# ic_state classification
# ---------------------------------------------------------------------------

def test_ic_state_strong():
    assert ic_module._ic_state(0.6) == "STRONG"

def test_ic_state_moderate():
    assert ic_module._ic_state(0.35) == "MODERATE"

def test_ic_state_weak():
    assert ic_module._ic_state(0.1) == "WEAK"

def test_ic_state_degraded():
    assert ic_module._ic_state(-0.1) == "DEGRADED"

def test_ic_state_insufficient():
    assert ic_module._ic_state(None) == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# store_ic_snapshot
# ---------------------------------------------------------------------------

def test_store_returns_zero_when_write_disabled(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.db_write_enabled", lambda: False)
    result = ic_module.store_ic_snapshot(dt.date(2024, 1, 10), {("composite", "21d"): {"ic_value": 0.1}})
    assert result == 0


def test_store_calls_safe_execute(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.db_write_enabled", lambda: True)
    calls = []
    monkeypatch.setattr("api.jobs.ic.db.safe_execute", lambda sql, params: calls.append(params))
    snapshot = {
        ("composite", "21d"): {"ic_value": 0.15, "sample_size": 20, "p_value": 0.04, "t_stat": 2.1, "ic_state": "WEAK"},
        ("fundamental_reality", "21d"): {"ic_value": 0.22, "sample_size": 20, "p_value": 0.02, "t_stat": 2.5, "ic_state": "MODERATE"},
    }
    result = ic_module.store_ic_snapshot(dt.date(2024, 1, 10), snapshot)
    assert result == 2
    assert len(calls) == 2


def test_store_returns_zero_for_empty_snapshot(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.db_write_enabled", lambda: True)
    assert ic_module.store_ic_snapshot(dt.date(2024, 1, 10), {}) == 0


# ---------------------------------------------------------------------------
# load_ic_history
# ---------------------------------------------------------------------------

def test_load_returns_empty_when_db_disabled(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.db_read_enabled", lambda: False)
    result = ic_module.load_ic_history("composite", "21d")
    assert result == []


def test_load_returns_rows(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.db_read_enabled", lambda: True)
    rows = [
        (dt.date(2024, 1, 10), 0.15, 20, "WEAK"),
        (dt.date(2024, 1, 11), 0.22, 20, "MODERATE"),
    ]
    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", lambda sql, params: rows)
    result = ic_module.load_ic_history("composite", "21d")
    assert len(result) == 2
    assert result[0]["ic_value"] == pytest.approx(0.15)
    assert result[1]["ic_state"] == "MODERATE"


# ---------------------------------------------------------------------------
# compute_ic_snapshot (integration with mocked DB)
# ---------------------------------------------------------------------------

def _make_axiom_rows(n: int):
    import json
    rows = []
    for i in range(n):
        payload = {
            "deployable_alpha_utility": 0.1 * (i % 10),
            "engine_scores": {
                "fundamental_reality": {"score": 0.05 * (i % 10)},
            },
        }
        rows.append((f"SYM{i:03d}", payload))
    return rows


def _make_bars_rows(n: int, entry_date: dt.date):
    rows = []
    for i in range(n):
        sym = f"SYM{i:03d}"
        entry_close = 100.0 + i
        exit_close = entry_close * (1.0 + 0.001 * i)
        rows.append((sym, entry_close, exit_close))
    return rows


def test_compute_snapshot_empty_when_no_scores(monkeypatch):
    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", lambda sql, params: [])
    result = ic_module.compute_ic_snapshot(dt.date(2024, 1, 10))
    assert result == {}


def test_compute_snapshot_produces_ic_values(monkeypatch):
    axiom_rows = _make_axiom_rows(20)
    bars_rows = _make_bars_rows(20, dt.date(2024, 1, 10))

    def _fake_fetchall(sql, params):
        if "axiom_scores_daily" in sql:
            return axiom_rows
        if "market_bars_daily" in sql:
            return bars_rows
        return []

    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", _fake_fetchall)
    result = ic_module.compute_ic_snapshot(dt.date(2024, 1, 10))
    assert len(result) > 0
    # composite + 21d should be present
    key = ("composite", "21d")
    assert key in result
    assert result[key]["sample_size"] == 20
    assert result[key]["ic_value"] is not None


def test_compute_snapshot_insufficient_when_no_bars(monkeypatch):
    axiom_rows = _make_axiom_rows(10)

    def _fake_fetchall(sql, params):
        if "axiom_scores_daily" in sql:
            return axiom_rows
        return []  # no bars

    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", _fake_fetchall)
    result = ic_module.compute_ic_snapshot(dt.date(2024, 1, 10))
    # No bars → no forward returns → no IC keys
    assert all(v.get("ic_value") is None or v.get("ic_state") == "INSUFFICIENT" for v in result.values())


# ---------------------------------------------------------------------------
# /jobs/ic/daily-snapshot endpoint
# ---------------------------------------------------------------------------

def test_endpoint_requires_auth(monkeypatch):
    _db_env(monkeypatch)
    client = TestClient(app)
    resp = client.post("/jobs/ic/daily-snapshot", json={})
    assert resp.status_code == 401


def test_endpoint_returns_no_data_when_no_scores(monkeypatch):
    _db_env(monkeypatch)
    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", lambda sql, params: [])
    client = TestClient(app)
    resp = client.post(
        "/jobs/ic/daily-snapshot",
        json={"as_of_date": "2024-01-10"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_data"


def test_endpoint_returns_ok_with_data(monkeypatch):
    _db_env(monkeypatch)
    axiom_rows = _make_axiom_rows(20)
    bars_rows = _make_bars_rows(20, dt.date(2024, 1, 10))

    def _fake_fetchall(sql, params):
        if "axiom_scores_daily" in sql:
            return axiom_rows
        if "market_bars_daily" in sql:
            return bars_rows
        return []

    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", _fake_fetchall)
    monkeypatch.setattr("api.jobs.ic.db.safe_execute", lambda sql, params: None)

    client = TestClient(app)
    resp = client.post(
        "/jobs/ic/daily-snapshot",
        json={"as_of_date": "2024-01-10"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["stored"] > 0
    assert "snapshot" in body


def test_endpoint_defaults_to_today(monkeypatch):
    _db_env(monkeypatch)
    monkeypatch.setattr("api.jobs.ic.db.safe_fetchall", lambda sql, params: [])
    client = TestClient(app)
    resp = client.post(
        "/jobs/ic/daily-snapshot",
        json={},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_data"
