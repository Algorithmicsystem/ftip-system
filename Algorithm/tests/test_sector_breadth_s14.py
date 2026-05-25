"""Session 14: Sector breadth + rotation heatmap tests."""
from __future__ import annotations

import datetime as dt
import statistics
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import security


def _db_env(monkeypatch):
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    monkeypatch.setenv("FTIP_DB_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_READ_ENABLED", "1")
    monkeypatch.setenv("FTIP_DB_WRITE_ENABLED", "1")
    security.reset_auth_cache()


AUTH = {"X-FTIP-API-Key": "secret"}


# ---------------------------------------------------------------------------
# compute_sector_breadth unit tests (pure logic, no DB)
# ---------------------------------------------------------------------------

def _make_rows(entries: List[tuple]) -> List[tuple]:
    """Simulate DB rows (sector, score, signal)."""
    return entries


class TestComputeSectorBreadth:
    """Unit tests for compute_sector_breadth logic (mocked DB)."""

    def _compute(self, rows):
        from api.jobs.sector_breadth import compute_sector_breadth
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.safe_fetchall.return_value = rows
            return compute_sector_breadth(dt.date(2025, 1, 2), 252)

    def test_empty_rows_returns_empty_list(self):
        result = self._compute([])
        assert result == []

    def test_single_sector_all_buy(self):
        rows = [
            ("technology", 0.8, "BUY"),
            ("technology", 0.7, "BUY"),
            ("technology", 0.6, "BUY"),
        ]
        result = self._compute(rows)
        assert len(result) == 1
        r = result[0]
        assert r["sector"] == "technology"
        assert r["symbol_count"] == 3
        assert r["buy_count"] == 3
        assert r["sell_count"] == 0
        assert r["hold_count"] == 0
        assert r["breadth_confirmation_score"] == 100.0
        assert r["breadth_state"] == "EXPANDING"

    def test_all_sell_sector(self):
        rows = [
            ("energy", 0.1, "SELL"),
            ("energy", 0.05, "SELL"),
            ("energy", 0.0, "SELL"),
        ]
        result = self._compute(rows)
        assert len(result) == 1
        r = result[0]
        assert r["breadth_confirmation_score"] == 0.0
        assert r["breadth_state"] == "CONTRACTING"

    def test_mixed_signals_neutral(self):
        rows = [
            ("healthcare", 0.5, "BUY"),
            ("healthcare", 0.4, "BUY"),
            ("healthcare", 0.3, "HOLD"),
            ("healthcare", 0.35, "HOLD"),
            ("healthcare", -0.1, "SELL"),
        ]
        result = self._compute(rows)
        r = result[0]
        # 2 BUY out of 5 = 40%; participation = 4/5 = 80%
        assert r["breadth_confirmation_score"] == 40.0
        assert r["participation_breadth_score"] == 80.0
        # 40% confirmation, not CONTRACTING (>30), not EXPANDING (<65) → NEUTRAL or STRESSED
        assert r["breadth_state"] in ("NEUTRAL", "STRESSED")

    def test_multiple_sectors_ranked(self):
        rows = [
            ("financials", 0.9, "BUY"),
            ("financials", 0.8, "BUY"),
            ("financials", 0.7, "BUY"),
            ("utilities", 0.1, "SELL"),
            ("utilities", 0.05, "SELL"),
            ("utilities", 0.0, "HOLD"),
        ]
        result = self._compute(rows)
        assert len(result) == 2
        assert result[0]["sector"] == "financials"
        assert result[1]["sector"] == "utilities"
        assert result[0]["rotation_rank"] == 1
        assert result[1]["rotation_rank"] == 2

    def test_rotation_rank_assigned_correctly(self):
        rows = [
            ("c_sector", 0.8, "BUY"),
            ("a_sector", 0.2, "SELL"),
            ("b_sector", 0.9, "BUY"),
            ("b_sector", 0.85, "BUY"),
        ]
        result = self._compute(rows)
        sectors_by_rank = {r["rotation_rank"]: r["sector"] for r in result}
        # b_sector: 2/2 = 100% BUY; c_sector: 1/1 = 100% BUY → tie → order preserved; a_sector: 0%
        assert sectors_by_rank[len(result)] == "a_sector"

    def test_avg_score_computed(self):
        rows = [
            ("technology", 0.6, "BUY"),
            ("technology", 0.4, "HOLD"),
        ]
        result = self._compute(rows)
        assert result[0]["avg_score"] == pytest.approx(0.5, abs=0.0001)

    def test_none_score_treated_as_zero(self):
        rows = [
            ("energy", None, "HOLD"),
            ("energy", 0.5, "BUY"),
        ]
        result = self._compute(rows)
        assert result[0]["avg_score"] == pytest.approx(0.25, abs=0.0001)

    def test_none_signal_treated_as_hold(self):
        rows = [
            ("tech", 0.5, None),
            ("tech", 0.6, "BUY"),
        ]
        result = self._compute(rows)
        r = result[0]
        assert r["hold_count"] == 1
        assert r["buy_count"] == 1

    def test_breadth_state_expanding_thresholds(self):
        # 7/10 BUY → 70%; 8/10 positive → 80%
        rows = [("tech", 0.5, "BUY")] * 7 + [("tech", 0.4, "HOLD")] + [("tech", -0.1, "SELL")] * 2
        result = self._compute(rows)
        assert result[0]["breadth_state"] == "EXPANDING"

    def test_breadth_state_contracting_by_confirmation(self):
        # 2/10 BUY → 20% ≤ 30%
        rows = [("energy", 0.5, "BUY")] * 2 + [("energy", -0.1, "SELL")] * 8
        result = self._compute(rows)
        assert result[0]["breadth_state"] == "CONTRACTING"

    def test_breadth_state_contracting_by_participation(self):
        # 3/10 positive (all others negative) → 30% ≤ 30%
        rows = [("energy", 0.5, "BUY")] * 3 + [("energy", -0.1, "HOLD")] * 7
        result = self._compute(rows)
        assert result[0]["breadth_state"] == "CONTRACTING"

    def test_query_uses_correct_as_of_column(self):
        """Verify the DB query uses 'as_of' not 'as_of_date'."""
        from api.jobs.sector_breadth import compute_sector_breadth
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.safe_fetchall.return_value = []
            compute_sector_breadth(dt.date(2025, 1, 2), 252)
            call_args = mock_db.safe_fetchall.call_args
            sql = call_args[0][0]
            assert "psd.as_of = %s" in sql
            assert "as_of_date" not in sql


# ---------------------------------------------------------------------------
# store_sector_breadth tests
# ---------------------------------------------------------------------------

class TestStoreSectorBreadth:
    def _store(self, sectors):
        from api.jobs.sector_breadth import store_sector_breadth
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.safe_execute.return_value = None
            return store_sector_breadth(dt.date(2025, 1, 2), sectors), mock_db

    def test_empty_list_returns_zero(self):
        count, _ = self._store([])
        assert count == 0

    def test_stores_each_sector(self):
        sectors = [
            {
                "sector": "technology", "symbol_count": 5,
                "buy_count": 4, "sell_count": 0, "hold_count": 1,
                "avg_score": 0.7, "breadth_confirmation_score": 80.0,
                "participation_breadth_score": 90.0, "breadth_state": "EXPANDING",
                "rotation_rank": 1,
            },
            {
                "sector": "energy", "symbol_count": 3,
                "buy_count": 1, "sell_count": 2, "hold_count": 0,
                "avg_score": 0.2, "breadth_confirmation_score": 33.3,
                "participation_breadth_score": 33.3, "breadth_state": "NEUTRAL",
                "rotation_rank": 2,
            },
        ]
        count, mock_db = self._store(sectors)
        assert count == 2
        assert mock_db.safe_execute.call_count == 2

    def test_db_exception_suppressed_count_decrements(self):
        from api.jobs.sector_breadth import store_sector_breadth
        sectors = [
            {
                "sector": "technology", "symbol_count": 5,
                "buy_count": 4, "sell_count": 0, "hold_count": 1,
                "avg_score": 0.7, "breadth_confirmation_score": 80.0,
                "participation_breadth_score": 90.0, "breadth_state": "EXPANDING",
                "rotation_rank": 1,
            }
        ]
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.safe_execute.side_effect = Exception("db error")
            count = store_sector_breadth(dt.date(2025, 1, 2), sectors)
        assert count == 0


# ---------------------------------------------------------------------------
# load_sector_breadth_latest tests
# ---------------------------------------------------------------------------

class TestLoadSectorBreadthLatest:
    def test_db_read_disabled_returns_none(self):
        from api.jobs.sector_breadth import load_sector_breadth_latest
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            assert load_sector_breadth_latest() is None

    def test_no_rows_returns_none(self):
        from api.jobs.sector_breadth import load_sector_breadth_latest
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.return_value = (None,)
            assert load_sector_breadth_latest() is None

    def test_returns_structured_result(self):
        from api.jobs.sector_breadth import load_sector_breadth_latest
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.return_value = (dt.date(2025, 1, 2),)
            mock_db.safe_fetchall.return_value = [
                ("technology", 5, 4, 0, 1, 0.7, 80.0, 90.0, "EXPANDING", 1),
                ("energy", 3, 1, 2, 0, 0.2, 33.3, 33.3, "NEUTRAL", 2),
            ]
            result = load_sector_breadth_latest()
        assert result is not None
        assert result["as_of_date"] == "2025-01-02"
        assert len(result["sectors"]) == 2
        assert result["sectors"][0]["sector"] == "technology"
        assert result["sectors"][0]["breadth_state"] == "EXPANDING"
        assert result["sectors"][1]["rotation_rank"] == 2

    def test_exception_returns_none(self):
        from api.jobs.sector_breadth import load_sector_breadth_latest
        with patch("api.jobs.sector_breadth.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.side_effect = Exception("db failure")
            assert load_sector_breadth_latest() is None


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestSectorBreadthRoutes:
    def test_daily_snapshot_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/jobs/sector-breadth/daily-snapshot", json={})
        assert resp.status_code == 401

    def test_latest_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/jobs/sector-breadth/latest")
        assert resp.status_code == 401

    def test_daily_snapshot_db_disabled_returns_skipped(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post(
            "/jobs/sector-breadth/daily-snapshot",
            json={},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_daily_snapshot_no_data(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.jobs.sector_breadth.db.safe_fetchall", lambda *a, **kw: [])
        client = TestClient(app)
        resp = client.post(
            "/jobs/sector-breadth/daily-snapshot",
            json={"as_of_date": "2025-01-02"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_data"
        assert data["sector_count"] == 0

    def test_daily_snapshot_stores_and_returns_sectors(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        rows = [
            ("technology", 0.8, "BUY"),
            ("technology", 0.7, "BUY"),
            ("energy", -0.2, "SELL"),
        ]
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_write_enabled", lambda: True)
        monkeypatch.setattr("api.jobs.sector_breadth.db.safe_fetchall", lambda *a, **kw: rows)
        monkeypatch.setattr("api.jobs.sector_breadth.db.safe_execute", lambda *a, **kw: None)
        client = TestClient(app)
        resp = client.post(
            "/jobs/sector-breadth/daily-snapshot",
            json={"as_of_date": "2025-01-02"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["sector_count"] == 2
        assert data["stored"] == 2

    def test_latest_no_data(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.jobs.sector_breadth.db.safe_fetchone", lambda *a, **kw: (None,))
        client = TestClient(app)
        resp = client.get("/jobs/sector-breadth/latest", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_data"

    def test_latest_returns_stored_data(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.jobs.sector_breadth.db.db_read_enabled", lambda: True)
        monkeypatch.setattr(
            "api.jobs.sector_breadth.db.safe_fetchone",
            lambda *a, **kw: (dt.date(2025, 1, 2),),
        )
        monkeypatch.setattr(
            "api.jobs.sector_breadth.db.safe_fetchall",
            lambda *a, **kw: [
                ("technology", 5, 4, 0, 1, 0.7, 80.0, 90.0, "EXPANDING", 1),
            ],
        )
        client = TestClient(app)
        resp = client.get("/jobs/sector-breadth/latest", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["as_of_date"] == "2025-01-02"
        assert len(data["sectors"]) == 1
        assert data["sectors"][0]["breadth_state"] == "EXPANDING"

    def test_routes_registered_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/jobs/sector-breadth/daily-snapshot" in paths
        assert "/jobs/sector-breadth/latest" in paths


# ---------------------------------------------------------------------------
# breadth.py bug fix test
# ---------------------------------------------------------------------------

class TestBreadthColumnFix:
    """Verify the breadth job uses 'as_of' not 'as_of_date'."""

    def test_market_breadth_uses_as_of_column(self):
        from api.jobs.breadth import compute_market_breadth
        with patch("api.jobs.breadth.db") as mock_db:
            mock_db.safe_fetchall.return_value = []
            compute_market_breadth(dt.date(2025, 1, 2), 252)
            call_args = mock_db.safe_fetchall.call_args
            sql = call_args[0][0]
            assert "as_of = %s" in sql
            assert "as_of_date" not in sql
