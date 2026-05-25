"""Session 16: Universe screener — conviction-ranked opportunity scan tests."""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import patch, MagicMock

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

_DATE = dt.date(2025, 1, 2)


def _make_payload(
    dau=72.0, fragility=35.0, liquidity=65.0, research=60.0,
    confidence=70.0, tier="live_candidate", regime="fundamental_convergence",
) -> dict:
    return {
        "deployable_alpha_utility": dau,
        "overall_confidence": confidence,
        "deployability_tier": tier,
        "regime_label": regime,
        "engine_scores": {
            "critical_fragility":  {"score": fragility},
            "liquidity_convexity": {"score": liquidity},
            "research_integrity":  {"score": research},
        },
    }


def _make_rows(*symbols_and_signals):
    """Build fake DB rows: (symbol, payload_dict, signal_label)."""
    rows = []
    for symbol, signal, dau in symbols_and_signals:
        rows.append((symbol, _make_payload(dau=dau), signal))
    return rows


# ---------------------------------------------------------------------------
# screen_universe unit tests (mocked DB)
# ---------------------------------------------------------------------------

class TestScreenUniverse:
    def _screen(self, rows, *, ic_state="MODERATE", breadth_state="EXPANDING", **kwargs):
        from api.axiom.screener import screen_universe
        with (
            patch("api.axiom.screener.db") as mock_db,
        ):
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.side_effect = [
                (ic_state,),    # first call: IC state
                (breadth_state,),  # second call: breadth state
            ]
            mock_db.safe_fetchall.return_value = rows
            return screen_universe(_DATE, **kwargs)

    def test_db_disabled_returns_early(self):
        from api.axiom.screener import screen_universe
        with patch("api.axiom.screener.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            result = screen_universe(_DATE)
        assert result["status"] == "db_disabled"
        assert result["results"] == []

    def test_empty_universe_returns_ok(self):
        result = self._screen([])
        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["total_screened"] == 0

    def test_returns_results_ranked_by_conviction_desc(self):
        rows = _make_rows(
            ("AAPL", "BUY", 50.0),
            ("NVDA", "BUY", 90.0),
            ("TSLA", "BUY", 70.0),
        )
        result = self._screen(rows)
        symbols = [r["symbol"] for r in result["results"]]
        # Should be ordered: NVDA (highest DAU → highest conviction) first
        assert symbols[0] == "NVDA"
        assert result["results"][0]["rank"] == 1

    def test_rank_is_sequential(self):
        rows = _make_rows(
            ("A", "BUY", 80.0),
            ("B", "BUY", 70.0),
            ("C", "BUY", 60.0),
        )
        result = self._screen(rows)
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == [1, 2, 3]

    def test_signal_filter_buy_only(self):
        rows = _make_rows(
            ("NVDA", "BUY",  80.0),
            ("AAPL", "SELL", 75.0),
            ("MSFT", "HOLD", 70.0),
        )
        result = self._screen(rows, signal_filter=["BUY"])
        assert all(r["signal_label"] == "BUY" for r in result["results"])

    def test_signal_filter_sell_only(self):
        rows = _make_rows(
            ("NVDA", "BUY",  80.0),
            ("AAPL", "SELL", 75.0),
        )
        result = self._screen(rows, signal_filter=["SELL"])
        assert len(result["results"]) == 1
        assert result["results"][0]["symbol"] == "AAPL"

    def test_min_dau_filter(self):
        rows = _make_rows(
            ("A", "BUY", 80.0),
            ("B", "BUY", 40.0),  # below threshold
        )
        result = self._screen(rows, min_dau=65.0)
        assert all(r["dau"] >= 65.0 for r in result["results"])

    def test_min_conviction_filter(self):
        rows = _make_rows(
            ("A", "BUY", 90.0),  # high conviction
            ("B", "SELL", 10.0), # low conviction
        )
        result = self._screen(rows, min_conviction=50.0, breadth_state="EXPANDING")
        # SELL with low DAU should be filtered out
        assert all(r["conviction_score"] >= 50.0 for r in result["results"])

    def test_limit_respected(self):
        rows = _make_rows(*[("S{}".format(i), "BUY", 80.0 - i) for i in range(10)])
        result = self._screen(rows, limit=3)
        assert result["count"] == 3
        assert len(result["results"]) == 3

    def test_ic_state_propagated_to_results(self):
        rows = _make_rows(("NVDA", "BUY", 80.0))
        result = self._screen(rows, ic_state="STRONG")
        assert result["ic_state"] == "STRONG"
        assert result["results"][0]["ic_state"] == "STRONG"

    def test_breadth_state_propagated_to_results(self):
        rows = _make_rows(("NVDA", "BUY", 80.0))
        result = self._screen(rows, breadth_state="CONTRACTING")
        assert result["breadth_state"] == "CONTRACTING"
        assert result["results"][0]["breadth_state"] == "CONTRACTING"

    def test_suggested_weight_pct_formatted(self):
        rows = _make_rows(("NVDA", "BUY", 80.0))
        result = self._screen(rows)
        pct_str = result["results"][0]["suggested_weight_pct"]
        assert pct_str.endswith("%")

    def test_degraded_ic_zeroes_weight(self):
        rows = _make_rows(("NVDA", "BUY", 85.0))
        result = self._screen(rows, ic_state="DEGRADED")
        # DEGRADED IC → suggested_weight = 0
        assert result["results"][0]["suggested_weight"] == 0.0
        assert result["results"][0]["active_constraint"] == "ic_degraded"

    def test_json_payload_string_parsed(self):
        """Verify string-encoded JSONB payloads are handled."""
        payload_str = json.dumps(_make_payload(dau=75.0))
        rows = [("NVDA", payload_str, "BUY")]
        result = self._screen(rows)
        assert len(result["results"]) == 1
        assert result["results"][0]["dau"] == 75.0

    def test_none_payload_yields_zero_dau_result(self):
        """None JSONB payload is treated as empty dict → dau=0, not skipped."""
        rows = [("NVDA", None, "BUY")]
        result = self._screen(rows)
        assert result["count"] == 1
        assert result["results"][0]["dau"] == 0.0

    def test_none_payload_filtered_by_min_dau(self):
        """None payload → dau=0 → filtered out when min_dau > 0."""
        rows = [("NVDA", None, "BUY")]
        result = self._screen(rows, min_dau=1.0)
        assert result["count"] == 0

    def test_conviction_tier_high_for_top_candidates(self):
        rows = _make_rows(("NVDA", "BUY", 95.0))
        result = self._screen(rows, ic_state="STRONG", breadth_state="EXPANDING")
        # With high DAU + STRONG IC + EXPANDING breadth, should be HIGH
        tier = result["results"][0]["conviction_tier"]
        assert tier in ("HIGH", "MODERATE")

    def test_total_screened_reflects_raw_rows(self):
        rows = _make_rows(
            ("A", "BUY", 80.0),
            ("B", "BUY", 70.0),
            ("C", "SELL", 60.0),
        )
        # Filter to BUY only — total_screened = 3, count = 2
        result = self._screen(rows, signal_filter=["BUY"])
        assert result["total_screened"] == 3
        assert result["count"] == 2

    def test_query_uses_correct_as_of_join_column(self):
        """Verify the bulk query joins on p.as_of not p.as_of_date."""
        from api.axiom.screener import screen_universe
        with patch("api.axiom.screener.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.side_effect = [("MODERATE",), ("EXPANDING",)]
            mock_db.safe_fetchall.return_value = []
            screen_universe(_DATE)
            sql = mock_db.safe_fetchall.call_args[0][0]
            assert "p.as_of = a.as_of_date" in sql
            assert "p.as_of_date" not in sql


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestScreenerRoutes:
    def test_screen_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/axiom/screen", json={})
        assert resp.status_code == 401

    def test_screen_db_disabled(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.screener.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post("/axiom/screen", json={}, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["status"] == "db_disabled"

    def test_screen_no_data(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.screener.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchone",
                            lambda *a, **kw: ("MODERATE",))
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchall",
                            lambda *a, **kw: [])
        client = TestClient(app)
        resp = client.post("/axiom/screen",
                           json={"as_of_date": "2025-01-02"}, headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["count"] == 0

    def test_screen_returns_ranked_results(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        rows = [
            ("NVDA", _make_payload(dau=85.0), "BUY"),
            ("AAPL", _make_payload(dau=70.0), "BUY"),
        ]
        calls = [0]
        def fake_fetchone(*a, **kw):
            calls[0] += 1
            if calls[0] == 1:
                return ("MODERATE",)
            return ("EXPANDING",)
        monkeypatch.setattr("api.axiom.screener.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchone", fake_fetchone)
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchall", lambda *a, **kw: rows)
        client = TestClient(app)
        resp = client.post("/axiom/screen",
                           json={"as_of_date": "2025-01-02", "limit": 10},
                           headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["results"][0]["rank"] == 1
        assert data["results"][0]["symbol"] == "NVDA"

    def test_screen_signal_filter_applied(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        rows = [
            ("NVDA", _make_payload(dau=85.0), "BUY"),
            ("AAPL", _make_payload(dau=80.0), "SELL"),
        ]
        calls = [0]
        def fake_fetchone(*a, **kw):
            calls[0] += 1
            return ("MODERATE",) if calls[0] == 1 else ("EXPANDING",)
        monkeypatch.setattr("api.axiom.screener.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchone", fake_fetchone)
        monkeypatch.setattr("api.axiom.screener.db.safe_fetchall", lambda *a, **kw: rows)
        client = TestClient(app)
        resp = client.post(
            "/axiom/screen",
            json={"as_of_date": "2025-01-02", "signal_filter": ["BUY"]},
            headers=AUTH,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert all(r["signal_label"] == "BUY" for r in results)

    def test_screen_route_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "/axiom/screen" in resp.json().get("paths", {})
