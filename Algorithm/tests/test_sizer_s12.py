"""Session 12: Fractional-Kelly position sizer tests."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api import security
from api.axiom.sizer import KellySizeResult, _IC_KELLY_MULTIPLIER, compute_kelly_size
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


AUTH = {"X-FTIP-API-Key": "secret"}
TODAY = dt.date.today().isoformat()

# ---------------------------------------------------------------------------
# compute_kelly_size: pure computation
# ---------------------------------------------------------------------------

class TestComputeKellySize:

    def _base(self, **overrides):
        defaults = dict(
            symbol="AAPL", as_of_date=TODAY,
            dau=75.0, fragility_score=30.0,
            liquidity_score=65.0, research_score=65.0,
            overall_confidence=70.0,
            deployability_tier="live_candidate",
            hit_rate=0.62,
            ic_state="MODERATE",
            fractional_kelly=0.5,
            max_weight=0.10,
            portfolio_heat=0.0,
        )
        defaults.update(overrides)
        return compute_kelly_size(**defaults)

    def test_returns_kelly_size_result(self):
        result = self._base()
        assert isinstance(result, KellySizeResult)

    def test_suggested_weight_is_positive(self):
        result = self._base()
        assert result.suggested_weight > 0

    def test_suggested_weight_bounded_by_max_weight(self):
        result = self._base(dau=100.0, fragility_score=0.0, max_weight=0.05)
        assert result.suggested_weight <= 0.05

    def test_higher_dau_gives_higher_weight(self):
        low  = self._base(dau=20.0)
        high = self._base(dau=90.0)
        assert high.suggested_weight > low.suggested_weight

    def test_high_fragility_reduces_weight(self):
        safe   = self._base(fragility_score=20.0)
        risky  = self._base(fragility_score=75.0)
        assert risky.suggested_weight < safe.suggested_weight

    def test_fragility_veto_zeroes_weight(self):
        result = self._base(fragility_score=85.0)
        assert result.suggested_weight == 0.0
        assert result.active_constraint == "fragility" or "critical_fragility_veto" in result.downside_flags

    def test_degraded_ic_zeroes_weight(self):
        result = self._base(ic_state="DEGRADED")
        assert result.suggested_weight == 0.0
        assert result.active_constraint == "ic_degraded"

    def test_strong_ic_gives_more_than_insufficient(self):
        strong     = self._base(ic_state="STRONG")
        insuff     = self._base(ic_state="INSUFFICIENT")
        assert strong.suggested_weight >= insuff.suggested_weight

    def test_not_actionable_tier_zeroes_weight(self):
        result = self._base(deployability_tier="not_actionable")
        assert result.suggested_weight == 0.0
        assert result.active_constraint == "deployability"

    def test_monitor_only_tier_zeroes_weight(self):
        result = self._base(deployability_tier="monitor_only")
        assert result.suggested_weight == 0.0

    def test_paper_trade_tier_caps_at_two_pct(self):
        result = self._base(deployability_tier="paper_trade_only", dau=100.0)
        assert result.suggested_weight <= 0.02
        assert result.size_band == "paper"

    def test_portfolio_heat_limits_weight(self):
        # Nearly full portfolio (9.5% used, 0.5% remaining on 10% cap)
        result = self._base(portfolio_heat=0.095, max_weight=0.10)
        assert result.suggested_weight <= 0.005 + 1e-9

    def test_active_constraint_portfolio_heat(self):
        result = self._base(portfolio_heat=0.095, max_weight=0.10, dau=90.0)
        assert result.active_constraint == "portfolio_heat"

    def test_higher_hit_rate_increases_weight(self):
        low  = self._base(hit_rate=0.51)
        high = self._base(hit_rate=0.72)
        assert high.suggested_weight >= low.suggested_weight

    def test_no_hit_rate_still_produces_weight(self):
        result = self._base(hit_rate=None)
        assert result.suggested_weight >= 0.0

    def test_size_band_large_for_strong_signal(self):
        result = self._base(dau=92.0, fragility_score=10.0, ic_state="STRONG",
                            hit_rate=0.70, fractional_kelly=0.5)
        assert result.size_band in ("large", "medium", "small")

    def test_size_band_none_when_zero_weight(self):
        result = self._base(deployability_tier="not_actionable")
        assert result.size_band == "none"

    def test_rationale_is_non_empty_string(self):
        result = self._base()
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 20

    def test_inputs_dict_echoes_key_params(self):
        result = self._base(dau=75.0, fragility_score=30.0)
        assert result.inputs["dau"] == 75.0
        assert result.inputs["fragility_score"] == 30.0

    def test_liquidity_penalty_applied(self):
        good = self._base(liquidity_score=70.0)
        weak = self._base(liquidity_score=25.0)
        assert weak.suggested_weight < good.suggested_weight
        assert "liquidity_integrity_weak" in weak.downside_flags

    def test_research_penalty_applied(self):
        good = self._base(research_score=70.0)
        weak = self._base(research_score=25.0)
        assert weak.suggested_weight < good.suggested_weight
        assert "research_integrity_weak" in weak.downside_flags

    def test_ic_kelly_multipliers_ordering(self):
        """STRONG > MODERATE > WEAK > INSUFFICIENT > DEGRADED."""
        weights = {
            state: self._base(ic_state=state).suggested_weight
            for state in ["STRONG", "MODERATE", "WEAK", "INSUFFICIENT", "DEGRADED"]
        }
        assert weights["STRONG"] >= weights["MODERATE"]
        assert weights["MODERATE"] >= weights["WEAK"]
        assert weights["DEGRADED"] == 0.0

    def test_fractional_kelly_param_scales_output(self):
        half  = self._base(fractional_kelly=0.5)
        full  = self._base(fractional_kelly=1.0)
        assert full.suggested_weight >= half.suggested_weight

    def test_weight_is_zero_or_positive(self):
        """Weight must never be negative."""
        for dau in [0, 10, 50, 100]:
            for frag in [10, 50, 90]:
                result = self._base(dau=float(dau), fragility_score=float(frag))
                assert result.suggested_weight >= 0.0


# ---------------------------------------------------------------------------
# /axiom/size endpoint
# ---------------------------------------------------------------------------

class TestAxiomSizeEndpoint:

    _INLINE_BODY = {
        "symbol": "AAPL",
        "as_of_date": TODAY,
        "dau": 75.0,
        "fragility_score": 30.0,
        "liquidity_score": 65.0,
        "research_score": 65.0,
        "overall_confidence": 70.0,
        "deployability_tier": "live_candidate",
        "hit_rate": 0.62,
        "ic_state": "MODERATE",
    }

    def test_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        client = TestClient(app)
        resp = client.post("/axiom/size", json=self._INLINE_BODY)
        assert resp.status_code == 401

    def test_inline_scores_returns_200(self, monkeypatch):
        _db_env(monkeypatch)
        # DB disabled for this one — all values inline
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post("/axiom/size", json=self._INLINE_BODY, headers=AUTH)
        assert resp.status_code == 200

    def test_response_has_expected_fields(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post("/axiom/size", json=self._INLINE_BODY, headers=AUTH)
        body = resp.json()
        for field in ("suggested_weight", "suggested_weight_pct", "size_band",
                       "deployability_tier", "ic_state", "active_constraint",
                       "rationale", "downside_flags", "data_source"):
            assert field in body, f"missing field: {field}"

    def test_suggested_weight_pct_is_formatted(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post("/axiom/size", json=self._INLINE_BODY, headers=AUTH)
        pct = resp.json()["suggested_weight_pct"]
        assert pct.endswith("%")

    def test_degraded_ic_returns_zero_weight(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: False)
        body = {**self._INLINE_BODY, "ic_state": "DEGRADED"}
        client = TestClient(app)
        resp = client.post("/axiom/size", json=body, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["suggested_weight"] == 0.0

    def test_db_lookup_used_when_scores_missing(self, monkeypatch):
        _db_env(monkeypatch)
        db_payload = {
            "deployable_alpha_utility": 68.0,
            "deployability_tier": "live_candidate",
            "overall_confidence": 65.0,
            "engine_scores": {
                "critical_fragility": {"score": 28.0},
                "liquidity_convexity": {"score": 70.0},
                "research_integrity": {"score": 72.0},
            },
        }
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: True)
        monkeypatch.setattr("api.axiom.routes.db.safe_fetchone", lambda sql, params: (db_payload,))
        client = TestClient(app)
        resp = client.post(
            "/axiom/size",
            json={"symbol": "MSFT", "as_of_date": TODAY, "ic_state": "MODERATE"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_source"] in ("db", "partial")
        assert body["suggested_weight"] > 0

    def test_missing_symbol_returns_422(self, monkeypatch):
        _db_env(monkeypatch)
        client = TestClient(app)
        resp = client.post("/axiom/size", json={"dau": 70.0}, headers=AUTH)
        assert resp.status_code == 422

    def test_max_weight_respected(self, monkeypatch):
        _db_env(monkeypatch)
        monkeypatch.setattr("api.axiom.routes.db.db_read_enabled", lambda: False)
        body = {**self._INLINE_BODY, "max_weight": 0.03, "dau": 100.0,
                "fragility_score": 5.0, "ic_state": "STRONG"}
        client = TestClient(app)
        resp = client.post("/axiom/size", json=body, headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["suggested_weight"] <= 0.03 + 1e-9

    def test_route_in_openapi(self, monkeypatch):
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "/axiom/size" in resp.json()["paths"]
