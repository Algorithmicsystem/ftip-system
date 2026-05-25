"""Session 15: Structured IC memo with lineage hash — tests."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from unittest.mock import patch

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
# Lineage hash: pure determinism tests
# ---------------------------------------------------------------------------

class TestLineageHash:
    def _canonical(self, **overrides):
        from api.axiom.memo import build_canonical_inputs
        defaults = dict(
            symbol="NVDA", as_of_date="2025-01-02",
            dau=78.5, fragility_score=40.0, liquidity_score=65.0,
            research_score=60.0, overall_confidence=70.0,
            deployability_tier="live_candidate", ic_state="MODERATE",
            hit_rate=0.62, fractional_kelly=0.5, max_weight=0.10,
            signal_label="BUY", regime_label="fundamental_convergence",
            breadth_state="EXPANDING", conviction_score=82.5,
            suggested_weight=0.048,
        )
        defaults.update(overrides)
        return build_canonical_inputs(**defaults)

    def test_same_inputs_same_hash(self):
        from api.axiom.memo import compute_lineage_hash
        c1 = self._canonical()
        c2 = self._canonical()
        assert compute_lineage_hash(c1) == compute_lineage_hash(c2)

    def test_different_dau_different_hash(self):
        from api.axiom.memo import compute_lineage_hash
        h1 = compute_lineage_hash(self._canonical(dau=78.5))
        h2 = compute_lineage_hash(self._canonical(dau=79.0))
        assert h1 != h2

    def test_different_symbol_different_hash(self):
        from api.axiom.memo import compute_lineage_hash
        h1 = compute_lineage_hash(self._canonical(symbol="NVDA"))
        h2 = compute_lineage_hash(self._canonical(symbol="AAPL"))
        assert h1 != h2

    def test_hash_is_sha256_hex(self):
        from api.axiom.memo import compute_lineage_hash
        h = compute_lineage_hash(self._canonical())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_symbol_case_normalized(self):
        from api.axiom.memo import build_canonical_inputs, compute_lineage_hash
        c_lower = self._canonical(symbol="nvda")
        c_upper = self._canonical(symbol="NVDA")
        assert c_lower["symbol"] == "NVDA"
        assert compute_lineage_hash(c_lower) == compute_lineage_hash(c_upper)

    def test_schema_version_included_in_hash(self):
        from api.axiom.memo import build_canonical_inputs, compute_lineage_hash, SCHEMA_VERSION
        canonical = self._canonical()
        assert canonical["schema_version"] == SCHEMA_VERSION
        # Mutating schema_version changes the hash
        c_modified = dict(canonical)
        c_modified["schema_version"] = "0.0"
        assert compute_lineage_hash(canonical) != compute_lineage_hash(c_modified)

    def test_none_hit_rate_preserved(self):
        from api.axiom.memo import build_canonical_inputs
        c = self._canonical(hit_rate=None)
        assert c["hit_rate"] is None

    def test_float_rounding_determinism(self):
        from api.axiom.memo import build_canonical_inputs, compute_lineage_hash
        # Tiny floating point differences that round to the same value should hash identically
        c1 = self._canonical(dau=78.50000000001)
        c2 = self._canonical(dau=78.50000000002)
        # Both round to 78.5 at 4dp
        assert compute_lineage_hash(c1) == compute_lineage_hash(c2)


# ---------------------------------------------------------------------------
# conviction_tier mapping
# ---------------------------------------------------------------------------

class TestConvictionTier:
    def test_high_at_75(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(75.0) == "HIGH"

    def test_high_above_75(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(99.9) == "HIGH"

    def test_moderate_at_50(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(50.0) == "MODERATE"

    def test_moderate_between_50_and_75(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(60.0) == "MODERATE"

    def test_low_at_25(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(25.0) == "LOW"

    def test_insufficient_below_25(self):
        from api.axiom.memo import conviction_tier
        assert conviction_tier(0.0) == "INSUFFICIENT"
        assert conviction_tier(24.9) == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# build_memo: structure checks
# ---------------------------------------------------------------------------

class TestBuildMemo:
    def _build(self, **overrides):
        from api.axiom.memo import build_memo
        defaults = dict(
            symbol="NVDA", as_of_date="2025-01-02",
            dau=78.5, fragility_score=40.0, liquidity_score=65.0,
            research_score=60.0, overall_confidence=70.0,
            deployability_tier="live_candidate", ic_state="MODERATE",
            hit_rate=0.62, fractional_kelly=0.5, max_weight=0.10,
            signal_label="BUY", regime_label="fundamental_convergence",
            breadth_state="EXPANDING", conviction_score=82.5,
            suggested_weight=0.048, kelly_gross_weight=0.096,
            fractional_kelly_applied=0.175, ic_kelly_multiplier=0.35,
            fragility_penalty_applied=0.0, active_constraint="kelly",
            size_band="medium", downside_flags=[], rationale="Test.",
            data_source="db",
        )
        defaults.update(overrides)
        return build_memo(**defaults)

    def test_returns_axiom_memo(self):
        from api.axiom.memo import AxiomMemo
        memo = self._build()
        assert isinstance(memo, AxiomMemo)

    def test_lineage_hash_present(self):
        memo = self._build()
        assert len(memo.lineage_hash) == 64

    def test_memo_id_is_uuid(self):
        import re
        memo = self._build()
        assert re.match(r"[0-9a-f-]{36}", memo.memo_id)

    def test_memo_body_has_required_sections(self):
        memo = self._build()
        body = memo.memo_body
        for key in ("executive_summary", "signal_context", "axiom_scorecard",
                    "conviction_analysis", "position_sizing", "ic_calibration",
                    "risk_flags", "compliance_attestation", "lineage_hash"):
            assert key in body, f"Missing section: {key}"

    def test_lineage_hash_in_body_matches_memo(self):
        memo = self._build()
        assert memo.memo_body["lineage_hash"] == memo.lineage_hash

    def test_symbol_normalized_to_upper(self):
        memo = self._build(symbol="nvda")
        assert memo.symbol == "NVDA"
        assert memo.memo_body["symbol"] == "NVDA"

    def test_conviction_tier_in_executive_summary(self):
        memo = self._build(conviction_score=82.5)
        assert memo.memo_body["executive_summary"]["conviction_tier"] == "HIGH"

    def test_headline_contains_symbol_and_signal(self):
        memo = self._build(symbol="AAPL", signal_label="SELL")
        assert "AAPL" in memo.memo_body["executive_summary"]["headline"]
        assert "SELL" in memo.memo_body["executive_summary"]["headline"]

    def test_risk_flags_include_insufficient_conviction(self):
        memo = self._build(conviction_score=10.0)
        assert "insufficient_conviction" in memo.memo_body["risk_flags"]

    def test_risk_flags_include_regime_unclassified(self):
        memo = self._build(regime_label="unknown")
        assert "regime_unclassified" in memo.memo_body["risk_flags"]

    def test_engine_scores_included_when_provided(self):
        engines = {"fundamental_reality": {"score": 0.7}}
        memo = self._build(engine_scores=engines)
        assert memo.memo_body["axiom_scorecard"]["engine_scores"] == engines

    def test_engine_scores_absent_when_none(self):
        memo = self._build(engine_scores=None)
        assert "engine_scores" not in memo.memo_body["axiom_scorecard"]

    def test_two_memos_same_inputs_same_hash(self):
        m1 = self._build()
        m2 = self._build()
        assert m1.lineage_hash == m2.lineage_hash

    def test_two_memos_same_inputs_different_memo_ids(self):
        m1 = self._build()
        m2 = self._build()
        assert m1.memo_id != m2.memo_id

    def test_compliance_attestation_present(self):
        memo = self._build()
        assert "FTIP AXIOM" in memo.memo_body["compliance_attestation"]


# ---------------------------------------------------------------------------
# store_memo / load_memo
# ---------------------------------------------------------------------------

class TestMemoStorage:
    def _make_memo(self):
        from api.axiom.memo import build_memo
        return build_memo(
            symbol="NVDA", as_of_date="2025-01-02",
            dau=78.5, fragility_score=40.0, liquidity_score=65.0,
            research_score=60.0, overall_confidence=70.0,
            deployability_tier="live_candidate", ic_state="MODERATE",
            hit_rate=None, fractional_kelly=0.5, max_weight=0.10,
            signal_label="BUY", regime_label="fundamental_convergence",
            breadth_state="EXPANDING", conviction_score=82.5,
            suggested_weight=0.048, kelly_gross_weight=0.096,
            fractional_kelly_applied=0.175, ic_kelly_multiplier=0.35,
            fragility_penalty_applied=0.0, active_constraint="kelly",
            size_band="medium", downside_flags=[], rationale=".",
            data_source="db",
        )

    def test_store_calls_safe_execute(self):
        from api.axiom.memo import store_memo
        memo = self._make_memo()
        with patch("api.axiom.memo.db") as mock_db:
            mock_db.safe_execute.return_value = None
            result = store_memo(memo)
        assert result is True
        assert mock_db.safe_execute.called

    def test_store_exception_returns_false(self):
        from api.axiom.memo import store_memo
        memo = self._make_memo()
        with patch("api.axiom.memo.db") as mock_db:
            mock_db.safe_execute.side_effect = Exception("db error")
            result = store_memo(memo)
        assert result is False

    def test_load_by_id_db_disabled_returns_none(self):
        from api.axiom.memo import load_memo_by_id
        with patch("api.axiom.memo.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            assert load_memo_by_id("some-uuid") is None

    def test_load_by_id_not_found_returns_none(self):
        from api.axiom.memo import load_memo_by_id
        with patch("api.axiom.memo.db") as mock_db:
            mock_db.db_read_enabled.return_value = True
            mock_db.safe_fetchone.return_value = None
            assert load_memo_by_id("nonexistent") is None

    def test_load_by_hash_db_disabled_returns_none(self):
        from api.axiom.memo import load_memo_by_hash
        with patch("api.axiom.memo.db") as mock_db:
            mock_db.db_read_enabled.return_value = False
            assert load_memo_by_hash("abc123") is None


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------

class TestMemoRoutes:
    def test_memo_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.post("/axiom/memo", json={"symbol": "NVDA"})
        assert resp.status_code == 401

    def test_memo_by_id_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/axiom/memo/test-uuid")
        assert resp.status_code == 401

    def test_memo_by_hash_requires_auth(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/axiom/memo/verify/abc123")
        assert resp.status_code == 401

    def test_memo_generates_inline(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.routes._load_axiom_scores", lambda *a: None)
        monkeypatch.setattr("api.axiom.routes._load_ic_state", lambda *a: "MODERATE")
        monkeypatch.setattr("api.axiom.routes._load_hit_rate", lambda *a: None)
        monkeypatch.setattr("api.axiom.routes._load_signal_context", lambda *a: {})
        monkeypatch.setattr("api.axiom.routes._load_breadth_state", lambda *a: "NEUTRAL")
        monkeypatch.setattr("api.axiom.memo.db.db_write_enabled", lambda: False)
        client = TestClient(app)
        resp = client.post(
            "/axiom/memo",
            json={
                "symbol": "NVDA",
                "as_of_date": "2025-01-02",
                "dau": 72.0,
                "fragility_score": 35.0,
                "deployability_tier": "live_candidate",
                "signal_label": "BUY",
                "store": False,
            },
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "lineage_hash" in data
        assert len(data["lineage_hash"]) == 64
        assert "memo" in data
        assert "executive_summary" in data["memo"]

    def test_memo_lineage_hash_deterministic(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.routes._load_axiom_scores", lambda *a: None)
        monkeypatch.setattr("api.axiom.routes._load_ic_state", lambda *a: "MODERATE")
        monkeypatch.setattr("api.axiom.routes._load_hit_rate", lambda *a: None)
        monkeypatch.setattr("api.axiom.routes._load_signal_context", lambda *a: {})
        monkeypatch.setattr("api.axiom.routes._load_breadth_state", lambda *a: "NEUTRAL")
        monkeypatch.setattr("api.axiom.memo.db.db_write_enabled", lambda: False)
        client = TestClient(app)
        body = {
            "symbol": "TSLA", "as_of_date": "2025-01-02",
            "dau": 70.0, "fragility_score": 42.0,
            "deployability_tier": "live_candidate",
            "signal_label": "BUY", "store": False,
        }
        r1 = client.post("/axiom/memo", json=body, headers=AUTH).json()
        r2 = client.post("/axiom/memo", json=body, headers=AUTH).json()
        assert r1["lineage_hash"] == r2["lineage_hash"]
        assert r1["memo"]["memo_id"] != r2["memo"]["memo_id"]

    def test_memo_by_id_404_when_not_found(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.routes.load_memo_by_id", lambda *a: None)
        client = TestClient(app)
        resp = client.get("/axiom/memo/nonexistent-uuid", headers=AUTH)
        assert resp.status_code == 404

    def test_memo_by_hash_404_when_not_found(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        monkeypatch.setattr("api.axiom.routes.load_memo_by_hash", lambda *a: None)
        client = TestClient(app)
        resp = client.get("/axiom/memo/verify/deadbeef", headers=AUTH)
        assert resp.status_code == 404

    def test_memo_by_id_returns_stored(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        stored = {"memo_id": "test-uuid", "symbol": "NVDA", "lineage_hash": "abc"}
        monkeypatch.setattr("api.axiom.routes.load_memo_by_id", lambda *a: stored)
        client = TestClient(app)
        resp = client.get("/axiom/memo/test-uuid", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["memo"]["symbol"] == "NVDA"

    def test_memo_by_hash_verified(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        stored = {"memo_id": "test-uuid", "symbol": "NVDA", "lineage_hash": "deadbeef"}
        monkeypatch.setattr("api.axiom.routes.load_memo_by_hash", lambda *a: stored)
        client = TestClient(app)
        resp = client.get("/axiom/memo/verify/deadbeef", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["verified"] is True
        assert data["memo"]["lineage_hash"] == "deadbeef"

    def test_routes_registered_in_openapi(self, monkeypatch):
        _db_env(monkeypatch)
        from api.main import app
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/axiom/memo" in paths
        assert "/axiom/memo/{memo_id}" in paths
        assert "/axiom/memo/verify/{lineage_hash}" in paths
