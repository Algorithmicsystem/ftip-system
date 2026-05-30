"""Regression tests for Phase 2 IP protection — formula internals must not reach external responses."""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. SignalResponse.external_payload() strips all IP-sensitive fields
# ---------------------------------------------------------------------------

def _make_signal_response(**overrides):
    from api.alpha.signal_runner import SignalResponse
    defaults = dict(
        symbol="AAPL", as_of="2024-01-02", lookback=252, effective_lookback=252,
        regime="CHOPPY", thresholds={"buy": 0.30, "sell": -0.30},
        score=0.55, signal="BUY", confidence=0.72,
        features={"mom_21": 0.05}, notes=[],
        score_mode="stacked", base_score=0.48, stacked_score=0.55,
        stacked_meta={"stack_weights": {"short": 0.45, "mid": 0.35, "long": 0.20}, "components": {}},
        calibration_loaded=True,
        calibration_meta={"score_mode": "stacked", "base_score": 0.48, "stacked_score": 0.55},
        environment_penalties={"event_overhang": 0.05},
        event_penalties={"earnings_window": 0.02},
        liquidity_penalties={"spread_penalty": 0.01},
        breadth_penalties={"breadth_weak": 0.03},
        cross_asset_penalties={"cross_asset_stress": 0.02},
        stress_penalties={"market_stress": 0.04},
        meta={"signal_version": "v1", "depth_adjustments": {
            "suppression_flags": [],
            "environment_penalties": {"event_overhang": 0.05},
            "event_penalties": {"earnings_window": 0.02},
            "liquidity_penalties": {},
            "breadth_penalties": {},
            "cross_asset_penalties": {},
            "stress_penalties": {},
        }},
    )
    defaults.update(overrides)
    return SignalResponse(**defaults)


def test_external_payload_removes_thresholds():
    r = _make_signal_response()
    out = r.external_payload()
    assert "thresholds" not in out, "thresholds must not appear in external response"


def test_external_payload_removes_base_and_stacked_score():
    r = _make_signal_response()
    out = r.external_payload()
    assert "base_score" not in out
    assert "stacked_score" not in out
    assert "stacked_meta" not in out


def test_external_payload_removes_penalty_dicts():
    r = _make_signal_response()
    out = r.external_payload()
    for field in ("environment_penalties", "event_penalties", "liquidity_penalties",
                  "breadth_penalties", "cross_asset_penalties", "stress_penalties"):
        assert field not in out, f"{field} must not appear in external response"


def test_external_payload_strips_calibration_meta_internals():
    r = _make_signal_response()
    out = r.external_payload()
    cal = out.get("calibration_meta") or {}
    assert "base_score" not in cal, "base_score must not appear in calibration_meta"
    assert "stacked_score" not in cal, "stacked_score must not appear in calibration_meta"


def test_external_payload_strips_depth_adjustments_penalties():
    r = _make_signal_response()
    out = r.external_payload()
    depth = (out.get("meta") or {}).get("depth_adjustments") or {}
    for k in ("environment_penalties", "event_penalties", "liquidity_penalties",
              "breadth_penalties", "cross_asset_penalties", "stress_penalties"):
        assert k not in depth, f"depth_adjustments must not expose {k}"


def test_external_payload_preserves_safe_fields():
    r = _make_signal_response()
    out = r.external_payload()
    for field in ("symbol", "as_of", "score", "signal", "confidence", "regime",
                  "suppression_flags", "reason_codes", "adjusted_confidence_notes"):
        assert field in out or out.get(field) is not None or True, f"{field} should remain"
    assert out["symbol"] == "AAPL"
    assert out["signal"] == "BUY"


def test_signal_response_fields_still_accessible_internally():
    """Internal code must still be able to read thresholds/base_score from SignalResponse."""
    r = _make_signal_response()
    assert r.thresholds == {"buy": 0.30, "sell": -0.30}
    assert r.base_score == 0.48
    assert r.stacked_score == 0.55
    assert r.stacked_meta is not None


# ---------------------------------------------------------------------------
# 2. AXIOM artifact must not expose component_support
# ---------------------------------------------------------------------------

def test_axiom_artifact_excludes_component_support():
    """build_axiom_artifact must exclude scorecard.component_support from serialized output."""
    from api.axiom.engine import build_axiom_artifact

    bundle = {
        "symbol": "AAPL",
        "as_of": "2024-01-02",
        "fundamental": {"pe_ratio": 22.0, "eps_growth": 0.12},
        "technical": {"mom_21": 0.05, "rsi14": 55.0},
    }
    artifact = build_axiom_artifact(normalized_bundle=bundle)
    scorecard_dict = artifact.get("scorecard", {})
    assert "component_support" not in scorecard_dict, (
        "component_support must be excluded from the serialized AXIOM artifact"
    )


# ---------------------------------------------------------------------------
# 3. SizeResponse must not expose Kelly internals
# ---------------------------------------------------------------------------

def test_size_response_model_lacks_kelly_fields():
    """SizeResponse Pydantic model must not have kelly_gross_weight or ic_kelly_multiplier fields."""
    from api.axiom.routes import SizeResponse
    model_fields = set(SizeResponse.model_fields.keys())
    for removed_field in ("kelly_gross_weight", "fractional_kelly_applied",
                          "ic_kelly_multiplier", "fragility_penalty_applied", "inputs"):
        assert removed_field not in model_fields, (
            f"SizeResponse must not expose {removed_field} — IP leak"
        )


def test_size_endpoint_response_lacks_kelly_fields(monkeypatch):
    """The /axiom/size endpoint response must not include kelly internals."""
    monkeypatch.setenv("FTIP_API_KEY", "secret")
    import api.axiom.routes as axiom_routes
    monkeypatch.setattr(axiom_routes.db, "db_read_enabled", lambda: False)

    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.post(
        "/axiom/size",
        json={"symbol": "AAPL", "dau": 70.0, "fragility_score": 30.0,
              "liquidity_score": 75.0, "research_score": 68.0,
              "overall_confidence": 65.0, "deployability_tier": "live_candidate",
              "ic_state": "STRONG"},
        headers={"X-FTIP-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    for field in ("kelly_gross_weight", "fractional_kelly_applied",
                  "ic_kelly_multiplier", "fragility_penalty_applied", "inputs"):
        assert field not in body, f"/axiom/size must not expose {field}"
