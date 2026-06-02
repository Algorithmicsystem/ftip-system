"""Phase 36: Production Hardening tests.

Covers:
  1.  macro_context.get_vix_level() returns None when FRED unavailable
  2.  macro_context.get_cardi_inputs() returns correct key structure
  3.  get_market_bubble_context with vix=35 → distress
  4.  get_market_bubble_context with vix=12 → euphoria
  5.  fragility breakdown includes mtrs_score
  6.  state_pricing breakdown includes cardi_score
  7.  sanitizer strips internal IP fields
  8.  sanitizer preserves score/confidence/coverage
  9.  trial endpoint with db disabled → db_disabled
  10. trial endpoint with db mocked → api_key and getting_started
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1 — macro_context: get_vix_level returns None without raising when FRED fails
# ---------------------------------------------------------------------------

class TestMacroContextVixDefaults:
    def test_vix_returns_none_when_fred_unavailable(self):
        from api.jobs.macro_context import get_vix_level, _cache
        # Clear cache so a fresh fetch is attempted
        _cache.clear()
        with patch("api.jobs.macro_context._fetch_latest", return_value=None):
            result = get_vix_level()
        assert result is None

    def test_vix_does_not_raise_on_exception(self):
        from api.jobs.macro_context import get_vix_level, _cache
        _cache.clear()
        with patch("api.jobs.macro_context._fetch_latest", side_effect=RuntimeError("network error")):
            result = get_vix_level()
        assert result is None


# ---------------------------------------------------------------------------
# 2 — macro_context: get_cardi_inputs returns correct structure
# ---------------------------------------------------------------------------

class TestMacroContextCardiInputsStructure:
    def test_has_required_keys(self):
        from api.jobs.macro_context import get_cardi_inputs, _cache
        _cache.clear()
        with patch("api.jobs.macro_context.get_term_spread", return_value=None):
            result = get_cardi_inputs()
        assert "carry_score" in result
        assert "value_score" in result
        assert "momentum_score" in result
        assert "defensive_score" in result

    def test_positive_spread_gives_carry_above_50(self):
        from api.jobs.macro_context import get_cardi_inputs, _cache
        _cache.clear()
        with patch("api.jobs.macro_context.get_term_spread", return_value=2.0):
            result = get_cardi_inputs()
        assert result["carry_score"] is not None
        assert result["carry_score"] > 50.0

    def test_inverted_curve_gives_carry_below_50(self):
        from api.jobs.macro_context import get_cardi_inputs, _cache
        _cache.clear()
        with patch("api.jobs.macro_context.get_term_spread", return_value=-0.5):
            result = get_cardi_inputs()
        assert result["carry_score"] is not None
        assert result["carry_score"] < 50.0

    def test_value_and_defensive_are_none(self):
        from api.jobs.macro_context import get_cardi_inputs, _cache
        _cache.clear()
        with patch("api.jobs.macro_context.get_term_spread", return_value=1.0):
            result = get_cardi_inputs()
        assert result["value_score"] is None
        assert result["defensive_score"] is None


# ---------------------------------------------------------------------------
# 3 — macro_context: vix=35 → distress
# ---------------------------------------------------------------------------

class TestMacroContextBubbleContextHighVix:
    def test_vix_35_returns_distress(self):
        from api.jobs.macro_context import get_market_bubble_context
        result = get_market_bubble_context(vix=35.0)
        assert result["kindleberger_stage"] == "distress"
        assert result["narrative_intensity"] == 75.0
        assert result["cape_z_score"] is None


# ---------------------------------------------------------------------------
# 4 — macro_context: vix=12 → euphoria
# ---------------------------------------------------------------------------

class TestMacroContextBubbleContextLowVix:
    def test_vix_12_returns_euphoria(self):
        from api.jobs.macro_context import get_market_bubble_context
        result = get_market_bubble_context(vix=12.0)
        assert result["kindleberger_stage"] == "euphoria"
        assert result["narrative_intensity"] == 35.0

    def test_vix_25_returns_boom(self):
        from api.jobs.macro_context import get_market_bubble_context
        result = get_market_bubble_context(vix=25.0)
        assert result["kindleberger_stage"] == "boom"

    def test_vix_18_returns_normal(self):
        from api.jobs.macro_context import get_market_bubble_context
        result = get_market_bubble_context(vix=18.0)
        assert result["kindleberger_stage"] == "normal"


# ---------------------------------------------------------------------------
# 5 — fragility breakdown includes mtrs_score
# ---------------------------------------------------------------------------

def _make_minimal_engine_input():
    """Build a minimal AxiomEngineInput for engine unit tests."""
    from api.axiom.contracts import (
        AxiomEngineInput,
        AxiomSupportContext,
        FragilityCandidateInputs,
        FundamentalCandidateInputs,
    )
    fragility = FragilityCandidateInputs(
        realized_vol_21d=0.20,
        realized_vol_63d=0.22,
        vol_of_vol_proxy=0.20,
        gap_pct=0.01,
        gap_instability_10d=0.25,
        abs_gap_mean_10d=0.012,
        return_dispersion_21d=0.012,
        return_dispersion_63d=0.014,
        downside_asymmetry_21d=0.95,
        downside_asymmetry_63d=0.94,
        maxdd_21d=-0.06,
        maxdd_63d=-0.12,
        maxdd_126d=-0.18,
        event_overhang_score=20.0,
        event_uncertainty_score=25.0,
        implementation_fragility_score=22.0,
        liquidity_quality_score=80.0,
        tradability_caution_score=24.0,
        overnight_gap_risk_score=18.0,
        friction_proxy_score=22.0,
        breadth_confirmation_score=70.0,
        cross_asset_conflict_score=22.0,
        market_stress_score=20.0,
        instability_score=22.0,
        volatility_stress_score=20.0,
        drawdown_sensitivity_score=22.0,
        anomaly_pressure_score=22.0,
        narrative_crowding_score=30.0,
        signal_fragility_score=20.0,
        regime_transition_score=22.0,
        regime_instability_score=22.0,
        coverage_score=70.0,
        provider_confidence=72.0,
    )
    fundamental = FundamentalCandidateInputs(
        coverage_score=70.0,
        provider_confidence=72.0,
    )
    support = AxiomSupportContext()
    return AxiomEngineInput(
        framework_version="axiom50_phase2_v1",
        symbol="TEST",
        as_of="2026-06-02",
        source_context={
            "market_bubble_context": {},
            "macro_cardi_inputs": {},
            "symbol_meta": {},
        },
        fundamental=fundamental,
        fragility=fragility,
        support=support,
        domain_coverage={},
        partial_engine_hints={},
        warnings=[],
    )


class TestFragilityMtrsInBreakdown:
    def test_mtrs_score_in_components(self):
        from api.axiom.engines import score_critical_fragility
        engine_input = _make_minimal_engine_input()
        result = score_critical_fragility(engine_input)
        assert "mtrs_score" in result.components

    def test_mtrs_score_is_not_none_by_default(self):
        from api.axiom.engines import score_critical_fragility
        engine_input = _make_minimal_engine_input()
        result = score_critical_fragility(engine_input)
        # Default 50.0 when no return_series
        assert result.components["mtrs_score"] == 50.0


# ---------------------------------------------------------------------------
# 6 — state_pricing breakdown includes cardi_score
# ---------------------------------------------------------------------------

class TestStatePricingCardiInBreakdown:
    def test_cardi_score_key_present_when_inputs_provided(self):
        from api.axiom.engines import score_state_pricing
        engine_input = _make_minimal_engine_input()
        # Provide real macro_cardi_inputs so cardi_score is computed (non-None)
        engine_input.source_context["macro_cardi_inputs"] = {
            "carry_score": 60.0,
            "value_score": 55.0,
            "momentum_score": 65.0,
            "defensive_score": 50.0,
        }
        result = score_state_pricing(engine_input)
        assert "cardi_score" in result.components

    def test_cardi_score_absent_when_no_macro_cardi_inputs(self):
        from api.axiom.engines import score_state_pricing
        engine_input = _make_minimal_engine_input()
        # Empty macro_cardi_inputs → compute_cardi not called → cardi_score=None
        # None is filtered from components dict per EngineScore construction
        engine_input.source_context["macro_cardi_inputs"] = {}
        result = score_state_pricing(engine_input)
        # cardi_score should NOT appear in components when it is None
        assert result.components.get("cardi_score") is None


# ---------------------------------------------------------------------------
# 7 — sanitizer strips internal IP fields
# ---------------------------------------------------------------------------

class TestSanitizerStripsIpFields:
    def _make_payload(self) -> Dict:
        return {
            "symbol": "AAPL",
            "composite_score": 72.5,
            "engine_scores": {
                "critical_fragility": {
                    "score": 35.0,
                    "confidence": 71.0,
                    "coverage": 68.0,
                    "components": {
                        "scps_component": 42.0,
                        "mtrs_score": 50.0,
                        "bfs_component": 38.0,
                        "volatility_instability_component": 40.0,
                    },
                },
                "state_pricing": {
                    "score": 65.0,
                    "confidence": 70.0,
                    "coverage": 65.0,
                    "components": {
                        "cardi_score": 55.0,
                        "caps_component": 60.0,
                        "macro_alignment_component": 62.0,
                    },
                },
            },
        }

    def test_strips_scps_component(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown(self._make_payload())
        comps = result["engine_scores"]["critical_fragility"]["components"]
        assert "scps_component" not in comps

    def test_strips_mtrs_score(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown(self._make_payload())
        comps = result["engine_scores"]["critical_fragility"]["components"]
        assert "mtrs_score" not in comps

    def test_strips_bfs_component(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown(self._make_payload())
        comps = result["engine_scores"]["critical_fragility"]["components"]
        assert "bfs_component" not in comps

    def test_strips_cardi_score(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown(self._make_payload())
        comps = result["engine_scores"]["state_pricing"]["components"]
        assert "cardi_score" not in comps

    def test_strips_caps_component(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown(self._make_payload())
        comps = result["engine_scores"]["state_pricing"]["components"]
        assert "caps_component" not in comps

    def test_strips_kle_score(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        payload = self._make_payload()
        payload["engine_scores"]["research_integrity"] = {
            "score": 70.0,
            "confidence": 72.0,
            "coverage": 70.0,
            "components": {"kle_score": 65.0, "other_component": 55.0},
        }
        result = sanitize_engine_breakdown(payload)
        comps = result["engine_scores"]["research_integrity"]["components"]
        assert "kle_score" not in comps


# ---------------------------------------------------------------------------
# 8 — sanitizer preserves score/confidence/coverage
# ---------------------------------------------------------------------------

class TestSanitizerPreservesScores:
    def test_preserves_score(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        payload = {
            "engine_scores": {
                "critical_fragility": {
                    "score": 35.0,
                    "confidence": 71.0,
                    "coverage": 68.0,
                    "components": {"scps_component": 42.0},
                }
            }
        }
        result = sanitize_engine_breakdown(payload)
        engine = result["engine_scores"]["critical_fragility"]
        assert engine["score"] == 35.0
        assert engine["confidence"] == 71.0
        assert engine["coverage"] == 68.0

    def test_preserves_non_ip_components(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        payload = {
            "engine_scores": {
                "critical_fragility": {
                    "score": 35.0,
                    "confidence": 71.0,
                    "coverage": 68.0,
                    "components": {
                        "scps_component": 42.0,
                        "volatility_instability_component": 40.0,
                        "gap_jump_risk_component": 35.0,
                    },
                }
            }
        }
        result = sanitize_engine_breakdown(payload)
        comps = result["engine_scores"]["critical_fragility"]["components"]
        assert "volatility_instability_component" in comps
        assert "gap_jump_risk_component" in comps

    def test_preserves_composite_score(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        payload = {"composite_score": 72.5, "engine_scores": {}}
        result = sanitize_engine_breakdown(payload)
        assert result["composite_score"] == 72.5

    def test_empty_payload_does_not_raise(self):
        from api.axiom.sanitizer import sanitize_engine_breakdown
        result = sanitize_engine_breakdown({})
        assert result == {"engine_scores": {}}


# ---------------------------------------------------------------------------
# 9 — trial endpoint with db disabled → db_disabled
# ---------------------------------------------------------------------------

class TestTrialEndpointNoDb:
    def test_returns_db_disabled_status(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=False):
            req = TrialRequest(org_name="Test Org", contact_email="test@example.com")
            result = create_trial(req)
        assert result["status"] == "db_disabled"

    def test_still_returns_api_key(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=False):
            req = TrialRequest(org_name="Test Org", contact_email="test@example.com")
            result = create_trial(req)
        assert result["api_key"].startswith("ftip_trial_")

    def test_still_returns_getting_started(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=False):
            req = TrialRequest(org_name="Test Org", contact_email="test@example.com")
            result = create_trial(req)
        assert "getting_started" in result
        assert "recommended_endpoints" in result["getting_started"]


# ---------------------------------------------------------------------------
# 10 — trial endpoint with db mocked → api_key and getting_started
# ---------------------------------------------------------------------------

class TestTrialEndpointWithDb:
    def test_returns_created_status(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="Acme Corp", contact_email="cfo@acme.com", sector_preference="investment")
            result = create_trial(req)
        assert result["status"] == "created"

    def test_returns_valid_api_key(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="Acme Corp", contact_email="cfo@acme.com")
            result = create_trial(req)
        assert isinstance(result["api_key"], str)
        assert len(result["api_key"]) > 20

    def test_returns_correct_org_name(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="Acme Corp", contact_email="cfo@acme.com")
            result = create_trial(req)
        assert result["org_name"] == "Acme Corp"

    def test_returns_14_day_expiry(self):
        import datetime as dt
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="Acme Corp", contact_email="cfo@acme.com")
            result = create_trial(req)
        expires = dt.date.fromisoformat(result["expires_at"])
        today = dt.date.today()
        delta = (expires - today).days
        assert delta == 14

    def test_investment_sector_endpoints(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="Acme", contact_email="x@x.com", sector_preference="investment")
            result = create_trial(req)
        paths = [ep["path"] for ep in result["getting_started"]["recommended_endpoints"]]
        assert any("prosperity" in p for p in paths)

    def test_pe_sector_endpoints(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="PE Fund", contact_email="gp@fund.com", sector_preference="pe")
            result = create_trial(req)
        paths = [ep["path"] for ep in result["getting_started"]["recommended_endpoints"]]
        assert any("pe" in p.lower() for p in paths)

    def test_unknown_sector_falls_back_to_investment(self):
        from api.jobs.onboarding import create_trial, TrialRequest
        with patch("api.jobs.tenant_auth.register_tenant", return_value=True):
            req = TrialRequest(org_name="X", contact_email="x@x.com", sector_preference="unknown_sector")
            result = create_trial(req)
        # Should not raise, falls back to investment endpoints
        assert len(result["getting_started"]["recommended_endpoints"]) > 0
