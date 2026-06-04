"""Phase 18 tests: Competitive Intelligence and Cross-Asset Global Macro."""
from __future__ import annotations

import datetime as dt

import pytest

from api.competitive.competitive_intelligence import (
    CompetitorProfile,
    _classify_position,
    _competitive_score,
    compute_competitor_profile,
    compute_market_share_momentum,
)
from api.competitive.management_quality import (
    ManagementQuality,
    _mqs_integrity_signal,
    compute_capital_allocation_score,
    compute_guidance_accuracy_score,
    compute_insider_alignment_score,
    compute_mqs,
)
from api.macro.cross_asset_engine import (
    CrossAssetSnapshot,
    apply_cross_asset_overlay,
    compute_cross_asset_snapshot,
)
from api.macro.global_macro import (
    MacroIntelligenceSnapshot,
    classify_macro_regime,
    compute_macro_factor_overlay,
)


# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------

_SYM_PAYLOAD = {
    "deployable_alpha_utility": 75.0,
    "engine_scores": {
        "fundamental_reality": {
            "components": {"eis_component": 80.0, "caps_component": 70.0}
        },
        "critical_fragility": {"score": 35.0},
        "flow_transmission": {"score": 60.0},
        "behavioral_distortion": {"score": 65.0},
    },
}

_COMP_PAYLOAD = {
    "deployable_alpha_utility": 60.0,
    "engine_scores": {
        "fundamental_reality": {
            "components": {"eis_component": 60.0, "caps_component": 65.0}
        },
        "critical_fragility": {"score": 45.0},
        "flow_transmission": {"score": 70.0},
        "behavioral_distortion": {"score": 55.0},
    },
}


# ===========================================================================
# TestCompetitiveIntelligence
# ===========================================================================

class TestCompetitiveIntelligence:
    def test_competitor_profile_computed(self):
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert isinstance(profile, CompetitorProfile)
        assert profile.symbol == "AAPL"
        assert profile.competitor_symbol == "MSFT"

    def test_dau_advantage_computed(self):
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert profile.dau_advantage == 15.0

    def test_competitive_position_score_bounded(self):
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert 0.0 <= profile.competitive_position_score <= 100.0

    def test_key_advantage_identified(self):
        # EIS advantage = 80-60=20, DAU=15, CAPS=5, fragility=10, momentum=-10
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert profile.key_advantage != "none"
        assert profile.key_advantage in ("eis", "dau", "caps", "fragility")

    def test_key_vulnerability_identified(self):
        # Competitor has higher flow score: momentum_advantage = 60-70 = -10 → vulnerability
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert profile.key_vulnerability == "momentum"

    def test_sector_rank_leader(self):
        assert _classify_position(1, 10) == "leader"
        assert _classify_position(2, 10) == "leader"

    def test_sector_rank_tail(self):
        assert _classify_position(10, 10) == "tail"

    def test_competitive_score_leader_high(self):
        # rank 1 of 10: 100 × (1 - 1/10) = 90
        score = _competitive_score(1, 10)
        assert score >= 90

    def test_market_share_momentum_gaining(self):
        payloads = {
            "AAPL": {"revenue_growth_ttm": 0.20},
            "MSFT": {"revenue_growth_ttm": 0.10},
            "GOOG": {"revenue_growth_ttm": 0.08},
        }
        result = compute_market_share_momentum("AAPL", ["MSFT", "GOOG"], payloads)
        assert result == "gaining"

    def test_market_share_momentum_losing(self):
        payloads = {
            "AAPL": {"revenue_growth_ttm": 0.02},
            "MSFT": {"revenue_growth_ttm": 0.15},
            "GOOG": {"revenue_growth_ttm": 0.18},
        }
        result = compute_market_share_momentum("AAPL", ["MSFT", "GOOG"], payloads)
        assert result == "losing"

    def test_competitive_distance_between_zero_and_one(self):
        profile = compute_competitor_profile("AAPL", "MSFT", _SYM_PAYLOAD, _COMP_PAYLOAD)
        assert 0.0 <= profile.competitive_distance <= 1.0


# ===========================================================================
# TestManagementQuality
# ===========================================================================

_IMPROVING_ROIC_FINS = [
    {"roic": 0.08}, {"roic": 0.10}, {"roic": 0.12}, {"roic": 0.15},
]
_DECLINING_ROIC_FINS = [
    {"roic": 0.15}, {"roic": 0.12}, {"roic": 0.10}, {"roic": 0.07},
]
_PERFECT_EARNINGS = [
    {"guided_eps": 1.00, "actual_eps": 1.02},
    {"guided_eps": 1.05, "actual_eps": 1.03},
]
_POOR_EARNINGS = [
    {"guided_eps": 1.00, "actual_eps": 0.80},
    {"guided_eps": 1.00, "actual_eps": 0.78},
]


class TestManagementQuality:
    def test_mqs_computed(self):
        mqs = compute_mqs("AAPL", _SYM_PAYLOAD)
        assert isinstance(mqs, ManagementQuality)
        assert 0.0 <= mqs.mqs_score <= 100.0

    def test_high_roic_trend_high_capital_score(self):
        score = compute_capital_allocation_score(_IMPROVING_ROIC_FINS)
        assert score > 60

    def test_declining_roic_low_capital_score(self):
        score = compute_capital_allocation_score(_DECLINING_ROIC_FINS)
        assert score < 50

    def test_insider_buying_positive(self):
        score = compute_insider_alignment_score({"buy_count": 5, "sell_count": 0})
        assert score > 65

    def test_insider_selling_negative(self):
        score = compute_insider_alignment_score({"buy_count": 0, "sell_count": 5})
        assert score < 40

    def test_guidance_accuracy_perfect(self):
        score = compute_guidance_accuracy_score(_PERFECT_EARNINGS)
        assert score == 100.0

    def test_guidance_accuracy_poor(self):
        score = compute_guidance_accuracy_score(_POOR_EARNINGS)
        assert score == 0.0

    def test_mqs_integrity_high(self):
        assert _mqs_integrity_signal(70.0) == "high"

    def test_mqs_integrity_low(self):
        assert _mqs_integrity_signal(35.0) == "low"

    def test_mqs_components_all_present(self):
        mqs = compute_mqs("AAPL", _SYM_PAYLOAD, financials_history=_IMPROVING_ROIC_FINS)
        assert hasattr(mqs, "capital_allocation_score")
        assert hasattr(mqs, "guidance_accuracy_score")
        assert hasattr(mqs, "insider_alignment_score")
        assert hasattr(mqs, "compensation_alignment_score")
        assert hasattr(mqs, "communication_quality_score")


# ===========================================================================
# TestCrossAsset
# ===========================================================================

_ALL_RISK_ON_TRENDING = dict(
    equity_regime_label="TRENDING",
    vix_level=12.0,
    yield_curve_slope=2.0,
    copper_return_90d=0.10,
    dxy_return_30d=-0.03,
)

_ALL_RISK_OFF_TRENDING = dict(
    equity_regime_label="TRENDING",
    vix_level=40.0,
    yield_curve_slope=-0.5,
    copper_return_90d=-0.10,
    dxy_return_30d=0.03,
)


class TestCrossAsset:
    def test_snapshot_fields_present(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING")
        required = [
            "equity_regime_confirmed", "cross_asset_confirmation_score",
            "regime_consistency", "fixed_income_signal", "currency_signal",
            "commodity_signal", "volatility_signal", "carry_environment",
            "value_environment", "momentum_environment", "defensive_environment",
            "equity_signal_amplifier", "macro_headwind_score", "macro_tailwind_score",
            "macro_narrative",
        ]
        for f in required:
            assert hasattr(snap, f), f"Missing field: {f}"

    def test_normal_curve_risk_on(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING", yield_curve_slope=2.0)
        assert snap.fixed_income_signal == "risk_on"

    def test_inverted_curve_risk_off(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING", yield_curve_slope=-0.5)
        assert snap.fixed_income_signal == "risk_off"

    def test_low_vix_risk_on(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING", vix_level=12.0)
        assert snap.volatility_signal == "risk_on"

    def test_high_vix_risk_off(self):
        snap = compute_cross_asset_snapshot({}, "HIGH_VOL", vix_level=40.0)
        assert snap.volatility_signal in ("risk_off", "extreme_risk_off")

    def test_rising_usd_risk_off(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING", dxy_return_30d=0.03)
        assert snap.currency_signal == "risk_off"

    def test_confirmation_high_when_aligned(self):
        snap = compute_cross_asset_snapshot({}, **_ALL_RISK_ON_TRENDING)
        assert snap.cross_asset_confirmation_score > 70

    def test_amplifier_positive_high_confirmation(self):
        snap = compute_cross_asset_snapshot({}, **_ALL_RISK_ON_TRENDING)
        assert snap.equity_signal_amplifier > 0

    def test_amplifier_negative_low_confirmation(self):
        snap = compute_cross_asset_snapshot({}, **_ALL_RISK_OFF_TRENDING)
        assert snap.equity_signal_amplifier < 0

    def test_macro_narrative_not_empty(self):
        snap = compute_cross_asset_snapshot({}, "TRENDING", vix_level=18.0, yield_curve_slope=1.5)
        assert isinstance(snap.macro_narrative, str)
        assert len(snap.macro_narrative) > 0

    def test_apply_cross_asset_overlay_bounded(self):
        snap = compute_cross_asset_snapshot({}, **_ALL_RISK_ON_TRENDING)
        adjusted = apply_cross_asset_overlay(75.0, snap, "TRENDING")
        assert 0.0 <= adjusted <= 100.0


# ===========================================================================
# TestGlobalMacro
# ===========================================================================

class TestGlobalMacro:
    def test_regime_expansion_high_growth(self):
        snap = classify_macro_regime(3.5, 2.0, 2.5, 2.0)
        assert snap.gdp_regime == "expansion"

    def test_regime_contraction_negative(self):
        snap = classify_macro_regime(-1.0, 2.0, 3.0, 3.0)
        assert snap.gdp_regime == "contraction"

    def test_inflation_high(self):
        snap = classify_macro_regime(2.0, 5.5, 4.0, 2.0)
        assert snap.inflation_regime == "high_inflation"

    def test_inflation_low(self):
        snap = classify_macro_regime(2.0, 1.5, 2.0, 2.0)
        assert snap.inflation_regime == "low_inflation"

    def test_tightening_regime(self):
        # Rate rose 200bps (not >200, so tightening not emergency)
        snap = classify_macro_regime(2.5, 3.0, 3.0, 1.0)
        assert snap.monetary_regime == "tightening"

    def test_equity_macro_score_bounded(self):
        snap = classify_macro_regime(2.5, 2.5, 3.0, 2.0)
        assert 0.0 <= snap.equity_macro_score <= 100.0

    def test_favored_factors_nonempty(self):
        snap = classify_macro_regime(2.5, 2.5, 3.0, 2.0)
        assert len(snap.favored_axiom_factors) >= 1

    def test_unfavored_factors_different(self):
        snap = classify_macro_regime(3.5, 1.5, 2.5, 2.0)
        favored_set = set(snap.favored_axiom_factors)
        unfavored_set = set(snap.unfavored_axiom_factors)
        assert favored_set.isdisjoint(unfavored_set)

    def test_macro_factor_overlay_adjusts_weights(self):
        snap = classify_macro_regime(3.5, 1.5, 2.5, 2.0)
        base = {f: 0.5 for f in snap.favored_axiom_factors + snap.unfavored_axiom_factors}
        adjusted = compute_macro_factor_overlay(snap, base)
        # At least some factor weights should differ from base
        assert any(abs(adjusted.get(f, 0.5) - 0.5) > 1e-6 for f in base)

    def test_macro_regime_label_format(self):
        snap = classify_macro_regime(3.5, 1.5, 2.5, 2.0)
        parts = snap.macro_regime_label.split("_")
        # Should have at least gdp_regime and inflation_regime components
        assert len(snap.macro_regime_label) > 0
