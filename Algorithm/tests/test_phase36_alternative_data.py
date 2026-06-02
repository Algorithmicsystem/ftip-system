"""Phase 7: Advanced Alternative Data Integration tests."""
from __future__ import annotations

import pytest

from api.axiom.engines.options_flow import compute_osms
from api.axiom.engines.narrative_intelligence import compute_nss, compute_nms, detect_narrative_inflection
from api.axiom.engines.institutional_flow import compute_ias
from api.axiom.engines.earnings_intelligence import compute_pess, evaluate_pess_flags
from api.alpha.canonical_signal import build_signal_from_features


# ---------------------------------------------------------------------------
# OSMS Tests
# ---------------------------------------------------------------------------

def test_osms_high_for_unusual_call_buying():
    """Unusual call volume + bullish IV skew + low PCR → OSMS > 65."""
    result = compute_osms({
        "call_volume": 30000,
        "avg_30d_call_volume": 10000,   # 3x avg → z=2.0
        "call_iv_atm": 0.35,
        "put_iv_atm": 0.25,
        "atm_iv": 0.30,
        "large_block_pct": 0.5,
        "put_call_ratio": 0.4,
    })
    assert result > 65.0, f"Expected OSMS > 65, got {result}"
    assert 0.0 <= result <= 100.0


def test_osms_neutral_when_no_data():
    """All None inputs → exactly 50.0."""
    result = compute_osms({})
    assert result == 50.0


def test_osms_low_for_extreme_put_buying():
    """High PCR + negative IV skew → OSMS < 35."""
    result = compute_osms({
        "call_volume": 5000,
        "avg_30d_call_volume": 10000,
        "call_iv_atm": 0.22,
        "put_iv_atm": 0.38,
        "atm_iv": 0.30,
        "large_block_pct": 0.1,
        "put_call_ratio": 1.8,
    })
    assert result < 35.0, f"Expected OSMS < 35, got {result}"


def test_osms_graceful_degradation():
    """Only 2 of 4 components provided → valid float in [0,100], not 50.0."""
    result = compute_osms({
        "put_call_ratio": 0.45,
        "call_volume": 20000,
        "avg_30d_call_volume": 10000,
    })
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0
    assert result != 50.0


def test_osms_integrated_in_trending_signal():
    """OSMS=80 in TRENDING regime → 'options_flow' in reason_codes."""
    features = {
        "signal_regime": "TRENDING",
        "rsi14": 60.0,
        "mom_5": 0.05,
        "mom_21": 0.10,
        "mom_63": 0.15,
        "trend_sma20_50": 0.05,
        "volume_z20": 1.0,
        "sentiment_score": 0.1,
        "maxdd_63d": 0.05,
        "atr_pct": 0.02,
        "volatility_ann": 0.20,
        "osms": 80.0,
        # penalty fields — zero them out
        "event_overhang_score": 0.0,
        "event_uncertainty_score": 0.0,
        "catalyst_burst_score": 0.0,
        "implementation_fragility_score": 0.0,
        "liquidity_quality_score": 80.0,
        "friction_proxy_score": 0.0,
        "execution_cleanliness_score": 80.0,
        "breadth_confirmation_score": 70.0,
        "internal_market_divergence_score": 0.0,
        "leadership_concentration_score": 0.0,
        "benchmark_confirmation_score": 70.0,
        "sector_confirmation_score": 70.0,
        "macro_asset_alignment_score": 70.0,
        "cross_asset_conflict_score": 0.0,
        "cross_asset_divergence_score": 0.0,
        "market_stress_score": 0.0,
        "spillover_risk_score": 0.0,
        "correlation_breakdown_proxy": 0.0,
        "volatility_shock_score": 0.0,
        "stress_transition_score": 0.0,
    }
    result = build_signal_from_features(features, symbol="TEST")
    assert "options_flow" in result["reason_codes"], (
        f"Expected 'options_flow' in reason_codes, got {result['reason_codes']}"
    )


# ---------------------------------------------------------------------------
# NSS / NMS Tests
# ---------------------------------------------------------------------------

def test_nss_high_for_novel_negative_news():
    """High novelty + low similarity + extreme negative sentiment + high relevance → NSS > 60."""
    result = compute_nss({
        "novelty_score": 90.0,
        "baseline_similarity": 0.1,
        "sentiment_score": 10.0,  # very negative
        "entity_relevance": 1.0,
    })
    assert result > 60.0, f"Expected NSS > 60, got {result}"
    assert 0.0 <= result <= 100.0


def test_nss_zero_for_no_news():
    """Empty dict → NSS = 0.0."""
    result = compute_nss({})
    assert result == 0.0


def test_nms_positive_momentum():
    """5-item list all sentiment=80 → NMS > 60."""
    history = [{"sentiment_score": 80.0} for _ in range(5)]
    result = compute_nms(history)
    assert result > 60.0, f"Expected NMS > 60, got {result}"
    assert 0.0 <= result <= 100.0


def test_narrative_inflection_detected():
    """First 5 avg≈20, last 5 avg≈80 → inflection_detected=True, direction=negative_to_positive."""
    history = (
        [{"sentiment_score": 20.0}] * 5
        + [{"sentiment_score": 80.0}] * 5
    )
    result = detect_narrative_inflection(history)
    assert result["inflection_detected"] is True
    assert result["direction_change"] == "negative_to_positive"
    assert result["confidence"] > 0.0


def test_narrative_inflection_not_detected():
    """Consistent sentiment=70 → inflection_detected=False."""
    history = [{"sentiment_score": 70.0} for _ in range(10)]
    result = detect_narrative_inflection(history)
    assert result["inflection_detected"] is False
    assert result["direction_change"] == "none"
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# IAS Tests
# ---------------------------------------------------------------------------

def test_ias_high_for_accumulation():
    """Dark pool buy + positive block + falling short interest → IAS > 70."""
    result = compute_ias({
        "dark_pool_buy_ratio": 0.7,
        "block_trade_direction": 60.0,
        "short_interest_change": -0.2,
    })
    assert result > 70.0, f"Expected IAS > 70, got {result}"
    assert 0.0 <= result <= 100.0


def test_ias_low_for_distribution():
    """Low dark pool + negative block + rising short interest → IAS < 35."""
    result = compute_ias({
        "dark_pool_buy_ratio": 0.25,
        "block_trade_direction": -70.0,
        "short_interest_change": 0.2,
    })
    assert result < 35.0, f"Expected IAS < 35, got {result}"


def test_ias_neutral_missing_data():
    """All None → IAS = 50.0."""
    result = compute_ias({})
    assert result == 50.0


def test_ias_integrated_in_flow_engine():
    """source_context with ias=80 → 'ias_score' in flow engine components."""
    from api.axiom.contracts import AxiomEngineInput
    from api.axiom.engines.flow import score_flow_transmission

    engine_input = AxiomEngineInput(
        framework_version="test_v1",
        symbol="TEST",
        as_of="2024-01-01",
        source_context={"ias": 80.0},
    )
    result = score_flow_transmission(engine_input)
    assert "ias_score" in result.components, (
        f"Expected 'ias_score' in components, got {list(result.components.keys())}"
    )
    assert result.components["ias_score"] == 80.0


# ---------------------------------------------------------------------------
# PESS Tests
# ---------------------------------------------------------------------------

def test_pess_high_for_deteriorating_earnings():
    """Falling EIS + rising accruals + negative guidance → PESS > 65."""
    result = compute_pess({
        "eis_trend_delta": -20.0,
        "accruals_acceleration": 0.08,
        "guidance_revision_velocity": -2.0,
    })
    assert result > 65.0, f"Expected PESS > 65, got {result}"
    assert 0.0 <= result <= 100.0


def test_pess_low_for_quality_earnings():
    """Rising EIS + low accruals + positive guidance → PESS < 40."""
    result = compute_pess({
        "eis_trend_delta": 20.0,
        "accruals_acceleration": -0.03,
        "guidance_revision_velocity": 2.0,
    })
    assert result < 40.0, f"Expected PESS < 40, got {result}"


def test_pess_fires_in_pre_earnings_window():
    """High PESS + days_to_earnings=30 → earnings_stress_flag=True."""
    pess_score = compute_pess({
        "eis_trend_delta": -25.0,
        "accruals_acceleration": 0.09,
        "guidance_revision_velocity": -2.5,
        "insider_sell_ratio": 3.0,
    })
    flags = evaluate_pess_flags(pess_score, days_to_earnings=30)
    assert flags["earnings_stress_flag"] is True, (
        f"Expected earnings_stress_flag=True with PESS={pess_score}"
    )
    assert flags["in_pre_earnings_window"] is True


def test_pess_does_not_fire_outside_window():
    """Even high PESS + days_to_earnings=90 → earnings_stress_flag=False."""
    pess_score = compute_pess({
        "eis_trend_delta": -25.0,
        "accruals_acceleration": 0.09,
        "guidance_revision_velocity": -2.5,
    })
    flags = evaluate_pess_flags(pess_score, days_to_earnings=90)
    assert flags["earnings_stress_flag"] is False
    assert flags["in_pre_earnings_window"] is False


def test_pess_suppression_in_canonical_signal():
    """pess=85, days_to_earnings=15 → 'earnings_risk' in suppression_flags."""
    features = {
        "signal_regime": "CHOPPY",
        "rsi14": 50.0,
        "mom_5": 0.0,
        "mom_21": 0.0,
        "mom_63": 0.0,
        "trend_sma20_50": 0.0,
        "volume_z20": 0.0,
        "sentiment_score": 0.0,
        "maxdd_63d": 0.0,
        "atr_pct": 0.02,
        "volatility_ann": 0.20,
        "pess": 85.0,
        "days_to_earnings": 15.0,
        # zero out all penalty fields
        "event_overhang_score": 0.0,
        "event_uncertainty_score": 0.0,
        "catalyst_burst_score": 0.0,
        "implementation_fragility_score": 0.0,
        "liquidity_quality_score": 80.0,
        "friction_proxy_score": 0.0,
        "execution_cleanliness_score": 80.0,
        "breadth_confirmation_score": 70.0,
        "internal_market_divergence_score": 0.0,
        "leadership_concentration_score": 0.0,
        "benchmark_confirmation_score": 70.0,
        "sector_confirmation_score": 70.0,
        "macro_asset_alignment_score": 70.0,
        "cross_asset_conflict_score": 0.0,
        "cross_asset_divergence_score": 0.0,
        "market_stress_score": 0.0,
        "spillover_risk_score": 0.0,
        "correlation_breakdown_proxy": 0.0,
        "volatility_shock_score": 0.0,
        "stress_transition_score": 0.0,
    }
    result = build_signal_from_features(features, symbol="TEST")
    assert "earnings_risk" in result["suppression_flags"], (
        f"Expected 'earnings_risk' in suppression_flags, got {result['suppression_flags']}"
    )


# ---------------------------------------------------------------------------
# Additional edge-case coverage
# ---------------------------------------------------------------------------

def test_osms_clamped_to_valid_range():
    """Extreme inputs must stay in [0, 100]."""
    result = compute_osms({
        "call_volume": 1_000_000,
        "avg_30d_call_volume": 100,
        "put_call_ratio": 0.30,
        "large_block_pct": 1.0,
    })
    assert 0.0 <= result <= 100.0


def test_nss_clamped_to_valid_range():
    """Extreme novelty/sentiment values must stay in [0, 100]."""
    result = compute_nss({
        "novelty_score": 100.0,
        "baseline_similarity": 0.0,
        "sentiment_score": 0.0,
        "entity_relevance": 1.0,
    })
    assert 0.0 <= result <= 100.0


def test_ias_returns_float():
    """Partial data → valid float."""
    result = compute_ias({"dark_pool_buy_ratio": 0.6})
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_pess_neutral_when_no_data():
    """Empty dict → PESS = 50.0."""
    result = compute_pess({})
    assert result == 50.0


def test_evaluate_pess_flags_returns_all_keys():
    """evaluate_pess_flags always returns all expected keys."""
    flags = evaluate_pess_flags(70.0, days_to_earnings=20)
    assert "earnings_stress_flag" in flags
    assert "high_earnings_risk" in flags
    assert "in_pre_earnings_window" in flags
    assert "pess" in flags


def test_nms_empty_history():
    """Empty history → NMS = 50.0."""
    result = compute_nms([])
    assert result == 50.0


def test_inflection_insufficient_data():
    """Fewer than 10 items → inflection_detected=False."""
    history = [{"sentiment_score": 80.0} for _ in range(5)]
    result = detect_narrative_inflection(history)
    assert result["inflection_detected"] is False


def test_osms_only_pcr():
    """Only PCR provided → score computed from that single component."""
    result_low_pcr = compute_osms({"put_call_ratio": 0.4})
    result_high_pcr = compute_osms({"put_call_ratio": 1.9})
    assert result_low_pcr > result_high_pcr, "Low PCR should score higher than high PCR"


def test_pess_high_risk_threshold():
    """PESS > 80 + days_to_earnings=10 → high_earnings_risk=True."""
    flags = evaluate_pess_flags(85.0, days_to_earnings=10)
    assert flags["high_earnings_risk"] is True
