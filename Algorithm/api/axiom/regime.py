from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import clamp, mean
from api.axiom.common import inverse_score
from api.axiom.contracts import (
    AxiomEngineInput,
    AxiomRegimeDecision,
    AxiomScorecard,
    EngineScore,
)


def classify_axiom_regime(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
    scorecard: AxiomScorecard,
) -> AxiomRegimeDecision:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    behavioral = engine_scores["behavioral_distortion"]
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    support = engine_input.support
    fragility_input = engine_input.fragility
    flags: list[str] = []
    label = "indeterminate"
    trade_family = "none"
    rationale = "AXIOM remains conservative because the seven-engine stack is still conflicted."
    alignment = float(scorecard.cross_engine_alignment or 0.0)
    timing = float(scorecard.timing_support or 0.0)
    path = float(scorecard.path_survivability or 0.0)
    evidence = float(scorecard.evidence_readiness or 0.0)
    false_positive = float(scorecard.false_positive_penalty or 0.0)
    mispricing = float(scorecard.mispricing_readiness or 0.0)
    maturity = float(scorecard.setup_maturity or 0.0)

    if (
        false_positive >= 72.0
        and (fragility.score or 0.0) >= 74.0
        and (
            (support.narrative_crowding_index or 0.0) >= 68.0
            or (support.hype_to_price_divergence_score or 0.0) >= 60.0
        )
    ):
        label = "euphoria_critical"
        rationale = "Narrative crowding, cross-engine false-positive pressure, and rising fragility now overwhelm the investable thesis, so AXIOM treats the regime as euphoric and unstable rather than actionable."
        flags.extend(["behavioral_overshoot", "fragility_dominant", "false_positive_pressure"])
    elif (
        (fragility_input.implementation_fragility_score or 0.0) >= 75.0
        and ((liquidity.score or 100.0) <= 42.0 or path <= 40.0)
    ):
        label = "liquidity_fracture"
        rationale = "Execution survivability has broken down relative to the apparent opportunity stack, so AXIOM classifies the regime as a liquidity fracture rather than a usable setup."
        flags.extend(["liquidity_fracture", "execution_breakdown", "path_survivability_low"])
    elif (
        (fundamental.score or 0.0) >= 68.0
        and (state_pricing.score or 0.0) >= 62.0
        and (fragility.score or 100.0) <= 45.0
        and (research.score or 0.0) >= 55.0
        and alignment >= 66.0
        and mispricing >= 60.0
        and false_positive <= 42.0
    ):
        label = "fundamental_convergence"
        trade_family = "convergence"
        rationale = "Fundamental reality, current pricing, and implementation quality are converging at the same time, so AXIOM treats the setup as a true convergence regime rather than a loose directional idea."
        flags.extend(["quality_supported", "cross_engine_alignment_high"])
    elif (
        (state_pricing.score or 0.0) >= 66.0
        and (fundamental.score or 0.0) >= 55.0
        and (flow.score or 0.0) >= 52.0
        and 42.0 <= (fragility.score or 0.0) <= 66.0
        and mispricing >= 54.0
        and path >= 46.0
        and false_positive <= 58.0
    ):
        label = "compensation_capture"
        trade_family = "compensation"
        rationale = "The market is offering usable compensation for residual macro or fragility risk, so AXIOM classifies the idea as compensation capture rather than clean convergence."
        flags.extend(["risk_premium_supported", "residual_risk_compensated"])
    elif (
        (behavioral.score or 0.0) >= 64.0
        and (flow.score or 0.0) >= 62.0
        and (support.narrative_crowding_index or 100.0) <= 58.0
        and (fragility_input.cross_asset_conflict_score or 100.0) < 45.0
        and timing >= 62.0
        and maturity >= 58.0
        and false_positive <= 48.0
    ):
        label = "behavioral_continuation"
        trade_family = "transmission"
        rationale = "Timing, flow transmission, and cleaner behavioral participation all remain supportive, so AXIOM sees a usable behavioral continuation rather than a late crowded chase."
        flags.extend(["trend_supported", "timing_supported"])
    elif (
        (liquidity.score or 0.0) >= 58.0
        and (support.negative_news_resilient_price_divergence or 0.0) >= 55.0
        and (fragility.score or 0.0) <= 58.0
        and (support.live_readiness_score or 0.0) >= 45.0
        and path >= 56.0
        and mispricing >= 50.0
    ):
        label = "convexity_opportunity"
        trade_family = "convexity"
        rationale = "The setup preserves asymmetry through adequate liquidity and path survivability, so AXIOM classifies it as a convexity opportunity rather than a simple bounce."
        flags.extend(["asymmetry_supported", "path_survivability_supported"])
    elif (
        (fragility_input.maxdd_63d or 0.0) <= -0.15
        and (fundamental.score or 0.0) >= 55.0
        and (behavioral.score or 0.0) >= 54.0
        and mispricing >= 50.0
        and evidence >= 46.0
    ):
        label = "recovery_reset"
        trade_family = "recovery"
        rationale = "The setup is emerging from a meaningful drawdown, but enough valuation and behavioral support has reset to justify a selective recovery classification."
        flags.extend(["drawdown_recovery", "reset_support"])

    confidence = clamp(
        mean(
            [
                scorecard.overall_coverage,
                scorecard.overall_confidence,
                alignment,
                evidence,
                fundamental.confidence,
                state_pricing.confidence,
                behavioral.confidence,
                flow.confidence,
                liquidity.confidence,
                research.confidence,
                inverse_score(fragility.confidence) if fragility.confidence else 50.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    return AxiomRegimeDecision(
        regime_label=label,
        trade_family=trade_family,
        confidence=round(confidence, 2),
        rationale=rationale,
        flags=sorted(
            set(
                flags
                + list(fragility.flags)
                + list(state_pricing.flags)
                + list(behavioral.flags)
                + list(flow.flags)
                + list(liquidity.flags)
                + list(research.flags)
            )
        ),
    )
