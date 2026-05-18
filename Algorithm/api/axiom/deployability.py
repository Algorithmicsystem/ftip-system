from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import clamp, mean
from api.axiom.common import inverse_score
from api.axiom.contracts import (
    AxiomDeployabilityDecision,
    AxiomEngineInput,
    AxiomRegimeDecision,
    AxiomScorecard,
    EngineScore,
)


def classify_axiom_deployability(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
    scorecard: AxiomScorecard,
    regime_decision: AxiomRegimeDecision,
) -> AxiomDeployabilityDecision:
    fragility = engine_scores["critical_fragility"]
    liquidity = engine_scores["liquidity_convexity"]
    research = engine_scores["research_integrity"]
    flow = engine_scores["flow_transmission"]
    invalidation_flags = list(fragility.flags)
    if scorecard.overall_coverage < 40.0:
        invalidation_flags.append("insufficient_axiom_coverage")
    if scorecard.overall_confidence < 35.0:
        invalidation_flags.append("insufficient_axiom_confidence")
    if (
        engine_input.fundamental.filing_recency_days is not None
        and engine_input.fundamental.filing_recency_days > 210
    ):
        invalidation_flags.append("stale_fundamental_backbone")
    if (liquidity.score or 0.0) < 42.0:
        invalidation_flags.append("weak_liquidity_integrity")
    if (research.score or 0.0) < 42.0:
        invalidation_flags.append("weak_research_integrity")
    if (engine_input.support.model_drift_score or 0.0) >= 55.0:
        invalidation_flags.append("active_drift")

    tier = "monitor_only"
    rationale = "AXIOM keeps the setup in monitor mode because deployability still needs cleaner liquidity, research, or fragility support."
    size_band = "none"
    monitoring_triggers = list(engine_input.support.monitoring_triggers)
    if (
        bool(engine_input.support.pause_required)
        or (fragility.score or 0.0) >= 82.0
        or scorecard.overall_coverage < 35.0
        or (research.score or 0.0) < 28.0
    ):
        tier = "not_actionable"
        size_band = "none"
        rationale = "High fragility, an active pause condition, or very weak research integrity makes the setup unsuitable for actionability."
    elif (
        scorecard.deployable_alpha_utility >= 62.0
        and scorecard.overall_coverage >= 66.0
        and scorecard.overall_confidence >= 60.0
        and (fragility.score or 100.0) <= 45.0
        and (liquidity.score or 0.0) >= 58.0
        and (research.score or 0.0) >= 62.0
        and (engine_input.support.live_readiness_score or 0.0) >= 75.0
        and regime_decision.regime_label not in {"euphoria_critical", "liquidity_fracture"}
    ):
        tier = "live_candidate"
        size_band = "medium" if (fragility.score or 0.0) > 32.0 else "large"
        rationale = "Coherent opportunity, contained fragility, usable liquidity, and stronger research integrity support live-candidate treatment."
    elif (
        scorecard.deployable_alpha_utility >= 48.0
        and scorecard.overall_coverage >= 45.0
        and scorecard.overall_confidence >= 40.0
        and (fragility.score or 100.0) <= 65.0
        and (liquidity.score or 0.0) >= 42.0
        and (research.score or 0.0) >= 42.0
    ):
        tier = "paper_trade_only"
        size_band = "small"
        rationale = "The setup has enough structure for paper-trade validation, but evidence, fragility, or liquidity still cap live deployability."
    elif scorecard.deployable_alpha_utility < 35.0:
        tier = "not_actionable"
        size_band = "none"
        rationale = "Deployable alpha utility is too weak once fragility, liquidity, and evidence penalties are applied."

    if tier == "live_candidate" and (engine_input.support.live_readiness_score or 0.0) < 72.0:
        tier = "paper_trade_only"
        size_band = "small"
        rationale = "AXIOM would otherwise like the setup, but existing readiness gates still cap it at paper-trade-only."
    if tier == "paper_trade_only" and regime_decision.regime_label in {"euphoria_critical", "liquidity_fracture"}:
        tier = "monitor_only"
        size_band = "none"
        rationale = "The regime profile is too unstable for paper-trade escalation despite some residual opportunity."
    if tier == "monitor_only" and scorecard.deployable_alpha_utility < 28.0:
        tier = "not_actionable"
        size_band = "none"

    flags = sorted(set(invalidation_flags))
    review_required = tier != "live_candidate"
    confidence = clamp(
        mean(
            [
                scorecard.overall_confidence,
                scorecard.overall_coverage,
                research.confidence,
                liquidity.confidence,
                flow.confidence,
                inverse_score(fragility.confidence) if fragility.confidence else 50.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    monitoring_triggers = sorted(
        set(
            monitoring_triggers
            + list(engine_input.support.confirmation_triggers)
            + list(engine_input.support.deterioration_triggers)
            + list(engine_input.support.invalidators)
            + list(engine_input.support.fragility_vetoes)
        )
    )
    return AxiomDeployabilityDecision(
        deployability_tier=tier,
        confidence=round(confidence, 2),
        rationale=rationale,
        flags=flags,
        review_required=review_required,
        invalidation_flags=flags,
        size_band_recommendation=size_band,
        monitoring_triggers=monitoring_triggers,
    )
