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
    alignment = float(scorecard.cross_engine_alignment or 0.0)
    evidence = float(scorecard.evidence_readiness or 0.0)
    path = float(scorecard.path_survivability or 0.0)
    false_positive = float(scorecard.false_positive_penalty or 0.0)
    exceptional = float(scorecard.exceptional_opportunity or 0.0)
    timing = float(scorecard.timing_support or 0.0)
    event_support = float(scorecard.event_overhang_support or 0.0)
    source_penalty = float(scorecard.source_strength_penalty or 0.0)
    recency_quality = float(scorecard.evidence_recency_quality or 0.0)
    catalyst_quality = float(scorecard.catalyst_quality or 0.0)
    estimate_support = float(scorecard.estimate_revision_support or 0.0)
    has_event_support = engine_input.support.event_overhang_support_or_penalty is not None
    has_source_penalty = engine_input.support.source_strength_penalty is not None
    has_recency_quality = engine_input.support.evidence_recency_quality is not None
    has_catalyst_quality = engine_input.support.catalyst_quality is not None
    has_estimate_support = engine_input.support.estimate_revision_support is not None
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
    if false_positive >= 60.0:
        invalidation_flags.append("false_positive_pressure")
    if path < 45.0:
        invalidation_flags.append("weak_path_survivability")
    if evidence < 48.0:
        invalidation_flags.append("weak_evidence_readiness")
    if has_source_penalty and source_penalty >= 60.0:
        invalidation_flags.append("weak_source_stack")
    if has_recency_quality and recency_quality < 45.0:
        invalidation_flags.append("stale_event_evidence")
    if (
        (has_event_support or has_catalyst_quality)
        and event_support < 42.0
        and catalyst_quality < 48.0
    ):
        invalidation_flags.append("event_overhang_drag")

    tier = "monitor_only"
    rationale = "AXIOM keeps the setup in monitor mode because deployability still needs cleaner liquidity, research, or fragility support."
    size_band = "none"
    monitoring_triggers = list(engine_input.support.monitoring_triggers)
    if (
        bool(engine_input.support.pause_required)
        or (fragility.score or 0.0) >= 82.0
        or scorecard.overall_coverage < 35.0
        or (research.score or 0.0) < 28.0
        or false_positive >= 76.0
    ):
        tier = "not_actionable"
        size_band = "none"
        rationale = "High fragility, active drift or pause conditions, or extreme false-positive pressure makes the setup unsuitable for actionability."
    elif (
        scorecard.deployable_alpha_utility >= 64.0
        and scorecard.overall_coverage >= 66.0
        and scorecard.overall_confidence >= 60.0
        and alignment >= 64.0
        and evidence >= 66.0
        and path >= 64.0
        and false_positive <= 34.0
        and (not has_source_penalty or source_penalty <= 38.0)
        and exceptional >= 64.0
        and (not has_event_support or event_support >= 54.0)
        and (not has_catalyst_quality or catalyst_quality >= 56.0)
        and (not has_recency_quality or recency_quality >= 60.0)
        and (fragility.score or 100.0) <= 45.0
        and (liquidity.score or 0.0) >= 58.0
        and (research.score or 0.0) >= 62.0
        and (engine_input.support.live_readiness_score or 0.0) >= 75.0
        and regime_decision.regime_label not in {"euphoria_critical", "liquidity_fracture"}
    ):
        tier = "live_candidate"
        size_band = "medium" if (fragility.score or 0.0) > 32.0 else "large"
        rationale = "Live-candidate status is earned because cross-engine alignment, evidence readiness, and path survivability all clear live-capital thresholds while false-positive pressure stays contained."
    elif (
        scorecard.deployable_alpha_utility >= 46.0
        and scorecard.overall_coverage >= 45.0
        and scorecard.overall_confidence >= 40.0
        and alignment >= 44.0
        and evidence >= 46.0
        and path >= 44.0
        and false_positive <= 58.0
        and (not has_source_penalty or source_penalty <= 64.0)
        and (not has_recency_quality or recency_quality >= 42.0)
        and (fragility.score or 100.0) <= 65.0
        and (liquidity.score or 0.0) >= 42.0
        and (research.score or 0.0) >= 42.0
    ):
        tier = "paper_trade_only"
        size_band = "small"
        rationale = "The setup is coherent enough for paper-trade validation, but evidence readiness, path survivability, or false-positive suppression still fall short of live deployment standards."
    elif scorecard.deployable_alpha_utility < 35.0:
        tier = "not_actionable"
        size_band = "none"
        rationale = "Deployable alpha utility is too weak once fragility, liquidity, and evidence penalties are applied."

    if tier == "live_candidate" and (engine_input.support.live_readiness_score or 0.0) < 72.0:
        tier = "paper_trade_only"
        size_band = "small"
        rationale = "AXIOM would otherwise like the setup, but existing readiness gates still cap it at paper-trade-only."
    if (
        tier == "live_candidate"
        and has_source_penalty
        and source_penalty >= 34.0
        and (not has_estimate_support or estimate_support < 52.0)
    ):
        tier = "paper_trade_only"
        size_band = "small"
        rationale = "The gross setup is strong, but the source stack and estimates/event follow-through are not yet clean enough for live escalation."
    if tier == "paper_trade_only" and regime_decision.regime_label in {"euphoria_critical", "liquidity_fracture"}:
        tier = "monitor_only"
        size_band = "none"
        rationale = "The regime profile is too unstable for paper-trade escalation despite some residual opportunity."
    if tier == "paper_trade_only" and false_positive >= 52.0 and timing < 52.0:
        tier = "monitor_only"
        size_band = "none"
        rationale = "Paper tracking is still too generous because timing support is weak while false-positive pressure remains elevated."
    if tier == "paper_trade_only" and (
        (has_source_penalty and source_penalty >= 58.0)
        or (has_recency_quality and recency_quality < 40.0)
        or (has_event_support and event_support < 38.0)
    ):
        tier = "monitor_only"
        size_band = "none"
        rationale = "Paper tracking is still too generous because event or source-stack quality remains too weak to trust the current setup path."
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
                alignment,
                evidence,
                path,
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
