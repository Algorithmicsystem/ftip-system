from __future__ import annotations

from typing import Any, Dict, List, Tuple

from api.axiom.common import rounded
from api.axiom.contracts import (
    AxiomDeployabilityDecision,
    AxiomEngineInput,
    AxiomRegimeDecision,
    AxiomScorecard,
    EngineScore,
)


def _score_value(payload: EngineScore) -> float:
    return float(payload.score) if payload.score is not None else 0.0


def _exceptionality_label(scorecard: AxiomScorecard) -> str:
    dau = float(scorecard.deployable_alpha_utility or 0.0)
    validated = float(scorecard.validated_edge or 0.0)
    if dau >= 70.0 and validated >= 60.0:
        return "exceptional"
    if dau >= 55.0 and validated >= 45.0:
        return "high_selectivity"
    if dau >= 40.0:
        return "selective"
    return "ordinary_or_constrained"


def _engine_stack_summary(engine_scores: Dict[str, EngineScore]) -> str:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    behavior = engine_scores["behavioral_distortion"]
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    return (
        f"Fundamental Reality {_score_value(fundamental):.1f}, State Pricing {_score_value(state_pricing):.1f}, "
        f"Behavioral Distortion {_score_value(behavior):.1f}, and Flow Transmission {_score_value(flow):.1f} "
        f"define the opportunity stack, while Liquidity and Convexity {_score_value(liquidity):.1f}, "
        f"Critical Fragility {_score_value(fragility):.1f}, and Research Integrity {_score_value(research):.1f} "
        "decide whether that signal can survive contact with real capital."
    )


def _why_now_summary(
    scorecard: AxiomScorecard,
    regime_decision: AxiomRegimeDecision,
    engine_scores: Dict[str, EngineScore],
) -> str:
    flow = engine_scores["flow_transmission"]
    state_pricing = engine_scores["state_pricing"]
    behavior = engine_scores["behavioral_distortion"]
    return (
        f"Why now: AXIOM sees {str(regime_decision.regime_label).replace('_', ' ')} conditions where "
        f"State Pricing {_score_value(state_pricing):.1f} and Flow Transmission {_score_value(flow):.1f} are strong enough "
        f"to convert the gross opportunity stack into validated edge {float(scorecard.validated_edge or 0.0):.1f}. "
        f"Behavioral Distortion {_score_value(behavior):.1f} indicates whether the tape is merely noisy or meaningfully out of line."
    )


def _unique_mispricing_summary(
    scorecard: AxiomScorecard,
    engine_scores: Dict[str, EngineScore],
) -> str:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    flow = engine_scores["flow_transmission"]
    fragility = engine_scores["critical_fragility"]
    if _score_value(fundamental) >= _score_value(state_pricing):
        primary_gap = "reality is stronger than current pricing"
    else:
        primary_gap = "pricing is moving before reality is fully validated"
    return (
        f"Unique mispricing: AXIOM believes {primary_gap}. Fundamental Reality {_score_value(fundamental):.1f}, "
        f"State Pricing {_score_value(state_pricing):.1f}, and Flow Transmission {_score_value(flow):.1f} imply the market is "
        f"not fully compensating for the setup, but Critical Fragility {_score_value(fragility):.1f} caps how aggressively that gap can be monetized."
    )


def _sorted_components(
    engine_scores: Dict[str, EngineScore],
    *,
    descending: bool,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    ranked: List[Tuple[str, str, float]] = []
    for engine_name, payload in engine_scores.items():
        for component_name, value in (payload.components or {}).items():
            ranked.append((engine_name, component_name, float(value)))
    ranked.sort(key=lambda item: item[2], reverse=descending)
    return [
        {
            "engine": engine_name,
            "component": component_name,
            "value": round(value, 2),
        }
        for engine_name, component_name, value in ranked[:limit]
    ]


def _engine_strength(
    engine_scores: Dict[str, EngineScore],
    *,
    strongest: bool,
) -> Dict[str, Any]:
    ranked: List[Tuple[str, float, float, float]] = []
    for engine_name, payload in engine_scores.items():
        if payload.score is None:
            continue
        ranked.append(
            (
                engine_name,
                float(payload.score),
                float(payload.coverage),
                float(payload.confidence),
            )
        )
    ranked.sort(key=lambda item: item[1], reverse=strongest)
    if not ranked:
        return {}
    engine_name, score, coverage, confidence = ranked[0]
    payload = engine_scores.get(engine_name)
    return {
        "engine": engine_name,
        "score": round(score, 2),
        "coverage": round(coverage, 2),
        "confidence": round(confidence, 2),
        "summary": payload.summary if payload else None,
    }


def build_axiom_explanation(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
    scorecard: AxiomScorecard,
    regime_decision: AxiomRegimeDecision,
    deployability_decision: AxiomDeployabilityDecision,
) -> Dict[str, Any]:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    behavior = engine_scores["behavioral_distortion"]
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    exceptionality = _exceptionality_label(scorecard)
    return {
        "summary": (
            f"AXIOM-50 Phase 2 sees gross opportunity {scorecard.gross_opportunity:.1f}, "
            f"friction burden {scorecard.friction_burden:.1f}, validated edge {scorecard.validated_edge:.1f}, "
            f"and deployable alpha utility {scorecard.deployable_alpha_utility:.1f} with deployability "
            f"{deployability_decision.deployability_tier}. The setup grades as {exceptionality.replace('_', ' ')} rather than routine."
        ),
        "gross_opportunity_reason": (
            f"Gross opportunity is led by Fundamental Reality ({fundamental.score if fundamental.score is not None else 'n/a'}), "
            f"State Pricing ({state_pricing.score if state_pricing.score is not None else 'n/a'}), and "
            f"Flow Transmission ({flow.score if flow.score is not None else 'n/a'}), with Behavioral Distortion treated as "
            "behavioral opportunity quality rather than raw hype intensity."
        ),
        "fragility_reason": (
            f"Critical Fragility contributes {fragility.score if fragility.score is not None else 'n/a'} / 100, "
            f"driven by {', '.join(fragility.flags[:4]) if fragility.flags else 'contained realized instability'}. "
            f"Liquidity and Convexity is {liquidity.score if liquidity.score is not None else 'n/a'} / 100 and "
            f"Research Integrity is {research.score if research.score is not None else 'n/a'} / 100."
        ),
        "deployability_reason": deployability_decision.rationale,
        "regime_reason": regime_decision.rationale,
        "proprietary_synthesis_summary": _engine_stack_summary(engine_scores),
        "why_now_summary": _why_now_summary(scorecard, regime_decision, engine_scores),
        "unique_mispricing_summary": _unique_mispricing_summary(
            scorecard, engine_scores
        ),
        "exceptionality_summary": (
            f"Exceptionality assessment is {exceptionality.replace('_', ' ')} because deployable alpha utility is "
            f"{scorecard.deployable_alpha_utility:.1f}, validated edge is {scorecard.validated_edge:.1f}, "
            f"and fragility drag is {scorecard.friction_burden:.1f}."
        ),
        "cross_engine_stack_summary": (
            f"Cross-engine stack: Fundamental Reality {_score_value(fundamental):.1f}, State Pricing {_score_value(state_pricing):.1f}, "
            f"Behavioral Distortion {_score_value(behavior):.1f}, and Flow Transmission {_score_value(flow):.1f} support the thesis, "
            f"while Liquidity and Convexity {_score_value(liquidity):.1f}, Critical Fragility {_score_value(fragility):.1f}, and Research Integrity {_score_value(research):.1f} govern how much of that signal is investable."
        ),
        "top_positive_components": _sorted_components(engine_scores, descending=True),
        "top_negative_components": _sorted_components(engine_scores, descending=False),
        "top_positive_drivers": _sorted_components(engine_scores, descending=True, limit=5),
        "top_negative_drivers": _sorted_components(engine_scores, descending=False, limit=5),
        "strongest_engine": _engine_strength(engine_scores, strongest=True),
        "weakest_engine": _engine_strength(engine_scores, strongest=False),
        "regime_rationale": regime_decision.rationale,
        "deployability_rationale": deployability_decision.rationale,
        "monitoring_triggers": deployability_decision.monitoring_triggers,
        "size_band_recommendation": deployability_decision.size_band_recommendation,
        "behavioral_engine_convention": "higher_is_behavioral_opportunity_quality",
        "dominant_invalidation_flags": deployability_decision.invalidation_flags,
        "coverage_note": (
            f"Overall AXIOM coverage is {scorecard.overall_coverage:.1f} / 100 and confidence is {scorecard.overall_confidence:.1f} / 100."
        ),
        "engine_scorecard": {
            name: {
                "score": rounded(payload.score),
                "coverage": rounded(payload.coverage),
                "confidence": rounded(payload.confidence),
                "status": payload.status,
            }
            for name, payload in engine_scores.items()
        },
    }
