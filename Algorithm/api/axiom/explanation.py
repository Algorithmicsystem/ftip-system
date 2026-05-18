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
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    return {
        "summary": (
            f"AXIOM-50 Phase 2 sees gross opportunity {scorecard.gross_opportunity:.1f}, "
            f"friction burden {scorecard.friction_burden:.1f}, validated edge {scorecard.validated_edge:.1f}, "
            f"and deployable alpha utility {scorecard.deployable_alpha_utility:.1f} with deployability "
            f"{deployability_decision.deployability_tier}."
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
