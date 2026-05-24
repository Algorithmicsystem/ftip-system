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


def _support_vs_drag_summary(engine_scores: Dict[str, EngineScore]) -> str:
    support = _mean_defined(
        [
            _score_value(engine_scores["fundamental_reality"]),
            _score_value(engine_scores["state_pricing"]),
            _score_value(engine_scores["behavioral_distortion"]),
            _score_value(engine_scores["flow_transmission"]),
        ]
    )
    drag = _mean_defined(
        [
            _score_value(engine_scores["critical_fragility"]),
            100.0 - _score_value(engine_scores["liquidity_convexity"]),
            100.0 - _score_value(engine_scores["research_integrity"]),
        ]
    )
    if support is None or drag is None:
        return "Support-versus-drag decomposition is incomplete."
    return (
        f"Opportunity support averages {support:.1f} while monetization drag averages {drag:.1f}. "
        "AXIOM treats that spread, rather than any single engine, as the real signature of whether a setup is ordinary, exceptional, or falsely attractive."
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
        f"Behavioral Distortion {_score_value(behavior):.1f} indicates whether the tape is merely noisy or meaningfully out of line, "
        "so timing depends on whether current positioning is amplifying a real compensation gap rather than just price motion."
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
        f"not fully compensating for the setup, but Critical Fragility {_score_value(fragility):.1f} caps how aggressively that gap can be monetized. "
        "The proprietary edge is not that any one engine is high; it is that cross-engine agreement is high enough to isolate where the market is still underpricing compensation versus execution risk."
    )


def _setup_character_summary(
    scorecard: AxiomScorecard,
    engine_scores: Dict[str, EngineScore],
) -> str:
    exceptionality = _exceptionality_label(scorecard)
    fragility = _score_value(engine_scores["critical_fragility"])
    research = _score_value(engine_scores["research_integrity"])
    if exceptionality == "exceptional" and fragility <= 40.0 and research >= 60.0:
        return "Setup character is exceptional: the opportunity stack is unusually broad, while fragility drag remains contained enough for real capital to participate."
    if exceptionality in {"high_selectivity", "selective"} and fragility <= 55.0:
        return "Setup character is selective rather than routine: several engines agree, but execution discipline still decides whether the edge becomes deployable."
    return "Setup character is ordinary or constrained: the setup may still be directionally right, but the investable edge is not broad enough to treat it as a high-conviction outlier."


def _false_positive_risk_summary(engine_scores: Dict[str, EngineScore]) -> str:
    fundamental = _score_value(engine_scores["fundamental_reality"])
    state_pricing = _score_value(engine_scores["state_pricing"])
    behavior = _score_value(engine_scores["behavioral_distortion"])
    flow = _score_value(engine_scores["flow_transmission"])
    fragility = _score_value(engine_scores["critical_fragility"])
    research = _score_value(engine_scores["research_integrity"])
    if (behavior + flow) / 2.0 > (fundamental + state_pricing) / 2.0 and fragility >= 50.0:
        return "False-positive risk is elevated because flow and behavioral support are outrunning the underlying reality stack while fragility remains high."
    if research < 50.0:
        return "False-positive risk is elevated because research integrity is too weak to trust the apparent edge at face value."
    return "False-positive risk is contained because the reality, pricing, and implementation layers are not being overwhelmed by crowding, instability, or weak research support."


def _decision_hierarchy_summary(
    scorecard: AxiomScorecard,
    engine_scores: Dict[str, EngineScore],
    deployability_decision: AxiomDeployabilityDecision,
) -> str:
    strongest = _engine_strength(engine_scores, strongest=True)
    weakest = _engine_strength(engine_scores, strongest=False)
    return (
        f"Decision hierarchy: gross opportunity {float(scorecard.gross_opportunity or 0.0):.1f} first establishes whether an idea is worth attention, "
        f"validated edge {float(scorecard.validated_edge or 0.0):.1f} then tests whether that opportunity survives cross-engine confirmation, "
        f"and deployable alpha utility {float(scorecard.deployable_alpha_utility or 0.0):.1f} finally decides whether the idea clears real-world friction. "
        f"The current strongest engine is {str(strongest.get('engine') or 'unknown').replace('_', ' ')} and the weakest is {str(weakest.get('engine') or 'unknown').replace('_', ' ')}, "
        f"which is why AXIOM lands on {deployability_decision.deployability_tier.replace('_', ' ')} rather than simply following directional enthusiasm."
    )


def _mean_defined(values: List[float]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


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
        "support_vs_drag_summary": _support_vs_drag_summary(engine_scores),
        "why_now_summary": _why_now_summary(scorecard, regime_decision, engine_scores),
        "unique_mispricing_summary": _unique_mispricing_summary(
            scorecard, engine_scores
        ),
        "setup_character_summary": _setup_character_summary(scorecard, engine_scores),
        "false_positive_risk_summary": _false_positive_risk_summary(engine_scores),
        "decision_hierarchy_summary": _decision_hierarchy_summary(
            scorecard, engine_scores, deployability_decision
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
