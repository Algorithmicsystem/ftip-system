from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp, mean
from api.axiom.contracts import (
    AxiomArtifact,
    AxiomEngineInput,
    AxiomScorecard,
    ENGINE_KEYS,
    EngineScore,
)
from api.axiom.deployability import classify_axiom_deployability
from api.axiom.explanation import build_axiom_explanation
from api.axiom.mappers import build_axiom_engine_input
from api.axiom.regime import classify_axiom_regime
from api.axiom.scorecard import build_axiom_scorecard
from api.axiom.engines import (
    score_behavioral_distortion,
    score_critical_fragility,
    score_flow_transmission,
    score_fundamental_reality,
    score_liquidity_convexity,
    score_research_integrity,
    score_state_pricing,
)


AXIOM_ARTIFACT_KIND = "assistant_axiom_artifact"
AXIOM_FRAMEWORK_VERSION = "axiom50_phase2_v1"


def _coverage_summary(
    *,
    engine_scores: Dict[str, EngineScore],
    engine_input: AxiomEngineInput,
) -> Dict[str, Any]:
    overall_coverage = clamp(
        mean([payload.coverage for payload in engine_scores.values()]) or 0.0,
        0.0,
        100.0,
    )
    overall_confidence = clamp(
        mean([payload.confidence for payload in engine_scores.values()]) or 0.0,
        0.0,
        100.0,
    )
    available = sum(1 for payload in engine_scores.values() if payload.status == "available")
    partial = sum(1 for payload in engine_scores.values() if payload.status == "partial")
    unavailable = sum(1 for payload in engine_scores.values() if payload.status == "unavailable")
    return {
        "overall_coverage": round(overall_coverage, 2),
        "overall_confidence": round(overall_confidence, 2),
        "available_engines": available,
        "partial_engines": partial,
        "unavailable_engines": unavailable,
        "implemented_engines": list(ENGINE_KEYS),
        "partial_engine_hints": engine_input.partial_engine_hints,
    }


def _diagnostics(
    *,
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
    scorecard: AxiomScorecard,
) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "engine_statuses": {
            name: {
                "status": payload.status,
                "coverage": payload.coverage,
                "confidence": payload.confidence,
            }
            for name, payload in engine_scores.items()
        },
        "source_warnings": engine_input.warnings,
        "domain_coverage": engine_input.domain_coverage,
        "scorecard_summary": scorecard.summary,
    }


def build_axiom_artifact(
    *,
    normalized_bundle: Dict[str, Any],
    job_context: Optional[Dict[str, Any]] = None,
    feature_factor_bundle: Optional[Dict[str, Any]] = None,
    strategy_bundle: Optional[Dict[str, Any]] = None,
    report_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    engine_input = build_axiom_engine_input(
        normalized_bundle,
        job_context=job_context,
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context=report_context,
    )
    engine_scores: Dict[str, EngineScore] = {
        "fundamental_reality": score_fundamental_reality(engine_input),
        "state_pricing": score_state_pricing(engine_input),
        "behavioral_distortion": score_behavioral_distortion(engine_input),
        "flow_transmission": score_flow_transmission(engine_input),
        "liquidity_convexity": score_liquidity_convexity(engine_input),
        "critical_fragility": score_critical_fragility(engine_input),
        "research_integrity": score_research_integrity(engine_input),
    }
    for engine_name in ENGINE_KEYS:
        engine_scores.setdefault(
            engine_name,
            EngineScore(
                score=None,
                confidence=0.0,
                coverage=0.0,
                status="unavailable",
                components={},
                flags=["missing_engine_registration"],
                summary="The engine was not registered in the Phase 2 AXIOM build.",
            ),
        )
    scorecard = build_axiom_scorecard(engine_input, engine_scores)
    regime_decision = classify_axiom_regime(engine_input, engine_scores, scorecard)
    deployability_decision = classify_axiom_deployability(
        engine_input,
        engine_scores,
        scorecard,
        regime_decision,
    )
    explanation = build_axiom_explanation(
        engine_input,
        engine_scores,
        scorecard,
        regime_decision,
        deployability_decision,
    )
    coverage_summary = _coverage_summary(
        engine_scores=engine_scores,
        engine_input=engine_input,
    )
    artifact = AxiomArtifact(
        framework_version=AXIOM_FRAMEWORK_VERSION,
        symbol=engine_input.symbol,
        as_of=engine_input.as_of,
        source_context=engine_input.source_context,
        engine_scores=engine_scores,
        scorecard=scorecard,
        regime_decision=regime_decision,
        deployability_decision=deployability_decision,
        gross_opportunity=scorecard.gross_opportunity,
        friction_burden=scorecard.friction_burden,
        validated_edge=scorecard.validated_edge,
        deployable_alpha_utility=scorecard.deployable_alpha_utility,
        regime_label=regime_decision.regime_label,
        trade_family=regime_decision.trade_family,
        deployability_tier=deployability_decision.deployability_tier,
        invalidation_flags=deployability_decision.invalidation_flags,
        explanation=explanation,
        coverage_summary=coverage_summary,
        diagnostics=_diagnostics(
            engine_input=engine_input,
            engine_scores=engine_scores,
            scorecard=scorecard,
        ),
    )
    return artifact.model_dump(mode="python")
