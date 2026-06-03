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
    payload = artifact.model_dump(mode="python", exclude={"scorecard": {"component_support"}})

    # Factor Model: compute and inject factor loadings + FCS + alpha decomposition
    try:
        from api.axiom.factors.factor_model import compute_all_factor_loadings
        from api.axiom.factors.alpha_decomposition import decompose_alpha
        from api.axiom.factors.regime_factor_matrix import compute_factor_composite_score

        _regime_for_factors = str(payload.get("regime_label") or "CHOPPY").upper()
        # Map AXIOM regime labels to factor matrix regimes
        _REGIME_MAP = {
            "BULL_TRENDING": "TRENDING", "TREND_CONFIRMED": "TRENDING",
            "BEAR_STRESS": "HIGH_VOL", "LIQUIDITY_FRACTURE": "HIGH_VOL",
            "RECOVERY_PHASE": "RECOVERY",
        }
        _regime_key = _REGIME_MAP.get(_regime_for_factors, _regime_for_factors)
        if _regime_key not in ("TRENDING", "CHOPPY", "HIGH_VOL", "RECOVERY"):
            _regime_key = "CHOPPY"

        # Build engine_scores dict for factor model (scores as dicts)
        _es = {}
        for _name, _escore in engine_scores.items():
            _es[_name] = {
                "score": _escore.score,
                "components": dict(_escore.components or {}),
            }

        # engine_inputs carries regime info
        _ei = {
            "regime_label": _regime_key,
            "regime_strength": float((engine_input.source_context or {}).get("regime_strength", 0.5)),
            "amqs_score": (engine_input.source_context or {}).get("amqs_score"),
        }

        _factor_loadings = compute_all_factor_loadings(_es, _ei)
        _fcs = compute_factor_composite_score(_factor_loadings, _regime_key)
        _decomp = decompose_alpha(payload, _factor_loadings, _regime_key,
                                   symbol=str(engine_input.symbol or ""),
                                   as_of_date=str(engine_input.as_of or ""))

        payload["factor_composite_score"] = _fcs
        payload["alpha_decomposition"] = _decomp.to_dict()
        payload["factor_loadings_summary"] = [
            {"factor_name": fl.factor_name, "loading": fl.loading, "t_stat": fl.t_stat}
            for fl in _factor_loadings
        ]
    except Exception:
        payload["factor_composite_score"] = 50.0
        payload["alpha_decomposition"] = {}
        payload["factor_loadings_summary"] = []
        _factor_loadings = []

    # ML Signal Layer: build feature vector and run inference
    try:
        from api.axiom.ml.feature_builder import build_feature_vector
        from api.axiom.ml.inference import compute_ml_signal_boost, predict_signal
        from api.assistant.phase3.common import clamp as _clamp

        _fv = build_feature_vector(payload, _factor_loadings, ic_state=None)
        _ml_pred = predict_signal(_fv, regime_label=payload.get("regime_label"))
        _ml_boost = compute_ml_signal_boost(
            float(payload.get("deployable_alpha_utility") or 0.0), _ml_pred
        )
        _ml_adjusted_dau = _clamp(
            float(payload.get("deployable_alpha_utility") or 0.0) + _ml_boost,
            0.0,
            100.0,
        )
        payload["ml_adjusted_dau"] = round(_ml_adjusted_dau, 2)
        payload["ml_signal"] = _ml_pred
    except Exception:
        payload["ml_adjusted_dau"] = payload.get("deployable_alpha_utility")
        payload["ml_signal"] = {
            "ml_prediction": None,
            "ml_confidence": None,
            "ml_model_version": "no_model_trained",
            "ml_available": False,
        }

    return payload
