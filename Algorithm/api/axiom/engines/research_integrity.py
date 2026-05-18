from __future__ import annotations

from typing import List

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.common import inverse_score, label_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore


_CALIBRATION_STATUS_SCORE = {
    "healthy": 90.0,
    "watch": 64.0,
    "degraded": 34.0,
    "critical": 10.0,
}

_MONOTONICITY_SCORE = {
    "higher_confidence_buckets_outperform": 84.0,
    "mixed": 48.0,
    "broken": 18.0,
}

_READINESS_STATUS_SCORE = {
    "ready": 88.0,
    "constrained": 54.0,
    "shadow_only": 42.0,
    "blocked": 18.0,
    "paused": 4.0,
}

_OPERATING_MODE_SCORE = {
    "live_like": 86.0,
    "limited_live": 76.0,
    "paper_shadow": 52.0,
    "shadow_only": 44.0,
    "paused": 6.0,
}

_SOURCE_SUITABILITY_SCORE = {
    "buyer_safe": 90.0,
    "cleaner_candidate": 74.0,
    "conditional_review_required": 48.0,
    "mixed_risk": 34.0,
    "restricted": 16.0,
}


def score_research_integrity(engine_input: AxiomEngineInput) -> EngineScore:
    support = engine_input.support
    fundamental = engine_input.fundamental
    fragility = engine_input.fragility

    matured_sample_component = bounded_score(
        support.matured_prediction_count,
        low=0.0,
        high=36.0,
    )
    hit_rate_component = bounded_score(support.hit_rate, low=0.35, high=0.7)
    net_edge_component = bounded_score(
        support.validation_net_edge,
        low=-0.03,
        high=0.04,
    )
    readiness_bucket_component = bounded_score(
        support.readiness_bucket_quality,
        low=-0.02,
        high=0.08,
    )
    suppression_edge_component = bounded_score(
        support.suppression_effect_edge_spread,
        low=-0.02,
        high=0.08,
    )
    monotonicity_component = label_score(
        support.ranking_monotonicity,
        _MONOTONICITY_SCORE,
        default=44.0,
    )
    calibration_status_component = label_score(
        support.calibration_health_status,
        _CALIBRATION_STATUS_SCORE,
        default=52.0,
    )
    readiness_status_component = label_score(
        support.model_readiness_status,
        _READINESS_STATUS_SCORE,
        default=52.0,
    )
    operating_mode_component = label_score(
        support.current_operating_mode,
        _OPERATING_MODE_SCORE,
        default=48.0,
    )
    source_suitability_component = label_score(
        support.buyer_demo_suitability,
        _SOURCE_SUITABILITY_SCORE,
        default=46.0,
    )

    evidence_quality_component = weighted_average(
        [
            (support.evaluation_consistency_score, 0.24),
            (matured_sample_component, 0.16),
            (fundamental.coverage_score, 0.14),
            (support.live_readiness_score, 0.16),
            (support.confidence_score, 0.12),
            (support.quality_score, 0.18),
        ]
    )
    out_of_sample_reliability_component = weighted_average(
        [
            (support.confidence_reliability_score, 0.26),
            (hit_rate_component, 0.16),
            (net_edge_component, 0.16),
            (readiness_bucket_component, 0.16),
            (suppression_edge_component, 0.14),
            (bounded_score(support.walkforward_window_count, low=0.0, high=6.0), 0.12),
        ]
    )
    calibration_component = weighted_average(
        [
            (support.confidence_reliability_score, 0.38),
            (calibration_status_component, 0.22),
            (monotonicity_component, 0.2),
            (
                support.actionable_vs_watchlist_return_spread
                and bounded_score(
                    support.actionable_vs_watchlist_return_spread,
                    low=-0.03,
                    high=0.06,
                ),
                0.2,
            ),
        ]
    )
    coverage_integrity_component = weighted_average(
        [
            (support.quality_score, 0.24),
            (fundamental.coverage_score, 0.18),
            (fundamental.provider_confidence, 0.16),
            (readiness_status_component, 0.18),
            (source_suitability_component, 0.12),
            (inverse_score(support.commercialization_risk_score), 0.12),
        ]
    )
    drift_penalty_component = weighted_average(
        [
            (inverse_score(support.model_drift_score), 0.42),
            (label_score(support.system_health_status, _CALIBRATION_STATUS_SCORE, default=54.0), 0.18),
            (operating_mode_component, 0.18),
            (100.0 if not support.pause_required else 0.0, 0.12),
            (inverse_score(fragility.market_stress_score), 0.1),
        ]
    )
    source_governance_component = weighted_average(
        [
            (source_suitability_component, 0.34),
            (inverse_score(support.commercialization_risk_score), 0.26),
            (readiness_status_component, 0.16),
            (operating_mode_component, 0.12),
            (inverse_score(support.model_drift_score), 0.12),
        ]
    )
    score = weighted_average(
        [
            (evidence_quality_component, 0.22),
            (out_of_sample_reliability_component, 0.22),
            (calibration_component, 0.18),
            (coverage_integrity_component, 0.16),
            (drift_penalty_component, 0.12),
            (source_governance_component, 0.1),
        ]
    )

    component_values = {
        "evidence_quality_component": rounded(evidence_quality_component),
        "out_of_sample_reliability_component": rounded(out_of_sample_reliability_component),
        "calibration_component": rounded(calibration_component),
        "coverage_integrity_component": rounded(coverage_integrity_component),
        "drift_penalty_component": rounded(drift_penalty_component),
        "source_governance_component": rounded(source_governance_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                engine_input.partial_engine_hints.get("research_integrity", 0.0),
                engine_input.domain_coverage.get("quality", 0.0),
                engine_input.domain_coverage.get("fundamentals", 0.0),
                (available_count / max(len(component_values), 1)) * 100.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    confidence = clamp(
        mean(
            [
                coverage,
                support.confidence_reliability_score,
                support.evaluation_consistency_score,
                inverse_score(support.model_drift_score),
                support.live_readiness_score,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if (support.matured_prediction_count or 0.0) < 4.0:
        flags.append("thin_out_of_sample_history")
    if (support.model_drift_score or 0.0) >= 55.0:
        flags.append("active_model_drift")
    if bool(support.pause_required):
        flags.append("pause_required")
    if (support.commercialization_risk_score or 0.0) >= 60.0:
        flags.append("mixed_risk_sources")
    if str(support.model_readiness_status or "") in {"constrained", "blocked", "paused"}:
        flags.append("deployment_readiness_constrained")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0.0 else "partial",
            components={},
            flags=flags or ["research_integrity_unavailable"],
            summary="Research Integrity cannot score the setup because validation, calibration, or governance evidence is too thin.",
        )

    status = "available" if coverage >= 62.0 and confidence >= 52.0 else "partial"
    summary = (
        f"Research Integrity reads {rounded(score)} / 100. Evidence quality is {rounded(evidence_quality_component)}, "
        f"out-of-sample reliability is {rounded(out_of_sample_reliability_component)}, and drift quality is "
        f"{rounded(drift_penalty_component)}."
    )
    return EngineScore(
        score=round(score, 2),
        confidence=round(confidence, 2),
        coverage=round(coverage, 2),
        status=status,
        components={key: value for key, value in component_values.items() if value is not None},
        flags=sorted(set(flags)),
        summary=summary,
    )
