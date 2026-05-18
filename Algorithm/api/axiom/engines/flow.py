from __future__ import annotations

from typing import List

from api.assistant.phase3.common import bounded_score, clamp, mean
from api.axiom.common import inverse_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore


def score_flow_transmission(engine_input: AxiomEngineInput) -> EngineScore:
    support = engine_input.support
    fragility = engine_input.fragility

    trend_quality_component = weighted_average(
        [
            (support.trend_quality_score, 0.32),
            (support.momentum_consistency_score, 0.24),
            (support.breakout_follow_through_score, 0.22),
            (support.market_structure_integrity_score, 0.22),
        ]
    )
    transmission_strength_component = weighted_average(
        [
            (bounded_score(support.ret_21d, low=-0.15, high=0.25), 0.22),
            (bounded_score(support.mom_vol_adj_21d, low=-0.8, high=1.2), 0.22),
            (support.price_volume_alignment_score, 0.2),
            (support.cross_domain_conviction_score, 0.18),
            (support.opportunity_quality_score, 0.18),
        ]
    )
    breadth_confirmation_component = weighted_average(
        [
            (fragility.breadth_confirmation_score, 0.34),
            (support.sector_confirmation_score, 0.24),
            (support.benchmark_relative_strength_score, 0.16),
            (support.sector_relative_strength_score, 0.14),
            (support.relative_context_quality_score, 0.12),
        ]
    )
    market_structure_component = weighted_average(
        [
            (support.market_structure_integrity_score, 0.34),
            (support.regime_stability_score, 0.2),
            (support.domain_agreement_score, 0.18),
            (inverse_score(fragility.regime_transition_score), 0.14),
            (inverse_score(fragility.regime_instability_score), 0.14),
        ]
    )
    flow_persistence_component = weighted_average(
        [
            (support.directional_persistence_score, 0.3),
            (support.relative_context_quality_score, 0.18),
            (support.idiosyncratic_strength_score, 0.16),
            (inverse_score(support.reversal_pressure_score), 0.18),
            (inverse_score(support.trend_exhaustion_score), 0.18),
        ]
    )
    conflict_penalty_component = weighted_average(
        [
            (inverse_score(support.domain_conflict_score), 0.28),
            (inverse_score(support.macro_conflict_score), 0.24),
            (inverse_score(fragility.cross_asset_conflict_score), 0.2),
            (inverse_score(fragility.market_stress_score), 0.16),
            (inverse_score(fragility.regime_transition_score), 0.12),
        ]
    )
    score = weighted_average(
        [
            (trend_quality_component, 0.22),
            (transmission_strength_component, 0.22),
            (breadth_confirmation_component, 0.16),
            (market_structure_component, 0.16),
            (flow_persistence_component, 0.14),
            (conflict_penalty_component, 0.1),
        ]
    )
    component_values = {
        "trend_quality_component": rounded(trend_quality_component),
        "transmission_strength_component": rounded(transmission_strength_component),
        "breadth_confirmation_component": rounded(breadth_confirmation_component),
        "market_structure_component": rounded(market_structure_component),
        "flow_persistence_component": rounded(flow_persistence_component),
        "conflict_penalty_component": rounded(conflict_penalty_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                engine_input.partial_engine_hints.get("flow_transmission", 0.0),
                engine_input.domain_coverage.get("market", 0.0),
                engine_input.domain_coverage.get("breadth", 0.0),
                engine_input.domain_coverage.get("cross_asset", 0.0),
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
                support.domain_agreement_score,
                support.market_structure_integrity_score,
                support.regime_stability_score,
                inverse_score(fragility.signal_fragility_score),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if (support.trend_exhaustion_score or 0.0) >= 65.0:
        flags.append("trend_exhaustion")
    if (support.reversal_pressure_score or 0.0) >= 60.0:
        flags.append("reversal_pressure")
    if (fragility.cross_asset_conflict_score or 0.0) >= 60.0:
        flags.append("cross_asset_conflict")
    if (fragility.breadth_confirmation_score or 100.0) <= 40.0:
        flags.append("weak_breadth_confirmation")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0.0 else "partial",
            components={},
            flags=flags or ["flow_transmission_unavailable"],
            summary="Flow Transmission cannot score the setup because market-structure, breadth, or relative-flow evidence is too thin.",
        )

    status = "available" if coverage >= 66.0 and confidence >= 55.0 else "partial"
    summary = (
        f"Flow Transmission reads {rounded(score)} / 100. Trend quality is {rounded(trend_quality_component)}, "
        f"transmission strength is {rounded(transmission_strength_component)}, and breadth confirmation is {rounded(breadth_confirmation_component)}."
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
