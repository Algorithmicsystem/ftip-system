from __future__ import annotations

from typing import List

from api.assistant.phase3.common import clamp, mean
from api.axiom.common import inverse_score, midrange_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore


def score_liquidity_convexity(engine_input: AxiomEngineInput) -> EngineScore:
    support = engine_input.support
    fragility = engine_input.fragility
    event_risk = str(fragility.event_risk_classification or "").lower()
    option_surface_available = bool(
        engine_input.source_context.get("option_surface_available")
    )

    liquidity_integrity_component = weighted_average(
        [
            (fragility.liquidity_quality_score, 0.34),
            (fragility.execution_cleanliness_score, 0.2),
            (support.execution_quality_score, 0.18),
            (inverse_score(fragility.tradability_caution_score), 0.16),
            (inverse_score(fragility.implementation_fragility_score), 0.12),
        ]
    )
    liquidation_feasibility_component = weighted_average(
        [
            (inverse_score(fragility.implementation_fragility_score), 0.3),
            (inverse_score(fragility.friction_proxy_score), 0.24),
            (inverse_score(fragility.overnight_gap_risk_score), 0.18),
            (inverse_score(fragility.market_stress_score), 0.14),
            (support.live_readiness_score, 0.14),
        ]
    )
    hedge_feasibility_component = weighted_average(
        [
            (inverse_score(fragility.cross_asset_conflict_score), 0.28),
            (inverse_score(support.macro_conflict_score), 0.18),
            (support.relative_context_quality_score, 0.18),
            (support.sector_confirmation_score, 0.14),
            (inverse_score(fragility.market_stress_score), 0.12),
            (inverse_score(support.idiosyncratic_weakness_score), 0.1),
        ]
    )
    raw_convexity_proxy = weighted_average(
        [
            (inverse_score(fragility.implementation_fragility_score), 0.22),
            (inverse_score(fragility.event_overhang_score), 0.16),
            (support.fundamental_durability_score, 0.2),
            (inverse_score(fragility.market_stress_score), 0.18),
            (inverse_score(support.trend_exhaustion_score), 0.12),
            (support.negative_news_resilient_price_divergence, 0.12),
        ]
    )
    convexity_value_component = midrange_score(
        raw_convexity_proxy,
        center=56.0,
        tolerance=56.0,
    )
    implied_vs_fundamental_risk_component = weighted_average(
        [
            (support.fundamental_durability_score, 0.24),
            (inverse_score(fragility.event_uncertainty_score), 0.22),
            (inverse_score(fragility.market_stress_score), 0.18),
            (inverse_score(support.macro_fragility_score), 0.18),
            (inverse_score(fragility.signal_fragility_score), 0.18),
        ]
    )
    execution_penalty_component = weighted_average(
        [
            (inverse_score(fragility.friction_proxy_score), 0.26),
            (inverse_score(fragility.implementation_fragility_score), 0.24),
            (inverse_score(fragility.overnight_gap_risk_score), 0.18),
            (inverse_score(fragility.tradability_caution_score), 0.16),
            (support.execution_quality_score, 0.16),
        ]
    )
    score = weighted_average(
        [
            (liquidity_integrity_component, 0.26),
            (liquidation_feasibility_component, 0.18),
            (hedge_feasibility_component, 0.16),
            (convexity_value_component, 0.12),
            (implied_vs_fundamental_risk_component, 0.12),
            (execution_penalty_component, 0.16),
        ]
    )
    component_values = {
        "liquidity_integrity_component": rounded(liquidity_integrity_component),
        "liquidation_feasibility_component": rounded(liquidation_feasibility_component),
        "hedge_feasibility_component": rounded(hedge_feasibility_component),
        "convexity_value_component": rounded(convexity_value_component),
        "implied_vs_fundamental_risk_component": rounded(implied_vs_fundamental_risk_component),
        "execution_penalty_component": rounded(execution_penalty_component),
    }
    available_count = sum(1 for value in component_values.values() if value is not None)
    coverage = clamp(
        mean(
            [
                engine_input.partial_engine_hints.get("liquidity_convexity", 0.0),
                engine_input.domain_coverage.get("liquidity", 0.0),
                engine_input.domain_coverage.get("stress", 0.0),
                85.0 if option_surface_available else 42.0,
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
                fragility.liquidity_quality_score,
                support.execution_quality_score,
                inverse_score(fragility.implementation_fragility_score),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    flags: List[str] = []
    if not option_surface_available:
        flags.append("no_option_surface_data")
    if (fragility.implementation_fragility_score or 0.0) >= 65.0:
        flags.append("implementation_fragility")
    if (fragility.friction_proxy_score or 0.0) >= 60.0:
        flags.append("high_execution_friction")
    if event_risk in {"event_distorted", "high_event_risk"}:
        flags.append("event_convexity_noise")
    if str(support.deployment_permission or "") in {"paper_shadow_only", "watchlist_only"}:
        flags.append("deployment_constrained")

    if score is None:
        return EngineScore(
            score=None,
            confidence=round(confidence, 2),
            coverage=round(coverage, 2),
            status="unavailable" if coverage <= 0.0 else "partial",
            components={},
            flags=flags or ["liquidity_convexity_unavailable"],
            summary="Liquidity and Convexity cannot score the setup because execution and stress-survivability evidence is too thin.",
        )

    status = "available" if coverage >= 62.0 and confidence >= 52.0 else "partial"
    summary = (
        f"Liquidity and Convexity reads {rounded(score)} / 100. Liquidity integrity is {rounded(liquidity_integrity_component)} "
        f"and execution survivability is {rounded(execution_penalty_component)}. "
        + (
            "Option-surface data is not available, so convexity is being treated as a proxy-only component."
            if not option_surface_available
            else "Convexity uses direct option-surface support."
        )
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
