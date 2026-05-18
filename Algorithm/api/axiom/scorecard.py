from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import bounded_score, clamp, mean, safe_float
from api.axiom.common import inverse_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, AxiomScorecard, EngineScore


def build_axiom_scorecard(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
) -> AxiomScorecard:
    fundamental = engine_scores["fundamental_reality"]
    state_pricing = engine_scores["state_pricing"]
    behavioral = engine_scores["behavioral_distortion"]
    flow = engine_scores["flow_transmission"]
    liquidity = engine_scores["liquidity_convexity"]
    fragility = engine_scores["critical_fragility"]
    research = engine_scores["research_integrity"]
    support = engine_input.support

    gross_opportunity = clamp(
        weighted_average(
            [
                (fundamental.score, 0.26),
                (state_pricing.score, 0.2),
                (flow.score, 0.18),
                (behavioral.score, 0.14),
                (liquidity.score, 0.08),
                (support.opportunity_quality_score, 0.07),
                (support.cross_domain_conviction_score, 0.04),
                (support.macro_alignment_score, 0.03),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    coverage_gap_penalty = clamp(
        100.0
        - (
            mean(
                [
                    fundamental.coverage,
                    state_pricing.coverage,
                    behavioral.coverage,
                    flow.coverage,
                    liquidity.coverage,
                    fragility.coverage,
                    research.coverage,
                ]
            )
            or 0.0
        ),
        0.0,
        100.0,
    )
    confidence_gap_penalty = clamp(
        100.0
        - (
            mean(
                [
                    fundamental.confidence,
                    state_pricing.confidence,
                    behavioral.confidence,
                    flow.confidence,
                    liquidity.confidence,
                    fragility.confidence,
                    research.confidence,
                    (safe_float(support.signal_confidence) or 0.0) * 100.0,
                ]
            )
            or 0.0
        ),
        0.0,
        100.0,
    )
    regime_conflict_penalty = mean(
        [
            safe_float(support.domain_conflict_score),
            safe_float(support.macro_conflict_score),
            safe_float(engine_input.fragility.cross_asset_conflict_score),
            safe_float(engine_input.fragility.regime_transition_score),
        ]
    )
    friction_burden = clamp(
        weighted_average(
            [
                (fragility.score, 0.32),
                (inverse_score(liquidity.score), 0.2),
                (inverse_score(research.score), 0.2),
                (safe_float(engine_input.fragility.implementation_fragility_score), 0.1),
                (safe_float(engine_input.fragility.friction_proxy_score), 0.08),
                (coverage_gap_penalty, 0.05),
                (confidence_gap_penalty, 0.03),
                (regime_conflict_penalty, 0.02),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    coverage_strength = clamp(
        mean(
            [
                fundamental.coverage,
                state_pricing.coverage,
                behavioral.coverage,
                flow.coverage,
                liquidity.coverage,
                fragility.coverage,
                research.coverage,
                safe_float(engine_input.fundamental.coverage_score),
                safe_float(engine_input.fragility.coverage_score),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    confidence_strength = clamp(
        mean(
            [
                fundamental.confidence,
                state_pricing.confidence,
                behavioral.confidence,
                flow.confidence,
                liquidity.confidence,
                fragility.confidence,
                research.confidence,
                (safe_float(support.signal_confidence) or 0.0) * 100.0,
                safe_float(support.quality_score),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    raw_validated_edge = weighted_average(
        [
            (gross_opportunity, 0.42),
            (inverse_score(friction_burden), 0.18),
            (research.score, 0.16),
            (liquidity.score, 0.08),
            (bounded_score(support.signal_score, low=-1.0, high=1.0), 0.08),
            (inverse_score(regime_conflict_penalty), 0.08),
        ]
    ) or 0.0
    validated_edge = clamp(
        raw_validated_edge
        * (0.35 + 0.65 * (coverage_strength / 100.0))
        * (0.32 + 0.68 * (confidence_strength / 100.0))
        * (0.4 + 0.6 * ((research.score or 0.0) / 100.0)),
        0.0,
        100.0,
    )
    deployable_alpha_utility = clamp(
        weighted_average(
            [
                (validated_edge, 0.48),
                (research.score, 0.18),
                (liquidity.score, 0.12),
                (flow.score, 0.08),
                (state_pricing.score, 0.08),
                (inverse_score(fragility.score), 0.06),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    deployable_alpha_utility = clamp(
        deployable_alpha_utility
        - max((safe_float(fragility.score) or 0.0) - 50.0, 0.0) * 0.42
        - max(58.0 - (safe_float(liquidity.score) or 0.0), 0.0) * 0.24
        - max(60.0 - (safe_float(research.score) or 0.0), 0.0) * 0.28
        - max(55.0 - coverage_strength, 0.0) * 0.2
        - max(58.0 - confidence_strength, 0.0) * 0.18
        - max((regime_conflict_penalty or 0.0) - 52.0, 0.0) * 0.18,
        0.0,
        100.0,
    )
    summary = (
        f"AXIOM scorecard sees gross opportunity {gross_opportunity:.1f}, friction burden {friction_burden:.1f}, "
        f"validated edge {validated_edge:.1f}, and deployable alpha utility {deployable_alpha_utility:.1f}. "
        f"Coverage is {coverage_strength:.1f} / 100, confidence is {confidence_strength:.1f} / 100, "
        f"and research integrity is {rounded(research.score) if research.score is not None else 'n/a'}."
    )
    return AxiomScorecard(
        gross_opportunity=round(gross_opportunity, 2),
        friction_burden=round(friction_burden, 2),
        validated_edge=round(validated_edge, 2),
        deployable_alpha_utility=round(deployable_alpha_utility, 2),
        overall_coverage=round(coverage_strength, 2),
        overall_confidence=round(confidence_strength, 2),
        component_support={
            "fundamental_reality": round(fundamental.score or 0.0, 2)
            if fundamental.score is not None
            else 0.0,
            "state_pricing": round(state_pricing.score or 0.0, 2)
            if state_pricing.score is not None
            else 0.0,
            "behavioral_distortion": round(behavioral.score or 0.0, 2)
            if behavioral.score is not None
            else 0.0,
            "flow_transmission": round(flow.score or 0.0, 2)
            if flow.score is not None
            else 0.0,
            "liquidity_convexity": round(liquidity.score or 0.0, 2)
            if liquidity.score is not None
            else 0.0,
            "critical_fragility": round(fragility.score or 0.0, 2)
            if fragility.score is not None
            else 0.0,
            "research_integrity": round(research.score or 0.0, 2)
            if research.score is not None
            else 0.0,
            "signal_confidence": round(
                (safe_float(support.signal_confidence) or 0.0) * 100.0,
                2,
            ),
            "regime_conflict_penalty": round(regime_conflict_penalty or 0.0, 2),
        },
        summary=summary,
    )
