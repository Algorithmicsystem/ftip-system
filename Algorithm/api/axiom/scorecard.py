from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import bounded_score, clamp, mean, safe_float
from api.axiom.common import inverse_score, rounded, weighted_average
from api.axiom.contracts import AxiomEngineInput, AxiomScorecard, EngineScore
from api.axiom.fusion import build_fusion_metrics


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
    fusion = build_fusion_metrics(engine_input, engine_scores)
    regime_profile = str(fusion.get("regime_weighting_profile") or "base_balance")
    gross_weight_map = {
        "transition_defensive": [
            (fundamental.score, 0.22),
            (state_pricing.score, 0.18),
            (flow.score, 0.12),
            (behavioral.score, 0.10),
            (liquidity.score, 0.12),
            (support.opportunity_quality_score, 0.10),
            (support.cross_domain_conviction_score, 0.08),
            (support.macro_alignment_score, 0.08),
        ],
        "trend_confirmation": [
            (fundamental.score, 0.20),
            (state_pricing.score, 0.22),
            (flow.score, 0.18),
            (behavioral.score, 0.16),
            (liquidity.score, 0.08),
            (support.opportunity_quality_score, 0.08),
            (support.cross_domain_conviction_score, 0.05),
            (support.macro_alignment_score, 0.03),
        ],
        "recovery_selective": [
            (fundamental.score, 0.24),
            (state_pricing.score, 0.18),
            (flow.score, 0.12),
            (behavioral.score, 0.10),
            (liquidity.score, 0.10),
            (support.opportunity_quality_score, 0.10),
            (support.cross_domain_conviction_score, 0.08),
            (support.macro_alignment_score, 0.08),
        ],
        "base_balance": [
            (fundamental.score, 0.22),
            (state_pricing.score, 0.20),
            (flow.score, 0.16),
            (behavioral.score, 0.12),
            (liquidity.score, 0.08),
            (support.opportunity_quality_score, 0.10),
            (support.cross_domain_conviction_score, 0.07),
            (support.macro_alignment_score, 0.05),
        ],
    }

    gross_opportunity = clamp(
        (
            weighted_average(gross_weight_map.get(regime_profile) or gross_weight_map["base_balance"])
            or 0.0
        )
        * (0.88 + 0.12 * ((safe_float(fusion.get("timing_support")) or 0.0) / 100.0)),
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
        safe_float(fusion.get("coverage_strength")) or 0.0,
        0.0,
        100.0,
    )
    confidence_strength = clamp(
        safe_float(fusion.get("confidence_strength")) or 0.0,
        0.0,
        100.0,
    )
    cross_engine_alignment = clamp(
        safe_float(fusion.get("cross_engine_alignment")) or 0.0,
        0.0,
        100.0,
    )
    timing_support = clamp(
        safe_float(fusion.get("timing_support")) or 0.0,
        0.0,
        100.0,
    )
    setup_maturity = clamp(
        safe_float(fusion.get("setup_maturity")) or 0.0,
        0.0,
        100.0,
    )
    mispricing_readiness = clamp(
        safe_float(fusion.get("mispricing_readiness")) or 0.0,
        0.0,
        100.0,
    )
    evidence_readiness = clamp(
        safe_float(fusion.get("evidence_readiness")) or 0.0,
        0.0,
        100.0,
    )
    path_survivability = clamp(
        safe_float(fusion.get("path_survivability")) or 0.0,
        0.0,
        100.0,
    )
    false_positive_penalty = clamp(
        safe_float(fusion.get("false_positive_penalty")) or 0.0,
        0.0,
        100.0,
    )
    exceptional_opportunity = clamp(
        safe_float(fusion.get("exceptional_opportunity")) or 0.0,
        0.0,
        100.0,
    )
    support_drag_spread = safe_float(fusion.get("support_drag_spread")) or 0.0
    event_overhang_support = clamp(
        safe_float(fusion.get("event_overhang_support")) or 0.0,
        0.0,
        100.0,
    )
    filings_change_signal = clamp(
        safe_float(fusion.get("filings_change_signal")) or 0.0,
        0.0,
        100.0,
    )
    catalyst_quality = clamp(
        safe_float(fusion.get("catalyst_quality")) or 0.0,
        0.0,
        100.0,
    )
    estimate_revision_support = clamp(
        safe_float(fusion.get("estimate_revision_support")) or 0.0,
        0.0,
        100.0,
    )
    source_strength_support = clamp(
        safe_float(fusion.get("source_strength_support")) or 0.0,
        0.0,
        100.0,
    )
    source_strength_penalty = clamp(
        safe_float(fusion.get("source_strength_penalty")) or 0.0,
        0.0,
        100.0,
    )
    premium_evidence_bonus = clamp(
        safe_float(fusion.get("premium_evidence_bonus")) or 0.0,
        0.0,
        100.0,
    )
    evidence_recency_quality = clamp(
        safe_float(fusion.get("evidence_recency_quality")) or 0.0,
        0.0,
        100.0,
    )
    raw_validated_edge = weighted_average(
        [
            (gross_opportunity, 0.18),
            (cross_engine_alignment, 0.18),
            (timing_support, 0.14),
            (mispricing_readiness, 0.14),
            (evidence_readiness, 0.12),
            (path_survivability, 0.12),
            (setup_maturity, 0.06),
            (event_overhang_support, 0.05),
            (catalyst_quality, 0.05),
            (evidence_recency_quality, 0.04),
            (source_strength_support, 0.04),
            (filings_change_signal, 0.03),
            (estimate_revision_support, 0.02),
            (bounded_score(support.signal_score, low=-1.0, high=1.0), 0.03),
            (clamp(50.0 + (support_drag_spread * 0.9), 0.0, 100.0), 0.03),
        ]
    ) or 0.0
    validated_edge = clamp(
        raw_validated_edge
        - false_positive_penalty * 0.24
        - max(55.0 - evidence_readiness, 0.0) * 0.16
        - max(55.0 - path_survivability, 0.0) * 0.18
        - max((regime_conflict_penalty or 0.0) - 50.0, 0.0) * 0.12
        - max(source_strength_penalty - 52.0, 0.0) * 0.14
        - max(52.0 - catalyst_quality, 0.0) * 0.08
        - max(50.0 - evidence_recency_quality, 0.0) * 0.08
        + max(exceptional_opportunity - 72.0, 0.0) * 0.05,
        0.0,
        100.0,
    )
    deployable_alpha_utility = clamp(
        weighted_average(
            [
                (validated_edge, 0.34),
                (path_survivability, 0.18),
                (evidence_readiness, 0.12),
                (timing_support, 0.1),
                (cross_engine_alignment, 0.1),
                (exceptional_opportunity, 0.08),
                (event_overhang_support, 0.06),
                (catalyst_quality, 0.05),
                (source_strength_support, 0.05),
                (evidence_recency_quality, 0.04),
                (research.score, 0.04),
                (liquidity.score, 0.04),
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
        - max(58.0 - path_survivability, 0.0) * 0.2
        - max(58.0 - evidence_readiness, 0.0) * 0.18
        - max(false_positive_penalty - 48.0, 0.0) * 0.26
        - max((regime_conflict_penalty or 0.0) - 52.0, 0.0) * 0.16
        - max(45.0 - cross_engine_alignment, 0.0) * 0.12
        - max(source_strength_penalty - 50.0, 0.0) * 0.18
        - max(52.0 - event_overhang_support, 0.0) * 0.12
        - max(52.0 - catalyst_quality, 0.0) * 0.08,
        0.0,
        100.0,
    )
    summary = (
        f"AXIOM scorecard sees gross opportunity {gross_opportunity:.1f}, friction burden {friction_burden:.1f}, "
        f"validated edge {validated_edge:.1f}, and deployable alpha utility {deployable_alpha_utility:.1f}. "
        f"Cross-engine alignment is {cross_engine_alignment:.1f}, timing support is {timing_support:.1f}, "
        f"mispricing readiness is {mispricing_readiness:.1f}, evidence readiness is {evidence_readiness:.1f}, "
        f"path survivability is {path_survivability:.1f}, false-positive penalty is {false_positive_penalty:.1f}, "
        f"catalyst quality is {catalyst_quality:.1f}, source-strength support is {source_strength_support:.1f}, "
        f"and evidence recency quality is {evidence_recency_quality:.1f}. "
        f"Coverage is {coverage_strength:.1f} / 100, confidence is {confidence_strength:.1f} / 100, "
        f"and regime weighting profile is {regime_profile.replace('_', ' ')}."
    )
    return AxiomScorecard(
        gross_opportunity=round(gross_opportunity, 2),
        friction_burden=round(friction_burden, 2),
        validated_edge=round(validated_edge, 2),
        deployable_alpha_utility=round(deployable_alpha_utility, 2),
        cross_engine_alignment=round(cross_engine_alignment, 2),
        timing_support=round(timing_support, 2),
        setup_maturity=round(setup_maturity, 2),
        mispricing_readiness=round(mispricing_readiness, 2),
        evidence_readiness=round(evidence_readiness, 2),
        path_survivability=round(path_survivability, 2),
        false_positive_penalty=round(false_positive_penalty, 2),
        exceptional_opportunity=round(exceptional_opportunity, 2),
        support_drag_spread=round(support_drag_spread, 2),
        event_overhang_support=round(event_overhang_support, 2),
        filings_change_signal=round(filings_change_signal, 2),
        catalyst_quality=round(catalyst_quality, 2),
        estimate_revision_support=round(estimate_revision_support, 2),
        source_strength_support=round(source_strength_support, 2),
        source_strength_penalty=round(source_strength_penalty, 2),
        premium_evidence_bonus=round(premium_evidence_bonus, 2),
        evidence_recency_quality=round(evidence_recency_quality, 2),
        regime_weighting_profile=regime_profile,
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
            "cross_engine_alignment": round(cross_engine_alignment, 2),
            "timing_support": round(timing_support, 2),
            "setup_maturity": round(setup_maturity, 2),
            "mispricing_readiness": round(mispricing_readiness, 2),
            "evidence_readiness": round(evidence_readiness, 2),
            "path_survivability": round(path_survivability, 2),
            "false_positive_penalty": round(false_positive_penalty, 2),
            "exceptional_opportunity": round(exceptional_opportunity, 2),
            "support_drag_spread": round(support_drag_spread, 2),
            "event_overhang_support": round(event_overhang_support, 2),
            "filings_change_signal": round(filings_change_signal, 2),
            "catalyst_quality": round(catalyst_quality, 2),
            "estimate_revision_support": round(estimate_revision_support, 2),
            "source_strength_support": round(source_strength_support, 2),
            "source_strength_penalty": round(source_strength_penalty, 2),
            "premium_evidence_bonus": round(premium_evidence_bonus, 2),
            "evidence_recency_quality": round(evidence_recency_quality, 2),
        },
        summary=summary,
    )
