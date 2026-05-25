from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from api.assistant.phase3.common import bounded_score, clamp, mean, safe_float
from api.axiom.common import inverse_score, weighted_average
from api.axiom.contracts import AxiomEngineInput, EngineScore


def _engine_value(payload: EngineScore) -> Optional[float]:
    return float(payload.score) if payload.score is not None else None


def _average_defined(values: Iterable[Optional[float]]) -> float:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return 0.0
    return float(sum(defined) / len(defined))


def _consistency_score(values: Iterable[Optional[float]]) -> float:
    defined = [float(value) for value in values if value is not None]
    if len(defined) < 2:
        return 50.0
    average = sum(defined) / len(defined)
    variance = sum((value - average) ** 2 for value in defined) / len(defined)
    dispersion = math.sqrt(max(variance, 0.0))
    return clamp(100.0 - (dispersion * 3.0), 0.0, 100.0)


def _regime_weighting_profile(engine_input: AxiomEngineInput) -> Dict[str, Any]:
    regime_hint = str(engine_input.support.regime_label or "").strip().lower()
    posture = str(engine_input.support.strategy_posture or "").strip().lower()
    maxdd_63d = safe_float(engine_input.fragility.maxdd_63d) or 0.0
    if "transition" in regime_hint:
        return {
            "label": "transition_defensive",
            "gross_weights": {
                "fundamental": 0.22,
                "state_pricing": 0.18,
                "flow": 0.12,
                "behavioral": 0.10,
                "liquidity": 0.12,
                "opportunity_quality": 0.10,
                "cross_domain_conviction": 0.08,
                "macro_alignment": 0.08,
            },
            "timing_multiplier": 0.92,
            "path_multiplier": 1.12,
            "false_positive_multiplier": 1.18,
            "fragility_multiplier": 1.18,
        }
    if "trend" in regime_hint or "continuation" in posture:
        return {
            "label": "trend_confirmation",
            "gross_weights": {
                "fundamental": 0.2,
                "state_pricing": 0.22,
                "flow": 0.18,
                "behavioral": 0.16,
                "liquidity": 0.08,
                "opportunity_quality": 0.08,
                "cross_domain_conviction": 0.05,
                "macro_alignment": 0.03,
            },
            "timing_multiplier": 1.1,
            "path_multiplier": 1.0,
            "false_positive_multiplier": 1.0,
            "fragility_multiplier": 1.0,
        }
    if maxdd_63d <= -0.15 or "recovery" in posture:
        return {
            "label": "recovery_selective",
            "gross_weights": {
                "fundamental": 0.24,
                "state_pricing": 0.18,
                "flow": 0.12,
                "behavioral": 0.1,
                "liquidity": 0.1,
                "opportunity_quality": 0.1,
                "cross_domain_conviction": 0.08,
                "macro_alignment": 0.08,
            },
            "timing_multiplier": 1.02,
            "path_multiplier": 1.06,
            "false_positive_multiplier": 1.06,
            "fragility_multiplier": 1.08,
        }
    return {
        "label": "base_balance",
        "gross_weights": {
            "fundamental": 0.22,
            "state_pricing": 0.2,
            "flow": 0.16,
            "behavioral": 0.12,
            "liquidity": 0.08,
            "opportunity_quality": 0.1,
            "cross_domain_conviction": 0.07,
            "macro_alignment": 0.05,
        },
        "timing_multiplier": 1.0,
        "path_multiplier": 1.0,
        "false_positive_multiplier": 1.0,
        "fragility_multiplier": 1.0,
    }


def build_fusion_metrics(
    engine_input: AxiomEngineInput,
    engine_scores: Dict[str, EngineScore],
) -> Dict[str, float | str]:
    support = engine_input.support
    fragility_input = engine_input.fragility
    fundamental = _engine_value(engine_scores["fundamental_reality"])
    state_pricing = _engine_value(engine_scores["state_pricing"])
    behavioral = _engine_value(engine_scores["behavioral_distortion"])
    flow = _engine_value(engine_scores["flow_transmission"])
    liquidity = _engine_value(engine_scores["liquidity_convexity"])
    fragility = _engine_value(engine_scores["critical_fragility"])
    research = _engine_value(engine_scores["research_integrity"])
    profile = _regime_weighting_profile(engine_input)

    opportunity_stack = _average_defined(
        [fundamental, state_pricing, behavioral, flow]
    )
    drag_stack = _average_defined(
        [
            fragility,
            inverse_score(liquidity),
            inverse_score(research),
            safe_float(fragility_input.implementation_fragility_score),
            safe_float(fragility_input.friction_proxy_score),
        ]
    )
    support_drag_spread = clamp(opportunity_stack - drag_stack, -100.0, 100.0)
    spread_score = clamp(50.0 + (support_drag_spread * 0.9), 0.0, 100.0)

    alignment = clamp(
        weighted_average(
            [
                (_average_defined([fundamental, state_pricing, behavioral, flow]), 0.34),
                (safe_float(support.domain_agreement_score), 0.14),
                (inverse_score(safe_float(support.domain_conflict_score)), 0.12),
                (inverse_score(safe_float(support.macro_conflict_score)), 0.1),
                (_consistency_score([fundamental, state_pricing, behavioral, flow]), 0.12),
                (safe_float(support.cross_domain_conviction_score), 0.1),
                (safe_float(support.market_structure_integrity_score), 0.08),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    timing_support = clamp(
        (weighted_average(
            [
                (state_pricing, 0.16),
                (flow, 0.16),
                (behavioral, 0.08),
                (safe_float(support.trend_quality_score), 0.1),
                (safe_float(support.momentum_consistency_score), 0.1),
                (safe_float(support.breakout_follow_through_score), 0.08),
                (safe_float(support.price_volume_alignment_score), 0.08),
                (safe_float(support.directional_persistence_score), 0.06),
                (safe_float(support.macro_alignment_score), 0.08),
                (safe_float(support.sector_confirmation_score), 0.04),
                (safe_float(support.benchmark_relative_strength_score), 0.04),
                (inverse_score(safe_float(support.reversal_pressure_score)), 0.04),
                (inverse_score(safe_float(support.trend_exhaustion_score)), 0.04),
                (inverse_score(safe_float(support.contradiction_score)), 0.02),
            ]
        )
        or 0.0)
        * float(profile["timing_multiplier"]),
        0.0,
        100.0,
    )
    setup_maturity = clamp(
        weighted_average(
            [
                (safe_float(support.opportunity_quality_score), 0.18),
                (safe_float(support.cross_domain_conviction_score), 0.14),
                (safe_float(support.market_structure_integrity_score), 0.1),
                (safe_float(support.regime_stability_score), 0.1),
                (safe_float(support.actionability_score), 0.1),
                ((safe_float(support.signal_confidence) or 0.0) * 100.0, 0.08),
                (safe_float(support.confidence_score), 0.08),
                (safe_float(support.relative_context_quality_score), 0.08),
                (safe_float(support.fundamental_durability_score), 0.08),
                (safe_float(support.execution_quality_score), 0.06),
                (safe_float(support.live_readiness_score), 0.05),
                ((safe_float(support.readiness_bucket_quality) or 0.0) * 100.0, 0.03),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    mispricing_readiness = clamp(
        weighted_average(
            [
                (fundamental, 0.18),
                (state_pricing, 0.18),
                (flow, 0.1),
                (safe_float(support.opportunity_quality_score), 0.1),
                (safe_float(support.cross_domain_conviction_score), 0.08),
                (safe_float(support.domain_agreement_score), 0.08),
                (inverse_score(safe_float(support.domain_conflict_score)), 0.08),
                (safe_float(support.fundamental_durability_score), 0.08),
                (bounded_score(safe_float(support.signal_score), low=-1.0, high=1.0), 0.06),
                (safe_float(support.negative_news_resilient_price_divergence), 0.04),
                (inverse_score(safe_float(support.hype_to_price_divergence_score)), 0.02),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    evidence_readiness = clamp(
        weighted_average(
            [
                (
                    _average_defined(
                        [
                            engine_scores["fundamental_reality"].coverage,
                            engine_scores["state_pricing"].coverage,
                            engine_scores["behavioral_distortion"].coverage,
                            engine_scores["flow_transmission"].coverage,
                            engine_scores["liquidity_convexity"].coverage,
                            engine_scores["critical_fragility"].coverage,
                            engine_scores["research_integrity"].coverage,
                            safe_float(engine_input.fundamental.coverage_score),
                            safe_float(engine_input.fragility.coverage_score),
                        ]
                    ),
                    0.34,
                ),
                (
                    _average_defined(
                        [
                            engine_scores["fundamental_reality"].confidence,
                            engine_scores["state_pricing"].confidence,
                            engine_scores["behavioral_distortion"].confidence,
                            engine_scores["flow_transmission"].confidence,
                            engine_scores["liquidity_convexity"].confidence,
                            engine_scores["critical_fragility"].confidence,
                            engine_scores["research_integrity"].confidence,
                            ((safe_float(support.signal_confidence) or 0.0) * 100.0),
                            safe_float(support.confidence_score),
                        ]
                    ),
                    0.28,
                ),
                (research, 0.14),
                (safe_float(support.quality_score), 0.08),
                (safe_float(engine_input.fundamental.reporting_completeness_score), 0.08),
                (safe_float(engine_input.fundamental.provider_confidence), 0.04),
                (safe_float(engine_input.fragility.provider_confidence), 0.04),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    path_survivability = clamp(
        (weighted_average(
            [
                (inverse_score(fragility), 0.24),
                (liquidity, 0.22),
                (research, 0.12),
                (safe_float(fragility_input.execution_cleanliness_score), 0.1),
                (safe_float(support.execution_quality_score), 0.08),
                (safe_float(support.live_readiness_score), 0.08),
                (inverse_score(safe_float(fragility_input.market_stress_score)), 0.06),
                (inverse_score(safe_float(fragility_input.implementation_fragility_score)), 0.06),
                (inverse_score(safe_float(fragility_input.friction_proxy_score)), 0.04),
            ]
        )
        or 0.0)
        * float(profile["path_multiplier"]),
        0.0,
        100.0,
    )
    momentum_outrun_penalty = clamp(
        (
            _average_defined([behavioral, flow])
            - _average_defined([fundamental, state_pricing])
        )
        * 2.25,
        0.0,
        100.0,
    )
    false_positive_penalty = clamp(
        (weighted_average(
            [
                (fragility, 0.2),
                (inverse_score(research), 0.14),
                (inverse_score(liquidity), 0.12),
                (safe_float(support.domain_conflict_score), 0.1),
                (safe_float(support.macro_conflict_score), 0.08),
                (safe_float(support.signal_fragility_index), 0.1),
                (safe_float(support.narrative_crowding_index), 0.08),
                (safe_float(support.hype_to_price_divergence_score), 0.06),
                (safe_float(support.contradiction_score), 0.04),
                (safe_float(support.positive_news_weak_price_divergence), 0.04),
                (momentum_outrun_penalty, 0.04),
            ]
        )
        or 0.0)
        * float(profile["false_positive_multiplier"]),
        0.0,
        100.0,
    )
    exceptional_opportunity = clamp(
        weighted_average(
            [
                (alignment, 0.18),
                (timing_support, 0.17),
                (mispricing_readiness, 0.17),
                (path_survivability, 0.16),
                (evidence_readiness, 0.12),
                (setup_maturity, 0.12),
                (spread_score, 0.08),
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    return {
        "regime_weighting_profile": str(profile["label"]),
        "cross_engine_alignment": round(alignment, 2),
        "timing_support": round(timing_support, 2),
        "setup_maturity": round(setup_maturity, 2),
        "mispricing_readiness": round(mispricing_readiness, 2),
        "evidence_readiness": round(evidence_readiness, 2),
        "path_survivability": round(path_survivability, 2),
        "false_positive_penalty": round(false_positive_penalty, 2),
        "exceptional_opportunity": round(exceptional_opportunity, 2),
        "support_drag_spread": round(support_drag_spread, 2),
        "spread_score": round(spread_score, 2),
        "opportunity_stack": round(opportunity_stack, 2),
        "drag_stack": round(drag_stack, 2),
        "coverage_strength": round(
            _average_defined(
                [
                    engine_scores["fundamental_reality"].coverage,
                    engine_scores["state_pricing"].coverage,
                    engine_scores["behavioral_distortion"].coverage,
                    engine_scores["flow_transmission"].coverage,
                    engine_scores["liquidity_convexity"].coverage,
                    engine_scores["critical_fragility"].coverage,
                    engine_scores["research_integrity"].coverage,
                ]
            ),
            2,
        ),
        "confidence_strength": round(
            _average_defined(
                [
                    engine_scores["fundamental_reality"].confidence,
                    engine_scores["state_pricing"].confidence,
                    engine_scores["behavioral_distortion"].confidence,
                    engine_scores["flow_transmission"].confidence,
                    engine_scores["liquidity_convexity"].confidence,
                    engine_scores["critical_fragility"].confidence,
                    engine_scores["research_integrity"].confidence,
                    ((safe_float(support.signal_confidence) or 0.0) * 100.0),
                    safe_float(support.quality_score),
                ]
            ),
            2,
        ),
    }
