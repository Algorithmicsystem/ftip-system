from __future__ import annotations

from typing import Any, Dict

from .common import compact_list, conviction_tier, mean_defined, safe_float


def _posture_shift(
    *,
    final_signal: str,
    strategy_posture: str,
    scenario_name: str,
    directional_score: float,
) -> str:
    if scenario_name == "bull":
        if final_signal == "BUY":
            return "stays BUY but upgrades toward higher-conviction actionable_long"
        if directional_score >= 0:
            return f"{strategy_posture} -> BUY / actionable_long"
        return "SELL -> HOLD if bearish pressure fades"
    if scenario_name == "bear":
        if final_signal == "SELL":
            return "stays SELL with a stronger defensive posture"
        if directional_score <= 0:
            return f"{strategy_posture} -> SELL / actionable_short"
        return "BUY -> HOLD or defensive watchlist if downside pressure builds"
    if scenario_name == "stress":
        return "any constructive posture -> no_trade / avoid_due_to_fragility"
    return f"maintains {strategy_posture}"


def build_scenario_matrix(
    *,
    job_context: Dict[str, Any],
    final_signal: str,
    strategy_posture: str,
    directional_score: float,
    actionability_score: float,
    confidence_score: float,
    components: Dict[str, Dict[str, Any]],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    veto_bundle: Dict[str, Any],
    invalidation_map: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    regime = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}
    macro = feature_factor_bundle.get("macro_alignment") or feature_factor_bundle.get("macro_sensitivity") or {}
    fragility = (
        feature_factor_bundle.get("fragility_intelligence")
        or feature_factor_bundle.get("volatility_risk_microstructure")
        or {}
    )
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    trend_component = components.get("trend_following") or {}
    macro_component = components.get("macro_alignment") or {}
    relative_component = components.get("relative_strength_cross_asset") or {}
    fundamental_component = components.get("fundamental_quality") or {}
    sentiment_component = components.get("sentiment_aware") or {}
    evidence_component = components.get("quality_evidence_confidence") or {}
    fragility_component = components.get("fragility_risk_veto") or {}

    bull_confidence = mean_defined(
        [
            max(directional_score, 0.0) * 100.0,
            trend_component.get("normalized_score"),
            relative_component.get("normalized_score"),
            macro_component.get("normalized_score"),
            actionability_score,
            100.0 - (safe_float(veto_bundle.get("actionability_penalty_total")) or 0.0),
        ]
    )
    bear_confidence = mean_defined(
        [
            max(-directional_score, 0.0) * 100.0,
            safe_float(composites.get("Signal Fragility Index")),
            safe_float(agreement.get("domain_conflict_score")),
            100.0 - (safe_float(macro.get("macro_alignment_score")) or 50.0),
            100.0 - (safe_float(relative_component.get("normalized_score")) or 50.0),
        ]
    )
    stress_confidence = mean_defined(
        [
            safe_float(composites.get("Signal Fragility Index")),
            safe_float(regime.get("regime_instability")),
            safe_float(composites.get("Narrative Crowding Index")),
            safe_float(macro.get("macro_fragility_score")),
            100.0 - (safe_float(evidence_component.get("normalized_score")) or 50.0),
        ]
    )

    scenario_matrix = {
        "base": {
            "summary": (
                f"Base case stays {final_signal} / {strategy_posture} on the {job_context.get('horizon') or 'active'} horizon because "
                f"trend-following {trend_component.get('normalized_score')} / 100, macro alignment {macro_component.get('normalized_score')} / 100, "
                f"fundamental quality {fundamental_component.get('normalized_score')} / 100, and actionability {actionability_score:.1f} / 100 are outweighing the active dampeners."
            ),
            "supporting_conditions": compact_list(
                [
                    "Regime stays stable enough to carry the current posture.",
                    "Cross-domain agreement remains constructive."
                    if (safe_float(agreement.get("domain_agreement_score")) or 0.0) >= 55
                    else "Cross-domain conflict must not worsen further.",
                    "Relative strength stays supportive."
                    if (safe_float(relative_component.get("normalized_score")) or 0.0) >= 55
                    else "Relative context must stabilize before any upgrade.",
                ]
            ),
            "risk_conditions": compact_list(
                [
                    "Fragility remains the main reason the posture is not more aggressive."
                    if (safe_float(composites.get("Signal Fragility Index")) or 0.0) >= 55
                    else None,
                    "Evidence-quality support remains capped."
                    if (safe_float(evidence_component.get("normalized_score")) or 0.0) < 55
                    else None,
                    "Narrative crowding stays a live risk."
                    if (safe_float(composites.get("Narrative Crowding Index")) or 0.0) >= 55
                    else None,
                ]
            ),
            "what_needs_to_improve": compact_list(invalidation_map.get("confirmation_triggers") or []),
            "expected_posture_shift": _posture_shift(
                final_signal=final_signal,
                strategy_posture=strategy_posture,
                scenario_name="base",
                directional_score=directional_score,
            ),
            "confidence_level": round(float(confidence_score), 2),
            "confidence_tier": conviction_tier(confidence_score),
            "fragility_notes": compact_list(
                [
                    "Base-case confidence is being reduced by active dampeners."
                    if veto_bundle.get("items")
                    else "No hard veto is active in the base case.",
                    *[
                        item.get("reason")
                        for item in (veto_bundle.get("items") or [])[:2]
                    ],
                ]
            ),
        },
        "bull": {
            "summary": (
                "Bull case assumes structure remains clean, regime stability improves or stays firm, and cross-domain conflict fades enough to let the opportunity graduate into a clearer actionable-long posture."
            ),
            "supporting_conditions": compact_list(
                [
                    "Breakout follow-through improves."
                    if (safe_float(market.get("breakout_distance_63d")) or 0.0) <= 0.02
                    else "Breakout follow-through stays intact.",
                    "Macro alignment remains supportive.",
                    "Relative strength continues to confirm the move.",
                    "Crowding stays contained while sentiment remains constructive.",
                ]
            ),
            "risk_conditions": compact_list(
                [
                    "Overcrowding without better price confirmation would block the bull upgrade.",
                    "A macro wobble or new conflict pulse would cap upside posture.",
                ]
            ),
            "what_needs_to_improve": compact_list(
                [
                    "Actionability needs to move higher."
                    if actionability_score < 68
                    else "Evidence quality needs to remain stable.",
                    "Evidence quality must not slip."
                    if (safe_float(evidence_component.get("normalized_score")) or 0.0) < 60
                    else "Domain conflict should keep fading.",
                ]
            ),
            "expected_posture_shift": _posture_shift(
                final_signal=final_signal,
                strategy_posture=strategy_posture,
                scenario_name="bull",
                directional_score=directional_score,
            ),
            "confidence_level": round(float(bull_confidence or 0.0), 2),
            "confidence_tier": conviction_tier(bull_confidence),
            "fragility_notes": compact_list(
                [
                    "Bull case fails quickly if fragility re-accelerates.",
                    "Narrative heat without cleaner participation would weaken the bull path."
                    if (safe_float(sentiment_component.get("normalized_score")) or 0.0) < 55
                    else None,
                ]
            ),
        },
        "bear": {
            "summary": (
                "Bear case assumes fragility, conflict, or macro drag become dominant enough to overwhelm the current supportive factors and push the setup toward a defensive or outright bearish posture."
            ),
            "supporting_conditions": compact_list(
                [
                    "Fragility and instability continue to rise.",
                    "Macro alignment deteriorates or relative strength rolls over.",
                    "Cross-domain conflict becomes the dominant read.",
                ]
            ),
            "risk_conditions": compact_list(
                [
                    "A cleaner regime and stronger evidence stack would invalidate the bear path.",
                    "Constructive price confirmation would reduce the plausibility of the bear case.",
                ]
            ),
            "what_needs_to_improve": compact_list(
                [
                    "Fragility would need to keep rising to validate the bear case."
                    if (safe_float(fragility.get("instability_score")) or 0.0) < 70
                    else "The bear case already has fragility support.",
                    "Macro conflict would need to stay elevated."
                    if (safe_float(macro.get("macro_conflict_score")) or 0.0) < 65
                    else None,
                ]
            ),
            "expected_posture_shift": _posture_shift(
                final_signal=final_signal,
                strategy_posture=strategy_posture,
                scenario_name="bear",
                directional_score=directional_score,
            ),
            "confidence_level": round(float(bear_confidence or 0.0), 2),
            "confidence_tier": conviction_tier(bear_confidence),
            "fragility_notes": compact_list(
                [
                    "Bear probability rises quickly if regime stability fails.",
                    "Relative weakness versus the benchmark would accelerate the bear transition.",
                ]
            ),
        },
        "stress": {
            "summary": (
                "Stress / adverse case assumes regime instability, macro conflict, narrative crowding, or evidence decay all worsen at once, which would collapse the setup into a no-trade or explicit risk-avoidance posture."
            ),
            "supporting_conditions": compact_list(
                [
                    "Regime transitions into high-volatility chop or instability.",
                    "Fragility crosses into a clearly elevated state.",
                    "Evidence freshness degrades while conflicts remain unresolved.",
                ]
            ),
            "risk_conditions": compact_list(
                [
                    "Stress probability falls if evidence quality improves and fragility recedes.",
                    "Sustained constructive confirmation would neutralize the stress path.",
                ]
            ),
            "what_needs_to_improve": compact_list(
                [
                    "Fragility must stay contained to avoid the stress path.",
                    "Freshness and coverage must remain stable.",
                ]
            ),
            "expected_posture_shift": _posture_shift(
                final_signal=final_signal,
                strategy_posture=strategy_posture,
                scenario_name="stress",
                directional_score=directional_score,
            ),
            "confidence_level": round(float(stress_confidence or 0.0), 2),
            "confidence_tier": conviction_tier(stress_confidence),
            "fragility_notes": compact_list(
                [
                    "Stress case is the main reason the engine carries explicit veto logic.",
                    "Elevated crowding, instability, and stale evidence are the fastest routes to invalidation.",
                ]
            ),
        },
    }

    return {
        "selected_scenario": str(job_context.get("scenario") or "base"),
        "scenarios": scenario_matrix,
        "scenario_transitions": {
            "base_to_bull": compact_list((invalidation_map.get("confirmation_triggers") or [])[:3]),
            "base_to_bear": compact_list((invalidation_map.get("deterioration_triggers") or [])[:3]),
            "to_invalidated": compact_list((invalidation_map.get("top_invalidators") or [])[:3]),
        },
    }
