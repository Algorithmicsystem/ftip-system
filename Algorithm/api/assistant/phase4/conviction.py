from __future__ import annotations

from typing import Any, Dict, List

from .common import (
    availability_penalty,
    centered_score,
    clamp,
    compact_list,
    confidence_quality_label,
    conviction_tier,
    fragility_tier,
    freshness_penalty,
    mean_defined,
    safe_float,
)


def build_fragility_vetoes(
    *,
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    strategy_components: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    quality = data_bundle.get("quality_provenance") or {}
    domain_availability = data_bundle.get("domain_availability") or {}
    market_structure = (
        feature_factor_bundle.get("market_structure")
        or feature_factor_bundle.get("multi_horizon_price_momentum")
        or {}
    )
    sentiment = (
        feature_factor_bundle.get("sentiment_narrative_intelligence")
        or feature_factor_bundle.get("sentiment_intelligence")
        or {}
    )
    macro = feature_factor_bundle.get("macro_alignment") or feature_factor_bundle.get("macro_sensitivity") or {}
    relative = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    fundamentals = (
        feature_factor_bundle.get("fundamental_durability")
        or feature_factor_bundle.get("fundamental_intelligence")
        or {}
    )
    regime = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    items: List[Dict[str, Any]] = []

    def add_veto(
        name: str,
        severity: str,
        reason: str,
        *,
        actionability_penalty: float,
        confidence_penalty: float,
        directional_multiplier: float = 1.0,
        hard_veto: bool = False,
    ) -> None:
        items.append(
            {
                "name": name,
                "severity": severity,
                "reason": reason,
                "actionability_penalty": round(float(actionability_penalty), 2),
                "confidence_penalty": round(float(confidence_penalty), 2),
                "directional_multiplier": round(float(directional_multiplier), 2),
                "hard_veto": hard_veto,
            }
        )

    fragility_score = safe_float(composites.get("Signal Fragility Index")) or 50.0
    regime_stability = safe_float(composites.get("Regime Stability Score")) or 50.0
    domain_conflict = safe_float(agreement.get("domain_conflict_score")) or 0.0
    narrative_crowding = safe_float(composites.get("Narrative Crowding Index")) or 50.0
    macro_alignment = safe_float(composites.get("Macro Alignment Score")) or 50.0
    macro_conflict = safe_float(macro.get("macro_conflict_score")) or 0.0
    fundamental_durability = safe_float(composites.get("Fundamental Durability Score")) or 50.0
    structure_integrity = safe_float(composites.get("Market Structure Integrity Score")) or 50.0
    missingness = min(safe_float(quality.get("missingness")) or 0.0, 0.3)
    freshness_risk = freshness_penalty(quality)
    relative_quality = safe_float(relative.get("relative_context_quality")) or 50.0

    if fragility_score >= 72 and regime_stability <= 45:
        add_veto(
            "high_fragility_unstable_regime",
            "high",
            "Fragility is elevated while regime stability is weak, so actionability is capped until the setup cleans up.",
            actionability_penalty=24,
            confidence_penalty=18,
            directional_multiplier=0.62,
            hard_veto=True,
        )
    if fragility_score >= 65 and regime_stability <= 40 and missingness >= 0.14:
        add_veto(
            "fragility_plus_evidence_break",
            "high",
            "Moderately extreme fragility combined with weak regime stability and poor evidence quality is enough to force a no-trade style veto.",
            actionability_penalty=18,
            confidence_penalty=15,
            directional_multiplier=0.68,
            hard_veto=True,
        )
    if domain_conflict >= 65:
        add_veto(
            "domain_conflict_dampener",
            "medium",
            "Cross-domain agreement is too weak to carry a high-conviction posture without further confirmation.",
            actionability_penalty=14,
            confidence_penalty=12,
            directional_multiplier=0.82,
        )
    if narrative_crowding >= 70 and (
        (safe_float(sentiment.get("positive_news_weak_price_divergence")) or 0.0) >= 55
        or (safe_float(market_structure.get("breakout_follow_through_score")) or 50.0) < 48
    ):
        add_veto(
            "crowded_narrative_without_price_confirmation",
            "medium",
            "Narrative intensity is outpacing clean price confirmation, which keeps the setup closer to watchlist than full actionability.",
            actionability_penalty=12,
            confidence_penalty=9,
            directional_multiplier=0.86,
        )
    if structure_integrity >= 65 and fundamental_durability <= 45:
        add_veto(
            "technical_strength_vs_weak_fundamentals",
            "medium",
            "Technical quality is stronger than the fundamental durability stack, so the posture is downgraded until durability catches up.",
            actionability_penalty=10,
            confidence_penalty=9,
            directional_multiplier=0.90,
        )
    if macro_alignment <= 45 or macro_conflict >= 65:
        add_veto(
            "macro_misalignment",
            "medium",
            "The setup is fighting the macro backdrop harder than the engine is willing to ignore.",
            actionability_penalty=11,
            confidence_penalty=10,
            directional_multiplier=0.88,
        )
    if freshness_risk >= 12 or missingness >= 0.14:
        add_veto(
            "evidence_staleness_or_missingness",
            "high" if freshness_risk >= 18 or missingness >= 0.18 else "medium",
            "Freshness or missingness has deteriorated enough to reduce trust in the computed posture.",
            actionability_penalty=16,
            confidence_penalty=16,
            directional_multiplier=0.78,
            hard_veto=freshness_risk >= 18 and missingness >= 0.14,
        )
    if relative_quality < 45:
        add_veto(
            "weak_relative_context",
            "low",
            "Benchmark / sector context is too weak to support aggressive posture sizing.",
            actionability_penalty=7,
            confidence_penalty=6,
            directional_multiplier=0.92,
        )

    explicit_availability_penalty = sum(
        [
            availability_penalty(domain_availability, "fundamentals"),
            availability_penalty(domain_availability, "cross_asset"),
        ]
    )
    if explicit_availability_penalty >= 15:
        add_veto(
            "thin_domain_coverage",
            "medium",
            "Key supporting domains are thin enough that the engine treats the posture as provisional.",
            actionability_penalty=10,
            confidence_penalty=11,
            directional_multiplier=0.88,
        )

    hard_veto = any(item["hard_veto"] for item in items)
    actionability_penalty_total = clamp(
        sum(float(item["actionability_penalty"]) for item in items),
        0.0,
        48.0,
    )
    confidence_penalty_total = clamp(
        sum(float(item["confidence_penalty"]) for item in items),
        0.0,
        45.0,
    )
    directional_multiplier = 1.0
    for item in items:
        directional_multiplier *= float(item["directional_multiplier"])
    directional_multiplier = clamp(directional_multiplier, 0.45, 1.0)

    return {
        "items": items,
        "hard_veto": hard_veto,
        "actionability_penalty_total": round(actionability_penalty_total, 2),
        "confidence_penalty_total": round(confidence_penalty_total, 2),
        "directional_multiplier": round(directional_multiplier, 4),
        "summary_flags": compact_list(item["name"] for item in items),
        "fragility_tier": fragility_tier(feature_factor_bundle.get("composite_intelligence", {}).get("Signal Fragility Index")),
        "dampened_components": [
            name
            for name, component in strategy_components.items()
            if (safe_float(component.get("score")) or 0.0) > 0.0 and (safe_float(component.get("weight")) or 0.0) >= 0.1
        ],
    }


def build_actionability_profile(
    *,
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    strategy_components: Dict[str, Dict[str, Any]],
    veto_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    quality = data_bundle.get("quality_provenance") or {}
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}
    regime = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}

    evidence_component = strategy_components.get("quality_evidence_confidence") or {}
    actionability_base = mean_defined(
        [
            composites.get("Opportunity Quality Score"),
            composites.get("Market Structure Integrity Score"),
            composites.get("Regime Stability Score"),
            composites.get("Cross-Domain Conviction Score"),
            100.0 - (safe_float(composites.get("Signal Fragility Index")) or 50.0),
            100.0 - (safe_float(composites.get("Narrative Crowding Index")) or 50.0),
            100.0 - (safe_float(agreement.get("domain_conflict_score")) or 0.0),
            evidence_component.get("normalized_score"),
        ]
    )
    actionability = clamp(
        (actionability_base or 50.0) - (safe_float(veto_bundle.get("actionability_penalty_total")) or 0.0),
        0.0,
        100.0,
    )
    if veto_bundle.get("hard_veto"):
        actionability = min(actionability, 34.0)

    if actionability >= 76:
        quality_of_setup = "clean"
    elif actionability >= 60:
        quality_of_setup = "selective"
    elif actionability >= 42:
        quality_of_setup = "fragile"
    else:
        quality_of_setup = "low_quality"

    return {
        "actionability_score": round(actionability, 2),
        "quality_of_setup": quality_of_setup,
        "actionability_base": round(float(actionability_base or 0.0), 2),
        "regime_label": regime.get("regime_label"),
        "degraded_by": compact_list(item["reason"] for item in (veto_bundle.get("items") or [])),
        "quality_missingness": round(float(min(safe_float(quality.get("missingness")) or 0.0, 1.0)), 4),
    }


def build_confidence_profile(
    *,
    directional_score: float,
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    strategy_components: Dict[str, Dict[str, Any]],
    actionability_profile: Dict[str, Any],
    veto_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    quality = data_bundle.get("quality_provenance") or {}
    domain_availability = data_bundle.get("domain_availability") or {}
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}
    relative = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    evidence_component = strategy_components.get("quality_evidence_confidence") or {}

    fragility_score = safe_float(composites.get("Signal Fragility Index")) or 50.0
    conflict_score = safe_float(agreement.get("domain_conflict_score")) or 0.0
    regime_stability = safe_float(composites.get("Regime Stability Score")) or 50.0
    cross_domain_conviction = safe_float(composites.get("Cross-Domain Conviction Score")) or 50.0
    evidence_quality = safe_float(evidence_component.get("normalized_score")) or 50.0
    scenario_stability = mean_defined(
        [
            regime_stability,
            100.0 - fragility_score,
            100.0 - conflict_score,
            evidence_quality,
        ]
    )

    confidence_base = mean_defined(
        [
            abs(directional_score) * 100.0,
            actionability_profile.get("actionability_score"),
            cross_domain_conviction,
            regime_stability,
            evidence_quality,
            scenario_stability,
        ]
    )

    degraders: List[str] = []
    penalties = []

    fresh_penalty = freshness_penalty(quality)
    if fresh_penalty > 0:
        penalties.append(fresh_penalty)
        degraders.append("Stale or only marginally usable inputs are lowering confidence.")

    missingness = min(safe_float(quality.get("missingness")) or 0.0, 0.3)
    if missingness > 0:
        penalties.append(missingness * 55.0)
        if missingness >= 0.1:
            degraders.append("Missingness is high enough to cap confidence.")

    if str((domain_availability.get("fundamentals") or {}).get("coverage_status") or "") in {
        "partial",
        "limited",
        "unavailable",
    }:
        penalties.append(7.0)
        degraders.append("Fundamental coverage is incomplete, so durability support is treated conservatively.")

    if str((domain_availability.get("cross_asset") or {}).get("coverage_status") or "") in {
        "limited",
        "unavailable",
    }:
        penalties.append(6.0)
        degraders.append("Weak benchmark / cross-asset context is reducing confidence quality.")

    if conflict_score >= 55:
        penalties.append((conflict_score - 50.0) * 0.35)
        degraders.append("Domain conflict is high enough to materially degrade conviction.")

    if fragility_score >= 55:
        penalties.append((fragility_score - 50.0) * 0.40)
        degraders.append("Fragility and instability are actively suppressing confidence.")

    if (safe_float(relative.get("relative_context_quality")) or 50.0) < 50:
        penalties.append((50.0 - (safe_float(relative.get("relative_context_quality")) or 50.0)) * 0.18)

    penalties.append(safe_float(veto_bundle.get("confidence_penalty_total")) or 0.0)

    confidence_score = clamp((confidence_base or 50.0) - sum(penalties), 5.0, 95.0)
    confidence_quality = confidence_quality_label(
        confidence_score,
        evidence_quality=evidence_quality,
        fragility=fragility_score,
    )
    calibration_status = (
        "degraded_provisional"
        if confidence_quality in {"fragile", "low_evidence"} or sum(penalties) >= 25
        else "probability_like_provisional"
    )

    if confidence_score < 42:
        degraders.append("Scenario stability is too weak for a fully actionable conviction tier.")

    uncertainty_notes = compact_list(
        degraders
        + [
            f"Scenario stability reads {round(float(scenario_stability or 0.0), 1)} / 100."
            if scenario_stability is not None
            else None,
            "Empirical calibration remains provisional until a dedicated evaluation layer is attached.",
        ]
    )

    return {
        "confidence_score": round(confidence_score, 2),
        "confidence": round(confidence_score / 100.0, 4),
        "conviction_tier": conviction_tier(confidence_score),
        "confidence_quality": confidence_quality,
        "calibration_status": calibration_status,
        "calibration_meta": {
            "method": "probability_like_normalization",
            "probability_like_confidence": round(confidence_score / 100.0, 4),
            "reliability_bucket": conviction_tier(confidence_score),
            "needs_empirical_calibration": True,
            "scenario_stability_score": round(float(scenario_stability or 0.0), 2),
        },
        "confidence_degraders": compact_list(degraders),
        "uncertainty_notes": uncertainty_notes,
        "scenario_stability_score": round(float(scenario_stability or 0.0), 2)
        if scenario_stability is not None
        else None,
    }
