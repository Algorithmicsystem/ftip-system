from __future__ import annotations

from typing import Any, Dict, Optional

from .common import (
    centered_score,
    compact_list,
    component_payload,
    coverage_from_payload,
    mean_defined,
)


def _component_weights(horizon: str, regime_label: str) -> Dict[str, float]:
    weights = {
        "trend_following": 0.19,
        "mean_reversion": 0.09,
        "sentiment_aware": 0.11,
        "macro_alignment": 0.12,
        "fundamental_quality": 0.14,
        "relative_strength_cross_asset": 0.12,
        "fragility_risk_veto": 0.14,
        "quality_evidence_confidence": 0.09,
    }
    horizon_text = str(horizon or "").strip().lower()
    regime = str(regime_label or "").strip().lower()

    if horizon_text in {"position", "long", "multi_week"}:
        weights["fundamental_quality"] += 0.05
        weights["macro_alignment"] += 0.02
        weights["mean_reversion"] -= 0.03
        weights["sentiment_aware"] -= 0.02
        weights["trend_following"] -= 0.02
    elif horizon_text in {"swing", "tactical", "short_term"}:
        weights["trend_following"] += 0.03
        weights["relative_strength_cross_asset"] += 0.02
        weights["sentiment_aware"] += 0.01
        weights["fundamental_quality"] -= 0.02

    if regime in {"trend", "trending"}:
        weights["trend_following"] += 0.06
        weights["relative_strength_cross_asset"] += 0.03
        weights["mean_reversion"] -= 0.04
    elif regime in {"chop", "choppy"}:
        weights["mean_reversion"] += 0.05
        weights["fragility_risk_veto"] += 0.02
        weights["trend_following"] -= 0.05
    elif regime == "squeeze":
        weights["trend_following"] += 0.02
        weights["mean_reversion"] += 0.01
        weights["fragility_risk_veto"] += 0.02
    elif regime == "high_vol":
        weights["fragility_risk_veto"] += 0.07
        weights["quality_evidence_confidence"] += 0.03
        weights["macro_alignment"] += 0.02
        weights["trend_following"] -= 0.06
        weights["mean_reversion"] -= 0.02
    elif regime == "transition":
        weights["fragility_risk_veto"] += 0.05
        weights["quality_evidence_confidence"] += 0.04
        weights["macro_alignment"] += 0.03
        weights["trend_following"] -= 0.05
        weights["mean_reversion"] -= 0.02

    total = sum(max(value, 0.02) for value in weights.values()) or 1.0
    return {key: max(value, 0.02) / total for key, value in weights.items()}


def build_strategy_components(
    *,
    job_context: Dict[str, Any],
    signal: Dict[str, Any],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    quality = data_bundle.get("quality_provenance") or {}
    domain_availability = data_bundle.get("domain_availability") or {}

    market_structure = (
        feature_factor_bundle.get("market_structure")
        or feature_factor_bundle.get("multi_horizon_price_momentum")
        or {}
    )
    regime = feature_factor_bundle.get("regime_intelligence") or feature_factor_bundle.get("regime_engine") or {}
    fragility = (
        feature_factor_bundle.get("fragility_intelligence")
        or feature_factor_bundle.get("volatility_risk_microstructure")
        or {}
    )
    sentiment = (
        feature_factor_bundle.get("sentiment_narrative_intelligence")
        or feature_factor_bundle.get("sentiment_intelligence")
        or {}
    )
    fundamentals = (
        feature_factor_bundle.get("fundamental_durability")
        or feature_factor_bundle.get("fundamental_intelligence")
        or {}
    )
    macro = feature_factor_bundle.get("macro_alignment") or feature_factor_bundle.get("macro_sensitivity") or {}
    relative = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    agreement = feature_factor_bundle.get("domain_agreement") or {}
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    regime_label = str(regime.get("regime_label") or "unknown")
    weights = _component_weights(str(job_context.get("horizon") or ""), regime_label)

    trend_support = mean_defined(
        [
            market_structure.get("trend_quality_score"),
            market_structure.get("momentum_consistency_score"),
            market_structure.get("breakout_follow_through_score"),
            regime.get("trend_quality"),
            regime.get("directional_persistence"),
            relative.get("benchmark_relative_strength"),
            relative.get("sector_confirmation_score"),
        ]
    )
    trend_penalty = mean_defined(
        [
            market_structure.get("trend_exhaustion_score"),
            market_structure.get("reversal_pressure_score"),
        ]
    )
    trend_raw = centered_score(trend_support) - max(centered_score(trend_penalty), 0.0) * 0.35

    pullback_score = mean_defined(
        [
            100.0 - (market_structure.get("support_resistance_pressure_score") or 50.0),
            100.0 - (market_structure.get("trend_exhaustion_score") or 50.0),
            fragility.get("clean_setup_score"),
            market_structure.get("range_compression_score"),
        ]
    )
    extension_penalty = mean_defined(
        [
            market_structure.get("trend_exhaustion_score"),
            sentiment.get("positive_news_weak_price_divergence"),
            sentiment.get("crowding_proxy_score"),
        ]
    )
    mean_reversion_raw = centered_score(pullback_score) - max(centered_score(extension_penalty), 0.0) * 0.30

    sentiment_support = mean_defined(
        [
            sentiment.get("sentiment_direction_score"),
            sentiment.get("sentiment_level_score"),
            sentiment.get("sentiment_trend_score"),
            sentiment.get("negative_news_resilient_price_divergence"),
        ]
    )
    sentiment_penalty = mean_defined(
        [
            sentiment.get("crowding_proxy_score"),
            sentiment.get("contradiction_score"),
            sentiment.get("hype_to_price_divergence_score"),
            sentiment.get("positive_news_weak_price_divergence"),
        ]
    )
    sentiment_raw = centered_score(sentiment_support) - max(centered_score(sentiment_penalty), 0.0) * 0.45

    macro_support = mean_defined(
        [
            macro.get("macro_alignment_score"),
            macro.get("growth_alignment_score"),
            macro.get("risk_on_risk_off_alignment"),
            macro.get("macro_regime_consistency"),
        ]
    )
    macro_penalty = mean_defined(
        [
            macro.get("macro_conflict_score"),
            macro.get("macro_fragility_score"),
            macro.get("inflation_stress_proxy"),
        ]
    )
    macro_raw = centered_score(macro_support) - max(centered_score(macro_penalty), 0.0) * 0.40

    fundamental_support = mean_defined(
        [
            composites.get("Fundamental Durability Score"),
            fundamentals.get("growth_quality_score"),
            fundamentals.get("profitability_quality_score"),
            fundamentals.get("balance_sheet_resilience_score"),
            fundamentals.get("cash_flow_durability_score"),
        ]
    )
    fundamental_penalty = mean_defined(
        [
            fundamentals.get("leverage_pressure_score"),
            100.0 - (fundamentals.get("filing_recency_score") or 50.0),
        ]
    )
    fundamental_raw = centered_score(fundamental_support) - max(centered_score(fundamental_penalty), 0.0) * 0.35

    relative_support = mean_defined(
        [
            relative.get("benchmark_relative_strength"),
            relative.get("sector_relative_strength"),
            relative.get("market_relative_momentum"),
            relative.get("sector_confirmation_score"),
            relative.get("idiosyncratic_strength_vs_market"),
        ]
    )
    relative_penalty = mean_defined(
        [
            relative.get("idiosyncratic_weakness_vs_market"),
            relative.get("cross_asset_divergence_score"),
        ]
    )
    relative_raw = centered_score(relative_support) - max(centered_score(relative_penalty), 0.0) * 0.28

    fragility_pressure = mean_defined(
        [
            composites.get("Signal Fragility Index"),
            fragility.get("instability_score"),
            regime.get("regime_instability"),
            agreement.get("domain_conflict_score"),
            composites.get("Narrative Crowding Index"),
            macro.get("macro_fragility_score"),
        ]
    )
    setup_support = mean_defined(
        [
            fragility.get("clean_setup_score"),
            regime.get("regime_confidence"),
            market_structure.get("participation_quality_score"),
            composites.get("Regime Stability Score"),
        ]
    )
    fragility_raw = centered_score(setup_support) - max(centered_score(fragility_pressure), 0.0) * 0.72
    fragility_veto_pressure = max(centered_score(fragility_pressure), 0.0)

    coverage_average = mean_defined(
        [
            coverage_from_payload({"meta": market_structure.get("meta")}),
            coverage_from_payload({"meta": fundamentals.get("meta")}),
            coverage_from_payload({"meta": sentiment.get("meta")}),
            coverage_from_payload({"meta": macro.get("meta")}),
            coverage_from_payload({"meta": relative.get("meta")}),
        ]
    )
    evidence_support = mean_defined(
        [
            (quality.get("quality_score") or 0.0),
            (1.0 - min(float(quality.get("missingness") or 0.0), 0.3)) * 100.0,
            (coverage_average or 0.0) * 100.0 if coverage_average is not None else None,
            100.0 - (agreement.get("confidence_penalty_from_conflict") or 0.0),
        ]
    )
    evidence_penalty = mean_defined(
        [
            min(float(quality.get("missingness") or 0.0), 0.3) * 100.0 * 2.0,
            12.0 if str((domain_availability.get("fundamentals") or {}).get("coverage_status") or "") in {"partial", "limited", "unavailable"} else 0.0,
            18.0 if str((domain_availability.get("cross_asset") or {}).get("coverage_status") or "") in {"limited", "unavailable"} else 0.0,
        ]
    )
    evidence_raw = centered_score(evidence_support) - max(centered_score(evidence_penalty), 0.0) * 0.55

    components = {
        "trend_following": component_payload(
            name="trend_following",
            raw_score=trend_raw,
            weight=weights["trend_following"],
            coverage=coverage_from_payload({"meta": market_structure.get("meta")}),
            rationale=compact_list(
                [
                    f"Regime is {regime_label}.",
                    "Uses multi-horizon momentum, breakout follow-through, and relative confirmation.",
                    "Trend exhaustion and reversal pressure explicitly reduce this component.",
                ]
            ),
            penalty_effect=max(centered_score(trend_penalty), 0.0),
        ),
        "mean_reversion": component_payload(
            name="mean_reversion",
            raw_score=mean_reversion_raw,
            weight=weights["mean_reversion"],
            coverage=coverage_from_payload({"meta": market_structure.get("meta")}),
            rationale=compact_list(
                [
                    "Looks for cleaner pullback / compression setups rather than late extension.",
                    "Narrative hype and extension pressure reduce mean-reversion quality.",
                ]
            ),
            penalty_effect=max(centered_score(extension_penalty), 0.0),
        ),
        "sentiment_aware": component_payload(
            name="sentiment_aware",
            raw_score=sentiment_raw,
            weight=weights["sentiment_aware"],
            coverage=coverage_from_payload({"meta": sentiment.get("meta")}),
            rationale=compact_list(
                [
                    "Blends sentiment level and trend with crowding, contradiction, and hype divergence.",
                    "Narrative strength without price confirmation is treated as a penalty.",
                ]
            ),
            penalty_effect=max(centered_score(sentiment_penalty), 0.0),
        ),
        "macro_alignment": component_payload(
            name="macro_alignment",
            raw_score=macro_raw,
            weight=weights["macro_alignment"],
            coverage=coverage_from_payload({"meta": macro.get("meta")}),
            rationale=compact_list(
                [
                    "Uses macro alignment, growth backdrop, and regime consistency rather than single-cause macro claims.",
                    "Macro conflict and macro fragility explicitly cap support.",
                ]
            ),
            penalty_effect=max(centered_score(macro_penalty), 0.0),
        ),
        "fundamental_quality": component_payload(
            name="fundamental_quality",
            raw_score=fundamental_raw,
            weight=weights["fundamental_quality"],
            coverage=coverage_from_payload({"meta": fundamentals.get("meta")}),
            rationale=compact_list(
                [
                    "Grounded in durability, profitability, balance-sheet resilience, and filing quality.",
                    "Leverage pressure and stale filing evidence reduce support.",
                ]
            ),
            penalty_effect=max(centered_score(fundamental_penalty), 0.0),
        ),
        "relative_strength_cross_asset": component_payload(
            name="relative_strength_cross_asset",
            raw_score=relative_raw,
            weight=weights["relative_strength_cross_asset"],
            coverage=coverage_from_payload({"meta": relative.get("meta")}),
            rationale=compact_list(
                [
                    "Separates idiosyncratic strength from broad beta and sector drift.",
                    "Weak market-relative behavior reduces this component quickly.",
                ]
            ),
            penalty_effect=max(centered_score(relative_penalty), 0.0),
        ),
        "fragility_risk_veto": component_payload(
            name="fragility_risk_veto",
            raw_score=fragility_raw,
            weight=weights["fragility_risk_veto"],
            coverage=coverage_from_payload({"meta": fragility.get("meta")}),
            rationale=compact_list(
                [
                    "Suppresses structurally weak setups when fragility, instability, or conflict climb.",
                    "A clean setup and stable regime can offset some veto pressure.",
                ]
            ),
            penalty_effect=fragility_veto_pressure,
            veto_effect=fragility_veto_pressure,
        ),
        "quality_evidence_confidence": component_payload(
            name="quality_evidence_confidence",
            raw_score=evidence_raw,
            weight=weights["quality_evidence_confidence"],
            coverage=coverage_average,
            rationale=compact_list(
                [
                    "Measures whether the setup is actually well-supported by fresh, complete evidence.",
                    "Coverage gaps and missingness reduce conviction before any posture is set.",
                ]
            ),
            penalty_effect=max(centered_score(evidence_penalty), 0.0),
        ),
    }

    return {
        "components": components,
        "weights": weights,
        "regime_label": regime_label,
        "market_snapshot": {
            "support_21d": market.get("support_21d"),
            "resistance_21d": market.get("resistance_21d"),
            "ret_5d": market.get("ret_5d"),
            "ret_21d": market.get("ret_21d"),
        },
        "signals": {
            "trend_support": trend_support,
            "trend_penalty": trend_penalty,
            "mean_reversion_pullback": pullback_score,
            "sentiment_penalty": sentiment_penalty,
            "macro_penalty": macro_penalty,
            "fragility_pressure": fragility_pressure,
            "evidence_support": evidence_support,
        },
    }
