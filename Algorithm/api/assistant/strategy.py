from __future__ import annotations

from typing import Any, Dict, List, Optional


STRATEGY_ARTIFACT_KIND = "strategy_artifact"
CHAT_GROUNDING_CONTEXT_KIND = "chat_grounding_context"


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _scaled_score(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return _clamp((float(value) - 50.0) / 50.0, -1.0, 1.0)


def _tier_from_score(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 75:
        return "high"
    if value >= 60:
        return "moderate"
    if value >= 45:
        return "balanced"
    return "low"


def _fragility_tier(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 75:
        return "elevated"
    if value >= 55:
        return "moderate"
    return "contained"


def _driver_entry(label: str, score: Optional[float], detail: str) -> Dict[str, Any]:
    return {
        "label": label,
        "score": round(float(score), 2) if score is not None else None,
        "detail": detail,
    }


def build_strategy_artifact(
    *,
    job_context: Dict[str, Any],
    signal: Dict[str, Any],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    technical = data_bundle.get("technical_market_structure") or {}
    fundamentals = data_bundle.get("fundamental_filing") or {}
    sentiment = data_bundle.get("sentiment_narrative_flow") or {}
    quality = data_bundle.get("quality_provenance") or {}
    price = feature_factor_bundle.get("multi_horizon_price_momentum") or {}
    vol = feature_factor_bundle.get("volatility_risk_microstructure") or {}
    regime = feature_factor_bundle.get("regime_engine") or {}
    sentiment_factor = feature_factor_bundle.get("sentiment_intelligence") or {}
    fundamentals_factor = feature_factor_bundle.get("fundamental_intelligence") or {}
    macro = feature_factor_bundle.get("macro_sensitivity") or {}
    relative = feature_factor_bundle.get("relative_peer") or {}
    domain_agreement = feature_factor_bundle.get("domain_agreement") or {}
    composite = feature_factor_bundle.get("composite_intelligence") or {}

    base_action = str(signal.get("action") or "HOLD").upper()
    signal_score = _safe_float(signal.get("score")) or 0.0
    base_confidence = _safe_float(signal.get("confidence")) or 0.0
    regime_label = regime.get("regime_label") or "chop"
    scenario = str(job_context.get("scenario") or "base")

    trend_score = _clamp(
        0.45 * _scaled_score(price.get("momentum_consistency_score"))
        + 0.35 * _scaled_score(composite.get("Market Structure Integrity Score"))
        + 0.10 * _scaled_score(relative.get("relative_strength_percentile"))
        + 0.10 * _scaled_score(price.get("directional_persistence_score")),
        -1.0,
        1.0,
    )
    mean_reversion_score = _clamp(
        0.60 * (-_scaled_score(price.get("reversal_pressure_score") or price.get("reversal_pressure_proxy")))
        + 0.40 * (-_scaled_score(price.get("exhaustion_score"))),
        -1.0,
        1.0,
    )
    sentiment_score = _clamp(
        0.55 * _scaled_score(sentiment_factor.get("sentiment_direction_score"))
        + 0.20 * (_safe_float(sentiment_factor.get("sentiment_level")) or 0.0)
        + 0.10 * (_safe_float(sentiment_factor.get("sentiment_trend")) or 0.0)
        - 0.15 * _scaled_score(composite.get("Narrative Crowding Index")),
        -1.0,
        1.0,
    )
    macro_score = _clamp(
        0.55 * _scaled_score(macro.get("Macro Alignment Score") or macro.get("macro_alignment_score"))
        + 0.20 * _scaled_score(macro.get("growth_alignment_score"))
        - 0.25 * _scaled_score(macro.get("macro_conflict_score"))
        - 0.20 * _scaled_score(macro.get("macro_stress_fragility")),
        -1.0,
        1.0,
    )
    quality_score = _clamp(
        0.55 * _scaled_score(composite.get("Fundamental Durability Score"))
        + 0.45 * _scaled_score(composite.get("Opportunity Quality Score")),
        -1.0,
        1.0,
    )
    agreement_score = _clamp(
        0.60 * _scaled_score(domain_agreement.get("domain_agreement_score"))
        - 0.40 * _scaled_score(domain_agreement.get("domain_conflict_score")),
        -1.0,
        1.0,
    )
    fragility_veto = _clamp(
        -0.75 * _scaled_score(composite.get("Signal Fragility Index"))
        - 0.25 * _scaled_score(domain_agreement.get("domain_conflict_score")),
        -1.0,
        1.0,
    )

    weights = {
        "trend_following": 0.24,
        "mean_reversion": 0.10,
        "sentiment_aware": 0.12,
        "macro_alignment": 0.12,
        "quality_fundamental": 0.16,
        "domain_agreement": 0.12,
        "fragility_risk_veto": 0.14,
    }
    if regime_label in {"trend", "trending"}:
        weights["trend_following"] = 0.36
        weights["mean_reversion"] = 0.08
        weights["domain_agreement"] = 0.10
    elif regime_label in {"chop", "choppy", "squeeze"}:
        weights["trend_following"] = 0.22
        weights["mean_reversion"] = 0.18
        weights["domain_agreement"] = 0.10
    elif regime_label == "high_vol":
        weights["fragility_risk_veto"] = 0.22
        weights["trend_following"] = 0.24
        weights["domain_agreement"] = 0.10
    elif regime_label == "transition":
        weights["macro_alignment"] = 0.18
        weights["fragility_risk_veto"] = 0.18
        weights["domain_agreement"] = 0.14
    total_weight = sum(weights.values()) or 1.0
    weights = {label: weight / total_weight for label, weight in weights.items()}

    component_scores = {
        "trend_following": {"score": trend_score, "weight": weights["trend_following"]},
        "mean_reversion": {"score": mean_reversion_score, "weight": weights["mean_reversion"]},
        "sentiment_aware": {"score": sentiment_score, "weight": weights["sentiment_aware"]},
        "macro_alignment": {"score": macro_score, "weight": weights["macro_alignment"]},
        "quality_fundamental": {"score": quality_score, "weight": weights["quality_fundamental"]},
        "domain_agreement": {"score": agreement_score, "weight": weights["domain_agreement"]},
        "fragility_risk_veto": {"score": fragility_veto, "weight": weights["fragility_risk_veto"]},
    }

    weighted_score = sum(component["score"] * component["weight"] for component in component_scores.values())
    alignment_bonus = 0.12 * signal_score
    combined_score = _clamp(weighted_score + alignment_bonus, -1.0, 1.0)

    disagreement_penalty = 0.0
    positive_components = sum(1 for component in component_scores.values() if component["score"] > 0.15)
    negative_components = sum(1 for component in component_scores.values() if component["score"] < -0.15)
    if positive_components and negative_components:
        disagreement_penalty += 0.12
    disagreement_penalty += ((_safe_float(domain_agreement.get("confidence_penalty_from_conflict")) or 0.0) / 100.0) * 0.22
    missingness = _safe_float(quality.get("missingness")) or 0.0
    disagreement_penalty += min(max(missingness, 0.0), 0.25)

    freshness_penalty = 0.0
    freshness = quality.get("freshness_summary") or {}
    for item in freshness.values():
        status = item.get("status")
        if status == "stale_but_usable":
            freshness_penalty += 0.04
        elif status == "stale":
            freshness_penalty += 0.08

    fragility_penalty = max((_safe_float(composite.get("Signal Fragility Index")) or 50.0) - 50.0, 0.0) / 100.0
    confidence = abs(combined_score)
    confidence *= 1.0 - _clamp(disagreement_penalty + freshness_penalty + fragility_penalty * 0.5, 0.0, 0.65)
    confidence = _clamp(confidence + 0.35 * base_confidence, 0.0, 1.0)

    final_signal = base_action
    if base_action == "BUY" and combined_score < 0.12:
        final_signal = "HOLD"
    elif base_action == "SELL" and combined_score > -0.12:
        final_signal = "HOLD"
    elif base_action == "HOLD":
        if combined_score >= 0.38:
            final_signal = "BUY"
        elif combined_score <= -0.38:
            final_signal = "SELL"
    elif combined_score >= 0.55:
        final_signal = "BUY"
    elif combined_score <= -0.55:
        final_signal = "SELL"

    if scenario == "bull" and final_signal == "HOLD" and combined_score > 0.18:
        final_signal = "BUY"
    elif scenario in {"bear", "stressed"} and final_signal == "BUY" and combined_score < 0.28:
        final_signal = "HOLD"

    contributor_candidates = [
        ("Trend-following component", trend_score, "Uses regime-aware momentum persistence and market-structure integrity."),
        ("Mean-reversion component", mean_reversion_score, "Measures whether extension or oversold/overbought conditions argue against pure trend chasing."),
        ("Sentiment-aware component", sentiment_score, "Blends sentiment level, trend, and crowding."),
        ("Macro-alignment component", macro_score, "Tests whether the setup aligns with cross-asset context instead of fighting it."),
        ("Quality/fundamental component", quality_score, "Rewards durability, filing recency, and business-quality proxies."),
        ("Domain-agreement component", agreement_score, "Rewards multi-domain confirmation and penalizes cross-domain conflict."),
        ("Fragility/risk veto", fragility_veto, "Penalizes unstable, stale, or crowded setups."),
    ]
    top_contributors = [
        _driver_entry(label, score, detail)
        for label, score, detail in sorted(contributor_candidates, key=lambda item: item[1], reverse=True)
        if score > 0.05
    ][:4]
    top_detractors = [
        _driver_entry(label, score, detail)
        for label, score, detail in sorted(contributor_candidates, key=lambda item: item[1])
        if score < -0.05
    ][:4]

    confidence_degraders: List[str] = []
    if disagreement_penalty > 0.1:
        confidence_degraders.append("Cross-domain components disagree enough to degrade conviction.")
    if (_safe_float(domain_agreement.get("domain_conflict_score")) or 0.0) >= 60:
        conflict_domains = [item.get("domain") for item in (domain_agreement.get("strongest_conflicting_domains") or [])]
        if conflict_domains:
            confidence_degraders.append(
                "Strong conflicting domains are "
                + ", ".join(str(domain) for domain in conflict_domains[:3])
                + "."
            )
    if freshness_penalty > 0.05:
        confidence_degraders.append("At least one evidence domain is stale or only marginally usable.")
    if missingness > 0.1:
        confidence_degraders.append("Missing data is large enough to cap confidence.")
    if (composite.get("Signal Fragility Index") or 0.0) >= 70:
        confidence_degraders.append("Fragility metrics are elevated and trigger a risk veto.")

    participant_fit = ["wait/no-trade"]
    if final_signal == "BUY":
        participant_fit = ["swing", "tactical", "risk-managed discretionary"]
    elif final_signal == "SELL":
        participant_fit = ["tactical", "risk-managed discretionary"]
    elif confidence >= 0.55:
        participant_fit = ["position", "risk-managed discretionary"]

    base_case = (
        f"The base case is {final_signal} on the {job_context.get('horizon')} horizon, with conviction anchored to regime "
        f"{regime_label} and cross-domain score {combined_score:.2f}."
    )
    upside_case = (
        "Upside improves if structure quality stays intact, crowding remains contained, and the quality/fundamental component stops detracting."
        if final_signal != "SELL"
        else "A less defensive outcome would require the negative components to fade and the market-structure profile to stabilize."
    )
    downside_case = (
        "Downside risk rises if fragility keeps climbing, relative strength slips against peers, or the macro overlay turns more hostile."
    )
    invalidation_conditions = [
        "A regime shift from trend/contained to transition or high-volatility instability.",
        "A material deterioration in freshness, missingness, or evidence quality.",
        "A reversal in the dominant driver set, especially if peer-relative behavior turns against the thesis.",
    ]

    if final_signal == "BUY" and market.get("support_21d") is not None:
        invalidation_conditions.append(
            f"Loss of the near support zone around {market.get('support_21d'):.2f} would weaken the constructive posture."
        )
    if final_signal == "SELL" and market.get("resistance_21d") is not None:
        invalidation_conditions.append(
            f"Recovery through the near resistance zone around {market.get('resistance_21d'):.2f} would undermine the defensive view."
        )

    return {
        "final_signal": final_signal,
        "combined_score": combined_score,
        "confidence": confidence,
        "conviction_tier": _tier_from_score((confidence or 0.0) * 100.0),
        "fragility_tier": _fragility_tier(composite.get("Signal Fragility Index")),
        "participant_fit": participant_fit,
        "component_scores": component_scores,
        "top_contributors": top_contributors,
        "top_detractors": top_detractors,
        "confidence_degraders": confidence_degraders,
        "base_case": base_case,
        "upside_case": upside_case,
        "downside_case": downside_case,
        "invalidation_conditions": invalidation_conditions,
        "where_least_certain": (
            "The model is least certain where macro alignment, sentiment crowding, fragility, and domain agreement point in different directions."
            if confidence_degraders
            else "The least-certain area is whether current momentum persists without becoming too crowded or fragile."
        ),
        "raw_signal_action": base_action,
    }
