from __future__ import annotations

from typing import Any, Dict

from .common import compact_list, fragility_tier, mean_defined, safe_float


def build_participant_profile(
    *,
    job_context: Dict[str, Any],
    final_signal: str,
    strategy_posture: str,
    actionability_score: float,
    confidence_score: float,
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    strategy_components: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
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
    relative = (
        feature_factor_bundle.get("cross_asset_relative_context")
        or feature_factor_bundle.get("relative_peer")
        or {}
    )
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    trend = strategy_components.get("trend_following") or {}
    mean_reversion = strategy_components.get("mean_reversion") or {}
    regime_label = str(regime.get("regime_label") or "unknown")
    horizon = str(job_context.get("horizon") or "swing")

    primary_fit = "wait / observe"
    fit_labels = ["wait / observe"]
    time_horizon_fit = f"{horizon}_observe"

    if strategy_posture in {"actionable_long", "trend_continuation_candidate"}:
        primary_fit = "swing trader"
        fit_labels = ["swing trader", "trend-continuation participant"]
        time_horizon_fit = f"{horizon}_trend"
        if (safe_float(composites.get("Fundamental Durability Score")) or 0.0) >= 65 and confidence_score >= 60:
            fit_labels.append("position trader")
    elif strategy_posture == "actionable_short":
        primary_fit = "tactical discretionary"
        fit_labels = ["tactical discretionary", "risk-managed short participant"]
        time_horizon_fit = f"{horizon}_defensive"
    elif strategy_posture == "opportunistic_reversal":
        primary_fit = "mean-reversion participant"
        fit_labels = ["mean-reversion participant", "tactical discretionary"]
        time_horizon_fit = f"{horizon}_reversal"
    elif strategy_posture in {"watchlist_positive", "watchlist_negative"}:
        primary_fit = "event-sensitive / catalyst watch"
        fit_labels = ["event-sensitive / catalyst watch", "wait / observe"]
        time_horizon_fit = f"{horizon}_watch"
    elif strategy_posture in {"fragile_hold", "no_trade", "wait"}:
        primary_fit = "no-trade / low quality" if actionability_score < 35 else "wait / observe"
        fit_labels = [primary_fit]
        time_horizon_fit = f"{horizon}_observe"

    if regime_label in {"chop", "choppy"} and (safe_float(mean_reversion.get("score")) or 0.0) > 0.12:
        fit_labels.append("mean-reversion participant")
    if (safe_float(sentiment.get("event_pressure_score")) or 0.0) >= 60:
        fit_labels.append("event-sensitive / catalyst watch")

    quality_of_setup = "low_quality"
    if actionability_score >= 75 and confidence_score >= 65:
        quality_of_setup = "clean"
    elif actionability_score >= 58:
        quality_of_setup = "selective"
    elif actionability_score >= 40:
        quality_of_setup = "fragile"

    return {
        "primary_participant_fit": primary_fit,
        "participant_fit": compact_list(fit_labels),
        "time_horizon_fit": time_horizon_fit,
        "quality_of_setup": quality_of_setup,
        "regime_label": regime_label,
        "fragility_tier": fragility_tier(
            feature_factor_bundle.get("composite_intelligence", {}).get("Signal Fragility Index")
        ),
        "fit_notes": compact_list(
            [
                "Trend-following evidence is dominant."
                if (safe_float(trend.get("score")) or 0.0) > 0.2
                else None,
                "Mean-reversion evidence is dominant."
                if (safe_float(mean_reversion.get("score")) or 0.0) > 0.2 and regime_label in {"chop", "choppy", "squeeze"}
                else None,
                "Event sensitivity is elevated."
                if (safe_float(sentiment.get("event_pressure_score")) or 0.0) >= 60
                else None,
                "Relative context is strong enough for active participation."
                if (safe_float(relative.get("relative_context_quality")) or 0.0) >= 60
                else None,
            ]
        ),
    }


def build_execution_posture(
    *,
    final_signal: str,
    strategy_posture: str,
    actionability_score: float,
    confidence_score: float,
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
    strategy_components: Dict[str, Dict[str, Any]],
    veto_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    market = data_bundle.get("market_price_volume") or {}
    market_structure = (
        feature_factor_bundle.get("market_structure")
        or feature_factor_bundle.get("multi_horizon_price_momentum")
        or {}
    )
    fragility = (
        feature_factor_bundle.get("fragility_intelligence")
        or feature_factor_bundle.get("volatility_risk_microstructure")
        or {}
    )
    composites = feature_factor_bundle.get("composite_intelligence") or {}

    entry_quality_proxy = mean_defined(
        [
            market_structure.get("breakout_follow_through_score"),
            market_structure.get("participation_quality_score"),
            fragility.get("clean_setup_score"),
            100.0 - (safe_float(composites.get("Narrative Crowding Index")) or 50.0),
        ]
    )
    signal_cleanliness = "noisy"
    if (entry_quality_proxy or 0.0) >= 72:
        signal_cleanliness = "clean"
    elif (entry_quality_proxy or 0.0) >= 56:
        signal_cleanliness = "workable"
    elif (entry_quality_proxy or 0.0) >= 42:
        signal_cleanliness = "fragile"

    if veto_bundle.get("hard_veto") or strategy_posture == "no_trade":
        preferred_posture = "avoid_due_to_fragility"
    elif final_signal == "HOLD" and strategy_posture in {"watchlist_positive", "watchlist_negative", "wait"}:
        preferred_posture = "wait_for_confirmation"
    elif actionability_score >= 70 and confidence_score >= 62 and signal_cleanliness == "clean":
        preferred_posture = "immediate_action"
    else:
        preferred_posture = "staged_watch"

    urgency_level = "low"
    if preferred_posture == "immediate_action":
        urgency_level = "measured_high"
    elif preferred_posture == "wait_for_confirmation":
        urgency_level = "measured"
    elif preferred_posture == "avoid_due_to_fragility":
        urgency_level = "none"

    patience_level = "high"
    if preferred_posture == "immediate_action":
        patience_level = "moderate"
    elif strategy_posture in {"actionable_short", "actionable_long"} and signal_cleanliness == "fragile":
        patience_level = "high"
    elif preferred_posture == "avoid_due_to_fragility":
        patience_level = "very_high"

    risk_context_summary = (
        f"Signal cleanliness is {signal_cleanliness}, entry quality reads {round(float(entry_quality_proxy or 0.0), 1)} / 100, "
        f"and fragility sits at {round(float(safe_float(composites.get('Signal Fragility Index')) or 0.0), 1)} / 100. "
        f"The engine therefore prefers {preferred_posture.replace('_', ' ')} with {urgency_level.replace('_', ' ')} urgency."
    )

    stop_invalidation_frame = None
    target_continuation_frame = None
    if final_signal == "BUY" and safe_float(market.get("support_21d")) is not None:
        stop_invalidation_frame = (
            f"Constructive posture weakens if price loses the nearby support reference around {safe_float(market.get('support_21d')):.2f}."
        )
    elif final_signal == "SELL" and safe_float(market.get("resistance_21d")) is not None:
        stop_invalidation_frame = (
            f"Defensive posture weakens if price reclaims the nearby resistance reference around {safe_float(market.get('resistance_21d')):.2f}."
        )
    if final_signal == "BUY":
        target_continuation_frame = "Continuation quality depends on cleaner follow-through and relative confirmation rather than pure extension."
    elif final_signal == "SELL":
        target_continuation_frame = "Continuation quality depends on fragility and relative weakness remaining dominant."
    else:
        target_continuation_frame = "Graduation out of HOLD depends on actionability improving rather than on raw signal magnitude alone."

    return {
        "preferred_posture": preferred_posture,
        "urgency_level": urgency_level,
        "patience_level": patience_level,
        "signal_cleanliness": signal_cleanliness,
        "entry_quality_proxy": round(float(entry_quality_proxy or 0.0), 2),
        "risk_context_summary": risk_context_summary,
        "stop_invalidation_frame": stop_invalidation_frame,
        "target_continuation_frame": target_continuation_frame,
        "execution_notes": compact_list(
            [
                "Hard veto active: posture should not be treated as immediately actionable."
                if veto_bundle.get("hard_veto")
                else None,
                "Confirmation is preferred over urgency because the setup is not fully clean."
                if preferred_posture == "wait_for_confirmation"
                else None,
                "Staged watch is preferred because the setup is selective rather than fully clean."
                if preferred_posture == "staged_watch"
                else None,
            ]
        ),
    }
