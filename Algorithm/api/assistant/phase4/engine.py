from __future__ import annotations

from typing import Any, Dict, List

from .common import (
    compact_list,
    signal_bias,
    safe_float,
    top_component_names,
)
from .components import build_strategy_components
from .conviction import (
    build_actionability_profile,
    build_confidence_profile,
    build_fragility_vetoes,
)
from .execution import build_execution_posture, build_participant_profile
from .invalidation import build_invalidation_map
from .scenarios import build_scenario_matrix


def _driver_entry(label: str, score: float, detail: str) -> Dict[str, Any]:
    return {
        "label": label,
        "score": round(float(score), 4),
        "detail": detail,
    }


def _weighted_directional_score(
    components: Dict[str, Dict[str, Any]],
    *,
    signal: Dict[str, Any],
    directional_multiplier: float,
) -> float:
    weighted = 0.0
    for component in components.values():
        weighted += (safe_float(component.get("score")) or 0.0) * (safe_float(component.get("weight")) or 0.0)
    weighted += signal_bias(signal) * 0.16
    weighted *= directional_multiplier
    if weighted > 1.0:
        return 1.0
    if weighted < -1.0:
        return -1.0
    return weighted


def _classify_posture(
    *,
    final_signal_hint: str,
    directional_score: float,
    actionability_score: float,
    confidence_score: float,
    regime_label: str,
    trend_score: float,
    mean_reversion_score: float,
    hard_veto: bool,
) -> Dict[str, str]:
    if hard_veto:
        if actionability_score < 26:
            return {"final_signal": "HOLD", "strategy_posture": "no_trade"}
        return {"final_signal": "HOLD", "strategy_posture": "fragile_hold"}

    if directional_score >= 0.44 and actionability_score >= 68 and confidence_score >= 58:
        if regime_label in {"trend", "trending", "squeeze"} and trend_score >= 0.18:
            return {"final_signal": "BUY", "strategy_posture": "trend_continuation_candidate"}
        return {"final_signal": "BUY", "strategy_posture": "actionable_long"}

    if directional_score <= -0.44 and actionability_score >= 68 and confidence_score >= 58:
        return {"final_signal": "SELL", "strategy_posture": "actionable_short"}

    if (
        mean_reversion_score >= 0.28
        and actionability_score >= 50
        and regime_label in {"chop", "choppy", "squeeze"}
        and directional_score > -0.12
    ):
        if confidence_score >= 50 and directional_score >= 0.18:
            return {"final_signal": "BUY", "strategy_posture": "opportunistic_reversal"}
        return {"final_signal": "HOLD", "strategy_posture": "opportunistic_reversal"}

    if directional_score >= 0.18:
        if actionability_score >= 60 and confidence_score >= 50 and final_signal_hint == "BUY":
            return {"final_signal": "BUY", "strategy_posture": "actionable_long"}
        return {"final_signal": "HOLD", "strategy_posture": "watchlist_positive"}

    if directional_score <= -0.18:
        if actionability_score >= 60 and confidence_score >= 50 and final_signal_hint == "SELL":
            return {"final_signal": "SELL", "strategy_posture": "actionable_short"}
        return {"final_signal": "HOLD", "strategy_posture": "watchlist_negative"}

    if actionability_score < 30 or confidence_score < 35:
        return {"final_signal": "HOLD", "strategy_posture": "no_trade"}

    return {"final_signal": "HOLD", "strategy_posture": "wait"}


def _strategy_summary(
    *,
    final_signal: str,
    strategy_posture: str,
    actionability_score: float,
    confidence_score: float,
    conviction_tier: str,
    participant_fit: List[str],
    veto_bundle: Dict[str, Any],
    regime_label: str,
) -> str:
    dampeners = veto_bundle.get("items") or []
    if dampeners:
        dampener_text = "; ".join(item.get("name", "").replace("_", " ") for item in dampeners[:3])
    else:
        dampener_text = "no active hard dampeners"
    participant_text = ", ".join(participant_fit[:2]) if participant_fit else "broad watchlist participants"
    return (
        f"The Phase 4 strategy engine sets a {final_signal} / {strategy_posture} posture with "
        f"{actionability_score:.1f} actionability, {confidence_score:.1f} confidence, and {conviction_tier} conviction. "
        f"The setup is being interpreted through a {regime_label} regime lens, fits {participant_text}, "
        f"and is being moderated by {dampener_text}."
    )


def build_strategy_artifact(
    *,
    job_context: Dict[str, Any],
    signal: Dict[str, Any],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    base_action = str(signal.get("action") or "HOLD").upper()
    component_bundle = build_strategy_components(
        job_context=job_context,
        signal=signal,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
    )
    components = component_bundle["components"]
    regime_label = component_bundle["regime_label"]

    veto_bundle = build_fragility_vetoes(
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy_components=components,
    )

    directional_score = _weighted_directional_score(
        components,
        signal=signal,
        directional_multiplier=safe_float(veto_bundle.get("directional_multiplier")) or 1.0,
    )

    actionability_profile = build_actionability_profile(
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy_components=components,
        veto_bundle=veto_bundle,
    )

    confidence_profile = build_confidence_profile(
        directional_score=directional_score,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy_components=components,
        actionability_profile=actionability_profile,
        veto_bundle=veto_bundle,
    )

    posture = _classify_posture(
        final_signal_hint=base_action,
        directional_score=directional_score,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
        confidence_score=safe_float(confidence_profile.get("confidence_score")) or 0.0,
        regime_label=regime_label,
        trend_score=safe_float((components.get("trend_following") or {}).get("score")) or 0.0,
        mean_reversion_score=safe_float((components.get("mean_reversion") or {}).get("score")) or 0.0,
        hard_veto=bool(veto_bundle.get("hard_veto")),
    )
    final_signal = posture["final_signal"]
    strategy_posture = posture["strategy_posture"]

    invalidation_map = build_invalidation_map(
        job_context=job_context,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        final_signal=final_signal,
        strategy_posture=strategy_posture,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
    )

    scenario_bundle = build_scenario_matrix(
        job_context=job_context,
        final_signal=final_signal,
        strategy_posture=strategy_posture,
        directional_score=directional_score,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
        confidence_score=safe_float(confidence_profile.get("confidence_score")) or 0.0,
        components=components,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        veto_bundle=veto_bundle,
        invalidation_map=invalidation_map,
    )

    participant_profile = build_participant_profile(
        job_context=job_context,
        final_signal=final_signal,
        strategy_posture=strategy_posture,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
        confidence_score=safe_float(confidence_profile.get("confidence_score")) or 0.0,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy_components=components,
    )
    execution_posture = build_execution_posture(
        final_signal=final_signal,
        strategy_posture=strategy_posture,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
        confidence_score=safe_float(confidence_profile.get("confidence_score")) or 0.0,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy_components=components,
        veto_bundle=veto_bundle,
    )

    top_contributors = []
    top_detractors = []
    for name, component in sorted(
        components.items(),
        key=lambda item: (safe_float(item[1].get("score")) or 0.0) * (safe_float(item[1].get("weight")) or 0.0),
        reverse=True,
    ):
        raw = safe_float(component.get("score")) or 0.0
        detail = "; ".join(component.get("notes") or [])
        if raw > 0.08 and len(top_contributors) < 4:
            top_contributors.append(_driver_entry(name.replace("_", " "), raw, detail))
    for name, component in sorted(
        components.items(),
        key=lambda item: (safe_float(item[1].get("score")) or 0.0) * (safe_float(item[1].get("weight")) or 0.0),
    ):
        raw = safe_float(component.get("score")) or 0.0
        detail = "; ".join(component.get("notes") or [])
        if raw < -0.08 and len(top_detractors) < 4:
            top_detractors.append(_driver_entry(name.replace("_", " "), raw, detail))

    strategy_summary = _strategy_summary(
        final_signal=final_signal,
        strategy_posture=strategy_posture,
        actionability_score=safe_float(actionability_profile.get("actionability_score")) or 0.0,
        confidence_score=safe_float(confidence_profile.get("confidence_score")) or 0.0,
        conviction_tier=str(confidence_profile.get("conviction_tier") or "unknown"),
        participant_fit=participant_profile.get("participant_fit") or [],
        veto_bundle=veto_bundle,
        regime_label=regime_label,
    )

    component_scores = {name: payload for name, payload in components.items()}
    scenario_matrix = scenario_bundle["scenarios"]

    return {
        "strategy_version": "phase4_institutional_v1",
        "raw_signal_action": base_action,
        "final_signal": final_signal,
        "strategy_posture": strategy_posture,
        "decision_posture": {
            "public_action": final_signal,
            "internal_posture": strategy_posture,
            "requested_scenario": scenario_bundle.get("selected_scenario"),
            "base_signal_action": base_action,
        },
        "combined_score": round(float(directional_score), 4),
        "strategy_summary": strategy_summary,
        "strategy_components": component_scores,
        "component_scores": component_scores,
        "top_contributors": top_contributors,
        "top_detractors": top_detractors,
        "base_case": scenario_matrix["base"]["summary"],
        "upside_case": scenario_matrix["bull"]["summary"],
        "downside_case": scenario_matrix["bear"]["summary"],
        "stress_case": scenario_matrix["stress"]["summary"],
        "scenario_matrix": scenario_matrix,
        "scenario_transitions": scenario_bundle["scenario_transitions"],
        "confidence_score": confidence_profile["confidence_score"],
        "confidence": confidence_profile["confidence"],
        "conviction_tier": confidence_profile["conviction_tier"],
        "confidence_quality": confidence_profile["confidence_quality"],
        "calibration_status": confidence_profile["calibration_status"],
        "calibration_meta": confidence_profile["calibration_meta"],
        "uncertainty_notes": confidence_profile["uncertainty_notes"],
        "confidence_degraders": confidence_profile["confidence_degraders"],
        "where_least_certain": " ".join((confidence_profile.get("uncertainty_notes") or [])[:2]),
        "actionability_score": actionability_profile["actionability_score"],
        "participant_fit": participant_profile["participant_fit"],
        "primary_participant_fit": participant_profile["primary_participant_fit"],
        "time_horizon_fit": participant_profile["time_horizon_fit"],
        "quality_of_setup": participant_profile["quality_of_setup"],
        "fit_notes": participant_profile["fit_notes"],
        "fragility_tier": participant_profile["fragility_tier"],
        "fragility_vetoes": veto_bundle["items"],
        "hard_veto": veto_bundle["hard_veto"],
        "veto_summary_flags": veto_bundle["summary_flags"],
        "invalidators": invalidation_map,
        "invalidation_conditions": invalidation_map["top_invalidators"],
        "confirmation_triggers": invalidation_map["confirmation_triggers"],
        "deterioration_triggers": invalidation_map["deterioration_triggers"],
        "execution_posture": execution_posture,
        "urgency_level": execution_posture["urgency_level"],
        "patience_level": execution_posture["patience_level"],
        "signal_cleanliness": execution_posture["signal_cleanliness"],
        "entry_quality_proxy": execution_posture["entry_quality_proxy"],
        "risk_context_summary": execution_posture["risk_context_summary"],
        "supporting_component_names": top_component_names(components, positive=True),
        "conflicting_component_names": top_component_names(components, positive=False),
    }
