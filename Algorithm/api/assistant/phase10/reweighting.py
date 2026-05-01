from __future__ import annotations

from typing import Any, Dict, List

from .common import monotonicity_strength, overfit_risk, safe_float, sample_confidence, slugify


_PENALTY_SCORES = {
    "Signal Fragility Index": "fragility_penalty",
    "Narrative Crowding Index": "crowding_penalty",
}


def _sample_size(item: Dict[str, Any]) -> int:
    results = item.get("bucket_results") or []
    return int(sum(int(result.get("matured_count") or 0) for result in results))


def build_reweighting_candidates(
    current_report: Dict[str, Any],
    regime_learning: Dict[str, Any],
    drift_alerts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evaluation = current_report.get("evaluation") or {}
    attribution = evaluation.get("factor_attribution_summary") or {}
    calibration = evaluation.get("calibration_summary") or {}
    score_items = attribution.get("proprietary_score_attribution") or []
    component_items = attribution.get("strategy_component_attribution") or []
    candidates: List[Dict[str, Any]] = []

    for item in score_items[:6]:
        score_name = str(item.get("score_name") or "")
        spread = abs(safe_float(item.get("favorable_vs_unfavorable_return_spread")) or 0.0)
        sample_size = _sample_size(item)
        if sample_size < 4 or spread < 0.015:
            continue
        monotonicity = str(item.get("monotonicity") or "mixed")
        confidence = 0.35 + min(0.45, spread * 6.0) + (0.15 * monotonicity_strength(monotonicity))
        confidence = round(min(0.95, confidence), 2)
        if score_name in _PENALTY_SCORES:
            change = {
                "target": _PENALTY_SCORES[score_name],
                "direction": "increase_penalty",
                "magnitude": 0.03 if sample_size < 10 else 0.05,
            }
            impact_area = "strategy_scoring_and_readiness_gates"
        else:
            change = {
                "target": score_name,
                "direction": "increase_weight" if spread > 0 else "decrease_weight",
                "magnitude": 0.02 if sample_size < 10 else 0.04,
            }
            impact_area = "strategy_scoring_and_ranking"
        candidates.append(
            {
                "reweighting_candidate": f"reweight_{slugify(score_name)}",
                "target_family": score_name,
                "suggested_weight_changes": [change],
                "rationale": (
                    f"{score_name} shows {monotonicity} bucket behavior with return spread {spread:.4f}, suggesting that its influence should be adjusted more deliberately."
                ),
                "sample_size": sample_size,
                "sample_confidence": sample_confidence(sample_size),
                "confidence_in_recommendation": confidence,
                "risk_of_overfit": overfit_risk(sample_size),
                "expected_impact_area": impact_area,
                "approval_required": True,
            }
        )

    for item in component_items[:4]:
        component_name = str(item.get("score_name") or "")
        spread = abs(safe_float(item.get("favorable_vs_unfavorable_return_spread")) or 0.0)
        sample_size = _sample_size(item)
        if sample_size < 4 or spread < 0.015:
            continue
        candidates.append(
            {
                "reweighting_candidate": f"reweight_component_{slugify(component_name)}",
                "target_family": component_name,
                "suggested_weight_changes": [
                    {
                        "target": component_name,
                        "direction": "increase_weight",
                        "magnitude": 0.02 if sample_size < 10 else 0.03,
                    }
                ],
                "rationale": (
                    f"{component_name} is emerging as a stronger differentiator inside the current attribution set."
                ),
                "sample_size": sample_size,
                "sample_confidence": sample_confidence(sample_size),
                "confidence_in_recommendation": round(min(0.9, 0.32 + spread * 5.0), 2),
                "risk_of_overfit": overfit_risk(sample_size),
                "expected_impact_area": "strategy_component_blend",
                "approval_required": True,
            }
        )

    reliability = safe_float(calibration.get("confidence_reliability_score"))
    if reliability is not None and reliability < 60.0:
        candidates.append(
            {
                "reweighting_candidate": "tighten_confidence_gates",
                "target_family": "confidence_and_deployment_gating",
                "suggested_weight_changes": [
                    {
                        "target": "confidence_to_conviction_mapping",
                        "direction": "tighten_thresholds",
                        "magnitude": 0.05,
                    },
                    {
                        "target": "live_readiness_gate",
                        "direction": "increase_minimum_threshold",
                        "magnitude": 0.04,
                    },
                ],
                "rationale": (
                    f"Confidence reliability is {reliability}, so calibration should carry more gatekeeping influence before live-readiness and conviction escalation."
                ),
                "sample_size": int(
                    sum(
                        int(item.get("matured_count") or 0)
                        for item in (calibration.get("bucketed_confidence_stats") or [])
                    )
                ),
                "sample_confidence": sample_confidence(
                    int(
                        sum(
                            int(item.get("matured_count") or 0)
                            for item in (calibration.get("bucketed_confidence_stats") or [])
                        )
                    )
                ),
                "confidence_in_recommendation": 0.78,
                "risk_of_overfit": "moderate",
                "expected_impact_area": "confidence_calibration_and_deployment_controls",
                "approval_required": True,
            }
        )

    if regime_learning.get("current_regime_is_weak"):
        note = regime_learning.get("current_regime_note") or {}
        regime_label = note.get("regime_label") or "current_regime"
        candidates.append(
            {
                "reweighting_candidate": f"regime_guard_{slugify(regime_label)}",
                "target_family": "regime_stability_and_fragility",
                "suggested_weight_changes": [
                    {
                        "target": "regime_stability_influence",
                        "direction": "increase_weight",
                        "magnitude": 0.04,
                    },
                    {
                        "target": "fragility_penalty",
                        "direction": "increase_penalty",
                        "magnitude": 0.03,
                    },
                ],
                "rationale": (
                    f"The active regime {regime_label} is currently a weaker condition, so regime stability and fragility should matter more in that context."
                ),
                "sample_size": int(note.get("sample_size") or 0),
                "sample_confidence": sample_confidence(int(note.get("sample_size") or 0)),
                "confidence_in_recommendation": 0.71,
                "risk_of_overfit": overfit_risk(int(note.get("sample_size") or 0)),
                "expected_impact_area": "regime_conditioning",
                "approval_required": True,
            }
        )

    if drift_alerts:
        candidates.append(
            {
                "reweighting_candidate": "deployment_drift_caution",
                "target_family": "deployment_and_ranking_suppression",
                "suggested_weight_changes": [
                    {
                        "target": "deployment_permission_rank",
                        "direction": "increase_weight",
                        "magnitude": 0.03,
                    },
                    {
                        "target": "portfolio_candidate_suppression_when_drifted",
                        "direction": "increase_penalty",
                        "magnitude": 0.02,
                    },
                ],
                "rationale": "Active drift alerts suggest that deployment and ranking logic should respond faster to reliability degradation.",
                "sample_size": len(drift_alerts),
                "sample_confidence": "limited",
                "confidence_in_recommendation": 0.66,
                "risk_of_overfit": "moderate",
                "expected_impact_area": "portfolio_and_deployment_controls",
                "approval_required": True,
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item.get("confidence_in_recommendation") or 0.0),
            item.get("sample_size") or 0,
        ),
        reverse=True,
    )
    return candidates[:8]
