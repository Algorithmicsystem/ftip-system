from __future__ import annotations

from typing import Any, Dict, List

from .common import safe_float, score_value


def build_signal_admission_control(
    report: Dict[str, Any],
    *,
    model_readiness: Dict[str, Any],
    deployment_mode_state: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    quality = report.get("quality") or {}
    agreement = report.get("domain_agreement") or {}
    proprietary_scores = report.get("proprietary_scores") or {}
    evaluation_consistency = model_readiness.get("evaluation_consistency") or {}

    readiness_score = safe_float(model_readiness.get("live_readiness_score")) or 0.0
    confidence_score = safe_float(strategy.get("confidence_score")) or 0.0
    actionability_score = safe_float(strategy.get("actionability_score")) or 0.0
    fragility_score = score_value(proprietary_scores.get("Signal Fragility Index")) or 100.0
    conflict_score = safe_float(agreement.get("domain_conflict_score")) or 100.0
    calibration_score = safe_float(evaluation_consistency.get("confidence_reliability_score")) or 0.0
    matured_predictions = int(evaluation_consistency.get("matured_prediction_count") or 0)

    reasons: List[str] = []
    primary_rejection = None

    admitted_for_strategy = readiness_score >= 30.0 and bool(quality.get("bars_ok", True))
    if not admitted_for_strategy:
        primary_rejection = "rejected_due_to_data_quality"
        reasons.append("core market-data quality is not strong enough for even internal strategy admission")

    admitted_for_paper = admitted_for_strategy and readiness_score >= 45.0 and fragility_score < 80.0
    if admitted_for_strategy and not admitted_for_paper:
        primary_rejection = primary_rejection or "rejected_due_to_weak_evidence"
        reasons.append("the setup can be analyzed, but evidence quality is not strong enough for disciplined paper admission")

    admitted_for_live = (
        admitted_for_paper
        and readiness_score >= 72.0
        and confidence_score >= 58.0
        and actionability_score >= 55.0
        and fragility_score <= 50.0
        and conflict_score <= 45.0
        and calibration_score >= 58.0
        and matured_predictions >= 8
    )
    if admitted_for_paper and not admitted_for_live:
        if fragility_score > 50.0:
            primary_rejection = primary_rejection or "rejected_due_to_fragility"
            reasons.append("fragility remains too high for live admission")
        if conflict_score > 45.0:
            primary_rejection = primary_rejection or "rejected_due_to_conflict"
            reasons.append("domain conflict remains above the live-admission tolerance")
        if calibration_score < 58.0 or matured_predictions < 8:
            primary_rejection = primary_rejection or "rejected_due_to_weak_evidence"
            reasons.append("evaluation and calibration support are not yet strong enough for live admission")

    if deployment_mode_state.get("active_mode") == "paused":
        primary_rejection = "rejected_due_to_paused_mode"
        reasons.append("the current deployment mode is paused")

    return {
        "admitted_for_strategy": admitted_for_strategy,
        "admitted_for_paper": admitted_for_paper,
        "admitted_for_live": admitted_for_live,
        "primary_decision": (
            "admitted_for_live"
            if admitted_for_live
            else "admitted_for_paper"
            if admitted_for_paper
            else "admitted_for_strategy"
            if admitted_for_strategy
            else primary_rejection or "rejected_due_to_data_quality"
        ),
        "rejection_reasons": reasons,
        "rejection_code": primary_rejection,
        "minimum_thresholds": {
            "live_readiness_score": readiness_score,
            "confidence_score": confidence_score,
            "actionability_score": actionability_score,
            "fragility_score": fragility_score,
            "domain_conflict_score": conflict_score,
            "calibration_score": calibration_score,
            "matured_prediction_count": matured_predictions,
        },
    }
