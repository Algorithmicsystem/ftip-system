from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import bool_score, clamp, compact_list, safe_float, score_value, status_score


def _coverage_score(report: Dict[str, Any]) -> float:
    quality_score = safe_float((report.get("quality") or {}).get("quality_score"))
    if quality_score is not None:
        return clamp(quality_score, 0.0, 100.0)

    availability = report.get("domain_availability") or {}
    if not availability:
        return 52.0
    scores: List[float] = []
    for entry in availability.values():
        scores.append(status_score((entry or {}).get("coverage_status")))
    if not scores:
        return 52.0
    return round(sum(scores) / len(scores), 2)


def _freshness_score(report: Dict[str, Any]) -> float:
    freshness = (report.get("freshness_summary") or {}).get("overall_status")
    return status_score(freshness)


def _fallback_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    availability = report.get("domain_availability") or {}
    total = 0
    fallback = 0
    fallback_domains: List[str] = []
    for label, entry in availability.items():
        total += 1
        if (entry or {}).get("fallback_used"):
            fallback += 1
            fallback_domains.append(str(label))
    ratio = (fallback / total) if total else 0.0
    return {
        "fallback_domain_count": fallback,
        "fallback_domain_ratio": round(ratio, 4),
        "fallback_domains": fallback_domains,
    }


def _evaluation_consistency(report: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = report.get("evaluation") or {}
    signal_overall = (evaluation.get("signal_scorecard") or {}).get("final_signal_overall") or {}
    strategy_scorecard = evaluation.get("strategy_scorecard") or {}
    calibration = evaluation.get("calibration_summary") or {}
    matured = safe_float(signal_overall.get("matured_count")) or 0.0
    hit_rate = safe_float(signal_overall.get("hit_rate"))
    actionable_spread = safe_float(strategy_scorecard.get("actionable_vs_watchlist_return_spread"))
    reliability = safe_float(calibration.get("confidence_reliability_score"))
    if reliability is None and matured <= 0:
        score = 34.0
    else:
        score = 32.0
        if hit_rate is not None:
            score += clamp(hit_rate * 38.0, 0.0, 38.0)
        if actionable_spread is not None:
            score += clamp(actionable_spread * 420.0, -12.0, 18.0)
        if reliability is not None:
            score = (score * 0.55) + (reliability * 0.45)
    return {
        "matured_prediction_count": int(matured),
        "hit_rate": hit_rate,
        "actionable_vs_watchlist_return_spread": actionable_spread,
        "confidence_reliability_score": reliability,
        "confidence_monotonicity": calibration.get("confidence_monotonicity"),
        "consistency_score": round(clamp(score, 0.0, 100.0), 2),
    }


def _degradation_matches(report: Dict[str, Any]) -> List[str]:
    evaluation = report.get("evaluation") or {}
    weakest = evaluation.get("weakest_conditions") or []
    regime_label = ((report.get("regime_intelligence") or {}).get("regime_label") or "").lower()
    fragility_tier = str((report.get("strategy") or {}).get("fragility_tier") or "").lower()
    freshness_status = str((report.get("freshness_summary") or {}).get("overall_status") or "").lower()
    matches: List[str] = []
    for item in weakest:
        dimension = str(item.get("dimension") or "").lower()
        label = str(item.get("label") or "").lower()
        if dimension == "regime_label" and label == regime_label:
            matches.append(f"historical weakness is concentrated in the current regime ({label})")
        if dimension == "fragility_tier" and label == fragility_tier:
            matches.append(f"historical weakness is concentrated in the current fragility tier ({label})")
        if dimension == "freshness_quality" and label == freshness_status:
            matches.append(f"historical weakness is concentrated in the current freshness state ({label})")
    return matches


def build_model_readiness(
    report: Dict[str, Any],
    *,
    deployment_mode_state: Dict[str, Any],
    prior_audit_summary: Dict[str, Any],
) -> Dict[str, Any]:
    quality = report.get("quality") or {}
    strategy = report.get("strategy") or {}
    agreement = report.get("domain_agreement") or {}
    proprietary_scores = report.get("proprietary_scores") or {}

    freshness_score = _freshness_score(report)
    coverage_score = _coverage_score(report)
    fallback = _fallback_summary(report)
    evaluation_consistency = _evaluation_consistency(report)
    calibration_score = evaluation_consistency["confidence_reliability_score"]
    if calibration_score is None:
        calibration_score = 36.0

    confidence_score = safe_float(strategy.get("confidence_score")) or 0.0
    actionability_score = safe_float(strategy.get("actionability_score")) or 0.0
    fragility_score = score_value(proprietary_scores.get("Signal Fragility Index")) or 55.0
    agreement_score = safe_float(agreement.get("domain_agreement_score")) or 50.0
    conflict_score = safe_float(agreement.get("domain_conflict_score")) or 50.0
    agreement_quality = clamp(agreement_score - (conflict_score * 0.55), 0.0, 100.0)
    source_health_score = round(
        (
            bool_score(quality.get("bars_ok"), yes=100.0, no=15.0)
            + bool_score(quality.get("news_ok"), yes=86.0, no=40.0)
            + bool_score(quality.get("sentiment_ok"), yes=82.0, no=35.0)
            + bool_score(quality.get("fundamentals_ok"), yes=86.0, no=46.0)
        )
        / 4.0,
        2,
    )
    degradation_matches = _degradation_matches(report)

    live_readiness_score = round(
        (
            freshness_score * 0.17
            + coverage_score * 0.17
            + calibration_score * 0.16
            + evaluation_consistency["consistency_score"] * 0.16
            + source_health_score * 0.12
            + confidence_score * 0.08
            + actionability_score * 0.06
            + agreement_quality * 0.08
            + (100.0 - fragility_score) * 0.10
        ),
        2,
    )

    blockers: List[str] = []
    notes: List[str] = []
    degradation_flags: List[str] = list(degradation_matches)

    if deployment_mode_state.get("active_mode") == "paused":
        blockers.append("the platform is currently in paused mode, so live-use support is disabled")
    if freshness_score < 45.0:
        blockers.append("freshness is too weak for disciplined deployment support")
    elif freshness_score < 70.0:
        notes.append("freshness is usable but not clean enough for aggressive trust")

    if coverage_score < 50.0:
        blockers.append("coverage quality is too thin or inconsistent for live trust")
    elif coverage_score < 68.0:
        notes.append("coverage quality is only partial, which should cap trust")

    if calibration_score < 50.0:
        blockers.append("confidence calibration quality is not strong enough for live escalation")
    elif calibration_score < 65.0:
        notes.append("confidence calibration remains only moderate")

    if evaluation_consistency["matured_prediction_count"] < 8:
        blockers.append("similar setups do not yet have enough matured observations for live escalation")
    elif evaluation_consistency["matured_prediction_count"] < 14:
        notes.append("matured sample size is still building and should keep review tight")

    if fragility_score >= 68.0:
        blockers.append("signal fragility is elevated enough to block live trust")
    elif fragility_score >= 54.0:
        notes.append("fragility remains elevated and should reduce deployment appetite")

    if conflict_score >= 62.0:
        blockers.append("domain conflict is too high for clean deployment support")
    elif conflict_score >= 48.0:
        notes.append("domain conflict is not dominant, but it still weakens trust")

    if fallback["fallback_domain_ratio"] >= 0.5:
        blockers.append("fallback usage is too concentrated across core domains")
        degradation_flags.append("core-domain fallback usage is elevated")
    elif fallback["fallback_domain_ratio"] >= 0.25:
        notes.append("fallback usage is elevated enough to warrant extra review")

    if degradation_matches:
        notes.extend(degradation_matches)

    if safe_float(quality.get("missingness")) is not None and safe_float(quality.get("missingness")) >= 0.15:
        blockers.append("missingness is too high for live capital support")
    elif safe_float(quality.get("missingness")) is not None and safe_float(quality.get("missingness")) >= 0.07:
        notes.append("missingness is non-trivial and should keep deployment cautious")

    if prior_audit_summary.get("recent_pause_recommendation_count", 0) >= 2:
        degradation_flags.append("recent audit history has repeated pause recommendations")
        blockers.append("recent deployment audits show repeated pause conditions")
    elif prior_audit_summary.get("recent_blocked_count", 0) >= 3:
        notes.append("recent audit history has repeatedly blocked deployment")

    if blockers:
        status = "blocked" if live_readiness_score < 55.0 else "constrained"
    elif live_readiness_score >= 78.0:
        status = "ready"
    elif live_readiness_score >= 63.0:
        status = "cautious"
    else:
        status = "constrained"

    evidence_quality_summary = (
        f"Readiness is {status} at {live_readiness_score:.1f} / 100, with freshness {freshness_score:.1f}, "
        f"coverage {coverage_score:.1f}, calibration {calibration_score:.1f}, consistency {evaluation_consistency['consistency_score']:.1f}, "
        f"and inverse fragility {(100.0 - fragility_score):.1f}. "
        f"Fallback usage spans {fallback['fallback_domain_count']} domains and recent degradation flags are {compact_list(degradation_flags) or ['none']}."
    )

    return {
        "model_readiness_status": status,
        "model_readiness_notes": compact_list(notes, limit=8),
        "live_readiness_score": live_readiness_score,
        "live_readiness_blockers": compact_list(blockers, limit=8),
        "recent_degradation_flags": compact_list(degradation_flags, limit=8),
        "evidence_quality_summary": evidence_quality_summary,
        "component_scores": {
            "freshness_score": round(freshness_score, 2),
            "coverage_score": round(coverage_score, 2),
            "calibration_score": round(calibration_score, 2),
            "evaluation_consistency_score": evaluation_consistency["consistency_score"],
            "source_health_score": round(source_health_score, 2),
            "agreement_quality_score": round(agreement_quality, 2),
            "inverse_fragility_score": round(100.0 - fragility_score, 2),
        },
        "evaluation_consistency": evaluation_consistency,
        "fallback_summary": fallback,
    }
