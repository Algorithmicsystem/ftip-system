from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, Dict, Iterable, List, Optional


CONTINUOUS_LEARNING_ARTIFACT_KIND = "assistant_continuous_learning_artifact"
CONTINUOUS_LEARNING_VERSION = "phase10_alpha_acceleration_v1"


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def compact_list(values: Iterable[Any], *, limit: int = 6) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = (
                value.get("label")
                or value.get("name")
                or value.get("title")
                or value.get("domain")
                or value.get("score_name")
                or str(value)
            )
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items


def score_value(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        return safe_float(payload.get("score"))
    return safe_float(payload)


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return text.strip("_") or "item"


def sample_confidence(sample_size: int) -> str:
    if sample_size >= 16:
        return "high"
    if sample_size >= 8:
        return "moderate"
    if sample_size >= 4:
        return "limited"
    return "exploratory"


def overfit_risk(sample_size: int, *, repeated: bool = False) -> str:
    if repeated and sample_size >= 10:
        return "moderate"
    if sample_size >= 16:
        return "low"
    if sample_size >= 8:
        return "moderate"
    return "high"


def macro_state(score: Any) -> str:
    number = safe_float(score)
    if number is None:
        return "mixed"
    if number >= 62.0:
        return "supportive"
    if number <= 42.0:
        return "conflicted"
    return "mixed"


def crowding_state(score: Any) -> str:
    number = safe_float(score)
    if number is None:
        return "balanced"
    if number >= 60.0:
        return "crowded"
    if number <= 40.0:
        return "uncrowded"
    return "balanced"


def fragility_state(score: Any) -> str:
    number = safe_float(score)
    if number is None:
        return "mixed"
    if number >= 60.0:
        return "fragile"
    if number <= 40.0:
        return "clean"
    return "mixed"


def monotonicity_strength(label: str) -> float:
    if label == "higher_confidence_buckets_outperform":
        return 1.0
    if label == "mixed":
        return 0.45
    if label == "lower_confidence_buckets_outperform":
        return 0.1
    return 0.25


def build_learning_snapshot(
    report: Dict[str, Any],
    *,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    proprietary_scores = report.get("proprietary_scores") or {}
    evaluation = report.get("evaluation") or {}
    calibration = evaluation.get("calibration_summary") or {}
    signal_scorecard = evaluation.get("signal_scorecard") or {}
    regime = report.get("regime_intelligence") or {}
    agreement = report.get("domain_agreement") or {}

    opportunity_quality = score_value(proprietary_scores.get("Opportunity Quality Score")) or 0.0
    fragility = score_value(proprietary_scores.get("Signal Fragility Index")) or 100.0
    crowding = score_value(proprietary_scores.get("Narrative Crowding Index")) or 50.0
    macro_alignment = score_value(proprietary_scores.get("Macro Alignment Score")) or 50.0
    regime_stability = score_value(proprietary_scores.get("Regime Stability Score")) or 50.0
    fundamental_durability = (
        score_value(proprietary_scores.get("Fundamental Durability Score")) or 50.0
    )
    cross_domain = (
        score_value(proprietary_scores.get("Cross-Domain Conviction Score")) or 50.0
    )
    market_structure = (
        score_value(proprietary_scores.get("Market Structure Integrity Score")) or 50.0
    )
    evaluation_reliability = safe_float(calibration.get("confidence_reliability_score")) or 50.0
    evaluation_hit_rate = safe_float(
        (signal_scorecard.get("final_signal_overall") or {}).get("hit_rate")
    )

    return {
        "report_id": report_id or report.get("report_id"),
        "session_id": session_id or report.get("session_id"),
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "final_signal": strategy.get("final_signal")
        or (report.get("signal") or {}).get("final_action")
        or (report.get("signal") or {}).get("action"),
        "strategy_posture": strategy.get("strategy_posture") or report.get("strategy_posture"),
        "conviction_tier": strategy.get("conviction_tier"),
        "confidence_score": safe_float(
            strategy.get("confidence_score") or report.get("confidence_score")
        )
        or 0.0,
        "actionability_score": safe_float(
            strategy.get("actionability_score") or report.get("actionability_score")
        )
        or 0.0,
        "deployment_permission": report.get("deployment_permission") or "analysis_only",
        "trust_tier": report.get("trust_tier") or "blocked",
        "live_readiness_score": safe_float(report.get("live_readiness_score")) or 0.0,
        "candidate_classification": report.get("candidate_classification") or "watchlist_candidate",
        "portfolio_candidate_score": safe_float(report.get("portfolio_candidate_score")) or 0.0,
        "portfolio_fit_quality": safe_float(report.get("portfolio_fit_quality")) or 0.0,
        "overlap_score": safe_float(report.get("overlap_score")) or 0.0,
        "redundancy_score": safe_float(report.get("redundancy_score")) or 0.0,
        "diversification_contribution_score": safe_float(
            report.get("diversification_contribution_score")
        )
        or 0.0,
        "size_band": report.get("size_band") or "watchlist only",
        "opportunity_quality_score": opportunity_quality,
        "cross_domain_conviction_score": cross_domain,
        "signal_fragility_index": fragility,
        "narrative_crowding_index": crowding,
        "macro_alignment_score": macro_alignment,
        "macro_state": macro_state(macro_alignment),
        "regime_label": regime.get("regime_label")
        or (report.get("key_features") or {}).get("regime_label")
        or "unknown",
        "regime_stability_score": regime_stability,
        "fundamental_durability_score": fundamental_durability,
        "market_structure_integrity_score": market_structure,
        "domain_agreement_score": safe_float(agreement.get("domain_agreement_score")) or 0.0,
        "domain_conflict_score": safe_float(agreement.get("domain_conflict_score")) or 0.0,
        "freshness_status": (report.get("freshness_summary") or {}).get("overall_status") or "unknown",
        "evaluation_reliability_score": evaluation_reliability,
        "evaluation_hit_rate": evaluation_hit_rate,
        "crowding_state": crowding_state(crowding),
        "fragility_state": fragility_state(fragility),
        "report_version": report.get("report_version"),
        "strategy_version": report.get("strategy_version") or strategy.get("strategy_version"),
    }
