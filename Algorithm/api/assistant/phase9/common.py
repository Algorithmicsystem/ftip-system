from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional


PORTFOLIO_CONSTRUCTION_ARTIFACT_KIND = "assistant_portfolio_construction_artifact"
PORTFOLIO_CONSTRUCTION_VERSION = "phase9_portfolio_construction_v1"


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
            label = value.get("label") or value.get("name") or value.get("domain") or str(value)
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


def permission_rank(permission: Any) -> float:
    mapping = {
        "scaled_live_eligible": 96.0,
        "limited_live_eligible": 84.0,
        "low_risk_live_eligible": 74.0,
        "paper_shadow_only": 52.0,
        "watchlist_only": 38.0,
        "analysis_only": 24.0,
        "blocked_weak_evidence": 10.0,
        "blocked_paused": 0.0,
    }
    return mapping.get(str(permission or "analysis_only"), 24.0)


def trust_rank(trust_tier: Any) -> float:
    mapping = {
        "validated_live": 96.0,
        "limited_live": 84.0,
        "conditional_live": 72.0,
        "paper_only": 44.0,
        "blocked": 20.0,
        "paused": 0.0,
    }
    return mapping.get(str(trust_tier or "blocked"), 20.0)


def macro_state(score: Any) -> str:
    number = safe_float(score)
    if number is None:
        return "mixed"
    if number >= 62.0:
        return "supportive"
    if number <= 42.0:
        return "conflicted"
    return "mixed"


def theme_tag(report: Dict[str, Any]) -> str:
    sentiment = (report.get("data_bundle") or {}).get("sentiment_narrative_flow") or {}
    topic_clusters = sentiment.get("topic_clusters") or sentiment.get("top_narratives") or []
    if topic_clusters:
        first = topic_clusters[0]
        if isinstance(first, dict):
            return str(first.get("topic") or first.get("label") or "broad_theme")
        return str(first)
    positive = (report.get("top_positive_drivers") or [])
    if positive:
        first_driver = positive[0]
        if isinstance(first_driver, dict):
            return str(first_driver.get("label") or "broad_theme")
    sector = ((report.get("data_bundle") or {}).get("symbol_meta") or {}).get("sector") or (
        (report.get("data_bundle") or {}).get("relative_context") or {}
    ).get("sector")
    return str(sector or "broad_theme")


def build_candidate_snapshot(
    report: Dict[str, Any],
    *,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    data_bundle = report.get("data_bundle") or {}
    market = data_bundle.get("market_price_volume") or {}
    relative = data_bundle.get("relative_context") or {}
    macro = data_bundle.get("macro_cross_asset") or {}
    symbol_meta = data_bundle.get("symbol_meta") or {}
    proprietary_scores = report.get("proprietary_scores") or {}
    evaluation = report.get("evaluation") or {}
    execution = strategy.get("execution_posture") or {}

    sector = symbol_meta.get("sector") or relative.get("sector") or "unknown"
    benchmark = (
        macro.get("benchmark_proxy")
        or relative.get("benchmark_proxy")
        or ((relative.get("benchmark_context") or {}).get("benchmark_symbol"))
        or "unknown"
    )
    regime_label = (
        (report.get("regime_intelligence") or {}).get("regime_label")
        or report.get("market_regime")
        or (report.get("key_features") or {}).get("regime_label")
        or "unknown"
    )
    macro_alignment_score = score_value(proprietary_scores.get("Macro Alignment Score"))
    primary_fit = strategy.get("primary_participant_fit") or compact_list(
        strategy.get("participant_fit") or [],
        limit=1,
    )
    primary_fit = (
        primary_fit[0]
        if isinstance(primary_fit, list) and primary_fit
        else primary_fit
        if isinstance(primary_fit, str)
        else "unknown"
    )

    return {
        "report_id": report_id or report.get("report_id"),
        "session_id": session_id or report.get("session_id"),
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "risk_mode": report.get("risk_mode"),
        "sector": sector,
        "benchmark_proxy": benchmark,
        "theme_tag": theme_tag(report),
        "regime_label": regime_label,
        "macro_state": macro_state(macro_alignment_score),
        "final_signal": strategy.get("final_signal")
        or (report.get("signal") or {}).get("final_action")
        or (report.get("signal") or {}).get("action"),
        "strategy_posture": strategy.get("strategy_posture") or report.get("strategy_posture"),
        "conviction_tier": strategy.get("conviction_tier"),
        "actionability_score": safe_float(
            strategy.get("actionability_score") or report.get("actionability_score")
        )
        or 0.0,
        "confidence_score": safe_float(
            strategy.get("confidence_score") or report.get("confidence_score")
        )
        or 0.0,
        "opportunity_quality_score": score_value(
            proprietary_scores.get("Opportunity Quality Score")
        )
        or 0.0,
        "cross_domain_conviction_score": score_value(
            proprietary_scores.get("Cross-Domain Conviction Score")
        )
        or 0.0,
        "signal_fragility_index": score_value(
            proprietary_scores.get("Signal Fragility Index")
        )
        or 100.0,
        "regime_stability_score": score_value(
            proprietary_scores.get("Regime Stability Score")
        )
        or 0.0,
        "fundamental_durability_score": score_value(
            proprietary_scores.get("Fundamental Durability Score")
        )
        or 0.0,
        "macro_alignment_score": macro_alignment_score or 0.0,
        "narrative_crowding_index": score_value(
            proprietary_scores.get("Narrative Crowding Index")
        )
        or 0.0,
        "evaluation_reliability_score": safe_float(
            ((evaluation.get("calibration_summary") or {}).get("confidence_reliability_score"))
        )
        or 50.0,
        "evaluation_hit_rate": safe_float(
            (((evaluation.get("signal_scorecard") or {}).get("final_signal_overall") or {}).get("hit_rate"))
        ),
        "deployment_mode": report.get("deployment_mode") or "research_only",
        "deployment_permission": report.get("deployment_permission") or "analysis_only",
        "trust_tier": report.get("trust_tier") or "blocked",
        "live_readiness_score": safe_float(report.get("live_readiness_score")) or 0.0,
        "risk_budget_tier": report.get("risk_budget_tier") or "none",
        "freshness_status": (report.get("freshness_summary") or {}).get("overall_status") or "unknown",
        "primary_participant_fit": primary_fit,
        "participant_fit": strategy.get("participant_fit") or [],
        "execution_preferred_posture": execution.get("preferred_posture") or "staged_watch",
        "signal_cleanliness": execution.get("signal_cleanliness") or "mixed_clean",
        "urgency_level": execution.get("urgency_level") or "measured",
        "patience_level": execution.get("patience_level") or "high",
        "entry_quality_proxy": safe_float(execution.get("entry_quality_proxy")),
        "ret_21d": safe_float(market.get("ret_21d")),
        "atr_pct": safe_float(market.get("atr_pct")),
        "realized_vol_21d": safe_float(market.get("realized_vol_21d") or (report.get("key_features") or {}).get("vol_21d")),
        "gap_pct": safe_float(market.get("gap_pct")),
        "gap_instability_10d": safe_float(market.get("gap_instability_10d")),
        "volume_anomaly": safe_float(market.get("volume_anomaly")),
        "horizon_days": safe_float((report.get("signal") or {}).get("horizon_days")),
        "current_size_band": report.get("size_band"),
        "current_weight_band": report.get("weight_band"),
        "current_candidate_classification": report.get("candidate_classification"),
    }
