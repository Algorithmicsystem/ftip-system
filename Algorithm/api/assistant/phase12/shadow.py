from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .common import clamp, compact_list, now_utc, safe_float


def build_shadow_decision_record(
    current_report: Dict[str, Any],
    *,
    report_id: Optional[str],
    session_id: Optional[str],
) -> Dict[str, Any]:
    strategy = current_report.get("strategy") or {}
    return {
        "recorded_at": now_utc(),
        "report_id": report_id,
        "session_id": session_id,
        "symbol": current_report.get("symbol"),
        "as_of_date": current_report.get("as_of_date"),
        "horizon": current_report.get("horizon"),
        "risk_mode": current_report.get("risk_mode"),
        "signal": strategy.get("final_signal")
        or (current_report.get("signal") or {}).get("final_action")
        or (current_report.get("signal") or {}).get("action"),
        "strategy_posture": current_report.get("strategy_posture"),
        "conviction_tier": current_report.get("conviction_tier"),
        "confidence_score": current_report.get("confidence_score"),
        "deployment_mode": current_report.get("deployment_mode"),
        "deployment_permission": current_report.get("deployment_permission"),
        "trust_tier": current_report.get("trust_tier"),
        "candidate_classification": current_report.get("candidate_classification"),
        "portfolio_fit_quality": current_report.get("portfolio_fit_quality"),
        "ranked_opportunity_score": current_report.get("ranked_opportunity_score"),
        "live_readiness_score": current_report.get("live_readiness_score"),
        "suppression_flags": list(current_report.get("suppression_flags") or []),
        "blockers": compact_list(
            [
                *(current_report.get("deployment_blockers") or []),
                *(current_report.get("candidate_blockers") or []),
            ],
            limit=8,
        ),
    }


def build_shadow_mode_summary(
    current_report: Dict[str, Any],
    *,
    shadow_decision_record: Dict[str, Any],
    prior_shadow_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    validation = current_report.get("canonical_validation") or {}
    linkage = validation.get("prediction_linkage_summary") or {}
    net_returns = validation.get("net_return_summary") or {}
    walkforward = validation.get("walkforward_summary") or {}
    current_mode = str(current_report.get("deployment_mode") or "research_only")
    permission = str(current_report.get("deployment_permission") or "analysis_only")
    live_readiness = safe_float(current_report.get("live_readiness_score")) or 0.0
    net_edge = safe_float(net_returns.get("average_edge_return"))
    hit_rate = safe_float(net_returns.get("hit_rate"))

    if current_mode == "paused" or permission == "blocked_paused":
        shadow_mode_status = "paused"
    elif current_mode in {"research_only", "paper_shadow"} or permission in {
        "analysis_only",
        "paper_shadow_only",
    }:
        shadow_mode_status = "active_shadow"
    elif permission.endswith("eligible"):
        shadow_mode_status = "live_candidate_monitor"
    else:
        shadow_mode_status = "watch_only"

    shadow_promotion_candidate = bool(
        shadow_mode_status == "active_shadow"
        and (linkage.get("matured_count") or 0) >= 8
        and (walkforward.get("window_count") or 0) >= 1
        and (net_edge or 0.0) > 0.0
        and (hit_rate or 0.0) >= 0.55
        and live_readiness >= 65.0
    )
    shadow_demotion_reason = None
    if current_report.get("pause_recommended"):
        shadow_demotion_reason = "Phase 8 drift controls are recommending pause conditions."
    elif current_report.get("degrade_to_paper_recommended"):
        shadow_demotion_reason = "Deployment controls are recommending a downgrade to paper or shadow mode."
    elif current_report.get("deployment_permission") == "paper_shadow_only":
        shadow_demotion_reason = current_report.get("deployment_rationale")

    live_like_records = [
        item
        for item in prior_shadow_records
        if str(item.get("deployment_permission") or "").endswith("eligible")
    ]
    blocked_records = [
        item
        for item in prior_shadow_records
        if str(item.get("deployment_permission") or "").startswith("blocked")
    ]
    paper_only_records = [
        item
        for item in prior_shadow_records
        if str(item.get("deployment_permission")) == "paper_shadow_only"
    ]

    reliability_score = clamp(
        (
            (50.0 if net_edge is None else (50.0 + (net_edge * 800.0)))
            + (0.0 if hit_rate is None else (hit_rate * 35.0))
            + (min((linkage.get("matured_count") or 0), 20) * 1.2)
        )
        / 2.0,
        0.0,
        100.0,
    )

    shadow_vs_realized_summary = (
        f"Shadow tracking currently has {linkage.get('matured_count') or 0} matured decisions, "
        f"average net edge {net_edge if net_edge is not None else 'n/a'}, hit rate {hit_rate if hit_rate is not None else 'n/a'}, "
        f"and {walkforward.get('window_count') or 0} walk-forward windows."
    )
    return {
        "shadow_mode_status": shadow_mode_status,
        "shadow_run_status": "recorded",
        "shadow_cohort": {
            "tracked_shadow_decisions": len(prior_shadow_records),
            "paper_only_count": len(paper_only_records),
            "live_like_count": len(live_like_records),
            "blocked_count": len(blocked_records),
        },
        "shadow_vs_realized_summary": shadow_vs_realized_summary,
        "shadow_reliability_summary": (
            f"Shadow reliability reads {round(reliability_score, 2)} / 100 with {linkage.get('matured_count') or 0} matured decisions and "
            f"{walkforward.get('window_count') or 0} walk-forward windows."
        ),
        "shadow_promotion_candidate": shadow_promotion_candidate,
        "shadow_demotion_reason": shadow_demotion_reason,
        "current_shadow_decision": shadow_decision_record,
        "shadow_reliability_score": round(reliability_score, 2),
    }
