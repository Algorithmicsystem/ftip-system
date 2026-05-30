from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import compact_list, safe_float


def _candidate_score(row: Dict[str, Any]) -> float:
    return float(
        safe_float(row.get("marginal_portfolio_utility"))
        or safe_float(row.get("portfolio_candidate_score"))
        or safe_float(row.get("ranked_opportunity_score"))
        or 0.0
    )


def _report_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    blockers = compact_list(
        [
            *(report.get("deployment_blockers") or []),
            *(report.get("candidate_blockers") or []),
            *(report.get("operational_alerts") or []),
        ],
        limit=4,
    )
    return {
        "symbol": report.get("symbol"),
        "signal": (report.get("strategy") or {}).get("final_signal")
        or (report.get("signal") or {}).get("final_action")
        or (report.get("signal") or {}).get("action"),
        "strategy_posture": report.get("strategy_posture")
        or (report.get("strategy") or {}).get("strategy_posture"),
        "candidate_classification": report.get("candidate_classification"),
        "deployment_permission": report.get("deployment_permission"),
        "trust_tier": report.get("trust_tier"),
        "current_operating_mode": report.get("current_operating_mode"),
        "portfolio_fit_quality": report.get("portfolio_fit_quality"),
        "ranked_opportunity_score": report.get("ranked_opportunity_score"),
        "marginal_portfolio_utility": report.get("marginal_portfolio_utility"),
        "size_band": report.get("size_band"),
        "review_reason": blockers[0] if blockers else report.get("overall_analysis"),
    }


def _unique_recent_reports(
    current_report: Dict[str, Any],
    recent_reports: Sequence[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    ordered: List[Dict[str, Any]] = [current_report]
    ordered.extend(recent_reports)
    output: List[Dict[str, Any]] = []
    for report in ordered:
        symbol = str(report.get("symbol") or "").upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        output.append(report)
        if len(output) >= limit:
            break
    return output


def _find_prior_same_symbol(
    current_report: Dict[str, Any],
    recent_reports: Sequence[Dict[str, Any]],
    *,
    current_report_id: str = "",
) -> Dict[str, Any]:
    current_symbol = str(current_report.get("symbol") or "")
    for report in recent_reports:
        if str(report.get("symbol") or "") != current_symbol:
            continue
        if current_report_id and str(report.get("report_id") or "") == current_report_id:
            continue
        return report
    return {}


def _changed_signals(current_report: Dict[str, Any], prior_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not prior_report:
        return [{"layer": "baseline", "description": "No prior same-symbol report is available, so this review is establishing the baseline shadow record.", "direction": "new", "prior": "n/a", "current": "n/a"}]
    changes: List[Dict[str, Any]] = []
    current_signal = (current_report.get("strategy") or {}).get("final_signal") or (
        current_report.get("signal") or {}
    ).get("action")
    prior_signal = (prior_report.get("strategy") or {}).get("final_signal") or (
        prior_report.get("signal") or {}
    ).get("action")
    if current_signal != prior_signal:
        changes.append({"layer": "signal", "description": f"Signal changed from {prior_signal or 'n/a'} to {current_signal or 'n/a'}.", "direction": "changed", "prior": prior_signal or "n/a", "current": current_signal or "n/a"})

    current_posture = current_report.get("strategy_posture") or (
        current_report.get("strategy") or {}
    ).get("strategy_posture")
    prior_posture = prior_report.get("strategy_posture") or (
        prior_report.get("strategy") or {}
    ).get("strategy_posture")
    if current_posture != prior_posture:
        changes.append({"layer": "posture", "description": f"Strategy posture changed from {prior_posture or 'n/a'} to {current_posture or 'n/a'}.", "direction": "changed", "prior": prior_posture or "n/a", "current": current_posture or "n/a"})

    if current_report.get("deployment_permission") != prior_report.get("deployment_permission"):
        changes.append({"layer": "deployment", "description": f"Deployment permission moved from {prior_report.get('deployment_permission') or 'n/a'} to {current_report.get('deployment_permission') or 'n/a'}.", "direction": "changed", "prior": prior_report.get("deployment_permission") or "n/a", "current": current_report.get("deployment_permission") or "n/a"})
    if current_report.get("trust_tier") != prior_report.get("trust_tier"):
        changes.append({"layer": "trust", "description": f"Trust tier shifted from {prior_report.get('trust_tier') or 'n/a'} to {current_report.get('trust_tier') or 'n/a'}.", "direction": "changed", "prior": prior_report.get("trust_tier") or "n/a", "current": current_report.get("trust_tier") or "n/a"})
    if current_report.get("current_operating_mode") != prior_report.get("current_operating_mode"):
        changes.append({"layer": "operating_mode", "description": f"Operating mode moved from {prior_report.get('current_operating_mode') or 'n/a'} to {current_report.get('current_operating_mode') or 'n/a'}.", "direction": "changed", "prior": prior_report.get("current_operating_mode") or "n/a", "current": current_report.get("current_operating_mode") or "n/a"})
    if current_report.get("candidate_classification") != prior_report.get("candidate_classification"):
        changes.append({"layer": "classification", "description": f"Candidate classification moved from {prior_report.get('candidate_classification') or 'n/a'} to {current_report.get('candidate_classification') or 'n/a'}.", "direction": "changed", "prior": prior_report.get("candidate_classification") or "n/a", "current": current_report.get("candidate_classification") or "n/a"})
    if not changes:
        changes.append({"layer": "none", "description": "No material change was detected versus the prior same-symbol report.", "direction": "unchanged", "prior": "n/a", "current": "n/a"})
    return changes


def build_daily_workflow(
    current_report: Dict[str, Any],
    *,
    current_report_id: str = "",
    recent_reports: Sequence[Dict[str, Any]],
    recent_shadow_records: Sequence[Dict[str, Any]],
    recent_incidents: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    triage_rows = [_report_snapshot(report) for report in _unique_recent_reports(current_report, recent_reports)]
    triage_rows.sort(key=_candidate_score, reverse=True)
    priority_watchlist = compact_list(
        (current_report.get("portfolio_construction") or {})
        .get("workflow", {})
        .get("prioritized_watchlist")
        or [row.get("symbol") for row in triage_rows if row.get("candidate_classification") != "blocked_candidate"],
        limit=6,
    )
    prior_report = _find_prior_same_symbol(
        current_report,
        recent_reports,
        current_report_id=current_report_id,
    )
    changed_signals = _changed_signals(current_report, prior_report)

    promotions = [
        row.get("symbol")
        for row in triage_rows
        if str(row.get("deployment_permission") or "").endswith("eligible")
    ]
    shadow_only = [
        row.get("symbol")
        for row in triage_rows
        if str(row.get("deployment_permission") or "") == "paper_shadow_only"
    ]
    blocked = [
        row.get("symbol")
        for row in triage_rows
        if str(row.get("deployment_permission") or "") in {"analysis_only", "blocked_paused"}
    ]
    warnings = compact_list(
        [
            *(current_report.get("deployment_blockers") or []),
            *(current_report.get("candidate_blockers") or []),
            *(current_report.get("deployment_risk_alerts") or []),
            *(current_report.get("degraded_domain_list") or []),
            *[
                item.get("summary") or item.get("alert_summary")
                for item in recent_incidents[:4]
                if isinstance(item, dict)
            ],
        ],
        limit=8,
    )
    operator_attention = compact_list(
        [
            warnings[0] if warnings else None,
            current_report.get("shadow_demotion_reason"),
            current_report.get("downgrade_reason"),
            current_report.get("replacement_candidate")
            and f"{current_report.get('replacement_candidate')} is currently a cleaner portfolio substitute.",
            current_report.get("pause_required") and "Pause conditions are active and require operator review first.",
        ],
        limit=6,
    )
    what_changed = changed_signals  # structured List[Dict] — callers use "description" for text
    changed_signals_text = [item["description"] for item in changed_signals]
    daily_summary = (
        f"Today’s operator triage has {len(triage_rows)} tracked candidates, "
        f"{len(promotions)} live-like names, {len(shadow_only)} shadow-only names, and {len(blocked)} blocked names. "
        f"Primary review focus: {(operator_attention or ['no acute warning'])[0]}"
    )

    return {
        "todays_candidate_triage": triage_rows[:6],
        "changed_signals": changed_signals_text,
        "priority_watchlist": priority_watchlist,
        "new_warnings_downgrades": warnings,
        "active_trust_state": {
            "deployment_mode": current_report.get("deployment_mode"),
            "deployment_permission": current_report.get("deployment_permission"),
            "trust_tier": current_report.get("trust_tier"),
            "current_operating_mode": current_report.get("current_operating_mode"),
            "shadow_mode_status": current_report.get("shadow_mode_status"),
            "system_health_status": current_report.get("system_health_status"),
        },
        "deployment_eligibility_snapshot": {
            "tracked_candidates": len(triage_rows),
            "live_like_candidates": len(promotions),
            "shadow_only_candidates": len(shadow_only),
            "blocked_candidates": len(blocked),
            "recent_shadow_records": len(recent_shadow_records),
        },
        "what_changed_panel": what_changed,
        "daily_operator_attention_items": operator_attention,
        "daily_operating_summary": daily_summary,
    }
