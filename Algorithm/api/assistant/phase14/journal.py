from __future__ import annotations

from typing import Any, Dict, Sequence

from .common import compact_list


def build_shadow_decision_journal(
    current_report: Dict[str, Any],
    *,
    current_shadow_record: Dict[str, Any],
    recent_shadow_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    recent_entries = []
    for row in recent_shadow_records[:6]:
        recent_entries.append(
            {
                "symbol": row.get("symbol"),
                "signal": row.get("signal"),
                "deployment_permission": row.get("deployment_permission"),
                "trust_tier": row.get("trust_tier"),
                "candidate_classification": row.get("candidate_classification"),
                "recorded_at": row.get("recorded_at"),
                "blockers": compact_list(row.get("blockers") or [], limit=3),
            }
        )
    review_note = (
        "The trust gate helped keep the setup in shadow mode while confidence, calibration, or fragility constraints remained unresolved."
        if str(current_report.get("deployment_permission") or "") == "paper_shadow_only"
        else "The setup is clearing higher-trust gates, but shadow logging should still be preserved for later review."
    )
    return {
        "shadow_decision_journal": {
            "current_shadow_decision": current_shadow_record,
            "recent_shadow_decisions": recent_entries,
            "shadow_decision_count": len(recent_shadow_records),
        },
        "operator_review_entry": {
            "review_focus": compact_list(
                [
                    current_report.get("shadow_mode_summary"),
                    current_report.get("deployment_rationale"),
                    current_report.get("portfolio_workflow_summary"),
                ],
                limit=3,
            ),
            "review_required": bool(current_report.get("human_review_required")),
        },
        "realized_followup": {
            "shadow_vs_realized_summary": current_report.get("shadow_vs_realized_summary"),
            "shadow_reliability_summary": current_report.get("shadow_reliability_summary"),
        },
        "decision_quality_note": review_note,
        "trust_gate_feedback": review_note,
        "candidate_outcome_review": current_report.get("canonical_validation_summary")
        or "Outcome review is still building as more shadow decisions mature.",
        "shadow_journal_summary": (
            f"Shadow journal now tracks {len(recent_shadow_records)} decisions. "
            f"Current review focus: {review_note}"
        ),
    }
