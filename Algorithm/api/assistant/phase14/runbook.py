from __future__ import annotations

from typing import Any, Dict

from .common import compact_list


def build_operator_runbook(current_report: Dict[str, Any]) -> Dict[str, Any]:
    pause_note = (
        "If pause conditions are active, stop relying on higher-trust deployment support until recovery criteria are met."
        if current_report.get("pause_required")
        else "No active pause condition is blocking the standard review sequence."
    )
    runbook = {
        "daily_workflow": [
            "Review today’s candidate triage and changed-signal panel first.",
            "Inspect new warnings, trust downgrades, and any blocked or stale candidates.",
            "Confirm whether the active operating mode and deployment permission still make sense.",
            "Record shadow interpretations before using the output in any higher-trust workflow.",
        ],
        "weekly_workflow": [
            "Review weekly signal quality, net edge, readiness spread, and suppression usefulness.",
            "Check drift, provider degradation, and concentration or overlap failures.",
            "Update the operator attention list and decide whether trust should be maintained, tightened, or reviewed.",
        ],
        "monthly_workflow": [
            "Review strongest and weakest setup families, persistent failure modes, and open research candidates.",
            "Re-rank improvement work across alpha, risk controls, data dependencies, and commercialization cleanup.",
            "Decide whether any trust promotion or demotion proposal deserves governed follow-up.",
        ],
        "incident_response": [
            pause_note,
            "Check provider freshness, fallback overuse, and current operational alerts.",
            "Move the system to shadow-first interpretation if downgrade conditions are active.",
            "Document the incident summary and revisit the trust checklist before resuming stronger reliance.",
        ],
    }
    return {
        "operator_runbook": runbook,
        "operator_runbook_summary": "Daily review starts with candidate triage and changed signals, weekly review centers on validation and drift, and monthly review turns failures and queue items into governed refinement work.",
        "runbook_attention_notes": compact_list(
            [
                pause_note,
                current_report.get("source_governance_summary"),
                current_report.get("drift_control_summary"),
            ],
            limit=4,
        ),
    }
