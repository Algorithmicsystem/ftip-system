from __future__ import annotations

from typing import Any, Dict, Sequence

from .common import compact_list


def build_monthly_refinement(
    current_report: Dict[str, Any],
    *,
    monthly_validation: Dict[str, Any],
    recent_incidents: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    learning = current_report.get("continuous_learning") or {}
    setup = learning.get("active_setup_archetype") or current_report.get("setup_archetype") or {}
    research_queue = list(current_report.get("improvement_queue") or [])
    cleanup_queue = list(current_report.get("commercial_cleanup_queue") or [])
    drift_alerts = list(current_report.get("learning_drift_alerts") or [])
    incidents = [
        item.get("summary") or item.get("alert_summary")
        for item in recent_incidents
        if isinstance(item, dict)
    ]
    priorities = []
    for item in research_queue[:5]:
        priorities.append(
            {
                "title": item.get("title") or item.get("target_family") or "research item",
                "priority": item.get("priority") or item.get("severity") or "review",
                "reason": item.get("linked_experiment_id")
                or item.get("observed_pattern")
                or item.get("description")
                or "Follow-up validation is still needed.",
            }
        )
    for item in cleanup_queue[:3]:
        priorities.append(
            {
                "title": f"Commercial cleanup: {item.get('source_name') or 'source'}",
                "priority": item.get("priority") or "high",
                "reason": item.get("cleanup_reason") or "Commercial stack cleanup remains pending.",
            }
        )
    for item in drift_alerts[:3]:
        priorities.append(
            {
                "title": item.get("affected_component") or "drift review",
                "priority": item.get("severity") or "review",
                "reason": item.get("evidence") or "Drift review remains open.",
            }
        )

    strongest = compact_list(monthly_validation.get("strongest_conditions") or [], limit=4)
    weakest = compact_list(monthly_validation.get("weakest_conditions") or [], limit=4)
    monthly_summary = (
        f"Monthly refinement is centered on archetype {setup.get('archetype_name') or 'n/a'}, "
        f"research priority {current_report.get('learning_priority') or 'observe'}, "
        f"and {len(priorities)} queued improvement or cleanup items."
    )
    return {
        "monthly_refinement_review": {
            "review_window_days": 30,
            "archetype_focus": setup.get("archetype_name"),
            "strongest_setup_families": strongest,
            "weakest_setup_families": weakest,
            "persistent_failure_modes": compact_list(
                monthly_validation.get("failure_modes") or [],
                limit=5,
            ),
            "source_governance_status": current_report.get("buyer_demo_suitability"),
            "recent_incident_count": len(incidents),
        },
        "research_priority_queue": priorities[:8],
        "improvement_candidate_summary": current_report.get("adaptation_queue_summary")
        or current_report.get("learning_summary")
        or "No monthly improvement candidate summary is available yet.",
        "trust_promotion_candidates": [],
        "trust_demotion_candidates": [],
        "monthly_operating_summary": monthly_summary,
    }
