from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .common import compact_list, overfit_risk, slugify


def build_experiment_registry(
    current_report: Dict[str, Any],
    reweighting_candidates: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    drift_alerts: List[Dict[str, Any]],
    prior_learning_artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    recurring_titles = Counter()
    for artifact in prior_learning_artifacts:
        registry = (artifact.get("experiment_registry") or {}).get("open_experiments") or []
        for item in registry:
            recurring_titles[str(item.get("title") or item.get("experiment_id") or "")] += 1

    linked_versions = {
        "report_version": current_report.get("report_version"),
        "strategy_version": current_report.get("strategy_version")
        or (current_report.get("strategy") or {}).get("strategy_version"),
        "evaluation_version": (current_report.get("evaluation") or {}).get("evaluation_version"),
        "deployment_mode": current_report.get("deployment_mode"),
    }

    experiments: List[Dict[str, Any]] = []
    for candidate in reweighting_candidates[:5]:
        title = f"Reweight {str(candidate.get('target_family') or 'factor').replace('_', ' ')}"
        recurring = recurring_titles[title]
        experiments.append(
            {
                "experiment_id": f"exp_{slugify(title)}",
                "title": title,
                "proposed_change": candidate.get("suggested_weight_changes") or [],
                "reason": candidate.get("rationale"),
                "supporting_data": compact_list(
                    [
                        candidate.get("expected_impact_area"),
                        f"sample {candidate.get('sample_size')}",
                        f"confidence {candidate.get('confidence_in_recommendation')}",
                    ],
                    limit=5,
                ),
                "validation_status": "under_review" if recurring else "proposed",
                "approval_status": "approval_required",
                "rollout_status": "not_started",
                "rollback_status": "not_applicable",
                "linked_versions": linked_versions,
                "sample_size": candidate.get("sample_size") or 0,
                "risk_of_overfit": candidate.get("risk_of_overfit") or "moderate",
            }
        )

    for hypothesis in hypotheses[:3]:
        title = hypothesis.get("hypothesis_title") or "research hypothesis"
        if any(item.get("title") == title for item in experiments):
            continue
        recurring = recurring_titles[str(title)]
        experiments.append(
            {
                "experiment_id": f"exp_{slugify(title)}",
                "title": title,
                "proposed_change": hypothesis.get("candidate_improvement"),
                "reason": hypothesis.get("observed_pattern"),
                "supporting_data": compact_list(hypothesis.get("supporting_evidence") or [], limit=5),
                "validation_status": "under_review" if recurring else "proposed",
                "approval_status": "approval_required",
                "rollout_status": "not_started",
                "rollback_status": "not_applicable",
                "linked_versions": linked_versions,
                "sample_size": hypothesis.get("sample_size") or 0,
                "risk_of_overfit": hypothesis.get("risk_of_overfit") or "moderate",
            }
        )

    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for artifact in prior_learning_artifacts:
        registry = artifact.get("experiment_registry") or {}
        approved.extend(registry.get("approved_improvements") or [])
        rejected.extend(registry.get("rejected_improvements") or [])

    experiments.sort(
        key=lambda item: (
            1 if item.get("validation_status") == "under_review" else 0,
            item.get("sample_size") or 0,
        ),
        reverse=True,
    )
    summary = (
        f"The experiment registry currently tracks {len(experiments)} open proposals, {len(approved)} approved improvements, and {len(rejected)} rejected proposals. "
        f"{len([item for item in experiments if item.get('validation_status') == 'under_review'])} proposals are recurring enough to warrant deeper review."
    )
    improvement_queue = []
    for item in experiments[:6]:
        improvement_queue.append(
            {
                "title": item.get("title"),
                "priority": "high"
                if item.get("validation_status") == "under_review" or (item.get("sample_size") or 0) >= 8
                else "moderate",
                "reason": item.get("reason"),
                "approval_required": True,
                "linked_experiment_id": item.get("experiment_id"),
            }
        )
    for alert in drift_alerts[:2]:
        improvement_queue.append(
            {
                "title": f"Investigate drift in {str(alert.get('affected_component') or 'component').replace('_', ' ')}",
                "priority": "high" if alert.get("severity") == "high" else "moderate",
                "reason": " / ".join(alert.get("evidence") or []),
                "approval_required": True,
                "linked_experiment_id": f"drift_{slugify(alert.get('affected_component') or 'component')}",
            }
        )
    return {
        "open_experiments": experiments[:8],
        "approved_improvements": approved[:8],
        "rejected_improvements": rejected[:8],
        "experiment_registry_summary": summary,
        "improvement_queue": improvement_queue[:8],
    }
