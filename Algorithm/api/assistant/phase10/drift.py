from __future__ import annotations

from typing import Any, Dict, List

from .common import compact_list, safe_float, slugify


def build_drift_alerts(
    current_report: Dict[str, Any],
    current_snapshot: Dict[str, Any],
    regime_learning: Dict[str, Any],
) -> List[Dict[str, Any]]:
    evaluation = current_report.get("evaluation") or {}
    calibration = evaluation.get("calibration_summary") or {}
    deployment = current_report.get("deployment_readiness") or {}
    drift_monitor = deployment.get("drift_monitor") or {}
    alerts: List[Dict[str, Any]] = []

    reliability = safe_float(calibration.get("confidence_reliability_score"))
    monotonicity = str(calibration.get("confidence_monotonicity") or "unknown")
    if reliability is not None and (reliability < 60.0 or monotonicity == "lower_confidence_buckets_outperform"):
        alerts.append(
            {
                "drift_id": f"drift_{slugify('confidence_calibration')}",
                "affected_component": "confidence_calibration",
                "severity": "high" if reliability < 50.0 else "moderate",
                "evidence": compact_list(
                    [
                        f"confidence reliability score is {reliability}",
                        f"monotonicity is {monotonicity}",
                        *(calibration.get("calibration_drift_notes") or []),
                    ],
                    limit=5,
                ),
                "adaptation_candidate": "tighten confidence-to-conviction mapping and make deployment gates less permissive until calibration recovers",
                "rollback_recommended": reliability < 45.0,
                "research_priority_increase": True,
            }
        )

    current_regime = str(current_snapshot.get("regime_label") or "unknown")
    if regime_learning.get("current_regime_is_weak"):
        note = regime_learning.get("current_regime_note") or {}
        alerts.append(
            {
                "drift_id": f"drift_{slugify(f'regime_{current_regime}')}",
                "affected_component": "regime_conditioning",
                "severity": "high"
                if (safe_float(note.get("average_reliability")) or 0.0) < 52.0
                else "moderate",
                "evidence": compact_list(
                    [
                        f"active regime {current_regime} appears in the weakest evaluation conditions",
                        f"average reliability in this regime is {note.get('average_reliability')}",
                        f"average hit rate in this regime is {note.get('average_hit_rate')}",
                    ],
                    limit=4,
                ),
                "adaptation_candidate": "increase confirmation requirements and fragility penalties when the active regime remains weak",
                "rollback_recommended": False,
                "research_priority_increase": True,
            }
        )

    current_archetype = (current_report.get("setup_archetype") or {}).get("archetype_name")
    if current_archetype:
        archetype_stats = (
            (current_report.get("signal_family_library") or {}).get("archetype_cohorts") or []
        )
        matching = next(
            (
                item
                for item in archetype_stats
                if item.get("archetype_name") == current_archetype
                or item.get("archetype_id")
                == (current_report.get("setup_archetype") or {}).get("archetype_id")
            ),
            None,
        )
        if matching and int(matching.get("sample_count") or 0) < 4:
            alerts.append(
                {
                    "drift_id": f"drift_{slugify(current_archetype)}_sparse",
                    "affected_component": "archetype_reliability",
                    "severity": "moderate",
                    "evidence": compact_list(
                        [
                            f"active archetype {current_archetype} only has {matching.get('sample_count')} tracked cohort examples",
                            "archetype reliability should remain exploratory until sample depth improves",
                        ],
                        limit=3,
                    ),
                    "adaptation_candidate": "keep archetype-specific weight changes in proposal mode until cohort depth improves",
                    "rollback_recommended": False,
                    "research_priority_increase": False,
                }
            )

    for item in drift_monitor.get("drift_alerts") or []:
        alerts.append(
            {
                "drift_id": f"drift_{slugify(item)}",
                "affected_component": "deployment_monitoring",
                "severity": "moderate",
                "evidence": [str(item)],
                "adaptation_candidate": "align learning proposals with deployment-readiness caution until the drift alert clears",
                "rollback_recommended": False,
                "research_priority_increase": True,
            }
        )

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for alert in alerts:
        drift_id = alert.get("drift_id")
        if drift_id in seen:
            continue
        seen.add(drift_id)
        deduped.append(alert)
    return deduped[:8]
