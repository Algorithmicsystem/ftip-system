from __future__ import annotations

from typing import Any, Dict, List, Sequence

from api import config, db
from api.ops import metrics_tracker

from .common import (
    STATUS_CRITICAL,
    STATUS_DEGRADED,
    STATUS_HEALTHY,
    STATUS_WATCH,
    clamp,
    compact_list,
    health_rank,
    mean,
    now_utc,
    safe_float,
    status_from_score,
)

_CRITICAL_DOMAINS = {"market", "technical", "cross_asset", "macro", "event", "quality"}


def _domain_availability(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    availability = report.get("domain_availability")
    if isinstance(availability, dict):
        return availability
    quality = (report.get("data_bundle") or {}).get("quality_provenance") or {}
    return quality.get("domain_availability") or {}


def _freshness_domains(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    freshness = report.get("freshness_summary") or {}
    domains = freshness.get("domains")
    return domains if isinstance(domains, dict) else {}


def _external_fabric_status(report: Dict[str, Any]) -> str:
    external = (report.get("data_bundle") or {}).get("external_data_fabric") or {}
    return str(external.get("status") or "disabled")


def _quality_provenance(report: Dict[str, Any]) -> Dict[str, Any]:
    return ((report.get("data_bundle") or {}).get("quality_provenance") or {})


def _provider_domain_health(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    freshness = _freshness_domains(report)
    availability = _domain_availability(report)
    output: List[Dict[str, Any]] = []
    seen = sorted(set(freshness.keys()) | set(availability.keys()))
    for domain in seen:
        freshness_status = str((freshness.get(domain) or {}).get("status") or "unknown")
        coverage_status = str((availability.get(domain) or {}).get("coverage_status") or "unknown")
        fallback_used = bool((availability.get(domain) or {}).get("fallback_used"))
        if freshness_status in {"stale", "unavailable"} or coverage_status in {
            "unavailable",
            "insufficient_history",
        }:
            status = STATUS_CRITICAL if domain in _CRITICAL_DOMAINS else STATUS_DEGRADED
        elif freshness_status in {"stale_but_usable", "limited", "mixed", "mixed_stale"}:
            status = STATUS_DEGRADED
        elif fallback_used or coverage_status == "partial":
            status = STATUS_WATCH
        else:
            status = STATUS_HEALTHY
        note_parts = [
            f"freshness {freshness_status}",
            f"coverage {coverage_status}",
        ]
        if fallback_used:
            note_parts.append(
                "fallback "
                + ", ".join((availability.get(domain) or {}).get("fallback_source") or [])
            )
        output.append(
            {
                "domain": domain,
                "status": status,
                "message": "; ".join(note_parts),
                "fallback_used": fallback_used,
            }
        )
    return output


def build_health_snapshot(current_report: Dict[str, Any]) -> Dict[str, Any]:
    freshness = _freshness_domains(current_report)
    availability = _domain_availability(current_report)
    quality = _quality_provenance(current_report)
    provider_notes = compact_list(quality.get("provider_notes") or [], limit=10)
    metrics = metrics_tracker.snapshot()
    total_requests = sum((metrics.get("request_counts") or {}).values())
    error_count = int(metrics.get("status_4xx") or 0) + int(metrics.get("status_5xx") or 0)
    error_rate = (error_count / total_requests) if total_requests else 0.0

    stale_domains = [
        name
        for name, payload in freshness.items()
        if str((payload or {}).get("status") or "") in {"stale", "stale_but_usable", "limited", "mixed_stale"}
    ]
    degraded_domains = [
        name
        for name, payload in freshness.items()
        if str((payload or {}).get("status") or "") in {"mixed", "partial"}
    ]
    fallback_domains = [
        name
        for name, payload in availability.items()
        if bool((payload or {}).get("fallback_used"))
    ]
    critical_missing_domains = [
        name
        for name, payload in availability.items()
        if name in _CRITICAL_DOMAINS
        and str((payload or {}).get("coverage_status") or "") in {"unavailable", "insufficient_history"}
    ]
    quality_score = safe_float(quality.get("quality_score")) or 60.0
    fallback_ratio = (
        float(len(fallback_domains)) / float(len(availability))
        if availability
        else 0.0
    )
    data_reliability_score = clamp(
        quality_score
        - (len(stale_domains) * 9.0)
        - (len(degraded_domains) * 4.0)
        - (len(fallback_domains) * 5.0)
        - (len(critical_missing_domains) * 12.0)
        - (len(provider_notes) * 2.0),
        0.0,
        100.0,
    )

    provider_domain_health = _provider_domain_health(current_report)
    provider_health_status = STATUS_HEALTHY
    for item in provider_domain_health:
        if health_rank(item.get("status")) > health_rank(provider_health_status):
            provider_health_status = str(item.get("status"))
    if fallback_ratio >= 0.45 and health_rank(provider_health_status) < health_rank(STATUS_DEGRADED):
        provider_health_status = STATUS_DEGRADED

    data_pipeline_score = clamp(
        90.0
        - (len(stale_domains) * 12.0)
        - (len(critical_missing_domains) * 14.0)
        - (fallback_ratio * 40.0)
        - (20.0 if _external_fabric_status(current_report) == "error" else 0.0),
        0.0,
        100.0,
    )
    artifact_ids = [
        current_report.get("prediction_record_id"),
        current_report.get("evaluation_artifact_id"),
        current_report.get("deployment_readiness_artifact_id"),
        current_report.get("portfolio_construction_artifact_id"),
        current_report.get("portfolio_risk_model_artifact_id"),
        current_report.get("learning_artifact_id"),
        current_report.get("canonical_validation_artifact_id"),
    ]
    artifact_pipeline_score = clamp(
        100.0 - (artifact_ids.count(None) + artifact_ids.count("")) * 16.0,
        0.0,
        100.0,
    )
    data_pipeline_health = status_from_score(data_pipeline_score)
    artifact_pipeline_health = status_from_score(artifact_pipeline_score)

    system_health_score = clamp(
        mean(
            [
                data_reliability_score,
                data_pipeline_score,
                artifact_pipeline_score,
                max(0.0, 100.0 - (error_rate * 100.0)),
                100.0 if config.llm_enabled() else 42.0,
                100.0 if db.db_enabled() else 0.0,
                100.0 if db.db_read_enabled() else 20.0,
            ]
        )
        or 0.0,
        0.0,
        100.0,
    )
    system_health_status = status_from_score(system_health_score)
    if critical_missing_domains or not db.db_enabled() or not db.db_read_enabled():
        system_health_status = STATUS_CRITICAL
    elif provider_health_status == STATUS_DEGRADED and health_rank(system_health_status) < health_rank(STATUS_DEGRADED):
        system_health_status = STATUS_DEGRADED

    return {
        "captured_at": now_utc(),
        "system_health_status": system_health_status,
        "system_health_score": round(system_health_score, 2),
        "provider_health_status": provider_health_status,
        "provider_degradation_notes": provider_notes,
        "provider_health_details": provider_domain_health,
        "data_pipeline_health": data_pipeline_health,
        "artifact_pipeline_health": artifact_pipeline_health,
        "failure_rate_summary": {
            "total_requests": total_requests,
            "status_4xx": int(metrics.get("status_4xx") or 0),
            "status_5xx": int(metrics.get("status_5xx") or 0),
            "error_rate": round(error_rate, 4),
            "rate_limit_hits": int(metrics.get("rate_limit_hits") or 0),
            "snapshot_runs": int(metrics.get("snapshot_runs") or 0),
            "strategy_graph_runs": int(metrics.get("strategy_graph_runs") or 0),
        },
        "stale_domain_summary": stale_domains,
        "degraded_domain_list": compact_list(
            [*critical_missing_domains, *stale_domains, *degraded_domains],
            limit=10,
        ),
        "fallback_overuse_summary": {
            "fallback_domain_count": len(fallback_domains),
            "fallback_domain_ratio": round(fallback_ratio, 3),
            "fallback_domains": compact_list(fallback_domains, limit=10),
            "status": (
                STATUS_CRITICAL
                if fallback_ratio >= 0.75
                else STATUS_DEGRADED
                if fallback_ratio >= 0.4
                else STATUS_WATCH
                if fallback_ratio > 0.0
                else STATUS_HEALTHY
            ),
        },
        "critical_domain_missing_flag": bool(critical_missing_domains),
        "critical_domains_missing": critical_missing_domains,
        "coverage_collapse_alert": bool(
            len(critical_missing_domains) >= 2 or len(stale_domains) >= 3
        ),
        "freshness_alert": bool(stale_domains),
        "fallback_overuse_alert": bool(fallback_ratio >= 0.4),
        "data_reliability_score": round(data_reliability_score, 2),
        "assistant_runtime": {
            "llm_enabled": config.llm_enabled(),
            "db_enabled": db.db_enabled(),
            "db_read_enabled": db.db_read_enabled(),
            "db_write_enabled": db.db_write_enabled(),
        },
        "external_data_fabric_status": _external_fabric_status(current_report),
    }
