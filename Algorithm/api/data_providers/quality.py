from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional

from api.source_governance import (
    active_source_profile,
    governance_risk_score,
    source_definition,
)


def _bounded_confidence(value: Any, *, default: float = 0.55) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def provider_freshness_status(
    *, end_date: Optional[dt.date] = None, generated_at: Optional[dt.datetime] = None
) -> str:
    if end_date is None:
        return "unknown"
    now = (generated_at or dt.datetime.now(dt.timezone.utc)).date()
    age_days = max((now - end_date).days, 0)
    if age_days <= 2:
        return "fresh"
    if age_days <= 7:
        return "recent"
    return "historical"


def provider_confidence(
    provider_name: Optional[str],
    *,
    fallback_used: bool = False,
    response_quality: str = "complete",
    override: Optional[float] = None,
) -> float:
    if override is not None:
        return _bounded_confidence(override)
    definition = source_definition(provider_name)
    governance_penalty = governance_risk_score(definition.get("governance_status")) / 200.0
    role = str(definition.get("fallback_role") or "")
    role_bonus = 0.1 if role.startswith("primary") else 0.04 if "secondary" in role else -0.03
    fallback_penalty = 0.08 if fallback_used else 0.0
    quality_penalty = 0.12 if response_quality == "partial" else 0.18 if response_quality == "degraded" else 0.0
    return _bounded_confidence(0.78 + role_bonus - governance_penalty - fallback_penalty - quality_penalty)


def provider_attempt(
    provider_name: str,
    *,
    status: str,
    source_type: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    response_quality: Optional[str] = None,
    fallback_used: bool = False,
) -> Dict[str, Any]:
    definition = source_definition(provider_name)
    resolved_source_type = source_type or definition.get("provider_type") or "unknown"
    return {
        "provider_name": definition.get("source_name") or provider_name,
        "display_name": definition.get("display_name") or provider_name,
        "source_type": resolved_source_type,
        "status": status,
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "fallback_role": definition.get("fallback_role"),
        "response_quality": response_quality,
        "fallback_used": fallback_used,
        "source_profile": active_source_profile(),
    }


def provider_warning_flags(
    *,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
    attempts: Optional[Iterable[Dict[str, Any]]] = None,
    error_summary: Optional[str] = None,
) -> List[str]:
    flags: List[str] = []
    attempts_list = list(attempts or [])
    if fallback_used:
        flags.append("fallback_chain_used")
    if partial_result:
        flags.append("partial_result")
    if response_quality == "partial":
        flags.append("response_partial")
    elif response_quality == "degraded":
        flags.append("response_degraded")
    if any(item.get("status") == "blocked" for item in attempts_list):
        flags.append("source_profile_blocked")
    if any(item.get("status") == "failed" for item in attempts_list):
        flags.append("provider_failures_present")
    if error_summary:
        flags.append("error_summary_present")
    return flags


def provider_strength_label(
    *,
    confidence: float,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
) -> str:
    if response_quality == "degraded" or confidence < 0.45:
        return "weak"
    if partial_result or fallback_used or response_quality == "partial" or confidence < 0.68:
        return "mixed"
    return "strong"


def provider_strength_summary(
    provider_name: str,
    *,
    source_type: str,
    freshness_status: str,
    confidence: float,
    fallback_used: bool = False,
    response_quality: str = "complete",
    partial_result: bool = False,
) -> str:
    strength = provider_strength_label(
        confidence=confidence,
        fallback_used=fallback_used,
        response_quality=response_quality,
        partial_result=partial_result,
    )
    return (
        f"{provider_name} supplied {source_type} with {freshness_status} freshness, "
        f"{strength} source strength, confidence {round(confidence, 3)}, "
        f"response quality {response_quality}, and fallback {'on' if fallback_used else 'off'}."
    )


def provider_result_metadata(
    provider_name: str,
    *,
    source_type: Optional[str] = None,
    end_date: Optional[dt.date] = None,
    fallback_used: bool = False,
    response_quality: str = "complete",
    error_summary: Optional[str] = None,
    attempts: Optional[Iterable[Dict[str, Any]]] = None,
    capability: Optional[str] = None,
    retry_policy: Optional[str] = None,
    partial_result: bool = False,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    definition = source_definition(provider_name)
    attempts_list = list(attempts or [])
    freshness_status = provider_freshness_status(end_date=end_date)
    resolved_source_type = source_type or definition.get("provider_type") or "unknown"
    resolved_confidence = round(
        provider_confidence(
            provider_name,
            fallback_used=fallback_used,
            response_quality=response_quality,
            override=confidence,
        ),
        3,
    )
    warning_flags = provider_warning_flags(
        fallback_used=fallback_used,
        response_quality=response_quality,
        partial_result=bool(partial_result),
        attempts=attempts_list,
        error_summary=error_summary,
    )
    return {
        "provider_name": definition.get("source_name") or provider_name,
        "provider_display_name": definition.get("display_name") or provider_name,
        "source_type": resolved_source_type,
        "freshness_status": freshness_status,
        "confidence": resolved_confidence,
        "fallback_used": fallback_used,
        "response_quality": response_quality,
        "error_summary": error_summary,
        "partial_result": bool(partial_result),
        "capability": capability or "data_fetch",
        "retry_policy": retry_policy or "best_available_fallback_chain",
        "fallback_role": definition.get("fallback_role"),
        "criticality": definition.get("criticality"),
        "source_profile": active_source_profile(),
        "attempts": attempts_list,
        "attempt_count": len(attempts_list),
        "source_warning_flags": warning_flags,
        "strength_label": provider_strength_label(
            confidence=resolved_confidence,
            fallback_used=fallback_used,
            response_quality=response_quality,
            partial_result=bool(partial_result),
        ),
        "source_strength_summary": provider_strength_summary(
            definition.get("source_name") or provider_name,
            source_type=resolved_source_type,
            freshness_status=freshness_status,
            confidence=resolved_confidence,
            fallback_used=fallback_used,
            response_quality=response_quality,
            partial_result=bool(partial_result),
        ),
        "provider_definition": definition,
    }


def summarize_provider_failures(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_provider: Dict[str, Dict[str, Any]] = {}
    by_reason: Dict[str, int] = {}
    for item in items:
        provider_name = str(item.get("provider_name") or "unknown")
        bucket = by_provider.setdefault(
            provider_name,
            {
                "provider_name": provider_name,
                "count": 0,
                "reason_codes": {},
                "response_quality": {},
            },
        )
        bucket["count"] += 1
        reason_code = str(item.get("reason_code") or "UNKNOWN")
        reasons = bucket["reason_codes"]
        reasons[reason_code] = int(reasons.get(reason_code, 0)) + 1
        by_reason[reason_code] = int(by_reason.get(reason_code, 0)) + 1
        quality = str(item.get("response_quality") or "unknown")
        response_quality = bucket["response_quality"]
        response_quality[quality] = int(response_quality.get(quality, 0)) + 1
    return {
        "provider_count": len(by_provider),
        "providers": sorted(by_provider.values(), key=lambda row: (-row["count"], row["provider_name"])),
        "top_reason_codes": [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in sorted(by_reason.items(), key=lambda row: (-row[1], row[0]))
        ],
    }


def summarize_provider_usage(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    fallback_used_count = 0
    response_quality_counts: Dict[str, int] = {}
    for item in items:
        provider_name = str(item.get("provider_name") or "unknown")
        counts[provider_name] = counts.get(provider_name, 0) + 1
        if item.get("fallback_used"):
            fallback_used_count += 1
        quality = str(item.get("response_quality") or "unknown")
        response_quality_counts[quality] = response_quality_counts.get(quality, 0) + 1
    return {
        "provider_count": len(counts),
        "providers": [
            {"provider_name": provider_name, "count": count}
            for provider_name, count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        ],
        "fallback_used_count": fallback_used_count,
        "response_quality_counts": response_quality_counts,
    }
