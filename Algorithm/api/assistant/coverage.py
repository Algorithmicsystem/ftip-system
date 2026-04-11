from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


RETURN_HORIZON_REQUIREMENTS = {
    "1d": 2,
    "5d": 6,
    "10d": 11,
    "21d": 22,
    "63d": 64,
    "126d": 127,
    "252d": 253,
}

TECHNICAL_HORIZON_REQUIREMENTS = {
    "10d": 10,
    "21d": 21,
    "63d": 63,
    "126d": 126,
}


def classify_horizon_coverage(
    row_count: int,
    *,
    requirements: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    requirements = dict(requirements or RETURN_HORIZON_REQUIREMENTS)
    available = [label for label, required in requirements.items() if row_count >= required]
    missing = [label for label, required in requirements.items() if row_count < required]
    status = "available"
    if missing and available:
        status = "partial"
    elif missing and not available:
        status = "insufficient_history"
    return {
        "coverage_status": status,
        "available_horizons": available,
        "missing_horizons": missing,
        "available_through": available[-1] if available else None,
        "missing_reason": "insufficient_history" if missing else None,
    }


def classify_domain_coverage(
    *,
    has_data: bool,
    coverage_score: Optional[float] = None,
    freshness_status: Optional[str] = None,
    available_horizons: Optional[Sequence[str]] = None,
    missing_horizons: Optional[Sequence[str]] = None,
    missing_reason: Optional[str] = None,
    relevant: bool = True,
) -> str:
    if not relevant:
        return "not_relevant"
    if freshness_status == "stale":
        return "stale"
    available_horizons = list(available_horizons or [])
    missing_horizons = list(missing_horizons or [])
    if available_horizons and missing_horizons:
        return "partial"
    if missing_reason == "insufficient_history" and not available_horizons:
        return "insufficient_history"
    if not has_data:
        return missing_reason or "unavailable"
    if freshness_status == "stale_but_usable":
        return "stale"
    if coverage_score is None:
        return "available"
    if coverage_score >= 0.75:
        return "available"
    if coverage_score > 0:
        return "partial"
    return "unavailable"


def availability_payload(
    *,
    has_data: bool,
    coverage_score: Optional[float] = None,
    freshness_status: Optional[str] = None,
    available_horizons: Optional[Sequence[str]] = None,
    missing_horizons: Optional[Sequence[str]] = None,
    missing_reason: Optional[str] = None,
    fallback_used: bool = False,
    fallback_source: Optional[Sequence[str]] = None,
    data_quality_note: Optional[str] = None,
    relevant: bool = True,
) -> Dict[str, Any]:
    status = classify_domain_coverage(
        has_data=has_data,
        coverage_score=coverage_score,
        freshness_status=freshness_status,
        available_horizons=available_horizons,
        missing_horizons=missing_horizons,
        missing_reason=missing_reason,
        relevant=relevant,
    )
    payload = {
        "coverage_status": status,
        "available_horizons": list(available_horizons or []),
        "missing_horizons": list(missing_horizons or []),
        "missing_reason": missing_reason,
        "fallback_used": bool(fallback_used),
        "fallback_source": list(fallback_source or []),
        "data_quality_note": data_quality_note,
    }
    if status == "insufficient_history" and payload["missing_reason"] is None:
        payload["missing_reason"] = "insufficient_history"
    if status == "unavailable" and payload["missing_reason"] is None:
        payload["missing_reason"] = "unavailable"
    return payload
