from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from api.assistant.coverage import availability_payload


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.median(clean))


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    return float(statistics.pstdev(clean))


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def first_available(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def score_100(
    value: Optional[float],
    *,
    low: float = -1.0,
    high: float = 1.0,
) -> Optional[float]:
    if value is None:
        return None
    if math.isclose(high, low):
        return 50.0
    clipped = clamp(float(value), low, high)
    return 100.0 * ((clipped - low) / (high - low))


def bounded_score(
    value: Optional[float],
    *,
    low: float,
    high: float,
    invert: bool = False,
) -> Optional[float]:
    scored = score_100(value, low=low, high=high)
    if scored is None:
        return None
    return 100.0 - scored if invert else scored


def inverse_metric(value: Optional[float], *, cap: float) -> Optional[float]:
    if value is None:
        return None
    clipped = clamp(float(value), 0.0, cap)
    return 1.0 - clipped / cap


def scaled_score(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return clamp((float(value) - 50.0) / 50.0, -1.0, 1.0)


def coverage_score(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    meta = (payload or {}).get("meta") or {}
    value = safe_float(meta.get("coverage_score"))
    if value is not None:
        return clamp(value, 0.0, 1.0)
    status = str(meta.get("coverage_status") or meta.get("status") or "unknown")
    if status in {"available", "fresh"}:
        return 1.0
    if status in {"partial", "stale_but_usable", "mixed"}:
        return 0.65
    if status in {"limited", "insufficient_history", "stale"}:
        return 0.4
    if status == "not_relevant":
        return 0.5
    return 0.0


def coverage_status_from_score(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "unavailable"
    if value >= 0.75:
        return "available"
    if value >= 0.4:
        return "partial"
    return "limited"


def component(name: str, value: Optional[float], weight: float, detail: str) -> Dict[str, Any]:
    return {
        "name": name,
        "value": round(float(value), 2) if value is not None else None,
        "weight": float(weight),
        "detail": detail,
    }


def weighted_average(components: Sequence[Dict[str, Any]]) -> Optional[float]:
    usable = [item for item in components if item.get("value") is not None and item.get("weight")]
    if not usable:
        return None
    total_weight = sum(float(item["weight"]) for item in usable)
    if total_weight <= 0:
        return None
    return sum(float(item["value"]) * float(item["weight"]) for item in usable) / total_weight


def component_support(
    components: Sequence[Dict[str, Any]],
    penalties: Optional[Sequence[Dict[str, Any]]] = None,
) -> float:
    all_items = list(components) + list(penalties or [])
    if not all_items:
        return 0.0
    total_weight = sum(max(float(item.get("weight") or 0.0), 0.0) for item in all_items)
    if total_weight <= 0:
        return 0.0
    available_weight = sum(
        max(float(item.get("weight") or 0.0), 0.0)
        for item in all_items
        if item.get("value") is not None
    )
    return clamp(available_weight / total_weight, 0.0, 1.0)


def freshness_penalty(quality_domain: Optional[Dict[str, Any]]) -> float:
    freshness = (quality_domain or {}).get("freshness_summary") or {}
    penalty = 0.0
    for item in freshness.values():
        status = str(item.get("status") or "")
        if status == "stale_but_usable":
            penalty += 3.0
        elif status == "stale":
            penalty += 7.0
        elif status == "limited":
            penalty += 4.0
    return penalty


def group_meta(
    *,
    coverage_inputs: Sequence[Optional[float]],
    components: Sequence[Dict[str, Any]],
    note: str,
) -> Dict[str, Any]:
    support = mean(list(coverage_inputs) + [component_support(components)])
    status = coverage_status_from_score(support)
    return {
        "coverage_score": support,
        "status": status,
        **availability_payload(
            has_data=bool(components),
            coverage_score=support,
            freshness_status=None,
            missing_reason="unavailable" if not components else None,
            data_quality_note=note,
        ),
    }


def as_percentile_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return clamp(number, 0.0, 100.0)

