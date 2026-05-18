from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from api.assistant.phase3.common import clamp


def rounded(value: Optional[float], *, digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def weighted_average(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    usable = [(value, weight) for value, weight in items if value is not None and weight > 0]
    if not usable:
        return None
    total_weight = sum(weight for _value, weight in usable)
    if total_weight <= 0:
        return None
    return sum(float(value) * float(weight) for value, weight in usable) / total_weight


def inverse_score(value: Optional[float], *, low: float = 0.0, high: float = 100.0) -> Optional[float]:
    if value is None:
        return None
    return clamp(high - (float(value) - low), 0.0, 100.0)


def midrange_score(
    value: Optional[float],
    *,
    center: float = 50.0,
    tolerance: float = 45.0,
) -> Optional[float]:
    if value is None or tolerance <= 0:
        return None
    distance = abs(float(value) - center)
    return clamp(100.0 - (distance / tolerance) * 100.0, 0.0, 100.0)


def nonnull_ratio(values: Iterable[Optional[float]]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    present = sum(1 for value in values_list if value is not None)
    return clamp((present / len(values_list)) * 100.0, 0.0, 100.0)


def label_score(
    label: object,
    mapping: Mapping[str, float],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    key = str(label or "").strip().lower()
    if not key:
        return default
    return mapping.get(key, default)
