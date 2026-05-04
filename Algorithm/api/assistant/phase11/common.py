from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


PORTFOLIO_RISK_MODEL_ARTIFACT_KIND = "assistant_portfolio_risk_model_artifact"
PORTFOLIO_RISK_MODEL_VERSION = "phase11_portfolio_risk_model_v1"


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


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_list(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = (
                value.get("label")
                or value.get("name")
                or value.get("title")
                or value.get("symbol")
                or str(value)
            )
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [float(value) for value in values if safe_float(value) is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def stdev(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    avg = sum(clean) / len(clean)
    variance = sum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    if variance < 0:
        return None
    return float(math.sqrt(variance))


def correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    clean_left: List[float] = []
    clean_right: List[float] = []
    for left_value, right_value in zip(left, right):
        if not math.isfinite(float(left_value)) or not math.isfinite(float(right_value)):
            continue
        clean_left.append(float(left_value))
        clean_right.append(float(right_value))
    if len(clean_left) < 3:
        return None
    left_mean = sum(clean_left) / len(clean_left)
    right_mean = sum(clean_right) / len(clean_right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(clean_left, clean_right)
    )
    left_std = stdev(clean_left)
    right_std = stdev(clean_right)
    if numerator == 0 or left_std in (None, 0.0) or right_std in (None, 0.0):
        return 0.0
    return float(numerator / ((len(clean_left) - 1) * left_std * right_std))


def covariance(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    clean_left: List[float] = []
    clean_right: List[float] = []
    for left_value, right_value in zip(left, right):
        if not math.isfinite(float(left_value)) or not math.isfinite(float(right_value)):
            continue
        clean_left.append(float(left_value))
        clean_right.append(float(right_value))
    if len(clean_left) < 3:
        return None
    left_mean = sum(clean_left) / len(clean_left)
    right_mean = sum(clean_right) / len(clean_right)
    return float(
        sum(
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(clean_left, clean_right)
        )
        / (len(clean_left) - 1)
    )


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> Optional[float]:
    shared = sorted(set(left.keys()) & set(right.keys()))
    if not shared:
        return None
    numerator = sum(float(left[key]) * float(right[key]) for key in shared)
    left_mag = math.sqrt(sum(float(left[key]) ** 2 for key in shared))
    right_mag = math.sqrt(sum(float(right[key]) ** 2 for key in shared))
    if left_mag == 0 or right_mag == 0:
        return None
    return float(numerator / (left_mag * right_mag))


def as_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        return None


def score_bucket(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "unknown"
    if number >= 75.0:
        return "high"
    if number >= 55.0:
        return "moderate"
    return "elevated"


def state_tag(value: Any, *, high_threshold: float = 60.0, low_threshold: float = 40.0) -> str:
    number = safe_float(value)
    if number is None:
        return "unknown"
    if number >= high_threshold:
        return "high"
    if number <= low_threshold:
        return "low"
    return "mixed"
