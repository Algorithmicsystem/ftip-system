from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any, Iterable, List, Optional, Sequence


PREDICTION_RECORD_KIND = "assistant_prediction_record"
EVALUATION_ARTIFACT_KIND = "assistant_evaluation_artifact"
EVALUATION_VERSION = "phase6_proof_of_edge_v1"

_HORIZON_DAY_MAP = {
    "short": 5,
    "swing": 21,
    "intermediate": 63,
    "long": 126,
}


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


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    return int(number) if number is not None else None


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def iso_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def as_date(value: Any) -> Optional[dt.date]:
    text = iso_date(value)
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def compact_list(values: Iterable[Any], *, limit: int = 5) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        items.append(str(value))
        if len(items) >= limit:
            break
    return items


def horizon_days(prediction: dict[str, Any]) -> int:
    direct = safe_int(prediction.get("horizon_days"))
    if direct is not None and direct > 0:
        return direct
    return _HORIZON_DAY_MAP.get(str(prediction.get("horizon") or "").lower(), 21)


def hold_band(days: int) -> float:
    base = 0.015 * math.sqrt(max(days, 1) / 5.0)
    return clamp(base, 0.0125, 0.08)


def score_value(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        return safe_float(payload.get("score"))
    return safe_float(payload)


def percentile_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value < 20:
        return "0-20"
    if value < 40:
        return "20-40"
    if value < 60:
        return "40-60"
    if value < 80:
        return "60-80"
    return "80-100"


def monotonic_label(series: Sequence[Optional[float]]) -> str:
    clean = [value for value in series if value is not None]
    if len(clean) < 2:
        return "insufficient"
    if all(left >= right for left, right in zip(clean, clean[1:])):
        return "favorable_buckets_outperform"
    if all(left <= right for left, right in zip(clean, clean[1:])):
        return "unfavorable_buckets_outperform"
    return "mixed"
