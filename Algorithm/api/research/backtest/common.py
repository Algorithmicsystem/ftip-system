from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any, Iterable, List, Optional, Sequence


CANONICAL_BACKTEST_VERSION = "phase10_canonical_backtest_v1"
WALKFORWARD_VERSION = "phase10_walkforward_v1"
RESEARCH_TRUTH_VERSION = "phase10_research_truth_v1"
CANONICAL_VALIDATION_ARTIFACT_KIND = "assistant_canonical_validation_artifact"
BACKTEST_VALIDATION_ARTIFACT_KIND = "canonical_backtest_validation_artifact"

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


def stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return None
    return float(statistics.pstdev(clean))


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


def horizon_days(label: Any) -> int:
    return _HORIZON_DAY_MAP.get(str(label or "").lower(), 21)


def hold_band(days: int) -> float:
    base = 0.015 * math.sqrt(max(days, 1) / 5.0)
    return clamp(base, 0.0125, 0.08)


def confidence_bucket(value: Optional[float]) -> str:
    score = safe_float(value)
    if score is None:
        return "unknown"
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def conviction_tier_from_confidence(confidence_score: Optional[float]) -> str:
    score = safe_float(confidence_score) or 0.0
    if score >= 85:
        return "very_high"
    if score >= 70:
        return "high"
    if score >= 50:
        return "moderate"
    if score >= 30:
        return "low"
    return "very_low"


def fragility_tier_from_score(score: Optional[float]) -> str:
    value = safe_float(score) or 0.0
    if value >= 75:
        return "acute"
    if value >= 58:
        return "elevated"
    if value >= 42:
        return "mixed"
    return "contained"


def compact_list(values: Iterable[Any], *, limit: int = 5) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        items.append(str(value))
        if len(items) >= limit:
            break
    return items


def monotonic_label(series: Sequence[Optional[float]]) -> str:
    clean = [value for value in series if value is not None]
    if len(clean) < 2:
        return "insufficient"
    if all(left >= right for left, right in zip(clean, clean[1:])):
        return "favorable_buckets_outperform"
    if all(left <= right for left, right in zip(clean, clean[1:])):
        return "unfavorable_buckets_outperform"
    return "mixed"

