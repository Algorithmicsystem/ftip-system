from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable, List, Optional, Sequence


OPERATING_WORKFLOW_ARTIFACT_KIND = "assistant_operating_workflow_artifact"
OPERATING_WORKFLOW_VERSION = "phase14_operating_workflow_v1"


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


def as_datetime(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min, tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed_date = dt.date.fromisoformat(text)
        except ValueError:
            return None
        return dt.datetime.combine(parsed_date, dt.time.min, tzinfo=dt.timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def compact_list(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = (
                value.get("summary")
                or value.get("title")
                or value.get("name")
                or value.get("label")
                or value.get("symbol")
                or value.get("source_name")
                or value.get("alert_summary")
                or value.get("failure_mode")
                or str(value)
            )
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items


def mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if safe_float(value) is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))
