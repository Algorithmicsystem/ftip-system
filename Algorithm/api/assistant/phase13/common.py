from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable, List, Optional


SOURCE_GOVERNANCE_ARTIFACT_KIND = "assistant_source_governance_artifact"
SOURCE_GOVERNANCE_VERSION = "phase13_source_governance_v1"


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


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_list(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = (
                value.get("summary")
                or value.get("title")
                or value.get("source_name")
                or value.get("display_name")
                or value.get("domain")
                or str(value)
            )
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items
