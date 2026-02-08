from __future__ import annotations

import datetime as dt
import uuid
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from api.main import Candle


def build_audit(
    candles: List["Candle"], as_of_date: dt.date, audit_no_lookahead: bool
) -> Dict[str, object]:
    last_used = None
    per_feature: Dict[str, str] = {}
    if candles:
        try:
            last_used = dt.date.fromisoformat(candles[-1].timestamp)
        except Exception:
            last_used = as_of_date
    if last_used is None:
        last_used = as_of_date
    no_lookahead_ok = bool(not audit_no_lookahead or last_used <= as_of_date)
    per_feature = {"default_window_end": last_used.isoformat()}
    return {
        "no_lookahead_ok": no_lookahead_ok,
        "last_candle_used": last_used.isoformat(),
        "per_feature_last_candle": per_feature,
        "trace_id": str(uuid.uuid4()),
    }


__all__ = ["build_audit"]
